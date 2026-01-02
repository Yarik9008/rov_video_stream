#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebRTC Video Streaming Server (LAN)

- Автоматически определяет локальный IP
- Поднимает HTTP signaling (aiohttp) для WebRTC offer/answer
- Рассылает broadcast (UDP) для автообнаружения клиентом в локальной сети
"""

import argparse
import asyncio
import atexit
import itertools
import json
import os
import platform
import signal
import socket
import threading
import time
from pathlib import Path
from logging import INFO
from fractions import Fraction

from Logger import Logger, loggingLevels


def get_local_ip() -> str:
    """Получает локальный IPv4 адрес.

    Важно: не полагаемся на доступ к интернету. Пробуем несколько способов и берём первый
    не-loopback IPv4 (не 127.* и не 169.254.*).
    """
    ips = get_local_ips()
    return ips[0] if ips else "127.0.0.1"


def _is_usable_ipv4(ip: str) -> bool:
    try:
        socket.inet_aton(ip)
    except OSError:
        return False
    if ip.startswith("127."):
        return False
    if ip.startswith("169.254."):
        return False
    return True


def get_local_ips() -> list[str]:
    """Возвращает список пригодных локальных IPv4 адресов (LAN)."""
    candidates: list[str] = []

    # 1) Попробовать hostname -> адреса
    try:
        _hostname = socket.gethostname()
        _name, _aliases, addrs = socket.gethostbyname_ex(_hostname)
        for ip in addrs:
            if _is_usable_ipv4(ip) and ip not in candidates:
                candidates.append(ip)
    except Exception:
        pass

    # 2) UDP connect-трюк, без отправки пакетов (нужен только выбор интерфейса)
    for target in (("1.1.1.1", 80), ("8.8.8.8", 80), ("10.255.255.255", 1)):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            try:
                s.connect(target)
                ip = s.getsockname()[0]
                if _is_usable_ipv4(ip) and ip not in candidates:
                    candidates.append(ip)
            finally:
                s.close()
        except Exception:
            continue

    return candidates


def get_broadcast_address(ip: str) -> str:
    """Получает broadcast адрес для данного IP (замена последнего октета на 255)."""
    try:
        parts = ip.split(".")
        if len(parts) == 4:
            parts[-1] = "255"
            return ".".join(parts)
    except Exception:
        pass
    return "255.255.255.255"


def calculate_optimal_bitrate(width: int, height: int, fps: int) -> int:
    """
    Рассчитывает оптимальный битрейт для максимального качества на основе разрешения и FPS.
    
    Формула: битрейт = (ширина * высота * FPS * бит_на_пиксель) / 1000
    Для максимального качества используем повышенное значение bits_per_pixel,
    чтобы получить минимальное сжатие (ближе к visually-lossless).
    
    Returns: битрейт в kbps
    """
    if width <= 0 or height <= 0 or fps <= 0:
        return 200000  # Fallback: 200 Mbps
    
    pixels = width * height
    # Примерные ориентиры:
    # - 1080p@30 при 0.5 bpp ≈ 31 Mbps
    # - 4K@60   при 0.5 bpp ≈ 249 Mbps
    bits_per_pixel = 0.5  # Для максимального качества
    bitrate_bps = pixels * fps * bits_per_pixel
    bitrate_kbps = int(bitrate_bps / 1000)
    
    # Ограничения:
    # - Минимум: 20000 kbps (20 Mbps), чтобы не деградировать качество на 1080p+
    # - Максимум: 500000 kbps (500 Mbps) для очень высоких разрешений
    bitrate_kbps = max(20000, min(500000, bitrate_kbps))
    
    return bitrate_kbps


def _lazy_imports():
    try:
        import cv2  # type: ignore
        from aiohttp import web  # type: ignore
        from aiortc import (  # type: ignore
            RTCPeerConnection,
            RTCSessionDescription,
            RTCConfiguration,
            RTCIceServer,
            RTCRtpSender,
        )
        from aiortc import VideoStreamTrack  # type: ignore
        from av import VideoFrame  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Для WebRTC нужны зависимости. Установите:\n"
            "  pip install -r requirements.txt\n"
            f"Детали: {e}"
        )
    return cv2, web, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender, VideoStreamTrack, VideoFrame


def _safe_client_ip(request) -> str:
    """Получить IP клиента из aiohttp request (учитывая прокси заголовки)."""
    try:
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            # берём первый IP из списка
            return xff.split(",")[0].strip()
    except Exception:
        pass
    try:
        return request.remote or "unknown"
    except Exception:
        return "unknown"


class _SharedFrameSource:
    """
    Общий источник кадров (камера/файл), который читает кадры в фоне и
    хранит последний кадр. Это позволяет:
    - множественные подключения (несколько клиентов одновременно)
    - повторные подключения без перезапуска сервера
    """

    def __init__(self, source: str, file_path: str, device_index: int, width: int, height: int, fps: int):
        self._source = source
        self._file_path = file_path
        self._device_index = int(device_index)
        self._fps = max(1, int(fps))
        # Всегда считаем target size (и для файла, и для камеры):
        # если камера игнорирует CAP_PROP_* — будем ресайзить кадр программно.
        self._target_width = int(width) if width else None
        self._target_height = int(height) if height else None

        self._lock = threading.Lock()
        self._latest = None  # numpy array (bgr)
        self._latest_ts = 0.0  # monotonic timestamp of latest frame
        self._running = False
        self._thread = None

        self._cv2 = None
        self._cap = None
        self._is_file = False

    def start(self):
        if self._running:
            return

        cv2, *_rest = _lazy_imports()
        self._cv2 = cv2

        if self._source == "file":
            if not self._file_path:
                raise ValueError("Не указан путь к файлу (используйте --file)")
            p = str(Path(self._file_path).resolve())
            if not os.path.exists(p):
                raise FileNotFoundError(f"Файл не найден: {p}")
            self._cap = cv2.VideoCapture(p)
            self._is_file = True
        else:
            self._cap = cv2.VideoCapture(int(self._device_index))
            self._is_file = False

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("Не удалось открыть источник видео (камера/файл).")

        # Best-effort: уменьшаем внутренний буфер захвата для минимальной задержки (не везде поддерживается).
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # попытка настроить камеру (только FPS; размер не трогаем — нужен нативный)
        if not self._is_file:
            try:
                if self._target_width and self._target_height:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._target_width))
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._target_height))
                self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            except Exception:
                pass

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        t = self._thread
        if t:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        self._thread = None

        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def get_latest(self):
        with self._lock:
            if self._latest is None:
                return None
            try:
                return self._latest.copy()
            except Exception:
                return None

    def get_latest_with_ts(self):
        """
        Возвращает (frame_copy, ts) где ts — monotonic timestamp последнего кадра.
        Нужен для минимальной задержки: трек ждёт именно НОВЫЙ кадр, без второго pacing.
        """
        with self._lock:
            if self._latest is None:
                return None, 0.0
            try:
                return self._latest.copy(), float(self._latest_ts)
            except Exception:
                return None, 0.0
    
    def get_video_size(self):
        """Возвращает реальные размеры видео (width, height) или None, если не определены."""
        if self._cap is None or self._cv2 is None:
            return None
        try:
            width = int(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            if width > 0 and height > 0:
                return (width, height)
        except Exception:
            pass
        # Если не удалось получить из cap, попробуем получить из последнего кадра
        with self._lock:
            if self._latest is not None:
                try:
                    h, w = self._latest.shape[:2]
                    return (w, h)
                except Exception:
                    pass
        return None

    def _run(self):
        cv2 = self._cv2
        cap = self._cap
        if cv2 is None or cap is None:
            return

        frame_period = 1.0 / float(self._fps)
        next_ts = time.time()

        while self._running:
            # pacing чтения (чтобы не жечь CPU)
            now = time.time()
            delay = next_ts - now
            if delay > 0:
                time.sleep(delay)
            next_ts = max(next_ts + frame_period, time.time())

            ok, frame = cap.read()
            if not ok or frame is None:
                if self._is_file:
                    # loop file
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception:
                        pass
                    ok, frame = cap.read()
                if not ok or frame is None:
                    continue

            # Ресайз применяем, если задан target size (и для файла, и для камеры)
            # Выбираем алгоритм интерполяции в зависимости от операции:
            # - INTER_AREA для уменьшения (лучшее качество при downscale)
            # - INTER_LANCZOS4 для увеличения (лучшее качество при upscale)
            if self._target_width and self._target_height:
                try:
                    current_h, current_w = frame.shape[:2]
                    if current_w != self._target_width or current_h != self._target_height:
                        # Определяем, увеличиваем или уменьшаем
                        scale_w = self._target_width / current_w
                        scale_h = self._target_height / current_h
                        is_upscaling = scale_w > 1.0 or scale_h > 1.0
                        
                        # Выбираем оптимальный алгоритм интерполяции
                        if is_upscaling:
                            # Для увеличения используем LANCZOS4 - лучшее качество
                            interpolation = cv2.INTER_LANCZOS4
                        else:
                            # Для уменьшения используем AREA - лучшее качество
                            interpolation = cv2.INTER_AREA
                        
                        frame = cv2.resize(
                            frame,
                            (self._target_width, self._target_height),
                            interpolation=interpolation,
                        )
                except Exception:
                    pass

            with self._lock:
                self._latest = frame
                self._latest_ts = time.monotonic()


def _make_capture_track(shared: _SharedFrameSource, fps: int):
    cv2, web, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender, VideoStreamTrack, VideoFrame = _lazy_imports()

    class _Track(VideoStreamTrack):
        def __init__(self, _shared: _SharedFrameSource):
            super().__init__()
            self._shared = _shared
            # Для минимальной задержки НЕ делаем второй pacing в треке.
            # Вместо этого ждём новый кадр от shared-источника и отправляем сразу.
            self._fps = max(1, int(fps))
            self._clock_rate = 90000
            self._time_base = Fraction(1, self._clock_rate)
            self._pts = 0
            self._pts_step = max(1, int(self._clock_rate / float(self._fps)))
            self._last_frame_ts = 0.0
            
        async def recv(self):
            # ждём НОВЫЙ кадр (камера/файл могут стартовать не сразу)
            # Важно: это снижает задержку, т.к. мы не “усыпляем” трек сверх чтения кадра.
            deadline = time.monotonic() + 2.0
            frame = None
            ts = 0.0

            while time.monotonic() < deadline:
                frame, ts = self._shared.get_latest_with_ts()
                if frame is not None and ts > self._last_frame_ts:
                    break
                await asyncio.sleep(0.001)

            if frame is None:
                raise RuntimeError("Нет кадров от источника (камера/файл).")

            self._last_frame_ts = ts
            self._pts += self._pts_step

            vf = VideoFrame.from_ndarray(frame, format="bgr24")
            vf.pts = self._pts
            vf.time_base = self._time_base
            return vf

        def stop(self):
            super().stop()

    return _Track(shared)


def _prefer_h264(pc):
    cv2, web, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender, VideoStreamTrack, VideoFrame = _lazy_imports()
    try:
        caps = RTCRtpSender.getCapabilities("video")
        codecs = [c for c in caps.codecs if c.mimeType.lower() in ("video/h264", "video/vp8")]
        codecs.sort(key=lambda c: 0 if c.mimeType.lower() == "video/h264" else 1)
        for t in pc.getTransceivers():
            if t.kind == "video" and codecs:
                t.setCodecPreferences(codecs)
    except Exception:
        pass


async def _wait_ice_complete(pc):
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_state():
        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=5)
    except asyncio.TimeoutError:
        pass


async def _apply_sender_quality(sender, target_bitrate_kbps: int, target_fps: int):
    """
    Пытаемся поднять качество через RTCRtpSender.setParameters():
    - maxBitrate (бит/с) - максимальный битрейт для максимального качества
    - maxFramerate (кадры/с)
    - scaleResolutionDownBy - не используем (сохраняем исходное разрешение)

    В зависимости от версии aiortc setParameters может быть корутиной.
    """
    try:
        params = sender.getParameters()
        if not getattr(params, "encodings", None):
            return
        if not params.encodings:
            return

        enc = params.encodings[0]
        if target_bitrate_kbps and target_bitrate_kbps > 0:
            # Устанавливаем максимальный битрейт для максимального качества
            enc.maxBitrate = int(target_bitrate_kbps) * 1000
            # Также устанавливаем minBitrate для стабильности (50% от max)
            if hasattr(enc, 'minBitrate'):
                enc.minBitrate = int(target_bitrate_kbps * 0.5) * 1000
        if target_fps and target_fps > 0:
            enc.maxFramerate = int(target_fps)
        
        # Отключаем масштабирование разрешения для максимального качества
        if hasattr(enc, 'scaleResolutionDownBy'):
            enc.scaleResolutionDownBy = 1.0

        res = sender.setParameters(params)
        if asyncio.iscoroutine(res):
            await res
    except Exception:
        # Если API/кодек не поддерживает параметры — просто игнорируем
        return


class DiscoveryBroadcaster:
    DISCOVERY_PORT = 5003

    def __init__(self, payload_factory, enabled: bool):
        self._payload_factory = payload_factory
        self._enabled = enabled
        self._running = False
        self._thread = None

    def start(self):
        if not self._enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def _run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except Exception:
                pass
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(1)

            while self._running:
                try:
                    payload = self._payload_factory()
                    message = json.dumps(payload).encode("utf-8")
                    # Шлём на несколько broadcast-адресов:
                    # - 255.255.255.255 (универсальный broadcast, часто работает)
                    # - вычисленные broadcast для найденных локальных IP (если /24 и т.п.)
                    targets = {"255.255.255.255"}
                    for ip in get_local_ips():
                        targets.add(get_broadcast_address(ip))
                    for bcast in targets:
                        try:
                            sock.sendto(message, (bcast, self.DISCOVERY_PORT))
                        except Exception:
                            continue
                    time.sleep(1.5)
                except Exception:
                    time.sleep(1.5)

            sock.close()
        except Exception:
            pass


async def run_server(args, logger: Logger):
    cv2, web, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender, VideoStreamTrack, VideoFrame = _lazy_imports()

    # host/local_ip
    if args.host is None:
        local_ip = get_local_ip()
        host = "0.0.0.0" if local_ip != "127.0.0.1" else "127.0.0.1"
    else:
        host = args.host
        local_ip = get_local_ip() if host == "0.0.0.0" else host

    cfg = RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])] if args.stun else []
    )

    pcs = set()
    pc_meta = {}
    pc_id_counter = itertools.count(1)
    shared = _SharedFrameSource(args.source, args.file, args.device, args.width, args.height, args.fps)
    shared.start()

    async def offer(request):
        client_ip = _safe_client_ip(request)
        user_agent = None
        try:
            user_agent = request.headers.get("User-Agent")
        except Exception:
            user_agent = None

        params = await request.json()
        offer_sdp = params["sdp"]
        offer_type = params["type"]

        pc = RTCPeerConnection(configuration=cfg)
        pc_id = next(pc_id_counter)
        pcs.add(pc)
        pc_meta[pc] = {
            "id": pc_id,
            "ip": client_ip,
            "ua": user_agent,
            "created": time.time(),
            "last_conn_state": None,
            "last_ice_state": None,
        }

        logger.info(f"client#{pc_id} CONNECT offer from {client_ip}" + (f" | UA: {user_agent}" if user_agent else ""), source="server")

        @pc.on("connectionstatechange")
        async def on_state_change():
            try:
                meta = pc_meta.get(pc, {})
                _id = meta.get("id", "?")
                state = getattr(pc, "connectionState", None)
                prev = meta.get("last_conn_state")
                if state != prev:
                    meta["last_conn_state"] = state
                    logger.info(f"client#{_id} PC state: {state}", source="server")

                if pc.connectionState in ("failed", "disconnected", "closed"):
                    pcs.discard(pc)
                    ip = meta.get("ip", "unknown")
                    logger.info(f"client#{_id} DISCONNECT ({pc.connectionState}) from {ip}", source="server")
                    try:
                        await pc.close()
                    except Exception:
                        pass
                    pc_meta.pop(pc, None)
            except Exception:
                pass

        @pc.on("iceconnectionstatechange")
        def on_ice_state_change():
            try:
                meta = pc_meta.get(pc, {})
                _id = meta.get("id", "?")
                state = getattr(pc, "iceConnectionState", None)
                prev = meta.get("last_ice_state")
                if state != prev:
                    meta["last_ice_state"] = state
                    logger.debug(f"client#{_id} ICE state: {state}", source="server")
            except Exception:
                pass

        try:
            # Важно: на каждое подключение создаём отдельный track,
            # чтобы поддержать множественных клиентов и реконнект.
            track = _make_capture_track(shared, args.fps)
            sender = pc.addTrack(track)
            _prefer_h264(pc)

            # Определяем битрейт: автоматический расчет в режиме максимального качества
            target_bitrate = int(args.bitrate)
            if args.max_quality:
                video_size = shared.get_video_size()
                if video_size:
                    width, height = video_size
                    target_bitrate = calculate_optimal_bitrate(width, height, args.fps)
                    logger.info(f"client#{pc_id} Автоматический битрейт для {width}x{height}@{args.fps}fps: {target_bitrate} kbps", source="server")

            # Качество: попросим encoder о более высоком битрейте/фпс
            await _apply_sender_quality(sender, target_bitrate, int(args.fps))

            await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await _wait_ice_complete(pc)

            logger.info(f"client#{pc_id} ANSWER ready (pc={pc.connectionState}, ice={pc.iceConnectionState})", source="server")
            return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
        except Exception as e:
            pcs.discard(pc)
            meta = pc_meta.pop(pc, {})
            _id = meta.get("id", pc_id)
            ip = meta.get("ip", client_ip)
            try:
                await pc.close()
            except Exception:
                pass
            logger.error(f"client#{_id} ERROR during setup from {ip}: {e}", source="server")
            raise web.HTTPInternalServerError(text=f"Ошибка установки соединения: {e}")

    async def index(_request):
        return web.json_response({"status": "ok", "backend": "webrtc"})

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=int(args.signal_port))
    await site.start()

    # Получаем реальное разрешение видео, если width или height равны 0
    actual_width = int(args.width) if args.width > 0 else None
    actual_height = int(args.height) if args.height > 0 else None
    
    if actual_width is None or actual_height is None:
        # Пытаемся получить реальное разрешение из видео
        video_size = shared.get_video_size()
        if video_size:
            actual_width = video_size[0]
            actual_height = video_size[1]
        else:
            # Fallback на значения по умолчанию
            actual_width = actual_width or 1280
            actual_height = actual_height or 720
    
    payload = {
        "backend": "webrtc",
        "host": local_ip,
        "signal_port": int(args.signal_port),
        "width": actual_width,
        "height": actual_height,
        "fps": int(args.fps),
    }

    broadcaster = DiscoveryBroadcaster(lambda: payload, enabled=(host != "127.0.0.1"))
    broadcaster.start()
    atexit.register(broadcaster.stop)

    logger.info("Запуск WebRTC сервера...", source="server")
    logger.info(f"Платформа: {platform.system()}", source="server")
    logger.info(f"Источник: {args.source}", source="server")
    if args.source == "webcam":
        logger.info(f"Камера: индекс {args.device}", source="server")
    else:
        logger.info(f"Файл: {args.file}", source="server")
    
    # Информация о качестве
    video_size = shared.get_video_size()
    if video_size:
        width, height = video_size
        logger.info(f"Разрешение видео: {width}x{height}", source="server")
        if args.max_quality:
            auto_bitrate = calculate_optimal_bitrate(width, height, args.fps)
            logger.info(f"Режим максимального качества: ВКЛ (автобитрейт: {auto_bitrate} kbps)", source="server")
        else:
            logger.info(f"Битрейт: {args.bitrate} kbps ({args.bitrate/1000:.1f} Mbps)", source="server")
    else:
        logger.info(f"Битрейт: {args.bitrate} kbps ({args.bitrate/1000:.1f} Mbps)", source="server")
    logger.info(f"FPS: {args.fps}", source="server")
    if args.width > 0 and args.height > 0:
        logger.info(f"Ресайз: {args.width}x{args.height}", source="server")
    else:
        logger.info("Ресайз: отключен (исходное разрешение)", source="server")
    
    logger.info(f"Локальный IP: {local_ip}", source="server")
    logger.info(f"Signaling: http://{local_ip}:{int(args.signal_port)}", source="server")
    if host != "127.0.0.1":
        logger.info(f"Автообнаружение: включено (UDP broadcast порт {DiscoveryBroadcaster.DISCOVERY_PORT})", source="server")
        logger.info("Клиент:  python3 client.py --auto", source="server")
    logger.info("Нажмите Ctrl+C для остановки.", source="server")

    stop_event = asyncio.Event()

    def _stop(*_):
        stop_event.set()

    if platform.system() != "Windows":
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, _stop)
        except Exception:
            pass
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, _stop)
    except Exception:
        # Windows fallback: SIGINT придёт как KeyboardInterrupt
        pass

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        broadcaster.stop()
        try:
            shared.stop()
        except Exception:
            pass
        try:
            alive = len(pcs)
            if alive:
                logger.info(f"shutdown: closing {alive} active client(s)", source="server")
        except Exception:
            pass
        coros = [pc.close() for pc in pcs]
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="WebRTC Video Streaming Server (LAN)")
    parser.add_argument("--source", choices=["webcam", "file"], default="webcam", help="Источник видео")
    parser.add_argument("--file", type=str, default=None, help="Путь к файлу (если --source file)")
    parser.add_argument("-t", "--test", action="store_true", default=False, help="Транслировать тестовое видео test.mp4 в исходном разрешении")
    parser.add_argument("--host", type=str, default=None, help="Хост для режима LAN (обычно 0.0.0.0)")
    parser.add_argument("--signal-port", type=int, default=8080, help="Порт HTTP signaling (по умолчанию 8080)")
    # По умолчанию пытаемся отдавать 4K (3840x2160).
    # Если хотите передавать в нативном разрешении источника — укажите: --width 0 --height 0
    parser.add_argument("--width", type=int, default=3840, help="Ширина (по умолчанию 3840 = 4K; 0 = исходное разрешение)")
    parser.add_argument("--height", type=int, default=2160, help="Высота (по умолчанию 2160 = 4K; 0 = исходное разрешение)")
    parser.add_argument("--fps", type=int, default=30, help="FPS (рекомендуется 30-60 для максимального качества)")
    # Максимальное качество включено по умолчанию.
    # argparse.BooleanOptionalAction создаёт флаги:
    #   --max-quality / --no-max-quality
    parser.add_argument(
        "--max-quality",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Режим максимального качества (по умолчанию ВКЛ): автоматический битрейт на основе разрешения, без ресайза",
    )
    parser.add_argument("--bitrate", type=int, default=200000, help="Целевой битрейт видео (kbps), по умолчанию 200000 (200 Mbps для максимального качества)")
    parser.add_argument("--device", type=int, default=0, help="Индекс камеры (для webcam)")
    parser.add_argument("--stun", action="store_true", default=False, help="Использовать публичный STUN (для LAN не нужно)")
    parser.add_argument("--log-level", type=str, default="info", choices=["spam", "debug", "verbose", "info", "warning", "error", "critical"], help="Уровень логирования")
    parser.add_argument("--log-path", type=str, default="logs", help="Путь к папке с логами")

    args = parser.parse_args()
    
    # Обработка флага -t/--test
    if args.test:
        test_video_path = Path(__file__).parent / "video" / "test.mp4"
        args.source = "file"
        args.file = str(test_video_path)
        args.width = 0  # 0 означает использовать исходное разрешение
        args.height = 0  # 0 означает использовать исходное разрешение
        # Для тестового видео включаем max-quality и пытаемся взять FPS из файла,
        # чтобы передавать в исходном качестве с минимальной задержкой.
        args.max_quality = True
        try:
            cv2, *_rest = _lazy_imports()
            if test_video_path.exists():
                cap = cv2.VideoCapture(str(test_video_path.resolve()))
                try:
                    file_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if file_fps >= 1.0 and file_fps <= 240.0:
                        args.fps = int(round(file_fps))
                finally:
                    cap.release()
            else:
                # Warning will be logged later by the logger
                pass
        except Exception:
            pass

        # Битрейт выставляем высоким как baseline (дальше в offer при max-quality он будет пересчитан по размеру/FPS)
        args.bitrate = 500000
    
    # Обработка режима максимального качества
    if args.max_quality:
        # В режиме максимального качества:
        # 1. Используем исходное разрешение (если не указано явно)
        if args.width == 0 and args.height == 0:
            # Разрешение будет определено автоматически из источника
            pass
        # 2. Автоматически рассчитываем битрейт на основе разрешения
        # (будет пересчитан после получения реального разрешения)
        # 3. Увеличиваем FPS если возможно
        if args.fps < 30:
            args.fps = 30
    
    log_level = loggingLevels.get(args.log_level, INFO)
    logger = Logger("server", args.log_path, log_level)
    
    try:
        asyncio.run(run_server(args, logger))
    except KeyboardInterrupt:
        logger.info("Остановка сервера по запросу пользователя", source="server")
    except Exception as e:
        logger.critical(f"Критическая ошибка сервера: {e}", source="server")


if __name__ == "__main__":
    main()

