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
import json
import os
import platform
import signal
import socket
import threading
import time
from pathlib import Path


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


def _make_capture_track(source: str, file_path: str, device_index: int, width: int, height: int, fps: int):
    cv2, web, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender, VideoStreamTrack, VideoFrame = _lazy_imports()

    class _Track(VideoStreamTrack):
        def __init__(self):
            super().__init__()
            self._fps = max(1, int(fps))
            self._width = int(width)
            self._height = int(height)
            self._frame_period = 1.0 / float(self._fps)
            self._next_ts = time.time()

            if source == "file":
                if not file_path:
                    raise ValueError("Не указан путь к файлу (используйте --file)")
                p = str(Path(file_path).resolve())
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Файл не найден: {p}")
                self._cap = cv2.VideoCapture(p)
                self._is_file = True
            else:
                self._cap = cv2.VideoCapture(int(device_index))
                self._is_file = False

            if not self._cap.isOpened():
                raise RuntimeError("Не удалось открыть источник видео (камера/файл).")

            # попытка настроить камеру
            if not self._is_file:
                try:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    self._cap.set(cv2.CAP_PROP_FPS, self._fps)
                except Exception:
                    pass
            
        async def recv(self):
            # pacing
            now = time.time()
            delay = self._next_ts - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_ts = max(self._next_ts + self._frame_period, time.time())

            pts, time_base = await self.next_timestamp()

            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._is_file:
                    # loop file
                    try:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception:
                        pass
                    ok, frame = self._cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Не удалось прочитать кадр.")

            # ensure size
            try:
                if frame.shape[1] != self._width or frame.shape[0] != self._height:
                    frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
            except Exception:
                pass

            vf = VideoFrame.from_ndarray(frame, format="bgr24")
            vf.pts = pts
            vf.time_base = time_base
            return vf

        def stop(self):
            try:
                self._cap.release()
            except Exception:
                pass
            super().stop()

    return _Track()


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


async def run_server(args):
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
    track = _make_capture_track(args.source, args.file, args.device, args.width, args.height, args.fps)

    async def offer(request):
        params = await request.json()
        offer_sdp = params["sdp"]
        offer_type = params["type"]

        pc = RTCPeerConnection(configuration=cfg)
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state_change():
            if pc.connectionState in ("failed", "disconnected", "closed"):
                await pc.close()
                pcs.discard(pc)

        pc.addTrack(track)
        _prefer_h264(pc)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_ice_complete(pc)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def index(_request):
        return web.json_response({"status": "ok", "backend": "webrtc"})

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=int(args.signal_port))
    await site.start()

    payload = {
        "backend": "webrtc",
        "host": local_ip,
        "signal_port": int(args.signal_port),
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
    }

    broadcaster = DiscoveryBroadcaster(lambda: payload, enabled=(host != "127.0.0.1"))
    broadcaster.start()
    atexit.register(broadcaster.stop)

    print("Запуск WebRTC сервера...")
    print(f"Платформа: {platform.system()}")
    print(f"Источник: {args.source}")
    if args.source == "webcam":
        print(f"Камера: индекс {args.device}")
    else:
        print(f"Файл: {args.file}")
    print(f"Локальный IP: {local_ip}")
    print(f"Signaling: http://{local_ip}:{int(args.signal_port)}")
    if host != "127.0.0.1":
        print(f"Автообнаружение: включено (UDP broadcast порт {DiscoveryBroadcaster.DISCOVERY_PORT})")
        print("Клиент:  python3 client.py --auto")
    print("\nНажмите Ctrl+C для остановки.\n")

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
            track.stop()
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
    parser.add_argument("--host", type=str, default=None, help="Хост для режима LAN (обычно 0.0.0.0)")
    parser.add_argument("--signal-port", type=int, default=8080, help="Порт HTTP signaling (по умолчанию 8080)")
    parser.add_argument("--width", type=int, default=640, help="Ширина")
    parser.add_argument("--height", type=int, default=480, help="Высота")
    parser.add_argument("--fps", type=int, default=30, help="FPS")
    parser.add_argument("--device", type=int, default=0, help="Индекс камеры (для webcam)")
    parser.add_argument("--stun", action="store_true", default=False, help="Использовать публичный STUN (для LAN не нужно)")

    args = parser.parse_args()
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()

