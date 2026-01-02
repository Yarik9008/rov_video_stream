#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebRTC Video Streaming Client (LAN)

- Автоматически находит сервер по UDP broadcast
- Делаёт WebRTC offer/answer через HTTP signaling
- Показывает видео через OpenCV
"""

import argparse
import asyncio
import json
import socket
import time


DISCOVERY_PORT = 5003


def discover_server(timeout: int = 5):
    """Автоматически находит WebRTC сервер через UDP broadcast."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1)
        sock.bind(("", DISCOVERY_PORT))

        print(f"Поиск сервера в локальной сети (таймаут: {timeout} сек)...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                data, _addr = sock.recvfrom(65535)
                server_info = json.loads(data.decode("utf-8"))
                if server_info.get("backend") != "webrtc":
                    continue
                # Важно: payload может содержать "не тот" IP (другая подсеть/VPN/127.0.0.1).
                # Самый надёжный адрес сервера — это IP отправителя UDP пакета.
                sender_ip = _addr[0]
                server_info["host"] = sender_ip
                sock.close()
                print(f"✓ Сервер найден (WebRTC): {server_info.get('host')}:{server_info.get('signal_port')}")
                return server_info
            except socket.timeout:
                continue
            except Exception:
                continue
        sock.close()
        return None
    except Exception:
        return None


def _lazy_imports():
    missing_modules = []
    try:
        import aiohttp  # type: ignore
    except ImportError as e:
        missing_modules.append(("aiohttp", str(e)))
    
    try:
        import cv2  # type: ignore
    except ImportError as e:
        missing_modules.append(("opencv-python", str(e)))
    
    try:
        from aiortc import (  # type: ignore
            RTCPeerConnection,
            RTCSessionDescription,
            RTCConfiguration,
            RTCIceServer,
            RTCRtpSender,
        )
    except ImportError as e:
        missing_modules.append(("aiortc", str(e)))
    
    if missing_modules:
        modules_str = " ".join([m[0] for m in missing_modules])
        details = "\n".join([f"  - {m[0]}: {m[1]}" for m in missing_modules])
        raise RuntimeError(
            f"Для WebRTC нужны зависимости. Установите недостающие модули:\n"
            f"  pip install {modules_str}\n\n"
            f"Или установите все зависимости:\n"
            f"  pip install aiohttp opencv-python aiortc\n\n"
            f"Детали ошибок:\n{details}"
        )
    
    return aiohttp, cv2, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender


def _prefer_h264(transceiver):
    aiohttp, cv2, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender = _lazy_imports()
    try:
        caps = RTCRtpSender.getCapabilities("video")
        codecs = [c for c in caps.codecs if c.mimeType.lower() in ("video/h264", "video/vp8")]
        codecs.sort(key=lambda c: 0 if c.mimeType.lower() == "video/h264" else 1)
        if codecs:
            transceiver.setCodecPreferences(codecs)
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


async def run_client(host: str, signal_port: int, stun: bool):
    aiohttp, cv2, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender = _lazy_imports()

    cfg = RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])] if stun else []
    )
    pc = RTCPeerConnection(configuration=cfg)

    transceiver = pc.addTransceiver("video", direction="recvonly")
    _prefer_h264(transceiver)

    stop_event = asyncio.Event()
    stats_text = "Delay n/a"
    debug_printed = False

    async def stats_loop():
        nonlocal stats_text
        while not stop_event.is_set():
            try:
                stats = await pc.getStats()
                
                # Временный отладочный вывод (один раз)
                nonlocal debug_printed
                if not debug_printed and len(stats) > 0:
                    print("\n=== Отладка: Доступные stats ===")
                    for stat_id, stat in list(stats.items())[:5]:  # Первые 5 для краткости
                        print(f"\nStat ID: {stat_id}")
                        print(f"  Type: {getattr(stat, 'type', 'N/A')}")
                        # Проверяем ключевые поля
                        for attr in ["state", "kind", "currentRoundTripTime", "totalRoundTripTime", 
                                     "jitterBufferDelay", "jitterBufferTargetDelay", "jitterBufferEmittedCount"]:
                            val = getattr(stat, attr, None)
                            if val is not None:
                                print(f"  {attr}: {val}")
                    print("===============================\n")
                    debug_printed = True
                
                rtt_ms = None
                jb_ms = None

                # Candidate-pair RTT (network) - более гибкий поиск
                for stat_id, stat in stats.items():
                    try:
                        if not hasattr(stat, "type"):
                            continue
                        if stat.type == "candidate-pair":
                            # Пробуем получить RTT из разных полей
                            for rtt_attr in ["currentRoundTripTime", "totalRoundTripTime", "roundTripTime"]:
                                crt = getattr(stat, rtt_attr, None)
                                if crt is not None:
                                    try:
                                        crt_val = float(crt)
                                        if crt_val > 0:
                                            rtt_ms = int(crt_val * 1000)
                                            break
                                    except (TypeError, ValueError):
                                        continue
                            if rtt_ms is not None:
                                break
                    except (AttributeError, TypeError, ValueError):
                        continue

                # Inbound RTP jitter buffer delay (playout) - более гибкий поиск
                for stat_id, stat in stats.items():
                    try:
                        if not hasattr(stat, "type"):
                            continue
                        if stat.type == "inbound-rtp":
                            # Проверяем kind, но не строго
                            kind = getattr(stat, "kind", "video")
                            if kind == "video" or kind is None:
                                # Пробуем разные поля для jitter buffer delay
                                for jb_attr in ["jitterBufferTargetDelay", "jitterBufferDelay"]:
                                    jbd_val = getattr(stat, jb_attr, None)
                                    if jbd_val is not None:
                                        try:
                                            jbd_float = float(jbd_val)
                                            if jbd_float > 0:
                                                if jb_attr == "jitterBufferTargetDelay":
                                                    jb_ms = int(jbd_float * 1000)
                                                else:
                                                    # Для jitterBufferDelay делим на emitted count если есть
                                                    emitted = getattr(stat, "jitterBufferEmittedCount", None)
                                                    if emitted and float(emitted) > 0:
                                                        jb_ms = int((jbd_float / float(emitted)) * 1000)
                                                    else:
                                                        jb_ms = int(jbd_float * 1000)
                                                break
                                        except (TypeError, ValueError):
                                            continue
                                if jb_ms is not None:
                                    break
                    except (AttributeError, TypeError, ValueError):
                        continue

                # Оценка задержки (ms): playout (JB) + половина RTT (в одну сторону)
                delay_est_ms = None
                if rtt_ms is not None and jb_ms is not None:
                    delay_est_ms = int(jb_ms + (rtt_ms / 2.0))
                elif jb_ms is not None:
                    delay_est_ms = int(jb_ms)
                elif rtt_ms is not None:
                    delay_est_ms = int(rtt_ms / 2.0)

                if delay_est_ms is None:
                    stats_text = "Delay n/a"
                else:
                    details = []
                    if rtt_ms is not None:
                        details.append(f"RTT {rtt_ms} ms")
                    if jb_ms is not None:
                        details.append(f"JB {jb_ms} ms")
                    suffix = f" ({' | '.join(details)})" if details else ""
                    stats_text = f"Delay {delay_est_ms} ms{suffix}"
            except Exception:
                pass
            await asyncio.sleep(0.5)

    asyncio.create_task(stats_loop())

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return

        async def consume():
            while True:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")

                # Overlay latency in the top-right corner
                try:
                    text = stats_text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.6
                    thickness = 2
                    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
                    x = max(8, int(img.shape[1]) - tw - 12)
                    y = 10 + th
                    cv2.rectangle(
                        img,
                        (x - 6, y - th - base - 6),
                        (x + tw + 6, y + base + 6),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(img, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
                except Exception:
                    pass

                cv2.imshow("WebRTC Video", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    stop_event.set()
                    break

        asyncio.create_task(consume())

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await _wait_ice_complete(pc)

    url = f"http://{host}:{int(signal_port)}/offer"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ) as resp:
            data = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))

    print("WebRTC соединение установлено. Нажмите Q/ESC в окне для выхода.")
    await stop_event.wait()

    await pc.close()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="WebRTC Video Streaming Client (LAN)")
    parser.add_argument("--host", type=str, default="auto", help='IP сервера или "auto" для автообнаружения')
    parser.add_argument("--signal-port", type=int, default=8080, help="Порт HTTP signaling (по умолчанию 8080)")
    parser.add_argument("--auto", action="store_true", default=False, help="Автообнаружение сервера в LAN")
    parser.add_argument("--stun", action="store_true", default=False, help="Использовать публичный STUN (для LAN не нужно)")
    args = parser.parse_args()

    auto = args.auto or (str(args.host).lower() == "auto")
    host = args.host
    signal_port = int(args.signal_port)

    if auto:
        info = discover_server(timeout=5)
        if not info:
            raise SystemExit("Сервер не найден в локальной сети.")
        host = info.get("host", host)
        signal_port = int(info.get("signal_port") or signal_port)

    asyncio.run(run_client(host, signal_port, args.stun))


if __name__ == "__main__":
    main()

