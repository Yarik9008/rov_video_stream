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
    
    # Создаём окно заранее для гарантии отображения
    window_name = "WebRTC Video"
    try:
        cv2.destroyWindow(window_name)  # Закрываем, если уже существует
    except:
        pass
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    # Показываем черный экран, чтобы окно появилось сразу
    import numpy as np
    blank_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(blank_img, "Подключение...", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imshow(window_name, blank_img)
    cv2.waitKey(1)  # Обновляем окно

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return

        async def consume():
            while True:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                cv2.imshow(window_name, img)
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
        cv2.destroyWindow(window_name)
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

