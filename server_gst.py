#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GStreamer WebRTC Video Streaming Server (LAN, NVIDIA NVENC)

- Signaling: HTTP (offer/answer + trickle ICE over HTTP)
  - POST /offer      -> {sdp,type}  returns {peer_id,sdp,type}
  - POST /candidate  -> {peer_id,sdpMLineIndex,candidate}
  - GET  /candidates?peer_id=...&since=... -> {candidates:[{sdpMLineIndex,candidate},...], next:int}
- Discovery: UDP broadcast compatible with existing client payloads
- Media: WebRTC via webrtcbin

Requirements:
- Installed GStreamer MSVC_x86_64 with webrtc + nvcodec plugins.
- Python bindings are shipped inside:
  C:\\Program Files\\GStreamer\\1.0\\msvc_x86_64\\lib\\site-packages
This script auto-adds that path and required DLL/type-lib directories.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from logging import INFO

from Logger import Logger, loggingLevels


def _setup_gstreamer_python_bindings() -> None:
    """
    Make `import gi` work with the official GStreamer MSVC bundle on Windows.
    """
    if platform.system() != "Windows":
        return

    gst_root = os.environ.get("GSTREAMER_1_0_ROOT_MSVC_X86_64") or r"C:\Program Files\GStreamer\1.0\msvc_x86_64"
    site_pkgs = os.path.join(gst_root, "lib", "site-packages")
    gi_typelib = os.path.join(gst_root, "lib", "girepository-1.0")
    gst_bin = os.path.join(gst_root, "bin")

    if os.path.isdir(site_pkgs) and site_pkgs not in os.sys.path:
        os.sys.path.insert(0, site_pkgs)

    # DLL search path for gi/_gi*.pyd and gstreamer DLLs
    try:
        if os.path.isdir(gst_bin):
            os.add_dll_directory(gst_bin)
    except Exception:
        pass

    # Typelib search path for GI
    if os.path.isdir(gi_typelib):
        os.environ.setdefault("GI_TYPELIB_PATH", gi_typelib)
        # Sometimes GI_TYPELIB_PATH must include existing value
        if gi_typelib not in os.environ.get("GI_TYPELIB_PATH", ""):
            os.environ["GI_TYPELIB_PATH"] = gi_typelib + os.pathsep + os.environ.get("GI_TYPELIB_PATH", "")


_setup_gstreamer_python_bindings()

try:
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    gi.require_version("GstSdp", "1.0")
    gi.require_version("GLib", "2.0")

    from gi.repository import GLib, Gst, GstSdp  # type: ignore
    # GstWebRTC импортируется лениво (см. _require_gstwebrtc()), чтобы --dry-run мог работать
    # даже если typelibs GstWebRTC не установлены.
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Не удалось импортировать/инициализировать GStreamer Python bindings (gi).\n"
        "Проверьте, что установлен GStreamer MSVC_x86_64 и доступна папка:\n"
        r"  C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\site-packages" "\n"
        "Также убедитесь, что вы используете Python той же версии, что и bindings.\n"
        f"Детали: {e}"
    )


def _require_gstwebrtc():
    """
    GstWebRTC GI namespace может отсутствовать в некоторых Windows сборках.
    Проверяем лениво, чтобы --dry-run работал без GstWebRTC typelibs.
    """
    try:
        gi.require_version("GstWebRTC", "1.0")
        from gi.repository import GstWebRTC  # type: ignore

        return GstWebRTC
    except Exception as _webrtc_err:
        raise RuntimeError(
            "В вашей установке GStreamer отсутствует GI namespace 'GstWebRTC'.\n"
            "Нужен файл 'GstWebRTC-1.0.typelib' в папке:\n"
            r"  C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\girepository-1.0" "\n\n"
            "Без него Python не сможет делать SDP offer/answer и ICE через webrtcbin.\n"
            "Решение: установить сборку GStreamer, где присутствует GstWebRTC typelibs.\n\n"
            f"Детали: {_webrtc_err}"
        )


def get_local_ip() -> str:
    """Best-effort локальный IPv4 (LAN)."""
    candidates: list[str] = []
    try:
        _name, _aliases, addrs = socket.gethostbyname_ex(socket.gethostname())
        candidates.extend(addrs)
    except Exception:
        pass
    for target in (("1.1.1.1", 80), ("8.8.8.8", 80), ("10.255.255.255", 1)):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            try:
                s.connect(target)
                candidates.append(s.getsockname()[0])
            finally:
                s.close()
        except Exception:
            continue
    for ip in candidates:
        if ip and not ip.startswith(("127.", "169.254.")):
            return ip
    return "127.0.0.1"


def get_broadcast_address(ip: str) -> str:
    try:
        parts = ip.split(".")
        if len(parts) == 4:
            parts[-1] = "255"
            return ".".join(parts)
    except Exception:
        pass
    return "255.255.255.255"


class DiscoveryBroadcaster:
    DISCOVERY_PORT = 5003

    def __init__(self, payload_factory, enabled: bool):
        self._payload_factory = payload_factory
        self._enabled = enabled
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self._enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        t = self._thread
        if t:
            try:
                t.join(timeout=1)
            except Exception:
                pass

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
                    targets = {get_broadcast_address(get_local_ip()), "255.255.255.255"}
                    for bcast in targets:
                        try:
                            sock.sendto(message, (bcast, self.DISCOVERY_PORT))
                        except Exception:
                            continue
                except Exception:
                    pass
                time.sleep(1.5)
            try:
                sock.close()
            except Exception:
                pass
        except Exception:
            return


def _element_exists(name: str) -> bool:
    return Gst.ElementFactory.find(name) is not None


def _make_best_video_sink() -> str:
    # минимальная задержка без копии в numpy
    for name in ("d3d11videosink", "glimagesink", "autovideosink"):
        if _element_exists(name):
            return name
    return "autovideosink"


def _make_video_source(device: int) -> str:
    # Windows: предпочитаем ksvideosrc
    if _element_exists("ksvideosrc"):
        return f"ksvideosrc device-index={int(device)}"
    if _element_exists("dshowvideosrc"):
        return f"dshowvideosrc device={int(device)}"
    # fallback
    return "autovideosrc"


def _nvenc_or_fallback(bitrate_kbps: int, fps: int) -> str:
    """
    Возвращает строку pipeline для H.264 encoder с максимально низкой задержкой.
    bitrate_kbps: kbit/sec (как в nvh264enc)
    """
    bitrate_kbps = max(100, int(bitrate_kbps))
    fps = max(1, int(fps))

    if _element_exists("nvh264enc"):
        # Максимально низкая задержка: no reordering, маленький GOP, SPS/PPS по IDR
        gop = fps  # ~1s GOP
        return (
            "nvh264enc "
            f"bitrate={bitrate_kbps} "
            "rc-mode=cbr "
            "preset=low-latency-hq "
            "tune=ultra-low-latency "
            "zerolatency=true "
            "bframes=0 "
            f"gop-size={gop} "
            "repeat-sequence-header=true"
        )

    # software fallback
    if _element_exists("x264enc"):
        # x264enc bitrate in kbit/sec
        keyint = fps
        return f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbps} key-int-max={keyint} bframes=0"

    # last resort
    return "openh264enc"


class _PeerSession:
    def __init__(self, peer_id: str, webrtcbin: Gst.Element):
        self.peer_id = peer_id
        self.webrtcbin = webrtcbin
        self._candidates: list[dict[str, Any]] = []
        self._cand_lock = threading.Lock()

    def add_local_candidate(self, sdp_mline_index: int, candidate: str):
        with self._cand_lock:
            self._candidates.append({"sdpMLineIndex": int(sdp_mline_index), "candidate": str(candidate)})

    def get_candidates_since(self, since: int) -> tuple[list[dict[str, Any]], int]:
        with self._cand_lock:
            since = max(0, int(since))
            out = self._candidates[since:]
            return out, since + len(out)


class GstWebRTCServer:
    def __init__(self, args, logger: Logger):
        self.args = args
        self.logger = logger

        self._mainloop = GLib.MainLoop()
        self._main_thread = threading.Thread(target=self._mainloop.run, daemon=True)

        self._sessions: dict[str, _PeerSession] = {}
        self._sessions_lock = threading.Lock()

        self._current_pipeline: Optional[Gst.Pipeline] = None
        self._current_peer_id: Optional[str] = None

    def start(self):
        Gst.init(None)
        self._main_thread.start()

    def stop(self):
        try:
            GLib.idle_add(self._stop_pipeline)
        except Exception:
            pass
        try:
            self._mainloop.quit()
        except Exception:
            pass

    def _stop_pipeline(self):
        if self._current_pipeline is not None:
            try:
                self._current_pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._current_pipeline = None
        self._current_peer_id = None
        return False

    def _build_pipeline(self) -> tuple[Gst.Pipeline, Gst.Element]:
        """
        Build pipeline for one peer. For simplicity, GPU mode supports one active peer at a time.
        """
        width = int(self.args.width)
        height = int(self.args.height)
        fps = int(self.args.fps)
        bitrate = int(self.args.bitrate)
        webrtc_latency_ms = int(self.args.webrtc_latency)

        webrtc_props = f"bundle-policy=max-bundle latency={webrtc_latency_ms}"
        if self.args.stun:
            webrtc_props += " stun-server=stun://stun.l.google.com:19302"

        if self.args.source == "file":
            file_path = str(Path(self.args.file).resolve())
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")

            if self.args.file_passthrough:
                # Passthrough: no re-encode (max quality + min latency), but no NVENC.
                pipe = (
                    f'filesrc location="{file_path}" ! qtdemux name=demux '
                    "demux.video_0 ! queue max-size-buffers=2 leaky=downstream ! "
                    "h264parse ! rtph264pay pt=96 config-interval=1 ! "
                    "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
                    f"webrtcbin name=webrtc {webrtc_props}"
                )
                self.logger.info("GST server: file passthrough (no re-encode)", source="server_gst")
            else:
                enc = _nvenc_or_fallback(bitrate, fps)
                pipe = (
                    f'filesrc location="{file_path}" ! qtdemux ! queue max-size-buffers=2 leaky=downstream ! '
                    "decodebin ! videoconvert ! video/x-raw,format=NV12 ! "
                    f"{enc} ! h264parse ! rtph264pay pt=96 config-interval=1 ! "
                    "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
                    f"webrtcbin name=webrtc {webrtc_props}"
                )
                self.logger.info("GST server: file decode+encode (NVENC preferred)", source="server_gst")
        else:
            src = _make_video_source(int(self.args.device))
            enc = _nvenc_or_fallback(bitrate, fps)
            # Note: we keep pipeline mostly CPU->NVENC. For even lower latency, can add d3d11/cuda upload.
            caps = f"video/x-raw,width={width},height={height},framerate={fps}/1"
            pipe = (
                f"{src} ! {caps} ! queue max-size-buffers=2 leaky=downstream ! "
                "videoconvert ! video/x-raw,format=NV12 ! "
                f"{enc} ! h264parse ! rtph264pay pt=96 config-interval=1 ! "
                "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
                f"webrtcbin name=webrtc {webrtc_props}"
            )
            self.logger.info("GST server: webcam encode (NVENC preferred)", source="server_gst")

        self.logger.info(f"GST pipeline: {pipe}", source="server_gst")
        pipeline = Gst.parse_launch(pipe)
        if not isinstance(pipeline, Gst.Pipeline):
            pipeline = pipeline  # type: ignore[assignment]
        webrtcbin = pipeline.get_by_name("webrtc")
        if webrtcbin is None:
            raise RuntimeError("Не найден webrtcbin в pipeline.")
        return pipeline, webrtcbin

    def handle_offer(self, offer_sdp: str, offer_type: str) -> dict[str, Any]:
        """
        Called from HTTP thread. Schedules work on GLib main loop and blocks for answer.
        """
        GstWebRTC = _require_gstwebrtc()
        peer_id = str(uuid.uuid4())
        done = threading.Event()
        result: dict[str, Any] = {}
        error: list[str] = []

        def _do_offer():
            try:
                # stop previous peer for simplicity
                self._stop_pipeline()

                pipeline, webrtcbin = self._build_pipeline()
                self._current_pipeline = pipeline
                self._current_peer_id = peer_id

                sess = _PeerSession(peer_id, webrtcbin)
                with self._sessions_lock:
                    self._sessions[peer_id] = sess

                def on_ice_candidate(_element, mlineindex, candidate):
                    try:
                        sess.add_local_candidate(int(mlineindex), str(candidate))
                    except Exception:
                        return

                webrtcbin.connect("on-ice-candidate", on_ice_candidate)

                # set remote description
                res, sdpmsg = GstSdp.sdp_message_new()
                if res != GstSdp.SDPResult.OK:
                    raise RuntimeError(f"SDP init error: {res}")
                res = GstSdp.sdp_message_parse_buffer(bytes(offer_sdp, "utf-8"), sdpmsg)
                if res != GstSdp.SDPResult.OK:
                    raise RuntimeError(f"SDP parse error: {res}")

                if str(offer_type).lower() != "offer":
                    raise RuntimeError(f"Unsupported SDP type: {offer_type}")

                offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdpmsg)

                promise = Gst.Promise.new()
                webrtcbin.emit("set-remote-description", offer, promise)
                promise.interrupt()

                def on_answer_created(promise: Gst.Promise, _user_data=None):
                    try:
                        reply = promise.get_reply()
                        answer = reply.get_value("answer")
                        webrtcbin.emit("set-local-description", answer, Gst.Promise.new())
                        sdp_text = answer.sdp.as_text()
                        result.update({"peer_id": peer_id, "type": "answer", "sdp": sdp_text})
                    except Exception as e:
                        error.append(str(e))
                    finally:
                        done.set()

                # create answer
                webrtcbin.emit("create-answer", None, Gst.Promise.new_with_change_func(on_answer_created, None))

                # start pipeline
                pipeline.set_state(Gst.State.PLAYING)
                return False
            except Exception as e:
                error.append(str(e))
                done.set()
                return False

        GLib.idle_add(_do_offer)
        done.wait(timeout=15)
        if error:
            raise RuntimeError(error[0])
        if not result:
            raise RuntimeError("Timeout creating answer.")
        return result

    def handle_remote_candidate(self, peer_id: str, sdp_mline_index: int, candidate: str) -> None:
        def _do():
            with self._sessions_lock:
                sess = self._sessions.get(peer_id)
            if not sess:
                return False
            try:
                sess.webrtcbin.emit("add-ice-candidate", int(sdp_mline_index), str(candidate))
            except Exception:
                pass
            return False

        GLib.idle_add(_do)

    def get_local_candidates(self, peer_id: str, since: int) -> dict[str, Any]:
        with self._sessions_lock:
            sess = self._sessions.get(peer_id)
        if not sess:
            return {"candidates": [], "next": int(since)}
        cands, nxt = sess.get_candidates_since(since)
        return {"candidates": cands, "next": nxt}


class SignalingHandler(BaseHTTPRequestHandler):
    server: "SignalingHTTPServer"  # typing: ignore[assignment]

    def _json_response(self, code: int, payload: dict[str, Any]):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/offer":
            body = self._read_json()
            try:
                sdp = str(body.get("sdp") or "")
                typ = str(body.get("type") or "offer")
                resp = self.server.webrtc_server.handle_offer(sdp, typ)
                self._json_response(200, resp)
            except Exception as e:
                self._json_response(500, {"error": str(e)})
            return

        if parsed.path == "/candidate":
            body = self._read_json()
            try:
                peer_id = str(body.get("peer_id") or "")
                mline = int(body.get("sdpMLineIndex") or 0)
                cand = str(body.get("candidate") or "")
                if peer_id and cand:
                    self.server.webrtc_server.handle_remote_candidate(peer_id, mline, cand)
                self._json_response(200, {"ok": True})
            except Exception as e:
                self._json_response(500, {"error": str(e)})
            return

        self._json_response(404, {"error": "not found"})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._json_response(200, {"status": "ok", "backend": "gst-webrtc"})
            return
        if parsed.path == "/candidates":
            qs = parse_qs(parsed.query or "")
            peer_id = (qs.get("peer_id") or [""])[0]
            since = int((qs.get("since") or ["0"])[0] or "0")
            self._json_response(200, self.server.webrtc_server.get_local_candidates(peer_id, since))
            return
        self._json_response(404, {"error": "not found"})


class SignalingHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, webrtc_server: GstWebRTCServer):
        super().__init__(server_address, RequestHandlerClass)
        self.webrtc_server = webrtc_server


def main():
    parser = argparse.ArgumentParser(description="GStreamer WebRTC Video Streaming Server (LAN, NVENC)")
    parser.add_argument("--source", choices=["webcam", "file"], default="webcam")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--file-passthrough", action="store_true", default=False, help="Для file: отправлять H.264 как есть (без NVENC)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--signal-port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--bitrate", type=int, default=250000, help="kbit/sec for encoder (e.g. 250000 = 250 Mbps)")
    parser.add_argument("--webrtc-latency", type=int, default=0, help="Jitterbuffer latency inside webrtcbin (ms)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stun", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False, help="Проверить сборку pipeline и выйти (без сети/стрима)")
    parser.add_argument("--log-level", type=str, default="info", choices=["spam", "debug", "verbose", "info", "warning", "error", "critical"])
    parser.add_argument("--log-path", type=str, default="logs")
    args = parser.parse_args()

    if args.source == "file" and not args.file:
        raise SystemExit("Укажите --file для --source file")

    log_level = loggingLevels.get(args.log_level, INFO)
    logger = Logger("server_gst", args.log_path, log_level)

    # Dry-run: just verify pipeline parse + element availability
    if args.dry_run:
        Gst.init(None)
        logger.info("Dry-run: checking GStreamer elements/pipeline...", source="server_gst")
        logger.info(f"GStreamer: {Gst.version_string()}", source="server_gst")
        logger.info(f"webrtcbin: {'OK' if _element_exists('webrtcbin') else 'MISSING'}", source="server_gst")
        logger.info(f"nvh264enc: {'OK' if _element_exists('nvh264enc') else 'MISSING'}", source="server_gst")
        logger.info(f"nvh264dec: {'OK' if _element_exists('nvh264dec') else 'MISSING'}", source="server_gst")
        tmp = GstWebRTCServer(args, logger)
        pipeline, _webrtc = tmp._build_pipeline()
        pipeline.set_state(Gst.State.READY)
        pipeline.set_state(Gst.State.NULL)
        logger.info("Dry-run: OK", source="server_gst")
        return

    local_ip = get_local_ip()
    payload = {
        "backend": "gst-webrtc",
        "host": local_ip,
        "signal_port": int(args.signal_port),
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
    }

    webrtc_server = GstWebRTCServer(args, logger)
    webrtc_server.start()

    broadcaster = DiscoveryBroadcaster(lambda: payload, enabled=(args.host != "127.0.0.1"))
    broadcaster.start()

    httpd = SignalingHTTPServer((args.host, int(args.signal_port)), SignalingHandler, webrtc_server=webrtc_server)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    logger.info("Запуск GStreamer WebRTC сервера (NVENC)...", source="server_gst")
    logger.info(f"GStreamer: {Gst.version_string()}", source="server_gst")
    logger.info(f"Signaling: http://{local_ip}:{int(args.signal_port)}", source="server_gst")
    logger.info(f"Discovery UDP: {DiscoveryBroadcaster.DISCOVERY_PORT}", source="server_gst")
    logger.info(f"webrtcbin: {'OK' if _element_exists('webrtcbin') else 'MISSING'}", source="server_gst")
    logger.info(f"nvh264enc: {'OK' if _element_exists('nvh264enc') else 'MISSING'}", source="server_gst")
    logger.info(f"nvh264dec: {'OK' if _element_exists('nvh264dec') else 'MISSING'}", source="server_gst")
    logger.info("Нажмите Ctrl+C для остановки.", source="server_gst")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            broadcaster.stop()
        except Exception:
            pass
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            webrtc_server.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

