#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GStreamer WebRTC Video Streaming Client (LAN, NVIDIA NVDEC)

- Auto-discovery via UDP broadcast (port 5003) like existing client
- Signaling via HTTP:
  - POST /offer -> gets {peer_id, sdp(answer), type}
  - POST /candidate (trickle ICE to server)
  - GET  /candidates (poll server candidates)
- Media via webrtcbin, decode with nvh264dec when available

This client prefers zero-copy display via d3d11videosink/glimagesink for lowest latency.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import threading
import time
import urllib.error
import urllib.request
from logging import INFO
from typing import Any, Optional

from Logger import Logger, loggingLevels


def _setup_gstreamer_python_bindings() -> None:
    if platform.system() != "Windows":
        return
    gst_root = os.environ.get("GSTREAMER_1_0_ROOT_MSVC_X86_64") or r"C:\Program Files\GStreamer\1.0\msvc_x86_64"
    site_pkgs = os.path.join(gst_root, "lib", "site-packages")
    gi_typelib = os.path.join(gst_root, "lib", "girepository-1.0")
    gst_bin = os.path.join(gst_root, "bin")

    if os.path.isdir(site_pkgs) and site_pkgs not in os.sys.path:
        os.sys.path.insert(0, site_pkgs)

    try:
        if os.path.isdir(gst_bin):
            os.add_dll_directory(gst_bin)
    except Exception:
        pass

    if os.path.isdir(gi_typelib):
        os.environ.setdefault("GI_TYPELIB_PATH", gi_typelib)
        if gi_typelib not in os.environ.get("GI_TYPELIB_PATH", ""):
            os.environ["GI_TYPELIB_PATH"] = gi_typelib + os.pathsep + os.environ.get("GI_TYPELIB_PATH", "")


_setup_gstreamer_python_bindings()

try:
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    gi.require_version("GstSdp", "1.0")
    gi.require_version("GLib", "2.0")

    from gi.repository import GLib, Gst, GstSdp  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Не удалось импортировать/инициализировать GStreamer Python bindings (gi).\n"
        "Убедитесь, что установлен GStreamer MSVC_x86_64 и доступен GI:\n"
        r"  C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\site-packages\gi" "\n"
        f"Детали: {e}"
    )


def _require_gstwebrtc():
    try:
        gi.require_version("GstWebRTC", "1.0")
        from gi.repository import GstWebRTC  # type: ignore

        return GstWebRTC
    except Exception as _webrtc_err:
        raise RuntimeError(
            "В вашей установке GStreamer отсутствует GI namespace 'GstWebRTC'.\n"
            "Нужен файл 'GstWebRTC-1.0.typelib' в папке:\n"
            r"  C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\girepository-1.0" "\n\n"
            "Без него Python не может управлять WebRTC сигналингом (offer/answer/ICE) через webrtcbin.\n"
            "Решение: доустановить typelibs WebRTC или использовать окружение GStreamer+PyGObject,\n"
            "в котором доступен GstWebRTC.\n\n"
            f"Детали: {_webrtc_err}"
        )


DISCOVERY_PORT = 5003


def discover_server(timeout: int = 5, logger: Optional[Logger] = None) -> Optional[dict[str, Any]]:
    """Автоматически находит WebRTC сервер через UDP broadcast."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1)
        sock.bind(("", DISCOVERY_PORT))
        start = time.time()
        if logger:
            logger.info(f"Поиск сервера (UDP {DISCOVERY_PORT}, timeout {timeout}s)...", source="client_gst")
        while time.time() - start < timeout:
            try:
                data, addr = sock.recvfrom(65535)
                info = json.loads(data.decode("utf-8"))
                if not isinstance(info, dict):
                    continue
                # accept both backends
                if info.get("backend") not in ("gst-webrtc", "webrtc"):
                    continue
                info["host"] = addr[0]
                sock.close()
                if logger:
                    # В Windows консоль часто в cp1251 — символ '✓' ломает логирование (UnicodeEncodeError).
                    logger.info(f"[OK] Сервер найден: {info.get('host')}:{info.get('signal_port')}", source="client_gst")
                return info
            except socket.timeout:
                continue
            except Exception:
                continue
        sock.close()
        return None
    except Exception:
        return None


def _http_json(url: str, payload: Optional[dict[str, Any]] = None, timeout: float = 5.0) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _element_exists(name: str) -> bool:
    return Gst.ElementFactory.find(name) is not None


def _best_sink_element() -> str:
    for name in ("d3d11videosink", "glimagesink", "autovideosink"):
        if _element_exists(name):
            return name
    return "autovideosink"


class GstWebRTCClient:
    def __init__(self, host: str, port: int, logger: Logger, webrtc_latency_ms: int = 0):
        self.host = host
        self.port = int(port)
        self.logger = logger
        self.webrtc_latency_ms = int(webrtc_latency_ms)

        self.peer_id: Optional[str] = None
        self._candidate_since = 0
        self._stop = threading.Event()

        self.pipeline: Optional[Gst.Pipeline] = None
        self.webrtcbin: Optional[Gst.Element] = None

        self._mainloop = GLib.MainLoop()

    @property
    def signal_base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        Gst.init(None)
        self._build_pipeline()
        assert self.pipeline is not None
        self.pipeline.set_state(Gst.State.PLAYING)

        # start polling candidates
        poll_t = threading.Thread(target=self._poll_remote_candidates, daemon=True)
        poll_t.start()

        self.logger.info("Запуск клиента (GStreamer/NVDEC). Q/ESC — закрыть окно.", source="client_gst")
        try:
            self._mainloop.run()
        finally:
            self.stop()

    def stop(self):
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            self._mainloop.quit()
        except Exception:
            pass

    def _build_pipeline(self):
        GstWebRTC = _require_gstwebrtc()
        # webrtcbin only; decoding chain is attached on pad-added
        pipe_str = f"webrtcbin name=webrtc bundle-policy=max-bundle latency={self.webrtc_latency_ms}"
        self.logger.info(f"GST pipeline: {pipe_str}", source="client_gst")
        pipeline = Gst.parse_launch(pipe_str)
        if not isinstance(pipeline, Gst.Pipeline):
            pipeline = pipeline  # type: ignore[assignment]
        webrtcbin = pipeline.get_by_name("webrtc")
        if webrtcbin is None:
            raise RuntimeError("Не найден webrtcbin в pipeline.")

        self.pipeline = pipeline
        self.webrtcbin = webrtcbin

        # Request recvonly transceiver for H264 RTP
        caps = Gst.Caps.from_string("application/x-rtp,media=video,encoding-name=H264,payload=96")
        webrtcbin.emit("add-transceiver", GstWebRTC.WebRTCRTPTransceiverDirection.RECVONLY, caps)

        webrtcbin.connect("on-negotiation-needed", self._on_negotiation_needed)
        webrtcbin.connect("on-ice-candidate", self._on_ice_candidate)
        webrtcbin.connect("pad-added", self._on_pad_added)

    def _on_negotiation_needed(self, _element):
        # Create offer and send to server
        assert self.webrtcbin is not None

        def on_offer_created(promise: Gst.Promise, _user_data=None):
            try:
                reply = promise.get_reply()
                offer = reply.get_value("offer")
                self.webrtcbin.emit("set-local-description", offer, Gst.Promise.new())
                sdp_text = offer.sdp.as_text()
                self.logger.info("Offer created, sending to server...", source="client_gst")

                resp = _http_json(
                    f"{self.signal_base}/offer",
                    {"sdp": sdp_text, "type": "offer"},
                    timeout=15,
                )
                if resp.get("error"):
                    raise RuntimeError(str(resp.get("error")))
                self.peer_id = str(resp.get("peer_id") or "")
                answer_sdp = str(resp.get("sdp") or "")
                if not self.peer_id or not answer_sdp:
                    raise RuntimeError("Invalid answer from server.")

                # Set remote description (answer)
                res, sdpmsg = GstSdp.sdp_message_new()
                if res != GstSdp.SDPResult.OK:
                    raise RuntimeError(f"SDP init error: {res}")
                res = GstSdp.sdp_message_parse_buffer(bytes(answer_sdp, "utf-8"), sdpmsg)
                if res != GstSdp.SDPResult.OK:
                    raise RuntimeError(f"SDP parse error: {res}")
                answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg)
                self.webrtcbin.emit("set-remote-description", answer, Gst.Promise.new())
                self.logger.info(f"Answer set (peer_id={self.peer_id})", source="client_gst")
            except Exception as e:
                self.logger.error(f"Negotiation error: {e}", source="client_gst")
                self.stop()

        self.webrtcbin.emit("create-offer", None, Gst.Promise.new_with_change_func(on_offer_created, None))

    def _on_ice_candidate(self, _element, sdp_mline_index: int, candidate: str):
        # Send local ICE candidates to server (trickle)
        if not self.peer_id:
            return
        try:
            _http_json(
                f"{self.signal_base}/candidate",
                {"peer_id": self.peer_id, "sdpMLineIndex": int(sdp_mline_index), "candidate": str(candidate)},
                timeout=5,
            )
        except Exception:
            # best-effort
            return

    def _poll_remote_candidates(self):
        # Poll candidates from server and add to webrtcbin
        while not self._stop.is_set():
            try:
                if not self.peer_id or not self.webrtcbin:
                    time.sleep(0.1)
                    continue

                url = f"{self.signal_base}/candidates?peer_id={self.peer_id}&since={self._candidate_since}"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                cands = data.get("candidates") or []
                nxt = int(data.get("next") or self._candidate_since)
                self._candidate_since = nxt
                for c in cands:
                    try:
                        mline = int(c.get("sdpMLineIndex") or 0)
                        cand = str(c.get("candidate") or "")
                        if cand:
                            # must run in GLib thread
                            GLib.idle_add(self.webrtcbin.emit, "add-ice-candidate", mline, cand)
                    except Exception:
                        continue
            except urllib.error.URLError:
                pass
            except Exception:
                pass
            time.sleep(0.05)

    def _on_pad_added(self, _element, pad: Gst.Pad):
        # Attach depay/decode chain for incoming RTP (H264)
        if not self.pipeline:
            return

        caps = pad.get_current_caps()
        caps_str = caps.to_string() if caps else ""
        if "application/x-rtp" not in caps_str:
            return

        self.logger.info(f"Incoming pad caps: {caps_str}", source="client_gst")

        queue = Gst.ElementFactory.make("queue", None)
        depay = Gst.ElementFactory.make("rtph264depay", None)
        h264parse = Gst.ElementFactory.make("h264parse", None)

        decoder_name = "nvh264dec" if _element_exists("nvh264dec") else "avdec_h264"
        decoder = Gst.ElementFactory.make(decoder_name, None)
        videoconvert = Gst.ElementFactory.make("videoconvert", None)

        sink_name = _best_sink_element()
        sink = Gst.ElementFactory.make(sink_name, None)
        if sink is None:
            sink = Gst.ElementFactory.make("autovideosink", None)

        if sink is not None:
            try:
                sink.set_property("sync", False)
            except Exception:
                pass

        for el in (queue, depay, h264parse, decoder, videoconvert, sink):
            if el is None:
                self.logger.error("Не удалось создать элементы decode pipeline.", source="client_gst")
                return
            self.pipeline.add(el)

        if not Gst.Element.link_many(queue, depay, h264parse, decoder, videoconvert, sink):
            self.logger.error("Не удалось связать элементы decode pipeline.", source="client_gst")
            return

        # Link incoming pad to queue sink pad
        sinkpad = queue.get_static_pad("sink")
        if sinkpad is None:
            return
        if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
            self.logger.error("Не удалось линковать входящий RTP pad.", source="client_gst")
            return

        # sync states
        for el in (queue, depay, h264parse, decoder, videoconvert, sink):
            el.sync_state_with_parent()

        self.logger.info(f"Decoder: {decoder_name} | Sink: {sink_name}", source="client_gst")

        # Optional: simple keyboard watcher for Q/ESC for sinks that create a window
        def _key_poll():
            # Nothing portable here without GUI bindings; user can Ctrl+C.
            return True

        GLib.timeout_add(1000, _key_poll)


def main():
    parser = argparse.ArgumentParser(description="GStreamer WebRTC Video Streaming Client (LAN, NVDEC)")
    parser.add_argument("--host", type=str, default="auto", help='IP сервера или "auto"')
    parser.add_argument("--signal-port", type=int, default=8080)
    parser.add_argument("--auto", action="store_true", default=False)
    parser.add_argument("--webrtc-latency", type=int, default=0, help="webrtcbin jitterbuffer latency (ms)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Проверить элементы/сборку pipeline и выйти")
    parser.add_argument("--log-level", type=str, default="info", choices=["spam", "debug", "verbose", "info", "warning", "error", "critical"])
    parser.add_argument("--log-path", type=str, default="logs")
    args = parser.parse_args()

    log_level = loggingLevels.get(args.log_level, INFO)
    logger = Logger("client_gst", args.log_path, log_level)

    # Нужно инициализировать GStreamer перед любыми вызовами Gst.*
    Gst.init(None)

    auto = args.auto or (str(args.host).lower() == "auto")
    host = str(args.host)
    port = int(args.signal_port)

    if auto:
        info = discover_server(timeout=5, logger=logger)
        if not info:
            raise SystemExit("Сервер не найден в локальной сети.")
        host = str(info.get("host") or host)
        port = int(info.get("signal_port") or port)

    logger.info(f"Подключение к {host}:{port}", source="client_gst")
    logger.info(f"GStreamer: {Gst.version_string()}", source="client_gst")
    logger.info(f"nvh264dec: {'OK' if _element_exists('nvh264dec') else 'MISSING'}", source="client_gst")
    logger.info(f"Video sink: {_best_sink_element()}", source="client_gst")

    if args.dry_run:
        # Just validate base GStreamer availability without requiring GstWebRTC typelibs.
        Gst.init(None)
        pipe_str = f"webrtcbin name=webrtc bundle-policy=max-bundle latency={int(args.webrtc_latency)}"
        pipeline = Gst.parse_launch(pipe_str)
        if isinstance(pipeline, Gst.Pipeline):
            pipeline.set_state(Gst.State.READY)
            pipeline.set_state(Gst.State.NULL)
        logger.info("Dry-run: base pipeline OK (note: negotiation requires GstWebRTC typelibs)", source="client_gst")
        return

    client = GstWebRTCClient(host, port, logger, webrtc_latency_ms=int(args.webrtc_latency))
    client.start()


if __name__ == "__main__":
    main()

