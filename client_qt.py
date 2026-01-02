#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebRTC Video Streaming Client (PyQt GUI)

- Автоматически находит сервер по UDP broadcast
- Делаёт WebRTC offer/answer через HTTP signaling
- Показывает видео через PyQt
"""

import argparse
import asyncio
import json
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Optional
from logging import INFO

from Logger import Logger, loggingLevels

PYQT6 = False
try:
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QLabel,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QSizePolicy,
        QWidget,
        QMessageBox,
        QLineEdit,
        QFormLayout,
    )
    PYQT6 = True
except ImportError:
    try:
        from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
        from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QFileDialog,
            QLabel,
            QMainWindow,
            QPushButton,
            QVBoxLayout,
            QHBoxLayout,
            QSizePolicy,
            QWidget,
            QMessageBox,
            QLineEdit,
            QFormLayout,
        )
    except ImportError:
        raise RuntimeError(
            "Для GUI нужен PyQt5 или PyQt6. Установите:\n"
            "  pip install PyQt5\n"
            "или\n"
            "  pip install PyQt6"
        )


DISCOVERY_PORT = 5003

# Предполагаемое начальное разрешение видео
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720


def apply_dark_theme(app: QApplication):
    """Жёстко задаём тёмную тему (Fusion + QPalette)."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    palette = QPalette()
    # Base colors
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(45, 140, 255))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(10, 10, 10))

    # Disabled
    try:
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(130, 130, 130))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(130, 130, 130))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(130, 130, 130))
    except Exception:
        pass

    app.setPalette(palette)

    # Tooltips
    app.setStyleSheet(
        "QToolTip { color: #dcdcdc; background-color: #2d2d2d; border: 1px solid #4a4a4a; }"
    )


class Mp4Recorder:
    """Запись входящего видео в MP4 (в фоне), чтобы не тормозить UI."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        self.path = str(path)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)

        self._q: "queue.Queue" = queue.Queue(maxsize=300)
        self._stop = threading.Event()
        self._thread = None
        self._dropped = 0

        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Для записи MP4 нужен opencv-python: {e}")

        self._cv2 = cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Не удалось открыть VideoWriter для: {self.path}")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def dropped(self) -> int:
        return int(self._dropped)

    def write(self, frame_bgr):
        """frame_bgr: numpy array uint8 HxWx3 (BGR)"""
        if self._stop.is_set():
            return
        try:
            self._q.put_nowait(frame_bgr)
        except queue.Full:
            self._dropped += 1

    def close(self):
        self._stop.set()
        t = self._thread
        if t:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        try:
            self._writer.release()
        except Exception:
            pass

    def _run(self):
        cv2 = self._cv2
        while (not self._stop.is_set()) or (not self._q.empty()):
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                # гарантируем размер
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                if (w, h) != (self.width, self.height):
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                self._writer.write(frame)
            except Exception:
                continue


def discover_server(timeout: int = 5, logger: Logger = None):
    """Автоматически находит WebRTC сервер через UDP broadcast."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1)
        sock.bind(("", DISCOVERY_PORT))

        start = time.time()
        while time.time() - start < timeout:
            try:
                data, _addr = sock.recvfrom(65535)
                server_info = json.loads(data.decode("utf-8"))
                if server_info.get("backend") != "webrtc":
                    continue
                sender_ip = _addr[0]
                server_info["host"] = sender_ip
                sock.close()
                if logger:
                    logger.info(f"Сервер найден (WebRTC): {server_info.get('host')}:{server_info.get('signal_port')}", source="client_qt")
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
        import numpy as np  # type: ignore
    except ImportError as e:
        missing_modules.append(("numpy", str(e)))
    
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
            f"  pip install aiohttp numpy aiortc\n\n"
            f"Детали ошибок:\n{details}"
        )
    
    return aiohttp, np, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender


def _prefer_h264(transceiver):
    _, _, RTCPeerConnection, _, _, _, RTCRtpSender = _lazy_imports()
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


class WebRTCClientThread(QThread):
    """Поток для выполнения WebRTC клиента в asyncio event loop."""
    frame_received = pyqtSignal(tuple)  # (bytes, width, height)
    status_changed = pyqtSignal(str)  # статус соединения
    error_occurred = pyqtSignal(str)  # ошибка

    def __init__(self, host: str, signal_port: int, stun: bool, logger: Logger = None):
        super().__init__()
        self.host = host
        self.signal_port = signal_port
        self.stun = stun
        self.logger = logger
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pc = None
        self._stop_event: Optional[asyncio.Event] = None

    def run(self):
        """Запускает asyncio event loop в отдельном потоке."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_client())
        except Exception as e:
            self.error_occurred.emit(f"Ошибка: {e}")
        finally:
            try:
                # Отменяем все задачи
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()

    async def _run_client(self):
        aiohttp, np, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender = _lazy_imports()

        if str(self.host).lower() == "auto":
            error_msg = "Host='auto' недопустим здесь. Используйте автообнаружение из GUI (кнопка/--auto)."
            if self.logger:
                self.logger.error(error_msg, source="client_qt")
            self.error_occurred.emit(error_msg)
            return

        self.status_changed.emit("Подключение...")
        if self.logger:
            self.logger.info(f"Подключение к {self.host}:{self.signal_port}", source="client_qt")
        
        cfg = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])] if self.stun else []
        )
        self._pc = RTCPeerConnection(configuration=cfg)

        @self._pc.on("iceconnectionstatechange")
        def on_ice_state_change():
            try:
                state = f"ICE: {self._pc.iceConnectionState}"
                self.status_changed.emit(state)
                if self.logger:
                    self.logger.debug(f"ICE state: {self._pc.iceConnectionState}", source="client_qt")
            except Exception:
                pass

        @self._pc.on("connectionstatechange")
        def on_conn_state_change():
            try:
                state = f"PC: {self._pc.connectionState}"
                self.status_changed.emit(state)
                if self.logger:
                    self.logger.info(f"Connection state: {self._pc.connectionState}", source="client_qt")
            except Exception:
                pass

        transceiver = self._pc.addTransceiver("video", direction="recvonly")
        _prefer_h264(transceiver)

        self._stop_event = asyncio.Event()

        @self._pc.on("track")
        def on_track(track):
            if track.kind != "video":
                return

            async def consume():
                self.status_changed.emit("Video track получен")
                if self.logger:
                    self.logger.info("Video track получен", source="client_qt")
                try:
                    while not self._stop_event.is_set():
                        try:
                            frame = await track.recv()
                            img = frame.to_ndarray(format="bgr24")
                            # Конвертируем BGR (OpenCV) в RGB для Qt
                            img_rgb = img[..., ::-1]  # BGR -> RGB
                            # Создаём непрерывную копию массива
                            img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
                            # Конвертируем в bytes для безопасной передачи через сигнал
                            img_bytes = img_rgb.tobytes()
                            self.frame_received.emit((img_bytes, img_rgb.shape[1], img_rgb.shape[0]))
                        except Exception as e:
                            if not self._stop_event.is_set():
                                error_msg = f"Ошибка получения кадра: {e}"
                                if self.logger:
                                    self.logger.error(error_msg, source="client_qt")
                                self.error_occurred.emit(error_msg)
                                import traceback
                                traceback.print_exc()
                            break
                except Exception as e:
                    if not self._stop_event.is_set():
                        error_msg = f"Ошибка потока: {e}"
                        if self.logger:
                            self.logger.error(error_msg, source="client_qt")
                        self.error_occurred.emit(error_msg)
                        import traceback
                        traceback.print_exc()

            asyncio.create_task(consume())

        try:
            self.status_changed.emit("Создание offer...")
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)
            self.status_changed.emit("ICE gathering...")
            await _wait_ice_complete(self._pc)

            url = f"http://{self.host}:{int(self.signal_port)}/offer"
            timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_connect=5, sock_read=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                self.status_changed.emit("Отправка offer...")
                async with session.post(
                    url,
                    json={"sdp": self._pc.localDescription.sdp, "type": self._pc.localDescription.type},
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"/offer HTTP {resp.status}: {text}")
                    self.status_changed.emit("Получение answer...")
                    data = await resp.json()

            await self._pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))
            self.status_changed.emit("RemoteDescription установлено")
            
            # Ждём пока соединение не будет прервано
            await self._stop_event.wait()

        except Exception as e:
            error_msg = f"Ошибка соединения: {e}"
            if self.logger:
                self.logger.error(error_msg, source="client_qt")
            self.error_occurred.emit(error_msg)
        finally:
            self.status_changed.emit("Отключение...")
            if self.logger:
                self.logger.info("Отключение от сервера", source="client_qt")
            try:
                if self._pc:
                    await self._pc.close()
            except Exception:
                pass
            self.status_changed.emit("Отключено")
            if self.logger:
                self.logger.info("Отключено", source="client_qt")

    def stop(self):
        """Останавливает клиент."""
        if self._stop_event and self._loop and not self._loop.is_closed():
            # Устанавливаем событие остановки в event loop
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self.wait(3000)  # Ждём максимум 3 секунды


class DiscoverThread(QThread):
    """Поток для поиска сервера (чтобы не трогать UI из threading.Thread)."""
    found = pyqtSignal(object)  # dict | None

    def __init__(self, timeout: int = 5, logger: Logger = None):
        super().__init__()
        self._timeout = int(timeout)
        self.logger = logger

    def run(self):
        try:
            info = discover_server(timeout=self._timeout, logger=self.logger)
        except Exception:
            info = None
        self.found.emit(info)


class VideoWidget(QLabel):
    """Виджет для отображения видео."""
    
    def __init__(self):
        super().__init__()
        self._last_frame_bytes: Optional[bytes] = None
        self._last_w: int = 0
        self._last_h: int = 0
        self.setText("Ожидание видео...")
        if PYQT6:
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        # Минимальный размер видео-виджета (16:9 пропорция, разумный минимум)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding if PYQT6 else QSizePolicy.Expanding,
                           QSizePolicy.Policy.Expanding if PYQT6 else QSizePolicy.Expanding)

    def update_frame(self, frame_data):
        """Обновляет изображение из bytes данных (RGB формат)."""
        try:
            if not isinstance(frame_data, tuple) or len(frame_data) != 3:
                return
            img_bytes, width, height = frame_data
            
            if not isinstance(img_bytes, bytes) or width <= 0 or height <= 0:
                return
            
            # Сохраняем ссылку на исходные данные
            self._last_frame_bytes = img_bytes
            self._last_w = int(width)
            self._last_h = int(height)

            bytes_per_line = 3 * width
            
            # Создаём QImage из bytes данных
            if PYQT6:
                q_image = QImage(self._last_frame_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                q_image = QImage(self._last_frame_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
            
            if q_image.isNull():
                return

            # Делаем deep copy, чтобы pixmap не зависел от внешнего буфера
            q_image = q_image.copy()
            pixmap = QPixmap.fromImage(q_image)
            
            # Масштабируем на размер виджета без сохранения пропорций
            widget_size = self.size()
            if widget_size.width() > 0 and widget_size.height() > 0:
                if PYQT6:
                    scaled_pixmap = pixmap.scaled(widget_size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
                else:
                    scaled_pixmap = pixmap.scaled(widget_size, Qt.IgnoreAspectRatio, Qt.FastTransformation)
                self.setPixmap(scaled_pixmap)
            else:
                self.setPixmap(pixmap)
        except Exception as e:
            print(f"Ошибка обновления кадра: {e}")
            import traceback
            traceback.print_exc()

    def redraw_last(self):
        """Перерисовать последний кадр (например, после ресайза окна)."""
        if self._last_frame_bytes and self._last_w > 0 and self._last_h > 0:
            self.update_frame((self._last_frame_bytes, self._last_w, self._last_h))


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self, logger: Logger = None):
        super().__init__()
        self.logger = logger
        self.client_thread: Optional[WebRTCClientThread] = None
        self._discover_thread: Optional[DiscoverThread] = None
        self._first_frame_seen: bool = False
        self._recorder: Optional[Mp4Recorder] = None
        self._record_pending: bool = False
        self._record_path: Optional[str] = None
        self._record_fps: float = 30.0
        # Aspect ratio видео для пропорционального изменения размера окна
        self._video_aspect_ratio: float = DEFAULT_VIDEO_WIDTH / DEFAULT_VIDEO_HEIGHT
        self._resizing: bool = False  # Флаг для предотвращения рекурсии при изменении размера
        self.init_ui()

    def init_ui(self):
        """Инициализация UI."""
        self.setWindowTitle("WebRTC Video Stream Client")

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Главный layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Виджет видео (создаём ДО подключения сигналов чекбоксов)
        self.video_widget = VideoWidget()

        # Панель управления
        control_layout = QHBoxLayout()
        
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("auto или IP адрес")
        self.host_edit.setText("auto")
        
        self.port_edit = QLineEdit()
        self.port_edit.setPlaceholderText("8080")
        self.port_edit.setText("8080")
        
        self.connect_btn = QPushButton("Подключиться")
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        self.discover_btn = QPushButton("Найти сервер")
        self.discover_btn.clicked.connect(self.discover_server)

        # Запись (клиент-side)
        self.rec_btn = QPushButton("REC")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setToolTip("Запись в MP4")
        self.rec_btn.toggled.connect(self.toggle_recording)
        self.rec_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 10px; }"
            "QPushButton:checked { background-color: #b00020; color: white; }"
        )
        
        control_layout.addWidget(QLabel("Хост:"))
        control_layout.addWidget(self.host_edit)
        control_layout.addWidget(QLabel("Порт:"))
        control_layout.addWidget(self.port_edit)
        control_layout.addWidget(self.discover_btn)
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.rec_btn)
        
        main_layout.addLayout(control_layout)

        # Видео (занимает все доступное пространство со stretch=1)
        main_layout.addWidget(self.video_widget, stretch=1)

        # Статус - жестко привязан к нижней границе окна
        self.status_label = QLabel("Готов к подключению")
        self.status_label.setStyleSheet("padding: 6px; background-color: #1e1e1e; color: #dcdcdc;")
        # Устанавливаем политику размера: Fixed по вертикали, чтобы статус имел фиксированную высоту
        if PYQT6:
            self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Фиксированная высота статус-бара (минимальная = максимальная)
        self.status_label.setMinimumHeight(30)
        self.status_label.setMaximumHeight(30)
        # Добавляем статус в layout со stretch=0, чтобы он оставался внизу
        main_layout.addWidget(self.status_label, stretch=0)
        
        # Вычисляем минимальный размер окна на основе минимальных размеров всех виджетов
        # Это делается после установки layout, чтобы Qt мог вычислить минимальный размер
        central_widget.updateGeometry()  # Обновляем геометрию для правильного расчета
        min_size = central_widget.minimumSizeHint()
        if min_size.isValid() and min_size.width() > 0 and min_size.height() > 0:
            # Устанавливаем минимальный размер окна на основе минимального размера центрального виджета
            # Добавляем небольшой запас для отступов окна (title bar, borders)
            self.setMinimumSize(min_size.width() + 20, min_size.height() + 40)
        else:
            # Fallback: устанавливаем минимальный размер, если вычисление не удалось
            self.setMinimumSize(520, 420)
        
        # Устанавливаем начальный размер окна пропорционально предполагаемому разрешению видео
        self._fit_window_to_video(DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT)

    def _fit_window_to_video(self, video_w: int, video_h: int):
        """Подогнать стартовый размер окна под разрешение видео (с ограничением по экрану)."""
        try:
            video_w = int(video_w)
            video_h = int(video_h)
            if video_w <= 0 or video_h <= 0:
                return

            # Обновляем aspect ratio видео
            self._video_aspect_ratio = video_w / video_h

            screen = QApplication.primaryScreen()
            if screen is None:
                return
            geom = screen.availableGeometry()
            max_w = int(geom.width() * 0.9)
            max_h = int(geom.height() * 0.9)

            # запас под панель управления + статус
            chrome_h = 140
            chrome_w = 60

            scale = min(max_w / float(video_w + chrome_w), max_h / float(video_h + chrome_h), 1.0)
            target_w = int(video_w * scale + chrome_w)
            target_h = int(video_h * scale + chrome_h)

            self._resizing = True
            self.resize(max(self.minimumWidth(), target_w), max(self.minimumHeight(), target_h))
            self._resizing = False
        except Exception:
            pass

    def on_frame(self, frame_data):
        """Промежуточный слот: первый кадр → подгоняем окно, потом рисуем."""
        try:
            if (not self._first_frame_seen) and isinstance(frame_data, tuple) and len(frame_data) == 3:
                _bytes, w, h = frame_data
                self._fit_window_to_video(int(w), int(h))
                self._first_frame_seen = True
        except Exception:
            pass

        # Запись: ждём первый кадр чтобы инициализировать writer
        try:
            if self._record_pending and isinstance(frame_data, tuple) and len(frame_data) == 3 and self._record_path:
                _bytes, w, h = frame_data
                self._recorder = Mp4Recorder(self._record_path, self._record_fps, int(w), int(h))
                self._record_pending = False
                self.status_label.setText(f"REC ▶ {self._record_path}")
        except Exception as e:
            self._record_pending = False
            self._recorder = None
            try:
                self.rec_btn.setChecked(False)
            except Exception:
                pass
            QMessageBox.critical(self, "Запись", f"Не удалось начать запись: {e}")

        # Если запись активна — пишем кадр
        if self._recorder:
            try:
                import numpy as np
                img_bytes, w, h = frame_data
                rgb = np.frombuffer(img_bytes, dtype=np.uint8).reshape((int(h), int(w), 3))
                bgr = rgb[:, :, ::-1].copy()
                self._recorder.write(bgr)
            except Exception:
                pass

        self.video_widget.update_frame(frame_data)

    def resizeEvent(self, event):
        # Если мы сами изменяем размер, не корректируем его
        if self._resizing:
            super().resizeEvent(event)
            try:
                self.video_widget.redraw_last()
            except Exception:
                pass
            return
        
        # Корректируем размер окна для сохранения пропорций видео
        try:
            new_size = event.size()
            old_size = event.oldSize()
            
            # Размеры элементов управления (chrome)
            chrome_h = 140
            chrome_w = 60
            
            # Вычисляем размер области видео
            video_area_w = new_size.width() - chrome_w
            video_area_h = new_size.height() - chrome_h
            
            # Вычисляем текущий aspect ratio области видео
            if video_area_h > 0:
                current_aspect = video_area_w / video_area_h
            else:
                current_aspect = self._video_aspect_ratio
            
            # Если пропорции не совпадают, корректируем размер
            if abs(current_aspect - self._video_aspect_ratio) > 0.01 and video_area_w > 0 and video_area_h > 0:
                # Определяем, какое измерение было изменено пользователем
                if old_size.isValid():
                    width_changed = abs(new_size.width() - old_size.width()) > abs(new_size.height() - old_size.height())
                else:
                    width_changed = True
                
                if width_changed:
                    # Пользователь изменил ширину - корректируем высоту
                    target_video_w = video_area_w
                    target_video_h = int(target_video_w / self._video_aspect_ratio)
                    target_h = target_video_h + chrome_h
                    target_w = new_size.width()
                else:
                    # Пользователь изменил высоту - корректируем ширину
                    target_video_h = video_area_h
                    target_video_w = int(target_video_h * self._video_aspect_ratio)
                    target_w = target_video_w + chrome_w
                    target_h = new_size.height()
                
                # Проверяем минимальные размеры
                min_w = self.minimumWidth()
                min_h = self.minimumHeight()
                target_w = max(min_w, target_w)
                target_h = max(min_h, target_h)
                
                # Устанавливаем корректный размер
                self._resizing = True
                self.resize(target_w, target_h)
                self._resizing = False
                return
        except Exception:
            pass
        
        super().resizeEvent(event)
        # Перерисовать последний кадр под новый размер окна
        try:
            self.video_widget.redraw_last()
        except Exception:
            pass

    def discover_server(self):
        """Поиск сервера в локальной сети."""
        self.status_label.setText("Поиск сервера...")
        self.discover_btn.setEnabled(False)

        # Если уже идёт поиск — не запускаем второй
        if self._discover_thread and self._discover_thread.isRunning():
            return

        self._discover_thread = DiscoverThread(timeout=5, logger=self.logger)

        def _on_found(info):
            try:
                if info:
                    self.host_edit.setText(info.get("host", "auto"))
                    self.port_edit.setText(str(info.get("signal_port", 8080)))
                    self.status_label.setText(f"Сервер найден: {info.get('host')}:{info.get('signal_port')}")
                else:
                    self.status_label.setText("Сервер не найден")
            finally:
                self.discover_btn.setEnabled(True)

        self._discover_thread.found.connect(_on_found)
        self._discover_thread.start()

    def toggle_connection(self):
        """Переключение подключения/отключения."""
        if self.client_thread and self.client_thread.isRunning():
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        """Подключение к серверу."""
        host = self.host_edit.text().strip()
        if not host:
            QMessageBox.warning(self, "Ошибка", "Укажите хост или используйте 'auto'")
            return
        
        try:
            signal_port = int(self.port_edit.text().strip() or "8080")
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Неверный порт")
            return

        # Если auto, ищем сервер
        if host.lower() == "auto":
            self.status_label.setText("Поиск сервера...")
            self.connect_btn.setEnabled(False)

            if self._discover_thread and self._discover_thread.isRunning():
                return

            self._discover_thread = DiscoverThread(timeout=5, logger=self.logger)

            def _on_found(info):
                if info:
                    host_found = info.get("host", host)
                    port_found = int(info.get("signal_port") or signal_port)
                    self.connect_btn.setEnabled(True)
                    self.start_client(host_found, port_found, False)
                else:
                    self.status_label.setText("Сервер не найден")
                    self.connect_btn.setEnabled(True)

            self._discover_thread.found.connect(_on_found)
            self._discover_thread.start()
        else:
            self.start_client(host, signal_port, False)

    def start_client(self, host: str, signal_port: int, stun: bool):
        """Запускает WebRTC клиент."""
        if self.client_thread and self.client_thread.isRunning():
            return

        self._first_frame_seen = False
        self.client_thread = WebRTCClientThread(host, signal_port, stun, logger=self.logger)
        if self.logger:
            self.logger.info(f"Запуск клиента к {host}:{signal_port}", source="client_qt")
        self.client_thread.frame_received.connect(self.on_frame)
        self.client_thread.status_changed.connect(self.status_label.setText)
        self.client_thread.error_occurred.connect(self.on_error)
        self.client_thread.finished.connect(self.on_client_finished)
        self.client_thread.start()

        self.connect_btn.setText("Отключиться")
        self.host_edit.setEnabled(False)
        self.port_edit.setEnabled(False)
        self.discover_btn.setEnabled(False)

    def disconnect(self):
        """Отключение от сервера."""
        if self.client_thread:
            if self.logger:
                self.logger.info("Отключение от сервера", source="client_qt")
            self.client_thread.stop()
            self.client_thread = None

        # Остановить запись при отключении
        self._stop_recording()
        
        self.connect_btn.setText("Подключиться")
        self.connect_btn.setEnabled(True)
        self.host_edit.setEnabled(True)
        self.port_edit.setEnabled(True)
        self.discover_btn.setEnabled(True)
        self.video_widget.setText("Отключено")
        self.status_label.setText("Отключено")

    def _stop_recording(self):
        rec = self._recorder
        self._recorder = None
        self._record_pending = False
        self._record_path = None
        if rec:
            try:
                rec.close()
            except Exception:
                pass

    def toggle_recording(self, enabled: bool):
        """Старт/стоп записи MP4 по кнопке."""
        if enabled:
            default_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            try:
                path, _flt = QFileDialog.getSaveFileName(self, "Сохранить запись", default_name, "MP4 Video (*.mp4)")
            except Exception:
                path = ""
            if not path:
                # пользователь отменил
                self.rec_btn.setChecked(False)
                return
            if not path.lower().endswith(".mp4"):
                path = path + ".mp4"

            self._record_path = path
            self._record_pending = True
            self.status_label.setText("REC: ожидание первого кадра...")
            if self.logger:
                self.logger.info(f"Начало записи: {path}", source="client_qt")
        else:
            if self.logger and self._record_path:
                self.logger.info(f"Остановка записи: {self._record_path}", source="client_qt")
            self._stop_recording()
            # статус не трогаем если подключены — оставим текущий
            if not (self.client_thread and self.client_thread.isRunning()):
                self.status_label.setText("Готов к подключению")

    def on_error(self, message: str):
        """Обработка ошибки."""
        if self.logger:
            self.logger.error(f"Ошибка GUI: {message}", source="client_qt")
        QMessageBox.critical(self, "Ошибка", message)
        self.disconnect()

    def on_client_finished(self):
        """Обработка завершения потока клиента."""
        if self.client_thread and not self.client_thread.isRunning():
            self.connect_btn.setText("Подключиться")
            self.host_edit.setEnabled(True)
            self.port_edit.setEnabled(True)
            self.discover_btn.setEnabled(True)

    def closeEvent(self, event):
        """Обработка закрытия окна."""
        if self.client_thread and self.client_thread.isRunning():
            self.disconnect()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="WebRTC Video Streaming Client (PyQt GUI)")
    parser.add_argument("--host", type=str, default=None, help='IP сервера (по умолчанию используется GUI)')
    parser.add_argument("--signal-port", type=int, default=None, help="Порт HTTP signaling")
    parser.add_argument("--auto", action="store_true", default=False, help="Автообнаружение сервера")
    parser.add_argument("--stun", action="store_true", default=False, help="Использовать публичный STUN")
    parser.add_argument("--log-level", type=str, default="info", choices=["spam", "debug", "verbose", "info", "warning", "error", "critical"], help="Уровень логирования")
    parser.add_argument("--log-path", type=str, default="logs", help="Путь к папке с логами")
    args = parser.parse_args()

    log_level = loggingLevels.get(args.log_level, INFO)
    logger = Logger("client_qt", args.log_path, log_level)

    app = QApplication(sys.argv)
    # HiDPI: более четкие pixmap'ы (если поддерживается)
    try:
        if PYQT6:
            app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        else:
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    apply_dark_theme(app)
    
    window = MainWindow(logger=logger)
    window.show()
    
    logger.info("Запуск GUI клиента", source="client_qt")

    # Если указаны параметры командной строки, подключаемся автоматически
    if args.host or args.auto:
        host = "auto" if args.auto else (args.host or "auto")
        signal_port = int(args.signal_port or 8080)

        # Заполняем поля и запускаем подключение из GUI-потока
        window.host_edit.setText(host)
        window.port_edit.setText(str(signal_port))

        # Важно: если host == "auto" — нужно пройти discovery, а не стартовать WebRTC на "auto"
        QTimer.singleShot(0, window.connect)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
