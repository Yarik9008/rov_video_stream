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
import socket
import sys
import threading
import time
from typing import Optional

PYQT6 = False
try:
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
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
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
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


def discover_server(timeout: int = 5):
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

    def __init__(self, host: str, signal_port: int, stun: bool):
        super().__init__()
        self.host = host
        self.signal_port = signal_port
        self.stun = stun
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
            self.error_occurred.emit("Host='auto' недопустим здесь. Используйте автообнаружение из GUI (кнопка/--auto).")
            return

        self.status_changed.emit("Подключение...")
        
        cfg = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])] if self.stun else []
        )
        self._pc = RTCPeerConnection(configuration=cfg)

        @self._pc.on("iceconnectionstatechange")
        def on_ice_state_change():
            try:
                self.status_changed.emit(f"ICE: {self._pc.iceConnectionState}")
            except Exception:
                pass

        @self._pc.on("connectionstatechange")
        def on_conn_state_change():
            try:
                self.status_changed.emit(f"PC: {self._pc.connectionState}")
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
                                self.error_occurred.emit(f"Ошибка получения кадра: {e}")
                                import traceback
                                traceback.print_exc()
                            break
                except Exception as e:
                    if not self._stop_event.is_set():
                        self.error_occurred.emit(f"Ошибка потока: {e}")
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
            self.error_occurred.emit(f"Ошибка соединения: {e}")
        finally:
            self.status_changed.emit("Отключение...")
            try:
                if self._pc:
                    await self._pc.close()
            except Exception:
                pass
            self.status_changed.emit("Отключено")

    def stop(self):
        """Останавливает клиент."""
        if self._stop_event and self._loop and not self._loop.is_closed():
            # Устанавливаем событие остановки в event loop
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self.wait(3000)  # Ждём максимум 3 секунды


class DiscoverThread(QThread):
    """Поток для поиска сервера (чтобы не трогать UI из threading.Thread)."""
    found = pyqtSignal(object)  # dict | None

    def __init__(self, timeout: int = 5):
        super().__init__()
        self._timeout = int(timeout)

    def run(self):
        try:
            info = discover_server(timeout=self._timeout)
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
        self._fit_to_window: bool = True
        self._hq_scaling: bool = True
        self.setText("Ожидание видео...")
        if PYQT6:
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding if PYQT6 else QSizePolicy.Expanding,
                           QSizePolicy.Policy.Expanding if PYQT6 else QSizePolicy.Expanding)

    def set_fit_to_window(self, enabled: bool):
        self._fit_to_window = bool(enabled)
        self.redraw_last()

    def set_hq_scaling(self, enabled: bool):
        self._hq_scaling = bool(enabled)
        self.redraw_last()

    def update_frame(self, frame_data):
        """Обновляет изображение из bytes данных (RGB формат)."""
        try:
            if not isinstance(frame_data, tuple) or len(frame_data) != 3:
                return
            img_bytes, width, height = frame_data
            
            if not isinstance(img_bytes, bytes) or width <= 0 or height <= 0:
                return
            
            # Важно: QImage(bytes, ...) НЕ владеет памятью и не копирует буфер.
            # Держим ссылку на bytes до следующего кадра + делаем copy() для гарантии.
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
            
            # Масштабируем с сохранением пропорций только если виджет имеет размер
            widget_size = self.size()
            if not self._fit_to_window:
                # 1:1 (без ухудшения качества из-за масштабирования)
                self.setPixmap(pixmap)
                return

            if widget_size.width() > 0 and widget_size.height() > 0:
                if PYQT6:
                    aspect = Qt.AspectRatioMode.KeepAspectRatio
                    transform = Qt.TransformationMode.SmoothTransformation if self._hq_scaling else Qt.TransformationMode.FastTransformation
                else:
                    aspect = Qt.KeepAspectRatio
                    transform = Qt.SmoothTransformation if self._hq_scaling else Qt.FastTransformation

                scaled_pixmap = pixmap.scaled(widget_size, aspect, transform)
                self.setPixmap(scaled_pixmap)
            else:
                # Если виджет еще не имеет размера, просто устанавливаем pixmap без масштабирования
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

    def __init__(self):
        super().__init__()
        self.client_thread: Optional[WebRTCClientThread] = None
        self._discover_thread: Optional[DiscoverThread] = None
        self._first_frame_seen: bool = False
        self.init_ui()

    def init_ui(self):
        """Инициализация UI."""
        self.setWindowTitle("WebRTC Video Stream Client")
        # Окно должно быть растягиваемым; минимальный размер делаем умеренным
        self.setMinimumSize(520, 420)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Главный layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

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

        # Качество отображения (клиент-side)
        self.fit_cb = QCheckBox("Fit")
        self.fit_cb.setChecked(True)
        self.fit_cb.toggled.connect(self.video_widget.set_fit_to_window)

        self.hq_cb = QCheckBox("HQ")
        self.hq_cb.setChecked(True)
        self.hq_cb.toggled.connect(self.video_widget.set_hq_scaling)
        
        control_layout.addWidget(QLabel("Хост:"))
        control_layout.addWidget(self.host_edit)
        control_layout.addWidget(QLabel("Порт:"))
        control_layout.addWidget(self.port_edit)
        control_layout.addWidget(self.discover_btn)
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.fit_cb)
        control_layout.addWidget(self.hq_cb)
        
        main_layout.addLayout(control_layout)

        # Виджет видео
        self.video_widget = VideoWidget()
        main_layout.addWidget(self.video_widget, stretch=1)

        # Статус
        self.status_label = QLabel("Готов к подключению")
        self.status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        main_layout.addWidget(self.status_label)

    def _fit_window_to_video(self, video_w: int, video_h: int):
        """Подогнать стартовый размер окна под разрешение видео (с ограничением по экрану)."""
        try:
            video_w = int(video_w)
            video_h = int(video_h)
            if video_w <= 0 or video_h <= 0:
                return

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

            self.resize(max(self.minimumWidth(), target_w), max(self.minimumHeight(), target_h))
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
        self.video_widget.update_frame(frame_data)

    def resizeEvent(self, event):
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

        self._discover_thread = DiscoverThread(timeout=5)

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

            self._discover_thread = DiscoverThread(timeout=5)

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
        self.client_thread = WebRTCClientThread(host, signal_port, stun)
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
            self.client_thread.stop()
            self.client_thread = None
        
        self.connect_btn.setText("Подключиться")
        self.connect_btn.setEnabled(True)
        self.host_edit.setEnabled(True)
        self.port_edit.setEnabled(True)
        self.discover_btn.setEnabled(True)
        self.video_widget.setText("Отключено")
        self.status_label.setText("Отключено")

    def on_error(self, message: str):
        """Обработка ошибки."""
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
    args = parser.parse_args()

    app = QApplication(sys.argv)
    # HiDPI: более четкие pixmap'ы (если поддерживается)
    try:
        if PYQT6:
            app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        else:
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    app.setStyle("Fusion")  # Современный стиль
    
    window = MainWindow()
    window.show()

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
