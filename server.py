#!/usr/bin/env python3
"""
GStreamer Video Streaming Server
Транслирует видео с веб-камеры или из файла по сети через RTP/UDP
"""

import subprocess
import sys
import argparse
import signal
import os
import platform
import threading

class VideoStreamServer:
    def __init__(self, source='webcam', file_path=None, host='127.0.0.1', port=5000, 
                 video_port=5004, audio_port=5006, width=640, height=480, fps=30):
        self.source = source
        self.file_path = file_path
        self.host = host
        self.port = port
        self.video_port = video_port
        self.audio_port = audio_port
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        
    def build_pipeline(self):
        """Строит GStreamer pipeline для сервера"""
        if self.source == 'webcam':
            # Захват с веб-камеры (кроссплатформенный)
            if platform.system() == 'Windows':
                # Windows использует mfvideosrc (Media Foundation) - современный элемент
                # Получаем видео в любом формате, затем конвертируем и масштабируем
                source_pipeline = (
                    f"mfvideosrc device-index=0 ! "
                    f"video/x-raw ! "
                    f"videoconvert ! "
                    f"videoscale ! "
                    f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                )
            else:
                # Linux использует v4l2src
                source_pipeline = (
                    f"v4l2src device=/dev/video0 ! "
                    f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                )
        elif self.source == 'file':
            if not self.file_path or not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Файл не найден: {self.file_path}")
            # Чтение из файла
            file_path_escaped = self.file_path.replace('\\', '/')
            source_pipeline = (
                f"filesrc location={file_path_escaped} ! "
                f"decodebin ! "
                f"videoconvert ! "
                f"videoscale ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            )
        else:
            raise ValueError(f"Неизвестный источник: {self.source}")
        
        # Кодирование и отправка через RTP
        pipeline = (
            source_pipeline +
            f"x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self.host} port={self.video_port}"
        )
        
        return pipeline
    
    def start(self):
        """Запускает сервер"""
        pipeline = self.build_pipeline()
        gst_cmd = 'gst-launch-1.0.exe' if platform.system() == 'Windows' else 'gst-launch-1.0'
        
        print(f"Запуск сервера трансляции...")
        print(f"Источник: {self.source}")
        print(f"Адрес: {self.host}:{self.video_port}")
        print(f"Разрешение: {self.width}x{self.height} @ {self.fps} fps")
        print(f"\nКоманда GStreamer:")
        print(f"{gst_cmd} -e {pipeline}")
        print("\nДля остановки нажмите Ctrl+C\n")
        
        try:
            # Запускаем GStreamer pipeline как одну команду
            cmd = f"{gst_cmd} -e {pipeline}"
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Выводим ошибки в реальном времени
            def print_stderr():
                for line in iter(self.process.stderr.readline, b''):
                    if line:
                        print(line.decode('utf-8', errors='ignore'), end='')
            
            stderr_thread = threading.Thread(target=print_stderr, daemon=True)
            stderr_thread.start()
            
            # Ожидаем завершения
            self.process.wait()
            
        except KeyboardInterrupt:
            print("\nОстановка сервера...")
            self.stop()
        except Exception as e:
            print(f"Ошибка: {e}")
            self.stop()
    
    def stop(self):
        """Останавливает сервер"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("Сервер остановлен")

def main():
    parser = argparse.ArgumentParser(description='GStreamer Video Streaming Server')
    parser.add_argument('--source', choices=['webcam', 'file'], default='webcam',
                        help='Источник видео: webcam или file (по умолчанию: webcam)')
    parser.add_argument('--file', type=str, default=None,
                        help='Путь к видео файлу (если source=file)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='IP адрес для отправки (по умолчанию: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5004,
                        help='Порт для видео (по умолчанию: 5004)')
    parser.add_argument('--width', type=int, default=640,
                        help='Ширина видео (по умолчанию: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Высота видео (по умолчанию: 480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Частота кадров (по умолчанию: 30)')
    
    args = parser.parse_args()
    
    # Проверка наличия GStreamer
    gst_cmd = 'gst-launch-1.0.exe' if platform.system() == 'Windows' else 'gst-launch-1.0'
    gst_found = False
    
    # На Windows пытаемся найти GStreamer в стандартных местах
    if platform.system() == 'Windows':
        possible_paths = [
            r"C:\Program Files\GStreamer\1.0\msvc_x86_64\bin",
            r"C:\Program Files\GStreamer\1.0\mingw_x86_64\bin",
            r"C:\gstreamer\1.0\msvc_x86_64\bin",
            r"C:\gstreamer\1.0\mingw_x86_64\bin",
        ]
        
        for gst_path in possible_paths:
            gst_exe = os.path.join(gst_path, gst_cmd)
            if os.path.exists(gst_exe):
                # Добавляем в PATH текущего процесса
                os.environ['PATH'] = gst_path + os.pathsep + os.environ.get('PATH', '')
                gst_found = True
                break
        
        if not gst_found:
            # Пытаемся найти через системный PATH
            try:
                subprocess.run([gst_cmd, '--version'], 
                              capture_output=True, check=True, shell=True)
                gst_found = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
    
    # Проверка через системный PATH (для всех платформ)
    if not gst_found:
        try:
            subprocess.run([gst_cmd, '--version'], 
                          capture_output=True, check=True, shell=(platform.system() == 'Windows'))
            gst_found = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    if not gst_found:
        print("Ошибка: GStreamer не установлен или не найден в PATH")
        print("Установите GStreamer: https://gstreamer.freedesktop.org/download/")
        if platform.system() == 'Windows':
            print("После установки добавьте GStreamer в PATH или перезапустите терминал")
        sys.exit(1)
    
    server = VideoStreamServer(
        source=args.source,
        file_path=args.file,
        host=args.host,
        video_port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    # Обработка сигналов для корректного завершения
    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
