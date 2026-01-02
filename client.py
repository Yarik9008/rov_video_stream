#!/usr/bin/env python3
"""
GStreamer Video Streaming Client
Принимает и воспроизводит видео, транслируемое сервером
"""

import subprocess
import sys
import argparse
import signal
import os
import threading
import platform

class VideoStreamClient:
    def __init__(self, host='127.0.0.1', video_port=5004, audio_port=5006):
        self.host = host
        self.video_port = video_port
        self.audio_port = audio_port
        self.process = None
        
    def build_pipeline(self):
        """Строит GStreamer pipeline для клиента"""
        # Прием RTP потока и воспроизведение
        pipeline = (
            f"udpsrc port={self.video_port} ! "
            f"application/x-rtp,encoding-name=H264,payload=96 ! "
            f"rtph264depay ! "
            f"h264parse ! "
            f"avdec_h264 ! "
            f"videoconvert ! "
            f"autovideosink sync=false"
        )
        
        return pipeline
    
    def start(self):
        """Запускает клиент"""
        pipeline = self.build_pipeline()
        gst_cmd = 'gst-launch-1.0.exe' if platform.system() == 'Windows' else 'gst-launch-1.0'
        
        print(f"Запуск клиента...")
        print(f"Подключение к: {self.host}:{self.video_port}")
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
            print("\nОстановка клиента...")
            self.stop()
        except Exception as e:
            print(f"Ошибка: {e}")
            self.stop()
    
    def stop(self):
        """Останавливает клиент"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("Клиент остановлен")

def main():
    parser = argparse.ArgumentParser(description='GStreamer Video Streaming Client')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='IP адрес сервера (по умолчанию: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5004,
                        help='Порт для видео (по умолчанию: 5004)')
    
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
    
    client = VideoStreamClient(
        host=args.host,
        video_port=args.port
    )
    
    # Обработка сигналов для корректного завершения
    def signal_handler(sig, frame):
        client.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        client.start()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
