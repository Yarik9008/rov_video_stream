#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GStreamer Video Streaming Server
Транслирует видео с веб-камеры или из файла по сети через RTP/UDP
Поддерживает Windows, Linux и macOS
"""

import subprocess
import sys
import argparse
import signal
import os
import platform
import threading
import shutil
import socket
import json
import time
import atexit
from pathlib import Path

def find_gstreamer():
    """Находит GStreamer на любой платформе и добавляет в PATH"""
    system = platform.system()
    gst_cmd = 'gst-launch-1.0.exe' if system == 'Windows' else 'gst-launch-1.0'
    
    # Сначала проверяем, доступен ли в PATH
    if shutil.which(gst_cmd):
        return gst_cmd
    
    # Windows: проверяем стандартные пути установки
    if system == 'Windows':
        possible_paths = [
            r"C:\Program Files\GStreamer\1.0\msvc_x86_64\bin",
            r"C:\Program Files\GStreamer\1.0\mingw_x86_64\bin",
            r"C:\Program Files (x86)\GStreamer\1.0\msvc_x86_64\bin",
            r"C:\Program Files (x86)\GStreamer\1.0\mingw_x86_64\bin",
            r"C:\gstreamer\1.0\msvc_x86_64\bin",
            r"C:\gstreamer\1.0\mingw_x86_64\bin",
        ]
        
        for gst_path in possible_paths:
            gst_exe = os.path.join(gst_path, gst_cmd)
            if os.path.exists(gst_exe):
                os.environ['PATH'] = gst_path + os.pathsep + os.environ.get('PATH', '')
                return gst_cmd
    
    # macOS: проверяем Homebrew пути и ищем через brew
    elif system == 'Darwin':
        possible_paths = [
            '/opt/homebrew/bin',
            '/usr/local/bin',
            '/opt/local/bin',  # MacPorts
        ]
        
        # Пытаемся найти через brew --prefix
        try:
            brew_prefix = subprocess.run(
                ['brew', '--prefix', 'gstreamer'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if brew_prefix.returncode == 0:
                brew_path = brew_prefix.stdout.strip()
                bin_path = os.path.join(brew_path, 'bin')
                if os.path.exists(bin_path):
                    possible_paths.insert(0, bin_path)
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        # Также проверяем стандартные пути Homebrew для плагинов
        try:
            for plugin in ['gst-plugins-base', 'gst-plugins-good', 'gst-plugins-bad', 'gst-plugins-ugly']:
                brew_prefix = subprocess.run(
                    ['brew', '--prefix', plugin],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if brew_prefix.returncode == 0:
                    brew_path = brew_prefix.stdout.strip()
                    bin_path = os.path.join(brew_path, 'bin')
                    if os.path.exists(bin_path) and bin_path not in possible_paths:
                        possible_paths.append(bin_path)
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        # Ищем GStreamer в найденных путях
        for gst_path in possible_paths:
            gst_exe = os.path.join(gst_path, gst_cmd)
            if os.path.exists(gst_exe):
                # Добавляем путь в начало PATH
                current_path = os.environ.get('PATH', '')
                if gst_path not in current_path:
                    os.environ['PATH'] = gst_path + os.pathsep + current_path
                return gst_cmd
        
        # Если не нашли напрямую, ищем через find
        try:
            find_result = subprocess.run(
                ['find', '/opt/homebrew', '/usr/local', '/opt/local', '-name', gst_cmd, '-type', 'f', '2>/dev/null'],
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            if find_result.returncode == 0 and find_result.stdout.strip():
                found_path = os.path.dirname(find_result.stdout.strip().split('\n')[0])
                if os.path.exists(found_path):
                    current_path = os.environ.get('PATH', '')
                    if found_path not in current_path:
                        os.environ['PATH'] = found_path + os.pathsep + current_path
                    return gst_cmd
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
    
    # Linux: обычно уже в PATH, но проверим стандартные места
    elif system == 'Linux':
        possible_paths = [
            '/usr/bin',
            '/usr/local/bin',
        ]
        
        for gst_path in possible_paths:
            gst_exe = os.path.join(gst_path, gst_cmd)
            if os.path.exists(gst_exe):
                current_path = os.environ.get('PATH', '')
                if gst_path not in current_path:
                    os.environ['PATH'] = gst_path + os.pathsep + current_path
                return gst_cmd
    
    return None

def get_local_ip():
    """Получает локальный IP адрес для сетевого интерфейса"""
    try:
        # Подключаемся к внешнему адресу (не отправляем данные)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # Не нужно реально подключаться, просто получаем локальный адрес
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def get_broadcast_address(ip):
    """Получает broadcast адрес для данного IP"""
    try:
        # Простой способ: заменяем последний октет на 255
        parts = ip.split('.')
        if len(parts) == 4:
            parts[-1] = '255'
            return '.'.join(parts)
    except Exception:
        pass
    return '255.255.255.255'

def check_gstreamer():
    """Проверяет наличие GStreamer и возвращает команду"""
    gst_cmd = find_gstreamer()
    
    if not gst_cmd:
        system = platform.system()
        print("Ошибка: GStreamer не установлен или не найден в PATH")
        print("\nУстановите GStreamer:")
        if system == 'Windows':
            print("  Скачайте с https://gstreamer.freedesktop.org/download/")
            print("  После установки добавьте GStreamer в PATH или перезапустите терминал")
        elif system == 'Darwin':
            print("  macOS: brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav")
        else:  # Linux
            print("  Ubuntu/Debian: sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav")
            print("  Fedora: sudo dnf install gstreamer1 gstreamer1-plugins-base gstreamer1-plugins-good gstreamer1-plugins-bad gstreamer1-plugins-ugly")
        sys.exit(1)
    
    # Проверяем, что команда действительно работает
    try:
        result = subprocess.run(
            [gst_cmd, '--version'],
            capture_output=True,
            check=True,
            shell=(platform.system() == 'Windows')
        )
        return gst_cmd
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ошибка: GStreamer найден, но не работает корректно")
        sys.exit(1)

class VideoStreamServer:
    # Порт для UDP broadcast обнаружения
    DISCOVERY_PORT = 5003
    
    def __init__(self, source='webcam', file_path=None, host=None, port=5000, 
                 video_port=5004, audio_port=5006, width=640, height=480, fps=30, device_index=0,
                 enable_discovery=True, bitrate=2000):
        self.source = source
        self.file_path = file_path
        
        # Автоматически определяем локальный IP для трансляции в сети
        if host is None:
            local_ip = get_local_ip()
            if local_ip and local_ip != '127.0.0.1':
                self.host = '0.0.0.0'  # Слушаем на всех интерфейсах
                self.local_ip = local_ip  # Сохраняем локальный IP для отображения
            else:
                self.host = '127.0.0.1'
                self.local_ip = '127.0.0.1'
        else:
            self.host = host
            if host == '0.0.0.0':
                self.local_ip = get_local_ip()
            else:
                self.local_ip = host
        
        self.port = port
        self.video_port = video_port
        self.audio_port = audio_port
        self.width = width
        self.height = height
        self.fps = fps
        self.device_index = device_index
        self.bitrate = bitrate
        self.process = None
        self.system = platform.system()
        # Автоматически включаем обнаружение если не localhost
        if self.host not in ('127.0.0.1', 'localhost'):
            self.enable_discovery = enable_discovery
        else:
            self.enable_discovery = False
        self.discovery_socket = None
        self.discovery_thread = None
        self.running = False
        
    def build_pipeline(self):
        """Строит GStreamer pipeline для сервера"""
        if self.source == 'webcam':
            # Захват с веб-камеры (кроссплатформенный)
            if self.system == 'Windows':
                # Windows использует mfvideosrc (Media Foundation)
                # do-timestamp=true: использовать временные метки для синхронизации
                source_pipeline = (
                    f"mfvideosrc device-index={self.device_index} do-timestamp=true ! "
                    f"video/x-raw ! "
                    f"videoconvert ! "
                    f"videoscale ! "
                    f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                )
            elif self.system == 'Darwin':
                # macOS использует avfvideosrc (AVFoundation)
                # do-timestamp=true: использовать временные метки для синхронизации
                source_pipeline = (
                    f"avfvideosrc device-index={self.device_index} do-timestamp=true ! "
                    f"video/x-raw ! "
                    f"videoconvert ! "
                    f"videoscale ! "
                    f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                )
            else:
                # Linux использует v4l2src
                # io-mode=2: режим памяти для уменьшения задержки
                device = f"/dev/video{self.device_index}" if self.device_index > 0 else "/dev/video0"
                source_pipeline = (
                    f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
                    f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                )
        elif self.source == 'file':
            if not self.file_path:
                raise ValueError("Не указан путь к файлу (используйте --file)")
            
            # Нормализуем путь для кроссплатформенности
            file_path = Path(self.file_path).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            
            # Экранируем путь для GStreamer (заменяем обратные слеши на прямые)
            file_path_str = str(file_path).replace('\\', '/')
            # Экранируем специальные символы в пути
            file_path_str = file_path_str.replace(' ', '\\ ').replace('(', '\\(').replace(')', '\\)')
            
            source_pipeline = (
                f"filesrc location={file_path_str} ! "
                f"decodebin ! "
                f"videoconvert ! "
                f"videoscale ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            )
        else:
            raise ValueError(f"Неизвестный источник: {self.source}")
        
        # Определяем адрес для отправки видео
        # Если host = '0.0.0.0', используем broadcast адрес для отправки на все интерфейсы
        # Это позволяет клиентам подключаться с любого IP в локальной сети
        video_host = self.host
        if self.host == '0.0.0.0':
            # Используем broadcast для отправки на все интерфейсы локальной сети
            video_host = get_broadcast_address(self.local_ip)
        
        # Кодирование H.265 (HEVC) и отправка через RTP
        # Используем аппаратный энкодер на macOS для лучшей производительности
        # Оптимизированные настройки для минимальной задержки:
        if self.system == 'Darwin':
            # macOS: используем аппаратный энкодер VideoToolbox (vtenc_h265)
            encoder = f"vtenc_h265 bitrate={self.bitrate} realtime=true allow-frame-reordering=false"
        elif self.system == 'Linux':
            # Linux: используем программный x265enc
            encoder = (
                f"x265enc tune=zerolatency bitrate={self.bitrate} speed-preset=ultrafast "
                f"key-int-max=30 threads=4 bframes=0"
            )
        else:
            # Windows: используем x265enc
            encoder = (
                f"x265enc tune=zerolatency bitrate={self.bitrate} speed-preset=ultrafast "
                f"key-int-max=30 threads=4 bframes=0"
            )
        
        pipeline = (
            source_pipeline +
            f"{encoder} ! "
            f"h265parse ! "
            f"rtph265pay config-interval=1 pt=96 mtu=1400 ! "
            f"udpsink host={video_host} port={self.video_port} buffer-size=65536 sync=false"
        )
        
        return pipeline
    
    def _discovery_service(self):
        """Отправляет UDP broadcast сообщения для обнаружения сервера"""
        try:
            # Получаем локальный IP
            local_ip = get_local_ip()
            broadcast_ip = get_broadcast_address(local_ip)
            
            # Создаем UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(1)
            
            # Информация о сервере
            server_info = {
                'host': local_ip,
                'video_port': self.video_port,
                'audio_port': self.audio_port,
                'width': self.width,
                'height': self.height,
                'fps': self.fps
            }
            message = json.dumps(server_info).encode('utf-8')
            
            while self.running:
                try:
                    # Отправляем broadcast сообщение
                    sock.sendto(message, (broadcast_ip, self.DISCOVERY_PORT))
                    time.sleep(2)  # Отправляем каждые 2 секунды
                except Exception:
                    # Игнорируем ошибки отправки
                    time.sleep(2)
            
            sock.close()
        except Exception:
            pass
    
    def start_discovery(self):
        """Запускает сервис обнаружения"""
        if not self.enable_discovery or self.host == '127.0.0.1':
            return
        
        self.running = True
        self.discovery_thread = threading.Thread(target=self._discovery_service, daemon=True)
        self.discovery_thread.start()
    
    def stop_discovery(self):
        """Останавливает сервис обнаружения"""
        self.running = False
        if self.discovery_thread:
            self.discovery_thread.join(timeout=1)
    
    def start(self):
        """Запускает сервер"""
        self.running = True
        
        pipeline = self.build_pipeline()
        gst_cmd = find_gstreamer()
        if not gst_cmd:
            gst_cmd = check_gstreamer()
        
        print(f"Запуск сервера трансляции...")
        print(f"Платформа: {self.system}")
        print(f"Источник: {self.source}")
        if self.source == 'webcam':
            print(f"Устройство: индекс {self.device_index}")
        elif self.source == 'file':
            print(f"Файл: {self.file_path}")
        print(f"Локальный IP: {self.local_ip}")
        print(f"Порт: {self.video_port}")
        print(f"Разрешение: {self.width}x{self.height} @ {self.fps} fps")
        print(f"Битрейт: {self.bitrate} kbps")
        
        # Запускаем сервис обнаружения
        # Включаем, если host не localhost и не 127.0.0.1
        if self.enable_discovery:
            self.start_discovery()
            print(f"Автообнаружение: включено (broadcast на порт {self.DISCOVERY_PORT})")
        
        print(f"\n{'='*60}")
        print(f"Сервер готов к трансляции!")
        if self.host != '127.0.0.1':
            print(f"\nДля подключения с другого компьютера:")
            print(f"  python client.py --host {self.local_ip} --port {self.video_port}")
            print(f"\nИли используйте автообнаружение:")
            print(f"  python client.py")
        else:
            print(f"\nДля подключения локально:")
            print(f"  python client.py --host {self.local_ip} --port {self.video_port}")
        print(f"{'='*60}")
        print(f"\nКоманда GStreamer:")
        print(f"{gst_cmd} -e {pipeline}")
        print("\nДля остановки нажмите Ctrl+C\n")
        
        try:
            # Запускаем GStreamer pipeline
            # Используем shell=True для всех платформ, так как pipeline - сложная строка
            # с пробелами и специальными символами
            cmd = f"{gst_cmd} -e {pipeline}"
            
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False
            )
            
            # Выводим ошибки в реальном времени
            def print_stderr():
                try:
                    for line in iter(self.process.stderr.readline, b''):
                        if line:
                            decoded = line.decode('utf-8', errors='ignore')
                            print(decoded, end='', flush=True)
                except Exception:
                    pass
            
            stderr_thread = threading.Thread(target=print_stderr, daemon=True)
            stderr_thread.start()
            
            # Ожидаем завершения
            self.process.wait()
            
        except KeyboardInterrupt:
            print("\nОстановка сервера...")
            self.stop()
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        """Останавливает сервер"""
        if not self.running:
            return
        
        self.running = False
        self.stop_discovery()
        
        if self.process:
            try:
                # Сначала пытаемся корректно завершить процесс
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Если процесс не завершился, принудительно убиваем
                    self.process.kill()
                    self.process.wait(timeout=2)
            except (ProcessLookupError, ValueError):
                # Процесс уже завершен
                pass
            except Exception as e:
                # В случае любой другой ошибки пытаемся убить процесс
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None
        
        print("Сервер остановлен")
    
    def __enter__(self):
        """Контекстный менеджер: вход"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход - автоматическое завершение"""
        self.stop()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='GStreamer Video Streaming Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Трансляция с веб-камеры (автоматически определяет IP в локальной сети)
  python server.py
  
  # Трансляция видео файла
  python server.py --source file --file video.mp4
  
  # Трансляция с другими параметрами
  python server.py --width 1280 --height 720 --fps 25
  
  # Использование другой веб-камеры (Windows/macOS)
  python server.py --device 1
  
  # Только локальная трансляция (без сети)
  python server.py --host 127.0.0.1
        """
    )
    parser.add_argument('--source', choices=['webcam', 'file'], default='webcam',
                        help='Источник видео: webcam или file (по умолчанию: webcam, автоматически)')
    parser.add_argument('--file', type=str, default=None,
                        help='Путь к видео файлу (если source=file)')
    parser.add_argument('--host', type=str, default=None,
                        help='IP адрес для отправки (по умолчанию: автоматически определяется локальный IP)')
    parser.add_argument('--port', type=int, default=5004,
                        help='Порт для видео (по умолчанию: 5004)')
    parser.add_argument('--width', type=int, default=640,
                        help='Ширина видео (по умолчанию: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Высота видео (по умолчанию: 480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Частота кадров (по умолчанию: 30)')
    parser.add_argument('--device', type=int, default=0,
                        help='Индекс видеоустройства (по умолчанию: 0)')
    parser.add_argument('--bitrate', type=int, default=2000,
                        help='Битрейт видео в kbps (по умолчанию: 2000, для низкой задержки используйте 1000-3000)')
    
    args = parser.parse_args()
    
    # Проверка наличия GStreamer
    check_gstreamer()
    
    server = VideoStreamServer(
        source=args.source,
        file_path=args.file,
        host=args.host,
        video_port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        device_index=args.device,
        bitrate=args.bitrate
    )
    
    # Регистрируем автоматическое завершение при выходе из программы
    atexit.register(server.stop)
    
    # Обработка сигналов для корректного завершения
    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)
    
    # На Windows SIGTERM может не работать, но SIGINT должен работать
    if platform.system() != 'Windows':
        signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Используем контекстный менеджер для гарантированного завершения
    try:
        with server:
            server.start()
    except KeyboardInterrupt:
        # Ctrl+C уже обрабатывается в signal_handler, но на всякий случай
        server.stop()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        server.stop()
        sys.exit(1)

if __name__ == '__main__':
    main()
