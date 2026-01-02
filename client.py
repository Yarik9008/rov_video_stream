#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GStreamer Video Streaming Client
Принимает и воспроизводит видео, транслируемое сервером
Поддерживает Windows, Linux и macOS
"""

import subprocess
import sys
import argparse
import signal
import os
import threading
import platform
import shutil
import socket
import json
import time
import atexit

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

def discover_server(timeout=5):
    """Автоматически находит сервер в локальной сети через UDP broadcast"""
    DISCOVERY_PORT = 5003
    
    try:
        # Создаем UDP socket для прослушивания
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1)
        sock.bind(('', DISCOVERY_PORT))
        
        print(f"Поиск сервера в локальной сети (таймаут: {timeout} сек)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                data, addr = sock.recvfrom(1024)
                try:
                    server_info = json.loads(data.decode('utf-8'))
                    sock.close()
                    print(f"✓ Сервер найден: {server_info['host']}:{server_info['video_port']}")
                    return server_info
                except (json.JSONDecodeError, KeyError):
                    continue
            except socket.timeout:
                continue
        
        sock.close()
        return None
    except Exception as e:
        return None

class VideoStreamClient:
    # Порт для UDP broadcast обнаружения
    DISCOVERY_PORT = 5003
    
    def __init__(self, host='127.0.0.1', video_port=5004, audio_port=5006, auto_discover=False):
        self.host = host
        self.video_port = video_port
        self.audio_port = audio_port
        self.process = None
        self.system = platform.system()
        self.auto_discover = auto_discover
        
    def build_pipeline(self):
        """Строит GStreamer pipeline для клиента"""
        # Прием RTP потока и воспроизведение
        # Оптимизированные настройки для минимальной задержки:
        # - buffer-size=65536: минимальный буфер для уменьшения задержки
        # - sync=false: отключение синхронизации для минимальной задержки
        # - drop-on-latency=true: сброс кадров при задержке
        # - max-lateness=-1: игнорировать задержку
        # autovideosink автоматически выберет подходящий видеовыход для платформы
        pipeline = (
            f"udpsrc port={self.video_port} buffer-size=65536 ! "
            f"application/x-rtp,encoding-name=H264,payload=96 ! "
            f"rtph264depay ! "
            f"h264parse ! "
            f"avdec_h264 ! "
            f"videoconvert ! "
            f"autovideosink sync=false drop-on-latency=true max-lateness=-1"
        )
        
        return pipeline
    
    def start(self):
        """Запускает клиент"""
        # Автоматический поиск сервера, если включен
        if self.auto_discover and self.host == '127.0.0.1':
            server_info = discover_server(timeout=5)
            if server_info:
                self.host = server_info['host']
                self.video_port = server_info['video_port']
                if 'audio_port' in server_info:
                    self.audio_port = server_info['audio_port']
            else:
                print("⚠ Сервер не найден в локальной сети, используем локальный адрес")
        
        pipeline = self.build_pipeline()
        gst_cmd = find_gstreamer()
        if not gst_cmd:
            gst_cmd = check_gstreamer()
        
        print(f"Запуск клиента...")
        print(f"Платформа: {self.system}")
        print(f"Подключение к: {self.host}:{self.video_port}")
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
            print("\nОстановка клиента...")
            self.stop()
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        """Останавливает клиент"""
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
        
        print("Клиент остановлен")
    
    def __enter__(self):
        """Контекстный менеджер: вход"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход - автоматическое завершение"""
        self.stop()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='GStreamer Video Streaming Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Автоматический поиск сервера в локальной сети
  python client.py --auto
  # или
  python client.py --host auto
  
  # Подключение к локальному серверу
  python client.py
  
  # Подключение к удаленному серверу
  python client.py --host 192.168.1.100 --port 5004
        """
    )
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='IP адрес сервера (по умолчанию: 127.0.0.1, используйте "auto" для автообнаружения)')
    parser.add_argument('--port', type=int, default=5004,
                        help='Порт для видео (по умолчанию: 5004)')
    parser.add_argument('--auto', action='store_true', default=False,
                        help='Автоматически найти сервер в локальной сети')
    
    args = parser.parse_args()
    
    # Если указан "auto" в host, включаем автообнаружение
    auto_discover = args.auto or (args.host.lower() == 'auto')
    if auto_discover:
        args.host = '127.0.0.1'  # Временно, будет заменено при обнаружении
    
    # Проверка наличия GStreamer
    check_gstreamer()
    
    client = VideoStreamClient(
        host=args.host,
        video_port=args.port,
        auto_discover=auto_discover
    )
    
    # Регистрируем автоматическое завершение при выходе из программы
    atexit.register(client.stop)
    
    # Обработка сигналов для корректного завершения
    def signal_handler(sig, frame):
        client.stop()
        sys.exit(0)
    
    # На Windows SIGTERM может не работать, но SIGINT должен работать
    if platform.system() != 'Windows':
        signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Используем контекстный менеджер для гарантированного завершения
    try:
        with client:
            client.start()
    except KeyboardInterrupt:
        # Ctrl+C уже обрабатывается в signal_handler, но на всякий случай
        client.stop()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        client.stop()
        sys.exit(1)

if __name__ == '__main__':
    main()
