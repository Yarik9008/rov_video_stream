# ROV Video Stream (WebRTC)

Минимальный проект для **низколатентной** трансляции видео по **WebRTC** в локальной сети:

- **Сервер**: захват камеры/файла через OpenCV → WebRTC
- **Клиент**: автообнаружение сервера по UDP broadcast → WebRTC → окно OpenCV

## Требования

- Python 3.9+ (рекомендуется 3.10+)
- Зависимости:

```bash
pip install -r requirements.txt
```

## Запуск

### Сервер (LAN, авто‑IP, broadcast discovery)

```bash
python3 server.py --host 0.0.0.0 --signal-port 8080
```

## GPU WebRTC (GStreamer, NVIDIA NVENC/NVDEC)

Если нужна **аппаратная** кодировка/декодировка на видеокарте (NVIDIA):
- **сервер**: `NVENC` (`nvh264enc`)
- **клиент**: `NVDEC` (`nvh264dec`)

### Требования (Windows)

У вас должен быть установлен **GStreamer MSVC x86_64** (как минимум плагины `webrtcbin` и `nvcodec`):

- `gst-inspect-1.0 webrtcbin`
- `gst-inspect-1.0 nvh264enc`
- `gst-inspect-1.0 nvh264dec`

Также для управления `webrtcbin` из Python нужен GI typelib:

- `GstWebRTC-1.0.typelib` (namespace `GstWebRTC`)

Он должен находиться в:

`C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\girepository-1.0`

Если `GstWebRTC-1.0.typelib` отсутствует, `server_gst.py/client_gst.py` не смогут делать offer/answer/ICE.

Проверка (должно вывести `OK`):

```bash
python -c "import sys; sys.path.insert(0, r'C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\site-packages'); import gi; gi.require_version('GstWebRTC','1.0'); from gi.repository import GstWebRTC; print('OK')"
```

Если падает `Namespace GstWebRTC not available`, значит в вашей сборке не установлены typelibs WebRTC.
Обычно помогает установка *Development* пакетов GStreamer, особенно для `-bad` (WebRTC относится к `gst-plugins-bad`).

Важно: в официальной сборке GStreamer Python bindings (`gi`) лежат внутри:

`C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\site-packages`

Скрипты `server_gst.py` / `client_gst.py` **автоматически добавляют** этот путь в `sys.path`
и настраивают `GI_TYPELIB_PATH`/DLL paths.

Если у вас нестандартный путь установки — задайте переменную окружения:

`GSTREAMER_1_0_ROOT_MSVC_X86_64`

### Запуск GPU сервера

```bash
python3 server_gst.py --host 0.0.0.0 --transport rtp --width 3840 --height 2160 --fps 60 --bitrate 250000
```

Примечания:
- `--bitrate` задаётся в **kbit/sec** (например 250000 = 250 Mbps)
- для минимальной задержки можно оставить `--webrtc-latency 0`

### Запуск GPU клиента (auto-discovery)

```bash
python3 client_gst.py --auto
```

### Почему по умолчанию RTP, а не WebRTC

На Windows в некоторых сборках GStreamer отсутствует `GstWebRTC-1.0.typelib`, и тогда Python не может делать negotiation через `webrtcbin`.
Поэтому `server_gst.py` по умолчанию использует **RTP/UDP**, который:
- не требует `GstWebRTC` typelibs
- всё равно использует **NVENC/NVDEC**
- даёт очень низкую задержку в LAN

RTP поток по умолчанию идёт в multicast `udp://239.255.0.1:5004`.
Если multicast в сети запрещён — укажите `--rtp-host <IP_клиента>` и откройте UDP порт `5004` в firewall.

### Режим file passthrough (макс. качество/мин. задержка, но без NVENC)

Если файл уже H.264 (например `mp4`), можно отправлять его **без перекодирования**
(это даёт максимально возможное качество и минимальную задержку, но NVENC не используется):

```bash
python3 server_gst.py --source file --file video/test.mp4 --file-passthrough --host 0.0.0.0
```

### Запуск сервера

**Оптимальные настройки по умолчанию** (баланс качества и задержки):

```bash
python3 server.py --host 0.0.0.0
```

**Параметры по умолчанию:**
- Разрешение: 1080p (1920x1080)
- FPS: 30
- Битрейт: автоматический расчет (~15 Mbps для 1080p@30fps)
- Задержка: минимальная

### Дополнительные опции

```bash
# Кастомное разрешение (например, 4K)
python3 server.py --host 0.0.0.0 --width 3840 --height 2160

# Нативное разрешение источника
python3 server.py --host 0.0.0.0 --width 0 --height 0

# Кастомный FPS
python3 server.py --host 0.0.0.0 --fps 60

# Ручной битрейт (по умолчанию автоматический)
python3 server.py --host 0.0.0.0 --bitrate 20000
```

**Важно**: 
- Битрейт рассчитывается автоматически на основе разрешения и FPS
- Для ручной настройки используйте `--bitrate` в kbps

По умолчанию сервер шлёт broadcast на UDP порт `5003`, и клиент может найти его автоматически.

### Клиент (автообнаружение)

```bash
python3 client.py --auto
```

### Клиент (вручную)

```bash
python3 client.py --host <IP_СЕРВЕРА> --signal-port 8080
```

## Параметры

### server.py
- **--source**: `webcam` (по умолчанию) или `file`
- **--file**: путь к файлу (для `--source file`)
- **--device**: индекс камеры (по умолчанию `0`)
- **--width/--height**: разрешение кадра (по умолчанию `1920x1080` = 1080p; `0` = исходное разрешение)
- **--fps**: частота кадров (по умолчанию `30`)
- **--bitrate**: битрейт в kbps (по умолчанию `10000` = 10 Mbps)
- **--low-latency**: режим низкой задержки (по умолчанию ВКЛ, используйте `--no-low-latency` для отключения)
- **--max-quality**: режим максимального качества (по умолчанию ВЫКЛ, автоматический расчет битрейта)
- **--signal-port**: порт HTTP signaling (по умолчанию `8080`)
- **--stun**: включить публичный STUN (для LAN обычно не нужно)

### client.py
- **--host**: IP сервера или `auto`
- **--auto**: автообнаружение по broadcast
- **--signal-port**: порт HTTP signaling (по умолчанию `8080`)
- **--stun**: включить публичный STUN (для LAN обычно не нужно)

## Управление

- Закрыть клиент: **Q** или **ESC** в окне видео
- Остановить сервер: **Ctrl+C**
