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

Важно: в официальной сборке GStreamer Python bindings (`gi`) лежат внутри:

`C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\site-packages`

Скрипты `server_gst.py` / `client_gst.py` **автоматически добавляют** этот путь в `sys.path`
и настраивают `GI_TYPELIB_PATH`/DLL paths.

Если у вас нестандартный путь установки — задайте переменную окружения:

`GSTREAMER_1_0_ROOT_MSVC_X86_64`

### Запуск GPU сервера

```bash
python3 server_gst.py --host 0.0.0.0 --signal-port 8080 --width 3840 --height 2160 --fps 60 --bitrate 250000
```

Примечания:
- `--bitrate` задаётся в **kbit/sec** (например 250000 = 250 Mbps)
- для минимальной задержки можно оставить `--webrtc-latency 0`

### Запуск GPU клиента (auto-discovery)

```bash
python3 client_gst.py --auto
```

### Режим file passthrough (макс. качество/мин. задержка, но без NVENC)

Если файл уже H.264 (например `mp4`), можно отправлять его **без перекодирования**
(это даёт максимально возможное качество и минимальную задержку, но NVENC не используется):

```bash
python3 server_gst.py --source file --file video/test.mp4 --file-passthrough --host 0.0.0.0
```

### Режим максимального качества

Для максимального качества используйте режим `--max-quality` (автоматический расчет битрейта):

```bash
python3 server.py --host 0.0.0.0 --max-quality --fps 60
```

Или вручную настройте параметры:

```bash
# Высокое качество (1080p, 30 FPS, 20 Mbps)
python3 server.py --host 0.0.0.0 --width 1920 --height 1080 --fps 30 --bitrate 20000

# Максимальное качество (4K, 60 FPS, 200 Mbps)
python3 server.py --host 0.0.0.0 --width 3840 --height 2160 --fps 60 --bitrate 200000
```

**Важно**: 
- По умолчанию сервер пытается отдавать **4K (3840x2160)**. Если нужно нативное разрешение источника — используйте `--width 0 --height 0`
- Битрейт по умолчанию: 200 Mbps (200000 kbps)
- В режиме `--max-quality` битрейт рассчитывается автоматически на основе разрешения и FPS

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
- **--width/--height**: разрешение кадра (по умолчанию `3840x2160` = 4K; `0` = исходное разрешение)
- **--fps**: частота кадров (по умолчанию `30`, рекомендуется `30-60` для максимального качества)
- **--bitrate**: битрейт в kbps (по умолчанию `200000` = 200 Mbps для максимального качества)
- **--max-quality**: режим максимального качества (автоматический расчет битрейта, без ресайза)
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
