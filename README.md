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
python server.py --host 0.0.0.0 --signal-port 8080
```

По умолчанию сервер шлёт broadcast на UDP порт `5003`, и клиент может найти его автоматически.

### Клиент (автообнаружение)

```bash
python client.py --auto
```

### Клиент (вручную)

```bash
python client.py --host <IP_СЕРВЕРА> --signal-port 8080
```

## Параметры

### server.py
- **--source**: `webcam` (по умолчанию) или `file`
- **--file**: путь к файлу (для `--source file`)
- **--device**: индекс камеры (по умолчанию `0`)
- **--width/--height/--fps**: параметры кадра
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
