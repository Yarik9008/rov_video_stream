# Сборка исполняемых файлов

## Требования

- Python 3.9+
- Все зависимости из `requirements.txt`
- PyInstaller (установится автоматически при запуске build.py)

## Быстрая сборка

### Windows (batch файл):
```bash
build.bat
```

### Кроссплатформенный способ (Python скрипт):
```bash
python build.py
```

## Ручная сборка с PyInstaller

Если нужно больше контроля над процессом сборки:

```bash
# Установка PyInstaller
pip install pyinstaller

# Сборка server.exe
pyinstaller --onefile --name server server.py

# Сборка client.exe
pyinstaller --onefile --name client client.py

# Сборка client_qt.exe
pyinstaller --onefile --name client_qt client_qt.py
```

## Результат

После сборки исполняемые файлы будут находиться в папке `dist/`:
- `server.exe` - сервер видеопотока
- `client.exe` - клиент с OpenCV GUI
- `client_qt.exe` - клиент с PyQt GUI

## Использование

Запуск исполняемых файлов аналогичен запуску Python скриптов:

```bash
# Сервер
dist\server.exe --host 0.0.0.0

# Клиент (GUI)
dist\client_qt.exe --auto
```

## Примечания

- Первый запуск может быть медленным (PyInstaller распаковывает архив)
- Размер файлов будет больше, чем исходные .py файлы (включают Python интерпретатор и зависимости)
- Для уменьшения размера можно использовать опцию `--onedir` вместо `--onefile` (создаст папку с файлами)
