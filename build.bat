@echo off
echo Building executables with PyInstaller...
echo.

REM Проверяем наличие PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

echo.
echo Building server.exe...
pyinstaller --onefile --name server --icon=NONE --clean server.py
if errorlevel 1 (
    echo Error building server.exe
    pause
    exit /b 1
)

echo.
echo Building client.exe...
pyinstaller --onefile --name client --icon=NONE --clean client.py
if errorlevel 1 (
    echo Error building client.exe
    pause
    exit /b 1
)

echo.
echo Building client_qt.exe...
pyinstaller --onefile --name client_qt --icon=NONE --clean client_qt.py
if errorlevel 1 (
    echo Error building client_qt.exe
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executables are in the 'dist' folder.
pause
