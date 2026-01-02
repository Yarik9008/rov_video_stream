# Скрипт для добавления GStreamer в PATH
# Запустите от имени администратора для добавления в системный PATH

Write-Host "Поиск GStreamer..." -ForegroundColor Yellow

# Возможные пути установки GStreamer
$possiblePaths = @(
    "C:\gstreamer\1.0\msvc_x86_64\bin",
    "C:\gstreamer\1.0\mingw_x86_64\bin",
    "C:\gstreamer\1.0\msvc_x86\bin",
    "C:\gstreamer\1.0\mingw_x86\bin",
    "$env:ProgramFiles\GStreamer\1.0\msvc_x86_64\bin",
    "$env:ProgramFiles\GStreamer\1.0\mingw_x86_64\bin",
    "$env:ProgramFiles(x86)\GStreamer\1.0\msvc_x86_64\bin",
    "$env:ProgramFiles(x86)\GStreamer\1.0\mingw_x86_64\bin"
)

$gstreamerPath = $null

# Поиск gst-launch-1.0.exe
foreach ($path in $possiblePaths) {
    if (Test-Path "$path\gst-launch-1.0.exe") {
        $gstreamerPath = $path
        Write-Host "GStreamer найден: $gstreamerPath" -ForegroundColor Green
        break
    }
}

# Если не найден, поиск по всей системе
if (-not $gstreamerPath) {
    Write-Host "Поиск GStreamer по всей системе (это может занять время)..." -ForegroundColor Yellow
    $found = Get-ChildItem -Path "C:\" -Filter "gst-launch-1.0.exe" -Recurse -ErrorAction SilentlyContinue -Depth 4 | Select-Object -First 1
    if ($found) {
        $gstreamerPath = $found.DirectoryName
        Write-Host "GStreamer найден: $gstreamerPath" -ForegroundColor Green
    }
}

if (-not $gstreamerPath) {
    Write-Host "GStreamer не найден автоматически." -ForegroundColor Red
    Write-Host "Пожалуйста, укажите путь к папке bin GStreamer вручную:" -ForegroundColor Yellow
    $gstreamerPath = Read-Host "Путь к bin (например, C:\gstreamer\1.0\msvc_x86_64\bin)"
    
    if (-not (Test-Path "$gstreamerPath\gst-launch-1.0.exe")) {
        Write-Host "Ошибка: gst-launch-1.0.exe не найден в указанном пути!" -ForegroundColor Red
        exit 1
    }
}

# Проверка текущего PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -like "*$gstreamerPath*") {
    Write-Host "GStreamer уже добавлен в PATH пользователя" -ForegroundColor Green
} else {
    # Добавление в PATH пользователя
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$gstreamerPath", "User")
    Write-Host "GStreamer добавлен в PATH пользователя" -ForegroundColor Green
}

# Обновление PATH в текущей сессии
$env:Path = "$env:Path;$gstreamerPath"

# Проверка
Write-Host "`nПроверка установки..." -ForegroundColor Yellow
$gstVersion = & "$gstreamerPath\gst-launch-1.0.exe" --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "GStreamer работает!" -ForegroundColor Green
    Write-Host $gstVersion
} else {
    Write-Host "Предупреждение: не удалось запустить gst-launch-1.0" -ForegroundColor Yellow
}

Write-Host "`nВАЖНО: Перезапустите терминал/IDE, чтобы изменения PATH вступили в силу!" -ForegroundColor Cyan
