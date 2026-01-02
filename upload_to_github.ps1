# Скрипт для загрузки репозитория на GitHub
# Использование: .\upload_to_github.ps1 -Username YOUR_USERNAME -RepoName video_stream

param(
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [Parameter(Mandatory=$false)]
    [string]$RepoName = "video_stream"
)

Write-Host "Подготовка к загрузке на GitHub..." -ForegroundColor Yellow

# Проверка наличия git
try {
    git --version | Out-Null
} catch {
    Write-Host "Ошибка: Git не установлен!" -ForegroundColor Red
    exit 1
}

# Проверка наличия коммитов
$commits = git log --oneline 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка: Нет коммитов в репозитории!" -ForegroundColor Red
    exit 1
}

Write-Host "`nТекущая ветка:" -ForegroundColor Cyan
git branch

Write-Host "`nПоследние коммиты:" -ForegroundColor Cyan
git log --oneline -5

Write-Host "`n=== Инструкция ===" -ForegroundColor Green
Write-Host "1. Создайте репозиторий на GitHub:" -ForegroundColor Yellow
Write-Host "   https://github.com/new" -ForegroundColor White
Write-Host "   Название: $RepoName" -ForegroundColor White
Write-Host "   НЕ добавляйте README, .gitignore или лицензию!" -ForegroundColor Yellow

Write-Host "`n2. После создания репозитория выполните:" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/$Username/$RepoName.git" -ForegroundColor White
Write-Host "   git branch -M main" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor White

Write-Host "`nИли выполните эту команду для автоматической загрузки:" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/$Username/$RepoName.git; git branch -M main; git push -u origin main" -ForegroundColor Cyan

$createNow = Read-Host "`nСоздали репозиторий на GitHub? (y/n)"
if ($createNow -eq "y" -or $createNow -eq "Y") {
    Write-Host "`nДобавление remote..." -ForegroundColor Yellow
    git remote remove origin 2>$null
    git remote add origin "https://github.com/$Username/$RepoName.git"
    
    Write-Host "Переименование ветки в main..." -ForegroundColor Yellow
    git branch -M main
    
    Write-Host "Загрузка на GitHub..." -ForegroundColor Yellow
    Write-Host "Введите ваш GitHub username и password (или Personal Access Token):" -ForegroundColor Cyan
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nУспешно загружено!" -ForegroundColor Green
        Write-Host "Репозиторий: https://github.com/$Username/$RepoName" -ForegroundColor Cyan
    } else {
        Write-Host "`nОшибка при загрузке. Проверьте:" -ForegroundColor Red
        Write-Host "- Правильность username и имени репозитория" -ForegroundColor Yellow
        Write-Host "- Что репозиторий создан на GitHub" -ForegroundColor Yellow
        Write-Host "- Используйте Personal Access Token вместо пароля" -ForegroundColor Yellow
    }
} else {
    Write-Host "`nВыполните команды вручную после создания репозитория." -ForegroundColor Yellow
}
