# Инструкция по загрузке на GitHub

## Вариант 1: Через веб-интерфейс GitHub

1. Перейдите на https://github.com/new
2. Создайте новый репозиторий:
   - Название: `video_stream` (или любое другое)
   - Описание: "GStreamer video streaming server and client"
   - Выберите Public или Private
   - **НЕ** добавляйте README, .gitignore или лицензию (они уже есть)
3. После создания репозитория выполните команды:

```bash
git remote add origin https://github.com/YOUR_USERNAME/video_stream.git
git branch -M main
git push -u origin main
```

Замените `YOUR_USERNAME` на ваш GitHub username.

## Вариант 2: Через GitHub CLI (если установлен)

```bash
gh repo create video_stream --public --source=. --remote=origin --push
```

## Вариант 3: Через SSH (если настроен SSH ключ)

```bash
git remote add origin git@github.com:YOUR_USERNAME/video_stream.git
git branch -M main
git push -u origin main
```

## После загрузки

Репозиторий будет доступен по адресу:
`https://github.com/YOUR_USERNAME/video_stream`
