@echo off
setlocal

:: === НАСТРОЙКИ ПУТЕЙ ===
set "PROJECT_DIR=%~dp0"
:: Путь к локальному окружению (папка env внутри проекта)
set "ENV_PATH=%PROJECT_DIR%env"

:: Путь к site-packages внутри локального окружения
set "SITE_PACKAGES=%ENV_PATH%\Lib\site-packages"


:: --- ЗАПУСК ПРОГРАММЫ ---
cd /d "%PROJECT_DIR%"

:: Активация окружения по пути (указываем полный путь к activate.bat вашей Conda)
:: Если путь к Conda другой — поправьте его тут
call "C:\Users\13ver\anaconda3\Scripts\activate.bat" "%ENV_PATH%"

echo Starting Comic Translate from LOCAL env...
python comic.py

pause