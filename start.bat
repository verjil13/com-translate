@echo off
setlocal

:: === НАСТРОЙКИ ПУТЕЙ ===
set "PROJECT_DIR=%~dp0"
set "ENV_PATH=%PROJECT_DIR%env"

cd /d "%PROJECT_DIR%"

:: Активация окружения
call "C:\Users\13ver\anaconda3\Scripts\activate.bat" "%ENV_PATH%"

:: === ИСПРАВЛЕНИЕ OpenMP КОНФЛИКТА ===
echo Fixing OpenMP conflict...

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "LLAMA_LIB=%ENV_PATH%\Lib\site-packages\llama_cpp\lib"

if exist "%LLAMA_LIB%\libomp140.x86_64.dll" (
    ren "%LLAMA_LIB%\libomp140.x86_64.dll" libomp140.x86_64.dll.bak
    echo [OK] libomp140.x86_64.dll renamed to .bak
) else (
    echo [INFO] libomp140.x86_64.dll not found
)

echo Starting Comic Translate from LOCAL env...
python comic.py
pause