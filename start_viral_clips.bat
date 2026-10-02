@echo off
setlocal EnableExtensions
title YouTube to Viral Clips

:: ============================================================
:: AUTO ADMIN
:: ============================================================

net session >nul 2>&1
if errorlevel 1 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

:: ============================================================
:: CONFIG
:: ============================================================

set "APP_DIR=E:\youtube-to-viral-clips"
set "APP_URL=http://127.0.0.1:5000"
set "OLLAMA_URL=http://127.0.0.1:11434/api/tags"

cd /d "%APP_DIR%"

cls
echo ============================================================
echo             YOUTUBE TO VIRAL CLIPS
echo ============================================================
echo.


:: ============================================================
:: CHECK PYTHON
:: ============================================================

echo [1/4] Checking Python 3.14...

py -3.14 --version
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.14 tidak ditemukan.
    echo.
    pause
    exit /b 1
)

echo.


:: ============================================================
:: CHECK OLLAMA
:: ============================================================

echo [2/4] Checking Ollama...

curl.exe -s "%OLLAMA_URL%" >nul 2>&1

if errorlevel 1 (
    echo Ollama isn't loaded.
    echo Starting Ollama...

    start "Ollama Server" /min cmd /c "ollama serve"

    echo Waiting 5 seconds for Ollama...
    timeout /t 5 /nobreak >nul
) else (
    echo Ollama already running.
)

:: Test again
curl.exe -s "%OLLAMA_URL%" >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Ollama gagal dijalankan.
    echo.
    echo Coba jalankan:
    echo ollama serve
    echo.
    pause
    exit /b 1
)

echo Ollama ready.
echo.


:: ============================================================
:: START BROWSER DELAYED
:: ============================================================

echo [3/4] Preparing browser...

start "" powershell -NoProfile -WindowStyle Hidden -Command ^
    "Start-Sleep -Seconds 4; Start-Process '%APP_URL%'"

echo Browser will open automatically.
echo.


:: ============================================================
:: START APP
:: ============================================================

echo [4/4] Starting Viral Clips...
echo.
echo ============================================================
echo.
echo Viral Clips is starting.
echo.
echo Keep this window open while using the application.
echo.
echo To stop:
echo Press CTRL+C
echo.
echo ============================================================
echo.

py -3.14 app.py


:: ============================================================
:: CLEANUP AFTER APP STOPS
:: ============================================================

echo.
echo ============================================================
echo Viral Clips stopped.
echo Cleaning temporary files...
echo ============================================================

if exist "%APP_DIR%\__pycache__" (
    rmdir /s /q "%APP_DIR%\__pycache__"
)

del /q "%TEMP%\tmp*.ass" >nul 2>&1

echo.
echo Cleanup complete.
echo.

pause
endlocal