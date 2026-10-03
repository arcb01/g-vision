@echo off
rem Double-click to start G-VISION: pulls the latest version (and reinstalls
rem dependencies only when they changed), then opens the app, which starts
rem Qwen (llama-server) and the Python backend itself and stops them when you
rem close it. Logs go to logs\. Pass --demo for the synthetic demo without Qwen.
rem
rem Everything below is one block so cmd reads it before git pull can rewrite
rem this file, and exit /b keeps it from reading on into the new version.
setlocal
(
  cd /d "%~dp0app"
  where node >nul 2>&1 && node "%~dp0app\update.js"
  if errorlevel 2 timeout /t 20
  if not exist "%~dp0app\node_modules\electron\dist\electron.exe" (
    echo Installing the app's dependencies, first run only...
    call npm install || (pause & exit /b 1)
  )
  start "" "%~dp0app\node_modules\electron\dist\electron.exe" "%~dp0app" %*
  exit /b
)
