@echo off
rem Double-click to start G-VISION: the app starts Qwen (llama-server) and the
rem Python backend itself and stops them when you close it. Logs go to logs\.
rem Pass --demo for the synthetic demo without Qwen.
setlocal
cd /d "%~dp0app"
if not exist node_modules\electron\dist\electron.exe (
  echo Installing the app's dependencies, first run only...
  call npm install || (pause & exit /b 1)
)
start "" "%~dp0app\node_modules\electron\dist\electron.exe" "%~dp0app" %*
