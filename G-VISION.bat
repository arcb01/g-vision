@echo off
rem Double-click to start G-VISION: installs Node.js and Python if they are
rem missing (app\prereqs.cmd), pulls the latest version (and reinstalls
rem dependencies only when they changed), then opens the app, which starts
rem Qwen (llama-server) and the Python backend itself and stops them when you
rem close it. Logs go to logs\. Pass --demo for the synthetic demo without Qwen,
rem or --server to start as the AI server for a gaming PC (see README).
rem
rem Everything below is one block so cmd reads it before git pull can rewrite
rem this file, and exit /b keeps it from reading on into the new version.
setlocal
(
  cd /d "%~dp0app"
  call "%~dp0app\prereqs.cmd" || (pause & exit /b 1)
  node "%~dp0app\update.js"
  if errorlevel 2 timeout /t 20
  if not exist "%~dp0app\node_modules\electron\package.json" (
    echo Installing the app's dependencies, first run only...
    call npm install || (pause & exit /b 1)
  )
  rem Electron 44 and later no longer download their binary during npm
  rem install. This does, or again after an update changes the version.
  node "%~dp0app\node_modules\electron\install.js"
  if not exist "%~dp0app\node_modules\electron\dist\electron.exe" (
    echo.
    echo Could not download Electron, the app's window. Check the internet
    echo connection and run G-VISION.bat again.
    pause
    exit /b 1
  )
  start "" "%~dp0app\node_modules\electron\dist\electron.exe" "%~dp0app" %*
  exit /b
)
