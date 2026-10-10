@echo off
rem Double-click to start G-VISION. app\setup.ps1 downloads whatever is
rem missing (Node.js, Git, Python) into runtime\ without installers or admin
rem rights, pulls the latest version (and reinstalls dependencies only when they
rem changed), then opens the app, which starts Qwen (llama-server) and the
rem Python backend itself and stops them when you close it. Logs go to logs\.
rem Pass --demo for the synthetic demo without Qwen, or --server to start as
rem the AI server for a gaming PC (see README).
rem
rem Everything below is one block so cmd reads it before git pull can rewrite
rem this file, and exit /b keeps it from reading on into the new version.
setlocal
(
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\setup.ps1" %*
  if errorlevel 1 (pause & exit /b 1)
  exit /b 0
)
