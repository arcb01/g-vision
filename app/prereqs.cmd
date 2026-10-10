@echo off
rem Called by G-VISION.bat first: makes sure Node.js 22+ and Python 3.11+
rem can be run from this window. Node.js is downloaded as a portable folder
rem into runtime\node (no installer, no admin rights) when none is found;
rem Python 3.12 is installed with winget. Exits with 1 and says what to
rem install when that is not possible.

for %%I in ("%~dp0..") do set "ROOT=%%~fI"

call :has_node && goto python
rem Installed but missing from this window's PATH, or downloaded before.
set "PATH=%ROOT%\runtime\node;%ProgramFiles%\nodejs;%PATH%"
call :has_node && goto python
echo Node.js 22 or newer was not found. Downloading it into runtime\node...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0get-node.ps1" -Dest "%ROOT%\runtime\node"
call :has_node && goto python
echo.
echo G-VISION needs Node.js 22 or newer and could not download it by itself.
echo Check the internet connection and run G-VISION.bat again, or install
echo the LTS version from https://nodejs.org and run G-VISION.bat again.
exit /b 1

:python
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%ProgramFiles%\Python312;%PATH%"
call :has_python && exit /b 0
echo Python 3.11 or newer was not found. Installing Python 3.12 with winget...
rem A per-user install needs no admin rights; fall back to the default scope.
call :winget Python.Python.3.12 --scope user || call :winget Python.Python.3.12
call :has_python && exit /b 0
echo.
echo G-VISION needs Python 3.11 or newer and could not install it by itself.
echo Install Python 3.12 from https://www.python.org/downloads/windows/
echo (tick "Add python.exe to PATH"), then run G-VISION.bat again.
exit /b 1

:has_node
node -e "process.exit(+process.versions.node.split('.')[0] < 22)" >nul 2>&1
exit /b

rem The Microsoft Store's python.exe stub fails this check too.
:has_python
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && exit /b 0
python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
exit /b

:winget
where winget >nul 2>&1 || (echo winget is not available on this PC. & exit /b 1)
winget install --id %1 --exact --silent --accept-package-agreements --accept-source-agreements %2 %3
exit /b
