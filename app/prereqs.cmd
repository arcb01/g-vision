@echo off
rem Called by G-VISION.bat first: makes sure Node.js 22+ and Python 3.11+
rem are installed, installing them with winget when they are not, and adds
rem them to PATH for this window (new windows get them from the system PATH).
rem Exits with 1 and says what to install when that is not possible.

call :has_node && goto python
echo Node.js 22 or newer was not found. Installing it with winget...
call :winget OpenJS.NodeJS.LTS
set "PATH=%ProgramFiles%\nodejs;%PATH%"
call :has_node && goto python
echo.
echo G-VISION needs Node.js 22 or newer and could not install it by itself.
echo Install the LTS version from https://nodejs.org, then run G-VISION.bat again.
exit /b 1

:python
call :has_python && exit /b 0
echo Python 3.11 or newer was not found. Installing Python 3.12 with winget...
call :winget Python.Python.3.12
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%ProgramFiles%\Python312;%PATH%"
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
winget install --id %1 --exact --silent --accept-package-agreements --accept-source-agreements
exit /b
