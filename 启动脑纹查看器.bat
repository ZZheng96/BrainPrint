@echo off
setlocal
cd /d "%~dp0"
set PORT=8766
set URL=http://127.0.0.1:%PORT%/

rem Server already running: just open the browser
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if %errorlevel%==0 (
  echo Viewer already running at %URL%
  start "" "%URL%"
  exit /b 0
)

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" if not exist ".venv\Scripts\python.exe" set PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe

echo Starting BrainPrint viewer at %URL%  (close this window to stop)
"%PY%" "vis\viewer_server.py" --port %PORT%
if errorlevel 1 pause
