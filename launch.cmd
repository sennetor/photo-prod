@echo off
rem Photo Prod launcher: starts the AI service and editor web server
rem (only if not already running), then opens the editor in the browser.

cd /d "%~dp0"

rem AI service (port 8765)
netstat -ano | findstr ":8765" | findstr "LISTENING" >nul
if errorlevel 1 (
  start "Photo Prod GenAI" /min "%~dp0sam-env\Scripts\python.exe" "%~dp0genai-service\server.py"
)

rem Editor web server (port 4173)
netstat -ano | findstr ":4173" | findstr "LISTENING" >nul
if errorlevel 1 (
  start "Photo Prod Editor" /min cmd /c npx --yes http-server "%~dp0editor" -p 4173 -c-1
)

rem brief wait for first-start binding (ping trick: works without stdin)
ping -n 3 127.0.0.1 >nul
start "" http://localhost:4173
