@echo off
rem Double-click to stop SLM Router (Windows). Downloaded models are kept for next time.
cd /d "%~dp0"
echo Stopping SLM Router...
docker compose down && echo Stopped.
echo.
pause
