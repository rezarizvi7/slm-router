@echo off
rem Double-click to start SLM Router (Windows).
setlocal
cd /d "%~dp0"
title SLM Router

echo === SLM Router ===
echo.

where docker >nul 2>&1
if errorlevel 1 (
  echo Docker is not installed.
  echo Install Docker Desktop from https://www.docker.com/products/docker-desktop
  echo then double-click Start again.
  start "" "https://www.docker.com/products/docker-desktop"
  goto :fail
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Starting Docker Desktop...
  if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
  for /l %%i in (1,1,60) do (
    docker info >nul 2>&1 && goto :docker_ok
    timeout /t 2 /nobreak >nul
  )
  echo Docker didn't start. Open Docker Desktop yourself, wait until it says it's running,
  echo then double-click Start again.
  goto :fail
)
:docker_ok

if not exist .env (
  copy /y .env.example .env >nul
  echo First-time setup.
  echo Paste your Anthropic API key so hard questions can go to Claude.
  echo ^(Get one at https://console.anthropic.com - or just press Enter to skip;
  echo  everything will then be answered by the free local model.^)
  set "KEY="
  set /p "KEY=API key: "
  call :savekey
  echo.
)

echo Starting everything. The FIRST time this downloads about 1.5 GB and can take 5-15 minutes.
echo.
docker compose up -d --build
if errorlevel 1 (
  echo.
  echo Something went wrong starting the containers ^(see the messages above^).
  goto :fail
)

echo.
echo Waiting for the router to be ready...
for /l %%i in (1,1,450) do (
  curl -s http://localhost:8080/healthz >nul 2>&1 && goto :ready
  timeout /t 2 /nobreak >nul
)
echo The router didn't come up in time. Recent log lines:
docker compose logs --tail 30 router
goto :fail

:ready
echo.
echo Ready!
echo   Chat:      http://localhost:8080
echo   Dashboard: http://localhost:3000
start "" "http://localhost:8080"
echo.
echo It keeps running in the background. Double-click Stop when you're done.
pause
exit /b 0

:savekey
if not defined KEY exit /b 0
powershell -NoProfile -Command "$k=$env:KEY; (Get-Content .env) | ForEach-Object { if ($_ -like 'ANTHROPIC_API_KEY=*') { 'ANTHROPIC_API_KEY=' + $k } else { $_ } } | Set-Content -Encoding ascii .env"
echo Saved.
exit /b 0

:fail
echo.
pause
exit /b 1
