@echo off
rem AgentX deployment bundle - Windows starter.
rem Starts the management dashboard (docker compose up -d), then opens the
rem access token in Notepad and the dashboard in your browser.
rem Requires Docker Desktop with WSL 2 integration (the default install).
setlocal
cd /d "%~dp0"

where wsl >nul 2>nul || (
  echo This starter needs WSL 2 ^(installed with Docker Desktop by default^).
  echo Alternatively run from a WSL shell:  docker compose up -d
  pause
  exit /b 1
)

rem Same-path bind mounts need a Linux-style path, so run compose via WSL
rem from this directory's /mnt/... translation.
for /f "usebackq delims=" %%p in (`wsl wslpath -a "%CD%"`) do set "LNXDIR=%%p"
if not defined LNXDIR (
  echo Could not translate this folder to a WSL path.
  pause
  exit /b 1
)

echo Starting the AgentX manager...
wsl -e bash -lc "cd '%LNXDIR%' && docker compose up -d" || (
  echo Start failed - is Docker Desktop running with WSL integration enabled?
  pause
  exit /b 1
)

echo Waiting for the access token...
:wait_token
if not exist ".manager-token" (
  timeout /t 2 /nobreak >nul
  goto wait_token
)

start "" "http://127.0.0.1:12320"
start "" notepad ".manager-token"
echo.
echo Dashboard: http://127.0.0.1:12320  (paste the token from Notepad)
endlocal
