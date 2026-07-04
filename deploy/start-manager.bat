@echo off
rem AgentX deployment bundle - Windows starter.
rem Starts the management dashboard (docker compose up -d), then opens the
rem access token in Notepad and the dashboard in your browser.
rem Requires Docker Desktop with WSL 2 integration and a Linux distro
rem (e.g. Ubuntu) - both part of the default Docker Desktop install flow.
setlocal
cd /d "%~dp0"

if not exist ".env.example" (
  echo Run this from the extracted agentx-deploy folder - .env.example is missing here.
  pause
  exit /b 1
)

rem First run: create .env from the template (secrets are generated
rem automatically on the dashboard's first Start - no editing required).
if not exist ".env" copy /y ".env.example" ".env" >nul

where wsl >nul 2>nul || (
  echo This starter needs WSL 2 ^(installed with Docker Desktop by default^).
  echo Alternatively run from a WSL shell:  docker compose up -d
  pause
  exit /b 1
)

rem Same-path bind mounts need a Linux-style path, so run compose via WSL
rem from this directory's /mnt/... translation. Note: wsl.exe exists even
rem when no Linux distro is installed - in that case its (UTF-16) error
rem text would land in LNXDIR, so validate the result looks like a path.
set "LNXDIR="
for /f "usebackq delims=" %%p in (`wsl wslpath -a "%CD%" 2^>nul`) do if not defined LNXDIR set "LNXDIR=%%p"
if not defined LNXDIR goto no_distro
if not "%LNXDIR:~0,1%"=="/" goto no_distro

echo Starting the AgentX manager...
rem Single quotes around the path on purpose: escaped double quotes would
rem toggle cmd's quote state and expose the && to cmd itself. Trade-off:
rem a path containing an apostrophe breaks (rare; documented).
wsl -e bash -lc "cd '%LNXDIR%' && docker compose up -d" || (
  echo Start failed - is Docker Desktop running with WSL integration enabled?
  pause
  exit /b 1
)

echo Waiting for the access token...
set /a tries=0
rem Flat loop on purpose: %tries% inside an if-block would expand at
rem block-parse time (cmd's delayed-expansion trap) and never advance.
:wait_token
if exist ".manager-token" goto token_ready
set /a tries+=1
if %tries% GEQ 60 goto token_timeout
timeout /t 2 /nobreak >nul
goto wait_token

:token_ready
start "" "http://127.0.0.1:12320"
start "" notepad ".manager-token"
echo.
echo Dashboard: http://127.0.0.1:12320  (paste the token from Notepad)
endlocal
exit /b 0

:token_timeout
echo The manager did not produce a token within 2 minutes. Check its logs:
echo   wsl -e bash -lc "cd '%LNXDIR%' && docker compose logs manager"
pause
exit /b 1

:no_distro
echo WSL has no Linux distribution ready. Install one ^(e.g. Ubuntu^):
echo   wsl --install -d Ubuntu
echo then enable it under Docker Desktop ^> Settings ^> Resources ^> WSL integration,
echo and run this starter again.
pause
exit /b 1
