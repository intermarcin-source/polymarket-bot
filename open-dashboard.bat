@echo off
cd /d "%~dp0"
echo ============================================
echo   POLYMARKET BOT - LIVE DASHBOARD
echo   Auto-refreshes every 60 seconds
echo   Close this window to stop refreshing
echo ============================================
echo.

:: First run: generate and open browser
echo [%TIME%] Generating dashboard...
venv\Scripts\python.exe generate_dashboard.py
echo.

:: Loop: regenerate every 60 seconds (browser auto-reloads via meta refresh)
:loop
timeout /t 60 /nobreak >nul
echo [%TIME%] Refreshing dashboard data...
venv\Scripts\python.exe generate_dashboard.py --no-open 2>nul || venv\Scripts\python.exe generate_dashboard.py
goto loop
