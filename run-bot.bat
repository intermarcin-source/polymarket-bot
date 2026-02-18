@echo off
cd /d "%~dp0"
echo ============================================
echo   POLYMARKET TRADING BOT
echo   Running continuously (5 min cycles)
echo   Press Ctrl+C to stop
echo ============================================
echo.
set PYTHONUTF8=1
venv\Scripts\python.exe main.py --interval 5 --max-cycles 0
pause
