@echo off
echo ==========================================
echo  Polymarket Trading Bot - Setup
echo ==========================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Download Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
pip install -r requirements.txt

echo [3/3] Creating .env file...
if not exist .env (
    copy .env.example .env
    echo.
    echo IMPORTANT: Edit the .env file with your API keys!
    echo   - ANTHROPIC_API_KEY: Your Anthropic API key
    echo   - XAI_API_KEY: Your xAI SuperGrok API key
    echo   - Keep SIMULATION_MODE=true until you're confident
    echo.
) else (
    echo .env already exists, skipping...
)

echo.
echo ==========================================
echo  Setup complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Edit .env with your API keys
echo   2. Run: python main.py --once    (single test cycle)
echo   3. Run: python main.py           (continuous trading)
echo   4. Run: python main.py --dashboard  (view performance)
echo.
pause
