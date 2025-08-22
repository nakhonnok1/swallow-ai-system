@echo off
echo ================================================
echo  AI Bird Tracking System - Quick Start
echo ================================================
echo.

echo 🔍 Checking Python environment...
cd /d "c:\Nakhonnok\swallow_ai"

if not exist "venv\" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

echo ✅ Virtual environment found

echo.
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 🚀 Starting AI Bird Tracking System...
echo 📊 Dashboard will be available at: http://localhost:5000
echo 🔧 System Management: http://localhost:5000/system
echo ❤️  Health Check: http://localhost:5000/health/detailed
echo.
echo 💡 Press Ctrl+C to stop the system
echo.

python app_working.py

echo.
echo 👋 System stopped. Press any key to exit...
pause > nul
