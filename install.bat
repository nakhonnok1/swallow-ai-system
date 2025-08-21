@echo off
echo ================================================
echo  AI Bird Tracking System - Installation
echo ================================================
echo.

echo 🔍 Setting up Python environment...
cd /d "c:\Nakhonnok\swallow_ai"

echo 📦 Creating virtual environment...
python -m venv venv

echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

echo 📦 Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Installation completed successfully!
echo.
echo 🚀 To start the system, run: start.bat
echo 📖 Read README.md for detailed instructions
echo.
pause
