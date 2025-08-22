@echo off
title Swallow AI - Complete System Launcher
color 0A

echo.
echo ========================================================
echo    🚀 SWALLOW AI - COMPLETE SYSTEM LAUNCHER
echo ========================================================
echo.

cd /d "%~dp0"

echo 📁 Current Directory: %CD%
echo.

echo 🔍 Checking Python...
python --version
if errorlevel 1 (
    echo ❌ Python ไม่พบ กรุณาติดตั้ง Python
    pause
    exit /b 1
)

echo.
echo 🔍 Checking required files...

if not exist "app_working.py" (
    echo ❌ ไม่พบ app_working.py
    pause
    exit /b 1
)

if not exist "ultra_smart_ai_agent.py" (
    echo ❌ ไม่พบ ultra_smart_ai_agent.py
    pause
    exit /b 1
)

if not exist "ai_agent_web.py" (
    echo ❌ ไม่พบ ai_agent_web.py
    pause
    exit /b 1
)

echo ✅ All required files found!
echo.

echo 🚀 เลือกการเริ่มต้น:
echo 1. เริ่มระบบหลัก (Main System) - Port 5000
echo 2. เริ่ม AI Agent Web - Port 8080  
echo 3. เริ่มทั้งหมด (All Systems)
echo 4. ใช้ Integration Controller
echo.

set /p choice="เลือก (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 เริ่มต้นระบบหลัก...
    echo 🌐 เปิดเว็บ: http://127.0.0.1:5000
    echo ⌨️ กด Ctrl+C เพื่อหยุด
    echo.
    python app_working.py
) else if "%choice%"=="2" (
    echo.
    echo 🚀 เริ่มต้น AI Agent Web...
    echo 🌐 เปิดเว็บ: http://127.0.0.1:8080
    echo ⌨️ กด Ctrl+C เพื่อหยุด
    echo.
    python ai_agent_web.py
) else if "%choice%"=="3" (
    echo.
    echo 🚀 เริ่มต้นระบบทั้งหมด...
    echo.
    echo 📱 Main System: http://127.0.0.1:5000
    echo 💬 AI Agent: http://127.0.0.1:8080
    echo.
    echo 🔧 เริ่มต้นระบบหลักก่อน...
    start "Swallow AI - Main System" python app_working.py
    
    echo ⏳ รอ 5 วินาที...
    timeout /t 5 /nobreak > nul
    
    echo 🔧 เริ่มต้น AI Agent Web...
    start "Swallow AI - AI Agent Web" python ai_agent_web.py
    
    echo.
    echo ✅ ระบบทั้งหมดเริ่มต้นแล้ว!
    echo 📱 Main System: http://127.0.0.1:5000
    echo 💬 AI Agent: http://127.0.0.1:8080
    echo.
    echo กด Enter เพื่อปิดหน้าต่างนี้...
    pause > nul
) else if "%choice%"=="4" (
    echo.
    echo 🔗 เริ่มต้น Integration Controller...
    python swallow_ai_integration.py
) else (
    echo ❌ ตัวเลือกไม่ถูกต้อง
    pause
)

echo.
echo 👋 ขอบคุณที่ใช้ Swallow AI!
pause
