@echo off
echo 🔴 เริ่มต้นระบบ AI ตรวจจับนกแอ่น 24 ชั่วโมง
echo ================================================

:: ตรวจสอบ Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ไม่พบ Python กรุณาติดตั้ง Python ก่อน
    pause
    exit /b 1
)

:: ติดตั้งแพ็คเกจที่ต้องการ
echo 📦 ติดตั้งแพ็คเกจที่จำเป็น...
pip install opencv-python numpy psutil matplotlib pandas requests --quiet

:: สร้างโฟลเดอร์ log
if not exist "logs" mkdir logs

:: เริ่มต้นระบบ
echo 🚀 เริ่มต้นระบบ...
echo.
echo เลือกโหมดการทำงาน:
echo 1. สตรีม 24 ชั่วโมง (ใช้งานจริง)
echo 2. ทดสอบกับวิดีโอ
echo 3. ติดตามสถิติอย่างเดียว
echo.

set /p choice="เลือก (1-3): "

if "%choice%"=="1" (
    echo 🔴 เริ่มสตรีม 24 ชั่วโมง...
    python live_stream_ai.py
) else if "%choice%"=="2" (
    echo 🎬 ทดสอบกับวิดีโอ...
    python quick_test.py
) else if "%choice%"=="3" (
    echo 📊 เริ่มติดตามสถิติ...
    python stream_monitor.py
) else (
    echo ❌ ตัวเลือกไม่ถูกต้อง
    pause
    exit /b 1
)

pause
