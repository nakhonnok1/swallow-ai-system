# 📹 คู่มือการตั้งค่า Video Source

## 🎯 วิธีการเปลี่ยน Video Source

เปิดไฟล์ `app_working.py` และแก้ไขบรรทัดที่ 175-178:

### 1. 📷 ใช้ IP Camera (กล้อง RTSP)
```python
VIDEO_SOURCE = "rtsp://ainok1:ainok123@192.168.1.102:554/stream1"  # IP Camera
# VIDEO_SOURCE = 0  # เว็บแคม
# VIDEO_SOURCE = "simulation"  # การจำลอง
```

### 2. 🎥 ใช้เว็บแคม (USB Camera)
```python
# VIDEO_SOURCE = "rtsp://ainok1:ainok123@192.168.1.102:554/stream1"  # IP Camera
VIDEO_SOURCE = 0  # เว็บแคม
# VIDEO_SOURCE = "simulation"  # การจำลอง
```

### 3. 🎭 ใช้การจำลอง (สำหรับทดสอบ)
```python
# VIDEO_SOURCE = "rtsp://ainok1:ainok123@192.168.1.102:554/stream1"  # IP Camera
# VIDEO_SOURCE = 0  # เว็บแคม
VIDEO_SOURCE = "simulation"  # การจำลอง
```

## 🔧 การตั้งค่า IP Camera

### ตัวอย่างการตั้งค่า RTSP URL:

1. **กล้อง Hikvision:**
   ```python
   VIDEO_SOURCE = "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
   ```

2. **กล้อง Dahua:**
   ```python
   VIDEO_SOURCE = "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
   ```

3. **กล้องทั่วไป:**
   ```python
   VIDEO_SOURCE = "rtsp://username:password@ip_address:port/stream_path"
   ```

## 🚀 การรีสตาร์ทระบบ

หลังจากแก้ไข VIDEO_SOURCE แล้ว:

1. **หยุดระบบ:** กด `Ctrl+C` ใน Terminal
2. **เริ่มใหม่:** รันคำสั่ง:
   ```bash
   cd C:\Nakhonnok\swallow_ai
   C:/Nakhonnok/swallow_ai/venv/Scripts/python.exe app_working.py
   ```

## 📊 การตรวจสอบสถานะ

เมื่อระบบเริ่มต้น จะแสดงสถานะดังนี้:

### ✅ เชื่อมต่อกล้องสำเร็จ:
```
🔍 ทดสอบการเชื่อมต่อกล้อง: rtsp://...
✅ เชื่อมต่อกล้องสำเร็จ!
📱 ระบบพร้อมใช้งาน:
   ✅ Live Video Feed (กล้องจริง)
   📹 Video Source: rtsp://...
```

### ❌ เชื่อมต่อกล้องไม่สำเร็จ:
```
🔍 ทดสอบการเชื่อมต่อกล้อง: 0
❌ ไม่สามารถเชื่อมต่อกล้องได้
📱 ระบบพร้อมใช้งาน:
   ✅ Live Video Feed (จำลองสมจริง)
   ⚠️ ไม่พบกล้องจริง - ใช้การจำลอง
```

## 🔍 การแก้ไขปัญหา

### ปัญหา: ไม่สามารถเชื่อมต่อ IP Camera
1. ตรวจสอบ IP Address และ Port
2. ตรวจสอบ username/password
3. ตรวจสอบการเชื่อมต่อเครือข่าย
4. ลองใช้ VLC Player ทดสอบ RTSP URL

### ปัญหา: ไม่พบเว็บแคม
1. ตรวจสอบการเชื่อมต่อ USB
2. ลองเปลี่ยนหมายเลข (0, 1, 2...)
3. ตรวจสอบ driver เว็บแคม

### ปัญหา: วีดีโอช้าหรือกระตุก
1. ตรวจสอบความเร็วเครือข่าย
2. ลดความละเอียดกล้อง
3. เปลี่ยน substream แทน mainstream

## 💡 เคล็ดลับ

1. **การใช้หลาย Video Source:** แก้ไขใน code เพื่อสลับอัตโนมัติ
2. **บันทึกวีดีโอ:** เพิ่ม feature การบันทึกในโค้ด
3. **ตรวจสอบคุณภาพ:** ดูสถิติ FPS และ resolution ใน log

---

📧 **ติดต่อสอบถาม:** หากมีปัญหาการตั้งค่า Video Source
