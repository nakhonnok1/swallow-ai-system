# 🚀 Swallow AI - Smart Bird Detection System

## � Project Overview
**Swallow AI** is an intelligent bird detection and monitoring system specifically designed for tracking swallow birds (นกนางแอ่น) with advanced object detection capabilities for security monitoring.

## ✨ Key Features

### 🪶 **Bird Detection & Monitoring**
- Real-time swallow bird counting (in/out)
- Advanced AI-powered detection with YOLO
- Live statistics and reporting
- Database logging and analytics

### �️ **Security & Intruder Detection**
- Advanced object detection for people, animals
- Motion detection fallback system
- Real-time alerts and notifications
- Smart filtering to reduce false positives

### 🤖 **Ultra Smart AI Agent**
- Intelligent chatbot with natural language processing
- Real-time data integration
- Learning capabilities with conversation history
- Multi-language support (Thai/English)

### 🌐 **Web Interface**
- Modern responsive web dashboard
- Real-time video streaming
- Interactive chat interface
- System monitoring and health checks

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: YOLOv8, OpenCV, Computer Vision
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript
- **Streaming**: RTSP, WebRTC support
- **APIs**: RESTful API architecture

## 📁 Project Structure

```
swallow_ai/
├── app_working.py              # Main application server
├── ultra_smart_ai_agent.py     # AI chatbot system
├── ai_agent_web.py            # Web interface for AI agent
├── advanced_object_detector.py # Security detection system
├── swallow_ai_integration.py   # System integration controller
├── test_ai_agent.py           # Testing suite
├── models.py                  # Database models
├── schemas.py                 # Data schemas
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam or IP camera (RTSP support)
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd swallow_ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**
   - Update camera settings in configuration files
   - Set up environment variables for security
   - Configure detection zones as needed

4. **Run the system**
   ```bash
   # Option 1: Use integration controller
   python swallow_ai_integration.py
   
   # Option 2: Start components individually
   python app_working.py        # Main system (Port 5000)
   python ai_agent_web.py       # AI Agent (Port 8080)
   ```

### 🌐 **Access Points**
- **Main Dashboard**: http://localhost:5000
- **AI Agent Chat**: http://localhost:8080
- **API Documentation**: http://localhost:5000/api

## 🔧 Configuration

### Camera Setup
Configure your camera source in the application:

```python
# For IP Camera (RTSP)
VIDEO_SOURCE = "rtsp://username:password@ip:port/stream"

# For USB Webcam
VIDEO_SOURCE = 0

# For Video File
VIDEO_SOURCE = "path/to/video.mp4"
```

### Environment Variables
Create a `.env` file for sensitive configuration:
```env
CAMERA_USERNAME=your_username
CAMERA_PASSWORD=your_password
CAMERA_IP=your_camera_ip
```

## 📊 API Endpoints

### Bird Detection
- `GET /api/statistics` - Bird counting statistics
- `GET /api/current-count` - Current bird count

### Security Detection  
- `GET /api/object-detection/stats` - Security detection stats
- `GET /api/object-detection/alerts` - Recent security alerts

### AI Agent
- `POST /api/ai-agent/chat` - Chat with AI agent
- `GET /api/ai-agent/status` - AI agent status

### System Health
- `GET /api/system-health` - System performance metrics

## 🤖 AI Agent Usage

The Ultra Smart AI Agent can answer questions about:
- Bird counting and statistics
- Security alerts and detections
- System status and health
- General information about swallows

### Example Questions:
- "นกเข้ากี่ตัว?" (How many birds entered?)
- "มีสิ่งแปลกปลอมไหม?" (Any intruders detected?)
- "สถานะระบบ" (System status)
- "ช่วยเหลือ" (Help)

## 🧪 Testing

Run the test suite:
```bash
python test_ai_agent.py
```

## 🛡️ Security Features

- Sensitive data protection with .gitignore
- Environment variable configuration
- Secure API endpoints
- Input validation and sanitization

## 📈 Performance

- Real-time processing at 15-30 FPS
- Low latency detection (< 100ms)
- Optimized for resource efficiency
- Scalable architecture

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation
- Use the AI Agent chat interface
- Review API documentation
- Test with the provided test suite

## 🔄 Updates

- **v1.0.0** - Initial release with basic bird detection
- **v2.0.0** - Added security detection capabilities  
- **v3.0.0** - Integrated Ultra Smart AI Agent
- **v4.0.0** - Complete system integration and web interface

---

**Built with ❤️ for wildlife monitoring and security**
│   └── app_master.py           # แอป Master Edition (แนะนำ)
│
├── 🏗️ Architecture Files
│   ├── models.py               # SQLAlchemy Database Models
│   ├── schemas.py              # Pydantic Data Validation
│   ├── config.py               # Enhanced Configuration
│   └── validators.py           # Data Validators
│
├── 🛡️ System Enhancement Files
│   ├── error_handler.py        # ระบบจัดการ Error
│   ├── performance_monitor.py  # ตรวจสอบประสิทธิภาพ
│   ├── backup_system.py        # ระบบ Backup อัตโนมัติ
│   └── api_extensions.py       # API เพิ่มเติม
│
├── 🎨 Frontend Files
│   ├── templates/
│   │   └── index.html          # Dashboard UI
│   └── static/                 # Static files (ถ้ามี)
│
├── 📄 Documentation
│   ├── README.md               # คู่มือการใช้งาน
│   └── requirements.txt        # Dependencies
│
└── 🗃️ Data & Logs
    ├── anomaly_images/         # ภาพ Anomaly
    ├── backups/               # ไฟล์ Backup
    ├── db.sqlite              # ฐานข้อมูลหลัก
    ├── app.log                # Log ระบบ
    └── yolov8n.pt             # AI Model
```

### 🔧 การตั้งค่าขั้นสูง

#### เปลี่ยนแหล่งที่มาของวิดีโอ
แก้ไขในไฟล์ `config.py`:
```python
# ใช้เว็บแคม
VIDEO_SOURCE = 0

# ใช้ไฟล์วิดีโอ
VIDEO_SOURCE = "path/to/video.mp4"

# ใช้ RTSP Stream
VIDEO_SOURCE = "rtsp://username:password@ip:port/stream"
```

#### การตั้งค่า Environment Variables
```bash
# Database
export SQLITE_DATABASE_URI="sqlite:///custom_path.db"

# AI Model Settings
export DETECTION_THRESHOLD=0.7
export BIRD_CONFIDENCE_THRESHOLD=0.8

# Backup Settings
export AUTO_BACKUP_ENABLED=true
export BACKUP_INTERVAL_HOURS=12

# Performance
export ANOMALY_COOLDOWN=15
```

#### การตั้งค่าผ่าน JSON File
สร้างไฟล์ `config.json`:
```json
{
  "DETECTION_THRESHOLD": 0.6,
  "BIRD_CONFIDENCE_THRESHOLD": 0.7,
  "COUNTING_LINE_Y": 350,
  "AUTO_BACKUP_ENABLED": true,
  "BACKUP_RETENTION_DAYS": 14
}
```

### 📊 API Endpoints สมบูรณ์

#### 🎯 Core APIs
| Endpoint | Method | คำอธิบาย |
|----------|--------|----------|
| `/` | GET | Dashboard หลัก |
| `/video_feed` | GET | Video streaming |
| `/stats` | GET | สถิติปัจจุบัน |
| `/health` | GET | Health check พื้นฐาน |

#### 📈 Analytics APIs
| Endpoint | Method | คำอธิบาย |
|----------|--------|----------|
| `/api/analysis` | GET | สถิติรายวัน |
| `/api/statistics/advanced` | GET | สถิติขั้นสูง |
| `/api/anomalies` | GET | รายการ Anomaly |
| `/api/anomalies/<id>/view` | POST | ทำเครื่องหมายดู |

#### 🔧 Management APIs
| Endpoint | Method | คำอธิบาย |
|----------|--------|----------|
| `/system` | GET | System Dashboard |
| `/health/detailed` | GET | Health check ละเอียด |
| `/api/system/performance` | GET | สถิติประสิทธิภาพ |
| `/api/system/backup` | POST | สร้าง Backup |
| `/api/system/logs` | GET | ดู Log ล่าสุด |
| `/api/config` | GET/PUT | จัดการการตั้งค่า |
| `/api/system/reset_stats` | POST | รีเซ็ตสถิติ |

### 🛡️ ฟีเจอร์ระบบป้องกันและความปลอดภัย

#### 1. **Error Handling System**
- บันทึก Error ทุกประเภทพร้อม Context
- Graceful degradation เมื่อเกิดปัญหา
- Auto-recovery mechanisms

#### 2. **Performance Monitoring**
- ตรวจสอบ CPU/Memory usage
- วัดอัตรา Frame processing
- แจ้งเตือนเมื่อประสิทธิภาพต่ำ

#### 3. **Auto Backup System**
- Backup ฐานข้อมูลอัตโนมัติ
- Export สถิติเป็น JSON
- ลบ Backup เก่าอัตโนมัติ

#### 4. **Configuration Management**
- โหลดการตั้งค่าจาก Environment Variables
- รองรับการตั้งค่าผ่าน JSON file
- ตรวจสอบความถูกต้องของการตั้งค่า

### 🐛 การแก้ไขปัญหา

#### ปัญหา: ไม่สามารถเชื่อมต่อกล้องได้
```bash
# ตรวจสอบ video source
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Failed')"
```

#### ปัญหา: Dependencies ขาดหาย
```bash
# ติดตั้งใหม่ทั้งหมด
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

#### ปัญหา: Database Error
```bash
# สำรองและรีเซ็ต Database
python -c "from backup_system import backup_system; backup_system.backup_database()"
rm db.sqlite  # ลบ database เก่า (จะสร้างใหม่อัตโนมัติ)
```

#### ปัญหา: Performance ต่ำ
- ลดค่า `DETECTION_THRESHOLD` 
- เพิ่มค่า `FRAME_PROCESSING_DELAY`
- ตรวจสอบ `/api/system/performance`

### � การใช้งานขั้นสูง

#### 1. **Real-time Monitoring**
```javascript
// JavaScript สำหรับ real-time updates
setInterval(async () => {
    const stats = await fetch('/stats').then(r => r.json());
    const performance = await fetch('/api/system/performance').then(r => r.json());
    // อัปเดต UI
}, 1000);
```

#### 2. **Custom API Integration**
```python
# Python client สำหรับ API
import requests

def get_bird_stats():
    response = requests.get('http://localhost:5000/stats')
    return response.json()

def create_backup():
    response = requests.post('http://localhost:5000/api/system/backup')
    return response.json()
```

#### 3. **Webhook Integration**
```python
# เพิ่มใน app_master.py สำหรับส่ง webhook
import requests

def send_anomaly_webhook(anomaly_data):
    webhook_url = "https://your-webhook-url.com"
    requests.post(webhook_url, json=anomaly_data)
```

### 🚀 การพัฒนาต่อยอด

#### ฟีเจอร์ที่แนะนำให้เพิ่ม:
1. **📱 Mobile App Integration**
2. **☁️ Cloud Storage Backup**
3. **📧 Email/LINE Notifications**
4. **🤖 Advanced AI Models**
5. **📊 Advanced Analytics Dashboard**
6. **🔐 User Authentication System**
7. **🌐 Multi-camera Support**
8. **📝 Report Generation**

### 📞 การสนับสนุน

#### Log Files สำหรับ Debug:
- `app.log` - Application logs
- `error.log` - Error logs  
- `performance.log` - Performance logs

#### Health Check URLs:
- `/health` - พื้นฐาน
- `/health/detailed` - ละเอียด
- `/api/system/performance` - ประสิทธิภาพ

#### Commands สำหรับ Maintenance:
```bash
# สำรองข้อมูล
python -c "from backup_system import backup_system; backup_system.auto_backup()"

# ตรวจสอบการตั้งค่า
python -c "from config import Config; Config.validate_config()"

# ดู Performance Stats
curl http://localhost:5000/api/system/performance | python -m json.tool
```

---
**🏆 AI Bird Tracking System - Master Edition**  
**เวอร์ชัน:** 3.0 Master  
**อัปเดตล่าสุด:** สิงหาคม 2025  
**สถานะ:** ✅ Production Ready
