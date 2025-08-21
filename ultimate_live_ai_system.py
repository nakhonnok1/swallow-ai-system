#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ULTIMATE LIVE STREAM AI SYSTEM V5 - เชื่อมต่อกล้อง + AI + Web App ครบครัน
รวม IP Camera + AI Detection + Live Stream + Web Dashboard

✅ คุณลักษณะ:
   🎥 เชื่อมต่อกล้อง ainok IP Camera (192.168.1.100)
   🤖 ใช้ AI หลัก V5 Ultra Precision สำหรับตรวจจับนก
   🌐 Web Dashboard แบบ Real-time 
   📊 สถิติการตรวจจับแบบสด
   🔄 Auto-reconnect เมื่อขาดการเชื่อมต่อ

🎉 Ready for Production!
"""

import cv2
import os
import time
import json
import threading
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
from collections import deque
import logging

# Import ระบบ AI หลัก
try:
    from ultimate_perfect_ai_MASTER import V5_UltimatePrecisionSwallowAI, EnhancedMasterBirdDetector
    AI_AVAILABLE = True
    print("✅ โหลด AI หลัก V5 สำเร็จ")
except Exception as e:
    AI_AVAILABLE = False
    print(f"⚠️ ไม่สามารถโหลด AI หลัก: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateLiveStreamAI:
    """🚀 ระบบ Live Stream AI แบบครบวงจร"""
    
    def __init__(self):
        print("🚀 เริ่มต้นระบบ Ultimate Live Stream AI V5")
        
        # การตั้งค่ากล้อง ainok
        self.camera_config = {
            'username': 'ainok1',
            'password': 'ainok123', 
            'ip': '192.168.1.100',
            'port': 554,
            'stream_path': '/stream1'
        }
        
        # ระบบกล้อง
        self.cap = None
        self.is_connected = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.running = False
        
        # ระบบ AI
        self.ai_system = None
        self.detector = None
        self._initialize_ai()
        
        # สถิติการตรวจจับ
        self.detection_stats = {
            'total_detections': 0,
            'entering_birds': 0,
            'exiting_birds': 0,
            'session_start': time.time(),
            'last_detection': None,
            'fps': 0,
            'frame_count': 0
        }
        
        # ผลลัพธ์ล่าสุด
        self.recent_results = deque(maxlen=100)
        
        print("✅ Ultimate Live Stream AI พร้อมใช้งาน!")
    
    def _initialize_ai(self):
        """เริ่มต้นระบบ AI"""
        try:
            if AI_AVAILABLE:
                self.ai_system = V5_UltimatePrecisionSwallowAI('mixed')
                self.detector = EnhancedMasterBirdDetector('mixed')
                print("✅ เริ่มต้นระบบ AI หลัก V5 สำเร็จ")
            else:
                print("⚠️ ใช้ระบบตรวจจับแบบพื้นฐาน")
                self.detector = None
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่มต้นระบบ AI: {e}")
            self.detector = None
    
    def build_rtsp_url(self):
        """สร้าง RTSP URL"""
        config = self.camera_config
        return f"rtsp://{config['username']}:{config['password']}@{config['ip']}:{config['port']}{config['stream_path']}"
    
    def setup_opencv_rtsp(self):
        """ตั้งค่า OpenCV สำหรับ RTSP"""
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp;'
            'timeout;10000000;'
            'reconnect;1;'
            'reconnect_at_eof;1;'
            'reconnect_streamed;1;'
            'reconnect_delay_max;2;'
            'fflags;nobuffer;'
            'flags;low_delay;'
            'strict;experimental'
        )
    
    def connect_camera(self):
        """เชื่อมต่อกล้อง"""
        try:
            print("🎥 กำลังเชื่อมต่อกล้อง ainok...")
            
            # ปิดกล้องเก่า
            if self.cap:
                self.cap.release()
            
            # ตั้งค่า OpenCV
            self.setup_opencv_rtsp()
            
            # เชื่อมต่อ
            rtsp_url = self.build_rtsp_url()
            logger.info(f"🔗 RTSP URL: {rtsp_url.replace(self.camera_config['password'], '***')}")
            
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"✅ เชื่อมต่อสำเร็จ! ขนาดเฟรม: {frame.shape}")
                    
                    with self.frame_lock:
                        self.current_frame = frame
                    
                    self.is_connected = True
                    return True
            
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
            return False
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดการเชื่อมต่อ: {e}")
            return False
    
    def start_live_stream(self):
        """เริ่ม Live Stream"""
        if not self.connect_camera():
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.capture_thread.start()
        print("🎬 เริ่ม Live Stream สำเร็จ")
        return True
    
    def _stream_loop(self):
        """วนรอบ Live Stream"""
        consecutive_failures = 0
        max_failures = 30
        frame_start_time = time.time()
        
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        # นับ FPS
                        self.detection_stats['frame_count'] += 1
                        current_time = time.time()
                        elapsed = current_time - frame_start_time
                        if elapsed >= 1.0:
                            self.detection_stats['fps'] = self.detection_stats['frame_count'] / elapsed
                            self.detection_stats['frame_count'] = 0
                            frame_start_time = current_time
                        
                        # ตรวจจับด้วย AI
                        processed_frame = self._process_frame_with_ai(frame)
                        
                        # อัปเดตเฟรมปัจจุบัน
                        with self.frame_lock:
                            self.current_frame = processed_frame
                        
                        consecutive_failures = 0
                        time.sleep(0.033)  # ~30 FPS
                        
                    else:
                        consecutive_failures += 1
                        logger.warning(f"⚠️ อ่านเฟรมไม่ได้ ({consecutive_failures}/{max_failures})")
                        
                        if consecutive_failures >= max_failures:
                            logger.error("❌ การจับภาพล้มเหลว - เชื่อมต่อใหม่")
                            self._reconnect()
                            consecutive_failures = 0
                        
                        time.sleep(0.1)
                else:
                    logger.error("❌ กล้องไม่ได้เชื่อมต่อ")
                    self._reconnect()
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ ข้อผิดพลาดในการประมวลผล: {e}")
                time.sleep(1)
    
    def _process_frame_with_ai(self, frame):
        """ประมวลผลเฟรมด้วย AI"""
        try:
            if self.detector:
                # ตรวจจับด้วย AI หลัก V5
                detections = self.detector.detect_smart(frame)
                
                # วาดผลลัพธ์
                for detection in detections:
                    bbox = detection.get('bbox', (0, 0, 0, 0))
                    confidence = detection.get('confidence', 0)
                    center = detection.get('center', (0, 0))
                    
                    # วาด bounding box
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # วาดจุดกลาง
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    
                    # แสดง confidence
                    cv2.putText(frame, f"{confidence:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # อัปเดตสถิติ
                self.detection_stats['total_detections'] += len(detections)
                if len(detections) > 0:
                    self.detection_stats['last_detection'] = time.time()
                
                # เก็บผลลัพธ์
                result = {
                    'timestamp': time.time(),
                    'detections': len(detections),
                    'frame_shape': frame.shape
                }
                self.recent_results.append(result)
            
            # เพิ่มข้อมูลสถานะ
            self._add_status_overlay(frame)
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการประมวลผล AI: {e}")
            return frame
    
    def _add_status_overlay(self, frame):
        """เพิ่มข้อมูลสถานะบนเฟรม"""
        # ข้อมูลสถานะ
        status_lines = [
            f"🎥 ainok Camera: {'Connected' if self.is_connected else 'Disconnected'}",
            f"🤖 AI: {'V5 Ultra' if self.detector else 'Basic'}",
            f"📊 Detections: {self.detection_stats['total_detections']}",
            f"⚡ FPS: {self.detection_stats['fps']:.1f}",
            f"⏰ {time.strftime('%H:%M:%S')}"
        ]
        
        # วาดพื้นหลัง
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # วาดข้อความ
        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _reconnect(self):
        """เชื่อมต่อใหม่"""
        logger.info("🔄 กำลังเชื่อมต่อใหม่...")
        self.is_connected = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        time.sleep(2)
        
        if self.connect_camera():
            logger.info("✅ เชื่อมต่อใหม่สำเร็จ")
        else:
            logger.error("❌ เชื่อมต่อใหม่ล้มเหลว")
    
    def get_current_frame(self):
        """รับเฟรมปัจจุบัน"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_stats(self):
        """รับสถิติ"""
        return self.detection_stats.copy()
    
    def stop(self):
        """หยุดระบบ"""
        print("🛑 หยุดระบบ...")
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False

# สร้างระบบหลัก
live_ai = UltimateLiveStreamAI()

# Flask App
app = Flask(__name__)

def generate_frames():
    """สร้างเฟรมสำหรับ streaming"""
    while True:
        frame = live_ai.get_current_frame()
        
        if frame is not None:
            # เข้ารหัสเป็น JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # สร้างเฟรมแสดงสถานะ
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            status = "Connecting to ainok camera..." if not live_ai.is_connected else "No Frame"
            cv2.putText(error_frame, status, (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def dashboard():
    """หน้าแดชบอร์ด"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Ultimate Live Stream AI - ainok Camera</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                min-height: 100vh;
            }
            .container { 
                max-width: 1400px; 
                margin: 0 auto; 
                padding: 20px; 
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: rgba(255,255,255,0.15);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.2);
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #00ff88;
                display: block;
            }
            .video-section {
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 30px;
                align-items: start;
            }
            .video-container {
                background: rgba(0,0,0,0.3);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
            }
            .video-stream {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            .controls {
                min-width: 250px;
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .btn {
                width: 100%;
                padding: 12px;
                margin: 8px 0;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .btn-primary { 
                background: linear-gradient(45deg, #00b4db, #0083b0);
                color: white;
            }
            .btn-success { 
                background: linear-gradient(45deg, #56ab2f, #a8e6cf);
                color: white;
            }
            .btn-danger { 
                background: linear-gradient(45deg, #ff416c, #ff4b2b);
                color: white;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .connected { background: #00ff88; }
            .disconnected { background: #ff4444; }
            
            @media (max-width: 768px) {
                .video-section {
                    grid-template-columns: 1fr;
                }
                .controls {
                    min-width: auto;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Ultimate Live Stream AI V5</h1>
                <h3>🎥 ainok IP Camera + AI Detection System</h3>
                <p>Real-time Bird Detection with V5 Ultra Precision AI</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-number" id="totalDetections">{{ stats.total_detections }}</span>
                    <div>Total Detections</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number" id="enteringBirds">{{ stats.entering_birds }}</span>
                    <div>Entering Birds</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number" id="exitingBirds">{{ stats.exiting_birds }}</span>
                    <div>Exiting Birds</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number" id="currentFps">{{ "%.1f"|format(stats.fps) }}</span>
                    <div>FPS</div>
                </div>
            </div>
            
            <div class="video-section">
                <div class="video-container">
                    <h3>📹 Live Stream</h3>
                    <img src="{{ url_for('video_feed') }}" class="video-stream" alt="Live Stream">
                </div>
                
                <div class="controls">
                    <h3>🎛️ System Control</h3>
                    
                    <div style="margin: 15px 0;">
                        <div id="cameraStatus">
                            <span class="status-indicator {{ 'connected' if connected else 'disconnected' }}"></span>
                            Camera: {{ 'Connected' if connected else 'Disconnected' }}
                        </div>
                        <div style="margin-top: 8px;">
                            <span class="status-indicator connected"></span>
                            AI: V5 Ultra Ready
                        </div>
                    </div>
                    
                    <button class="btn btn-success" onclick="refreshPage()">🔄 Refresh</button>
                    <button class="btn btn-primary" onclick="testConnection()">🔗 Test Camera</button>
                    <button class="btn btn-primary" onclick="viewStats()">📊 View Stats</button>
                    <button class="btn btn-danger" onclick="stopSystem()">⏹️ Stop System</button>
                    
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.3);">
                        <h4>📋 Camera Info</h4>
                        <small>
                            IP: 192.168.1.100<br>
                            Stream: /stream1<br>
                            Resolution: Full HD<br>
                            Protocol: RTSP
                        </small>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function refreshPage() {
                location.reload();
            }
            
            function testConnection() {
                fetch('/api/test_connection')
                    .then(r => r.json())
                    .then(data => {
                        alert('Connection Test: ' + (data.success ? '✅ Success' : '❌ Failed'));
                        if (data.success) location.reload();
                    })
                    .catch(e => alert('Error: ' + e));
            }
            
            function viewStats() {
                fetch('/api/stats')
                    .then(r => r.json())
                    .then(data => {
                        const statsText = Object.entries(data)
                            .map(([key, value]) => `${key}: ${value}`)
                            .join('\\n');
                        alert('📊 System Stats:\\n\\n' + statsText);
                    });
            }
            
            function stopSystem() {
                if(confirm('Stop the system?')) {
                    fetch('/api/stop').then(() => {
                        alert('🛑 System stopped');
                        location.reload();
                    });
                }
            }
            
            // อัปเดตสถิติทุก 3 วินาที
            setInterval(() => {
                fetch('/api/stats')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('totalDetections').textContent = data.total_detections;
                        document.getElementById('enteringBirds').textContent = data.entering_birds;
                        document.getElementById('exitingBirds').textContent = data.exiting_birds;
                        document.getElementById('currentFps').textContent = data.fps.toFixed(1);
                        
                        // อัปเดตสถานะกล้อง
                        const indicator = document.querySelector('#cameraStatus .status-indicator');
                        const text = document.querySelector('#cameraStatus');
                        if (data.connected) {
                            indicator.className = 'status-indicator connected';
                            text.innerHTML = '<span class="status-indicator connected"></span>Camera: Connected';
                        } else {
                            indicator.className = 'status-indicator disconnected';
                            text.innerHTML = '<span class="status-indicator disconnected"></span>Camera: Disconnected';
                        }
                    })
                    .catch(e => console.log('Stats update failed:', e));
            }, 3000);
        </script>
    </body>
    </html>
    ''', stats=live_ai.get_stats(), connected=live_ai.is_connected)

@app.route('/video_feed')
def video_feed():
    """ส่งวีดีโอสตรีม"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    """API สถิติ"""
    stats = live_ai.get_stats()
    stats['connected'] = live_ai.is_connected
    return jsonify(stats)

@app.route('/api/test_connection')
def api_test_connection():
    """ทดสอบการเชื่อมต่อ"""
    success = live_ai.connect_camera()
    if success and not live_ai.running:
        live_ai.start_live_stream()
    
    return jsonify({
        'success': success,
        'connected': live_ai.is_connected,
        'message': '✅ Connected' if success else '❌ Failed'
    })

@app.route('/api/stop')
def api_stop():
    """หยุดระบบ"""
    live_ai.stop()
    return jsonify({'message': '🛑 System stopped'})

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 เริ่มต้นระบบ Ultimate Live Stream AI V5")
    print("🎥 กำลังเชื่อมต่อกล้อง ainok...")
    
    # เริ่ม Live Stream
    if live_ai.start_live_stream():
        print("✅ เริ่ม Live Stream สำเร็จ")
    else:
        print("⚠️ ไม่สามารถเริ่ม Live Stream (จะลองใหม่ผ่าน web interface)")
    
    # เริ่ม web server
    print("🌐 เริ่ม Web Dashboard ที่ http://localhost:5000")
    print("📱 เปิดเบราว์เซอร์ไปที่ http://localhost:5000 เพื่อดู Live Stream + AI Detection")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 หยุดระบบ...")
        live_ai.stop()

if __name__ == '__main__':
    main()
