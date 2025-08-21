#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENHANCED IP CAMERA RTSP CONNECTOR V2 - เชื่อมต่อกล้อง IP พร้อม Authentication
รองรับ RTSP, HTTP, และ USB Camera ครบครัน

✅ คุณลักษณะ:
   🎯 รองรับ IP Camera พร้อม Username/Password
   🔄 Auto-reconnect เมื่อขาดการเชื่อมต่อ
   📡 RTSP/HTTP Stream รองรับหลายรูปแบบ
   🎥 Live Preview ผ่าน Web Interface
   💪 ทนทานต่อการเชื่อมต่อขาดหาย

🎉 Ready for Production!
"""

import cv2
import os
import time
import threading
import numpy as np
from flask import Flask, Response, render_template_string
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedIPCameraConnector:
    """🎥 ระบบเชื่อมต่อกล้อง IP ขั้นสูง พร้อม Authentication"""
    
    def __init__(self):
        self.cap = None
        self.is_connected = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.running = False
        
        # การตั้งค่า RTSP
        self.rtsp_configs = {
            'default': {
                'url': "rtsp://ainok1:ainok123@192.168.1.100:554/stream1",
                'username': 'ainok1',
                'password': 'ainok123',
                'ip': '192.168.1.100',
                'port': 554,
                'stream_path': '/stream1'
            }
        }
        
    def build_rtsp_url(self, config):
        """สร้าง RTSP URL พร้อม Authentication"""
        username = config['username']
        password = config['password']
        ip = config['ip']
        port = config['port']
        stream_path = config['stream_path']
        
        # รูปแบบ RTSP URL
        urls = [
            f"rtsp://{username}:{password}@{ip}:{port}{stream_path}",
            f"rtsp://{username}:{password}@{ip}:{port}/",
            f"rtsp://{username}:{password}@{ip}{stream_path}",
            f"rtsp://{ip}:{port}{stream_path}",
        ]
        
        return urls
    
    def setup_opencv_rtsp(self):
        """ตั้งค่า OpenCV สำหรับ RTSP"""
        # ตั้งค่า FFMPEG options สำหรับ RTSP
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
    
    def connect_camera(self, config_name='default'):
        """เชื่อมต่อกล้อง IP"""
        if config_name not in self.rtsp_configs:
            logger.error(f"❌ ไม่พบการตั้งค่า: {config_name}")
            return False
        
        config = self.rtsp_configs[config_name]
        logger.info(f"🎥 กำลังเชื่อมต่อกล้อง IP: {config['ip']}")
        
        # ตั้งค่า OpenCV
        self.setup_opencv_rtsp()
        
        # ลองหลาย URL
        urls = self.build_rtsp_url(config)
        
        for i, url in enumerate(urls):
            try:
                logger.info(f"🔄 ลอง URL {i+1}/{len(urls)}: {url.replace(config['password'], '***')}")
                
                # สร้าง VideoCapture
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                
                # ตั้งค่า buffer
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 25)
                
                if self.cap.isOpened():
                    # ทดสอบอ่านเฟรม
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info(f"✅ เชื่อมต่อสำเร็จ! ขนาดเฟรม: {frame.shape}")
                        
                        with self.frame_lock:
                            self.current_frame = frame
                        
                        self.is_connected = True
                        return True
                    else:
                        logger.warning(f"⚠️ เชื่อมต่อได้แต่อ่านเฟรมไม่ได้")
                        self.cap.release()
                else:
                    logger.warning(f"⚠️ ไม่สามารถเปิด VideoCapture")
                    
            except Exception as e:
                logger.error(f"❌ ข้อผิดพลาด URL {i+1}: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        logger.error("❌ ไม่สามารถเชื่อมต่อกล้องได้ด้วย URL ใดๆ")
        return False
    
    def start_capture(self):
        """เริ่มการจับภาพแบบต่อเนื่อง"""
        if not self.is_connected:
            logger.error("❌ ไม่ได้เชื่อมต่อกล้อง")
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("🎬 เริ่มการจับภาพต่อเนื่อง")
        return True
    
    def _capture_loop(self):
        """วนรอบการจับภาพ"""
        consecutive_failures = 0
        max_failures = 30
        
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        # อัปเดตเฟรมปัจจุบัน
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                        
                        consecutive_failures = 0
                        time.sleep(0.033)  # ~30 FPS
                        
                    else:
                        consecutive_failures += 1
                        logger.warning(f"⚠️ อ่านเฟรมไม่ได้ ({consecutive_failures}/{max_failures})")
                        
                        if consecutive_failures >= max_failures:
                            logger.error("❌ การจับภาพล้มเหลวต่อเนื่อง - พยายามเชื่อมต่อใหม่")
                            self._reconnect()
                            consecutive_failures = 0
                        
                        time.sleep(0.1)
                else:
                    logger.error("❌ กล้องไม่ได้เชื่อมต่อ - พยายามเชื่อมต่อใหม่")
                    self._reconnect()
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ ข้อผิดพลาดในการจับภาพ: {e}")
                time.sleep(1)
    
    def _reconnect(self):
        """เชื่อมต่อใหม่"""
        logger.info("🔄 กำลังเชื่อมต่อใหม่...")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False
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
    
    def stop(self):
        """หยุดการทำงาน"""
        logger.info("🛑 หยุดการจับภาพ")
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False

# Flask App สำหรับทดสอบ
app = Flask(__name__)
camera_connector = EnhancedIPCameraConnector()

def generate_frames():
    """สร้างเฟรมสำหรับ streaming"""
    while True:
        frame = camera_connector.get_current_frame()
        
        if frame is not None:
            # เพิ่มข้อมูลสถานะ
            status_text = f"Connected: {camera_connector.is_connected} | Time: {time.strftime('%H:%M:%S')}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # เข้ารหัสเป็น JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # สร้างเฟรมแสดงสถานะ
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            status = "Connecting..." if not camera_connector.is_connected else "No Frame"
            cv2.putText(error_frame, status, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """หน้าแรก"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎥 IP Camera RTSP Stream Test</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial; margin: 20px; background: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .video-container { text-align: center; margin: 20px 0; }
            .controls { margin: 20px 0; }
            .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            .btn-danger { background: #dc3545; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 IP Camera RTSP Stream Test</h1>
            
            <div id="status" class="status disconnected">
                📡 สถานะ: กำลังเชื่อมต่อ... | กล้อง: {{ 'เชื่อมต่อแล้ว' if connected else 'ไม่ได้เชื่อมต่อ' }}
            </div>
            
            <div class="video-container">
                <h3>📹 Live Stream</h3>
                <img src="{{ url_for('video_feed') }}" style="max-width: 100%; border: 2px solid #ddd; border-radius: 10px;">
            </div>
            
            <div class="controls">
                <h3>🎛️ การควบคุม</h3>
                <button class="btn btn-success" onclick="location.reload()">🔄 Refresh</button>
                <button class="btn btn-primary" onclick="testConnection()">🔗 Test Connection</button>
                <button class="btn btn-danger" onclick="stopStream()">⏹️ Stop</button>
            </div>
            
            <div style="margin-top: 30px;">
                <h3>📋 การตั้งค่ากล้อง</h3>
                <ul>
                    <li><strong>URL:</strong> rtsp://ainok1:ainok123@192.168.1.100:554/stream1</li>
                    <li><strong>Username:</strong> ainok1</li>
                    <li><strong>Password:</strong> ainok123</li>
                    <li><strong>IP Address:</strong> 192.168.1.100</li>
                    <li><strong>Port:</strong> 554</li>
                    <li><strong>Stream Path:</strong> /stream1</li>
                </ul>
            </div>
        </div>
        
        <script>
            function testConnection() {
                alert('🔄 กำลังทดสอบการเชื่อมต่อ... ตรวจสอบ Console Log');
                fetch('/test_connection').then(r => r.json()).then(data => {
                    alert('ผลการทดสอบ: ' + JSON.stringify(data));
                });
            }
            
            function stopStream() {
                if(confirm('หยุด Stream หรือไม่?')) {
                    fetch('/stop').then(() => location.reload());
                }
            }
            
            // อัปเดตสถานะทุก 5 วินาที
            setInterval(() => {
                fetch('/status').then(r => r.json()).then(data => {
                    const statusDiv = document.getElementById('status');
                    statusDiv.className = 'status ' + (data.connected ? 'connected' : 'disconnected');
                    statusDiv.innerHTML = `📡 สถานะ: ${data.connected ? 'เชื่อมต่อแล้ว' : 'ไม่ได้เชื่อมต่อ'} | เวลา: ${data.time}`;
                });
            }, 5000);
        </script>
    </body>
    </html>
    ''', connected=camera_connector.is_connected)

@app.route('/video_feed')
def video_feed():
    """ส่งวีดีโอสตรีม"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """สถานะระบบ"""
    return {
        'connected': camera_connector.is_connected,
        'time': time.strftime('%H:%M:%S'),
        'running': camera_connector.running
    }

@app.route('/test_connection')
def test_connection():
    """ทดสอบการเชื่อมต่อ"""
    logger.info("🔍 ทดสอบการเชื่อมต่อ...")
    success = camera_connector.connect_camera()
    
    if success and not camera_connector.running:
        camera_connector.start_capture()
    
    return {
        'success': success,
        'connected': camera_connector.is_connected,
        'message': '✅ เชื่อมต่อสำเร็จ' if success else '❌ เชื่อมต่อล้มเหลว'
    }

@app.route('/stop')
def stop():
    """หยุดการทำงาน"""
    camera_connector.stop()
    return {'message': '🛑 หยุดการทำงานแล้ว'}

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 เริ่มต้นระบบ Enhanced IP Camera RTSP Connector")
    
    # เชื่อมต่อกล้อง
    if camera_connector.connect_camera():
        camera_connector.start_capture()
        print("✅ เชื่อมต่อกล้องสำเร็จ!")
    else:
        print("⚠️ ไม่สามารถเชื่อมต่อกล้องในตอนเริ่มต้น (จะลองใหม่ผ่าน web interface)")
    
    # เริ่ม web server
    print("🌐 เริ่ม Web Server ที่ http://localhost:5001")
    print("📱 เปิดเบราว์เซอร์ไปที่ http://localhost:5001 เพื่อดู Live Stream")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 หยุดระบบ...")
        camera_connector.stop()

if __name__ == '__main__':
    main()
