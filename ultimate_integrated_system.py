#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ULTIMATE INTEGRATED SWALLOW AI SYSTEM V5 - เชื่อมต่อครบทุกระบบ
รวม AI หลัก + Live Stream + Web App + กล้อง + Database ครบครัน

✅ คุณลักษณะเด่น:
   🎯 เชื่อมต่อกล้อง USB, IP Camera, Webcam
   🤖 ใช้ AI หลัก V5 Ultra Precision
   🌐 Web Dashboard แบบ Real-time
   📊 บันทึกฐานข้อมูลอัตโนมัติ
   🔄 Live Stream 24/7
   📱 API สำหรับแอพมือถือ

🎉 Ready for Production!
"""

import cv2
import numpy as np
import time
import json
import threading
import logging
import sqlite3
import os
import gc
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
from flask import Flask, jsonify, render_template, Response, request, render_template_string
from werkzeug.serving import make_server
import base64

# Import ระบบ AI หลัก
try:
    from ultimate_perfect_ai_MASTER import V5_UltimatePrecisionSwallowAI, EnhancedMasterBirdDetector
    AI_AVAILABLE = True
    print("✅ โหลด AI หลัก V5 สำเร็จ")
except Exception as e:
    AI_AVAILABLE = False
    print(f"⚠️ ไม่สามารถโหลด AI หลัก: {e}")

# Import ระบบ Live Stream
try:
    from live_stream_ai import MemoryEfficientDetector, LiveStreamManager
    LIVE_STREAM_AVAILABLE = True
    print("✅ โหลด Live Stream สำเร็จ")
except Exception as e:
    LIVE_STREAM_AVAILABLE = False
    print(f"⚠️ ไม่สามารถโหลด Live Stream: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_integrated_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimateIntegratedSystem:
    """🚀 ระบบรวมครบทุกอย่างสำหรับตรวจจับนกแอ่น"""
    
    def __init__(self):
        print("🚀 เริ่มต้นระบบ Ultimate Integrated Swallow AI V5")
        
        # การตั้งค่าระบบ
        self.config = self._load_config()
        
        # ระบบ AI หลัก
        self.ai_system = None
        self.detector = None
        self._initialize_ai()
        
        # ระบบกล้อง
        self.camera = None
        self.camera_thread = None
        self.camera_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # ระบบบันทึกข้อมูล
        self.db_manager = DatabaseManager()
        
        # ระบบ Live Stream
        self.live_results = deque(maxlen=1000)
        self.detection_stats = {
            'total_detections': 0,
            'entering_birds': 0,
            'exiting_birds': 0,
            'session_start': datetime.now(),
            'last_detection': None
        }
        
        # Flask Web App
        self.app = Flask(__name__)
        self.setup_routes()
        
        print("✅ Ultimate Integrated System พร้อมใช้งาน!")
    
    def _load_config(self):
        """โหลดการตั้งค่า"""
        default_config = {
            'camera': {
                'source': 'rtsp://ainok1:ainok123@192.168.1.100:554/stream1',  # IP Camera RTSP
                'backup_source': 0,  # 0 = webcam สำรอง
                'width': 640,
                'height': 480,
                'fps': 15,
                'rtsp_transport': 'tcp',  # tcp หรือ udp
                'buffer_size': 1,
                'timeout': 10
            },
            'ai': {
                'mode': 'mixed',  # enter, exit, mixed
                'confidence_threshold': 0.25,
                'use_yolo': True,
                'enable_motion_detection': True
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            },
            'database': {
                'path': 'ultimate_integrated_results.db',
                'backup_interval': 3600
            }
        }
        
        try:
            if os.path.exists('integrated_config.json'):
                with open('integrated_config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print("✅ โหลดการตั้งค่าจากไฟล์")
                return {**default_config, **config}
        except Exception as e:
            print(f"⚠️ ใช้การตั้งค่าเริ่มต้น: {e}")
        
        return default_config
    
    def _initialize_ai(self):
        """เริ่มต้นระบบ AI"""
        try:
            if AI_AVAILABLE:
                self.ai_system = V5_UltimatePrecisionSwallowAI(self.config['ai']['mode'])
                self.detector = EnhancedMasterBirdDetector(self.config['ai']['mode'])
                print("✅ เริ่มต้นระบบ AI หลัก V5 สำเร็จ")
            else:
                # ใช้ระบบ backup
                if LIVE_STREAM_AVAILABLE:
                    self.detector = MemoryEfficientDetector()
                    print("✅ ใช้ระบบ AI สำรอง")
                else:
                    self.detector = None
                    print("⚠️ ไม่มีระบบ AI ที่ใช้ได้")
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่มต้นระบบ AI: {e}")
            self.detector = None
    
    def connect_camera(self, source=None):
        """เชื่อมต่อกล้อง - รองรับ IP Camera, RTSP, และ USB Camera"""
        if source is None:
            source = self.config['camera']['source']
        
        try:
            print(f"🎥 กำลังเชื่อมต่อกล้อง: {source}")
            
            # ปิดกล้องเก่า (ถ้ามี)
            if self.camera is not None:
                self.camera.release()
            
            # เชื่อมต่อกล้องใหม่
            self.camera = cv2.VideoCapture(source)
            
            # ตั้งค่าพิเศษสำหรับ RTSP
            if isinstance(source, str) and source.startswith('rtsp://'):
                print("🔗 ตั้งค่าสำหรับ RTSP Stream...")
                
                # ตั้งค่า buffer และ transport protocol
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera'].get('buffer_size', 1))
                
                # ตั้งค่า timeout
                self.camera.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config['camera'].get('timeout', 10) * 1000)
                
                # ตั้งค่า read timeout
                self.camera.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                
                print("⏳ รอการเชื่อมต่อ RTSP...")
                time.sleep(2)  # รอให้ RTSP เชื่อมต่อ
            
            if not self.camera.isOpened():
                # ลองใช้ backup source
                backup_source = self.config['camera'].get('backup_source', 0)
                print(f"🔄 ลองเชื่อมต่อ backup camera: {backup_source}")
                
                if self.camera is not None:
                    self.camera.release()
                
                self.camera = cv2.VideoCapture(backup_source)
                
                if not self.camera.isOpened():
                    raise Exception(f"ไม่สามารถเปิดกล้องหลักและสำรอง: {source}, {backup_source}")
                else:
                    print("✅ เชื่อมต่อ backup camera สำเร็จ")
            
            # ตั้งค่ากล้อง
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            self.camera.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            
            # ทดสอบอ่านเฟรมหลายครั้ง
            print("🔍 ทดสอบการอ่านเฟรม...")
            test_attempts = 5
            frame_read_success = False
            
            for attempt in range(test_attempts):
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    frame_read_success = True
                    print(f"✅ อ่านเฟรมสำเร็จในครั้งที่ {attempt + 1} - ขนาด: {frame.shape}")
                    break
                else:
                    print(f"⚠️ ครั้งที่ {attempt + 1}: ไม่สามารถอ่านเฟรม")
                    time.sleep(1)
            
            if not frame_read_success:
                raise Exception("ไม่สามารถอ่านเฟรมจากกล้องหลังจากทดสอบหลายครั้ง")
            
            print(f"✅ เชื่อมต่อกล้องสำเร็จ - ขนาด: {frame.shape}")
            return True
            
        except Exception as e:
            print(f"❌ ไม่สามารถเชื่อมต่อกล้อง: {e}")
            
            # แสดงข้อมูลเพิ่มเติมสำหรับการ debug
            print("🔧 ข้อมูลการ Debug:")
            print(f"   Source: {source}")
            print(f"   OpenCV Version: {cv2.__version__}")
            print(f"   Camera Object: {self.camera}")
            
            return False
    
    def start_camera_stream(self):
        """เริ่ม Live Stream จากกล้อง"""
        if self.camera_running:
            print("⚠️ กล้องทำงานอยู่แล้ว")
            return
        
        if not self.connect_camera():
            print("❌ ไม่สามารถเชื่อมต่อกล้อง")
            return
        
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        
        print("🎥 เริ่ม Live Stream จากกล้อง")
    
    def _camera_loop(self):
        """Loop การประมวลผลกล้อง - ปรับปรุงสำหรับ RTSP"""
        frame_count = 0
        consecutive_failures = 0
        max_failures = 10
        
        while self.camera_running and self.camera is not None:
            try:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"⚠️ ไม่สามารถอ่านเฟรมจากกล้อง (ครั้งที่ {consecutive_failures})")
                    
                    if consecutive_failures >= max_failures:
                        print("❌ การอ่านเฟรมล้มเหลวต่อเนื่อง - พยายามเชื่อมต่อใหม่")
                        if self.connect_camera():
                            consecutive_failures = 0
                            continue
                        else:
                            print("❌ ไม่สามารถเชื่อมต่อกล้องใหม่ได้")
                            break
                    
                    time.sleep(0.5)
                    continue
                
                # รีเซ็ตตัวนับความล้มเหลว
                consecutive_failures = 0
                
                # ตรวจสอบขนาดเฟรม
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    print("⚠️ เฟรมมีขนาดไม่ถูกต้อง")
                    continue
                
                # บันทึกเฟรมปัจจุบัน
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # ตรวจจับนก (ทุก 3 เฟรม เพื่อประสิทธิภาพ)
                if frame_count % 3 == 0 and self.detector is not None:
                    self._process_frame_detection(frame, frame_count)
                
                frame_count += 1
                
                # ปรับ sleep time ตาม FPS
                sleep_time = 1.0 / self.config['camera']['fps']
                time.sleep(sleep_time)
                
            except Exception as e:
                consecutive_failures += 1
                print(f"❌ ข้อผิดพลาดในการประมวลผลเฟรม: {e}")
                
                if consecutive_failures >= max_failures:
                    print("❌ ข้อผิดพลาดต่อเนื่องมากเกินไป - หยุดการประมวลผล")
                    break
                
                time.sleep(1)
        
        print("🛑 หยุดการประมวลผลกล้อง")
    
    def _process_frame_detection(self, frame, frame_count):
        """ประมวลผลการตรวจจับในเฟรม"""
        try:
            start_time = time.time()
            
            # ตรวจจับด้วย AI หลัก
            if self.ai_system and hasattr(self.detector, 'detect_smart'):
                detections = self.detector.detect_smart(frame, self.config['ai']['mode'])
            elif hasattr(self.detector, 'detect_birds'):
                detections = self.detector.detect_birds(frame)
            else:
                detections = []
            
            processing_time = time.time() - start_time
            
            # วิเคราะห์ผลลัพธ์
            entering = len([d for d in detections if d.get('direction') == 'entering'])
            exiting = len([d for d in detections if d.get('direction') == 'exiting'])
            total = len(detections)
            
            # บันทึกผลลัพธ์
            result = {
                'timestamp': datetime.now().isoformat(),
                'frame_number': frame_count,
                'detections': total,
                'entering': entering,
                'exiting': exiting,
                'processing_time': processing_time,
                'detections_data': detections
            }
            
            self.live_results.append(result)
            
            # อัปเดตสถิติ
            self._update_stats(result)
            
            # บันทึกลงฐานข้อมูล
            if total > 0:
                self.db_manager.save_detection_result(result)
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการตรวจจับ: {e}")
    
    def _update_stats(self, result):
        """อัปเดตสถิติการตรวจจับ"""
        self.detection_stats['total_detections'] += result['detections']
        self.detection_stats['entering_birds'] += result['entering']
        self.detection_stats['exiting_birds'] += result['exiting']
        
        if result['detections'] > 0:
            self.detection_stats['last_detection'] = result['timestamp']
    
    def stop_camera_stream(self):
        """หยุด Live Stream"""
        self.camera_running = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        print("🛑 หยุด Live Stream")
    
    def get_current_frame_encoded(self):
        """รับเฟรมปัจจุบันแบบ encoded สำหรับ web"""
        with self.frame_lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
            else:
                # สร้างเฟรมว่าง
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Feed", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # วาดข้อมูลบนเฟรม
        self._draw_info_on_frame(frame)
        
        # Encode เป็น JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _draw_info_on_frame(self, frame):
        """วาดข้อมูลบนเฟรม"""
        # วาดสถิติ
        info_text = [
            f"Mode: {self.config['ai']['mode'].upper()}",
            f"Total: {self.detection_stats['total_detections']}",
            f"Entering: {self.detection_stats['entering_birds']}",
            f"Exiting: {self.detection_stats['exiting_birds']}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def setup_routes(self):
        """ตั้งค่า Web Routes"""
        
        @self.app.route('/')
        def dashboard():
            """หน้าแดชบอร์ดหลัก"""
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/status')
        def api_status():
            """API สถานะระบบ"""
            return jsonify({
                'status': 'running',
                'camera_connected': self.camera is not None,
                'camera_running': self.camera_running,
                'ai_available': AI_AVAILABLE,
                'detector_available': self.detector is not None,
                'uptime': str(datetime.now() - self.detection_stats['session_start']),
                'config': self.config
            })
        
        @self.app.route('/api/stats')
        def api_stats():
            """API สถิติการตรวจจับ"""
            return jsonify(self.detection_stats)
        
        @self.app.route('/api/live-data')
        def api_live_data():
            """API ข้อมูล Live"""
            recent_results = list(self.live_results)[-10:]  # 10 ผลล่าสุด
            return jsonify({
                'recent_detections': recent_results,
                'current_stats': self.detection_stats,
                'frame_encoded': self.get_current_frame_encoded()
            })
        
        @self.app.route('/api/camera/start')
        def api_camera_start():
            """API เริ่มกล้อง"""
            self.start_camera_stream()
            return jsonify({'status': 'started'})
        
        @self.app.route('/api/camera/stop')
        def api_camera_stop():
            """API หยุดกล้อง"""
            self.stop_camera_stream()
            return jsonify({'status': 'stopped'})
        
        @self.app.route('/api/test-rtsp')
        def api_test_rtsp():
            """API ทดสอบการเชื่อมต่อ RTSP"""
            rtsp_url = "rtsp://ainok1:ainok123@192.168.1.100:554/stream1"
            
            print(f"🔍 ทดสอบการเชื่อมต่อ RTSP: {rtsp_url}")
            
            try:
                # ทดสอบด้วย cv2.VideoCapture
                test_cap = cv2.VideoCapture(rtsp_url)
                test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                test_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    test_cap.release()
                    
                    if ret and frame is not None:
                        return jsonify({
                            'status': 'success',
                            'message': 'RTSP connection successful',
                            'frame_shape': frame.shape
                        })
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': 'RTSP opened but cannot read frame'
                        })
                else:
                    test_cap.release()
                    return jsonify({
                        'status': 'error',
                        'message': 'Cannot open RTSP connection'
                    })
                    
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'RTSP test failed: {str(e)}'
                })
        
        @self.app.route('/api/camera/source', methods=['POST'])
        def api_camera_source():
            """API เปลี่ยนแหล่งกล้อง"""
            data = request.get_json()
            source = data.get('source', 0)
            
            # แปลงเป็นตัวเลขถ้าเป็นไปได้
            try:
                source = int(source)
            except:
                pass
            
            success = self.connect_camera(source)
            return jsonify({'status': 'success' if success else 'failed'})
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """API การตั้งค่า"""
            if request.method == 'POST':
                new_config = request.get_json()
                self.config.update(new_config)
                
                # บันทึกการตั้งค่า
                with open('integrated_config.json', 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                
                return jsonify({'status': 'updated'})
            else:
                return jsonify(self.config)
    
    def run_web_server(self):
        """รัน Web Server"""
        try:
            print(f"🌐 เริ่ม Web Server: http://{self.config['web']['host']}:{self.config['web']['port']}")
            
            server = make_server(
                self.config['web']['host'], 
                self.config['web']['port'], 
                self.app
            )
            
            server.serve_forever()
            
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่ม Web Server: {e}")
    
    def process_video_file(self, video_path):
        """ประมวลผลไฟล์วีดีโอ"""
        if not self.ai_system:
            print("❌ ไม่มีระบบ AI สำหรับประมวลผลวีดีโอ")
            return None
        
        try:
            print(f"🎬 ประมวลผลวีดีโอ: {video_path}")
            results = self.ai_system.process_video_v5(video_path)
            
            # บันทึกผลลัพธ์
            self.db_manager.save_video_result(video_path, results)
            
            print("✅ ประมวลผลวีดีโอเสร็จสิ้น")
            return results
            
        except Exception as e:
            print(f"❌ ไม่สามารถประมวลผลวีดีโอ: {e}")
            return None

class DatabaseManager:
    """📊 ระบบจัดการฐานข้อมูล"""
    
    def __init__(self, db_path='ultimate_integrated_results.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """เริ่มต้นฐานข้อมูล"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ตารางผลลัพธ์การตรวจจับ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER,
                    total_detections INTEGER,
                    entering_birds INTEGER,
                    exiting_birds INTEGER,
                    processing_time REAL,
                    source_type TEXT DEFAULT 'live',
                    data TEXT
                )
            ''')
            
            # ตารางผลลัพธ์วีดีโอ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_frames INTEGER,
                    entering_birds INTEGER,
                    exiting_birds INTEGER,
                    processing_time REAL,
                    fps REAL,
                    results TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ เริ่มต้นฐานข้อมูลสำเร็จ")
            
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่มต้นฐานข้อมูล: {e}")
    
    def save_detection_result(self, result):
        """บันทึกผลการตรวจจับ"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, frame_number, total_detections, entering_birds, exiting_birds, processing_time, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp'],
                result['frame_number'],
                result['detections'],
                result['entering'],
                result['exiting'],
                result['processing_time'],
                json.dumps(result['detections_data'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ ไม่สามารถบันทึกผลการตรวจจับ: {e}")
    
    def save_video_result(self, video_path, results):
        """บันทึกผลการประมวลผลวีดีโอ"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO video_results 
                (video_path, timestamp, total_frames, entering_birds, exiting_birds, processing_time, fps, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_path,
                datetime.now().isoformat(),
                results.get('frames_processed', 0),
                results.get('entering', 0),
                results.get('exiting', 0),
                results.get('processing_time', 0),
                results.get('fps', 0),
                json.dumps(results)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ ไม่สามารถบันทึกผลวีดีโอ: {e}")

# HTML Template สำหรับ Dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Ultimate Swallow AI V5 Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00ff88; margin-bottom: 10px; }
        .status-panel { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { background: #2a2a2a; padding: 20px; border-radius: 10px; border-left: 4px solid #00ff88; }
        .status-card h3 { color: #00ff88; margin-bottom: 10px; }
        .video-panel { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 30px; }
        .video-container { background: #2a2a2a; padding: 20px; border-radius: 10px; }
        .controls-panel { background: #2a2a2a; padding: 20px; border-radius: 10px; }
        .btn { background: #00ff88; color: #000; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #00cc6a; }
        .btn.danger { background: #ff4444; color: #fff; }
        .btn.danger:hover { background: #cc3333; }
        .live-frame { width: 100%; max-width: 640px; border: 2px solid #00ff88; border-radius: 10px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-item { background: #333; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .stat-label { font-size: 12px; color: #ccc; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Ultimate Swallow AI V5 Dashboard</h1>
            <p>ระบบตรวจจับนกแอ่นแบบครบครัน - Live Stream + AI + Analytics</p>
        </div>

        <div class="status-panel">
            <div class="status-card">
                <h3>📊 System Status</h3>
                <div id="system-status">Loading...</div>
            </div>
            <div class="status-card">
                <h3>🎥 Camera Status</h3>
                <div id="camera-status">Loading...</div>
            </div>
            <div class="status-card">
                <h3>🤖 AI Status</h3>
                <div id="ai-status">Loading...</div>
            </div>
        </div>

        <div class="video-panel">
            <div class="video-container">
                <h3>📹 Live Video Feed</h3>
                <img id="live-frame" class="live-frame" src="data:image/jpeg;base64," alt="Live Feed">
            </div>
            <div class="controls-panel">
                <h3>🎛️ Controls</h3>
                <button class="btn" onclick="startCamera()">▶️ Start Camera</button>
                <button class="btn danger" onclick="stopCamera()">⏹️ Stop Camera</button>
                <br><br>
                <h4>Camera Source:</h4>
                <input type="text" id="camera-source" placeholder="0 or IP camera URL" style="width: 100%; padding: 8px; margin: 10px 0; background: #333; color: #fff; border: 1px solid #555; border-radius: 5px;">
                <button class="btn" onclick="changeCameraSource()">🔄 Change Source</button>
                
                <br><br>
                <h4>AI Mode:</h4>
                <select id="ai-mode" style="width: 100%; padding: 8px; margin: 10px 0; background: #333; color: #fff; border: 1px solid #555; border-radius: 5px;">
                    <option value="mixed">Mixed (เข้า-ออก)</option>
                    <option value="enter">Enter Only (เข้าอย่างเดียว)</option>
                    <option value="exit">Exit Only (ออกอย่างเดียว)</option>
                </select>
            </div>
        </div>

        <div class="status-card">
            <h3>📈 Live Statistics</h3>
            <div class="stats-grid" id="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="total-detections">0</div>
                    <div class="stat-label">Total Detections</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="entering-birds">0</div>
                    <div class="stat-label">Entering Birds</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="exiting-birds">0</div>
                    <div class="stat-label">Exiting Birds</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="session-time">00:00:00</div>
                    <div class="stat-label">Session Time</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ฟังก์ชันอัปเดตข้อมูล
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('system-status').innerHTML = `
                    <div>Status: <span style="color: #00ff88;">Running</span></div>
                    <div>Uptime: ${data.uptime}</div>
                `;
                
                document.getElementById('camera-status').innerHTML = `
                    <div>Connected: <span style="color: ${data.camera_connected ? '#00ff88' : '#ff4444'};">${data.camera_connected ? 'Yes' : 'No'}</span></div>
                    <div>Running: <span style="color: ${data.camera_running ? '#00ff88' : '#ff4444'};">${data.camera_running ? 'Yes' : 'No'}</span></div>
                `;
                
                document.getElementById('ai-status').innerHTML = `
                    <div>AI Available: <span style="color: ${data.ai_available ? '#00ff88' : '#ff4444'};">${data.ai_available ? 'Yes' : 'No'}</span></div>
                    <div>Detector: <span style="color: ${data.detector_available ? '#00ff88' : '#ff4444'};">${data.detector_available ? 'Ready' : 'Not Ready'}</span></div>
                `;
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }

        async function updateLiveData() {
            try {
                const response = await fetch('/api/live-data');
                const data = await response.json();
                
                // อัปเดตสถิติ
                document.getElementById('total-detections').textContent = data.current_stats.total_detections;
                document.getElementById('entering-birds').textContent = data.current_stats.entering_birds;
                document.getElementById('exiting-birds').textContent = data.current_stats.exiting_birds;
                
                // อัปเดตเฟรม
                if (data.frame_encoded) {
                    document.getElementById('live-frame').src = 'data:image/jpeg;base64,' + data.frame_encoded;
                }
            } catch (error) {
                console.error('Error updating live data:', error);
            }
        }

        // ฟังก์ชันควบคุม
        async function startCamera() {
            try {
                await fetch('/api/camera/start');
                console.log('Camera started');
            } catch (error) {
                console.error('Error starting camera:', error);
            }
        }

        async function stopCamera() {
            try {
                await fetch('/api/camera/stop');
                console.log('Camera stopped');
            } catch (error) {
                console.error('Error stopping camera:', error);
            }
        }

        async function changeCameraSource() {
            const source = document.getElementById('camera-source').value;
            try {
                await fetch('/api/camera/source', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: source })
                });
                console.log('Camera source changed');
            } catch (error) {
                console.error('Error changing camera source:', error);
            }
        }

        // เริ่มการอัปเดตอัตโนมัติ
        updateStatus();
        updateLiveData();
        setInterval(updateStatus, 5000);  // อัปเดตทุก 5 วินาที
        setInterval(updateLiveData, 1000); // อัปเดตทุก 1 วินาที
    </script>
</body>
</html>
'''

def main():
    """ฟังก์ชันหลักสำหรับรันระบบ"""
    print("🚀 เริ่มต้น Ultimate Integrated Swallow AI System V5")
    
    # สร้างระบบ
    system = UltimateIntegratedSystem()
    
    # เริ่มกล้อง
    system.start_camera_stream()
    
    try:
        # รัน Web Server
        system.run_web_server()
    except KeyboardInterrupt:
        print("\n🛑 หยุดระบบ...")
    finally:
        # ปิดระบบ
        system.stop_camera_stream()
        print("✅ ปิดระบบเรียบร้อย")

if __name__ == "__main__":
    main()
