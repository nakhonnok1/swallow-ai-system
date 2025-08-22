#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate AI Bird Tracking System - CLEANED & OPTIMIZED V8
===============================================================================
- ทำความสะอาดโค้ดซ้ำซ้อน ปรับปรุงการจัดการโครงสร้าง
- เชื่อมต่อกับ Enhanced Database และ Enhanced API Routes อย่างสมบูรณ์
- ลดความซับซ้อน เพิ่มความเสถียร และง่ายต่อการพัฒนาต่อ
- รองรับ Windows Unicode และมีระบบ fallback ที่แข็งแกร่ง
===============================================================================
"""

import os
import sys
import cv2
import time
import json
import sqlite3
import logging
import threading
import datetime as dt
import numpy as np
from typing import List, Dict, Any, Optional

# -------- Windows Unicode Support --------
import locale
import codecs
if sys.platform.startswith('win'):
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass

# -------- Core Flask Imports --------
from flask import Flask, jsonify, render_template, Response, request, send_from_directory
from jinja2 import TemplateNotFound

# -------- System Monitoring --------
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# -------- Enhanced Integrations --------
try:
    from enhanced_database import db_manager
    ENHANCED_DB_AVAILABLE = True
    print("✅ Enhanced Database loaded successfully")
except Exception as e:
    ENHANCED_DB_AVAILABLE = False
    print(f"⚠️ Enhanced Database not available: {e}")

try:
    from enhanced_api_routes import setup_enhanced_api_routes
    ENHANCED_API_AVAILABLE = True
    print("✅ Enhanced API Routes loaded successfully")
except Exception as e:
    ENHANCED_API_AVAILABLE = False
    print(f"⚠️ Enhanced API Routes not available: {e}")
    def setup_enhanced_api_routes(app):
        pass

# -------- AI System Integrations --------
# Main AI System
try:
    from ultimate_perfect_ai_MASTER import V5_UltimatePrecisionSwallowAI, EnhancedMasterBirdDetector
    MAIN_AI_AVAILABLE = True
except Exception as e:
    MAIN_AI_AVAILABLE = False
    print(f"Warning: Main AI system not available: {e}")

# Advanced Object Detector
try:
    from advanced_object_detector import AdvancedObjectDetector
    ADVANCED_DETECTOR_AVAILABLE = True
except Exception as e:
    ADVANCED_DETECTOR_AVAILABLE = False

# Ultra Safe Detector
try:
    from ultra_safe_detector import UltraSafeDetector
    ULTRA_SAFE_DETECTOR_AVAILABLE = True
except Exception as e:
    ULTRA_SAFE_DETECTOR_AVAILABLE = False

# Simple YOLO Detector (fallback)
try:
    from simple_yolo_detector import SimpleYOLODetector
    SIMPLE_YOLO_AVAILABLE = True
except Exception as e:
    SIMPLE_YOLO_AVAILABLE = False

# Enhanced Ultra Smart AI Agent
try:
    from enhanced_ultra_smart_ai_agent import EnhancedUltraSmartAIAgent
    ENHANCED_AI_CHATBOT_AVAILABLE = True
except Exception as e:
    ENHANCED_AI_CHATBOT_AVAILABLE = False
    print(f"Warning: Enhanced Ultra Smart AI Agent not available: {e}")

# Intelligent Intruder Detection
try:
    from intelligent_intruder_integration import IntelligentIntruderIntegration, create_integration_system
    INTRUDER_DETECTION_AVAILABLE = True
except Exception as e:
    INTRUDER_DETECTION_AVAILABLE = False
    print(f"Warning: Intelligent Intruder Detection not available: {e}")

# Configuration
try:
    from config import Config as AppConfig
except Exception as e:
    class AppConfig:
        YOLO_MODEL_PATH = 'yolov8n.pt'
        ANOMALY_DB_PATH = 'anomaly_alerts.db'
        BIRD_DB_PATH = 'swallow_smart_stats.db'
        LOG_LEVEL = 'INFO'
        DEBUG_MODE = False

# -------- Logging Configuration --------
logging.basicConfig(
    level=getattr(logging, str(AppConfig.LOG_LEVEL).upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_ai_master.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('swallow_ai')

# -------- Flask Application --------
app = Flask(__name__)

# -------- Configuration Constants --------
FPS_LIMIT = 30
DETECTION_COOLDOWN = 0.5
DB_PATH = "object_detection_alerts.db"
VIDEO_SOURCE = "rtsp://ainok1:ainok123@192.168.1.100:554/stream1"

# -------- Global State Variables --------
start_time = time.time()
video_feed_active = True
current_frame = None
frame_lock = threading.Lock()

camera_props: Dict[str, Any] = {
    'resolution': (640, 480),
    'fps': 30,
    'rtsp_url': VIDEO_SOURCE if isinstance(VIDEO_SOURCE, str) else '',
    'model': 'default',
    'location': 'main_entrance',
}

# -------- Utility Functions --------
def frame_quality(frame: Optional[np.ndarray]) -> Dict[str, float]:
    """Calculate frame quality metrics"""
    if frame is None:
        return {'brightness': 0.0, 'sharpness': 0.0}
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        sharpness = float(np.var(gray)) / 255.0
        return {'brightness': brightness, 'sharpness': sharpness}
    except Exception:
        return {'brightness': 0.0, 'sharpness': 0.0}

def get_uptime() -> str:
    """Get system uptime as formatted string"""
    uptime_seconds = int(time.time() - start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# -------- Database Management --------
def init_detection_database():
    """Initialize basic detection database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            detection_type TEXT,
            confidence REAL,
            bbox_x INTEGER,
            bbox_y INTEGER,
            bbox_w INTEGER,
            bbox_h INTEGER,
            frame_info TEXT,
            camera_source TEXT,
            ai_model TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Basic detection database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def save_detection(detection_data: Dict[str, Any]) -> bool:
    """Save detection to both basic and enhanced databases"""
    try:
        # Save to basic database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        bbox = detection_data.get('bbox', [0, 0, 0, 0])
        cursor.execute('''
        INSERT INTO detections (detection_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, 
                               frame_info, camera_source, ai_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection_data.get('class', 'unknown'),
            detection_data.get('confidence', 0.0),
            bbox[0], bbox[1], bbox[2], bbox[3],
            json.dumps(detection_data.get('frame_info', {})),
            detection_data.get('camera_source', 'main_entrance'),
            detection_data.get('ai_model', 'enhanced_system')
        ))
        
        conn.commit()
        conn.close()
        
        # Mirror to enhanced database if available
        if ENHANCED_DB_AVAILABLE:
            try:
                db_manager.save_detection(detection_data)
            except Exception as e:
                logger.debug(f"Enhanced DB save failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save detection: {e}")
        return False

# -------- Core System Classes --------
class BirdCounter:
    """Bird counting and statistics management"""
    
    def __init__(self):
        self.birds_in = 0
        self.birds_out = 0
        self.current_count = 0
        self.last_detection = dt.datetime.now()
        self.detection_history: List[Dict[str, Any]] = []

    def update_from_detection(self, stats: Dict[str, Any]):
        """Update counts from detection results"""
        if not isinstance(stats, dict):
            return
            
        self.birds_in = stats.get('entering', stats.get('birds_in', self.birds_in))
        self.birds_out = stats.get('exiting', stats.get('birds_out', self.birds_out))
        self.current_count = max(0, int(self.birds_in) - int(self.birds_out))
        self.last_detection = dt.datetime.now()
        
        # Add to history
        self.detection_history.append({
            'timestamp': self.last_detection.isoformat(),
            'birds_in': self.birds_in,
            'birds_out': self.birds_out,
            'current_count': self.current_count
        })
        
        # Keep only last 100 entries
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'birds_in': self.birds_in,
            'birds_out': self.birds_out,
            'current_count': self.current_count,
            'last_detection': self.last_detection.isoformat(),
            'total_detections': len(self.detection_history)
        }

class CameraManager:
    """Enhanced camera management with robust error handling"""
    
    def __init__(self, video_source: Any):
        self.video_source = video_source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.last_frame: Optional[np.ndarray] = None
        self.error_count = 0
        self.max_errors = 10
        
    def connect(self) -> bool:
        """Connect to camera with retry logic"""
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.video_source)
            
            if self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_props['resolution'][0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_props['resolution'][1])
                self.cap.set(cv2.CAP_PROP_FPS, camera_props['fps'])
                
                # Test read
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    self.is_connected = True
                    self.error_count = 0
                    logger.info(f"Camera connected successfully: {self.video_source}")
                    return True
                    
            self.is_connected = False
            logger.error(f"Failed to connect to camera: {self.video_source}")
            return False
            
        except Exception as e:
            self.is_connected = False
            logger.error(f"Camera connection error: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame with error handling"""
        if not self.is_connected or not self.cap:
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame = frame.copy()
                self.error_count = 0
                return frame
            else:
                self.error_count += 1
                if self.error_count >= self.max_errors:
                    logger.warning("Too many read errors, attempting reconnection")
                    self.connect()
                return self.last_frame
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Frame read error: {e}")
            return self.last_frame

    def release(self):
        """Release camera resources"""
        try:
            if self.cap:
                self.cap.release()
            self.is_connected = False
            logger.info("Camera released")
        except Exception as e:
            logger.error(f"Camera release error: {e}")

class AIDetector:
    """Unified AI detection system with multiple detectors"""
    
    def __init__(self):
        self.detectors = {}
        self.current_detector = None
        self.detection_enabled = False
        
        # Initialize available detectors
        self._initialize_detectors()
        
        # Setup intruder detection if available
        if INTRUDER_DETECTION_AVAILABLE:
            try:
                self.intruder_system = create_integration_system()
                logger.info("Intruder detection system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize intruder detection: {e}")
                self.intruder_system = None
        else:
            self.intruder_system = None
            
        # Setup AI chatbot if available
        if ENHANCED_AI_CHATBOT_AVAILABLE:
            try:
                self.ai_agent = EnhancedUltraSmartAIAgent()
                logger.info("AI chatbot initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AI chatbot: {e}")
                self.ai_agent = None
        else:
            self.ai_agent = None

    def _initialize_detectors(self):
        """Initialize all available AI detectors"""
        # Ultra Safe Detector (primary)
        if ULTRA_SAFE_DETECTOR_AVAILABLE:
            try:
                self.detectors['ultra_safe'] = UltraSafeDetector()
                self.current_detector = 'ultra_safe'
                self.detection_enabled = True
                logger.info("Ultra Safe Detector initialized")
            except Exception as e:
                logger.error(f"Ultra Safe Detector initialization failed: {e}")
        
        # Advanced Object Detector (secondary)
        if ADVANCED_DETECTOR_AVAILABLE:
            try:
                self.detectors['advanced'] = AdvancedObjectDetector()
                if not self.current_detector:
                    self.current_detector = 'advanced'
                    self.detection_enabled = True
                logger.info("Advanced Object Detector initialized")
            except Exception as e:
                logger.error(f"Advanced Object Detector initialization failed: {e}")
        
        # Simple YOLO (fallback)
        if SIMPLE_YOLO_AVAILABLE:
            try:
                self.detectors['simple_yolo'] = SimpleYOLODetector()
                if not self.current_detector:
                    self.current_detector = 'simple_yolo'
                    self.detection_enabled = True
                logger.info("Simple YOLO Detector initialized")
            except Exception as e:
                logger.error(f"Simple YOLO Detector initialization failed: {e}")

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Perform object detection using the current detector"""
        if not self.detection_enabled or not self.current_detector:
            return []
            
        try:
            detector = self.detectors.get(self.current_detector)
            if not detector:
                return []
                
            # Use appropriate detection method based on detector type
            if self.current_detector == 'ultra_safe':
                _, detections, stats = detector.detect_birds_realtime(
                    frame, camera_props, frame_quality(frame)
                )
                return detections if isinstance(detections, list) else []
                
            elif self.current_detector == 'advanced':
                return detector.detect_objects(frame, camera_props, frame_quality(frame))
                
            elif self.current_detector == 'simple_yolo':
                return detector.detect_birds(frame)
                
        except Exception as e:
            logger.error(f"Detection error with {self.current_detector}: {e}")
            
        return []

    def get_chat_response(self, message: str, language: str = 'th') -> str:
        """Get AI chatbot response"""
        if self.ai_agent:
            try:
                context = {
                    'language': language,
                    'system_status': self.get_system_status(),
                    'timestamp': dt.datetime.now().isoformat()
                }
                return self.ai_agent.get_response(message, context)
            except Exception as e:
                logger.error(f"AI chatbot error: {e}")
                
        # Fallback response
        if language == 'th':
            return "ขออภัย ระบบ AI Agent ไม่พร้อมใช้งาน ณ ขณะนี้"
        else:
            return "Sorry, AI Agent is not available at the moment"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'detection_enabled': self.detection_enabled,
            'current_detector': self.current_detector,
            'available_detectors': list(self.detectors.keys()),
            'intruder_detection': self.intruder_system is not None,
            'ai_chatbot': self.ai_agent is not None,
            'uptime': get_uptime()
        }

    def setup_flask_integration(self, app):
        """Setup Flask integration for intruder detection"""
        if self.intruder_system:
            try:
                self.intruder_system.setup_flask_integration(app)
                self.intruder_system.add_camera_stream(
                    'main_camera', VIDEO_SOURCE, 'main_entrance'
                )
                logger.info("Flask integration setup completed")
            except Exception as e:
                logger.error(f"Flask integration setup failed: {e}")

# -------- Initialize System Components --------
bird_counter = BirdCounter()
camera_manager = CameraManager(VIDEO_SOURCE)
ai_detector = AIDetector()

# Setup Flask integration
ai_detector.setup_flask_integration(app)

# -------- Video Processing --------
def video_processing_thread():
    """Main video processing thread with AI detection"""
    global current_frame
    
    # Connect to camera
    if not camera_manager.connect():
        logger.error("Failed to connect to camera in video processing thread")
    
    while video_feed_active:
        try:
            frame = camera_manager.read_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Perform AI detection
            detections = ai_detector.detect_objects(frame)
            
            # Process detections
            processed_frame = frame.copy()
            detection_count = 0
            
            for detection in detections:
                try:
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    confidence = detection.get('confidence', 0.0)
                    class_name = detection.get('class', 'object')
                    
                    # Draw bounding box
                    x, y, w, h = map(int, bbox)
                    color = (0, 255, 0) if 'bird' in class_name.lower() else (255, 0, 0)
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(processed_frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Save detection
                    detection_data = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'camera_source': 'main_entrance',
                        'ai_model': ai_detector.current_detector,
                        'frame_info': frame_quality(frame)
                    }
                    
                    if save_detection(detection_data):
                        detection_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing detection: {e}")
            
            # Update bird counter if applicable
            if detection_count > 0:
                bird_counter.update_from_detection({
                    'birds_in': bird_counter.birds_in + detection_count,
                    'total_detections': detection_count
                })
            
            # Add system info to frame
            timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(processed_frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Detections: {detection_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"AI: {ai_detector.current_detector or 'None'}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Update current frame
            with frame_lock:
                current_frame = processed_frame
            
            time.sleep(1 / FPS_LIMIT)
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            time.sleep(1)

def generate_video_feed():
    """Generate video feed for web streaming"""
    while True:
        try:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    # Create placeholder frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "System Initializing...", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, "Enhanced Edition V8", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"Uptime: {get_uptime()}", (50, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1/30)  # 30 FPS limit
            
        except Exception as e:
            logger.error(f"Video feed error: {e}")
            time.sleep(0.1)

# -------- Flask Routes --------

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Main dashboard route"""
    try:
        return render_template('index.html')
    except TemplateNotFound:
        return '''
        <!DOCTYPE html>
        <html><head><title>Swallow AI Dashboard - V8</title></head>
        <body style="font-family: Arial; padding: 20px; background: #f0f0f0;">
            <h1>🪶 Swallow AI Dashboard - Enhanced V8</h1>
            <p><strong>เทมเพลต index.html ไม่พบ</strong></p>
            <div style="margin: 20px 0;">
                <a href="/video_feed" style="padding: 10px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px;">📹 ดูสตรีมวิดีโอ</a>
                <a href="/api/system-health" style="padding: 10px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px;">🔧 สถานะระบบ</a>
                <a href="/api/statistics" style="padding: 10px; background: #17a2b8; color: white; text-decoration: none; border-radius: 5px;">📊 สถิติ</a>
            </div>
            <div style="margin: 20px 0;">
                <h3>System Status:</h3>
                <p>🎥 Camera: ''' + ('Connected' if camera_manager.is_connected else 'Disconnected') + '''</p>
                <p>🤖 AI Detection: ''' + ('Enabled' if ai_detector.detection_enabled else 'Disabled') + '''</p>
                <p>⏱️ Uptime: ''' + get_uptime() + '''</p>
            </div>
        </body></html>
        '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_video_feed(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/anomaly_images/<path:filename>')
def serve_anomaly_image(filename):
    """Serve anomaly detection images"""
    try:
        return send_from_directory('../anomaly_images', filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({'error': 'Image not found'}), 404

# -------- API Routes --------

@app.route('/api/system-health')
def api_system_health():
    """Get comprehensive system health status"""
    try:
        health_data = {
            'status': 'healthy',
            'uptime': get_uptime(),
            'camera': {
                'connected': camera_manager.is_connected,
                'source': VIDEO_SOURCE,
                'resolution': camera_props['resolution'],
                'fps': camera_props['fps']
            },
            'ai_system': ai_detector.get_system_status(),
            'bird_counter': bird_counter.get_stats(),
            'integrations': {
                'enhanced_database': ENHANCED_DB_AVAILABLE,
                'enhanced_api': ENHANCED_API_AVAILABLE,
                'intruder_detection': INTRUDER_DETECTION_AVAILABLE,
                'ai_chatbot': ENHANCED_AI_CHATBOT_AVAILABLE
            },
            'timestamp': dt.datetime.now().isoformat()
        }
        
        # Add system metrics if available
        if PSUTIL_AVAILABLE:
            try:
                import psutil
                health_data['system_metrics'] = {
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('.').percent
                }
            except Exception:
                pass
        
        return jsonify(health_data)
        
    except Exception as e:
        logger.error(f"System health check error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/statistics')
def api_statistics():
    """Get system statistics"""
    try:
        if ENHANCED_DB_AVAILABLE:
            # Get from enhanced database
            stats = db_manager.get_statistics(7)  # Last 7 days
        else:
            # Fallback statistics
            stats = {
                'period_days': 7,
                'total_detections': bird_counter.birds_in + bird_counter.birds_out,
                'entries': bird_counter.birds_in,
                'exits': bird_counter.birds_out,
                'current_count': bird_counter.current_count,
                'last_detection': bird_counter.last_detection.isoformat()
            }
        
        stats['timestamp'] = dt.datetime.now().isoformat()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """AI chatbot endpoint"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message'].strip()
        language = data.get('language', 'th')
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        response = ai_detector.get_chat_response(message, language)
        
        return jsonify({
            'response': response,
            'timestamp': dt.datetime.now().isoformat(),
            'language': language
        })
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({'error': 'Chat service unavailable'}), 500

@app.route('/api/recent-detections')
def api_recent_detections():
    """Get recent detection data"""
    try:
        if ENHANCED_DB_AVAILABLE:
            # Get from enhanced database
            activities = db_manager.get_recent_activities(limit=10)
            return jsonify({'activities': activities})
        else:
            # Get from basic database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT detection_type, confidence, timestamp, ai_model
            FROM detections 
            ORDER BY timestamp DESC 
            LIMIT 10
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            detections = []
            for row in rows:
                detections.append({
                    'type': row[0],
                    'confidence': row[1],
                    'timestamp': row[2],
                    'ai_model': row[3]
                })
            
            return jsonify({'detections': detections})
            
    except Exception as e:
        logger.error(f"Recent detections error: {e}")
        return jsonify({'error': str(e)}), 500

# -------- Error Handlers --------

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# -------- Enhanced API Integration --------
if ENHANCED_API_AVAILABLE:
    try:
        setup_enhanced_api_routes(app)
        logger.info("Enhanced API routes setup completed")
    except Exception as e:
        logger.error(f"Enhanced API routes setup failed: {e}")

# -------- Main Application Entry Point --------
def main():
    """Main application entry point"""
    logger.info('🚀 Starting Ultimate AI Bird Tracking System - Enhanced Edition V8')
    
    # Initialize enhanced database if available
    if ENHANCED_DB_AVAILABLE:
        try:
            logger.info("Enhanced Database system ready")
        except Exception as e:
            logger.error(f"Enhanced Database initialization failed: {e}")
    
    # Initialize basic detection database
    init_detection_database()
    
    # Start video processing thread
    logger.info("🎥 Starting video processing thread...")
    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()
    
    # Connect to camera
    logger.info("📹 Connecting to camera...")
    if camera_manager.connect():
        logger.info("✅ Camera connected successfully")
    else:
        logger.warning("⚠️ Camera connection failed - system will continue with placeholder frames")
    
    # Log system status
    logger.info(f"🤖 AI Detection: {'Enabled' if ai_detector.detection_enabled else 'Disabled'}")
    logger.info(f"🔧 Enhanced Database: {'Available' if ENHANCED_DB_AVAILABLE else 'Not Available'}")
    logger.info(f"🛡️ Intruder Detection: {'Available' if INTRUDER_DETECTION_AVAILABLE else 'Not Available'}")
    logger.info(f"💬 AI Chatbot: {'Available' if ENHANCED_AI_CHATBOT_AVAILABLE else 'Not Available'}")
    
    logger.info("🌐 Starting Flask web server...")
    logger.info("📍 Access the system at: http://127.0.0.1:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=AppConfig.DEBUG_MODE, threaded=True)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down system...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Cleanup
        global video_feed_active
        video_feed_active = False
        camera_manager.release()
        logger.info("🔄 System shutdown complete")

if __name__ == '__main__':
    main()
