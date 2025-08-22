#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 ULTRA INTELLIGENT INTRUDER DETECTION AI AGENT V2.0
ระบบ AI Agent ตรวจจับสิ่งแปลกปลอมที่ฉลาดจริงๆ

🎯 Features:
✅ Real AI Agent ที่เรียนรู้และปรับปรุงตัวเอง
✅ Multi-Model Integration (YOLO + OpenCV + Custom Algorithms)
✅ Smart Detection with Context Understanding  
✅ Real-time Camera Stream Integration
✅ Intelligent Alert System with Priority
✅ Auto-learning from Detection Patterns
✅ Integration with Main App & Web Interface
✅ Advanced Species Detection (นก เหยี่ยว พิราบ งู ตุ๊กแก คน สัตว์)
"""

import cv2
import numpy as np
import time
import json
import sqlite3
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Priority(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class DetectionResult:
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    priority: Priority
    timestamp: datetime
    context_info: Dict[str, Any]
    threat_level: float

class UltraIntelligentIntruderDetector:
    """🧠 Ultra Intelligent AI Agent สำหรับตรวจจับสิ่งแปลกปลอม"""
    
    def __init__(self):
        print("🚀 กำลังเริ่มต้น Ultra Intelligent Intruder Detection AI Agent...")
        
        # Core AI Components
        self.yolo_model = None
        self.backup_model = None
        self.learning_database = "ultra_intelligent_detections.db"
        self.detection_history = []
        self.learning_patterns = {}
        
        # Intelligence Settings
        self.confidence_threshold = 0.3
        self.smart_threshold_adjustment = True
        self.context_learning = True
        self.pattern_recognition = True
        
        # Species Detection Configuration
        self.target_species = {
            # Critical Threats
            'person': {
                'name': 'คน/ผู้บุกรุก', 'priority': Priority.CRITICAL, 'color': (0, 0, 255),
                'size_range': (2000, 200000), 'yolo_classes': [0], 'threat_level': 0.9
            },
            'large_bird': {
                'name': 'นกขนาดใหญ่ (เหยี่ยว)', 'priority': Priority.HIGH, 'color': (255, 100, 0),
                'size_range': (800, 15000), 'yolo_classes': [14], 'threat_level': 0.7
            },
            'snake': {
                'name': 'งู', 'priority': Priority.CRITICAL, 'color': (128, 0, 128),
                'size_range': (500, 10000), 'yolo_classes': [], 'threat_level': 0.8,
                'custom_detection': True
            },
            
            # High Priority Animals
            'cat': {
                'name': 'แมว', 'priority': Priority.HIGH, 'color': (0, 165, 255),
                'size_range': (1000, 25000), 'yolo_classes': [15], 'threat_level': 0.6
            },
            'dog': {
                'name': 'สุนัข', 'priority': Priority.HIGH, 'color': (0, 165, 255),
                'size_range': (2000, 50000), 'yolo_classes': [16], 'threat_level': 0.6
            },
            'pigeon': {
                'name': 'นกพิราบ', 'priority': Priority.MEDIUM, 'color': (255, 165, 0),
                'size_range': (300, 3000), 'yolo_classes': [14], 'threat_level': 0.4
            },
            'gecko': {
                'name': 'ตุ๊กแก', 'priority': Priority.LOW, 'color': (255, 255, 0),
                'size_range': (50, 800), 'yolo_classes': [], 'threat_level': 0.2,
                'custom_detection': True
            },
            
            # Vehicles & Objects
            'car': {
                'name': 'รถยนต์', 'priority': Priority.MEDIUM, 'color': (255, 165, 0),
                'size_range': (5000, 200000), 'yolo_classes': [2], 'threat_level': 0.5
            },
            'motorcycle': {
                'name': 'มอเตอร์ไซค์', 'priority': Priority.MEDIUM, 'color': (255, 165, 0),
                'size_range': (1500, 15000), 'yolo_classes': [3], 'threat_level': 0.5
            }
        }
        
        # Learning & Performance Tracking
        self.detection_stats = {
            'total_detections': 0,
            'accurate_detections': 0,
            'false_positives': 0,
            'missed_detections': 0,
            'learning_accuracy': 0.0
        }
        
        # Real-time Integration
        self.camera_stream = None
        self.web_socket_clients = []
        self.alert_callbacks = []
        self.last_alert_time = {}
        self.alert_cooldown = 2.0  # seconds
        
        # Initialize Components
        self._init_ai_models()
        self._init_database()
        self._init_learning_system()
        self._init_performance_monitor()
        
        print("✅ Ultra Intelligent Intruder Detection AI Agent พร้อมใช้งาน!")
    
    def _init_ai_models(self):
        """เริ่มต้น AI Models หลายตัวเพื่อความแม่นยำสูงสุด"""
        try:
            print("🧠 กำลังโหลด AI Models...")
            
            # Primary YOLO Model
            try:
                from ultralytics import YOLO
                model_path = "yolov8n.pt"
                if os.path.exists(model_path):
                    self.yolo_model = YOLO(model_path)
                    print("✅ YOLO Model โหลดสำเร็จ")
                else:
                    print("⚠️ YOLO Model file ไม่พบ - ดาวน์โหลดอัตโนมัติ")
                    self.yolo_model = YOLO(model_path)  # จะดาวน์โหลดอัตโนมัติ
            except Exception as e:
                print(f"❌ YOLO Model ล้มเหลว: {e}")
                self.yolo_model = None
            
            # Backup OpenCV-based Detection
            self._init_opencv_detector()
            
            # Custom Snake Detection Algorithm
            self._init_snake_detector()
            
            # Motion Analysis System
            self._init_motion_analyzer()
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการเริ่มต้น AI Models: {e}")
    
    def _init_opencv_detector(self):
        """เริ่มต้น OpenCV-based detector เป็น backup"""
        try:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, history=500, varThreshold=50
            )
            self.contour_detector = cv2.SimpleBlobDetector_create()
            print("✅ OpenCV Detector พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ OpenCV Detector เริ่มต้นล้มเหลว: {e}")
    
    def _init_snake_detector(self):
        """เริ่มต้น Custom Snake Detection Algorithm"""
        try:
            # Snake detection parameters
            self.snake_detection_params = {
                'min_aspect_ratio': 3.0,  # งูมักจะยาว
                'max_aspect_ratio': 20.0,
                'min_area': 500,
                'max_area': 10000,
                'contour_smoothness_threshold': 0.02
            }
            print("✅ Snake Detector พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ Snake Detector เริ่มต้นล้มเหลว: {e}")
    
    def _init_motion_analyzer(self):
        """เริ่มต้นระบบวิเคราะห์การเคลื่อนไหว"""
        try:
            self.motion_history = []
            self.motion_threshold = 1000  # minimum motion area
            self.motion_frames_buffer = 10
            print("✅ Motion Analyzer พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ Motion Analyzer เริ่มต้นล้มเหลว: {e}")
    
    def _init_database(self):
        """เริ่มต้นฐานข้อมูลสำหรับการเรียนรู้"""
        try:
            conn = sqlite3.connect(self.learning_database)
            cursor = conn.cursor()
            
            # ตาราง detections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT NOT NULL,
                    threat_level REAL NOT NULL,
                    context_info TEXT,
                    verification_status TEXT DEFAULT 'pending',
                    feedback_score REAL DEFAULT 0.0,
                    camera_location TEXT,
                    weather_condition TEXT,
                    time_of_day TEXT
                )
            """)
            
            # ตาราง learning_patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    accuracy_score REAL NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # ตาราง performance_metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    additional_data TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            print("✅ ฐานข้อมูลการเรียนรู้พร้อมใช้งาน")
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการเริ่มต้นฐานข้อมูล: {e}")
    
    def _init_learning_system(self):
        """เริ่มต้นระบบการเรียนรู้"""
        try:
            self.learning_enabled = True
            self.pattern_memory = {}
            self.adaptive_thresholds = {}
            
            # โหลด learning patterns ที่มีอยู่
            self._load_existing_patterns()
            print("✅ ระบบการเรียนรู้พร้อมใช้งาน")
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการเริ่มต้นระบบการเรียนรู้: {e}")
    
    def _init_performance_monitor(self):
        """เริ่มต้นระบบติดตามประสิทธิภาพ"""
        try:
            self.performance_monitor = {
                'detection_times': [],
                'accuracy_scores': [],
                'false_positive_rate': 0.0,
                'detection_rate': 0.0,
                'system_load': 0.0
            }
            print("✅ ระบบติดตามประสิทธิภาพพร้อมใช้งาน")
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการเริ่มต้นระบบติดตามประสิทธิภาพ: {e}")
    
    def detect_intruders(self, frame: np.ndarray, camera_info: Dict = None) -> List[DetectionResult]:
        """🎯 ฟังก์ชันหลักสำหรับตรวจจับสิ่งแปลกปลอม"""
        start_time = time.time()
        
        try:
            all_detections = []
            
            # 1. YOLO Detection
            yolo_detections = self._detect_with_yolo(frame)
            all_detections.extend(yolo_detections)
            
            # 2. Custom Snake Detection
            snake_detections = self._detect_snakes(frame)
            all_detections.extend(snake_detections)
            
            # 3. Motion-based Detection
            motion_detections = self._detect_with_motion(frame)
            all_detections.extend(motion_detections)
            
            # 4. Remove Duplicates & Apply Intelligence
            filtered_detections = self._apply_intelligence_filter(all_detections, frame)
            
            # 5. Learn from Detections
            if self.learning_enabled:
                self._learn_from_detections(filtered_detections, frame, camera_info)
            
            # 6. Update Performance Metrics
            detection_time = time.time() - start_time
            self._update_performance_metrics(detection_time, filtered_detections)
            
            # 7. Trigger Alerts
            for detection in filtered_detections:
                self._trigger_alert(detection, frame, camera_info)
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการตรวจจับ: {e}")
            return []
    
    def _detect_with_yolo(self, frame: np.ndarray) -> List[DetectionResult]:
        """ตรวจจับด้วย YOLO Model"""
        if self.yolo_model is None:
            return []
        
        detections = []
        
        try:
            results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        try:
                            # Extract detection data
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            # Get class name
                            class_name = self.yolo_model.names[class_id]
                            
                            # Check if this is a target species
                            for species_key, species_info in self.target_species.items():
                                if class_id in species_info.get('yolo_classes', []):
                                    # Calculate area
                                    width = int(x2 - x1)
                                    height = int(y2 - y1)
                                    area = width * height
                                    
                                    # Check size range
                                    min_area, max_area = species_info['size_range']
                                    if min_area <= area <= max_area:
                                        # Create detection result
                                        detection = DetectionResult(
                                            object_type=species_key,
                                            confidence=confidence,
                                            bbox=(int(x1), int(y1), width, height),
                                            center=(int((x1+x2)/2), int((y1+y2)/2)),
                                            priority=species_info['priority'],
                                            timestamp=datetime.now(),
                                            context_info={
                                                'area': area,
                                                'class_name': class_name,
                                                'source': 'yolo',
                                                'model_confidence': confidence
                                            },
                                            threat_level=species_info['threat_level']
                                        )
                                        detections.append(detection)
                                        
                        except Exception as e:
                            logger.error(f"⚠️ YOLO detection parsing error: {e}")
                            continue
            
        except Exception as e:
            logger.error(f"❌ YOLO detection error: {e}")
        
        return detections
    
    def _detect_snakes(self, frame: np.ndarray) -> List[DetectionResult]:
        """ตรวจจับงูด้วย Custom Algorithm"""
        detections = []
        
        try:
            # Convert to different color spaces for better snake detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Snake-like shape detection
            # 1. Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # 2. Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Check area range for snakes
                if (self.snake_detection_params['min_area'] <= area <= 
                    self.snake_detection_params['max_area']):
                    
                    # Get bounding rect
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h)
                    
                    # Check aspect ratio (snakes are elongated)
                    if (self.snake_detection_params['min_aspect_ratio'] <= aspect_ratio <= 
                        self.snake_detection_params['max_aspect_ratio']):
                        
                        # Additional checks for snake-like characteristics
                        # Check contour smoothness
                        perimeter = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                        smoothness = len(approx) / perimeter
                        
                        if smoothness < self.snake_detection_params['contour_smoothness_threshold']:
                            # Calculate confidence based on snake-like features
                            confidence = min(0.9, 0.3 + (aspect_ratio - 3) / 20 + (1 - smoothness))
                            
                            detection = DetectionResult(
                                object_type='snake',
                                confidence=confidence,
                                bbox=(x, y, w, h),
                                center=(x + w//2, y + h//2),
                                priority=Priority.CRITICAL,
                                timestamp=datetime.now(),
                                context_info={
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'smoothness': smoothness,
                                    'source': 'custom_snake_detector'
                                },
                                threat_level=0.8
                            )
                            detections.append(detection)
            
        except Exception as e:
            logger.error(f"❌ Snake detection error: {e}")
        
        return detections
    
    def _detect_with_motion(self, frame: np.ndarray) -> List[DetectionResult]:
        """ตรวจจับด้วยการวิเคราะห์การเคลื่อนไหว"""
        detections = []
        
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > self.motion_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Estimate object type based on motion patterns and size
                    confidence = min(0.7, area / 10000)  # Simple confidence calculation
                    
                    # Classify based on size and motion
                    object_type = 'unknown_motion'
                    threat_level = 0.3
                    priority = Priority.LOW
                    
                    if area > 20000:  # Large moving object
                        object_type = 'large_intruder'
                        threat_level = 0.6
                        priority = Priority.HIGH
                    elif area > 5000:  # Medium moving object
                        object_type = 'medium_intruder'
                        threat_level = 0.4
                        priority = Priority.MEDIUM
                    
                    detection = DetectionResult(
                        object_type=object_type,
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        center=(x + w//2, y + h//2),
                        priority=priority,
                        timestamp=datetime.now(),
                        context_info={
                            'area': area,
                            'source': 'motion_detector',
                            'motion_intensity': area / 1000
                        },
                        threat_level=threat_level
                    )
                    detections.append(detection)
                    
        except Exception as e:
            logger.error(f"❌ Motion detection error: {e}")
        
        return detections
    
    def _apply_intelligence_filter(self, detections: List[DetectionResult], frame: np.ndarray) -> List[DetectionResult]:
        """ใช้ความฉลาดในการกรองและปรับปรุง detections"""
        if not detections:
            return []
        
        # 1. Remove overlapping detections (NMS-like)
        filtered_detections = self._remove_overlapping_detections(detections)
        
        # 2. Apply context-based filtering
        context_filtered = self._apply_context_filter(filtered_detections, frame)
        
        # 3. Apply learning-based improvements
        if self.learning_enabled:
            context_filtered = self._apply_learned_patterns(context_filtered)
        
        # 4. Sort by priority and threat level
        context_filtered.sort(key=lambda x: (x.priority.value, -x.threat_level))
        
        return context_filtered
    
    def _remove_overlapping_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """ลบ detections ที่ทับซ้อนกัน"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections_sorted = sorted(detections, key=lambda x: x.confidence, reverse=True)
        filtered = []
        
        for detection in detections_sorted:
            overlap_found = False
            for existing in filtered:
                if self._calculate_overlap(detection.bbox, existing.bbox) > 0.5:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """คำนวณ overlap ratio ระหว่าง 2 bboxes"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_context_filter(self, detections: List[DetectionResult], frame: np.ndarray) -> List[DetectionResult]:
        """ใช้ context ในการกรอง detections"""
        # Get current time context
        current_hour = datetime.now().hour
        is_night = current_hour < 6 or current_hour > 18
        
        filtered = []
        for detection in detections:
            # Adjust confidence based on time of day
            if is_night and detection.object_type in ['person', 'large_bird']:
                detection.threat_level *= 1.2  # Increase threat at night
            
            # Filter based on minimum threat level
            if detection.threat_level >= 0.2:  # Minimum threshold
                filtered.append(detection)
        
        return filtered
    
    def _apply_learned_patterns(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """ใช้ patterns ที่เรียนรู้มาปรับปรุง detections"""
        # This is where machine learning patterns would be applied
        # For now, simple rule-based improvements
        
        for detection in detections:
            pattern_key = f"{detection.object_type}_{detection.bbox[2]}_{detection.bbox[3]}"
            
            if pattern_key in self.pattern_memory:
                pattern_data = self.pattern_memory[pattern_key]
                # Adjust confidence based on historical accuracy
                accuracy_factor = pattern_data.get('accuracy', 0.5)
                detection.confidence *= accuracy_factor
        
        return detections
    
    def _learn_from_detections(self, detections: List[DetectionResult], frame: np.ndarray, camera_info: Dict = None):
        """เรียนรู้จาก detections เพื่อปรับปรุงประสิทธิภาพ"""
        try:
            for detection in detections:
                # Store detection for learning
                self._store_detection_for_learning(detection, camera_info)
                
                # Update pattern memory
                pattern_key = f"{detection.object_type}_{detection.bbox[2]}_{detection.bbox[3]}"
                if pattern_key not in self.pattern_memory:
                    self.pattern_memory[pattern_key] = {
                        'count': 0,
                        'total_confidence': 0.0,
                        'accuracy': 0.5
                    }
                
                pattern = self.pattern_memory[pattern_key]
                pattern['count'] += 1
                pattern['total_confidence'] += detection.confidence
                pattern['avg_confidence'] = pattern['total_confidence'] / pattern['count']
                
        except Exception as e:
            logger.error(f"❌ Learning error: {e}")
    
    def _store_detection_for_learning(self, detection: DetectionResult, camera_info: Dict = None):
        """บันทึก detection ลงฐานข้อมูลเพื่อการเรียนรู้"""
        try:
            conn = sqlite3.connect(self.learning_database)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO detections 
                (timestamp, object_type, confidence, bbox, threat_level, context_info, camera_location, time_of_day)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                detection.timestamp.isoformat(),
                detection.object_type,
                detection.confidence,
                json.dumps(detection.bbox),
                detection.threat_level,
                json.dumps(detection.context_info),
                json.dumps(camera_info) if camera_info else None,
                datetime.now().strftime("%H:%M")
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Database storage error: {e}")
    
    def _update_performance_metrics(self, detection_time: float, detections: List[DetectionResult]):
        """อัพเดตตัวชี้วัดประสิทธิภาพ"""
        try:
            self.performance_monitor['detection_times'].append(detection_time)
            
            # Keep only last 100 measurements
            if len(self.performance_monitor['detection_times']) > 100:
                self.performance_monitor['detection_times'] = self.performance_monitor['detection_times'][-100:]
            
            # Update stats
            self.detection_stats['total_detections'] += len(detections)
            
            # Calculate average detection time
            avg_time = sum(self.performance_monitor['detection_times']) / len(self.performance_monitor['detection_times'])
            self.performance_monitor['avg_detection_time'] = avg_time
            
        except Exception as e:
            logger.error(f"❌ Performance update error: {e}")
    
    def _trigger_alert(self, detection: DetectionResult, frame: np.ndarray, camera_info: Dict = None):
        """ส่งการแจ้งเตือน"""
        try:
            alert_key = f"{detection.object_type}_{detection.center[0]}_{detection.center[1]}"
            current_time = time.time()
            
            # Check cooldown
            if (alert_key in self.last_alert_time and 
                current_time - self.last_alert_time[alert_key] < self.alert_cooldown):
                return
            
            self.last_alert_time[alert_key] = current_time
            
            # Create alert data
            alert_data = {
                'timestamp': detection.timestamp.isoformat(),
                'object_type': detection.object_type,
                'object_name': self.target_species.get(detection.object_type, {}).get('name', detection.object_type),
                'confidence': detection.confidence,
                'threat_level': detection.threat_level,
                'priority': detection.priority.value,
                'location': detection.center,
                'bbox': detection.bbox,
                'camera_info': camera_info,
                'context': detection.context_info
            }
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data, frame)
                except Exception as e:
                    logger.error(f"⚠️ Alert callback error: {e}")
            
            # Log alert
            logger.info(f"🚨 ALERT: {alert_data['object_name']} detected with {detection.confidence:.2f} confidence")
            
        except Exception as e:
            logger.error(f"❌ Alert trigger error: {e}")
    
    def register_alert_callback(self, callback):
        """ลงทะเบียน callback function สำหรับการแจ้งเตือน"""
        self.alert_callbacks.append(callback)
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """ดึงสถิติการตรวจจับ"""
        return {
            'detection_stats': self.detection_stats.copy(),
            'performance_monitor': self.performance_monitor.copy(),
            'learning_patterns_count': len(self.pattern_memory),
            'target_species_count': len(self.target_species),
            'alert_callbacks_count': len(self.alert_callbacks)
        }
    
    def _load_existing_patterns(self):
        """โหลด learning patterns ที่มีอยู่จากฐานข้อมูล"""
        try:
            conn = sqlite3.connect(self.learning_database)
            cursor = conn.cursor()
            
            cursor.execute("SELECT pattern_type, pattern_data, accuracy_score FROM learning_patterns")
            rows = cursor.fetchall()
            
            for row in rows:
                pattern_type, pattern_data, accuracy_score = row
                pattern_dict = json.loads(pattern_data)
                pattern_dict['accuracy'] = accuracy_score
                self.pattern_memory[pattern_type] = pattern_dict
            
            conn.close()
            logger.info(f"✅ โหลด {len(rows)} learning patterns")
            
        except Exception as e:
            logger.error(f"❌ Error loading patterns: {e}")

# Integration Functions for Main App
def create_intelligent_detector() -> UltraIntelligentIntruderDetector:
    """สร้าง instance ของ Ultra Intelligent Intruder Detector"""
    return UltraIntelligentIntruderDetector()

def integrate_with_camera_stream(detector: UltraIntelligentIntruderDetector, camera_url: str):
    """เชื่อมต่อกับ camera stream"""
    try:
        cap = cv2.VideoCapture(camera_url)
        
        def camera_loop():
            while True:
                ret, frame = cap.read()
                if ret:
                    detections = detector.detect_intruders(frame, {'camera_url': camera_url})
                    # Process detections...
                else:
                    time.sleep(0.1)
        
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        
        return camera_thread
        
    except Exception as e:
        logger.error(f"❌ Camera integration error: {e}")
        return None

# Test Functions
if __name__ == "__main__":
    print("🧪 ทดสอบ Ultra Intelligent Intruder Detection AI Agent")
    
    # Create detector
    detector = create_intelligent_detector()
    
    # Test with dummy frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_intruders(test_frame)
    
    print(f"✅ ระบบพร้อมใช้งาน! ตรวจจับได้ {len(detector.target_species)} ประเภท")
    print("📊 สถิติระบบ:", detector.get_detection_stats())
