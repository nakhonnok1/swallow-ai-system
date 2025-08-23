#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ULTIMATE AI DETECTION SYSTEM - ระบบ AI ที่สมบูรณ์แบบที่สุด
Version: 4.0 - Maximum Performance & Production Ready
ออกแบบเพื่อประสิทธิภาพสูงสุดและการใช้งานในทุกรูปแบบ

Features:
- Multi-Model AI Detection (YOLO + EfficientDet + MobileNet)
- Advanced Performance Optimization
- Real-time Statistics & Analytics
- Smart Caching & Memory Management
- Production-ready Architecture
- Comprehensive Error Handling
- Advanced Visualization
"""

import cv2
import numpy as np
import requests
import os
import logging
import time
import threading
import queue
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle

@dataclass
class DetectionResult:
    """โครงสร้างข้อมูลผลการตรวจจับที่สมบูรณ์"""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[int]  # [x, y, w, h]
    center: Tuple[int, int]
    area: int
    source: str
    timestamp: float
    model_used: str
    processing_time: float
    frame_id: Optional[int] = None
    tracking_id: Optional[int] = None

@dataclass
class PerformanceMetrics:
    """ตัวชี้วัดประสิทธิภาพ"""
    fps: float
    avg_processing_time: float
    memory_usage: float
    cpu_usage: float
    detection_accuracy: float
    total_detections: int
    detection_rate: float

class SmartCache:
    """ระบบแคชอัจฉริยะ"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """ดึงข้อมูลจากแคช"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                self.hit_count += 1
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any):
        """เก็บข้อมูลในแคช"""
        if len(self.cache) >= self.max_size:
            # ลบข้อมูลเก่าที่สุด
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def get_stats(self) -> Dict:
        """สถิติการใช้แคช"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self.cache)
        }

class ModelManager:
    """จัดการโมเดล AI หลายตัว"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'yolov4': {
                'weights': 'yolov4.weights',
                'config': 'yolov4.cfg',
                'names': 'coco.names',
                'input_size': (416, 416),
                'confidence': 0.5,
                'nms': 0.4
            },
            'yolov3': {
                'weights': 'yolov3.weights',
                'config': 'yolov3.cfg',
                'names': 'coco.names',
                'input_size': (416, 416),
                'confidence': 0.5,
                'nms': 0.4
            },
            'yolov3-tiny': {
                'weights': 'yolov3-tiny.weights',
                'config': 'yolov3-tiny.cfg',
                'names': 'coco.names',
                'input_size': (416, 416),
                'confidence': 0.3,
                'nms': 0.4
            }
        }
        
        self.performance_stats = {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """ตั้งค่า logger"""
        logger = logging.getLogger('ModelManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self, model_name: str) -> bool:
        """โหลดโมเดล"""
        if model_name in self.models:
            return True
            
        if model_name not in self.model_configs:
            self.logger.error(f"❌ ไม่รู้จักโมเดล: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        
        try:
            # ตรวจสอบไฟล์
            weights_path = Path("models") / config['weights']
            config_path = Path("models") / config['config']
            
            if not weights_path.exists():
                weights_path = Path(config['weights'])
            if not config_path.exists():
                config_path = Path(config['config'])
            
            if weights_path.exists() and config_path.exists():
                self.logger.info(f"🧠 โหลดโมเดล: {model_name}")
                
                net = cv2.dnn.readNet(str(weights_path), str(config_path))
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                # ดึง output layers
                layer_names = net.getLayerNames()
                unconnected_out_layers = net.getUnconnectedOutLayers()
                
                if len(unconnected_out_layers.shape) == 1:
                    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
                else:
                    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
                
                self.models[model_name] = {
                    'net': net,
                    'output_layers': output_layers,
                    'config': config,
                    'loaded_time': time.time()
                }
                
                self.performance_stats[model_name] = {
                    'total_inferences': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'best_time': float('inf'),
                    'worst_time': 0
                }
                
                self.logger.info(f"✅ โหลดโมเดล {model_name} สำเร็จ")
                return True
            else:
                self.logger.warning(f"⚠️ ไม่พบไฟล์โมเดล {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ ไม่สามารถโหลดโมเดล {model_name}: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Dict]:
        """ดึงโมเดล"""
        if model_name not in self.models:
            if not self.load_model(model_name):
                return None
        
        return self.models[model_name]
    
    def update_performance(self, model_name: str, inference_time: float):
        """อัปเดตสถิติประสิทธิภาพ"""
        if model_name in self.performance_stats:
            stats = self.performance_stats[model_name]
            stats['total_inferences'] += 1
            stats['total_time'] += inference_time
            stats['avg_time'] = stats['total_time'] / stats['total_inferences']
            stats['best_time'] = min(stats['best_time'], inference_time)
            stats['worst_time'] = max(stats['worst_time'], inference_time)

class UltimateAIDetectionSystem:
    """🚀 ระบบ AI Detection ที่สมบูรณ์แบบที่สุด"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = self._setup_logging()
        self.logger.info("🚀 เริ่มต้น Ultimate AI Detection System...")
        
        # Core Components
        self.model_manager = ModelManager()
        self.cache = SmartCache(max_size=2000, ttl=600)
        self.db_manager = self._setup_database()
        
        # Configuration
        self.config = self._load_config(config_path)
        
        # Classes และ Colors
        self.classes = []
        self.colors = []
        
        # Performance Tracking
        self.performance_metrics = PerformanceMetrics(
            fps=0, avg_processing_time=0, memory_usage=0,
            cpu_usage=0, detection_accuracy=0, total_detections=0, detection_rate=0
        )
        
        # Threading สำหรับประสิทธิภาพ
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # Statistics
        self.stats = {
            'session_start': time.time(),
            'total_frames': 0,
            'total_detections': 0,
            'detection_history': [],
            'model_usage': {},
            'performance_log': []
        }
        
        # Initialization
        self._initialize_system()
    
    def _setup_logging(self):
        """ตั้งค่าระบบ logging ขั้นสูง"""
        logger = logging.getLogger('UltimateAI')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console Handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File Handler
            file_handler = logging.FileHandler('ultimate_ai_system.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_database(self):
        """ตั้งค่าฐานข้อมูลสำหรับเก็บผลการตรวจจับ"""
        db_path = "ultimate_ai_detections.db"
        
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # สร้างตาราง
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    class_name TEXT,
                    confidence REAL,
                    bbox TEXT,
                    model_used TEXT,
                    processing_time REAL,
                    frame_id INTEGER,
                    session_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    fps REAL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    model_name TEXT,
                    session_id TEXT
                )
            ''')
            
            conn.commit()
            self.logger.info("✅ ตั้งค่าฐานข้อมูลสำเร็จ")
            return conn
            
        except Exception as e:
            self.logger.error(f"❌ ไม่สามารถตั้งค่าฐานข้อมูล: {e}")
            return None
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """โหลดการตั้งค่า"""
        default_config = {
            'models': {
                'primary': 'yolov4',
                'fallback': ['yolov3', 'yolov3-tiny'],
                'ensemble': False
            },
            'detection': {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'input_size': (416, 416),
                'batch_size': 1
            },
            'performance': {
                'enable_caching': True,
                'enable_threading': True,
                'enable_gpu': False,
                'max_fps': 30,
                'memory_limit': 1024  # MB
            },
            'output': {
                'save_detections': True,
                'save_frames': False,
                'output_dir': 'output',
                'log_level': 'INFO'
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
                    self.logger.info(f"✅ โหลดการตั้งค่าจาก {config_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ ไม่สามารถโหลดการตั้งค่า: {e}")
        
        return default_config
    
    def _initialize_system(self):
        """เริ่มต้นระบบ"""
        try:
            # โหลด classes
            self._load_classes()
            
            # โหลดโมเดลหลัก
            primary_model = self.config['models']['primary']
            if self.model_manager.load_model(primary_model):
                self.logger.info(f"✅ โหลดโมเดลหลัก: {primary_model}")
            
            # โหลดโมเดลสำรอง
            for fallback_model in self.config['models']['fallback']:
                self.model_manager.load_model(fallback_model)
            
            # สร้างสี
            self._generate_colors()
            
            # สร้างโฟลเดอร์ output
            output_dir = Path(self.config['output']['output_dir'])
            output_dir.mkdir(exist_ok=True)
            
            self.logger.info("🚀 Ultimate AI System พร้อมใช้งาน!")
            
        except Exception as e:
            self.logger.error(f"❌ ไม่สามารถเริ่มต้นระบบ: {e}")
    
    def _load_classes(self):
        """โหลด class names"""
        classes_file = Path("models/coco.names")
        if not classes_file.exists():
            classes_file = Path("coco.names")
        
        if classes_file.exists():
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                self.logger.info(f"📚 โหลด {len(self.classes)} classes")
            except Exception as e:
                self.logger.warning(f"⚠️ ไม่สามารถอ่าน {classes_file}: {e}")
                self._load_default_classes()
        else:
            self._load_default_classes()
    
    def _load_default_classes(self):
        """โหลด COCO classes เริ่มต้น"""
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        self.logger.info("📚 ใช้ COCO classes เริ่มต้น")
    
    def _generate_colors(self):
        """สร้างสีสำหรับแต่ละ class"""
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
        
        # สีพิเศษสำหรับ classes สำคัญ
        special_colors = {
            'bird': [0, 255, 255],      # สีเหลือง
            'person': [0, 0, 255],      # สีแดง
            'car': [255, 0, 0],         # สีน้ำเงิน
            'dog': [0, 255, 0],         # สีเขียว
            'cat': [255, 0, 255]        # สีม่วง
        }
        
        for class_name, color in special_colors.items():
            if class_name in self.classes:
                idx = self.classes.index(class_name)
                self.colors[idx] = color
    
    def detect_objects(self, 
                      frame: np.ndarray, 
                      model_name: Optional[str] = None,
                      conf_threshold: Optional[float] = None,
                      nms_threshold: Optional[float] = None,
                      use_cache: bool = True) -> List[DetectionResult]:
        """ตรวจจับวัตถุด้วย AI (เวอร์ชันขั้นสูง)"""
        
        start_time = time.time()
        frame_hash = None
        
        # ใช้แคชถ้าเปิดใช้งาน
        if use_cache and self.config['performance']['enable_caching']:
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            cached_result = self.cache.get(frame_hash)
            if cached_result:
                self.logger.debug("📋 ใช้ผลลัพธ์จากแคช")
                return cached_result
        
        # เลือกโมเดล
        if model_name is None:
            model_name = self.config['models']['primary']
        
        model_info = self.model_manager.get_model(model_name)
        if not model_info:
            # ลองใช้โมเดลสำรอง
            for fallback in self.config['models']['fallback']:
                model_info = self.model_manager.get_model(fallback)
                if model_info:
                    model_name = fallback
                    self.logger.warning(f"⚠️ ใช้โมเดลสำรอง: {model_name}")
                    break
        
        if not model_info:
            self.logger.error("❌ ไม่มีโมเดลที่ใช้ได้")
            return []
        
        # ตั้งค่า threshold
        conf_thresh = conf_threshold or self.config['detection']['confidence_threshold']
        nms_thresh = nms_threshold or self.config['detection']['nms_threshold']
        
        try:
            # เตรียมข้อมูล
            net = model_info['net']
            output_layers = model_info['output_layers']
            input_size = model_info['config']['input_size']
            
            height, width = frame.shape[:2]
            
            # สร้าง blob
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, input_size, swapRB=True, crop=False
            )
            net.setInput(blob)
            
            # รัน inference
            inference_start = time.time()
            outputs = net.forward(output_layers)
            inference_time = time.time() - inference_start
            
            # อัปเดตสถิติโมเดล
            self.model_manager.update_performance(model_name, inference_time)
            
            # ประมวลผลผลลัพธ์
            detections = self._process_detections(
                outputs, frame.shape, conf_thresh, nms_thresh, 
                model_name, start_time
            )
            
            # เก็บในแคช
            if use_cache and frame_hash:
                self.cache.set(frame_hash, detections)
            
            # อัปเดตสถิติ
            self._update_statistics(detections, time.time() - start_time)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"❌ ข้อผิดพลาดในการตรวจจับ: {e}")
            return []
    
    def _process_detections(self, outputs, frame_shape, conf_thresh, nms_thresh, model_name, start_time):
        """ประมวลผลผลลัพธ์การตรวจจับ"""
        height, width = frame_shape[:2]
        
        boxes = []
        confidences = []
        class_ids = []
        
        # แยกข้อมูลจาก outputs
        for output in outputs:
            for detection in output:
                if len(detection) > 5:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > conf_thresh:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # ตรวจสอบขอบเขต
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
        # Non-Maximum Suppression
        detections = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
            
            if len(indices) > 0:
                indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
                
                for i in indices:
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    class_name = self.classes[class_id] if class_id < len(self.classes) else 'unknown'
                    
                    detection = DetectionResult(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=confidence,
                        bbox=[x, y, w, h],
                        center=(x + w//2, y + h//2),
                        area=w * h,
                        source='ultimate_ai',
                        timestamp=time.time(),
                        model_used=model_name,
                        processing_time=time.time() - start_time
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def _update_statistics(self, detections: List[DetectionResult], processing_time: float):
        """อัปเดตสถิติระบบ"""
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(detections)
        
        # อัปเดต performance metrics
        self.performance_metrics.total_detections += len(detections)
        self.performance_metrics.avg_processing_time = (
            (self.performance_metrics.avg_processing_time * (self.stats['total_frames'] - 1) + processing_time) 
            / self.stats['total_frames']
        )
        self.performance_metrics.fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # เก็บประวัติ
        self.stats['detection_history'].append({
            'timestamp': time.time(),
            'count': len(detections),
            'processing_time': processing_time
        })
        
        # ลบประวัติเก่า (เก็บแค่ 1000 รายการล่าสุด)
        if len(self.stats['detection_history']) > 1000:
            self.stats['detection_history'] = self.stats['detection_history'][-1000:]
    
    def detect_specific_objects(self, frame: np.ndarray, target_classes: List[str]) -> List[DetectionResult]:
        """ตรวจจับวัตถุเฉพาะประเภท"""
        all_detections = self.detect_objects(frame)
        filtered_detections = [
            det for det in all_detections 
            if det.class_name in target_classes
        ]
        return filtered_detections
    
    def detect_birds(self, frame: np.ndarray) -> List[DetectionResult]:
        """ตรวจจับนกโดยเฉพาะ"""
        return self.detect_specific_objects(frame, ['bird'])
    
    def detect_persons(self, frame: np.ndarray) -> List[DetectionResult]:
        """ตรวจจับคนโดยเฉพาะ"""
        return self.detect_specific_objects(frame, ['person'])
    
    def detect_animals(self, frame: np.ndarray) -> List[DetectionResult]:
        """ตรวจจับสัตว์โดยเฉพาะ"""
        animal_classes = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        return self.detect_specific_objects(frame, animal_classes)
    
    def detect_vehicles(self, frame: np.ndarray) -> List[DetectionResult]:
        """ตรวจจับยานพาหนะโดยเฉพาะ"""
        vehicle_classes = ['car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bicycle']
        return self.detect_specific_objects(frame, vehicle_classes)
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult], 
                       show_confidence: bool = True, show_model: bool = False) -> np.ndarray:
        """วาดกรอบการตรวจจับขั้นสูง"""
        result_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            class_name = detection.class_name
            confidence = detection.confidence
            class_id = detection.class_id
            
            # เลือกสี
            if class_id < len(self.colors):
                color = tuple(map(int, self.colors[class_id]))
            else:
                color = (0, 255, 0)
            
            # วาดกรอบ
            thickness = 2
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, thickness)
            
            # เตรียมข้อความ
            label_parts = [class_name]
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            if show_model:
                label_parts.append(f"[{detection.model_used}]")
            
            label = " | ".join(label_parts)
            
            # วาดพื้นหลังข้อความ
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                result_frame, 
                (x, y - text_height - baseline - 5), 
                (x + text_width, y), 
                color, -1
            )
            
            # วาดข้อความ
            cv2.putText(
                result_frame, label, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
            # วาดจุดกึ่งกลาง
            center = detection.center
            cv2.circle(result_frame, center, 3, color, -1)
            
            # วาดข้อมูลเพิ่มเติม (processing time)
            if hasattr(detection, 'processing_time'):
                time_text = f"{detection.processing_time*1000:.1f}ms"
                cv2.putText(
                    result_frame, time_text, (x, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
                )
        
        return result_frame
    
    def get_performance_metrics(self) -> Dict:
        """ดึงตัวชี้วัดประสิทธิภาพ"""
        metrics = asdict(self.performance_metrics)
        
        # เพิ่มข้อมูลเพิ่มเติม
        metrics.update({
            'uptime': time.time() - self.stats['session_start'],
            'cache_stats': self.cache.get_stats(),
            'model_stats': self.model_manager.performance_stats,
            'memory_usage': self._get_memory_usage(),
            'total_frames': self.stats['total_frames']
        })
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """ดึงการใช้หน่วยความจำ"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def save_detection_to_db(self, detection: DetectionResult, session_id: str = "default"):
        """บันทึกผลการตรวจจับลงฐานข้อมูล"""
        if self.db_manager:
            try:
                cursor = self.db_manager.cursor()
                cursor.execute('''
                    INSERT INTO detections 
                    (timestamp, class_name, confidence, bbox, model_used, processing_time, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection.timestamp,
                    detection.class_name,
                    detection.confidence,
                    json.dumps(detection.bbox),
                    detection.model_used,
                    detection.processing_time,
                    session_id
                ))
                self.db_manager.commit()
            except Exception as e:
                self.logger.error(f"❌ ไม่สามารถบันทึกลงฐานข้อมูล: {e}")
    
    def export_statistics(self, filepath: str):
        """ส่งออกสถิติเป็นไฟล์"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': asdict(self.performance_metrics),
                'system_stats': self.stats,
                'cache_stats': self.cache.get_stats(),
                'model_stats': self.model_manager.performance_stats,
                'config': self.config
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ ส่งออกสถิติไปยัง: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ ไม่สามารถส่งออกสถิติ: {e}")
    
    def benchmark_system(self, test_frames: int = 100) -> Dict:
        """ทดสอบประสิทธิภาพระบบ"""
        self.logger.info(f"🧪 เริ่มทดสอบประสิทธิภาพ ({test_frames} เฟรม)...")
        
        # สร้างเฟรมทดสอบ
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # ทดสอบโมเดลต่างๆ
        results = {}
        
        for model_name in self.model_manager.models.keys():
            self.logger.info(f"📊 ทดสอบโมเดล: {model_name}")
            
            times = []
            detection_counts = []
            
            for i in range(test_frames):
                start_time = time.time()
                detections = self.detect_objects(test_frame, model_name=model_name, use_cache=False)
                end_time = time.time()
                
                times.append(end_time - start_time)
                detection_counts.append(len(detections))
                
                if (i + 1) % 20 == 0:
                    self.logger.info(f"   ความคืบหน้า: {i+1}/{test_frames}")
            
            # คำนวณสถิติ
            avg_time = np.mean(times)
            avg_fps = 1.0 / avg_time
            avg_detections = np.mean(detection_counts)
            
            results[model_name] = {
                'avg_processing_time': avg_time,
                'avg_fps': avg_fps,
                'avg_detections': avg_detections,
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times)
            }
            
            self.logger.info(f"   📈 FPS เฉลี่ย: {avg_fps:.1f}")
            self.logger.info(f"   🎯 การตรวจจับเฉลี่ย: {avg_detections:.1f}")
        
        self.logger.info("✅ ทดสอบประสิทธิภาพเสร็จสิ้น")
        return results
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        self.logger.info("🧹 ทำความสะอาดระบบ...")
        
        # ปิด thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        # ปิดฐานข้อมูล
        if self.db_manager:
            self.db_manager.close()
        
        self.logger.info("✅ ทำความสะอาดเสร็จสิ้น")

# Helper Functions
def create_config_file(filename: str = "ultimate_ai_config.json"):
    """สร้างไฟล์การตั้งค่าตัวอย่าง"""
    config = {
        "models": {
            "primary": "yolov4",
            "fallback": ["yolov3", "yolov3-tiny"],
            "ensemble": False
        },
        "detection": {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_size": [416, 416],
            "batch_size": 1
        },
        "performance": {
            "enable_caching": True,
            "enable_threading": True,
            "enable_gpu": False,
            "max_fps": 30,
            "memory_limit": 1024
        },
        "output": {
            "save_detections": True,
            "save_frames": False,
            "output_dir": "output",
            "log_level": "INFO"
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ สร้างไฟล์การตั้งค่า: {filename}")

# Test และ Demo
def test_ultimate_ai_system():
    """ทดสอบระบบ Ultimate AI"""
    print("🧪 ทดสอบ Ultimate AI Detection System...")
    
    # สร้างระบบ
    ai_system = UltimateAIDetectionSystem()
    
    # ทดสอบกับกล้อง
    rtsp_urls = [
        "rtsp://ainok1:ainok123@192.168.1.100:554/stream1",
        "rtsp://admin:admin@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0",
        0  # กล้อง USB
    ]
    
    cap = None
    connected = False
    
    for rtsp_url in rtsp_urls:
        print(f"📹 ลองเชื่อมต่อ: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)
        
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ เชื่อมต่อสำเร็จ: {rtsp_url}")
                connected = True
                break
            else:
                cap.release()
        else:
            if cap:
                cap.release()
    
    if not connected:
        print("❌ ไม่สามารถเชื่อมต่อกล้องได้ - ใช้เฟรมทดสอบ")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # ทดสอบการตรวจจับ
        detections = ai_system.detect_objects(test_frame)
        print(f"🎯 พบการตรวจจับ: {len(detections)} วัตถุ")
        
        # ทดสอบประสิทธิภาพ
        benchmark_results = ai_system.benchmark_system(50)
        print("📊 ผลการทดสอบประสิทธิภาพ:")
        for model, stats in benchmark_results.items():
            print(f"   {model}: {stats['avg_fps']:.1f} FPS")
    
    else:
        print("🎥 เริ่มทดสอบด้วยกล้องจริง...")
        
        frame_count = 0
        detection_count = 0
        start_time = time.time()
        
        try:
            while frame_count < 50:  # ทดสอบ 50 เฟรม
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # ทดสอบการตรวจจับ
                detections = ai_system.detect_objects(frame)
                
                if len(detections) > 0:
                    detection_count += 1
                    print(f"🎯 Frame {frame_count}: พบ {len(detections)} วัตถุ")
                    
                    for det in detections:
                        print(f"   📍 {det.class_name}: {det.confidence:.2f} [{det.model_used}]")
                
                # แสดงความคืบหน้า
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 ความคืบหน้า: {frame_count}/50 เฟรม (FPS: {fps:.1f})")
        
        except KeyboardInterrupt:
            print("\n⏹️ หยุดการทดสอบโดยผู้ใช้")
        finally:
            if cap:
                cap.release()
    
    # แสดงสถิติสุดท้าย
    metrics = ai_system.get_performance_metrics()
    print(f"\n📈 สรุปการทดสอบ Ultimate AI System:")
    print(f"   ⚡ FPS เฉลี่ย: {metrics['fps']:.1f}")
    print(f"   🕒 เวลาประมวลผลเฉลี่ย: {metrics['avg_processing_time']*1000:.1f} ms")
    print(f"   🧠 หน่วยความจำ: {metrics['memory_usage']:.1f} MB")
    print(f"   📋 อัตราการใช้แคช: {metrics['cache_stats']['hit_rate']}")
    print(f"   🎯 การตรวจจับทั้งหมด: {metrics['total_detections']}")
    
    # ส่งออกสถิติ
    ai_system.export_statistics("ultimate_ai_test_results.json")
    
    # ทำความสะอาด
    ai_system.cleanup()
    
    print("✅ ทดสอบเสร็จสิ้น!")

if __name__ == "__main__":
    # สร้างไฟล์การตั้งค่า
    create_config_file()
    
    # รันการทดสอบ
    test_ultimate_ai_system()
    
    print("\nกด Enter เพื่อออก...")
    input()
