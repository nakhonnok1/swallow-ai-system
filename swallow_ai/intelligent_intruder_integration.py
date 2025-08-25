#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 Ultra Intelligent Intruder Detection System - Production Ready
ระบบตรวจจับสิ่งแปลกปลอมที่เป็น AI Agent จริงๆ พร้อมใช้งาน 100%
เชื่อมต่อกับ Ultimate AI System ทั้งหมด
"""

import os
import sys
import cv2
import numpy as np
import json
import time
import sqlite3
import threading
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import base64

# Flask imports with error handling
try:
    from flask import Flask, jsonify, request, Response
except ImportError:
    Flask = None
    print("⚠️ Flask not available")

# For type hinting only
if TYPE_CHECKING:
    from flask import Flask

# AI/ML Libraries
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available, using backup detection")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """ระดับความเสี่ยง"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionPriority(Enum):
    """ลำดับความสำคัญ"""
    NORMAL = 1
    ELEVATED = 2
    HIGH = 3
    URGENT = 4
    EMERGENCY = 5

class IntruderStats:
    """Global intruder detection statistics"""
    
    def __init__(self):
        self.total_intruders = 0
        self.daily_intruders = 0
        self.last_detection = None
        self.detection_count_today = 0
        self.reset_daily_stats()
    
    def reset_daily_stats(self):
        """Reset daily statistics at midnight"""
        import datetime as dt
        current_date = dt.datetime.now().date()
        if not hasattr(self, 'last_date') or self.last_date != current_date:
            self.daily_intruders = 0
            self.detection_count_today = 0
            self.last_date = current_date
    
    def add_detection(self, count: int = 1):
        """Add intruder detection"""
        import datetime as dt
        self.reset_daily_stats()
        self.total_intruders += count
        self.daily_intruders += count
        self.detection_count_today += 1
        self.last_detection = dt.datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        self.reset_daily_stats()
        return {
            'total_intruders': self.total_intruders,
            'daily_intruders': self.daily_intruders,
            'detection_count_today': self.detection_count_today,
            'last_detection': self.last_detection.isoformat() if self.last_detection else None
        }

@dataclass
class IntruderDetection:
    """ข้อมูลการตรวจจับสิ่งแปลกปลอม"""
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    threat_level: ThreatLevel
    priority: DetectionPriority
    timestamp: str
    camera_id: str
    description: str

class UltraIntelligentIntruderDetector:
    """🤖 AI Agent ตรวจจับสิ่งแปลกปลอมที่ชาญฉลาด"""
    
    def __init__(self):
        """เริ่มต้นระบบ AI Agent"""
        print("🧠 เริ่มต้น Ultra Intelligent Intruder Detector...")

        # Core Settings - ปรับแต่งให้เป็น AI เอเจนต์ที่ดีที่สุด
        self.confidence_threshold = 0.45  # เพิ่ม threshold เพื่อลด false positive
        self.detection_interval = 3  # ตรวจจับบ่อยขึ้นเพื่อความแม่นยำ
        self.frame_count = 0
        
        # ขนาดขั้นต่ำสำหรับการจำแนกประเภท (pixel area)
        self.min_large_bird_size = 5000  # นกขนาดใหญ่ต้องมีขนาดอย่างน้อย 5000 pixels
        self.min_small_object_size = 1000  # วัตถุเล็กๆ ที่จะไม่สนใจ

        # นกนางแอ่น - ไม่ใช่สิ่งแปลกปลอม (รายการนกที่อนุญาต)
        self.allowed_birds = [
            'swallow', 'martin', 'hirundo', 'delichon',  # นกนางแอ่นทุกชนิด
            'small_bird', 'tiny_bird', 'sparrow', 'finch'  # นกเล็กๆ ทั่วไป
        ]
        
        # นกขนาดใหญ่ที่เป็นภัยคุกคาม
        self.large_predator_birds = [
            'falcon', 'eagle', 'hawk', 'kite', 'buzzard',  # นกล่าสัตว์
            'owl', 'barn_owl', 'horned_owl',  # นกฮูก
            'crow', 'raven', 'magpie',  # นกอีกา (ขนาดใหญ่)
        ]

        # 🎯 Label mapping ที่ปรับปรุงแล้ว - แยกแยะนกนางแอ่นอย่างชาญฉลาด (ไม่รวมยานพาหนะ)
        self.label_alias = {
            # 🐦 นกนางแอ่นและนกเล็ก (ไม่ใช่สิ่งแปลกปลอม - เป็นเป้าหมายที่ต้องปกป้อง)
            'swallow': ['swallow', 'martin', 'hirundo', 'barn_swallow', 'house_martin', 'delichon'],
            'small_bird': ['sparrow', 'finch', 'wren', 'robin', 'tit', 'tiny_bird', 'small_passerine'],
            
            # 🦅 นกขนาดใหญ่ที่เป็นภัยคุกคาม (สิ่งแปลกปลอม - ต้องมีขนาดใหญ่อย่างน้อย 5000 pixels)
            'large_predator': ['falcon', 'eagle', 'hawk', 'kite', 'buzzard', 'owl', 'barn_owl', 'horned_owl'],
            'large_bird': ['crow', 'raven', 'magpie', 'pigeon', 'dove', 'large_corvid'],
            
            # 🐾 สัตว์เลี้ยงลูกด้วยนมที่เป็นภัยคุกคาม (สิ่งแปลกปลอม)
            'mammal_predator': ['cat', 'dog', 'fox', 'weasel', 'rat', 'mouse', 'ferret', 'marten'],
            
            # 🐍 สัตว์เลื้อยคลาน (สิ่งแปลกปลอมอันตราย)
            'reptile': ['snake', 'python', 'cobra', 'lizard', 'gecko', 'monitor', 'viper'],
            
            # 👤 มนุษย์ (สิ่งแปลกปลอมระดับวิกฤต)
            'human': ['person', 'man', 'woman', 'child', 'people', 'human']
        }
        
        # 🚫 รายการที่ไม่ถือเป็นสิ่งแปลกปลอม (ยานพาหนะและวัตถุอื่นๆ ที่ไม่เป็นภัย)
        self.non_intruder_objects = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle',
            'airplane', 'boat', 'train', 'ship',
            'umbrella', 'bag', 'suitcase', 'chair', 'table',
            'bench', 'bottle', 'cup', 'phone', 'laptop'
        ]

        # 🎯 การจำแนกสิ่งแปลกปลอม - AI Agent ที่ชาญฉลาดที่สุด (ไม่รวมยานพาหนะและนกนางแอ่น)
        self.threat_objects = {
            # 👥 มนุษย์ - ภัยคุกคามวิกฤตระดับสูงสุด 
            'person': {'threat': ThreatLevel.CRITICAL, 'priority': DetectionPriority.EMERGENCY, 'is_intruder': True},
            'human': {'threat': ThreatLevel.CRITICAL, 'priority': DetectionPriority.EMERGENCY, 'is_intruder': True},
            
            # 🐱🐶 สัตว์เลี้ยงลูกด้วยนม - ภัยคุกคามสูง (ล่าสัตว์และรบกวนนก)
            'cat': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.URGENT, 'is_intruder': True},
            'dog': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.URGENT, 'is_intruder': True},
            'fox': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.URGENT, 'is_intruder': True},
            'weasel': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.URGENT, 'is_intruder': True},
            
            # 🐭 สัตว์ฟันแทะ - ภัยคุกคามปานกลาง (กินไข่และลูกนก)
            'rat': {'threat': ThreatLevel.MEDIUM, 'priority': DetectionPriority.HIGH, 'is_intruder': True},
            'mouse': {'threat': ThreatLevel.MEDIUM, 'priority': DetectionPriority.HIGH, 'is_intruder': True},
            
            # 🐍 สัตว์เลื้อยคลาน - ภัยคุกคามวิกฤต (อันตรายมากต่อไข่และลูกนก)
            'snake': {'threat': ThreatLevel.CRITICAL, 'priority': DetectionPriority.EMERGENCY, 'is_intruder': True},
            'python': {'threat': ThreatLevel.CRITICAL, 'priority': DetectionPriority.EMERGENCY, 'is_intruder': True},
            'cobra': {'threat': ThreatLevel.CRITICAL, 'priority': DetectionPriority.EMERGENCY, 'is_intruder': True},
            'lizard': {'threat': ThreatLevel.MEDIUM, 'priority': DetectionPriority.HIGH, 'is_intruder': True},
            'gecko': {'threat': ThreatLevel.LOW, 'priority': DetectionPriority.NORMAL, 'is_intruder': False},  # ตุกแกไม่เป็นภัย
            'monitor': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.HIGH, 'is_intruder': True},
            
            # 🦅 นกขนาดใหญ่ที่เป็นภัยคุกคาม (ต้องมีขนาดใหญ่อย่างน้อย 5000 pixels)
            'falcon': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.URGENT, 'is_intruder': True, 'min_size': 5000},
            'eagle': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.URGENT, 'is_intruder': True, 'min_size': 8000},
            'hawk': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.URGENT, 'is_intruder': True, 'min_size': 4000},
            'owl': {'threat': ThreatLevel.HIGH, 'priority': DetectionPriority.URGENT, 'is_intruder': True, 'min_size': 3000},
            'crow': {'threat': ThreatLevel.MEDIUM, 'priority': DetectionPriority.HIGH, 'is_intruder': True, 'min_size': 4000},
            'raven': {'threat': ThreatLevel.MEDIUM, 'priority': DetectionPriority.HIGH, 'is_intruder': True, 'min_size': 5000},
            'magpie': {'threat': ThreatLevel.MEDIUM, 'priority': DetectionPriority.HIGH, 'is_intruder': True, 'min_size': 3000},
            
            # 🐦 นกนางแอ่นและนกเล็ก - เป็นเป้าหมายที่ต้องปกป้อง (ไม่ใช่สิ่งแปลกปลอม!)
            'swallow': {'threat': ThreatLevel.LOW, 'priority': DetectionPriority.NORMAL, 'is_target_species': True, 'is_intruder': False},
            'small_bird': {'threat': ThreatLevel.LOW, 'priority': DetectionPriority.NORMAL, 'is_target_species': True, 'is_intruder': False},
        }

        # Initialize AI Models
        self._initialize_models()

        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'threat_alerts': 0,
            'false_positives': 0,
            'accuracy_score': 0.0
        }

        print("✅ Ultra Intelligent Intruder Detector พร้อมใช้งาน!")
    
    def _summarize_threats(self, detections: List[IntruderDetection]) -> Dict[str, Any]:
        """สรุปภัยคุกคาม - ฟังก์ชันใหม่สำหรับ app_working.py"""
        summary = {
            'total': len(detections),
            'by_threat_level': {},
            'by_object_type': {},
            'highest_threat': None,
            'avg_confidence': 0.0
        }
        
        if not detections:
            return summary
        
        # นับตาม threat level
        for detection in detections:
            level = detection.threat_level.value
            summary['by_threat_level'][level] = summary['by_threat_level'].get(level, 0) + 1
            
            obj_type = detection.object_type
            summary['by_object_type'][obj_type] = summary['by_object_type'].get(obj_type, 0) + 1
        
        # หา threat level สูงสุด
        threat_priorities = {
            ThreatLevel.CRITICAL: 4,
            ThreatLevel.HIGH: 3,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.LOW: 1
        }
        
        highest_priority = 0
        for detection in detections:
            priority = threat_priorities.get(detection.threat_level, 0)
            if priority > highest_priority:
                highest_priority = priority
                summary['highest_threat'] = detection.threat_level.value
        
        # คำนวณ confidence เฉลี่ย
        if detections:
            summary['avg_confidence'] = sum(d.confidence for d in detections) / len(detections)
        
        return summary
        
    def detect_objects(self, frame: np.ndarray, camera_id: str = "default", camera_props: dict = None) -> List[IntruderDetection]:
        """🔍 ตรวจจับวัตถุและสิ่งแปลกปลอมอย่างชาญฉลาด (optimized for low CPU)"""
        self.frame_count += 1
        # ตรวจจับทุก N เฟรมเท่านั้น
        if self.frame_count % self.detection_interval != 0:
            return []

        detections = []
        current_time = datetime.now().isoformat()

        # ดึงความสามารถของกล้อง (resolution, fps, etc.)
        camera_info = {}
        if camera_props is not None:
            camera_info = camera_props
        else:
            try:
                if hasattr(frame, 'shape'):
                    h, w = frame.shape[:2]
                    camera_info['resolution'] = f"{w}x{h}"
            except Exception:
                camera_info['resolution'] = 'unknown'

        try:
            # YOLO Detection (Primary)
            if 'yolo' in self.models:
                yolo_detections = self._yolo_detection(frame, camera_id, current_time)
                detections.extend(yolo_detections)

            # Backup Detection System (motion)
            if not detections:
                backup_detections = self._backup_detection(frame, camera_id, current_time)
                detections.extend(backup_detections)

            # AI Analysis & Threat Assessment
            analyzed_detections = self._ai_threat_analysis(detections, frame)

            # Update statistics
            self._update_statistics(analyzed_detections)

            # เพิ่มข้อมูลกล้องใน description
            for d in analyzed_detections:
                d.description += f" | Camera: {camera_id} | {camera_info.get('resolution', '')}"

            return analyzed_detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _initialize_models(self):
        """เริ่มต้น AI Models - เชื่อมต่อกับ Ultimate AI System"""
        self.models = {}
        
        # Ultimate AI Vision System Integration
        try:
            print("🚀 โหลด Ultimate AI Vision System...")
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            
            # พยายามโหลด Ultimate AI Vision System ก่อน
            try:
                from opencv_yolo_detector import OpenCVYOLODetector
                self.ultimate_ai_vision = OpenCVYOLODetector()
                if self.ultimate_ai_vision.available:
                    print("✅ Ultimate AI Vision System โหลดสำเร็จ")
                    self.models['ultimate_ai'] = self.ultimate_ai_vision
                    self.models['yolo'] = self.ultimate_ai_vision  # สำหรับ backward compatibility
                else:
                    print("⚠️ Ultimate AI Vision ไม่พร้อม")
                    self.models['ultimate_ai'] = None
                    self.models['yolo'] = None
            except Exception as e:
                print(f"⚠️ Ultimate AI Vision Error: {e}")
                self.models['ultimate_ai'] = None
                self.models['yolo'] = None
            
            # Fallback: ลอง Simple AI Detector
            if self.models['ultimate_ai'] is None:
                try:
                    from simple_ai_detector import SimpleYOLODetector
                    self.simple_ai = SimpleYOLODetector()
                    print("✅ Simple AI Detector โหลดเป็น fallback")
                    self.models['fallback_ai'] = self.simple_ai
                except Exception as e:
                    print(f"⚠️ Simple AI Detector Error: {e}")
                    self.models['fallback_ai'] = None
                
        except Exception as e:
            print(f"❌ ไม่สามารถโหลด AI System ใดๆ: {e}")
            self.models['ultimate_ai'] = None
            self.models['yolo'] = None
            self.models['fallback_ai'] = None
        
        # MediaPipe (for person detection)
        if MEDIAPIPE_AVAILABLE:
            try:
                import mediapipe as mp
                self.models['mediapipe'] = mp.solutions.objectron
                print("✅ MediaPipe Model โหลดสำเร็จ")
            except Exception as e:
                print(f"❌ Error loading MediaPipe: {e}")
        
        # Enhanced Ultra Smart AI Agent Integration
        try:
            from enhanced_ultra_smart_ai_agent import EnhancedUltraSmartAIAgent
            self.ai_chatbot = EnhancedUltraSmartAIAgent()
            self.models['ai_chatbot'] = self.ai_chatbot
            print("✅ Enhanced Ultra Smart AI Agent โหลดสำเร็จ")
        except Exception as e:
            print(f"⚠️ Enhanced AI Chatbot ไม่พร้อม: {e}")
            self.models['ai_chatbot'] = None
        
        # Ultimate Swallow AI Agent Integration
        try:
            from ultimate_perfect_ai_MASTER import UltimateSwallowAIAgent
            self.ultimate_swallow_ai = UltimateSwallowAIAgent(video_type="mixed")
            self.models['ultimate_swallow_ai'] = self.ultimate_swallow_ai
            print("✅ Ultimate Swallow AI Agent โหลดสำเร็จ")
        except Exception as e:
            print(f"⚠️ Ultimate Swallow AI ไม่พร้อม: {e}")
            self.models['ultimate_swallow_ai'] = None
        
        # Backup detection system
        self.models['backup'] = True
        print("✅ Backup Detection System พร้อมใช้งาน")
    
    def _yolo_detection(self, frame: np.ndarray, camera_id: str, timestamp: str) -> List[IntruderDetection]:
        """Ultimate AI-based detection - ใช้ Ultimate AI Vision System พร้อมการแยกแยะนกนางแอ่น"""
        detections = []
        
        # ลำดับความสำคัญ: Ultimate AI > Fallback AI > None
        ai_detector = None
        detector_name = "none"
        
        if self.models.get('ultimate_ai') is not None:
            ai_detector = self.models['ultimate_ai']
            detector_name = "Ultimate AI Vision"
        elif self.models.get('fallback_ai') is not None:
            ai_detector = self.models['fallback_ai']
            detector_name = "Fallback AI"
        elif self.models.get('yolo') is not None:
            ai_detector = self.models['yolo']
            detector_name = "OpenCV YOLO"
        
        if ai_detector is None:
            print("⚠️ ไม่มี AI detector ที่พร้อมใช้งาน")
            return detections
            
        try:
            # ใช้ Ultimate AI Vision System
            print(f"🔍 ใช้ {detector_name} สำหรับการตรวจจับ...")
            ai_detections = ai_detector.detect_objects(frame, conf_threshold=self.confidence_threshold)
            
            for det in ai_detections:
                class_name = det['class']
                confidence = det['confidence']
                x, y, w, h = det['bbox']
                center = det['center']
                
                # คำนวณขนาดของวัตถุ
                object_area = w * h
                
                # กรองวัตถุที่ไม่ใช่สิ่งแปลกปลอม (ยานพาหนะ, วัตถุทั่วไป)
                if not self._filter_non_intruders(class_name, object_area, confidence):
                    continue  # ไม่ใช่สิ่งแปลกปลอม
                
                # กรองวัตถุเล็กๆ ที่ไม่สำคัญ
                if object_area < self.min_small_object_size:
                    continue
                
                # Map label alias สำหรับการจำแนกประเภท - AI ที่ชาญฉลาด
                original_class = class_name
                mapped_class = class_name
                
                # จำแนกประเภทด้วย AI ที่ชาญฉลาด
                for main_label, aliases in self.label_alias.items():
                    if class_name.lower() in [alias.lower() for alias in aliases]:
                        mapped_class = main_label
                        break
                
                # 🐦 ตรวจสอบว่าเป็นนกนางแอ่นหรือไม่ (AI Agent ที่ชาญฉลาดที่สุด)
                is_swallow = self._is_swallow_detection(mapped_class, original_class, object_area, confidence)
                
                if is_swallow:
                    # นกนางแอ่น - ไม่ใช่สิ่งแปลกปลอม แต่บันทึกไว้เพื่อการติดตาม
                    print(f"🐦✅ นกนางแอ่นตรวจพบ: {original_class} (confidence: {confidence:.2%}, ขนาด: {object_area}px)")
                    continue  # ไม่เพิ่มในรายการสิ่งแปลกปลอม
                
                # 🦅 ตรวจสอบนกขนาดใหญ่ที่เป็นภัยคุกคาม
                if mapped_class in ['large_predator', 'large_bird']:
                    if not self._is_large_bird_threat(mapped_class, original_class, object_area, confidence):
                        continue  # ไม่เป็นภัยคุกคาม (ขนาดเล็กเกินไป)
                
                # ตรวจสอบว่าเป็นสิ่งแปลกปลอมหรือไม่
                if mapped_class in self.threat_objects:
                    threat_info = self.threat_objects[mapped_class]
                    
                    # ถ้าเป็นพันธุ์เป้าหมาย (นกนางแอ่น) ให้ข้าม
                    if threat_info.get('is_target_species', False):
                        print(f"🎯 พันธุ์เป้าหมาย: {original_class} - ไม่ใช่สิ่งแปลกปลอม")
                        continue
                        
                else:
                    # สำหรับวัตถุที่ไม่รู้จัก - ให้ระวัง
                    threat_info = {
                        'threat': ThreatLevel.MEDIUM, 
                        'priority': DetectionPriority.ELEVATED
                    }
                    print(f"❓ วัตถุไม่ทราบประเภท: {original_class}")
                
                # สร้าง detection สำหรับสิ่งแปลกปลอม
                detection = IntruderDetection(
                    object_type=mapped_class,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    center=center,
                    threat_level=threat_info['threat'],
                    priority=threat_info['priority'],
                    timestamp=timestamp,
                    camera_id=camera_id,
                    description=f"{detector_name} detected {original_class} → {mapped_class} (size: {object_area}px, conf: {confidence:.2%})"
                )
                detections.append(detection)
                
                # Log การตรวจจับสิ่งแปลกปลอม
                if threat_info['threat'] in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    print(f"🚨 INTRUDER ALERT: {mapped_class} ({original_class}) detected with {confidence:.2%} confidence")
                elif threat_info['threat'] == ThreatLevel.MEDIUM:
                    print(f"⚠️ MEDIUM THREAT: {mapped_class} ({original_class}) detected")
                
        except Exception as e:
            logger.error(f"{detector_name} detection error: {e}")
            print(f"❌ {detector_name} เกิดข้อผิดพลาด: {e}")
            
        return detections
    
    def _is_swallow_detection(self, mapped_class: str, original_class: str, object_area: int, confidence: float) -> bool:
        """🐦 ตรวจสอบว่าเป็นนกนางแอ่นหรือไม่ (AI ที่ชาญฉลาดที่สุด)"""
        
        # 1. ตรวจสอบจาก mapped class - นกนางแอ่นและนกเล็กไม่ใช่สิ่งแปลกปลอม
        if mapped_class in ['swallow', 'small_bird']:
            print(f"✅ ระบุเป็นนกนางแอ่น/นกเล็ก: {mapped_class}")
            return True
        
        # 2. ตรวจสอบจาก original class - คำสำคัญนกนางแอ่น
        swallow_keywords = [
            'swallow', 'martin', 'hirundo', 'delichon', 'barn_swallow', 
            'house_martin', 'red_rumped_swallow', 'cliff_swallow'
        ]
        if any(keyword in original_class.lower() for keyword in swallow_keywords):
            print(f"✅ ตรวจพบนกนางแอ่นจากคำสำคัญ: {original_class}")
            return True
        
        # 3. ตรวจสอบขนาด - นกนางแอ่นมีขนาดเล็กถึงปานกลาง (< 3000 pixels)
        if mapped_class == 'bird' and object_area < 3000:
            if confidence > 0.6:
                print(f"🤔 นกขนาดเล็ก (น่าจะเป็นนกนางแอ่น): {original_class}, ขนาด: {object_area}")
                return True
        
        # 4. นกเล็กๆ ทั่วไป - ถือเป็นกลุ่มเดียวกับนกนางแอ่น
        small_bird_keywords = [
            'sparrow', 'finch', 'wren', 'robin', 'tit', 'small', 'tiny',
            'warbler', 'flycatcher', 'chat', 'pipit'
        ]
        if any(keyword in original_class.lower() for keyword in small_bird_keywords):
            print(f"✅ ตรวจพบนกเล็ก (ไม่ใช่สิ่งแปลกปลอม): {original_class}")
            return True
        
        # 5. ตรวจสอบ confidence + ขนาด - กรณีไม่แน่ใจ
        if mapped_class == 'bird':
            # ถ้า confidence สูงแต่ขนาดเล็ก = น่าจะเป็นนกนางแอ่น
            if confidence > 0.7 and object_area < 4000:
                print(f"🎯 นกขนาดเล็ก confidence สูง (น่าจะเป็นนกนางแอ่น): {confidence:.2f}")
                return True
            # ถ้า confidence ต่ำแต่ขนาดเล็กมาก = อาจเป็นนกนางแอ่นไกลๆ
            if confidence > 0.4 and object_area < 2000:
                print(f"🔍 นกขนาดเล็กมาก (อาจเป็นนกนางแอ่นไกล): ขนาด {object_area}")
                return True
        
        # 6. ตรวจสอบว่าไม่ใช่นกขนาดใหญ่ที่เป็นภัยคุกคาม
        large_bird_threats = [
            'falcon', 'eagle', 'hawk', 'kite', 'buzzard', 'owl', 
            'crow', 'raven', 'magpie', 'large', 'big'
        ]
        if any(keyword in original_class.lower() for keyword in large_bird_threats):
            # ตรวจสอบขนาดเพิ่มเติม - ถ้าขนาดเล็กอาจเป็น false detection
            if object_area < self.min_large_bird_size:
                print(f"🚫 นกขนาดใหญ่แต่ขนาดเล็ก (น่าจะเป็น false detection): {original_class}")
                return True  # ขนาดเล็กเกินไป น่าจะเป็นนกนางแอ่น
        
        return False
    
    def _is_large_bird_threat(self, mapped_class: str, original_class: str, object_area: int, confidence: float) -> bool:
        """🦅 ตรวจสอบว่าเป็นนกขนาดใหญ่ที่เป็นภัยคุกคามจริงหรือไม่"""
        
        # 1. ต้องมีขนาดใหญ่พอที่จะเป็นภัยคุกคาม
        if object_area < self.min_large_bird_size:
            print(f"🚫 นกขนาดเล็กเกินไป ไม่เป็นภัย: {original_class}, ขนาด: {object_area}")
            return False  # ขนาดเล็กเกินไป ไม่เป็นภัย
        
        # 2. ตรวจสอบจาก threat_objects ที่กำหนด min_size
        threat_info = self.threat_objects.get(mapped_class, {})
        min_required_size = threat_info.get('min_size', self.min_large_bird_size)
        
        if object_area < min_required_size:
            print(f"🔍 ขนาดไม่ถึงขั้นต่ำสำหรับ {mapped_class}: {object_area} < {min_required_size}")
            return False
        
        # 3. ต้องมี confidence ที่เพียงพอ
        if confidence < self.confidence_threshold:
            print(f"⚠️ Confidence ต่ำเกินไป: {confidence:.2f} < {self.confidence_threshold}")
            return False
        
        # 4. ตรวจสอบคำสำคัญนกล่าสัตว์
        predator_keywords = [
            'falcon', 'eagle', 'hawk', 'kite', 'buzzard', 'owl', 
            'barn_owl', 'horned_owl', 'crow', 'raven', 'magpie'
        ]
        
        if mapped_class in ['large_predator', 'large_bird']:
            print(f"🚨 ตรวจพบนกขนาดใหญ่ที่เป็นภัยคุกคาม: {mapped_class}")
            return True
        
        if any(keyword in original_class.lower() for keyword in predator_keywords):
            print(f"🦅 ตรวจพบนกล่าสัตว์: {original_class}")
            return True
        
        return False
    
    def _filter_non_intruders(self, detection_class: str, object_area: int, confidence: float) -> bool:
        """🚫 กรองวัตถุที่ไม่ใช่สิ่งแปลกปลอม (ยานพาหนะ, วัตถุทั่วไป)"""
        
        # 1. ตรวจสอบรายการที่ไม่ใช่สิ่งแปลกปลอม
        if detection_class.lower() in self.non_intruder_objects:
            print(f"🚫 ไม่ใช่สิ่งแปลกปลอม: {detection_class}")
            return False  # ไม่ใช่สิ่งแปลกปลอม
        
        # 2. ยานพาหนะและเครื่องจักร
        vehicle_keywords = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle',
            'airplane', 'boat', 'train', 'ship', 'helicopter'
        ]
        if any(keyword in detection_class.lower() for keyword in vehicle_keywords):
            print(f"🚗 ตรวจพบยานพาหนะ (ไม่ใช่สิ่งแปลกปลอม): {detection_class}")
            return False
        
        # 3. วัตถุในครัวเรือนและเครื่องใช้
        household_keywords = [
            'chair', 'table', 'bench', 'umbrella', 'bag', 'suitcase',
            'bottle', 'cup', 'phone', 'laptop', 'book', 'clock'
        ]
        if any(keyword in detection_class.lower() for keyword in household_keywords):
            print(f"🏠 ตรวจพบเครื่องใช้ในบ้าน (ไม่ใช่สิ่งแปลกปลอม): {detection_class}")
            return False
        
        # 4. วัตถุที่มีขนาดเล็กเกินไปที่ไม่น่าจะเป็นภัย
        if object_area < self.min_small_object_size:
            print(f"🔍 วัตถุขนาดเล็กเกินไป: {detection_class}, ขนาด: {object_area}")
            return False
        
        print(f"✅ ระบุเป็นสิ่งแปลกปลอม: {detection_class}")
        return True  # เป็นสิ่งแปลกปลอมที่ต้องสนใจ
    
    def _backup_detection(self, frame: np.ndarray, camera_id: str, timestamp: str) -> List[IntruderDetection]:
        """Backup detection using traditional computer vision"""
        detections = []
        
        try:
            # Simple motion detection or basic CV techniques
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect contours (simplified example)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w//2, y + h//2)
                    
                    detection = IntruderDetection(
                        object_type="unknown_object",
                        confidence=0.6,
                        bbox=(x, y, w, h),
                        center=center,
                        threat_level=ThreatLevel.MEDIUM,
                        priority=DetectionPriority.ELEVATED,
                        timestamp=timestamp,
                        camera_id=camera_id,
                        description="Object detected by backup system"
                    )
                    
                    detections.append(detection)
            
        except Exception as e:
            logger.error(f"Backup detection error: {e}")
        
        return detections
    
    def _ai_threat_analysis(self, detections: List[IntruderDetection], frame: np.ndarray) -> List[IntruderDetection]:
        """🧠 AI-powered threat analysis and enhancement"""
        enhanced_detections = []
        
        for detection in detections:
            # AI Enhancement
            enhanced_detection = self._enhance_detection(detection, frame)
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _enhance_detection(self, detection: IntruderDetection, frame: np.ndarray) -> IntruderDetection:
        """🧠 เพิ่มความแม่นยำของการตรวจจับด้วย AI ที่ชาญฉลาด"""
        
        # คำนวณขนาดวัตถุจากพิกเซล
        x, y, w, h = detection.bbox
        object_area = w * h
        
        # ปรับระดับความเสี่ยงตามขนาดและ confidence
        if detection.confidence > 0.8:
            if detection.threat_level == ThreatLevel.HIGH:
                detection.threat_level = ThreatLevel.CRITICAL
                detection.priority = DetectionPriority.EMERGENCY
        
        # วิเคราะห์เพิ่มเติมสำหรับนกขนาดใหญ่
        if detection.object_type in ['large_predator', 'large_bird']:
            if object_area < self.min_large_bird_size:
                # ลดระดับความเสี่ยงถ้านกขนาดไม่ใหญ่พอ
                if detection.threat_level == ThreatLevel.HIGH:
                    detection.threat_level = ThreatLevel.MEDIUM
                    detection.priority = DetectionPriority.ELEVATED
                detection.description += f" | ขนาดเล็กกว่าที่คาดหวัง ({object_area}px)"
        
        # เพิ่มข้อมูลสำหรับการวิเคราะห์
        threat_descriptions = {
            ThreatLevel.LOW: "ระดับเสี่ยงต่ำ - ไม่เป็นอันตรายต่อนกนางแอ่น",
            ThreatLevel.MEDIUM: "ระดับเสี่ยงปานกลาง - ควรเฝ้าระวัง", 
            ThreatLevel.HIGH: "ระดับเสี่ยงสูง - อาจเป็นภัยต่อนกนางแอ่น",
            ThreatLevel.CRITICAL: "ระดับเสี่ยงวิกฤต - อันตรายต่อนกนางแอ่นมาก"
        }
        
        object_analysis = {
            'person': "บุคคลในพื้นที่ - ตรวจสอบสิทธิ์การเข้าถึง",
            'human': "มนุษย์ในพื้นที่ - อาจรบกวนนกนางแอ่น",
            'cat': "แมว - นักล่าตัวฉกาจที่อันตรายต่อนก",
            'dog': "สุนัข - อาจไล่ล่าหรือรบกวนนก",
            'snake': "งู - อันตรายมากต่อไข่และลูกนก",
            'large_predator': "นกล่าสัตว์ขนาดใหญ่ - ภัยคุกคามสูงต่อนกนางแอ่น",
            'large_bird': "นกขนาดใหญ่ - อาจแย่งที่ทำรังหรือล่าสัตว์",
            'rat': "หนู - อาจกินไข่หรือลูกนก",
            'mouse': "หนูเล็ก - ภัยคุกคามน้อยแต่ควรระวัง"
        }
        
        analysis = object_analysis.get(detection.object_type, "วัตถุไม่ทราบประเภท")
        detection.description += f" | {threat_descriptions[detection.threat_level]} | {analysis}"
        
        return detection
    
    def _update_statistics(self, detections: List[IntruderDetection]):
        """อัพเดทสถิติการตรวจจับ"""
        self.detection_stats['total_detections'] += len(detections)
        
        threat_count = sum(1 for d in detections if d.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL])
        self.detection_stats['threat_alerts'] += threat_count
        
        # Calculate accuracy (simplified)
        if self.detection_stats['total_detections'] > 0:
            self.detection_stats['accuracy_score'] = (
                (self.detection_stats['total_detections'] - self.detection_stats['false_positives']) / 
                self.detection_stats['total_detections']
            )
    
    def analyze_with_ai_chatbot(self, detections: List[IntruderDetection], frame: np.ndarray) -> Dict[str, Any]:
        """ใช้ AI Chatbot วิเคราะห์ผลการตรวจจับ"""
        if self.models.get('ai_chatbot') is None:
            return {'analysis': 'AI Chatbot ไม่พร้อมใช้งาน', 'recommendations': []}
        
        try:
            # สร้างข้อมูลสำหรับ AI Chatbot
            detection_summary = {
                'total_detections': len(detections),
                'threat_levels': {},
                'object_types': {},
                'confidence_scores': []
            }
            
            for detection in detections:
                # นับ threat levels
                threat_level = detection.threat_level.value
                detection_summary['threat_levels'][threat_level] = detection_summary['threat_levels'].get(threat_level, 0) + 1
                
                # นับ object types
                obj_type = detection.object_type
                detection_summary['object_types'][obj_type] = detection_summary['object_types'].get(obj_type, 0) + 1
                
                # เก็บ confidence scores
                detection_summary['confidence_scores'].append(detection.confidence)
            
            # ถาม AI Chatbot เพื่อวิเคราะห์
            analysis_query = f"วิเคราะห์การตรวจจับสิ่งแปลกปลอม: พบ {len(detections)} รายการ"
            if detection_summary['threat_levels']:
                threat_info = ", ".join([f"{k}: {v}" for k, v in detection_summary['threat_levels'].items()])
                analysis_query += f" ระดับความเสี่ยง: {threat_info}"
            
            ai_response = self.models['ai_chatbot'].get_response(analysis_query, context=detection_summary)
            
            return {
                'analysis': ai_response,
                'summary': detection_summary,
                'ai_available': True
            }
            
        except Exception as e:
            return {
                'analysis': f'เกิดข้อผิดพลาดในการวิเคราะห์: {e}',
                'summary': {},
                'ai_available': False
            }
    
    def integrate_with_bird_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        """เชื่อมต่อกับระบบตรวจจับนก Ultimate Swallow AI"""
        if self.models.get('ultimate_swallow_ai') is None:
            return {'bird_stats': 'Ultimate Swallow AI ไม่พร้อมใช้งาน'}
        
        try:
            # ใช้ Ultimate Swallow AI วิเคราะห์เฟรม
            bird_results = self.models['ultimate_swallow_ai'].process_frame_agent(frame)
            
            bird_analysis = {
                'bird_detections': len(bird_results.get('detections', [])),
                'bird_stats': self.models['ultimate_swallow_ai'].get_realtime_stats(),
                'detailed_analytics': self.models['ultimate_swallow_ai'].get_detailed_analytics(),
                'frame_analysis': bird_results
            }
            
            return bird_analysis
            
        except Exception as e:
            return {'bird_stats': f'เกิดข้อผิดพลาดในการวิเคราะห์นก: {e}'}
    
    def get_comprehensive_analysis(self, frame: np.ndarray, camera_id: str = "default") -> Dict[str, Any]:
        """วิเคราะห์ครบถ้วนทั้งสิ่งแปลกปลอมและนก"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'camera_id': camera_id,
            'intruder_detection': {},
            'bird_detection': {},
            'ai_analysis': {},
            'recommendations': []
        }
        
        try:
            # 1. ตรวจจับสิ่งแปลกปลอม
            intruder_detections = self.detect_objects(frame, camera_id)
            results['intruder_detection'] = {
                'detections': [asdict(d) for d in intruder_detections],
                'count': len(intruder_detections),
                'threat_summary': self._summarize_threats(intruder_detections)
            }
            
            # 2. ตรวจจับนก (ถ้ามี Ultimate Swallow AI)
            bird_analysis = self.integrate_with_bird_detection(frame)
            results['bird_detection'] = bird_analysis
            
            # 3. วิเคราะห์ด้วย AI Chatbot
            ai_analysis = self.analyze_with_ai_chatbot(intruder_detections, frame)
            results['ai_analysis'] = ai_analysis
            
            # 4. สรุปและคำแนะนำ
            recommendations = self._generate_recommendations(intruder_detections, bird_analysis)
            results['recommendations'] = recommendations
            
        except Exception as e:
            results['error'] = f'เกิดข้อผิดพลาดในการวิเคราะห์: {e}'
        
        return results
    
    def _summarize_threats(self, detections: List[IntruderDetection]) -> Dict[str, Any]:
        """สรุปภัยคุกคาม"""
        summary = {
            'total': len(detections),
            'by_threat_level': {},
            'by_object_type': {},
            'highest_threat': None,
            'avg_confidence': 0.0
        }
        
        if not detections:
            return summary
        
        # นับตาม threat level
        for detection in detections:
            threat = detection.threat_level.value
            summary['by_threat_level'][threat] = summary['by_threat_level'].get(threat, 0) + 1
            
            obj_type = detection.object_type
            summary['by_object_type'][obj_type] = summary['by_object_type'].get(obj_type, 0) + 1
        
        # หา threat level สูงสุด
        threat_priorities = {
            ThreatLevel.CRITICAL: 4,
            ThreatLevel.HIGH: 3,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.LOW: 1
        }
        
        highest_priority = 0
        for detection in detections:
            priority = threat_priorities.get(detection.threat_level, 0)
            if priority > highest_priority:
                highest_priority = priority
                summary['highest_threat'] = detection.threat_level.value
        
        # คำนวณ confidence เฉลี่ย
        if detections:
            summary['avg_confidence'] = sum(d.confidence for d in detections) / len(detections)
        
        return summary
    
    def _generate_recommendations(self, intruder_detections: List[IntruderDetection], bird_analysis: Dict) -> List[str]:
        """🎯 สร้างคำแนะนำเฉพาะเจาะจงสำหรับการป้องกันนกนางแอ่น"""
        recommendations = []
        
        # วิเคราะห์การตรวจจับสิ่งแปลกปลอม
        if intruder_detections:
            critical_threats = [d for d in intruder_detections if d.threat_level == ThreatLevel.CRITICAL]
            high_threats = [d for d in intruder_detections if d.threat_level == ThreatLevel.HIGH]
            
            # ภัยคุกคามระดับวิกฤต
            if critical_threats:
                for threat in critical_threats:
                    if threat.object_type in ['person', 'human']:
                        recommendations.append(f"🚨 พบบุคคลในพื้นที่ - ตรวจสอบสิทธิ์การเข้าถึงทันที")
                    elif threat.object_type in ['snake', 'python', 'cobra']:
                        recommendations.append(f"� อันตราย! พบงูในพื้นที่ - เสี่ยงต่อไข่และลูกนกนางแอ่น")
                
            # ภัยคุกคามระดับสูง
            if high_threats:
                predator_count = len([d for d in high_threats if d.object_type in ['cat', 'dog', 'fox']])
                if predator_count > 0:
                    recommendations.append(f"🐱 พบสัตว์นักล่า {predator_count} ตัว - อันตรายต่อนกนางแอ่น")
                
                large_bird_count = len([d for d in high_threats if d.object_type in ['large_predator', 'falcon', 'eagle', 'hawk']])
                if large_bird_count > 0:
                    recommendations.append(f"🦅 พบนกล่าสัตว์ขนาดใหญ่ {large_bird_count} ตัว - อาจล่านกนางแอ่น")
            
            # สัตว์ฟันแทะ
            rodent_count = len([d for d in intruder_detections if d.object_type in ['rat', 'mouse']])
            if rodent_count > 0:
                recommendations.append(f"� พบสัตว์ฟันแทะ {rodent_count} ตัว - อาจกินไข่นกนางแอ่น")
                
        # วิเคราะห์ข้อมูลนก
        if 'bird_stats' in bird_analysis and isinstance(bird_analysis['bird_stats'], dict):
            bird_stats = bird_analysis['bird_stats']
            current_birds = bird_stats.get('current_count', 0)
            
            if current_birds > 0:
                if intruder_detections:
                    high_risk_intruders = [d for d in intruder_detections if d.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
                    if high_risk_intruders:
                        recommendations.append(f"⚠️ มีนกนางแอ่น {current_birds} ตัว และสิ่งแปลกปลอม {len(high_risk_intruders)} รายการ - เสี่ยงสูงมาก!")
                    else:
                        recommendations.append(f"🐦 มีนกนางแอ่น {current_birds} ตัว ในพื้นที่ - หลีกเลี่ยงการรบกวน")
                else:
                    recommendations.append(f"✅ มีนกนางแอ่น {current_birds} ตัว และไม่พบภัยคุกคาม - สภาพแวดล้อมดี")
            else:
                if intruder_detections:
                    recommendations.append(f"🚫 ไม่มีนกนางแอ่นแต่มีสิ่งแปลกปลอม - อาจเป็นสาเหตุที่นกหลบหนี")
        
        # คำแนะนำเฉพาะกรณี
        if not intruder_detections and not current_birds:
            recommendations.append("🔍 ไม่พบนกหรือสิ่งแปลกปลอม - ตรวจสอบการตั้งค่ากล้องและ AI")
        
        if not recommendations:
            recommendations.append("✅ สภาพแวดล้อมปลอดภัยสำหรับนกนางแอ่น - ไม่พบภัยคุกคาม")
        
        # เพิ่มคำแนะนำการป้องกันทั่วไป
        if intruder_detections:
            recommendations.append("💡 แนะนำ: ตรวจสอบรั้วและจุดเข้าออก เพิ่มระบบป้องกันตามความจำเป็น")
        
        return recommendations

class IntelligentIntruderIntegration:
    """🔗 ระบบเชื่อมต่อ AI Intruder Detection กับแอพหลัก"""

    def __init__(self, app=None):
        print("🚀 เริ่มต้น Intelligent Intruder Integration System...")
        
        # Core Components
        self.detector = UltraIntelligentIntruderDetector()
        self.app = app
        
        # Camera Management
        self.camera_streams = {}
        self.active_cameras = {}
        self.detection_threads = {}
        
        # Database
        self.db_path = 'intelligent_intruder_detections.db'
        self._initialize_database()
        
        # Notification System
        self.notification_callbacks = []
        
        # Performance Monitoring
        self.performance_stats = {
            'frames_processed': 0,
            'detections_made': 0,
            'alerts_sent': 0,
            'average_fps': 0.0,
            'system_uptime': time.time()
        }
        
        print("✅ Intelligent Intruder Integration ตั้งค่าเสร็จสิ้น")
    
    def _initialize_database(self):
        """เริ่มต้นฐานข้อมูลสำหรับเก็บข้อมูลการตรวจจับ"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ตาราง intruder_detections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intruder_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    threat_level TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_width INTEGER NOT NULL,
                    bbox_height INTEGER NOT NULL,
                    center_x INTEGER NOT NULL,
                    center_y INTEGER NOT NULL,
                    description TEXT,
                    image_data TEXT
                )
            """)
            
            # ตาราง system_performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    frames_per_second REAL,
                    detection_accuracy REAL,
                    alert_response_time REAL,
                    system_load REAL,
                    memory_usage REAL
                )
            """)
            
            conn.commit()
            conn.close()
            print("✅ Integration Database พร้อมใช้งาน")
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการเริ่มต้น Integration Database: {e}")
    
    def setup_flask_integration(self, app):
        """ตั้งค่าการเชื่อมต่อกับ Flask App"""
        self.app = app
        
        # Register API Routes
        self._register_api_routes()
        
        print("✅ Flask Integration ตั้งค่าเสร็จสิ้น")
    
    def process_frame_for_intruders(self, frame: np.ndarray, camera_id: str = "main_camera") -> List[Dict[str, Any]]:
        """Process frame for intruders - NO DUPLICATE CODE - Central method for all intruder detection"""
        if frame is None:
            return []
        
        try:
            # Use UltraIntelligentIntruderDetector directly
            raw_detections = self.detector.detect_objects(frame, camera_id)
            
            # Convert to format expected by app_working.py
            formatted_detections = []
            for detection in raw_detections:
                detection_dict = {
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'class': detection.object_type,
                    'threat_level': detection.threat_level.value if hasattr(detection.threat_level, 'value') else str(detection.threat_level),
                    'priority': detection.priority.value if hasattr(detection.priority, 'value') else str(detection.priority),
                    'type': 'intruder_detection',
                    'description': detection.description,
                    'center': detection.center,
                    'timestamp': detection.timestamp,
                    'camera_id': detection.camera_id
                }
                formatted_detections.append(detection_dict)
            
            # Save to database
            self._save_detections_to_db(raw_detections, frame)
            
            return formatted_detections
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return []
    
    def _register_api_routes(self):
        """ลงทะเบียน API Routes สำหรับ Intruder Detection"""
        
        if self.app is None:
            print("⚠️ Flask app not available")
            return
        
        @self.app.route('/api/intruder/detect', methods=['POST'])
        def api_detect_intruders():
            """API สำหรับตรวจจับสิ่งแปลกปลอมจากภาพ"""
            try:
                if 'image' not in request.files:
                    return jsonify({'success': False, 'error': 'No image provided'})
                
                file = request.files['image']
                camera_id = request.form.get('camera_id', 'api_upload')
                
                # Convert to OpenCV format
                file_bytes = np.frombuffer(file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return jsonify({'success': False, 'error': 'Invalid image format'})
                
                # Perform detection
                detections = self.detector.detect_objects(frame, camera_id)
                
                # Save to database
                self._save_detections_to_db(detections, frame)
                
                # Convert detections to JSON-serializable format
                results = []
                for detection in detections:
                    results.append({
                        'object_type': detection.object_type,
                        'confidence': detection.confidence,
                        'threat_level': detection.threat_level.value,
                        'priority': detection.priority.value,
                        'bbox': detection.bbox,
                        'center': detection.center,
                        'description': detection.description,
                        'timestamp': detection.timestamp
                    })
                
                return jsonify({
                    'success': True,
                    'detections': results,
                    'total_detections': len(results),
                    'threat_count': len([d for d in detections if d.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]])
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/intruder/status', methods=['GET'])
        def api_intruder_status():
            """API สำหรับตรวจสอบสถานะระบบ"""
            try:
                uptime = time.time() - self.performance_stats['system_uptime']
                
                status = {
                    'system_active': True,
                    'detector_ready': self.detector is not None,
                    'active_cameras': len(self.active_cameras),
                    'total_detections': self.detector.detection_stats['total_detections'],
                    'threat_alerts': self.detector.detection_stats['threat_alerts'],
                    'accuracy_score': self.detector.detection_stats['accuracy_score'],
                    'uptime_seconds': uptime,
                    'performance': self.performance_stats
                }
                
                return jsonify({'success': True, 'status': status})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/intruder/cameras', methods=['GET'])
        def api_list_cameras():
            """API สำหรับดูรายการกล้องที่เชื่อมต่อ"""
            try:
                cameras = []
                for camera_id, info in self.active_cameras.items():
                    cameras.append({
                        'camera_id': camera_id,
                        'location': info.get('location', 'Unknown'),
                        'status': info.get('status', 'active'),
                        'last_detection': info.get('last_detection', None)
                    })
                
                return jsonify({
                    'success': True,
                    'cameras': cameras,
                    'total_cameras': len(cameras)
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        print("✅ API Routes ลงทะเบียนเสร็จสิ้น")
        
        # Enhanced API Routes for AI Integration
        @self.app.route('/api/intruder/comprehensive-analysis', methods=['POST'])
        def api_comprehensive_analysis():
            """API สำหรับการวิเคราะห์ครบถ้วน (รวมนกและสิ่งแปลกปลอม)"""
            try:
                if 'image' not in request.files:
                    return jsonify({'success': False, 'error': 'No image provided'})
                
                file = request.files['image']
                camera_id = request.form.get('camera_id', 'api_analysis')
                
                # Convert to OpenCV format
                file_bytes = np.frombuffer(file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return jsonify({'success': False, 'error': 'Invalid image format'})
                
                # Perform comprehensive analysis
                analysis = self.detector.get_comprehensive_analysis(frame, camera_id)
                
                return jsonify({
                    'success': True,
                    'analysis': analysis
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/intruder/ai-analysis', methods=['POST'])
        def api_ai_analysis():
            """API สำหรับการวิเคราะห์ด้วย AI Chatbot"""
            try:
                data = request.get_json()
                query = data.get('query', 'วิเคราะห์สถานการณ์ปัจจุบัน')
                
                if 'image' in request.files:
                    file = request.files['image']
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        detections = self.detector.detect_objects(frame, 'ai_analysis')
                        ai_analysis = self.detector.analyze_with_ai_chatbot(detections, frame)
                        return jsonify({
                            'success': True,
                            'ai_analysis': ai_analysis
                        })
                
                return jsonify({'success': False, 'error': 'No valid image provided'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/intruder/bird-integration', methods=['POST'])
        def api_bird_integration():
            """API สำหรับเชื่อมต่อกับระบบตรวจจับนก"""
            try:
                if 'image' not in request.files:
                    return jsonify({'success': False, 'error': 'No image provided'})
                
                file = request.files['image']
                file_bytes = np.frombuffer(file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return jsonify({'success': False, 'error': 'Invalid image format'})
                
                # Perform bird analysis
                bird_analysis = self.detector.integrate_with_bird_detection(frame)
                
                return jsonify({
                    'success': True,
                    'bird_analysis': bird_analysis
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/intruder/ai-models', methods=['GET'])
        def api_ai_models_status():
            """API สำหรับตรวจสอบสถานะ AI Models"""
            try:
                models_status = {}
                for model_name, model_instance in self.detector.models.items():
                    if model_instance is not None:
                        if hasattr(model_instance, 'available'):
                            models_status[model_name] = {
                                'loaded': True,
                                'available': model_instance.available,
                                'type': type(model_instance).__name__
                            }
                        else:
                            models_status[model_name] = {
                                'loaded': True,
                                'available': True,
                                'type': type(model_instance).__name__
                            }
                    else:
                        models_status[model_name] = {
                            'loaded': False,
                            'available': False,
                            'type': None
                        }
                
                return jsonify({
                    'success': True,
                    'models': models_status,
                    'total_models': len(models_status),
                    'active_models': len([m for m in models_status.values() if m['loaded']])
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        print("✅ Enhanced AI Integration API Routes ลงทะเบียนเสร็จสิ้น")
    
    def add_camera_stream(self, camera_id: str, camera_url: str, location: str = "Unknown", username: str = None, password: str = None) -> bool:
        """เพิ่มกล้องเข้าระบบ (debug RTSP connection)"""
        try:
            if camera_id in self.active_cameras:
                print(f"⚠️ Camera {camera_id} already exists")
                return False

            # Build RTSP URL with credentials if provided
            url = camera_url
            if username and password and camera_url.startswith("rtsp://"):
                # Insert credentials into RTSP URL if not already present
                if "@" not in camera_url:
                    url = camera_url.replace("rtsp://", f"rtsp://{username}:{password}@")
            print(f"[DEBUG] Connecting to camera: {url}")

            # Test camera connection
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print(f"❌ Cannot connect to camera {camera_id}: {url}")
                return False
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"❌ Camera {camera_id} connected but no video stream (check credentials, stream path, firewall, or camera config)")
                cap.release()
                return False
            cap.release()

            # Add to active cameras
            self.active_cameras[camera_id] = {
                'url': url,
                'location': location,
                'status': 'active',
                'last_detection': None,
                'detection_count': 0
            }
            print(f"✅ Camera {camera_id} added successfully")
            return True
        except Exception as e:
            print(f"❌ Error adding camera {camera_id}: {e}")
            return False
        def add_camera_stream(self, camera_id: str, camera_url: str, location: str = "Unknown") -> bool:
            """เพิ่มกล้องเข้าระบบ (พร้อม auto-reconnect)"""
            try:
                if camera_id in self.active_cameras:
                    print(f"⚠️ Camera {camera_id} already exists")
                    return False
                # Test camera connection
                cap = cv2.VideoCapture(camera_url)
                if not cap.isOpened():
                    print(f"❌ Cannot connect to camera {camera_id}: {camera_url}")
                    return False
                cap.release()
                self.active_cameras[camera_id] = {
                    'url': camera_url,
                    'location': location,
                    'status': 'active',
                    'last_detection': None,
                    'detection_count': 0,
                    'reconnect_attempts': 0
                }
                print(f"✅ Camera {camera_id} added successfully")
                return True
            except Exception as e:
                print(f"❌ Error adding camera {camera_id}: {e}")
                return False
    
    def start_camera_monitoring(self, camera_id: str, detection_interval: float = 1.0):
        """เริ่มการตรวจสอบกล้องแบบ real-time"""
        if camera_id not in self.active_cameras:
            print(f"❌ Camera {camera_id} not found")
            return False
        
        def monitor_camera():
            camera_info = self.active_cameras[camera_id]
            cap = cv2.VideoCapture(camera_info['url'])
            
            while camera_info.get('status') == 'active':
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"⚠️ Failed to read from camera {camera_id}")
                        break
                    
                    # Perform detection
                    detections = self.detector.detect_objects(frame, camera_id)
                    
                    if detections:
                        # Update camera info
                        camera_info['last_detection'] = datetime.now().isoformat()
                        camera_info['detection_count'] += len(detections)
                        
                        # Save to database
                        self._save_detections_to_db(detections, frame)
                        
                        # Send notifications
                        self._send_notifications(detections, camera_id)
                    
                    # Update performance stats
                    self.performance_stats['frames_processed'] += 1
                    
                    time.sleep(detection_interval)
                    
                except Exception as e:
                    print(f"❌ Error in camera monitoring {camera_id}: {e}")
                    break
            
            cap.release()
            print(f"🛑 Monitoring stopped for camera {camera_id}")
        
        # Start monitoring thread
        thread = threading.Thread(target=monitor_camera, daemon=True)
        thread.start()
        self.detection_threads[camera_id] = thread
        
        print(f"🎥 Started monitoring camera {camera_id}")
        return True
        def start_camera_monitoring(self, camera_id: str, detection_interval: float = 1.0):
            """เริ่มการตรวจสอบกล้องแบบ real-time (auto-reconnect/self-healing)"""
            if camera_id not in self.active_cameras:
                print(f"❌ Camera {camera_id} not found")
                return False
            def monitor_camera():
                camera_info = self.active_cameras[camera_id]
                while camera_info.get('status') == 'active':
                    try:
                        cap = cv2.VideoCapture(camera_info['url'])
                        fail_count = 0
                        while camera_info.get('status') == 'active':
                            ret, frame = cap.read()
                            if not ret:
                                fail_count += 1
                                print(f"⚠️ Failed to read from camera {camera_id} (fail {fail_count})")
                                if fail_count > 5:
                                    print(f"🔄 Attempting reconnect for camera {camera_id}")
                                    cap.release()
                                    time.sleep(2)
                                    cap = cv2.VideoCapture(camera_info['url'])
                                    fail_count = 0
                                    camera_info['reconnect_attempts'] += 1
                                continue
                            fail_count = 0
                            detections = self.detector.detect_objects(frame, camera_id)
                            if detections:
                                camera_info['last_detection'] = datetime.now().isoformat()
                                camera_info['detection_count'] += len(detections)
                                # ส่งผลลัพธ์ไปยัง callback/notification
                                self._notify_realtime(camera_id, detections, frame)
                            self.performance_stats['frames_processed'] += 1
                            time.sleep(detection_interval)
                        cap.release()
                    except Exception as e:
                        print(f"❌ Error in camera monitoring {camera_id}: {e}")
                        time.sleep(2)
                print(f"🛑 Monitoring stopped for camera {camera_id}")
            thread = threading.Thread(target=monitor_camera, daemon=True)
            thread.start()
            self.detection_threads[camera_id] = thread
            print(f"🎥 Started monitoring camera {camera_id}")
            return True
    def _notify_realtime(self, camera_id, detections, frame):
        """ส่งผลลัพธ์ real-time (REST API/Callback/Notification)"""
        # 1. ส่ง snapshot พร้อม bbox (base64)
        for detection in detections:
            if detection.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                x, y, w, h = detection.bbox
                crop = frame[y:y+h, x:x+w]
                _, buffer = cv2.imencode('.jpg', crop)
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                payload = {
                    'camera_id': camera_id,
                    'object_type': detection.object_type,
                    'confidence': detection.confidence,
                    'threat_level': detection.threat_level.value,
                    'priority': detection.priority.value,
                    'timestamp': detection.timestamp,
                    'snapshot': img_b64,
                    'description': detection.description
                }
                # 2. เรียก callback ทุกตัว
                for cb in self.notification_callbacks:
                    try:
                        cb(payload)
                    except Exception as e:
                        print(f"❌ Callback error: {e}")
                # 3. (optionally) ส่ง REST API/แจ้งเตือนอื่นๆ ได้ที่นี่

    def add_notification_callback(self, callback):
        """เพิ่ม callback function สำหรับการแจ้งเตือน real-time"""
        self.notification_callbacks.append(callback)

    def health_check(self):
        """REST API endpoint: /api/intruder/health"""
        try:
            health = self.get_system_health()
            return json.dumps({'success': True, 'health': health})
        except Exception as e:
            return json.dumps({'success': False, 'error': str(e)})
    
    def stop_camera_monitoring(self, camera_id: str):
        """หยุดการตรวจสอบกล้อง"""
        if camera_id in self.active_cameras:
            self.active_cameras[camera_id]['status'] = 'inactive'
            print(f"🛑 Stopped monitoring camera {camera_id}")
            return True
        return False
    
    def _save_detections_to_db(self, detections: List[IntruderDetection], frame: np.ndarray):
        """บันทึกข้อมูลการตรวจจับลงฐานข้อมูล"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for detection in detections:
                # Encode image as base64 (optional, for critical detections)
                image_data = None
                if detection.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_data = base64.b64encode(buffer).decode('utf-8')
                
                cursor.execute("""
                    INSERT INTO intruder_detections 
                    (timestamp, camera_id, object_type, confidence, threat_level, priority,
                     bbox_x, bbox_y, bbox_width, bbox_height, center_x, center_y, 
                     description, image_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    detection.timestamp,
                    detection.camera_id,
                    detection.object_type,
                    detection.confidence,
                    detection.threat_level.value,
                    detection.priority.value,
                    detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3],
                    detection.center[0], detection.center[1],
                    detection.description,
                    image_data
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error saving detections to database: {e}")
    
    def _send_notifications(self, detections: List[IntruderDetection], camera_id: str):
        """ส่งการแจ้งเตือน"""
        for detection in detections:
            if detection.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                notification = {
                    'type': 'intruder_alert',
                    'camera_id': camera_id,
                    'detection': asdict(detection),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Call notification callbacks
                for callback in self.notification_callbacks:
                    try:
                        callback(notification)
                    except Exception as e:
                        print(f"❌ Notification callback error: {e}")
                
                self.performance_stats['alerts_sent'] += 1
    
    def add_notification_callback(self, callback):
        """เพิ่ม callback function สำหรับการแจ้งเตือน"""
        self.notification_callbacks.append(callback)
    
    def get_detection_history(self, camera_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """ดึงประวัติการตรวจจับ"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if camera_id:
                cursor.execute("""
                    SELECT * FROM intruder_detections 
                    WHERE camera_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (camera_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM intruder_detections 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            detections = []
            for row in rows:
                detection_dict = dict(zip(columns, row))
                detections.append(detection_dict)
            
            conn.close()
            return detections
            
        except Exception as e:
            print(f"❌ Error getting detection history: {e}")
            return []
    
    def get_system_health(self) -> Dict[str, Any]:
        """ตรวจสอบสุขภาพระบบ"""
        uptime = time.time() - self.performance_stats['system_uptime']
        
        health = {
            'status': 'healthy',
            'uptime_seconds': uptime,
            'uptime_formatted': str(timedelta(seconds=int(uptime))),
            'detector_stats': self.detector.detection_stats,
            'performance_stats': self.performance_stats,
            'active_cameras': len(self.active_cameras),
            'database_status': os.path.exists(self.db_path),
            'models_loaded': list(self.detector.models.keys())
        }
        
        return health

def create_intelligent_detector() -> UltraIntelligentIntruderDetector:
    """สร้าง AI Detector instance"""
    return UltraIntelligentIntruderDetector()

def create_integration_system(app=None) -> IntelligentIntruderIntegration:
    """สร้าง Integration System instance"""
    return IntelligentIntruderIntegration(app)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 ทดสอบ Ultra Intelligent Intruder Detection System...")
    
    # Create detector
    detector = create_intelligent_detector()
    
    # Create integration system
    integration = create_integration_system()
    
    # Test with sample frame (black image)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_objects(test_frame, "test_camera")
    
    print(f"✅ ทดสอบเสร็จสิ้น - พบการตรวจจับ: {len(detections)} รายการ")
    print(f"📊 สถิติระบบ: {detector.detection_stats}")
    print(f"🏥 สุขภาพระบบ: {integration.get_system_health()}")
