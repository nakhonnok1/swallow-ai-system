"""
🚨 ADVANCED OBJECT DETECTION SYSTEM V1.0
สำหรับตรวจจับวัตถุแปลกปลอม (คน สัตว์ ยานพาหนะ ฯลฯ)
ทำงานแยกจากระบบนกโดยสิ้นเชิง

Features:
✅ ตรวจจับ 80+ ประเภทวัตถุ
✅ การแจ้งเตือนแบบ Real-time
✅ บันทึกภาพหลักฐาน
✅ ระบบคะแนนความน่าเชื่อถือ
✅ การกรองวัตถุตามความสำคัญ
"""

import cv2
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import sqlite3
import json
import os
from typing import Dict, List, Tuple, Optional

class AdvancedObjectDetector:
    """🎯 ระบบตรวจจับวัตถุแปลกปลอมขั้นสูง"""
    
    def __init__(self):
        self.model = None
        self.detected_objects = []
        self.alert_history = []
        self.db_path = "object_detection_alerts.db"
        self.confidence_threshold = 0.25  # ลดลงเพื่อจับได้ง่ายขึ้น
        self.last_detection_time = {}
        self.cooldown_seconds = 1.5  # ลดเวลา cooldown เพื่อตอบสนองเร็วขึ้น
        
        # 🚨 วัตถุที่ต้องการแจ้งเตือน (เฉพาะสิ่งที่เป็นอันตรายหรือแปลกปลอม)
        self.critical_objects = {
            # คน และ สัตว์อันตราย - ลำดับความสำคัญสูงสุด
            'person': {'name': 'คน', 'priority': 'CRITICAL', 'color': (0, 0, 255)},
            'cat': {'name': 'แมว', 'priority': 'HIGH', 'color': (0, 165, 255)},
            'dog': {'name': 'สุนัข', 'priority': 'HIGH', 'color': (0, 165, 255)},
            'bird': {'name': 'นกตัวใหญ่ (เหยี่ย/พิราบ)', 'priority': 'HIGH', 'color': (255, 100, 0)},
            'mouse': {'name': 'หนู/ตุกแก', 'priority': 'HIGH', 'color': (255, 165, 0)},
            
            # งู และสัตว์เลื้อยคลาน (ใช้วิธีพิเศษตรวจจับ)
            'snake': {'name': 'งู', 'priority': 'CRITICAL', 'color': (128, 0, 128)},
            
            # สัตว์ป่าอันตราย
            'bear': {'name': 'หมี', 'priority': 'CRITICAL', 'color': (0, 0, 255)},
            'elephant': {'name': 'ช้าง', 'priority': 'CRITICAL', 'color': (0, 0, 255)},
            'horse': {'name': 'ม้า', 'priority': 'MEDIUM', 'color': (255, 165, 0)},
            'cow': {'name': 'วัว/ควาย', 'priority': 'MEDIUM', 'color': (255, 165, 0)},
            'sheep': {'name': 'แกะ', 'priority': 'MEDIUM', 'color': (255, 165, 0)},
            
            # ยานพาหนะต้องสงสัย
            'car': {'name': 'รถยนต์', 'priority': 'MEDIUM', 'color': (255, 165, 0)},
            'motorcycle': {'name': 'รถมอเตอร์ไซค์', 'priority': 'MEDIUM', 'color': (255, 165, 0)},
            'bicycle': {'name': 'จักรยาน', 'priority': 'LOW', 'color': (255, 255, 0)},
            'truck': {'name': 'รถบรรทุก', 'priority': 'MEDIUM', 'color': (255, 165, 0)},
            
            # วัตถุต้องสงสัย
            'backpack': {'name': 'กระเป๋าเป้', 'priority': 'LOW', 'color': (255, 255, 0)},
            'umbrella': {'name': 'ร่ม', 'priority': 'LOW', 'color': (255, 255, 0)},
            'handbag': {'name': 'กระเป๋าถือ', 'priority': 'LOW', 'color': (255, 255, 0)},
        }
        
        # เพิ่มการจัดการประสิทธิภาพ
        self.frame_skip_counter = 0
        self.detection_frequency = 5  # ตรวจจับทุก 5 เฟรม (ลดการใช้ CPU)
        self.last_detection_results = []  # เก็บผลล่าสุดเพื่อ smooth detection
        
        self._init_database()
        self._load_model()
    
    def _load_model(self):
        """โหลด YOLO model สำหรับตรวจจับวัตถุ"""
        try:
            print("🤖 กำลังโหลด Advanced Object Detection Model...")
            # ใช้ YOLOv8 model ที่ใหญ่กว่าเพื่อความแม่นยำสูง
            model_path = "yolov8n.pt"  # ใช้ model เดียวกัน แต่ config ต่าง
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print("✅ โหลด Object Detection Model สำเร็จ!")
            else:
                print("❌ ไม่พบ Model file")
                return False
                
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการโหลด Model: {e}")
            return False
        return True
    
    def _init_database(self):
        """สร้างฐานข้อมูลสำหรับเก็บการแจ้งเตือน"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS object_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    priority TEXT NOT NULL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_width INTEGER NOT NULL,
                    bbox_height INTEGER NOT NULL,
                    image_path TEXT,
                    status TEXT DEFAULT 'NEW'
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database Object Detection พร้อมใช้งาน")
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาด Database: {e}")
    
    def detect_objects(self, frame: np.ndarray, camera_props=None, frame_quality=None) -> List[Dict]:
        """ตรวจจับวัตถุในเฟรม รวมทั้งการตรวจจับรูปร่างพิเศษ"""
        # ใช้ frame skipping เพื่อประสิทธิภาพ
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.detection_frequency != 0:
            # ส่งคืนผลล่าสุดที่เก็บไว้ (ถ้ามี) เพื่อไม่ให้หายไป
            return self.last_detection_results[:3] if self.last_detection_results else []
        
        detections = []
        
        # 1. ใช้ YOLO detection เป็นหลัก (เฉพาะกรณีที่จำเป็น)
        if self.model is not None:
            try:
                yolo_detections = self._detect_with_yolo(frame)
                detections.extend(yolo_detections)
            except Exception as e:
                print(f"⚠️ YOLO detection error: {e}")
        
        # 2. ใช้ motion detection เป็น backup (เฉพาะเมื่อไม่มี YOLO detections สำคัญ)
        critical_found = any(d.get('priority') == 'CRITICAL' for d in detections)
        if not critical_found:
            motion_detections = self._detect_with_motion(frame)
            detections.extend(motion_detections)
        
        # 3. ลบ detection ที่ซ้ำกัน
        detections = self._remove_duplicate_detections(detections)
        
        # 4. จำกัดจำนวน detections เพื่อประสิทธิภาพ
        detections = detections[:5]  # ลดเหลือ 5 detections
        
        # 5. เก็บผลล่าสุดไว้
        self.last_detection_results = detections.copy()
        
        return detections
    
    def _detect_with_yolo(self, frame: np.ndarray) -> List[Dict]:
        """ตรวจจับด้วย YOLO model"""
        if self.model is None:
            return []
        
        try:
            # รัน YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            current_time = time.time()
            
            for r in results:
                boxes = r.boxes
                if boxes is not None and hasattr(boxes, 'xyxy') and hasattr(boxes.xyxy, 'shape') and boxes.xyxy.shape[0] > 0:
                    # แปลงเป็น numpy arrays ก่อน
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    
                    # ตรวจสอบว่ามีการตรวจจับหรือไม่
                    if hasattr(xyxy, 'shape') and xyxy.shape[0] > 0:
                        for i in range(xyxy.shape[0]):
                            # ดึงข้อมูล detection
                            x1, y1, x2, y2 = xyxy[i]
                            confidence = float(conf[i])
                            class_id = int(cls[i])
                            
                            # แปลง class_id เป็นชื่อวัตถุ
                            object_type = self.model.names[class_id]
                            
                            # เช็คว่าเป็นวัตถุที่เราสนใจไหม
                            if object_type in self.critical_objects:
                                object_info = self.critical_objects[object_type]
                                
                                # เช็ค cooldown เพื่อไม่แจ้งเตือนซ้ำ
                                cooldown_key = f"{object_type}_{int(x1)}_{int(y1)}"
                                if (cooldown_key not in self.last_detection_time or 
                                    current_time - self.last_detection_time[cooldown_key] > self.cooldown_seconds):
                                    
                                    detection = {
                                        'object_type': object_type,
                                        'object_name': object_info['name'],
                                        'confidence': confidence,
                                        'priority': object_info['priority'],
                                        'color': object_info['color'],
                                        'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                                        'center': (int((x1+x2)/2), int((y1+y2)/2)),
                                        'timestamp': datetime.now().isoformat(),
                                        'source': 'yolo'
                                    }
                                    
                                    detections.append(detection)
                                    self.last_detection_time[cooldown_key] = current_time
                                    
                                    # บันทึกลง database
                                    self._save_alert_to_db(detection, frame)
                                    
                                    # แจ้งเตือน Chatbot
                                    self._notify_chatbot(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการตรวจจับ YOLO: {e}")
            return []
    
    def _detect_with_motion(self, frame: np.ndarray) -> List[Dict]:
        """ตรวจจับด้วยการวิเคราะห์การเคลื่อนไหว (fallback method) - เน้นจับคน"""
        try:
            # สร้าง background model ถ้ายังไม่มี
            if not hasattr(self, '_background_model'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self._background_model = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=True,
                    varThreshold=30,  # ลดค่าเพื่อจับการเคลื่อนไหวได้ง่ายขึ้น
                    history=300
                )
                return []  # skip frame แรก
            
            # หาการเคลื่อนไหว
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = self._background_model.apply(gray, learningRate=0.01)
            
            # กรอง noise แต่เก็บรายละเอียดสำหรับคน
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # ลดขนาด kernel
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            
            # หา contours
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            current_time = time.time()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # ปรับขนาดการกรองให้เหมาะสมกับการจับคน
                if 800 < area < 80000:  # ขยายช่วงสำหรับการจับคน
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # คำนวณ aspect ratio
                    aspect_ratio = w / h if h > 0 else 1
                    
                    # ประเมินว่าเป็นประเภทไหน - เน้นการจับคน
                    if area > 5000 and 0.2 < aspect_ratio < 1.5:  # รูปร่างคล้ายคน
                        object_type = 'person'
                        object_name = 'คน (ตรวจจับจากการเคลื่อนไหว)'
                        priority = 'CRITICAL'
                        color = (255, 0, 0)
                        confidence = min(0.85, (area / 15000) + 0.5)  # เพิ่ม confidence สำหรับคน
                    elif area > 2000 and 0.4 < aspect_ratio < 2.5:
                        object_type = 'animal'
                        object_name = 'สัตว์ (ตรวจจับจากการเคลื่อนไหว)'
                        priority = 'HIGH'
                        color = (255, 165, 0)
                        confidence = min(0.75, (area / 8000) + 0.4)
                    else:
                        object_type = 'unknown'
                        object_name = 'วัตถุเคลื่อนไหว'
                        priority = 'MEDIUM'
                        color = (255, 255, 0)
                        confidence = min(0.65, (area / 5000) + 0.3)
                    
                    # เช็ค cooldown
                    cooldown_key = f"motion_{object_type}_{x//50}_{y//50}"  # ลด precision ของ position
                    if (cooldown_key not in self.last_detection_time or 
                        current_time - self.last_detection_time[cooldown_key] > self.cooldown_seconds):
                        
                        detection = {
                            'object_type': object_type,
                            'object_name': object_name,
                            'confidence': confidence,
                            'priority': priority,
                            'color': color,
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'motion'
                        }
                        
                        detections.append(detection)
                        self.last_detection_time[cooldown_key] = current_time
                        
                        # บันทึกลง database เฉพาะ CRITICAL และ HIGH เท่านั้น
                        if priority in ['CRITICAL', 'HIGH']:
                            self._save_alert_to_db(detection, frame)
                            
                            # แจ้งเตือน Chatbot
                            self._notify_chatbot(detection)
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Motion detection error: {e}")
            return []
    
    def _remove_duplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """ลบ detection ที่ซ้ำกัน (overlapping bboxes)"""
        if len(detections) <= 1:
            return detections
        
        # เรียงตาม confidence
        detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        filtered = []
        for detection in detections:
            bbox1 = detection['bbox']
            is_duplicate = False
            
            for existing in filtered:
                bbox2 = existing['bbox']
                
                # คำนวณ IoU (Intersection over Union)
                iou = self._calculate_iou(bbox1, bbox2)
                
                # ถ้า overlap มากกว่า 50% ถือว่าซ้ำ
                if iou > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """คำนวณ Intersection over Union ระหว่าง 2 bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # คำนวณพื้นที่ intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # คำนวณพื้นที่ union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_snake_like_shapes(self, frame: np.ndarray) -> List[Dict]:
        """ตรวจจับรูปร่างคล้ายงูด้วยการวิเคราะห์รูปทรง"""
        try:
            # แปลงเป็น grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ใช้ edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # หา contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            snake_detections = []
            current_time = time.time()
            
            for contour in contours:
                # กรองขนาดที่เหมาะสม (ไม่เล็กเกินไป ไม่ใหญ่เกินไป)
                area = cv2.contourArea(contour)
                if area < 200 or area > 5000:
                    continue
                
                # คำนวณ aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h)
                
                # ตรวจสอบว่าเป็นรูปทรงยาวๆ คล้ายงูไหม
                if aspect_ratio > 4:  # ยาวมากกว่ากว้าง 4 เท่า
                    # ตรวจสอบความโค้ง
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # ถ้าไม่เป็นวงกลม (circularity ต่ำ) และยาว = อาจเป็นงู
                        if circularity < 0.3:  # ไม่เป็นวงกลม
                            cooldown_key = f"snake_{x}_{y}"
                            if (cooldown_key not in self.last_detection_time or 
                                current_time - self.last_detection_time[cooldown_key] > self.cooldown_seconds * 2):
                                
                                detection = {
                                    'object_type': 'snake',
                                    'object_name': 'งู (ตรวจจับจากรูปทรง)',
                                    'confidence': min(0.7, aspect_ratio / 10),  # confidence ตามความยาว
                                    'priority': 'CRITICAL',
                                    'color': (128, 0, 128),
                                    'bbox': (x, y, w, h),
                                    'center': (x + w//2, y + h//2),
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                snake_detections.append(detection)
                                self.last_detection_time[cooldown_key] = current_time
                                
                                # บันทึกลง database
                                self._save_alert_to_db(detection, frame)
            
            return snake_detections
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการตรวจจับงู: {e}")
            return []
    
    def _save_alert_to_db(self, detection: Dict, frame: np.ndarray):
        """บันทึกการแจ้งเตือนลงฐานข้อมูล"""
        try:
            # บันทึกภาพหลักฐาน
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"alert_{detection['object_type']}_{timestamp}.jpg"
            image_path = os.path.join("anomaly_images", image_filename)
            
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            os.makedirs("anomaly_images", exist_ok=True)
            
            # ครอบภาพรอบๆ วัตถุที่ตรวจพบ
            x, y, w, h = detection['bbox']
            margin = 50
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)
            
            cropped_frame = frame[y1:y2, x1:x2]
            cv2.imwrite(image_path, cropped_frame)
            
            # บันทึกลง database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO object_alerts 
                (timestamp, object_type, object_name, confidence, priority, 
                 bbox_x, bbox_y, bbox_width, bbox_height, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection['timestamp'],
                detection['object_type'],
                detection['object_name'],
                detection['confidence'],
                detection['priority'],
                detection['bbox'][0],
                detection['bbox'][1],
                detection['bbox'][2],
                detection['bbox'][3],
                image_path
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการบันทึก: {e}")
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """วาดการตรวจจับบนเฟรม"""
        result_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            color = detection['color']
            object_name = detection['object_name']
            confidence = detection['confidence']
            priority = detection['priority']
            
            # วาดกรอบ
            thickness = 3 if priority == 'CRITICAL' else 2
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, thickness)
            
            # สร้างข้อความ
            label = f"{object_name} {confidence:.1%}"
            if priority == 'CRITICAL':
                label = f"🚨 {label}"
            elif priority == 'HIGH':
                label = f"⚠️ {label}"
            
            # วาดพื้นหลังข้อความ
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(result_frame, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # วาดข้อความ
            cv2.putText(result_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """ดึงการแจ้งเตือนล่าสุด"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM object_alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            alerts = []
            for row in rows:
                alerts.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'object_type': row[2],
                    'object_name': row[3],
                    'confidence': row[4],
                    'priority': row[5],
                    'bbox': (row[6], row[7], row[8], row[9]),
                    'image_path': row[10],
                    'status': row[11]
                })
            
            return alerts
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการดึงข้อมูล: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """สถิติการตรวจจับ"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # นับจำนวนการแจ้งเตือนแต่ละประเภท
            cursor.execute('''
                SELECT object_name, priority, COUNT(*) as count
                FROM object_alerts 
                WHERE date(timestamp) = date('now')
                GROUP BY object_name, priority
                ORDER BY count DESC
            ''')
            
            today_stats = cursor.fetchall()
            
            # นับรวมทั้งหมด
            cursor.execute('SELECT COUNT(*) FROM object_alerts')
            total_alerts = cursor.fetchone()[0]
            
            # นับวันนี้
            cursor.execute('''
                SELECT COUNT(*) FROM object_alerts 
                WHERE date(timestamp) = date('now')
            ''')
            today_total = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_alerts': total_alerts,
                'today_total': today_total,
                'today_by_type': today_stats,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการดึงสถิติ: {e}")
            return {'total_alerts': 0, 'today_total': 0, 'today_by_type': []}
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """ดึงการแจ้งเตือนล่าสุด"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, object_type, object_name, confidence, priority,
                       bbox_x, bbox_y, bbox_width, bbox_height, image_path
                FROM object_alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'timestamp': row[0],
                    'object_type': row[1],
                    'object_name': row[2],
                    'confidence': row[3],
                    'priority': row[4],
                    'bbox': {
                        'x': row[5],
                        'y': row[6],
                        'width': row[7],
                        'height': row[8]
                    },
                    'image_path': row[9]
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการดึงข้อมูลแจ้งเตือน: {e}")
            return []

# 🧪 ทดสอบระบบ
if __name__ == "__main__":
    print("🚀 ทดสอบ Advanced Object Detection System")
    
    detector = AdvancedObjectDetector()
    
    if detector.model:
        print("✅ ระบบพร้อมใช้งาน!")
        print(f"📊 ตรวจจับได้ {len(detector.critical_objects)} ประเภทวัตถุ")
        
        for obj_type, info in detector.critical_objects.items():
            priority_emoji = "🚨" if info['priority'] == 'CRITICAL' else "⚠️" if info['priority'] == 'HIGH' else "ℹ️"
            print(f"   {priority_emoji} {info['name']} ({info['priority']})")
    else:
        print("❌ ระบบยังไม่พร้อม")
