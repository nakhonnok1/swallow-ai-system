"""
🔴 AI ตรวจจับนกแอ่นสำหรับสตรีม 24 ชั่วโมง
ระบบที่เสถียร ประหยัดทรัพยากร และทำงานต่อเนื่อง
"""

import cv2
import cv2
import numpy as np
import time
import json
import threading
import logging
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
import gc
import psutil

# เซ็ตอัพ logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_stream.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryEfficientDetector:
    """🔬 ตัวตรวจจับนกที่ประหยัดหน่วยความจำ"""
    
    def __init__(self):
        # ใช้พารามิเตอร์ที่ดีที่สุดจากการปรับแต่ง
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=4,  # จากอัลกอริทึมเจเนติก
            history=50       # ลดลงเพื่อประหยัดหน่วยความจำ
        )
        
        self.min_contour_area = 3     # จากอัลกอริทึมเจเนติก
        self.max_contour_area = 1088  # จากอัลกอริทึมเจเนติก
        
        # ประหยัดหน่วยความจำ
        self.frame_skip = 2  # ประมวลผลทุก 2 เฟรม
        self.frame_count = 0
        
    def detect_birds(self, frame: np.ndarray) -> List[Dict]:
        """🔍 ตรวจจับนกแบบประหยัดทรัพยากร"""
        
        self.frame_count += 1
        
        # ข้ามเฟรมเพื่อประหยัดทรัพยากร
        if self.frame_count % self.frame_skip != 0:
            return []
        
        # ลดขนาดเฟรมเพื่อประมวลผลเร็วขึ้น
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # เตรียมเฟรม
        frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame_blur)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # หา contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_contour_area <= area <= self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # ปรับกลับตามขนาดเฟรมจริง
                if width > 640:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)
                    area = area / (scale * scale)
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'confidence': min(1.0, area / 100.0)
                })
        
        return detections

class LightweightTracker:
    """🎯 ตัวติดตามนกแบบเบา"""
    
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = 80
        self.max_age = 30  # ลดลงเพื่อล้างข้อมูลเก่า
        self.max_path_length = 20  # จำกัดประวัติเส้นทาง
        
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """🔄 อัปเดตการติดตาม"""
        
        if not detections:
            # อายุการติดตาม +1
            for track in self.tracks.values():
                track['age'] += 1
            
            # ลบ tracks ที่เก่าเกินไป
            self.tracks = {tid: track for tid, track in self.tracks.items() 
                          if track['age'] < self.max_age}
            
            return list(self.tracks.values())
        
        # จับคู่การตรวจจับกับ tracks
        unmatched_detections = []
        
        for detection in detections:
            detection_center = detection['center']
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track in self.tracks.items():
                if track['age'] > 5:  # ข้าม tracks ที่เก่าเกินไป
                    continue
                    
                track_center = track['center']
                distance = np.sqrt(
                    (detection_center[0] - track_center[0])**2 +
                    (detection_center[1] - track_center[1])**2
                )
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                # อัปเดต track ที่มีอยู่
                track = self.tracks[best_track_id]
                track['center'] = detection_center
                track['bbox'] = detection['bbox']
                track['age'] = 0
                
                # อัปเดตประวัติเส้นทาง
                track['path_history'].append(detection_center)
                if len(track['path_history']) > self.max_path_length:
                    track['path_history'].pop(0)
                
                # คำนวณการเคลื่อนไหวรวม
                if len(track['path_history']) > 1:
                    total_movement = 0
                    for i in range(1, len(track['path_history'])):
                        movement = np.sqrt(
                            (track['path_history'][i][0] - track['path_history'][i-1][0])**2 +
                            (track['path_history'][i][1] - track['path_history'][i-1][1])**2
                        )
                        total_movement += movement
                    track['total_movement'] = total_movement
                else:
                    track['total_movement'] = 0
            else:
                unmatched_detections.append(detection)
        
        # สร้าง tracks ใหม่
        for detection in unmatched_detections:
            self.tracks[self.next_id] = {
                'id': self.next_id,
                'center': detection['center'],
                'bbox': detection['bbox'],
                'path_history': [detection['center']],
                'age': 0,
                'total_movement': 0,
                'area': detection['area']
            }
            self.next_id += 1
        
        # อายุการติดตาม +1 สำหรับ tracks ที่ไม่ได้อัปเดต
        for track in self.tracks.values():
            if track['age'] > 0:
                track['age'] += 1
        
        # ลบ tracks ที่เก่าเกินไป
        self.tracks = {tid: track for tid, track in self.tracks.items() 
                      if track['age'] < self.max_age}
        
        return list(self.tracks.values())

class StreamingEntranceAnalyzer:
    """🚪 ระบบวิเคราะห์ทางเข้าสำหรับสตรีม"""
    
    def __init__(self, entrance_zone: Dict):
        self.entrance_zone = entrance_zone
        self.detection_radius = 94  # จากอัลกอริทึมเจเนติก
        
        # เก็บสถิติ
        self.daily_entries = 0
        self.daily_exits = 0
        self.hourly_stats = deque(maxlen=24)  # เก็บข้อมูล 24 ชั่วโมง
        
        # ฐานข้อมูล
        self.init_database()
        
        # การนับที่ไม่ซ้ำ
        self.counted_entries = set()
        self.counted_exits = set()
        self.last_cleanup = time.time()
        
    def init_database(self):
        """🗄️ เริ่มต้นฐานข้อมูล"""
        self.conn = sqlite3.connect('live_stream_data.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bird_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                track_id INTEGER,
                confidence REAL,
                position_x INTEGER,
                position_y INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour DATETIME,
                entries INTEGER DEFAULT 0,
                exits INTEGER DEFAULT 0
            )
        ''')
        
        self.conn.commit()
    
    def analyze_tracks(self, tracks: List[Dict]) -> Dict:
        """🔍 วิเคราะห์ tracks"""
        
        if not self.entrance_zone:
            return {'new_entries': 0, 'new_exits': 0}
        
        new_entries = 0
        new_exits = 0
        entrance_center = (self.entrance_zone['center_x'], self.entrance_zone['center_y'])
        
        for track in tracks:
            track_id = track['id']
            track_center = track['center']
            
            # ตรวจสอบระยะทางถึงทางเข้า
            distance = np.sqrt(
                (track_center[0] - entrance_center[0])**2 +
                (track_center[1] - entrance_center[1])**2
            )
            
            if distance <= self.detection_radius:
                # วิเคราะห์ทิศทาง
                direction = self.analyze_direction(track, entrance_center)
                
                if direction == 'entering' and track_id not in self.counted_entries:
                    if self.is_valid_entry(track):
                        self.counted_entries.add(track_id)
                        new_entries += 1
                        self.daily_entries += 1
                        
                        # บันทึกลงฐานข้อมูล
                        self.log_event('entry', track_id, track_center, 0.8)
                        
                elif direction == 'exiting' and track_id not in self.counted_exits:
                    if self.is_valid_exit(track):
                        self.counted_exits.add(track_id)
                        new_exits += 1
                        self.daily_exits += 1
                        
                        # บันทึกลงฐานข้อมูล
                        self.log_event('exit', track_id, track_center, 0.8)
        
        # ล้างข้อมูลเก่าทุก 5 นาที
        current_time = time.time()
        if current_time - self.last_cleanup > 300:  # 5 นาที
            self.cleanup_old_data()
            self.last_cleanup = current_time
        
        return {
            'new_entries': new_entries,
            'new_exits': new_exits,
            'daily_entries': self.daily_entries,
            'daily_exits': self.daily_exits
        }
    
    def analyze_direction(self, track: Dict, entrance_center: Tuple[int, int]) -> str:
        """🧭 วิเคราะห์ทิศทาง"""
        
        path = track['path_history']
        if len(path) < 3:
            return 'unknown'
        
        # ดูการเคลื่อนไหวล่าสุด 3 จุด
        recent_points = path[-3:]
        
        # คำนวณการเคลื่อนไหวเข้าหาหรือออกจากทางเข้า
        distances = []
        for point in recent_points:
            distance = np.sqrt(
                (point[0] - entrance_center[0])**2 +
                (point[1] - entrance_center[1])**2
            )
            distances.append(distance)
        
        # ถ้าระยะทางลดลง = เข้าใกล้ = กำลังเข้า
        if distances[0] > distances[-1]:
            return 'entering'
        # ถ้าระยะทางเพิ่มขึ้น = ออกห่าง = กำลังออก
        elif distances[0] < distances[-1]:
            return 'exiting'
        else:
            return 'unknown'
    
    def is_valid_entry(self, track: Dict) -> bool:
        """✅ ตรวจสอบการเข้าที่ถูกต้อง"""
        return (track['total_movement'] > 5.0 and 
                len(track['path_history']) >= 3 and
                track['area'] >= 3)
    
    def is_valid_exit(self, track: Dict) -> bool:
        """✅ ตรวจสอบการออกที่ถูกต้อง"""
        return (track['total_movement'] > 5.0 and 
                len(track['path_history']) >= 3 and
                track['area'] >= 3)
    
    def log_event(self, event_type: str, track_id: int, position: Tuple[int, int], confidence: float):
        """📝 บันทึกเหตุการณ์"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO bird_events (event_type, track_id, confidence, position_x, position_y)
                VALUES (?, ?, ?, ?, ?)
            ''', (event_type, track_id, confidence, position[0], position[1]))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Database error: {e}")
    
    def cleanup_old_data(self):
        """🧹 ล้างข้อมูลเก่า"""
        # ล้าง counted sets ทุก 10 นาที
        if len(self.counted_entries) > 1000:
            self.counted_entries.clear()
        if len(self.counted_exits) > 1000:
            self.counted_exits.clear()
        
        # ล้างข้อมูลในฐานข้อมูลที่เก่ากว่า 7 วัน
        try:
            cursor = self.conn.cursor()
            week_ago = datetime.now() - timedelta(days=7)
            cursor.execute('DELETE FROM bird_events WHERE timestamp < ?', (week_ago,))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

class LiveStreamAI:
    """🔴 AI สำหรับสตรีม 24 ชั่วโมง"""
    
    def __init__(self, stream_source: str = 0, entrance_zone: Dict = None):
        self.stream_source = stream_source
        self.detector = MemoryEfficientDetector()
        self.tracker = LightweightTracker()
        self.entrance_analyzer = StreamingEntranceAnalyzer(entrance_zone or self.load_entrance_config())
        
        # สถิติการทำงาน
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
        self.start_time = time.time()
        self.last_fps_update = time.time()
        
        # การจัดการหน่วยความจำ
        self.memory_check_interval = 300  # ตรวจสอบทุก 5 นาที
        self.last_memory_check = time.time()
        
        # การบันทึก
        self.running = False
        
        logger.info("🔴 Live Stream AI เริ่มต้นพร้อมใช้งาน")
    
    def load_entrance_config(self) -> Dict:
        """📂 โหลดการตั้งค่าทางเข้า"""
        try:
            with open('entrance_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if 'entrance_zones' in config and config['entrance_zones']:
                first_zone = config['entrance_zones'][0]
                return {
                    'center_x': first_zone[0],
                    'center_y': first_zone[1],
                    'width': 100,
                    'height': 100,
                    'x': first_zone[0] - 50,
                    'y': first_zone[1] - 50
                }
        except Exception as e:
            logger.warning(f"ไม่สามารถโหลดการตั้งค่าทางเข้า: {e}")
        
        return {
            'center_x': 816,
            'center_y': 297,
            'width': 100,
            'height': 100,
            'x': 766,
            'y': 247
        }
    
    def start_stream(self):
        """▶️ เริ่มสตรีม"""
        
        logger.info(f"🎬 เริ่มเชื่อมต่อ: {self.stream_source}")
        
        cap = cv2.VideoCapture(self.stream_source)
        if not cap.isOpened():
            logger.error(f"❌ ไม่สามารถเชื่อมต่อ: {self.stream_source}")
            return
        
        # ตั้งค่า buffer ให้เล็ก
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.running = True
        logger.info("🟢 เริ่มการประมวลผลสตรีม 24 ชั่วโมง")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ ไม่สามารถอ่านเฟรม - พยายามเชื่อมต่อใหม่")
                    time.sleep(1)
                    continue
                
                # ประมวลผลเฟรม
                self.process_frame(frame)
                
                # ตรวจสอบหน่วยความจำ
                self.check_memory_usage()
                
                # ลดการใช้ CPU
                time.sleep(0.01)  # 10ms
                
        except KeyboardInterrupt:
            logger.info("🛑 ได้รับสัญญาณหยุด")
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาด: {e}")
        finally:
            cap.release()
            self.cleanup()
            logger.info("🏁 หยุดการทำงาน")
    
    def process_frame(self, frame: np.ndarray):
        """🎬 ประมวลผลเฟรม"""
        
        self.frame_count += 1
        frame_start_time = time.time()
        
        # ตรวจจับและติดตาม
        detections = self.detector.detect_birds(frame)
        tracks = self.tracker.update_tracks(detections)
        
        # วิเคราะห์ทางเข้า
        entrance_results = self.entrance_analyzer.analyze_tracks(tracks)
        
        # คำนวณ FPS
        frame_time = time.time() - frame_start_time
        self.fps_counter.append(frame_time)
        
        # แสดงสถิติทุก 30 วินาที
        if time.time() - self.last_fps_update > 30:
            self.print_stats(entrance_results)
            self.last_fps_update = time.time()
    
    def check_memory_usage(self):
        """💾 ตรวจสอบการใช้หน่วยความจำ"""
        
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return
        
        # ตรวจสอบหน่วยความจำ
        process = psutil.Process()
        memory_percent = process.memory_percent()
        
        if memory_percent > 80:  # ถ้าใช้มากกว่า 80%
            logger.warning(f"⚠️ หน่วยความจำเกือบเต็ม: {memory_percent:.1f}%")
            
            # บังคับ garbage collection
            gc.collect()
            
            # ล้างข้อมูลเก่า
            self.tracker.tracks.clear()
            self.entrance_analyzer.cleanup_old_data()
            
            logger.info("🧹 ล้างหน่วยความจำเสร็จ")
        
        self.last_memory_check = current_time
    
    def print_stats(self, entrance_results: Dict):
        """📊 แสดงสถิติ"""
        
        uptime = time.time() - self.start_time
        avg_fps = len(self.fps_counter) / sum(self.fps_counter) if self.fps_counter else 0
        
        logger.info(f"""
🔴 === สถิติสตรีม 24 ชั่วโมง ===
⏱️ เวลาทำงาน: {uptime/3600:.1f} ชั่วโมง
🎬 เฟรมประมวลผล: {self.frame_count:,}
⚡ FPS เฉลี่ย: {avg_fps:.1f}
🐦 นกเข้าวันนี้: {entrance_results['daily_entries']}
🐦 นกออกวันนี้: {entrance_results['daily_exits']}
🎯 ทำงานต่อเนื่อง: ✅
        """)
    
    def stop_stream(self):
        """⏹️ หยุดสตรีม"""
        self.running = False
    
    def cleanup(self):
        """🧹 ล้างข้อมูล"""
        if hasattr(self.entrance_analyzer, 'conn'):
            self.entrance_analyzer.conn.close()
        gc.collect()

def main():
    """🚀 เริ่มต้นระบบ"""
    
    # ตัวเลือกแหล่งสตรีม
    STREAM_SOURCES = {
        '0': 0,  # กล้องเว็บแคม
        'rtsp': 'rtsp://ainok1:ainok123@192.168.1.100:554/stream1',  # กล้อง IP
        'file': 'training_videos/mixed_behavior/mixed_001.mp4.mp4'  # ไฟล์วิดีโอ
    }
    
    print("🔴 ระบบ AI ตรวจจับนกแอ่นสำหรับสตรีม 24 ชั่วโมง")
    print("=" * 50)
    print("เลือกแหล่งสตรีม:")
    print("0 - กล้องเว็บแคม")
    print("1 - กล้อง IP (RTSP)")
    print("2 - ไฟล์วิดีโอทดสอบ")
    
    choice = input("เลือก (0-2): ").strip()
    
    if choice == '0':
        stream_source = 0
    elif choice == '1':
        stream_source = input("ใส่ RTSP URL: ").strip() or STREAM_SOURCES['rtsp']
    elif choice == '2':
        stream_source = STREAM_SOURCES['file']
    else:
        stream_source = 0
    
    # สร้างและเริ่ม AI
    ai = LiveStreamAI(stream_source)
    
    try:
        ai.start_stream()
    except KeyboardInterrupt:
        ai.stop_stream()

if __name__ == "__main__":
    main()
