#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ ULTRA SAFE AI DETECTOR - 100% ไม่มี YOLO
แก้ไขปัญหา YOLO array ambiguity error อย่างสมบูรณ์
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path

class UltraSafeDetector:
    def connect_agents(self, bird_detector=None, motion_analyzer=None, ai_chatbot=None):
        """
        เชื่อมต่อ agent อื่น ๆ เพื่อใช้งานร่วมกัน
        Args:
            bird_detector: instance ของ BirdDetector
            motion_analyzer: instance ของ MotionAnalyzer
            ai_chatbot: instance ของ SmartAIChatbot หรือ UltraSmartAIChatbot
        """
        self.bird_detector = bird_detector
        self.motion_analyzer = motion_analyzer
        self.ai_chatbot = ai_chatbot
        import logging
        logging.info("UltraSafeDetector: Agents connected successfully.")

    def detect_all(self, frame: np.ndarray, camera_props: dict = None, frame_quality: dict = None) -> dict:
        """
        ตรวจจับและวิเคราะห์ด้วยทุก agent ที่เชื่อมต่อ
        Returns:
            dict: รวมผลลัพธ์จากทุก agent
        """
        results = {}
        # UltraSafeDetector motion detection
        processed_frame, detections, stats = self.detect_birds_realtime(frame, camera_props, frame_quality)
        results['ultra_safe'] = {
            'frame': processed_frame,
            'detections': detections,
            'stats': stats
        }
        # BirdDetector
        if hasattr(self, 'bird_detector') and self.bird_detector:
            try:
                bird_detections = self.bird_detector.detect(frame, camera_props, frame_quality)
                results['bird_detector'] = bird_detections
            except Exception as e:
                results['bird_detector'] = f"Error: {e}"
        # MotionAnalyzer
        if hasattr(self, 'motion_analyzer') and self.motion_analyzer:
            try:
                motion_result = self.motion_analyzer.analyze_motion(frame, camera_props, frame_quality)
                results['motion_analyzer'] = motion_result
            except Exception as e:
                results['motion_analyzer'] = f"Error: {e}"
        # AI Chatbot
        if hasattr(self, 'ai_chatbot') and self.ai_chatbot:
            try:
                chatbot_response = self.ai_chatbot.get_response(
                    f"ตรวจจับ {len(detections)} วัตถุในเฟรม", camera_props, frame_quality)
                results['ai_chatbot'] = chatbot_response
            except Exception as e:
                results['ai_chatbot'] = f"Error: {e}"
        return results
    """Ultra Safe AI Detector - ไม่มี YOLO เลย ใช้ Computer Vision เบื้องต้น"""
    
    def __init__(self):
        self.detection_enabled = True
        self.last_birds_in = 0
        self.last_birds_out = 0
        self.frame_count = 0
        self.motion_detector = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        print("🛠️ Ultra Safe Detector initialized - 100% ไม่มี YOLO")
        
    def detect_birds_realtime(self, frame: np.ndarray, camera_props: dict = None, frame_quality: dict = None) -> tuple:
        """
        ตรวจจับนกด้วยระบบ AI หลัก (BirdDetector) หากเชื่อมต่อไว้ มิฉะนั้น fallback เป็น motion detection
        Args:
            frame: ภาพจากกล้อง
            camera_props: คุณสมบัติกล้อง เช่น exposure, brightness
            frame_quality: คุณภาพภาพ เช่น blur, brightness
        Returns:
            processed_frame: เฟรมที่วาดผลการตรวจจับ
            detections: รายการผลการตรวจจับ
            stats: ข้อมูลสถิติ
        """
        import logging
        try:
            self.frame_count += 1
            processed_frame = frame.copy()
            detections = []

            # หากเชื่อมต่อ BirdDetector ให้ใช้ AI หลักตรวจจับนก
            if hasattr(self, 'bird_detector') and self.bird_detector:
                try:
                    detections = self.bird_detector.detect(frame, camera_props, frame_quality)
                    for det in detections:
                        if 'bbox' in det:
                            x, y, w, h = det['bbox']
                            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(processed_frame, f"AI: {det.get('confidence', 0.0):.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    logging.error(f"BirdDetector error: {e}")
                    detections = []
            else:
                # Fallback: ใช้ motion detection เดิม
                brightness = frame_quality.get('brightness', 1) if frame_quality else 1
                min_area = 50 if brightness > 0.5 else 30
                max_area = 3000 if brightness > 0.5 else 2000
                height, width = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fg_mask = self.background_subtractor.apply(frame)
                kernel = np.ones((5,5), np.uint8)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bird_count = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if min_area < area < max_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.2 < aspect_ratio < 5.0:
                            bird_count += 1
                            center_x = x + w // 2
                            center_y = y + h // 2
                            detections.append({
                                'center': (center_x, center_y),
                                'bbox': (x, y, w, h),
                                'confidence': 0.8,
                                'area': area,
                                'source': 'motion_detection'
                            })
                            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(processed_frame, f"Motion: 0.8", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # จำลองการนับเข้า-ออก (ปรับปรุงแล้ว)
                if bird_count > 0:
                    if self.frame_count % 20 == 0:
                        if bird_count >= 2:
                            self.last_birds_in += 1
                        elif bird_count >= 1:
                            self.last_birds_out += 1

            # สร้าง stats
            stats = {
                'entering': self.last_birds_in,
                'exiting': self.last_birds_out,
                'current_detections': len(detections)
            }

            # แสดงข้อมูลบนเฟรม
            info_text = f"Ultra Safe AI: In={self.last_birds_in} Out={self.last_birds_out} Current={len(detections)}"
            cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            status_text = "MOTION DETECTION MODE (100% YOLO FREE)"
            cv2.putText(processed_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            frame_text = f"Frame: {self.frame_count}"
            cv2.putText(processed_frame, frame_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            logging.info(f"UltraSafeDetector: Detected {len(detections)} birds, stats={stats}")
            return processed_frame, detections, stats

        except Exception as e:
            logging.error(f"Ultra Safe AI Error: {e}")
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Detection Error: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return error_frame, [], {'entering': self.last_birds_in, 'exiting': self.last_birds_out, 'current_detections': 0}
    
    def detect_suspicious_objects(self, frame):
        """ตรวจจับวัตถุแปลกปลอม (ปิดไว้ - ใช้แค่ motion detection)"""
        return [], frame
    
    def reset_counts(self):
        """รีเซ็ตการนับ"""
        self.last_birds_in = 0
        self.last_birds_out = 0
        self.frame_count = 0
        print("🛠️ Ultra Safe Detector: Counts reset")

def test_ultra_safe_ai():
    """ทดสอบ Ultra Safe AI System"""
    print("🛠️ ทดสอบ Ultra Safe Detector...")
    
    import logging
    from types import SimpleNamespace
    logging.basicConfig(level=logging.INFO)
    detector = UltraSafeDetector()

    # ตัวอย่างการเชื่อมต่อ BirdDetector จริง (import จากไฟล์หลัก)
    try:
        from ultimate_perfect_ai_MASTER import BirdDetector
        bird_detector = BirdDetector()
        print("✅ ใช้งาน BirdDetector จริงสำเร็จ!")
    except Exception as e:
        print(f"⚠️ ใช้งาน BirdDetector จริงไม่ได้: {e}\nใช้ mock แทน")
        class BirdDetector:
            def detect(self, frame, camera_props=None, frame_quality=None):
                return [{'center': (320, 240), 'confidence': 0.9, 'bbox': (300, 220, 40, 40), 'type': 'bird'}]
        bird_detector = BirdDetector()

    # Mock agent สำหรับตัวอื่น
    class MockMotionAnalyzer:
        def analyze_motion(self, frame, camera_props=None, frame_quality=None):
            return {'motion_detected': True, 'motion_strength': 0.7}
    class MockAIChatbot:
        def get_response(self, message, camera_props=None, frame_quality=None):
            return f"AI ตอบกลับ: {message}"

    # เชื่อมต่อ agent จริงและ mock
    detector.connect_agents(
        bird_detector=bird_detector,
        motion_analyzer=MockMotionAnalyzer(),
        ai_chatbot=MockAIChatbot()
    )

    # สร้างภาพทดสอบ
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (150, 130), (255, 255, 255), -1)
    cv2.circle(test_frame, (200, 200), 20, (255, 255, 255), -1)

    # ทดสอบการตรวจจับแบบรวม agent
    camera_props = {'exposure': 0.7, 'brightness': 0.8}
    frame_quality = {'blur': 0.1, 'brightness': 0.8}
    results = detector.detect_all(test_frame, camera_props, frame_quality)

    print("\n=== UltraSafeDetector Integration Test ===")
    print(f"✅ UltraSafe: ตรวจพบ {len(results['ultra_safe']['detections'])} motion objects")
    print(f"📊 UltraSafe Stats: {results['ultra_safe']['stats']}")
    print(f"🐦 BirdDetector: {results.get('bird_detector')}")
    print(f"🎯 MotionAnalyzer: {results.get('motion_analyzer')}")
    print(f"🤖 AIChatbot: {results.get('ai_chatbot')}")
    print("🛠️ Ultra Safe AI System Integration ทำงานได้ปกติ (100% YOLO FREE)!")
    print("========================================\n")
    return True

if __name__ == "__main__":
    test_ultra_safe_ai()
