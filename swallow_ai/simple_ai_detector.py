#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMPLE AI DETECTOR - ใช้ OpenCV DNN แทน Ultralytics YOLO
"""

import cv2
import numpy as np

class SimpleYOLODetector:
    """AI detector ที่ใช้ OpenCV DNN - เสถียร 100%"""
    
    def __init__(self):
        print("🔧 เริ่มต้น Simple AI Detector...")
        try:
            # ใช้ OpenCV AI Detector
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from opencv_yolo_detector import OpenCVYOLODetector
            
            self.opencv_ai = OpenCVYOLODetector()
            if self.opencv_ai.available:
                print("✅ Simple AI Detector พร้อมใช้งาน")
                self.available = True
            else:
                print("⚠️ OpenCV AI ไม่พร้อม")
                self.available = False
                
        except Exception as e:
            print(f"❌ ไม่สามารถโหลด AI: {e}")
            self.available = False
    
    def detect_birds(self, frame):
        """ตรวจจับนกในเฟรม - ใช้ AI จริงๆ"""
        if not self.available:
            return []
        
        try:
            # ใช้ OpenCV AI Detection
            bird_detections = self.opencv_ai.detect_birds(frame)
            
            detections = []
            for det in bird_detections:
                detections.append({
                    'center': det['center'],
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'area': det['bbox'][2] * det['bbox'][3],  # width * height
                    'source': 'opencv_ai'
                })
                
            return detections
            
        except Exception as e:
            print(f"⚠️ AI Detection error: {e}")
            return []

# Test simple detector
if __name__ == "__main__":
    detector = SimpleYOLODetector()
    
    if detector.available:
        print("🧪 ทดสอบ Simple AI Detector...")
        
        # ทดสอบกับ RTSP
        rtsp_url = "rtsp://ainok1:ainok123@192.168.1.100:554/stream1"
        cap = cv2.VideoCapture(rtsp_url)
        
        if cap.isOpened():
            print("✅ เชื่อมต่อ RTSP สำเร็จ")
            
            frame_count = 0
            detection_count = 0
            
            while frame_count < 20:  # ทดสอบ 20 เฟรม
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame_count += 1
                
                # ทดสอบการตรวจจับ
                detections = detector.detect_birds(frame)
                
                if len(detections) > 0:
                    detection_count += 1
                    print(f"🎯 Frame {frame_count}: AI พบนก {len(detections)} ตัว")
                    
                    for i, det in enumerate(detections):
                        print(f"   นก {i+1}: ความมั่นใจ {det['confidence']:.2f}, พื้นที่ {det['area']}")
                        
                if frame_count % 5 == 0:
                    print(f"📊 ประมวลผลแล้ว {frame_count}/20 เฟรม | การตรวจจับ: {detection_count}")
                    
            cap.release()
            
            print(f"\n📈 สรุปผลการทดสอบ Simple AI:")
            print(f"   📷 เฟรมทั้งหมด: {frame_count}")
            print(f"   🎯 เฟรมที่จับได้: {detection_count}")
            print(f"   📊 อัตราการตรวจจับ: {(detection_count/frame_count)*100:.1f}%")
            
            if detection_count > 0:
                print("✅ Simple AI ทำงานได้!")
            else:
                print("⚠️ ไม่พบการตรวจจับ")
        else:
            print("❌ ไม่สามารถเชื่อมต่อ RTSP")
    else:
        print("❌ Simple AI ไม่พร้อมใช้งาน")
