#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMPLE YOLO DETECTOR - แก้ปัญหา array error โดยสร้างใหม่ให้ใช้งานง่าย
"""

import cv2
import numpy as np

class SimpleYOLODetector:
    """YOLO detector ที่ใช้งานได้ไม่มี array error"""
    
    def __init__(self):
        print("🔧 เริ่มต้น Simple YOLO Detector...")
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            self.available = True
            print("✅ Simple YOLO พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ ไม่สามารถโหลด YOLO: {e}")
            self.yolo_model = None
            self.available = False
    
    def detect_birds(self, frame):
        """ตรวจจับนกในเฟรม - วิธีง่ายๆ ไม่มี error"""
        if not self.available or self.yolo_model is None:
            return []
        
        detections = []
        
        try:
            # รัน YOLO
            results = self.yolo_model(frame, verbose=False, conf=0.2)
            
            # ตรวจสอบผลลัพธ์อย่างปลอดภัย
            if results and len(results) > 0:
                result = results[0]
                
                # ตรวจสอบว่ามี boxes หรือไม่
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # ตรวจสอบว่ามีข้อมูลหรือไม่
                    if hasattr(boxes, 'data') and len(boxes.data) > 0:
                        # วนลูปแต่ละ detection
                        for i in range(len(boxes.data)):
                            try:
                                # ดึงข้อมูลแต่ละตัว
                                detection = boxes.data[i].cpu().numpy()
                                
                                if len(detection) >= 6:
                                    x1, y1, x2, y2, conf, cls = detection[:6]
                                    
                                    # ตรวจสอบว่าเป็นนก (class 14) หรือไม่
                                    if int(cls) == 14 and float(conf) > 0.2:
                                        # คำนวณตำแหน่งกลาง
                                        center_x = int((x1 + x2) / 2)
                                        center_y = int((y1 + y2) / 2)
                                        width = int(x2 - x1)
                                        height = int(y2 - y1)
                                        
                                        # ตรวจสอบขนาดที่สมเหตุสมผล
                                        area = width * height
                                        if 100 <= area <= 5000:
                                            detections.append({
                                                'center': (center_x, center_y),
                                                'bbox': (int(x1), int(y1), width, height),
                                                'confidence': float(conf),
                                                'area': area,
                                                'source': 'simple_yolo'
                                            })
                                            
                            except Exception as detection_error:
                                print(f"⚠️ Detection parsing error: {detection_error}")
                                continue
                                
        except Exception as e:
            print(f"⚠️ YOLO error: {e}")
            return []
        
        return detections

# Test simple detector
if __name__ == "__main__":
    detector = SimpleYOLODetector()
    
    if detector.available:
        print("🧪 ทดสอบ Simple YOLO Detector...")
        
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
                    print(f"🎯 Frame {frame_count}: พบนก {len(detections)} ตัว")
                    
                    for i, det in enumerate(detections):
                        print(f"   นก {i+1}: ความมั่นใจ {det['confidence']:.2f}, พื้นที่ {det['area']}")
                        
                if frame_count % 5 == 0:
                    print(f"📊 ประมวลผลแล้ว {frame_count}/20 เฟรม | การตรวจจับ: {detection_count}")
                    
            cap.release()
            
            print(f"\n📈 สรุปผลการทดสอบ Simple YOLO:")
            print(f"   📷 เฟรมทั้งหมด: {frame_count}")
            print(f"   🎯 เฟรมที่จับได้: {detection_count}")
            print(f"   📊 อัตราการตรวจจับ: {(detection_count/frame_count)*100:.1f}%")
            
            if detection_count > 0:
                print("✅ Simple YOLO ทำงานได้!")
            else:
                print("⚠️ ไม่พบการตรวจจับ")
        else:
            print("❌ ไม่สามารถเชื่อมต่อ RTSP")
    else:
        print("❌ Simple YOLO ไม่พร้อมใช้งาน")
