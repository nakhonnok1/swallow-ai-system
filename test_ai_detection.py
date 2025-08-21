#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ทดสอบ AI Detection - ตรวจสอบว่า AI จับนกได้หรือไม่
"""

import cv2
import numpy as np
from ultimate_perfect_ai_MASTER import V5_UltimatePrecisionSwallowAI

def test_ai_with_rtsp():
    """ทดสอบ AI กับ RTSP stream"""
    print("🧪 เริ่มทดสอบ V5 ULTRA PRECISION AI กับ RTSP...")
    
    # สร้าง AI detector
    ai_detector = V5_UltimatePrecisionSwallowAI('mixed')  # ใช้ mode mixed
    
    # เชื่อมต่อกล้อง
    rtsp_url = "rtsp://ainok1:ainok123@192.168.1.101:554/stream1"
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ ไม่สามารถเชื่อมต่อ RTSP")
        return
        
    print("✅ เชื่อมต่อ RTSP สำเร็จ")
    print("🔍 กำลังทดสอบการตรวจจับ...")
    
    frame_count = 0
    detection_count = 0
    
    while frame_count < 50:  # ทดสอบ 50 เฟรม
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_count += 1
        
        try:
            # ทดสอบการตรวจจับด้วย process_frame_v5
            processed_frame, results = ai_detector.process_frame_v5(frame, frame_count)
            
            if results and 'detections' in results:
                detections = results['detections']
                if len(detections) > 0:
                    detection_count += 1
                    print(f"🎯 Frame {frame_count}: พบนก {len(detections)} ตัว")
                    
                    # แสดงรายละเอียด
                    for i, det in enumerate(detections):
                        if 'confidence' in det:
                            print(f"   นก {i+1}: ความมั่นใจ {det['confidence']:.2f}")
                            
            if frame_count % 10 == 0:
                print(f"📊 ประมวลผลแล้ว {frame_count}/50 เฟรม | การตรวจจับ: {detection_count}")
                
        except Exception as e:
            print(f"❌ Error ในการประมวลผล: {e}")
            import traceback
            traceback.print_exc()
            break  # หยุดเมื่อเกิด error เพื่อดู stacktrace
            
    cap.release()
    
    print(f"\n📈 สรุปผลการทดสอบ:")
    print(f"   📷 เฟรมทั้งหมด: {frame_count}")
    print(f"   🎯 เฟรมที่จับได้: {detection_count}")
    print(f"   📊 อัตราการตรวจจับ: {(detection_count/frame_count)*100:.1f}%")
    
    if detection_count == 0:
        print("⚠️ AI ไม่ได้ตรวจจับอะไรเลย - ต้องตรวจสอบการตั้งค่า")
    else:
        print("✅ AI ทำงานปกติ")

if __name__ == "__main__":
    test_ai_with_rtsp()
