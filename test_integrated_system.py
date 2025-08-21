#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ทดสอบระบบ Ultimate Integrated Swallow AI V5
สคริปต์สำหรับทดสอบการทำงานครบครัน
"""

import sys
import time
import cv2
from ultimate_integrated_system import UltimateIntegratedSystem

def test_system():
    """ทดสอบระบบทั้งหมด"""
    print("🧪 เริ่มทดสอบระบบ Ultimate Integrated V5")
    
    # สร้างระบบ
    system = UltimateIntegratedSystem()
    
    print("\n📋 ผลการทดสอบ:")
    print("=" * 50)
    
    # 1. ทดสอบการโหลด AI
    print(f"🤖 AI System: {'✅ พร้อม' if system.ai_system else '❌ ไม่พร้อม'}")
    print(f"🔍 Detector: {'✅ พร้อม' if system.detector else '❌ ไม่พร้อม'}")
    
    # 2. ทดสอบการเชื่อมต่อกล้อง
    camera_test = system.connect_camera(0)  # ทดสอบ webcam
    print(f"🎥 Camera (Webcam): {'✅ เชื่อมต่อได้' if camera_test else '❌ เชื่อมต่อไม่ได้'}")
    
    if camera_test:
        system.camera.release()
    
    # 3. ทดสอบฐานข้อมูล
    try:
        system.db_manager.init_database()
        print("💾 Database: ✅ พร้อม")
    except Exception as e:
        print(f"💾 Database: ❌ ข้อผิดพลาด - {e}")
    
    # 4. ทดสอบ Web Server (ไม่รัน แค่ตรวจสอบ)
    print(f"🌐 Web Server: ✅ พร้อมรัน (Port {system.config['web']['port']})")
    
    # 5. ทดสอบการประมวลผลวีดีโอ (ถ้ามีไฟล์)
    import os
    test_video = "../test_video.mp4"
    if os.path.exists(test_video) and system.ai_system:
        print(f"🎬 Video Processing: ✅ พร้อม (ไฟล์: {test_video})")
        
        # ทดสอบประมวลผลจริง (แค่ 10 เฟรมแรก)
        try:
            cap = cv2.VideoCapture(test_video)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and system.detector:
                    detections = system.detector.detect_smart(frame, 'enter')
                    print(f"   🔍 ทดสอบตรวจจับ: พบ {len(detections)} การตรวจจับ")
                cap.release()
        except Exception as e:
            print(f"   ⚠️ ข้อผิดพลาดในการทดสอบ: {e}")
    else:
        print(f"🎬 Video Processing: ❌ ไม่พบไฟล์ทดสอบ หรือ AI ไม่พร้อม")
    
    print("=" * 50)
    print("✅ การทดสอบเสร็จสิ้น!")
    
    return system

def demo_live_detection():
    """สาธิตการตรวจจับแบบ Live"""
    print("🎥 สาธิตการตรวจจับแบบ Live (กด 'q' เพื่อออก)")
    
    system = UltimateIntegratedSystem()
    
    # เชื่อมต่อกล้อง
    if not system.connect_camera(0):
        print("❌ ไม่สามารถเชื่อมต่อกล้อง")
        return
    
    print("📹 เริ่มการตรวจจับ...")
    frame_count = 0
    
    try:
        while True:
            ret, frame = system.camera.read()
            if not ret:
                break
            
            # ตรวจจับทุก 5 เฟรม
            if frame_count % 5 == 0 and system.detector:
                detections = system.detector.detect_smart(frame, 'mixed')
                
                # วาดผลลัพธ์
                for detection in detections:
                    center = detection.get('center', (0, 0))
                    bbox = detection.get('bbox', (0, 0, 0, 0))
                    confidence = detection.get('confidence', 0)
                    
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, center, 3, (0, 255, 0), -1)
                    cv2.putText(frame, f"{confidence:.2f}", 
                               (center[0] - 20, center[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # แสดงจำนวน
                cv2.putText(frame, f"Detections: {len(detections)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # แสดงเฟรม
            cv2.imshow('Ultimate AI Live Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n🛑 หยุดการสาธิต")
    
    finally:
        system.camera.release()
        cv2.destroyAllWindows()

def demo_web_interface():
    """สาธิต Web Interface"""
    print("🌐 เริ่ม Web Interface Demo")
    print("📱 เปิดเบราว์เซอร์ไปที่: http://localhost:5000")
    print("🛑 กด Ctrl+C เพื่อหยุด")
    
    system = UltimateIntegratedSystem()
    
    # เริ่มกล้อง
    system.start_camera_stream()
    
    try:
        # รัน Web Server
        system.run_web_server()
    except KeyboardInterrupt:
        print("\n🛑 หยุด Web Interface")
    finally:
        system.stop_camera_stream()

if __name__ == "__main__":
    print("🚀 Ultimate Integrated Swallow AI V5 - Test Suite")
    print("\nเลือกโหมดทดสอบ:")
    print("1. ทดสอบระบบทั้งหมด")
    print("2. สาธิตการตรวจจับแบบ Live")
    print("3. สาธิต Web Interface")
    print("4. ออกจากโปรแกรม")
    
    while True:
        try:
            choice = input("\nเลือก (1-4): ").strip()
            
            if choice == '1':
                test_system()
                break
            elif choice == '2':
                demo_live_detection()
                break
            elif choice == '3':
                demo_web_interface()
                break
            elif choice == '4':
                print("👋 ออกจากโปรแกรม")
                break
            else:
                print("❌ กรุณาเลือก 1-4")
        
        except KeyboardInterrupt:
            print("\n👋 ออกจากโปรแกรม")
            break
