#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎥 CAMERA CONNECTION TESTER
ทดสอบการเชื่อมต่อกล้องวงจรปิด RTSP
===============================================================================
"""

import cv2
import time
import sys
import datetime
import threading
import numpy as np

# กำหนดค่าการเชื่อมต่อกล้อง
CAMERA_CONFIG = {
    'main_camera': 'rtsp://ainok1:ainok123@192.168.1.100:554/stream1',
    'backup_camera': 'rtsp://ainok1:ainok123@192.168.1.101:554/stream1',
    'usb_camera': 0
}

def test_single_camera(camera_url, camera_name):
    """ทดสอบการเชื่อมต่อกล้องตัวเดียว"""
    print(f"\n🔍 ทดสอบ {camera_name}")
    print(f"📡 URL: {camera_url}")
    print("-" * 50)
    
    try:
        # เปิดการเชื่อมต่อ
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลด buffer lag
        
        if not cap.isOpened():
            print(f"❌ ไม่สามารถเชื่อมต่อกล้อง {camera_name}")
            return False
            
        print(f"✅ เชื่อมต่อกล้อง {camera_name} สำเร็จ!")
        
        # ลองอ่านกรอบแรก
        ret, frame = cap.read()
        if ret:
            height, width, channels = frame.shape
            print(f"📹 ความละเอียด: {width}x{height}")
            print(f"🎨 Channels: {channels}")
            
            # ทดสอบอ่านหลายกรอบ
            frames_read = 0
            start_time = time.time()
            
            for i in range(30):  # ทดสอบ 30 กรอบ
                ret, frame = cap.read()
                if ret:
                    frames_read += 1
                else:
                    break
                    
            elapsed_time = time.time() - start_time
            fps = frames_read / elapsed_time if elapsed_time > 0 else 0
            
            print(f"⚡ FPS ที่วัดได้: {fps:.2f}")
            print(f"📊 กรอบที่อ่านได้: {frames_read}/30")
            
            if fps > 10:
                print(f"🎉 {camera_name} ทำงานได้ดีเยี่ยม! (FPS > 10)")
            elif fps > 5:
                print(f"👍 {camera_name} ทำงานได้พอใช้ (FPS 5-10)")
            else:
                print(f"⚠️ {camera_name} ทำงานช้า (FPS < 5)")
                
        else:
            print(f"❌ ไม่สามารถอ่านกรอบภาพจาก {camera_name}")
            cap.release()
            return False
            
        cap.release()
        print(f"✅ ทดสอบ {camera_name} เสร็จสิ้น")
        return True
        
    except Exception as e:
        print(f"💥 เกิดข้อผิดพลาดขณะทดสอบ {camera_name}: {e}")
        return False

def test_all_cameras():
    """ทดสอบกล้องทั้งหมด"""
    print("=" * 60)
    print("🎥 ULTIMATE CAMERA CONNECTION TESTER")
    print(f"⏰ เวลาทดสอบ: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {}
    
    # ทดสอบกล้อง RTSP หลัก
    print("\n🔧 ทดสอบกล้อง RTSP หลัก...")
    results['main_camera'] = test_single_camera(
        CAMERA_CONFIG['main_camera'], 
        "กล้องหลัก (Main RTSP)"
    )
    
    # ทดสอบกล้อง USB (ถ้ามี)
    print("\n🔧 ทดสอบกล้อง USB...")
    results['usb_camera'] = test_single_camera(
        CAMERA_CONFIG['usb_camera'], 
        "กล้อง USB"
    )
    
    # สรุปผลการทดสอบ
    print("\n" + "=" * 60)
    print("📋 สรุปผลการทดสอบ")
    print("=" * 60)
    
    for camera, success in results.items():
        status = "✅ ใช้งานได้" if success else "❌ ใช้งานไม่ได้"
        print(f"   {camera}: {status}")
    
    working_cameras = sum(results.values())
    total_cameras = len(results)
    
    print(f"\n🎯 ผลรวม: {working_cameras}/{total_cameras} กล้องใช้งานได้")
    
    if working_cameras > 0:
        print("🎉 ระบบพร้อมใช้งาน!")
        if 'main_camera' in results and results['main_camera']:
            print("💡 แนะนำใช้กล้อง RTSP หลักสำหรับการผลิต")
    else:
        print("⚠️ ไม่พบกล้องที่ใช้งานได้ กรุณาตรวจสอบ:")
        print("   - การเชื่อมต่อเครือข่าย")
        print("   - ข้อมูลการเข้าสู่ระบบ (username/password)")
        print("   - IP Address และ Port")
        print("   - การตั้งค่ากล้อง RTSP")
    
    return results

def live_camera_test(camera_url="rtsp://ainok1:ainok123@192.168.1.100:554/stream1"):
    """แสดงภาพสดจากกล้อง (กด 'q' เพื่อออก)"""
    print(f"\n🎬 เปิดภาพสดจากกล้อง...")
    print(f"📡 URL: {camera_url}")
    print("💡 กด 'q' เพื่อปิดหน้าต่าง")
    
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้")
        return
        
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ ไม่สามารถอ่านกรอบภาพ")
                break
                
            frame_count += 1
            
            # เพิ่มข้อมูลลงในภาพ
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"Live Camera Test", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {current_time}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Ultimate Camera Test', frame)
            
            # กด 'q' เพื่อออก
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ ผู้ใช้หยุดการทดสอบ")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ ปิดการทดสอบภาพสดเรียบร้อย")

def main():
    """ฟังก์ชันหลัก"""
    while True:
        print("\n🎥 ULTIMATE CAMERA TESTER")
        print("1. ทดสอบการเชื่อมต่อกล้องทั้งหมด")
        print("2. ดูภาพสดจากกล้องหลัก")
        print("3. ทดสอบกล้อง RTSP เฉพาะ")
        print("4. ออกจากโปรแกรม")
        
        choice = input("\n👆 เลือกตัวเลือก (1-4): ").strip()
        
        if choice == '1':
            test_all_cameras()
        elif choice == '2':
            live_camera_test()
        elif choice == '3':
            url = input("🔗 ใส่ RTSP URL: ").strip()
            if url:
                live_camera_test(url)
        elif choice == '4':
            print("👋 ขอบคุณที่ใช้งาน!")
            break
        else:
            print("❌ ตัวเลือกไม่ถูกต้อง")
            
        input("\n⏳ กด Enter เพื่อดำเนินการต่อ...")

if __name__ == "__main__":
    main()
