#!/usr/bin/env python3
"""
🧪 ทดสอบระบบ AI ตรวจจับสิ่งแปลกปลอม
เพื่อยืนยันว่าระบบสามารถจับคน สัตว์ และสิ่งแปลกปลอมได้
"""

import cv2
import requests
import json
import time
from advanced_object_detector import AdvancedObjectDetector

def test_object_detector():
    """ทดสอบ AdvancedObjectDetector โดยตรง"""
    print("🧪 ทดสอบ Advanced Object Detector...")
    
    detector = AdvancedObjectDetector()
    
    if not detector.model:
        print("❌ Model ไม่โหลด")
        return False
    
    # ทดสอบด้วยกล้อง
    cap = cv2.VideoCapture("rtsp://ainok1:ainok123@192.168.1.100:554/stream1")
    
    if not cap.isOpened():
        print("❌ ไม่สามารถเชื่อมต่อกล้อง")
        return False
    
    print("✅ เชื่อมต่อกล้องสำเร็จ")
    print("🎯 กำลังทดสอบการตรวจจับ...")
    print("📝 ยืนหรือเคลื่อนไหวหน้ากล้องเพื่อทดสอบ")
    
    frame_count = 0
    detection_count = 0
    
    while frame_count < 100:  # ทดสอบ 100 เฟรม
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # ทดสอบการตรวจจับ
        detections = detector.detect_objects(frame)
        
        if detections:
            detection_count += 1
            print(f"🎯 เฟรม {frame_count}: พบ {len(detections)} วัตถุ")
            
            for i, det in enumerate(detections):
                print(f"   {i+1}. {det['object_name']} - {det['confidence']:.1%} ({det['priority']})")
        
        # แสดงภาพ (ถ้าต้องการ)
        if detections:
            display_frame = detector.draw_detections(frame, detections)
            cv2.imshow('Detection Test', cv2.resize(display_frame, (640, 480)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    success_rate = (detection_count / frame_count) * 100
    print(f"\n📊 ผลการทดสอบ:")
    print(f"   🎯 เฟรมที่มีการตรวจจับ: {detection_count}/{frame_count}")
    print(f"   📈 อัตราความสำเร็จ: {success_rate:.1f}%")
    
    return success_rate > 5  # ถือว่าสำเร็จถ้าตรวจจับได้มากกว่า 5%

def test_api_endpoints():
    """ทดสอบ API endpoints"""
    print("\n🌐 ทดสอบ API endpoints...")
    
    base_url = "http://127.0.0.1:5000"
    
    endpoints = [
        "/api/object-detection/status",
        "/api/object-detection/stats", 
        "/api/object-detection/alerts"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {endpoint}: OK")
                print(f"   📄 ข้อมูล: {json.dumps(data, indent=2, ensure_ascii=False)}")
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")

def test_live_detection():
    """ทดสอบการตรวจจับแบบ real-time ผ่าน API"""
    print("\n🔴 ทดสอบการตรวจจับแบบ real-time...")
    
    base_url = "http://127.0.0.1:5000"
    
    for i in range(10):
        try:
            # ตรวจสอบสถานะ
            response = requests.get(f"{base_url}/api/object-detection/status", timeout=3)
            if response.status_code == 200:
                status = response.json()
                print(f"🔄 รอบ {i+1}: Model loaded={status.get('model_loaded')}, Enabled={status.get('enabled')}")
            
            # ตรวจสอบแจ้งเตือนล่าสุด
            response = requests.get(f"{base_url}/api/object-detection/alerts", timeout=3)
            if response.status_code == 200:
                alerts = response.json()
                if alerts:
                    print(f"🚨 มีแจ้งเตือน {len(alerts)} รายการ:")
                    for alert in alerts[:3]:  # แสดง 3 รายการล่าสุด
                        print(f"   - {alert.get('object_name')} ({alert.get('confidence'):.1%}) เวลา {alert.get('timestamp')}")
                else:
                    print("   ไม่มีแจ้งเตือน")
            
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 เริ่มทดสอบระบบ AI ตรวจจับสิ่งแปลกปลอม")
    print("=" * 60)
    
    # ทดสอบ Object Detector
    success = test_object_detector()
    
    if success:
        print("✅ การทดสอบ Object Detector สำเร็จ!")
    else:
        print("❌ การทดสอบ Object Detector ล้มเหลว")
    
    # ทดสอบ API
    test_api_endpoints()
    
    # ทดสอบการตรวจจับแบบ real-time
    test_live_detection()
    
    print("\n🏁 การทดสอบเสร็จสิ้น")
