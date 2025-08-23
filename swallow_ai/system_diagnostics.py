#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 ULTIMATE SYSTEM DIAGNOSIS & CAMERA CHECKER
ระบบตรวจสอบการเชื่อมต่อและสถานะระบบ AI ครบวงจร
===============================================================================
"""

import cv2
import requests
import json
import time
import sys
import os
from datetime import datetime

class SystemDiagnostics:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.rtsp_url = "rtsp://ainok1:ainok123@192.168.1.100:554/stream1"
        
    def check_flask_server(self):
        """ตรวจสอบ Flask Server"""
        print("🌐 ตรวจสอบ Flask Server...")
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                print("✅ Flask Server ทำงานปกติ")
                return True
            else:
                print(f"⚠️ Flask Server ตอบกลับรหัส: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ ไม่สามารถเชื่อมต่อ Flask Server")
            return False
        except Exception as e:
            print(f"❌ ข้อผิดพลาด Flask Server: {e}")
            return False
    
    def check_video_feed(self):
        """ตรวจสอบ Video Feed API"""
        print("📹 ตรวจสอบ Video Feed...")
        try:
            response = requests.get(f"{self.base_url}/video_feed", timeout=15, stream=True)
            if response.status_code == 200:
                # อ่านข้อมูลเล็กน้อยเพื่อทดสอบ
                chunk = next(response.iter_content(1024))
                if b'Content-Type: image/jpeg' in chunk:
                    print("✅ Video Feed ทำงานปกติ")
                    return True
                else:
                    print("⚠️ Video Feed ตอบกลับข้อมูลไม่ถูกต้อง")
                    return False
            else:
                print(f"❌ Video Feed ตอบกลับรหัส: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ ข้อผิดพลาด Video Feed: {e}")
            return False
    
    def check_api_endpoints(self):
        """ตรวจสอบ API Endpoints"""
        print("🔌 ตรวจสอบ API Endpoints...")
        endpoints = [
            "/api/stats",
            "/api/statistics", 
            "/api/notifications",
            "/api/database-stats",
            "/api/anomaly-images"
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {endpoint} ทำงานปกติ")
                    results[endpoint] = True
                else:
                    print(f"⚠️ {endpoint} รหัส: {response.status_code}")
                    results[endpoint] = False
            except Exception as e:
                print(f"❌ {endpoint} ข้อผิดพลาด: {e}")
                results[endpoint] = False
        
        return results
    
    def check_camera_direct(self):
        """ตรวจสอบการเชื่อมต่อกล้องโดยตรง"""
        print("🎥 ตรวจสอบการเชื่อมต่อกล้องโดยตรง...")
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                print("❌ ไม่สามารถเปิดการเชื่อมต่อกล้อง")
                return False
            
            print("✅ เปิดการเชื่อมต่อกล้องสำเร็จ")
            
            # ทดสอบอ่านกรอบ
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, c = frame.shape
                print(f"✅ อ่านกรอบภาพสำเร็จ - ขนาด: {w}x{h}, ช่อง: {c}")
                
                # ทดสอบ FPS
                start_time = time.time()
                frame_count = 0
                
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"📊 FPS ที่วัดได้: {fps:.2f}")
                
                cap.release()
                return True
            else:
                print("❌ ไม่สามารถอ่านกรอบภาพ")
                cap.release()
                return False
                
        except Exception as e:
            print(f"❌ ข้อผิดพลาดการเชื่อมต่อกล้อง: {e}")
            return False
    
    def check_system_files(self):
        """ตรวจสอบไฟล์ระบบที่สำคัญ"""
        print("📂 ตรวจสอบไฟล์ระบบ...")
        
        important_files = [
            "app_working.py",
            "config.py", 
            "ultimate_ai_config.py",
            "yolov4.weights",
            "yolov4.cfg",
            "coco.names",
            "templates/index.html"
        ]
        
        results = {}
        for file_path in important_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {file_path} - ขนาด: {size:,} bytes")
                results[file_path] = True
            else:
                print(f"❌ {file_path} - ไม่พบไฟล์")
                results[file_path] = False
        
        return results
    
    def get_system_info(self):
        """รวบรวมข้อมูลระบบ"""
        print("💻 รวบรวมข้อมูลระบบ...")
        
        info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': sys.version,
            'opencv_version': cv2.__version__,
            'current_directory': os.getcwd(),
            'rtsp_url': self.rtsp_url,
            'flask_url': self.base_url
        }
        
        for key, value in info.items():
            print(f"📋 {key}: {value}")
        
        return info
    
    def run_full_diagnosis(self):
        """เรียกใช้การตรวจสอบครบวงจร"""
        print("=" * 70)
        print("🔍 ULTIMATE SYSTEM DIAGNOSIS - เริ่มตรวจสอบระบบ")
        print(f"⏰ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        results = {}
        
        # 1. ข้อมูลระบบ
        results['system_info'] = self.get_system_info()
        print()
        
        # 2. ไฟล์ระบบ
        results['system_files'] = self.check_system_files()
        print()
        
        # 3. การเชื่อมต่อกล้อง
        results['camera_connection'] = self.check_camera_direct()
        print()
        
        # 4. Flask Server
        results['flask_server'] = self.check_flask_server()
        print()
        
        # 5. Video Feed
        results['video_feed'] = self.check_video_feed()
        print()
        
        # 6. API Endpoints
        results['api_endpoints'] = self.check_api_endpoints()
        print()
        
        # สรุปผล
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """แสดงสรุปผลการตรวจสอบ"""
        print("=" * 70)
        print("📋 สรุปผลการตรวจสอบระบบ")
        print("=" * 70)
        
        # ตรวจสอบแต่ละหมวด
        sections = [
            ('🎥 การเชื่อมต่อกล้อง', results.get('camera_connection', False)),
            ('🌐 Flask Server', results.get('flask_server', False)),
            ('📹 Video Feed', results.get('video_feed', False))
        ]
        
        for name, status in sections:
            status_text = "✅ ปกติ" if status else "❌ มีปัญหา"
            print(f"{name}: {status_text}")
        
        # API Endpoints
        api_results = results.get('api_endpoints', {})
        working_apis = sum(1 for v in api_results.values() if v)
        total_apis = len(api_results)
        print(f"🔌 API Endpoints: {working_apis}/{total_apis} ทำงานปกติ")
        
        # ไฟล์ระบบ
        file_results = results.get('system_files', {})
        existing_files = sum(1 for v in file_results.values() if v)
        total_files = len(file_results)
        print(f"📂 ไฟล์ระบบ: {existing_files}/{total_files} พร้อมใช้งาน")
        
        # สถานะรวม
        print("\n" + "=" * 70)
        critical_systems = [
            results.get('camera_connection', False),
            results.get('flask_server', False),
            results.get('video_feed', False)
        ]
        
        if all(critical_systems):
            print("🎉 ระบบพร้อมใช้งานครบถ้วน!")
            print("🌐 เข้าถึงได้ที่: http://localhost:5000")
        elif any(critical_systems):
            print("⚠️ ระบบทำงานบางส่วน - ตรวจสอบปัญหาที่แสดงข้างต้น")
        else:
            print("❌ ระบบมีปัญหาหลายจุด - กรุณาตรวจสอบการตั้งค่า")
        
        print("=" * 70)

def main():
    """ฟังก์ชันหลัก"""
    diagnostics = SystemDiagnostics()
    
    while True:
        print("\n🔍 ULTIMATE SYSTEM DIAGNOSTICS")
        print("1. 🔍 ตรวจสอบระบบครบวงจร")
        print("2. 🎥 ทดสอบกล้องเฉพาะ")
        print("3. 🌐 ทดสอบ Web Server เฉพาะ")
        print("4. 🔌 ทดสอบ API เฉพาะ")
        print("5. 🚪 ออกจากโปรแกรม")
        
        choice = input("\n👆 เลือกตัวเลือก (1-5): ").strip()
        
        if choice == '1':
            diagnostics.run_full_diagnosis()
        elif choice == '2':
            diagnostics.check_camera_direct()
        elif choice == '3':
            diagnostics.check_flask_server()
            diagnostics.check_video_feed()
        elif choice == '4':
            diagnostics.check_api_endpoints()
        elif choice == '5':
            print("👋 ขอบคุณที่ใช้งาน System Diagnostics!")
            break
        else:
            print("❌ ตัวเลือกไม่ถูกต้อง")
        
        input("\n⏳ กด Enter เพื่อดำเนินการต่อ...")

if __name__ == "__main__":
    main()
