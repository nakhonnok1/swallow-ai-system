#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🔧 Direct Camera Connection Test
ทดสอบการเชื่อมต่อกล้องโดยตรงหลายแบบ
"""

import cv2
import numpy as np
import time
import os

def test_camera_sources():
    """ทดสอบแหล่งกล้องต่างๆ"""
    print("🎯 เริ่มทดสอบการเชื่อมต่อกล้อง")
    
    camera_sources = [
        # IP Camera - Primary RTSP
        "rtsp://ainok1:ainok123@192.168.1.100:554/stream1",
        "rtsp://ainok1:ainok123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://ainok1:ainok123@192.168.1.100/cam/realmonitor?channel=1&subtype=0",
        "rtsp://ainok1:ainok123@192.168.1.100:554/live",
        "rtsp://ainok1:ainok123@192.168.1.100:554/",
        
        # Alternative ports
        "rtsp://ainok1:ainok123@192.168.1.100:80/stream1",
        "rtsp://ainok1:ainok123@192.168.1.100:8554/stream1",
        
        # Different protocols
        "http://ainok1:ainok123@192.168.1.100/video.cgi",
        "http://192.168.1.100/cgi-bin/mjpeg?resolution=640x480",
        
        # USB Cameras
        0, 1, 2,
        
        # Test video file
        "test_video.mp4",
        "../test_video.mp4"
    ]
    
    working_sources = []
    
    for i, source in enumerate(camera_sources):
        print(f"\n📷 ทดสอบ [{i+1}/{len(camera_sources)}]: {source}")
        
        try:
            cap = cv2.VideoCapture(source)
            
            # Set timeout for network cameras
            if isinstance(source, str) and source.startswith(('rtsp', 'http')):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 seconds timeout
            
            # Check if camera opened successfully
            if cap.isOpened():
                print(f"✅ เปิดกล้องสำเร็จ")
                
                # Try to read frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✅ อ่านเฟรมสำเร็จ - ขนาด: {width}x{height}")
                    
                    # Test multiple frames
                    frame_count = 0
                    for _ in range(5):
                        ret, frame = cap.read()
                        if ret:
                            frame_count += 1
                        time.sleep(0.1)
                    
                    if frame_count >= 3:
                        print(f"🎉 กล้องทำงานเสถียร ({frame_count}/5 เฟรม)")
                        working_sources.append(source)
                        
                        # Save test frame
                        if ret and frame is not None:
                            test_filename = f"test_frame_{i}.jpg"
                            cv2.imwrite(test_filename, frame)
                            print(f"💾 บันทึกเฟรมทดสอบ: {test_filename}")
                    else:
                        print(f"⚠️ กล้องไม่เสถียร ({frame_count}/5 เฟรม)")
                else:
                    print("❌ ไม่สามารถอ่านเฟรมได้")
            else:
                print("❌ ไม่สามารถเปิดกล้องได้")
                
        except Exception as e:
            print(f"❌ ข้อผิดพลาด: {e}")
        
        finally:
            try:
                cap.release()
            except:
                pass
    
    print(f"\n🎯 สรุปผลการทดสอบ:")
    if working_sources:
        print(f"✅ พบกล้องที่ใช้งานได้: {len(working_sources)} แหล่ง")
        for source in working_sources:
            print(f"   - {source}")
    else:
        print("❌ ไม่พบกล้องที่ใช้งานได้")
    
    return working_sources

def test_network_connection():
    """ทดสอบการเชื่อมต่อเครือข่าย"""
    print("\n🌐 ทดสอบการเชื่อมต่อเครือข่าย")
    
    import subprocess
    import socket
    
    # Test ping to camera IP
    try:
        result = subprocess.run(['ping', '192.168.1.100', '-n', '4'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ping ไปยัง 192.168.1.100 สำเร็จ")
        else:
            print("❌ Ping ไปยัง 192.168.1.100 ล้มเหลว")
            print(result.stdout)
    except Exception as e:
        print(f"❌ ไม่สามารถ ping ได้: {e}")
    
    # Test port connectivity
    ports_to_test = [554, 80, 8554, 8080]
    for port in ports_to_test:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(('192.168.1.100', port))
            if result == 0:
                print(f"✅ Port {port} เปิดอยู่")
            else:
                print(f"❌ Port {port} ปิดอยู่")
            sock.close()
        except Exception as e:
            print(f"❌ ไม่สามารถทดสอบ port {port}: {e}")

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 เริ่มการทดสอบกล้องแบบครบวงจร")
    print("=" * 50)
    
    # Test network first
    test_network_connection()
    
    print("\n" + "=" * 50)
    
    # Test camera sources
    working_sources = test_camera_sources()
    
    print("\n" + "=" * 50)
    print("🏁 การทดสอบเสร็จสิ้น")
    
    if working_sources:
        print(f"\n💡 แนะนำให้ใช้แหล่งกล้องแรกที่ทำงาน: {working_sources[0]}")
        
        # Update config
        config_update = f'''
# อัปเดตไฟล์ config.json
VIDEO_SOURCE = "{working_sources[0]}"
'''
        print(config_update)
    else:
        print("\n⚠️ ไม่พบกล้องที่ใช้งานได้ กรุณาตรวจสอบ:")
        print("   1. การเชื่อมต่อเครือข่าย")
        print("   2. ที่อยู่ IP ของกล้อง")
        print("   3. Username และ Password")
        print("   4. Port และ stream path")

if __name__ == "__main__":
    main()
