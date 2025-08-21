#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 IP Camera Connection Tester
ทดสอบการเชื่อมต่อกล้อง IP ainok
"""

import cv2
import time
import os

def test_ip_camera():
    print("🔍 ทดสอบการเชื่อมต่อกล้อง IP Camera...")
    
    # URL ต่างๆ ที่จะทดสอบ
    test_urls = [
        "rtsp://ainok1:ainok123@192.168.1.100:554/stream1",
        "rtsp://ainok1:ainok123@192.168.1.100:554/stream",
        "rtsp://ainok1:ainok123@192.168.1.100:554/",
        "rtsp://192.168.1.100:554/stream1",
        "http://192.168.1.100:8080/video",
        0,  # USB Camera fallback
        1   # USB Camera fallback 2
    ]
    
    for i, url in enumerate(test_urls):
        print(f"\n🧪 ทดสอบ #{i+1}: {url}")
        
        try:
            # ตั้งค่า RTSP options
            if isinstance(url, str) and url.startswith('rtsp://'):
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'rtsp_transport;tcp;timeout;5000000;reconnect;1;'
                    'reconnect_at_eof;1;reconnect_streamed;1;'
                    'fflags;nobuffer;flags;low_delay'
                )
            
            cap = cv2.VideoCapture(url)
            
            if cap is not None and cap.isOpened():
                print(f"✅ เชื่อมต่อสำเร็จ!")
                
                # ทดสอบอ่านเฟรม
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"📐 ขนาดเฟรม: {width}x{height}")
                    print(f"🎨 ช่องสี: {frame.shape[2] if len(frame.shape) == 3 else 1}")
                    
                    # ทดสอบหลายเฟรม
                    success_count = 0
                    for test_frame in range(5):
                        ret, _ = cap.read()
                        if ret:
                            success_count += 1
                        time.sleep(0.1)
                    
                    print(f"📊 อ่านเฟรมสำเร็จ: {success_count}/5")
                    
                    if success_count >= 3:
                        print(f"🎉 กล้องทำงานปกติ! URL: {url}")
                        cap.release()
                        return url
                else:
                    print(f"❌ เชื่อมต่อได้แต่อ่านเฟรมไม่ได้")
            else:
                print(f"❌ เชื่อมต่อไม่ได้")
                
        except Exception as e:
            print(f"❌ ข้อผิดพลาด: {e}")
        
        finally:
            try:
                if 'cap' in locals():
                    cap.release()
            except:
                pass
    
    print(f"\n💔 ไม่พบกล้องที่ใช้งานได้")
    return None

def test_network_connectivity():
    """ทดสอบการเชื่อมต่อเครือข่าย"""
    import subprocess
    
    print("\n🌐 ทดสอบการเชื่อมต่อเครือข่าย...")
    
    try:
        # ทดสอบ ping กล้อง IP
        result = subprocess.run(['ping', '-n', '3', '192.168.1.100'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Ping กล้อง IP สำเร็จ - อุปกรณ์เชื่อมต่อได้")
        else:
            print("❌ Ping กล้อง IP ไม่สำเร็จ - ตรวจสอบ IP หรือเครือข่าย")
            print(f"Output: {result.stdout}")
    except Exception as e:
        print(f"❌ ไม่สามารถทดสอบ ping: {e}")

if __name__ == "__main__":
    print("🚀 เริ่มต้นการทดสอบระบบกล้อง")
    
    # ทดสอบเครือข่าย
    test_network_connectivity()
    
    # ทดสอบกล้อง
    working_url = test_ip_camera()
    
    if working_url:
        print(f"\n🎊 สรุป: กล้องที่ใช้งานได้คือ {working_url}")
        print("✅ คุณสามารถใช้ URL นี้ในแอพหลักได้")
    else:
        print(f"\n😢 สรุป: ไม่พบกล้องที่ใช้งานได้")
        print("💡 แนะนำ:")
        print("   1. ตรวจสอบ IP Address และ Port ของกล้อง")
        print("   2. ตรวจสอบ Username/Password")
        print("   3. ตรวจสอบว่ากล้องเปิด RTSP service หรือไม่")
        print("   4. ลองใช้ USB Camera แทน (0 หรือ 1)")
