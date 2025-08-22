#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Source Detector - หาแหล่งกล้องที่ใช้ได้
"""
import cv2

def find_available_cameras():
    print("🔍 Scanning for available camera sources...")
    available_sources = []
    
    # ทดสอบ camera index 0-5
    for i in range(6):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    print(f"✅ Camera {i}: Working - Resolution {w}x{h}")
                    available_sources.append(i)
                else:
                    print(f"❌ Camera {i}: Opened but no frame")
                cap.release()
            else:
                print(f"❌ Camera {i}: Cannot open")
        except Exception as e:
            print(f"❌ Camera {i}: Error - {e}")
    
    # ทดสอบ DirectShow backend (Windows)
    for i in range(3):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    print(f"✅ Camera {i} (DirectShow): Working - Resolution {w}x{h}")
                    available_sources.append(f"{i}_dshow")
                cap.release()
        except Exception as e:
            print(f"❌ Camera {i} (DirectShow): Error - {e}")
    
    # สร้างไฟล์วิดีโอทดสอบ
    print("\n🎬 Creating test video...")
    create_test_video()
    
    return available_sources

def create_test_video():
    """สร้างไฟล์วิดีโอทดสอบสำหรับใช้แทนกล้อง"""
    import numpy as np
    import os
    
    try:
        video_path = 'test_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        for frame_num in range(100):  # 5 วินาที
            # สร้างเฟรมทดสอบ
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # พื้นหลังสีเทา
            frame[:] = (50, 50, 50)
            
            # วาดวงกลมเคลื่อนที่ (จำลองนก)
            center_x = int(320 + 200 * np.sin(frame_num * 0.1))
            center_y = int(240 + 100 * np.cos(frame_num * 0.15))
            cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), -1)
            
            # ข้อความ
            cv2.putText(frame, f'Test Video Frame {frame_num}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Simulated Bird Movement', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # เส้นนับ
            cv2.line(frame, (480, 60), (480, 420), (0, 255, 255), 3)
            cv2.putText(frame, 'IN', (430, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'OUT', (490, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Test video created: {os.path.abspath(video_path)}")
        return os.path.abspath(video_path)
        
    except Exception as e:
        print(f"❌ Failed to create test video: {e}")
        return None

if __name__ == '__main__':
    sources = find_available_cameras()
    print(f"\n📊 Summary: Found {len(sources)} working camera sources")
    for src in sources:
        print(f"  - {src}")
    
    if not sources:
        print("\n💡 No cameras found. You can use the test video:")
        print("   $env:VIDEO_SOURCE = 'test_video.mp4'")
    else:
        print(f"\n💡 Recommended camera source: {sources[0]}")
        print(f"   $env:VIDEO_SOURCE = '{sources[0]}'")
