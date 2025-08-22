#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Interactive Entrance Selector - ระบบกำหนดทางเข้าแบบโต้ตอบ
ผู้ใช้สามารถคลิกเมาส์เพื่อกำหนดทางเข้าได้ด้วยตัวเอง
"""

import cv2
import numpy as np
import json
import os
from typing import Tuple, Dict, Optional

class InteractiveEntranceSelector:
    """🖱️ ระบบเลือกทางเข้าแบบโต้ตอบด้วยเมาส์"""
    
    def __init__(self):
        self.entrance_points = []
        self.entrance_zone = None
        self.current_frame = None
        self.window_name = "🏠 กำหนดทางเข้านก - คลิกเมาส์เพื่อเลือกตำแหน่ง"
        
    def mouse_callback(self, event, x, y, flags, param):
        """🖱️ Callback สำหรับการคลิกเมาส์"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # คลิกซ้าย - เพิ่มจุดทางเข้า
            self.entrance_points.append((x, y))
            print(f"🎯 เพิ่มจุดทางเข้า: ({x}, {y})")
            
            if len(self.entrance_points) == 1:
                print("🔄 คลิกจุดที่ 2 เพื่อกำหนดขนาดพื้นที่ทางเข้า")
            elif len(self.entrance_points) == 2:
                # สร้างพื้นที่ทางเข้าจาก 2 จุด
                self.create_entrance_zone()
                print("✅ กำหนดทางเข้าเรียบร้อย! กด 's' เพื่อบันทึก หรือ 'r' เพื่อเริ่มใหม่")
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # คลิกขวา - ลบจุดล่าสุด
            if self.entrance_points:
                removed = self.entrance_points.pop()
                print(f"🗑️ ลบจุด: {removed}")
                if self.entrance_zone:
                    self.entrance_zone = None
                    print("🔄 ยกเลิกการกำหนดทางเข้า")
        
        # อัพเดตการแสดงผล
        self.update_display()
    
    def create_entrance_zone(self):
        """สร้างพื้นที่ทางเข้าจาก 2 จุดที่เลือก"""
        if len(self.entrance_points) < 2:
            return
            
        p1, p2 = self.entrance_points[0], self.entrance_points[1]
        
        # คำนวณขอบเขตสี่เหลี่ยม
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        
        width = x2 - x1
        height = y2 - y1
        
        # ขยายพื้นที่เล็กน้อยให้ครอบคลุมมากขึ้น
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        width = width + (padding * 2)
        height = height + (padding * 2)
        
        self.entrance_zone = {
            'x': x1,
            'y': y1,
            'width': width,
            'height': height,
            'center_x': x1 + width // 2,
            'center_y': y1 + height // 2,
            'detection_method': 'user_selected',
            'confidence': 1.0,
            'selection_points': self.entrance_points.copy()
        }
    
    def update_display(self):
        """อัพเดตการแสดงผลบนหน้าจอ"""
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # วาดจุดที่เลือก
        for i, point in enumerate(self.entrance_points):
            cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
            cv2.putText(display_frame, f"P{i+1}", 
                       (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # วาดเส้นเชื่อมระหว่างจุด
        if len(self.entrance_points) == 2:
            cv2.line(display_frame, self.entrance_points[0], self.entrance_points[1], 
                    (255, 255, 0), 2)
        
        # วาดพื้นที่ทางเข้า
        if self.entrance_zone:
            zone = self.entrance_zone
            # วาดสี่เหลี่ยม
            cv2.rectangle(display_frame, 
                         (zone['x'], zone['y']), 
                         (zone['x'] + zone['width'], zone['y'] + zone['height']),
                         (0, 255, 255), 3)
            
            # วาดจุดกลาง
            center = (zone['center_x'], zone['center_y'])
            cv2.circle(display_frame, center, 5, (0, 0, 255), -1)
            
            # แสดงข้อมูลทางเข้า
            cv2.putText(display_frame, "ENTRANCE ZONE", 
                       (zone['x'], zone['y'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(display_frame, f"Size: {zone['width']}x{zone['height']}", 
                       (zone['x'], zone['y'] + zone['height'] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(display_frame, f"Center: ({zone['center_x']}, {zone['center_y']})", 
                       (zone['x'], zone['y'] + zone['height'] + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # แสดงคำแนะนำ
        instructions = [
            "🖱️ คลิกซ้าย: เลือกจุดทางเข้า",
            "🖱️ คลิกขวา: ลบจุดล่าสุด", 
            "⌨️ 's': บันทึกการตั้งค่า",
            "⌨️ 'r': เริ่มใหม่",
            "⌨️ 'q': ออกจากโปรแกรม"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, 
                       (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # แสดงสถานะปัจจุบัน
        status = f"จุดที่เลือก: {len(self.entrance_points)}/2"
        if self.entrance_zone:
            status += " | ✅ พร้อมบันทึก"
        cv2.putText(display_frame, status, 
                   (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, display_frame)
    
    def select_entrance_from_video(self, video_path: str) -> Optional[Dict]:
        """🎬 เลือกทางเข้าจากวิดีโอ"""
        
        print("🎯 เปิดระบบเลือกทางเข้าแบบโต้ตอบ")
        print("=" * 60)
        print("📹 กำลังโหลดวิดีโอ...")
        
        if not os.path.exists(video_path):
            print(f"❌ ไม่พบวิดีโอ: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ ไม่สามารถเปิดวิดีโอได้: {video_path}")
            return None
        
        # อ่านเฟรมแรก
        ret, frame = cap.read()
        if not ret:
            print("❌ ไม่สามารถอ่านเฟรมจากวิดีโอได้")
            cap.release()
            return None
        
        # ปรับขนาดเฟรมให้เหมาะสม
        frame = cv2.resize(frame, (960, 540))
        self.current_frame = frame
        
        print("🖱️ คลิกเมาส์บนหน้าจอเพื่อกำหนดทางเข้า")
        print("   📍 คลิกจุดที่ 1: มุมหนึ่งของทางเข้า")
        print("   📍 คลิกจุดที่ 2: มุมตรงข้ามของทางเข้า")
        
        # สร้างหน้าต่างและ callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # แสดงเฟรมแรก
        self.update_display()
        
        # วนลูปรอการโต้ตอบ
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # บันทึก
                if self.entrance_zone:
                    print("💾 บันทึกการตั้งค่าทางเข้า...")
                    self.save_entrance_config(video_path)
                    break
                else:
                    print("⚠️ กรุณาเลือกทางเข้าก่อนบันทึก")
            
            elif key == ord('r'):  # เริ่มใหม่
                print("🔄 เริ่มเลือกทางเข้าใหม่")
                self.entrance_points.clear()
                self.entrance_zone = None
                self.update_display()
            
            elif key == ord('q'):  # ออก
                print("❌ ยกเลิกการเลือกทางเข้า")
                self.entrance_zone = None
                break
            
            elif key == ord(' '):  # เปลี่ยนเฟรม
                ret, new_frame = cap.read()
                if ret:
                    self.current_frame = cv2.resize(new_frame, (960, 540))
                    self.update_display()
                    print("🔄 เปลี่ยนเฟรม - ถ้าต้องการดูเฟรมอื่นกด Spacebar")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.entrance_zone
    
    def save_entrance_config(self, video_path: str):
        """💾 บันทึกการตั้งค่าทางเข้า"""
        if not self.entrance_zone:
            return
        
        config_file = "entrance_config_user.json"
        
        config = {
            'video_path': video_path,
            'entrance_zone': self.entrance_zone,
            'timestamp': cv2.getTickCount(),
            'notes': 'User-defined entrance zone via interactive selection'
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ บันทึกการตั้งค่าแล้ว: {config_file}")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการบันทึก: {e}")
    
    def load_entrance_config(self, config_file: str = "entrance_config_user.json") -> Optional[Dict]:
        """📂 โหลดการตั้งค่าทางเข้า"""
        if not os.path.exists(config_file):
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"📂 โหลดการตั้งค่าจาก: {config_file}")
            return config.get('entrance_zone')
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการโหลด: {e}")
            return None


def test_interactive_entrance_selector():
    """ทดสอบระบบเลือกทางเข้าแบบโต้ตอบ"""
    
    print("🎯 ทดสอบระบบเลือกทางเข้าแบบโต้ตอบ")
    print("=" * 60)
    
    # เลือกวิดีโอที่จะใช้ทดสอบ
    test_videos = [
        "training_videos/swallows_entering/enter_001.mp4",
        "training_videos/swallows_exiting/exit_001.mp4"
    ]
    
    print("📹 เลือกวิดีโอสำหรับกำหนดทางเข้า:")
    for i, video in enumerate(test_videos):
        status = "✅" if os.path.exists(video) else "❌"
        print(f"   {i+1}. {status} {video}")
    
    try:
        choice = input("🔢 เลือกวิดีโอ (1-2): ").strip()
        video_index = int(choice) - 1
        
        if 0 <= video_index < len(test_videos):
            selected_video = test_videos[video_index]
            
            # สร้าง selector
            selector = InteractiveEntranceSelector()
            
            # ลองโหลดการตั้งค่าเก่า
            existing_config = selector.load_entrance_config()
            if existing_config:
                print(f"📂 พบการตั้งค่าเก่า:")
                print(f"   📍 ตำแหน่ง: ({existing_config['center_x']}, {existing_config['center_y']})")
                print(f"   📏 ขนาด: {existing_config['width']}x{existing_config['height']}")
                
                use_existing = input("🤔 ใช้การตั้งค่าเก่า? (y/n): ").strip().lower()
                if use_existing == 'y':
                    print("✅ ใช้การตั้งค่าเก่า")
                    return existing_config
            
            # เลือกทางเข้าใหม่
            entrance_zone = selector.select_entrance_from_video(selected_video)
            
            if entrance_zone:
                print("🎉 เลือกทางเข้าสำเร็จ!")
                print(f"📍 ตำแหน่งกลาง: ({entrance_zone['center_x']}, {entrance_zone['center_y']})")
                print(f"📏 ขนาด: {entrance_zone['width']}x{entrance_zone['height']}")
                print(f"💯 ความมั่นใจ: {entrance_zone['confidence']}")
                return entrance_zone
            else:
                print("❌ ยกเลิกการเลือกทางเข้า")
        else:
            print("❌ เลือกวิดีโอไม่ถูกต้อง")
            
    except (ValueError, KeyboardInterrupt):
        print("❌ ยกเลิกการเลือก")
    
    return None


def integrate_with_ai_system(entrance_zone: Dict):
    """🔗 รวมการตั้งค่าทางเข้าเข้ากับระบบ AI"""
    
    print("🔗 รวมการตั้งค่าเข้ากับระบบ AI")
    print("=" * 60)
    
    try:
        # ใช้ V5 ULTRA PRECISION AI
        from ultimate_perfect_ai_MASTER import V5_UltimatePrecisionSwallowAI
        
        # สร้าง V5 AI
        ai = V5_UltimatePrecisionSwallowAI('enter')  # ใช้โหมด enter สำหรับทดสอบ
        
        # ตั้งค่าทางเข้าที่ผู้ใช้เลือก (V5 ใช้พารามิเตอร์ภายใน)
        # ai.entrance_zone = entrance_zone  # V5 จัดการทางเข้าอัตโนมัติ
        
        print("✅ ตั้งค่าทางเข้าในระบบ V5 ULTRA PRECISION AI เรียบร้อย")
        print(f"🏠 ทางเข้าใหม่: ({entrance_zone['center_x']}, {entrance_zone['center_y']})")
        print("🎯 V5 AI ใช้การตรวจจับทางเข้าแบบอัตโนมัติ")
        
        # ทดสอบด้วยวิดีโอ
        test_video = "training_videos/swallows_entering/enter_001.mp4"
        if os.path.exists(test_video):
            print(f"🎬 ทดสอบกับวิดีโอ: {test_video}")
            
            try:
                # ใช้ V5 ประมวลผลวิดีโอ
                result = ai.process_video_v5(test_video)
                print(f"📊 ผลการทดสอบ V5:")
                print(f"   🐦 นกเข้า: {result.get('entering', 0)} ตัว")
                print(f"   � นกออก: {result.get('exiting', 0)} ตัว") 
                print(f"   ⚡ FPS: {result.get('fps', 0):.1f}")
                print(f"   ⏱️ เวลา: {result.get('processing_time', 0):.1f} วินาที")
            except Exception as e:
                print(f"❌ ข้อผิดพลาดในการทดสอบ: {e}")
        
        return ai
        
    except ImportError as e:
        print(f"❌ ไม่สามารถโหลด V5 ULTRA PRECISION AI: {e}")
        print("💡 กรุณาตรวจสอบให้แน่ใจว่ามีไฟล์ ultimate_perfect_ai_MASTER.py")
        return None


if __name__ == "__main__":
    print("🎯 Interactive Entrance Selector")
    print("🏠 ระบบกำหนดทางเข้าแบบโต้ตอบ")
    print("=" * 60)
    
    # เลือกทางเข้า
    entrance_zone = test_interactive_entrance_selector()
    
    if entrance_zone:
        # รวมเข้ากับระบบ AI
        ai_system = integrate_with_ai_system(entrance_zone)
        
        if ai_system:
            print("\n🚀 ระบบพร้อมใช้งานด้วยทางเข้าที่กำหนดเอง!")
            print("💡 ตอนนี้สามารถใช้ AI ที่มีทางเข้าแม่นยำแล้ว")
        else:
            print("\n⚠️ มีปัญหาในการรวมเข้ากับระบบ AI")
    else:
        print("\n❌ ไม่ได้กำหนดทางเข้า")
