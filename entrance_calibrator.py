#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Entrance Calibrator
เครื่องมือคาลิเบรต "กรอบช่องเข้า + เส้น 4 เส้น" ให้พอดีกับวีดีโอเทรน โดยไม่ทับไฟล์เดิม
- โหลด/บันทึกค่าลง entrance_config.json
- ใช้เมาส์ลากกรอบ + ปรับเส้นแบบอินเทอร์แอคทีฟ
"""

import cv2
import json
import os
from datetime import datetime

DEFAULT = {
    'frame_width': 960,
    'frame_height': 540,
    'entrance_zone': {'x': 450, 'y': 200, 'width': 100, 'height': 80, 'center_y': 240},
    'counting_lines': {'gate': 550},
    'line_mode': '1'  # '1' = 1 gate line only
}

BASE_DIR = os.path.dirname(__file__)
CFG_FILE = os.path.join(BASE_DIR, 'entrance_config.json')

class EntranceCalibrator:
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.cfg = self.load_cfg()
        self.dragging = False
        self.drag_offset = (0, 0)
        self.active_line = None
        self.window = 'Entrance Calibrator'

    def load_cfg(self):
        if os.path.exists(CFG_FILE):
            with open(CFG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # เติมค่าเริ่มต้นหากไม่มี
                if 'line_mode' not in data:
                    data['line_mode'] = '1'
                if 'counting_lines' not in data:
                    data['counting_lines'] = DEFAULT['counting_lines']
                if 'entrance_zone' not in data:
                    data['entrance_zone'] = DEFAULT['entrance_zone']
                return data
        return DEFAULT.copy()

    def save_cfg(self):
        with open(CFG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cfg, f, indent=2, ensure_ascii=False)
        print(f'💾 บันทึกค่า: {CFG_FILE}')

    def on_mouse(self, event, x, y, flags, param):
        ez = self.cfg['entrance_zone']
        cl = self.cfg['counting_lines']
        mode = self.cfg.get('line_mode', '1')
        if event == cv2.EVENT_LBUTTONDOWN:
            # ตรวจสอบว่าคลิกภายในกรอบเข้า
            if ez['x'] <= x <= ez['x']+ez['width'] and ez['y'] <= y <= ez['y']+ez['height']:
                self.dragging = True
                self.drag_offset = (x - ez['x'], y - ez['y'])
                self.active_line = None
            else:
                # ตรวจสอบคลิกใกล้เส้น (±8px)
                if mode == 'rect':
                    visible_lines = []
                elif mode == '1':
                    visible_lines = ['gate']
                elif mode == '2':
                    visible_lines = ['pre_entry', 'main']
                else:
                    visible_lines = ['pre_entry', 'entry', 'main', 'exit']
                for name in visible_lines:
                    lx = cl.get(name)
                    if lx is None:
                        continue
                    if abs(x - lx) <= 8:
                        self.active_line = name
                        break
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            ez['x'] = max(0, x - self.drag_offset[0])
            ez['y'] = max(0, y - self.drag_offset[1])
            ez['center_y'] = ez['y'] + ez['height']//2
        elif event == cv2.EVENT_MOUSEMOVE and self.active_line:
            cl[self.active_line] = max(0, x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.active_line = None

    def draw_overlay(self, frame):
        ez = self.cfg['entrance_zone']
        cl = self.cfg['counting_lines']
        mode = self.cfg.get('line_mode', '1')
        out = frame.copy()
        # กรอบช่องเข้า
        cv2.rectangle(out, (ez['x'], ez['y']), (ez['x']+ez['width'], ez['y']+ez['height']), (0,255,255), 2)
        cv2.putText(out, 'ENTRANCE', (ez['x'], ez['y']-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # เส้นนับ
        colors = {'pre_entry': (255,255,0), 'entry': (0,255,0), 'main': (0,0,255), 'exit': (255,0,255), 'gate': (0,165,255)}
        h = out.shape[0]
        if mode == 'rect':
            visible_lines = []
        elif mode == '1':
            visible_lines = ['gate']
        elif mode == '2':
            visible_lines = ['pre_entry', 'main']
        else:
            visible_lines = ['pre_entry', 'entry', 'main', 'exit']
        for name in visible_lines:
            lx = cl.get(name)
            if lx is None:
                continue
            color = colors.get(name, (200,200,200))
            
            # วาดเส้นเกตให้สั้นลง เท่ากับความสูงของกรอบประตู
            if name == 'gate':
                ez = self.cfg['entrance_zone']
                y1 = ez['y']
                y2 = ez['y'] + ez['height']
                cv2.line(out, (lx, y1), (lx, y2), color, 3)  # เส้นหนาขึ้นเพื่อให้เห็นชัด
                cv2.putText(out, 'GATE', (lx+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # เส้นอื่นยังคงยาวเต็มหน้าจอ
                cv2.line(out, (lx, 0), (lx, h), color, 2)
                cv2.putText(out, name.upper(), (lx+5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # แถบช่วยเหลือและที่บันทึกไฟล์
        source = getattr(self, 'current_source', 'None')
        mode_text = {'rect': 'RECT-ONLY', '1': '1-LINE', '2': '2-LINE', '4': '4-LINE'}.get(mode, mode)
        help1 = f"MODE: {mode_text} (0/1/2/4) | s=Save | o=Open video | q=Quit | Source: {source}"
        help2 = f"Save path: {os.path.abspath(CFG_FILE)}"
        cv2.putText(out, help1, (10, h-24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(out, help2, (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        return out

    def run(self):
        # เปิดไฟล์วิดีโอ ถ้าไม่มีให้ลองเปิดเว็บแคม (หลีกเลี่ยงจอดำ)
        cap = None
        self.current_source = 'None'
        if self.video_path and os.path.exists(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            if getattr(cap, 'isOpened', lambda: False)():
                self.current_source = os.path.basename(self.video_path)
        if cap is None or not getattr(cap, 'isOpened', lambda: False)():
            try:
                tmp = cv2.VideoCapture(0)
                if getattr(tmp, 'isOpened', lambda: False)():
                    cap = tmp
                    self.current_source = 'Webcam(0)'
                else:
                    tmp.release()
            except Exception:
                cap = None
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        print('วิธีใช้: ลากกรอบช่องเข้า | คลิกใกล้เส้นเพื่อเลือกแล้วลาก | 2/4=สลับโหมดเส้น | s=บันทึก | q=ออก')
        frame = None
        import numpy as np
        while True:
            if cap is not None:
                ok, f = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, f = cap.read()
                if not ok:
                    f = np.zeros((DEFAULT['frame_height'], DEFAULT['frame_width'], 3), dtype=np.uint8)
                frame = cv2.resize(f, (DEFAULT['frame_width'], DEFAULT['frame_height']))
            else:
                frame = np.zeros((DEFAULT['frame_height'], DEFAULT['frame_width'], 3), dtype=np.uint8)
            view = self.draw_overlay(frame)
            cv2.imshow(self.window, view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('s'):
                self.save_cfg()
            elif key == ord('1'):
                self.cfg['line_mode'] = '1'
                cl = self.cfg['counting_lines']
                # ลบเส้นเก่าและเก็บเฉพาะ gate
                gate_x = cl.get('gate', 570)
                self.cfg['counting_lines'] = {'gate': gate_x}
                cl = self.cfg['counting_lines']
                cl.setdefault('gate', 550)
            elif key == ord('o'):
                # เปิดกล่องเลือกไฟล์วิดีโอ
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk(); root.withdraw()
                    filepath = filedialog.askopenfilename(title='เลือกไฟล์วิดีโอ',
                                                          filetypes=[('Video Files','*.mp4;*.avi;*.mov;*.mkv'), ('All Files','*.*')])
                    root.destroy()
                    if filepath and os.path.exists(filepath):
                        if cap is not None:
                            cap.release()
                        cap = cv2.VideoCapture(filepath)
                        if getattr(cap, 'isOpened', lambda: False)():
                            self.video_path = filepath
                            self.current_source = os.path.basename(filepath)
                            print(f"🎞️ เปิดวิดีโอ: {filepath}")
                        else:
                            print("⚠️ เปิดวิดีโอไม่สำเร็จ")
                    else:
                        print("❌ ไม่ได้เลือกไฟล์")
                except Exception as e:
                    print(f"❌ ไม่สามารถเปิดกล่องเลือกไฟล์: {e}")
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()
        if cap is not None:
            cap.release()

if __name__ == '__main__':
    # ค้นหาวิดีโออัตโนมัติใต้โฟลเดอร์ training_videos ทั้งหมด
    base = os.path.join(BASE_DIR, 'training_videos')
    video = None
    try:
        if os.path.isdir(base):
            for dirpath, _, filenames in os.walk(base):
                for name in filenames:
                    if name.lower().endswith(('.mp4','.avi','.mov','.mkv')):
                        video = os.path.join(dirpath, name)
                        raise StopIteration
    except StopIteration:
        pass
    EntranceCalibrator(video).run()
