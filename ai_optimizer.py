# -*- coding: utf-8 -*-
"""
AI Performance Optimizer
สำหรับปรับปรุงประสิทธิภาพการประมวลผลของระบบ AI
"""

import time
import threading
import queue
from typing import Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from config import Config

class PerformanceOptimizer:
    """คลาสสำหรับปรับปรุงประสิทธิภาพการทำงานของ AI"""
    
    def __init__(self, model: YOLO):
        self.model = model
        self.config = Config()
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.is_running = False
        
        # Performance settings
        self.frame_skip = getattr(self.config, 'FRAME_SKIP', 2)
        self.performance_mode = getattr(self.config, 'PERFORMANCE_MODE', 'FAST')
        self.roi_enabled = getattr(self.config, 'ROI_ENABLED', True)
        
        # ROI settings
        self.roi_x = getattr(self.config, 'ROI_X', 0)
        self.roi_y = getattr(self.config, 'ROI_Y', 100)
        self.roi_width = getattr(self.config, 'ROI_WIDTH', 640)
        self.roi_height = getattr(self.config, 'ROI_HEIGHT', 280)
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
    def optimize_frame(self, frame: np.ndarray) -> np.ndarray:
        """ปรับปรุงภาพก่อนการประมวลผล"""
        
        # ROI Processing - ประมวลผลเฉพาะพื้นที่ที่สำคัญ
        if self.roi_enabled:
            h, w = frame.shape[:2]
            roi_x = min(self.roi_x, w - 1)
            roi_y = min(self.roi_y, h - 1)
            roi_w = min(self.roi_width, w - roi_x)
            roi_h = min(self.roi_height, h - roi_y)
            
            frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # Performance mode optimization
        if self.performance_mode == 'FAST':
            # ลดความละเอียดเพื่อความเร็ว
            if frame.shape[1] > 416:
                scale = 416 / frame.shape[1]
                new_width = 416
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
                
        elif self.performance_mode == 'BALANCED':
            # ความละเอียดปานกลาง
            if frame.shape[1] > 640:
                scale = 640 / frame.shape[1]
                new_width = 640
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
                
        # ACCURATE mode ใช้ความละเอียดเต็ม
        
        return frame
    
    def start_async_processing(self):
        """เริ่มการประมวลผลแบบ async"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_async_processing(self):
        """หยุดการประมวลผลแบบ async"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
    
    def _process_frames(self):
        """ประมวลผลภาพในเธรดแยก"""
        while self.is_running:
            try:
                # รับภาพจาก queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # ปรับปรุงภาพ
                optimized_frame = self.optimize_frame(frame)
                
                # ทำ detection
                start_time = time.time()
                results = self.model(optimized_frame, verbose=False)
                process_time = time.time() - start_time
                
                # ส่งผลลัพธ์ไปยัง result queue
                self.result_queue.put({
                    'results': results,
                    'process_time': process_time,
                    'original_frame': frame,
                    'optimized_frame': optimized_frame
                })
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """เพิ่มภาพเข้าสู่ queue สำหรับการประมวลผล"""
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            # ถ้า queue เต็ม ให้ข้ามเฟรมนี้
            return False
    
    def get_result(self, timeout: float = 0.1) -> Optional[dict]:
        """รับผลลัพธ์การประมวลผล"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def update_fps(self):
        """อัพเดตการนับ FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_performance_stats(self) -> dict:
        """รับสถิติประสิทธิภาพ"""
        return {
            'fps': round(self.current_fps, 1),
            'frame_queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'performance_mode': self.performance_mode,
            'roi_enabled': self.roi_enabled,
            'frame_skip': self.frame_skip
        }
    
    def adjust_performance_mode(self, fps_target: float = 15.0):
        """ปรับโหมดประสิทธิภาพอัตโนมัติตาม FPS ปัจจุบัน"""
        if self.current_fps < fps_target * 0.5:  # น้อยกว่า 50% ของเป้าหมาย
            if self.performance_mode != 'FAST':
                self.performance_mode = 'FAST'
                print(f"🔄 Switched to FAST mode (FPS: {self.current_fps:.1f})")
                
        elif self.current_fps > fps_target * 1.2:  # มากกว่า 120% ของเป้าหมาย
            if self.performance_mode != 'ACCURATE':
                self.performance_mode = 'ACCURATE'
                print(f"🔄 Switched to ACCURATE mode (FPS: {self.current_fps:.1f})")
                
        elif self.current_fps > fps_target * 0.8:  # 80-120% ของเป้าหมาย
            if self.performance_mode != 'BALANCED':
                self.performance_mode = 'BALANCED'
                print(f"🔄 Switched to BALANCED mode (FPS: {self.current_fps:.1f})")

class SmartFrameSelector:
    """เลือกเฟรมที่สำคัญสำหรับการประมวลผล"""
    
    def __init__(self, skip_frames: int = 2):
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.last_frame = None
        self.motion_threshold = 30.0
    
    def should_process_frame(self, frame: np.ndarray) -> bool:
        """ตัดสินใจว่าควรประมวลผลเฟรมนี้หรือไม่"""
        
        # Frame skipping
        self.frame_counter += 1
        if self.frame_counter % (self.skip_frames + 1) != 0:
            return False
        
        # Motion detection
        if self.last_frame is not None:
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            
            # คำนวณความแตกต่างระหว่างเฟรม
            diff = cv2.absdiff(gray_current, gray_last)
            motion_score = np.mean(diff)
            
            # ถ้าไม่มีการเคลื่อนไหวมาก ให้ข้ามเฟรม
            if motion_score < self.motion_threshold:
                return False
        
        self.last_frame = frame.copy()
        return True
    
    def reset(self):
        """รีเซ็ตตัวเลือกเฟรม"""
        self.frame_counter = 0
        self.last_frame = None
