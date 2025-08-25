#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED VIDEO PROCESSING - Single Stream, Dual AI Detection
==========================================================
ระบบประมวลผลวิดีโอแบบรวมศูนย์ที่ใช้วิดีโอสตรีมเดียว
แต่มี AI Detection 2 ระบบทำงานร่วมกันอย่างมีประสิทธิภาพ
==========================================================
"""

import cv2
import time
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Optional

# Global variables
current_frame = None
frame_lock = threading.Lock()
video_feed_active = True

logger = logging.getLogger('unified_video')

class UnifiedVideoProcessor:
    """Unified video processor with dual AI detection systems"""
    
    def __init__(self, camera_manager, ai_detector, bird_counter, intruder_stats, performance_monitor):
        self.camera_manager = camera_manager
        self.ai_detector = ai_detector
        self.bird_counter = bird_counter
        self.intruder_stats = intruder_stats
        self.performance_monitor = performance_monitor
        
        # Performance optimization settings
        self.detection_interval = 3  # Run AI detection every 3 frames
        self.fps_limit = 30
        
        # Caching for smooth display
        self._last_bird_count = 0
        self._last_intruder_count = 0
        self._cached_detections = {'birds': [], 'intruders': []}
        
    def process_single_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """Process single frame with unified dual AI detection"""
        try:
            # Enhance frame quality
            enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
            processed_frame = enhanced_frame.copy()
            
            bird_count = 0
            intruder_count = 0
            
            # Optimize AI detection frequency to prevent stuttering
            should_detect = (frame_count % self.detection_interval == 0) or (frame_count < 30)
            
            if should_detect:
                logger.info(f"🤖 UNIFIED AI Processing frame {frame_count}")
                
                # ============ BIRD AI DETECTION (BLUE BOXES) ============
                bird_detections = self._detect_birds(enhanced_frame, frame_count)
                bird_count = self._draw_bird_detections(processed_frame, bird_detections)
                
                # ============ INTRUDER AI DETECTION (RED BOXES) ============  
                intruder_detections = self._detect_intruders(enhanced_frame, frame_count)
                intruder_count = self._draw_intruder_detections(processed_frame, intruder_detections)
                
                # Cache results for smoother display
                self._last_bird_count = bird_count
                self._last_intruder_count = intruder_count
                self._cached_detections = {
                    'birds': bird_detections,
                    'intruders': intruder_detections
                }
                
            else:
                # Use cached results for smoother display
                bird_count = self._last_bird_count
                intruder_count = self._last_intruder_count
                
                # Draw cached detections for visual continuity
                if self._cached_detections['birds']:
                    self._draw_bird_detections(processed_frame, self._cached_detections['birds'])
                if self._cached_detections['intruders']:
                    self._draw_intruder_detections(processed_frame, self._cached_detections['intruders'])
            
            # Add unified system info overlay
            self._add_system_overlay(processed_frame, frame_count, bird_count, intruder_count)
            
            # Update statistics
            self._update_statistics(bird_count, intruder_count)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Unified frame processing error: {e}")
            return frame
    
    def _detect_birds(self, frame: np.ndarray, frame_count: int) -> List[Dict[str, Any]]:
        """Detect birds using bird AI system"""
        try:
            bird_detections = self.ai_detector.detect_birds(frame)
            
            # Add test detection for demonstration (removable in production)
            if frame_count % 120 == 0:
                bird_detections.append({
                    'bbox': [100, 100, 150, 120],
                    'confidence': 0.92,
                    'class': 'swallow',
                    'type': 'bird_detection'
                })
            
            return bird_detections
            
        except Exception as e:
            logger.error(f"Bird detection error: {e}")
            return []
    
    def _detect_intruders(self, frame: np.ndarray, frame_count: int) -> List[Dict[str, Any]]:
        """Detect intruders/objects using intruder AI system"""
        try:
            intruder_detections = self.ai_detector.detect_intruders(frame)
            
            # Fallback detection with bird filtering
            if not intruder_detections and frame_count % 180 == 0:
                general_detections = self.ai_detector.detect_objects(frame)
                intruder_detections = self._filter_non_bird_objects(general_detections)
            
            return intruder_detections
            
        except Exception as e:
            logger.error(f"Intruder detection error: {e}")
            return []
    
    def _filter_non_bird_objects(self, detections: List[Dict]) -> List[Dict]:
        """Filter out birds and small animals from general detections"""
        bird_exclusions = [
            'bird', 'swallow', 'pigeon', 'dove', 'sparrow', 'crow', 'eagle',
            'cat', 'kitten', 'mouse', 'rat', 'squirrel', 'rabbit'
        ]
        
        filtered = []
        for detection in detections:
            class_name = detection.get('class', '').lower()
            if not any(term in class_name for term in bird_exclusions):
                # Add threat level assessment
                if class_name in ['person', 'human', 'man', 'woman']:
                    detection['threat_level'] = 'medium'
                elif class_name in ['car', 'truck', 'vehicle']:
                    detection['threat_level'] = 'low'
                elif class_name in ['knife', 'gun', 'weapon']:
                    detection['threat_level'] = 'critical'
                else:
                    detection['threat_level'] = 'low'
                filtered.append(detection)
        
        return filtered
    
    def _draw_bird_detections(self, frame: np.ndarray, detections: List[Dict]) -> int:
        """Draw bird detections with blue bounding boxes"""
        count = 0
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0.0)
            class_name = detection.get('class', 'bird')
            
            if len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
                x, y, w, h = map(int, bbox)
                color = (255, 0, 0)  # Blue in BGR
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw label
                label = f"🐦 {class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - 25), (x + label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                count += 1
        
        return count
    
    def _draw_intruder_detections(self, frame: np.ndarray, detections: List[Dict]) -> int:
        """Draw intruder detections with red bounding boxes"""
        count = 0
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0.0)
            class_name = detection.get('class', 'object')
            threat_level = detection.get('threat_level', 'low')
            
            if len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
                x, y, w, h = map(int, bbox)
                color = (0, 0, 255)  # Red in BGR
                thickness = 4 if threat_level in ['high', 'critical'] else 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                
                # Draw label with threat indicator
                threat_emoji = "🚨" if threat_level in ['high', 'critical'] else "⚠️"
                label = f"{threat_emoji} {class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - 25), (x + label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                count += 1
        
        return count
    
    def _add_system_overlay(self, frame: np.ndarray, frame_count: int, bird_count: int, intruder_count: int):
        """Add system information overlay to frame"""
        import datetime as dt
        
        timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # System status
        cv2.putText(frame, f"🤖 UNIFIED AI: Active | Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"🐦 Birds: {bird_count} | 🚨 Objects: {intruder_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Color legend
        cv2.putText(frame, "🐦 BLUE = Birds | 🚨 RED = Intruders/Objects", 
                   (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # AI status indicator
        ai_status = "🟢 UNIFIED AI" if bird_count > 0 or intruder_count > 0 else "🟡 MONITORING"
        cv2.putText(frame, ai_status, (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if "UNIFIED" in ai_status else (0, 255, 255), 2)
    
    def _update_statistics(self, bird_count: int, intruder_count: int):
        """Update system statistics"""
        try:
            if bird_count > 0:
                self.bird_counter.update_from_detection({
                    'birds_in': self.bird_counter.birds_in + bird_count,
                    'total_detections': bird_count
                })
            
            if intruder_count > 0:
                self.intruder_stats.add_detection(intruder_count)
        except Exception as e:
            logger.error(f"Statistics update error: {e}")
    
    def start_unified_processing(self):
        """Start unified video processing thread"""
        global current_frame, video_feed_active
        
        logger.info("🎬 UNIFIED Video Processing Started")
        logger.info("⚡ Optimized: Single stream, dual AI, no stuttering")
        logger.info("🔵 Bird AI + 🔴 Intruder AI working together")
        
        frame_count = 0
        last_fps_update = time.time()
        fps_counter = 0
        
        while video_feed_active:
            try:
                frame_start_time = time.time()
                
                # Get frame from camera
                frame = self.camera_manager.read_frame()
                if frame is None:
                    logger.warning("⚠️ No frame - retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                fps_counter += 1
                
                # FPS calculation
                current_time = time.time()
                if current_time - last_fps_update >= 5.0:
                    fps = fps_counter / (current_time - last_fps_update)
                    logger.info(f"📊 UNIFIED FPS: {fps:.2f}")
                    self.performance_monitor.record_frame_time(1.0 / fps if fps > 0 else 0)
                    last_fps_update = current_time
                    fps_counter = 0
                
                # Process frame with unified AI
                processed_frame = self.process_single_frame(frame, frame_count)
                
                # Update global current frame
                with frame_lock:
                    current_frame = processed_frame.copy()
                
                # Frame rate limiting
                time.sleep(1 / self.fps_limit)
                
            except Exception as e:
                logger.error(f"Unified processing error: {e}")
                time.sleep(1)

def create_unified_processor(camera_manager, ai_detector, bird_counter, intruder_stats, performance_monitor):
    """Factory function to create unified video processor"""
    return UnifiedVideoProcessor(
        camera_manager=camera_manager,
        ai_detector=ai_detector,
        bird_counter=bird_counter,
        intruder_stats=intruder_stats,
        performance_monitor=performance_monitor
    )
