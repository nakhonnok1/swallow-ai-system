# -*- coding: utf-8 -*-
"""
AI Performance Dashboard
แดชบอร์ดสำหรับตรวจสอบประสิทธิภาพ AI แบบ real-time
"""

import json
import time
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict
import threading

@dataclass
class PerformanceMetrics:
    """คลาสสำหรับเก็บข้อมูลประสิทธิภาพ"""
    timestamp: str
    fps: float
    detection_count: int
    processing_time: float
    memory_usage: float
    cpu_usage: float
    model_confidence: float
    anomaly_count: int
    
class AIPerformanceDashboard:
    """แดชบอร์ดสำหรับตรวจสอบประสิทธิภาพ AI"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_metrics = {
            'fps': 0.0,
            'total_detections': 0,
            'session_detections': 0,
            'avg_processing_time': 0.0,
            'uptime': 0.0,
            'performance_grade': 'Unknown'
        }
        
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def add_metrics(self, 
                   fps: float,
                   detection_count: int,
                   processing_time: float,
                   memory_usage: float = 0.0,
                   cpu_usage: float = 0.0,
                   model_confidence: float = 0.0,
                   anomaly_count: int = 0):
        """เพิ่มข้อมูลประสิทธิภาพใหม่"""
        
        with self.lock:
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                fps=fps,
                detection_count=detection_count,
                processing_time=processing_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                model_confidence=model_confidence,
                anomaly_count=anomaly_count
            )
            
            self.metrics_history.append(metrics)
            
            # จำกัดขนาดประวัติ
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            
            # อัพเดตสถิติปัจจุบัน
            self._update_current_metrics()
    
    def _update_current_metrics(self):
        """อัพเดตสถิติปัจจุบัน"""
        if not self.metrics_history:
            return
            
        recent_metrics = self.metrics_history[-10:]  # 10 ค่าล่าสุด
        
        # คำนวณค่าเฉลี่ย
        self.current_metrics['fps'] = sum(m.fps for m in recent_metrics) / len(recent_metrics)
        self.current_metrics['avg_processing_time'] = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
        self.current_metrics['session_detections'] = sum(m.detection_count for m in self.metrics_history)
        self.current_metrics['uptime'] = time.time() - self.start_time
        
        # คำนวณเกรดประสิทธิภาพ
        self.current_metrics['performance_grade'] = self._calculate_performance_grade()
    
    def _calculate_performance_grade(self) -> str:
        """คำนวณเกรดประสิทธิภาพ"""
        fps = self.current_metrics['fps']
        
        if fps >= 25:
            return 'A+ (Excellent)'
        elif fps >= 20:
            return 'A (Very Good)'
        elif fps >= 15:
            return 'B+ (Good)'
        elif fps >= 10:
            return 'B (Fair)'
        elif fps >= 5:
            return 'C (Poor)'
        else:
            return 'D (Very Poor)'
    
    def get_dashboard_data(self) -> Dict:
        """รับข้อมูลแดชบอร์ดทั้งหมด"""
        with self.lock:
            dashboard_data = {
                'current_metrics': self.current_metrics.copy(),
                'recent_history': [asdict(m) for m in self.metrics_history[-20:]],  # 20 ค่าล่าสุด
                'performance_summary': self._get_performance_summary(),
                'recommendations': self._get_recommendations()
            }
            
            return dashboard_data
    
    def _get_performance_summary(self) -> Dict:
        """สรุปประสิทธิภาพ"""
        if not self.metrics_history:
            return {}
            
        all_fps = [m.fps for m in self.metrics_history]
        all_processing_times = [m.processing_time for m in self.metrics_history]
        
        return {
            'max_fps': max(all_fps),
            'min_fps': min(all_fps),
            'avg_fps': sum(all_fps) / len(all_fps),
            'max_processing_time': max(all_processing_times),
            'min_processing_time': min(all_processing_times),
            'total_samples': len(self.metrics_history)
        }
    
    def _get_recommendations(self) -> List[str]:
        """ให้คำแนะนำสำหรับการปรับปรุงประสิทธิภาพ"""
        recommendations = []
        current_fps = self.current_metrics['fps']
        
        if current_fps < 10:
            recommendations.append("🔴 Performance is very low. Consider reducing image resolution or using GPU acceleration.")
            recommendations.append("💡 Try switching to FAST performance mode.")
            
        elif current_fps < 15:
            recommendations.append("🟡 Performance could be improved. Consider optimizing detection settings.")
            recommendations.append("💡 Check CPU usage and close unnecessary applications.")
            
        elif current_fps >= 25:
            recommendations.append("🟢 Excellent performance! You can try increasing accuracy settings.")
            recommendations.append("💡 Consider switching to ACCURATE mode for better detection quality.")
            
        # ตรวจสอบ processing time
        avg_time = self.current_metrics['avg_processing_time']
        if avg_time > 0.1:  # มากกว่า 100ms
            recommendations.append("⏱️ Processing time is high. Consider using ROI or reducing frame rate.")
            
        return recommendations
    
    def export_metrics(self, filename: str = None) -> str:
        """ส่งออกข้อมูลเป็น JSON"""
        if filename is None:
            filename = f"ai_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        data = self.get_dashboard_data()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return filename
    
    def print_live_dashboard(self):
        """แสดงแดชบอร์ดแบบ live"""
        import os
        
        # Clear screen (Windows)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("🤖 AI BIRD TRACKING - PERFORMANCE DASHBOARD")
        print("=" * 60)
        
        # Current metrics
        current = self.current_metrics
        print(f"📊 CURRENT PERFORMANCE")
        print(f"   FPS: {current['fps']:.1f}")
        print(f"   Grade: {current['performance_grade']}")
        print(f"   Processing Time: {current['avg_processing_time']:.3f}s")
        print(f"   Session Detections: {current['session_detections']}")
        print(f"   Uptime: {current['uptime']:.0f}s")
        
        print(f"\n📈 PERFORMANCE HISTORY (Last 10 readings)")
        if self.metrics_history:
            recent = self.metrics_history[-10:]
            for i, metrics in enumerate(recent, 1):
                time_str = metrics.timestamp.split('T')[1][:8]  # แสดงเฉพาะเวลา
                print(f"   {i:2d}. {time_str} | FPS: {metrics.fps:5.1f} | Detections: {metrics.detection_count:2d}")
        
        # Recommendations
        recommendations = self._get_recommendations()
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations[:3]:  # แสดง 3 ข้อแรก
                print(f"   {rec}")
        
        print("=" * 60)
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Global dashboard instance
dashboard = AIPerformanceDashboard()

def update_dashboard_metrics(**kwargs):
    """ฟังก์ชันสำหรับอัพเดตข้อมูลแดชบอร์ด"""
    dashboard.add_metrics(**kwargs)

def show_live_dashboard():
    """แสดงแดชบอร์ดแบบ live"""
    dashboard.print_live_dashboard()

def get_dashboard_json():
    """รับข้อมูลแดชบอร์ดเป็น JSON"""
    return dashboard.get_dashboard_data()
