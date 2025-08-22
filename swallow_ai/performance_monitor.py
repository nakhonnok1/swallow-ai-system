# performance_monitor.py - ระบบตรวจสอบประสิทธิภาพ
import time
import psutil
import threading
from datetime import datetime
from collections import deque

class PerformanceMonitor:
    def __init__(self, max_records=1000):
        self.max_records = max_records
        self.cpu_usage = deque(maxlen=max_records)
        self.memory_usage = deque(maxlen=max_records)
        self.processing_times = deque(maxlen=max_records)
        self.frame_rates = deque(maxlen=max_records)
        self.last_frame_time = time.time()
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """เริ่มการตรวจสอบประสิทธิภาพ"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("📊 Performance monitoring started")

    def stop_monitoring(self):
        """หยุดการตรวจสอบประสิทธิภาพ"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("⏹️ Performance monitoring stopped")

    def _monitor_loop(self):
        """Loop หลักสำหรับตรวจสอบประสิทธิภาพ"""
        while self.monitoring:
            try:
                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append({
                    'timestamp': datetime.now(),
                    'value': cpu_percent
                })

                # Memory Usage
                memory = psutil.virtual_memory()
                self.memory_usage.append({
                    'timestamp': datetime.now(),
                    'value': memory.percent,
                    'available_mb': memory.available / 1024 / 1024
                })

            except Exception as e:
                print(f"❌ Monitoring error: {e}")
            
            time.sleep(5)  # ตรวจสอบทุก 5 วินาที

    def log_processing_time(self, process_name, start_time):
        """บันทึกเวลาการประมวลผล"""
        processing_time = time.time() - start_time
        self.processing_times.append({
            'timestamp': datetime.now(),
            'process': process_name,
            'duration': processing_time
        })

    def log_frame_rate(self):
        """บันทึกอัตราเฟรม"""
        current_time = time.time()
        frame_interval = current_time - self.last_frame_time
        fps = 1.0 / frame_interval if frame_interval > 0 else 0
        
        self.frame_rates.append({
            'timestamp': datetime.now(),
            'fps': fps,
            'interval': frame_interval
        })
        
        self.last_frame_time = current_time

    def get_stats(self):
        """ดึงสถิติประสิทธิภาพ"""
        try:
            stats = {
                'current_time': datetime.now().isoformat(),
                'cpu': {
                    'current': self.cpu_usage[-1]['value'] if self.cpu_usage else 0,
                    'average': sum(item['value'] for item in self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
                },
                'memory': {
                    'current': self.memory_usage[-1]['value'] if self.memory_usage else 0,
                    'available_mb': self.memory_usage[-1]['available_mb'] if self.memory_usage else 0
                },
                'processing': {
                    'average_time': sum(item['duration'] for item in self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                    'total_processes': len(self.processing_times)
                },
                'frame_rate': {
                    'current_fps': self.frame_rates[-1]['fps'] if self.frame_rates else 0,
                    'average_fps': sum(item['fps'] for item in self.frame_rates) / len(self.frame_rates) if self.frame_rates else 0
                }
            }
            return stats
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {}

    def get_health_status(self):
        """ตรวจสอบสถานะความแข็งแรงของระบบ"""
        stats = self.get_stats()
        
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        # ตรวจสอบ CPU
        if stats.get('cpu', {}).get('current', 0) > 80:
            health['issues'].append('High CPU usage')
            health['status'] = 'warning'
        
        # ตรวจสอบ Memory
        if stats.get('memory', {}).get('current', 0) > 85:
            health['issues'].append('High memory usage')
            health['status'] = 'warning'
        
        # ตรวจสอบ Frame Rate
        current_fps = stats.get('frame_rate', {}).get('current_fps', 0)
        if current_fps < 10:
            health['issues'].append('Low frame rate')
            health['status'] = 'warning'
        
        if len(health['issues']) > 2:
            health['status'] = 'critical'
            
        return health

# สร้าง instance สำหรับใช้งาน
performance_monitor = PerformanceMonitor()
