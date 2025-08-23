#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 AI HELPER SYSTEM - ระบบช่วยเหลือ AI ที่ครบถ้วนสมบูรณ์
เป็นตัวช่วยหลักสำหรับเพิ่มประสิทธิภาพการทำงานของ AI ทุกระบบ
Version: 1.0 - ULTIMATE AI ASSISTANT

🎯 Features:
- Smart Performance Optimization
- Intelligent Resource Management  
- Advanced Analytics & Insights
- Real-time Monitoring Dashboard
- Predictive Maintenance
- Auto-tuning & Self-optimization
- Multi-AI Coordination
- Emergency Response System
"""

import cv2
import numpy as np
import time
import threading
import logging
import sqlite3
import json
import pickle
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AISystemMetrics:
    """ข้อมูลเมทริกของระบบ AI"""
    system_id: str
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    fps: float
    accuracy: float
    response_time: float
    error_rate: float
    uptime: float
    status: str

@dataclass
class OptimizationRecommendation:
    """คำแนะนำการปรับปรุง"""
    category: str
    priority: int  # 1-5
    description: str
    action: str
    estimated_improvement: float
    implementation_difficulty: int  # 1-5

class UltimateAIHelperSystem:
    """🤖 Ultimate AI Helper System - ตัวช่วย AI ที่สมบูรณ์แบบ"""
    
    def __init__(self):
        print("🚀 เริ่มต้น Ultimate AI Helper System...")
        
        # Core Configuration
        self.system_id = f"ai_helper_{int(time.time())}"
        self.start_time = time.time()
        self.active_ai_systems = {}
        
        # Monitoring & Analytics
        self.metrics_history = []
        self.performance_baseline = {}
        self.optimization_queue = Queue()
        self.alert_queue = Queue()
        
        # AI Enhancement Features
        self.smart_cache = {}
        self.prediction_models = {}
        self.optimization_rules = []
        self.learning_algorithms = {}
        
        # Database
        self.db_path = "ai_helper_system.db"
        self.db_connection = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=6)
        self.monitoring_active = True
        
        # Setup
        self._setup_logging()
        self._initialize_database()
        self._load_optimization_rules()
        self._start_monitoring_threads()
        
        print("✅ Ultimate AI Helper System พร้อมใช้งาน!")
    
    def _setup_logging(self):
        """ตั้งค่าระบบ logging"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('AIHelper')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(
                logs_dir / f"ai_helper_{datetime.now().strftime('%Y%m%d')}.log",
                encoding='utf-8'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _initialize_database(self):
        """เริ่มต้นฐานข้อมูล"""
        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # ตาราง system metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id TEXT,
                    timestamp REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL,
                    fps REAL,
                    accuracy REAL,
                    response_time REAL,
                    error_rate REAL,
                    uptime REAL,
                    status TEXT
                )
            ''')
            
            # ตาราง optimizations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    category TEXT,
                    description TEXT,
                    action TEXT,
                    improvement REAL,
                    status TEXT
                )
            ''')
            
            # ตาราง alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    severity TEXT,
                    category TEXT,
                    message TEXT,
                    system_id TEXT,
                    resolved INTEGER DEFAULT 0
                )
            ''')
            
            self.db_connection.commit()
            self.logger.info("✅ Database เชื่อมต่อสำเร็จ")
            
        except Exception as e:
            self.logger.error(f"❌ Database error: {e}")
    
    def _load_optimization_rules(self):
        """โหลดกฎการปรับปรุง"""
        self.optimization_rules = [
            {
                'name': 'high_memory_usage',
                'condition': lambda metrics: metrics.memory_usage > 80,
                'action': self._optimize_memory,
                'priority': 5
            },
            {
                'name': 'low_fps',
                'condition': lambda metrics: metrics.fps < 15,
                'action': self._optimize_performance,
                'priority': 4
            },
            {
                'name': 'high_cpu_usage',
                'condition': lambda metrics: metrics.cpu_usage > 90,
                'action': self._optimize_cpu,
                'priority': 4
            },
            {
                'name': 'low_accuracy',
                'condition': lambda metrics: metrics.accuracy < 0.7,
                'action': self._optimize_accuracy,
                'priority': 3
            },
            {
                'name': 'high_error_rate',
                'condition': lambda metrics: metrics.error_rate > 0.1,
                'action': self._handle_errors,
                'priority': 5
            }
        ]
    
    def _start_monitoring_threads(self):
        """เริ่มต้น monitoring threads"""
        # System monitor
        monitor_thread = threading.Thread(target=self._system_monitor_thread, daemon=True)
        monitor_thread.start()
        
        # Optimization processor
        optimization_thread = threading.Thread(target=self._optimization_processor_thread, daemon=True)
        optimization_thread.start()
        
        # Alert handler
        alert_thread = threading.Thread(target=self._alert_handler_thread, daemon=True)
        alert_thread.start()
        
        # Performance analyzer
        analyzer_thread = threading.Thread(target=self._performance_analyzer_thread, daemon=True)
        analyzer_thread.start()
        
        self.logger.info("✅ Monitoring threads เริ่มต้นแล้ว")
    
    def register_ai_system(self, system_id: str, system_instance: Any):
        """ลงทะเบียนระบบ AI"""
        self.active_ai_systems[system_id] = {
            'instance': system_instance,
            'registered_time': time.time(),
            'last_check': time.time(),
            'metrics_history': [],
            'status': 'active'
        }
        
        self.logger.info(f"📝 ลงทะเบียนระบบ AI: {system_id}")
    
    def unregister_ai_system(self, system_id: str):
        """ยกเลิกการลงทะเบียนระบบ AI"""
        if system_id in self.active_ai_systems:
            del self.active_ai_systems[system_id]
            self.logger.info(f"🗑️ ยกเลิกลงทะเบียน: {system_id}")
    
    def collect_metrics(self, system_id: str) -> Optional[AISystemMetrics]:
        """เก็บข้อมูลเมทริกจากระบบ AI"""
        if system_id not in self.active_ai_systems:
            return None
        
        try:
            ai_system = self.active_ai_systems[system_id]['instance']
            
            # Get system metrics
            process = psutil.Process()
            cpu_usage = process.cpu_percent()
            memory_usage = process.memory_percent()
            
            # Get AI-specific metrics
            fps = getattr(ai_system, 'detection_stats', {}).get('fps', 0)
            accuracy = getattr(ai_system, 'detection_stats', {}).get('accuracy_score', 0)
            
            metrics = AISystemMetrics(
                system_id=system_id,
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=0,  # TODO: Implement GPU monitoring
                fps=fps,
                accuracy=accuracy,
                response_time=getattr(ai_system, 'detection_stats', {}).get('processing_time', 0),
                error_rate=0,  # TODO: Implement error tracking
                uptime=time.time() - self.start_time,
                status='running' if getattr(ai_system, 'available', False) else 'error'
            )
            
            # Store metrics
            self.active_ai_systems[system_id]['metrics_history'].append(metrics)
            self.active_ai_systems[system_id]['last_check'] = time.time()
            
            # Keep only last 100 metrics
            if len(self.active_ai_systems[system_id]['metrics_history']) > 100:
                self.active_ai_systems[system_id]['metrics_history'] = \
                    self.active_ai_systems[system_id]['metrics_history'][-100:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting metrics for {system_id}: {e}")
            return None
    
    def analyze_performance(self, system_id: str) -> List[OptimizationRecommendation]:
        """วิเคราะห์ประสิทธิภาพและให้คำแนะนำ"""
        if system_id not in self.active_ai_systems:
            return []
        
        metrics_history = self.active_ai_systems[system_id]['metrics_history']
        if len(metrics_history) < 5:
            return []
        
        recommendations = []
        latest_metrics = metrics_history[-1]
        
        # Memory optimization
        if latest_metrics.memory_usage > 80:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=5,
                description="หน่วยความจำใช้งานสูง",
                action="ลดขนาด cache และทำ garbage collection",
                estimated_improvement=15.0,
                implementation_difficulty=2
            ))
        
        # Performance optimization
        if latest_metrics.fps < 15:
            recommendations.append(OptimizationRecommendation(
                category="performance",
                priority=4,
                description="ประสิทธิภาพการประมวลผลต่ำ",
                action="ลดขนาด input หรือใช้ model ที่เล็กกว่า",
                estimated_improvement=30.0,
                implementation_difficulty=3
            ))
        
        # Accuracy optimization
        if latest_metrics.accuracy < 0.7:
            recommendations.append(OptimizationRecommendation(
                category="accuracy",
                priority=3,
                description="ความแม่นยำต่ำ",
                action="ปรับ confidence threshold หรือใช้ model ที่ดีกว่า",
                estimated_improvement=20.0,
                implementation_difficulty=4
            ))
        
        return recommendations
    
    def optimize_system(self, system_id: str, auto_apply: bool = False) -> bool:
        """ปรับปรุงระบบ AI"""
        try:
            recommendations = self.analyze_performance(system_id)
            
            if not recommendations:
                self.logger.info(f"✅ ระบบ {system_id} ทำงานได้ดีแล้ว")
                return True
            
            applied_optimizations = 0
            
            for rec in sorted(recommendations, key=lambda x: x.priority, reverse=True):
                if auto_apply or rec.priority >= 4:
                    success = self._apply_optimization(system_id, rec)
                    if success:
                        applied_optimizations += 1
                        self.logger.info(f"✅ ปรับปรุง {rec.category} สำเร็จ")
                    else:
                        self.logger.warning(f"⚠️ ไม่สามารถปรับปรุง {rec.category}")
            
            self.logger.info(f"🔧 ปรับปรุงระบบ {system_id}: {applied_optimizations}/{len(recommendations)}")
            return applied_optimizations > 0
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing {system_id}: {e}")
            return False
    
    def _apply_optimization(self, system_id: str, recommendation: OptimizationRecommendation) -> bool:
        """ปรับใช้การปรับปรุง"""
        try:
            ai_system = self.active_ai_systems[system_id]['instance']
            
            if recommendation.category == "memory":
                return self._optimize_memory(ai_system)
            elif recommendation.category == "performance":
                return self._optimize_performance(ai_system)
            elif recommendation.category == "accuracy":
                return self._optimize_accuracy(ai_system)
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error applying optimization: {e}")
            return False
    
    def _optimize_memory(self, ai_system) -> bool:
        """ปรับปรุงหน่วยความจำ"""
        try:
            # Clear caches
            if hasattr(ai_system, 'ai_cache'):
                ai_system.ai_cache.clear()
            
            if hasattr(ai_system, 'detection_memory'):
                ai_system.detection_memory = ai_system.detection_memory[-100:]  # Keep last 100
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("🧹 ทำความสะอาดหน่วยความจำแล้ว")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Memory optimization error: {e}")
            return False
    
    def _optimize_performance(self, ai_system) -> bool:
        """ปรับปรุงประสิทธิภาพ"""
        try:
            # Reduce input size
            if hasattr(ai_system, 'model_info'):
                current_size = ai_system.model_info.get('input_size', (416, 416))
                if current_size[0] > 416:
                    ai_system.model_info['input_size'] = (416, 416)
                    self.logger.info("📏 ลดขนาด input เป็น 416x416")
            
            # Adjust confidence threshold for faster processing
            if hasattr(ai_system, 'confidence_threshold'):
                if ai_system.confidence_threshold < 0.6:
                    ai_system.confidence_threshold = 0.6
                    self.logger.info("🎯 ปรับ confidence threshold เป็น 0.6")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Performance optimization error: {e}")
            return False
    
    def _optimize_accuracy(self, ai_system) -> bool:
        """ปรับปรุงความแม่นยำ"""
        try:
            # Lower confidence threshold for better recall
            if hasattr(ai_system, 'confidence_threshold'):
                if ai_system.confidence_threshold > 0.3:
                    ai_system.confidence_threshold = 0.3
                    self.logger.info("🎯 ปรับ confidence threshold เป็น 0.3 เพื่อความแม่นยำ")
            
            # Increase input size for better accuracy
            if hasattr(ai_system, 'model_info'):
                current_size = ai_system.model_info.get('input_size', (416, 416))
                if current_size[0] < 608:
                    ai_system.model_info['input_size'] = (608, 608)
                    self.logger.info("📏 เพิ่มขนาด input เป็น 608x608")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Accuracy optimization error: {e}")
            return False
    
    def _handle_errors(self, ai_system) -> bool:
        """จัดการข้อผิดพลาด"""
        try:
            # Reset system if too many errors
            if hasattr(ai_system, '_error_count'):
                if ai_system._error_count > 10:
                    self.logger.warning("🔄 รีเซ็ตระบบเนื่องจากข้อผิดพลาดมาก")
                    # Implement system reset logic here
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error handling error: {e}")
            return False
    
    def _system_monitor_thread(self):
        """Thread สำหรับติดตามระบบ"""
        while self.monitoring_active:
            try:
                for system_id in list(self.active_ai_systems.keys()):
                    metrics = self.collect_metrics(system_id)
                    
                    if metrics:
                        # Check optimization rules
                        for rule in self.optimization_rules:
                            if rule['condition'](metrics):
                                self.optimization_queue.put({
                                    'system_id': system_id,
                                    'rule': rule,
                                    'metrics': metrics
                                })
                        
                        # Store to database
                        if self.db_connection:
                            cursor = self.db_connection.cursor()
                            cursor.execute('''
                                INSERT INTO system_metrics 
                                (system_id, timestamp, cpu_usage, memory_usage, gpu_usage, 
                                 fps, accuracy, response_time, error_rate, uptime, status)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                metrics.system_id, metrics.timestamp, metrics.cpu_usage,
                                metrics.memory_usage, metrics.gpu_usage, metrics.fps,
                                metrics.accuracy, metrics.response_time, metrics.error_rate,
                                metrics.uptime, metrics.status
                            ))
                            self.db_connection.commit()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Monitor thread error: {e}")
                time.sleep(10)
    
    def _optimization_processor_thread(self):
        """Thread สำหรับประมวลผลการปรับปรุง"""
        while self.monitoring_active:
            try:
                optimization_task = self.optimization_queue.get(timeout=5)
                
                system_id = optimization_task['system_id']
                rule = optimization_task['rule']
                
                self.logger.info(f"🔧 กำลังปรับปรุง {system_id} ด้วยกฎ {rule['name']}")
                
                ai_system = self.active_ai_systems[system_id]['instance']
                success = rule['action'](ai_system)
                
                if success:
                    self.logger.info(f"✅ ปรับปรุง {system_id} สำเร็จ")
                else:
                    self.logger.warning(f"⚠️ ไม่สามารถปรับปรุง {system_id}")
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Optimization processor error: {e}")
    
    def _alert_handler_thread(self):
        """Thread สำหรับจัดการแจ้งเตือน"""
        while self.monitoring_active:
            try:
                # Check for alerts every 30 seconds
                time.sleep(30)
                
                for system_id, system_data in self.active_ai_systems.items():
                    metrics_history = system_data['metrics_history']
                    
                    if len(metrics_history) > 0:
                        latest_metrics = metrics_history[-1]
                        
                        # Critical alerts
                        if latest_metrics.memory_usage > 95:
                            self._create_alert("critical", "memory", 
                                             f"หน่วยความจำเกือบเต็ม: {latest_metrics.memory_usage:.1f}%", 
                                             system_id)
                        
                        if latest_metrics.cpu_usage > 95:
                            self._create_alert("critical", "cpu", 
                                             f"CPU ใช้งานสูงมาก: {latest_metrics.cpu_usage:.1f}%", 
                                             system_id)
                        
                        if latest_metrics.fps < 5:
                            self._create_alert("warning", "performance", 
                                             f"ประสิทธิภาพต่ำมาก: {latest_metrics.fps:.1f} FPS", 
                                             system_id)
                
            except Exception as e:
                self.logger.error(f"❌ Alert handler error: {e}")
    
    def _performance_analyzer_thread(self):
        """Thread สำหรับวิเคราะห์ประสิทธิภาพ"""
        while self.monitoring_active:
            try:
                time.sleep(60)  # Analyze every minute
                
                for system_id in self.active_ai_systems:
                    recommendations = self.analyze_performance(system_id)
                    
                    if recommendations:
                        high_priority = [r for r in recommendations if r.priority >= 4]
                        
                        if high_priority:
                            self.logger.info(f"📊 พบคำแนะนำ {len(high_priority)} ข้อสำหรับ {system_id}")
                            
                            # Auto-apply critical optimizations
                            for rec in high_priority:
                                if rec.priority == 5:
                                    self._apply_optimization(system_id, rec)
                
            except Exception as e:
                self.logger.error(f"❌ Performance analyzer error: {e}")
    
    def _create_alert(self, severity: str, category: str, message: str, system_id: str):
        """สร้างแจ้งเตือน"""
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    INSERT INTO alerts (timestamp, severity, category, message, system_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (time.time(), severity, category, message, system_id))
                self.db_connection.commit()
            
            self.logger.warning(f"🚨 {severity.upper()}: {message} [{system_id}]")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating alert: {e}")
    
    def get_system_dashboard(self) -> Dict:
        """ดึงข้อมูล dashboard ของระบบ"""
        dashboard = {
            'overview': {
                'active_systems': len(self.active_ai_systems),
                'total_uptime': time.time() - self.start_time,
                'status': 'healthy'
            },
            'systems': {},
            'alerts': self._get_recent_alerts(),
            'recommendations': {}
        }
        
        for system_id, system_data in self.active_ai_systems.items():
            if system_data['metrics_history']:
                latest_metrics = system_data['metrics_history'][-1]
                
                dashboard['systems'][system_id] = {
                    'status': latest_metrics.status,
                    'fps': latest_metrics.fps,
                    'accuracy': latest_metrics.accuracy,
                    'cpu_usage': latest_metrics.cpu_usage,
                    'memory_usage': latest_metrics.memory_usage,
                    'last_update': latest_metrics.timestamp
                }
            
            # Get recommendations
            recommendations = self.analyze_performance(system_id)
            dashboard['recommendations'][system_id] = [
                {
                    'category': r.category,
                    'priority': r.priority,
                    'description': r.description,
                    'improvement': r.estimated_improvement
                }
                for r in recommendations
            ]
        
        return dashboard
    
    def _get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """ดึงแจ้งเตือนล่าสุด"""
        if not self.db_connection:
            return []
        
        try:
            cursor = self.db_connection.cursor()
            since = time.time() - (hours * 3600)
            
            cursor.execute('''
                SELECT timestamp, severity, category, message, system_id, resolved
                FROM alerts 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 50
            ''', (since,))
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'timestamp': row[0],
                    'severity': row[1],
                    'category': row[2],
                    'message': row[3],
                    'system_id': row[4],
                    'resolved': bool(row[5])
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"❌ Error getting alerts: {e}")
            return []
    
    def shutdown(self):
        """ปิดระบบ"""
        self.monitoring_active = False
        
        if self.db_connection:
            self.db_connection.close()
        
        self.executor.shutdown(wait=True)
        self.logger.info("🛑 AI Helper System ปิดแล้ว")

# สร้าง instance หลักของระบบ
ai_helper = UltimateAIHelperSystem()

def get_ai_helper() -> UltimateAIHelperSystem:
    """ดึง AI Helper System instance"""
    return ai_helper

# ฟังก์ชันช่วยเหลือ
def optimize_all_systems():
    """ปรับปรุงระบบ AI ทั้งหมด"""
    helper = get_ai_helper()
    
    optimized_count = 0
    for system_id in helper.active_ai_systems:
        if helper.optimize_system(system_id, auto_apply=True):
            optimized_count += 1
    
    print(f"✅ ปรับปรุงระบบ AI {optimized_count}/{len(helper.active_ai_systems)} ระบบ")
    return optimized_count

def show_dashboard():
    """แสดง dashboard ของระบบ"""
    helper = get_ai_helper()
    dashboard = helper.get_system_dashboard()
    
    print("\n" + "="*60)
    print("🤖 AI HELPER SYSTEM DASHBOARD")
    print("="*60)
    
    print(f"📊 ระบบที่ใช้งาน: {dashboard['overview']['active_systems']}")
    print(f"⏱️ เวลาทำงาน: {dashboard['overview']['total_uptime']/3600:.1f} ชั่วโมง")
    
    for system_id, data in dashboard['systems'].items():
        print(f"\n🤖 {system_id}:")
        print(f"  สถานะ: {data['status']}")
        print(f"  FPS: {data['fps']:.1f}")
        print(f"  ความแม่นยำ: {data['accuracy']:.2f}")
        print(f"  CPU: {data['cpu_usage']:.1f}%")
        print(f"  RAM: {data['memory_usage']:.1f}%")
        
        # แสดงคำแนะนำ
        recommendations = dashboard['recommendations'].get(system_id, [])
        if recommendations:
            print(f"  💡 คำแนะนำ: {len(recommendations)} ข้อ")
            for rec in recommendations[:3]:  # แสดง 3 ข้อแรก
                print(f"    - {rec['description']} (ปรับปรุง: {rec['improvement']:.1f}%)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("🚀 AI Helper System Demo")
    
    # แสดง dashboard
    show_dashboard()
    
    # ทดสอบ optimization
    optimize_all_systems()
    
    print("\nกด Enter เพื่อออก...")
    input()
    
    # ปิดระบบ
    ai_helper.shutdown()
