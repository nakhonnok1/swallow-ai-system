#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ULTIMATE AI CONFIGURATION CENTER
ศูนย์รวมการตั้งค่า AI System ทั้งหมด
Version: 1.0 - COMPREHENSIVE CONFIGURATION

🚀 รวมการตั้งค่า:
- AI Vision Detection Settings
- Performance Optimization Settings  
- Helper System Settings
- Database Settings
- Monitoring & Analytics Settings
"""

import os
from pathlib import Path

class UltimateAIConfig:
    """การตั้งค่า AI System แบบครบถ้วน"""
    
    def __init__(self):
        # 🏠 Base Directory
        self.BASE_DIR = Path(__file__).parent
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.CACHE_DIR = self.BASE_DIR / "cache"
        
        # สร้างโฟลเดอร์ที่จำเป็น
        for directory in [self.MODELS_DIR, self.LOGS_DIR, self.DATA_DIR, self.CACHE_DIR]:
            directory.mkdir(exist_ok=True)
    
    # ⚙️ AI VISION DETECTION SETTINGS
    @property
    def VISION_CONFIG(self):
        return {
            # Model Configuration
            'yolo_config': str(self.BASE_DIR / 'yolov4.cfg'),
            'yolo_weights': str(self.BASE_DIR / 'yolov4.weights'),
            'yolo_names': str(self.BASE_DIR / 'coco.names'),
            'yolov8_model': str(self.BASE_DIR / 'yolov8n.pt'),
            
            # Detection Parameters
            'confidence_threshold': 0.4,
            'nms_threshold': 0.45,
            'input_width': 608,
            'input_height': 608,
            'scale_factor': 1/255.0,
            'mean_subtraction': (0, 0, 0),
            'swap_rb': True,
            'crop': False,
            
            # Processing Settings
            'enable_gpu': True,
            'gpu_memory_fraction': 0.8,
            'enable_multithreading': True,
            'max_threads': 4,
            'queue_size': 10,
            'enable_smart_tracking': True,
            'tracking_max_age': 30,
            'tracking_min_hits': 3,
            
            # Target Classes (COCO dataset)
            'target_classes': [
                'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'car', 'motorbike',
                'bus', 'truck', 'boat', 'bicycle'
            ],
            
            # Alert Classes
            'intruder_classes': ['person'],
            'animal_classes': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
            'vehicle_classes': ['car', 'motorbike', 'bus', 'truck', 'boat', 'bicycle'],
            
            # Color Scheme (BGR format)
            'colors': {
                'person': (0, 0, 255),      # Red
                'bird': (0, 255, 0),        # Green  
                'animal': (255, 0, 0),      # Blue
                'vehicle': (0, 255, 255),   # Yellow
                'default': (255, 255, 255)  # White
            }
        }
    
    # ⚡ PERFORMANCE OPTIMIZATION SETTINGS
    @property
    def PERFORMANCE_CONFIG(self):
        return {
            # GPU Acceleration
            'enable_gpu_acceleration': True,
            'gpu_device_id': 0,
            'gpu_memory_limit': 0.8,
            'enable_tensorrt': False,  # เปิดถ้ามี TensorRT
            'enable_opencl': True,
            
            # CPU Optimization
            'cpu_threads': os.cpu_count(),
            'enable_cpu_optimization': True,
            'cpu_affinity': None,  # None = auto
            'enable_hyperthreading': True,
            
            # Memory Management
            'enable_smart_caching': True,
            'cache_size_mb': 512,
            'cache_ttl_seconds': 300,
            'enable_memory_pool': True,
            'memory_pool_size_mb': 256,
            'enable_garbage_collection': True,
            'gc_threshold': 100,
            
            # Parallel Processing
            'enable_parallel_processing': True,
            'max_workers': 4,
            'batch_size': 4,
            'enable_async_processing': True,
            'async_queue_size': 20,
            
            # Model Optimization
            'enable_model_quantization': False,
            'quantization_type': 'int8',
            'enable_model_pruning': False,
            'pruning_ratio': 0.1,
            'enable_model_fusion': True,
            
            # Frame Processing
            'enable_frame_skipping': True,
            'frame_skip_ratio': 0.1,
            'enable_dynamic_resolution': True,
            'min_resolution': (320, 240),
            'max_resolution': (1920, 1080),
            'enable_smart_cropping': True,
            
            # Monitoring
            'enable_performance_monitoring': True,
            'monitoring_interval': 5.0,
            'enable_auto_optimization': True,
            'optimization_threshold': 0.8
        }
    
    # 🤖 AI HELPER SYSTEM SETTINGS
    @property
    def HELPER_CONFIG(self):
        return {
            # System Monitoring
            'enable_system_monitoring': True,
            'monitoring_interval': 10.0,
            'resource_check_interval': 5.0,
            'alert_threshold_cpu': 80.0,
            'alert_threshold_memory': 85.0,
            'alert_threshold_fps': 10.0,
            
            # Performance Analysis
            'enable_performance_analysis': True,
            'analysis_window_size': 100,
            'analysis_interval': 30.0,
            'enable_trend_analysis': True,
            'trend_window_hours': 24,
            
            # Auto Optimization
            'enable_auto_optimization': True,
            'optimization_interval': 300.0,  # 5 minutes
            'optimization_threshold': 0.75,
            'max_optimization_attempts': 3,
            'enable_learning_optimization': True,
            
            # Predictive Maintenance
            'enable_predictive_maintenance': True,
            'prediction_window_hours': 168,  # 1 week
            'maintenance_threshold': 0.7,
            'enable_anomaly_detection': True,
            'anomaly_sensitivity': 0.8,
            
            # Database Settings
            'database_file': str(self.DATA_DIR / 'ai_helper_system.db'),
            'backup_interval_hours': 24,
            'max_backup_files': 7,
            'enable_data_compression': True,
            
            # Notifications
            'enable_notifications': True,
            'notification_methods': ['log', 'console'],
            'enable_email_alerts': False,
            'email_settings': {
                'smtp_server': '',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'recipients': []
            }
        }
    
    # 🗄️ DATABASE SETTINGS
    @property
    def DATABASE_CONFIG(self):
        return {
            # Main Databases
            'ai_agent_memory': str(self.DATA_DIR / 'ai_agent_memory.db'),
            'enhanced_ai_system': str(self.DATA_DIR / 'enhanced_ai_system.db'),
            'intelligent_intruder_detections': str(self.DATA_DIR / 'intelligent_intruder_detections.db'),
            'object_detection_alerts': str(self.DATA_DIR / 'object_detection_alerts.db'),
            'swallow_smart_stats': str(self.DATA_DIR / 'swallow_smart_stats.db'),
            'ultimate_ai_agent': str(self.DATA_DIR / 'ultimate_ai_agent.db'),
            
            # Connection Settings
            'connection_timeout': 30,
            'enable_wal_mode': True,
            'enable_foreign_keys': True,
            'cache_size': 2000,
            'temp_store': 'memory',
            
            # Backup Settings
            'enable_auto_backup': True,
            'backup_interval_hours': 6,
            'max_backup_files': 10,
            'backup_directory': str(self.DATA_DIR / 'backups'),
            
            # Performance Settings
            'enable_connection_pooling': True,
            'max_connections': 10,
            'enable_query_optimization': True,
            'enable_indexing': True,
            
            # Data Retention
            'retention_days': 90,
            'enable_data_archiving': True,
            'archive_interval_days': 30,
            'enable_data_cleanup': True
        }
    
    # 📊 MONITORING & ANALYTICS SETTINGS
    @property
    def MONITORING_CONFIG(self):
        return {
            # Logging Configuration
            'log_level': 'INFO',
            'log_file': str(self.LOGS_DIR / 'ultimate_ai_system.log'),
            'max_log_size_mb': 100,
            'max_log_files': 5,
            'enable_console_logging': True,
            'enable_file_logging': True,
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            
            # Metrics Collection
            'enable_metrics_collection': True,
            'metrics_interval': 30.0,
            'enable_system_metrics': True,
            'enable_ai_metrics': True,
            'enable_performance_metrics': True,
            
            # Analytics
            'enable_analytics': True,
            'analytics_interval': 300.0,
            'enable_trend_analysis': True,
            'enable_pattern_recognition': True,
            'enable_anomaly_detection': True,
            
            # Dashboard
            'enable_dashboard': True,
            'dashboard_port': 8080,
            'dashboard_host': '0.0.0.0',
            'enable_real_time_updates': True,
            'update_interval': 5.0,
            
            # Alerts
            'enable_alerting': True,
            'alert_channels': ['log', 'console'],
            'alert_thresholds': {
                'cpu_usage': 85.0,
                'memory_usage': 90.0,
                'disk_usage': 95.0,
                'fps_drop': 50.0,
                'detection_accuracy': 0.7,
                'response_time': 1000.0  # ms
            }
        }
    
    # 🎥 CAMERA & INPUT SETTINGS
    @property
    def CAMERA_CONFIG(self):
        return {
            # Default Camera Sources
            'rtsp_sources': [
                {
                    'name': 'Main Camera',
                    'url': 'rtsp://ainok1:ainok123@192.168.1.100:554/stream1',
                    'enabled': True,
                    'resolution': (1920, 1080),
                    'fps': 30
                },
                {
                    'name': 'Secondary Camera',
                    'url': 'rtsp://ainok1:ainok123@192.168.1.101:554/stream1',
                    'enabled': False,
                    'resolution': (1280, 720),
                    'fps': 25
                }
            ],
            
            'usb_cameras': [
                {
                    'name': 'USB Camera 1',
                    'device_id': 0,
                    'enabled': True,
                    'resolution': (640, 480),
                    'fps': 30
                }
            ],
            
            # Connection Settings
            'connection_timeout': 10.0,
            'reconnect_attempts': 5,
            'reconnect_delay': 2.0,
            'enable_auto_reconnect': True,
            
            # Buffer Settings
            'enable_buffering': True,
            'buffer_size': 3,
            'enable_frame_dropping': True,
            'max_frame_age': 1.0,
            
            # Recording Settings
            'enable_recording': False,
            'recording_directory': str(self.DATA_DIR / 'recordings'),
            'recording_format': 'mp4',
            'recording_quality': 'high',
            'max_recording_duration': 3600,  # 1 hour
            'auto_delete_old_recordings': True,
            'max_recording_age_days': 7
        }
    
    # 🎯 AI CHATBOT SETTINGS
    @property
    def CHATBOT_CONFIG(self):
        return {
            # Model Settings
            'model_name': 'enhanced_ultra_smart_ai',
            'max_tokens': 2048,
            'temperature': 0.7,
            'enable_context_memory': True,
            'memory_window': 10,
            
            # Response Settings
            'max_response_time': 30.0,
            'enable_streaming': True,
            'enable_safety_filter': True,
            'enable_translation': True,
            'default_language': 'th',  # Thai
            
            # Learning Settings
            'enable_learning': True,
            'learning_rate': 0.001,
            'enable_feedback_learning': True,
            'enable_conversation_analysis': True,
            
            # Integration Settings
            'enable_vision_integration': True,
            'enable_system_commands': True,
            'enable_data_queries': True,
            'enable_real_time_updates': True
        }
    
    def get_all_configs(self):
        """รับการตั้งค่าทั้งหมด"""
        return {
            'vision': self.VISION_CONFIG,
            'performance': self.PERFORMANCE_CONFIG,
            'helper': self.HELPER_CONFIG,
            'database': self.DATABASE_CONFIG,
            'monitoring': self.MONITORING_CONFIG,
            'camera': self.CAMERA_CONFIG,
            'chatbot': self.CHATBOT_CONFIG
        }
    
    def save_config_to_file(self, filename='ultimate_ai_config.json'):
        """บันทึกการตั้งค่าเป็นไฟล์"""
        import json
        
        config_file = self.BASE_DIR / filename
        all_configs = self.get_all_configs()
        
        # แปลง Path objects เป็น string
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        configs_serializable = convert_paths(all_configs)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(configs_serializable, f, indent=2, ensure_ascii=False)
        
        return config_file
    
    def load_config_from_file(self, filename='ultimate_ai_config.json'):
        """โหลดการตั้งค่าจากไฟล์"""
        import json
        
        config_file = self.BASE_DIR / filename
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return None
    
    def validate_config(self):
        """ตรวจสอบความถูกต้องของการตั้งค่า"""
        issues = []
        
        # ตรวจสอบไฟล์ model
        vision_config = self.VISION_CONFIG
        model_files = [
            ('YOLO Config', vision_config['yolo_config']),
            ('YOLO Weights', vision_config['yolo_weights']),
            ('YOLO Names', vision_config['yolo_names']),
            ('YOLOv8 Model', vision_config['yolov8_model'])
        ]
        
        for name, filepath in model_files:
            if not Path(filepath).exists():
                issues.append(f"❌ {name} ไม่พบ: {filepath}")
        
        # ตรวจสอบการตั้งค่า GPU
        perf_config = self.PERFORMANCE_CONFIG
        if perf_config['enable_gpu_acceleration']:
            try:
                import cv2
                if not cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    issues.append("⚠️ GPU acceleration เปิดอยู่แต่ไม่พบ CUDA device")
            except:
                issues.append("⚠️ ไม่สามารถตรวจสอบ CUDA ได้")
        
        # ตรวจสอบโฟลเดอร์
        directories = [self.MODELS_DIR, self.LOGS_DIR, self.DATA_DIR, self.CACHE_DIR]
        for directory in directories:
            if not directory.exists():
                issues.append(f"❌ โฟลเดอร์ไม่พบ: {directory}")
        
        return issues

# สร้าง instance หลัก
config = UltimateAIConfig()

# ฟังก์ชันสำหรับการใช้งานง่าย
def get_vision_config():
    """รับการตั้งค่า AI Vision"""
    return config.VISION_CONFIG

def get_performance_config():
    """รับการตั้งค่า Performance"""
    return config.PERFORMANCE_CONFIG

def get_helper_config():
    """รับการตั้งค่า AI Helper"""
    return config.HELPER_CONFIG

def get_database_config():
    """รับการตั้งค่า Database"""
    return config.DATABASE_CONFIG

def get_monitoring_config():
    """รับการตั้งค่า Monitoring"""
    return config.MONITORING_CONFIG

def get_camera_config():
    """รับการตั้งค่า Camera"""
    return config.CAMERA_CONFIG

def get_chatbot_config():
    """รับการตั้งค่า Chatbot"""
    return config.CHATBOT_CONFIG

def validate_all_configs():
    """ตรวจสอบการตั้งค่าทั้งหมด"""
    return config.validate_config()

if __name__ == "__main__":
    # ทดสอบการทำงาน
    print("🎯 ULTIMATE AI CONFIGURATION CENTER")
    print("="*60)
    
    # ตรวจสอบการตั้งค่า
    print("🔍 ตรวจสอบการตั้งค่า...")
    issues = validate_all_configs()
    
    if not issues:
        print("✅ การตั้งค่าทั้งหมดถูกต้อง")
    else:
        print("⚠️ พบปัญหาในการตั้งค่า:")
        for issue in issues:
            print(f"   {issue}")
    
    # บันทึกการตั้งค่า
    print("\n💾 บันทึกการตั้งค่า...")
    config_file = config.save_config_to_file()
    print(f"✅ บันทึกเป็นไฟล์: {config_file}")
    
    # แสดงสรุปการตั้งค่า
    print("\n📋 สรุปการตั้งค่า:")
    all_configs = config.get_all_configs()
    for section, settings in all_configs.items():
        print(f"   📂 {section.upper()}: {len(settings)} รายการ")
    
    print("\n🚀 พร้อมใช้งาน Ultimate AI System!")
