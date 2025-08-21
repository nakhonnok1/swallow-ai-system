#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swallow AI System Status Report
"""
import sys
sys.path.append('C:/Nakhonnok')

def check_system_status():
    print("🔍 Swallow AI System Status Check")
    print("=" * 50)
    
    try:
        from swallow_ai.app_working import (
            app, bird_counter, real_camera, ai_detector, 
            ultra_safe, ADVANCED_DETECTOR_AVAILABLE, 
            SIMPLE_YOLO_AVAILABLE, ULTRA_SAFE_AVAILABLE,
            OBJECT_DETECTION_AVAILABLE
        )
        print("✅ Core modules imported successfully")
        
        print(f"📷 Camera connection: {'🟢 Ready' if real_camera.is_connected else '🔴 Waiting for source'}")
        print(f"🤖 Advanced Object Detection: {'🟢 Active' if ADVANCED_DETECTOR_AVAILABLE else '🔴 Fallback'}")
        print(f"🎯 Simple YOLO Detection: {'🟢 Available' if SIMPLE_YOLO_AVAILABLE else '🔴 Fallback'}")
        print(f"🛡️ Ultra Safe Detection: {'🟢 Active' if ULTRA_SAFE_AVAILABLE else '🔴 Fallback'}")
        print(f"🔍 Object Detection Status: {'🟢 Model Loaded' if OBJECT_DETECTION_AVAILABLE else '🔴 No Model'}")
        
        print(f"🐦 Bird Counter: In={bird_counter.birds_in}, Out={bird_counter.birds_out}, Current={bird_counter.current_count}")
        
        # Test Flask app
        with app.test_client() as client:
            health = client.get('/api/system-health')
            if health.status_code == 200:
                data = health.json
                print(f"💾 System Health: CPU={data['system']['cpu_usage']}%, Memory={data['system']['memory_usage']}%")
                print(f"⚡ AI Performance: {data['ai_performance']['performance_score'].upper()}")
                print(f"🌐 API Status: {data['connectivity']['api_status'].upper()}")
            
            detection_status = client.get('/api/object-detection/status')
            if detection_status.status_code == 200:
                det_data = detection_status.json
                print(f"🔎 Detection Model: {'🟢 Loaded' if det_data['model_loaded'] else '🔴 Not Loaded'}")
                print(f"🔄 Detection Enabled: {'🟢 Yes' if det_data['enabled'] else '🔴 No'}")
        
        print("\n🚀 System Ready for:")
        print("   • Video streaming at /video_feed")
        print("   • Dashboard at /dashboard")
        print("   • API endpoints for statistics and insights")
        print("   • Real-time bird counting and anomaly detection")
        
        return True
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False

if __name__ == '__main__':
    success = check_system_status()
    exit(0 if success else 1)
