#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔌 MAIN APP INTEGRATION - Enhanced Intruder Detection
อัพเดต app_working.py ให้รองรับระบบตรวจจับสิ่งแปลกปลอมใหม่

🎯 Features:
✅ Integration with Ultra Intelligent Intruder Detector
✅ Real-time Alert System
✅ Camera Stream Integration 
✅ API Endpoints for Intruder Detection
✅ Performance Monitoring
✅ Database Integration
"""

import sys
import os
import traceback
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import json
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our intelligent intruder detection system
try:
    from intelligent_intruder_integration import (
        IntelligentIntruderIntegration, 
        get_integration_instance, 
        setup_intruder_integration
    )
    INTRUDER_DETECTION_AVAILABLE = True
    print("✅ Ultra Intelligent Intruder Detection System imported successfully")
except ImportError as e:
    print(f"⚠️ Intruder Detection System not available: {e}")
    INTRUDER_DETECTION_AVAILABLE = False

# Import existing components with fallbacks
try:
    from enhanced_ultra_smart_ai_agent import EnhancedUltraSmartAIAgent
    AI_AGENT_AVAILABLE = True
except ImportError:
    AI_AGENT_AVAILABLE = False
    class EnhancedUltraSmartAIAgent:
        def get_response(self, message):
            return "Enhanced AI Agent not available"

try:
    from ultra_safe_detector import UltraSafeDetector
    SAFE_DETECTOR_AVAILABLE = True
except ImportError:
    SAFE_DETECTOR_AVAILABLE = False
    class UltraSafeDetector:
        def __init__(self):
            self.detection_enabled = True
        def detect_birds_realtime(self, frame, camera_props=None, frame_quality=None):
            return frame, [], {'birds_in': 0, 'birds_out': 0}

class EnhancedAIDetector:
    """🚀 Enhanced AI Detection System with Intruder Detection"""
    
    def __init__(self):
        print("🚀 กำลังเริ่มต้น Enhanced AI Detection System...")
        
        # Core Components
        self.ai_agent = None
        self.bird_detector = None
        self.intruder_integration = None
        
        # System Status
        self.system_status = {
            'ai_agent': False,
            'bird_detection': False,
            'intruder_detection': False,
            'camera_connection': False,
            'last_update': datetime.now().isoformat()
        }
        
        # Performance Metrics
        self.performance_metrics = {
            'total_detections': 0,
            'intruder_alerts': 0,
            'bird_detections': 0,
            'uptime_start': datetime.now(),
            'average_response_time': 0.0
        }
        
        # Initialize components
        self._init_ai_agent()
        self._init_bird_detector()
        self._init_intruder_detection()
        
        print("✅ Enhanced AI Detection System พร้อมใช้งาน!")
    
    def _init_ai_agent(self):
        """เริ่มต้น Enhanced AI Agent"""
        try:
            if AI_AGENT_AVAILABLE:
                self.ai_agent = EnhancedUltraSmartAIAgent()
                self.system_status['ai_agent'] = True
                print("✅ Enhanced Ultra Smart AI Agent โหลดสำเร็จ")
            else:
                print("⚠️ Enhanced AI Agent ไม่พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่มต้น AI Agent: {e}")
    
    def _init_bird_detector(self):
        """เริ่มต้น Bird Detection System"""
        try:
            if SAFE_DETECTOR_AVAILABLE:
                self.bird_detector = UltraSafeDetector()
                self.system_status['bird_detection'] = True
                print("✅ Bird Detection System โหลดสำเร็จ")
            else:
                print("⚠️ Bird Detection System ไม่พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่มต้น Bird Detector: {e}")
    
    def _init_intruder_detection(self):
        """เริ่มต้น Intruder Detection System"""
        try:
            if INTRUDER_DETECTION_AVAILABLE:
                self.intruder_integration = get_integration_instance()
                self.system_status['intruder_detection'] = True
                print("✅ Ultra Intelligent Intruder Detection System โหลดสำเร็จ")
            else:
                print("⚠️ Intruder Detection System ไม่พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่มต้น Intruder Detection: {e}")
    
    def setup_flask_integration(self, app):
        """ตั้งค่า Flask Integration"""
        try:
            # Setup Intruder Detection APIs
            if self.intruder_integration:
                setup_intruder_integration(app)
                print("✅ Intruder Detection APIs ลงทะเบียนเสร็จสิ้น")
            
            # Add enhanced system status API
            @app.route('/api/system/enhanced-status', methods=['GET'])
            def get_enhanced_system_status():
                """ดึงสถานะระบบแบบครบถ้วน"""
                try:
                    enhanced_status = self.get_enhanced_system_status()
                    return jsonify({
                        'success': True,
                        'data': enhanced_status
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    })
            
            # Add unified detection API
            @app.route('/api/detection/unified', methods=['POST'])
            def unified_detection():
                """API รวมสำหรับการตรวจจับทั้งหมด"""
                try:
                    if 'image' not in request.files:
                        return jsonify({'success': False, 'error': 'No image provided'})
                    
                    file = request.files['image']
                    image_data = file.read()
                    
                    # Convert to OpenCV format
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        return jsonify({'success': False, 'error': 'Invalid image format'})
                    
                    # Run unified detection
                    results = self.unified_frame_detection(frame)
                    
                    return jsonify({
                        'success': True,
                        'results': results
                    })
                    
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)})
            
            print("✅ Enhanced Flask Integration สำเร็จ")
            
        except Exception as e:
            print(f"❌ Flask Integration ล้มเหลว: {e}")
    
    def unified_frame_detection(self, frame):
        """การตรวจจับรวมทั้งนกและสิ่งแปลกปลอม"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'bird_detection': {},
            'intruder_detection': {},
            'combined_alerts': [],
            'system_performance': {}
        }
        
        start_time = datetime.now()
        
        try:
            # 1. Bird Detection
            if self.bird_detector:
                try:
                    processed_frame, bird_detections, bird_stats = self.bird_detector.detect_birds_realtime(frame)
                    results['bird_detection'] = {
                        'success': True,
                        'detections': bird_detections,
                        'statistics': bird_stats,
                        'bird_count': len(bird_detections)
                    }
                    self.performance_metrics['bird_detections'] += len(bird_detections)
                except Exception as e:
                    results['bird_detection'] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # 2. Intruder Detection
            if self.intruder_integration and self.intruder_integration.detector:
                try:
                    intruder_detections = self.intruder_integration.detector.detect_intruders(frame)
                    
                    formatted_detections = []
                    for detection in intruder_detections:
                        formatted_detections.append({
                            'object_type': detection.object_type,
                            'confidence': detection.confidence,
                            'threat_level': detection.threat_level,
                            'priority': detection.priority.value,
                            'bbox': detection.bbox,
                            'center': detection.center,
                            'timestamp': detection.timestamp.isoformat()
                        })
                    
                    results['intruder_detection'] = {
                        'success': True,
                        'detections': formatted_detections,
                        'total_detections': len(formatted_detections)
                    }
                    
                    # Add to combined alerts if critical
                    for detection in intruder_detections:
                        if detection.threat_level > 0.5:
                            results['combined_alerts'].append({
                                'type': 'intruder',
                                'message': f"พบ {detection.object_type} ที่ตำแหน่ง {detection.center}",
                                'priority': detection.priority.value,
                                'threat_level': detection.threat_level
                            })
                    
                    self.performance_metrics['intruder_alerts'] += len(formatted_detections)
                    
                except Exception as e:
                    results['intruder_detection'] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # 3. System Performance
            processing_time = (datetime.now() - start_time).total_seconds()
            results['system_performance'] = {
                'processing_time_seconds': processing_time,
                'frame_size': f"{frame.shape[1]}x{frame.shape[0]}",
                'components_active': sum(self.system_status.values())
            }
            
            # Update metrics
            self.performance_metrics['total_detections'] += 1
            if self.performance_metrics['average_response_time'] == 0:
                self.performance_metrics['average_response_time'] = processing_time
            else:
                self.performance_metrics['average_response_time'] = (
                    self.performance_metrics['average_response_time'] * 0.9 + 
                    processing_time * 0.1
                )
            
        except Exception as e:
            results['error'] = str(e)
            print(f"❌ Unified detection error: {e}")
        
        return results
    
    def get_chat_response(self, message):
        """ดึงการตอบสนองจาก AI Agent"""
        try:
            if self.ai_agent:
                return self.ai_agent.get_response(message)
            else:
                return "AI Agent ไม่พร้อมใช้งาน ขออธิการสำหรับความไม่สะดวก"
        except Exception as e:
            return f"ขออภัย เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"
    
    def get_enhanced_system_status(self):
        """ดึงสถานะระบบแบบครบถ้วน"""
        uptime = datetime.now() - self.performance_metrics['uptime_start']
        
        status = {
            'system_status': self.system_status,
            'performance_metrics': {
                **self.performance_metrics,
                'uptime_seconds': uptime.total_seconds(),
                'uptime_human': str(uptime).split('.')[0]  # Remove microseconds
            },
            'component_details': {
                'ai_agent': {
                    'available': AI_AGENT_AVAILABLE,
                    'active': self.system_status['ai_agent'],
                    'type': 'EnhancedUltraSmartAIAgent'
                },
                'bird_detection': {
                    'available': SAFE_DETECTOR_AVAILABLE,
                    'active': self.system_status['bird_detection'],
                    'type': 'UltraSafeDetector'
                },
                'intruder_detection': {
                    'available': INTRUDER_DETECTION_AVAILABLE,
                    'active': self.system_status['intruder_detection'],
                    'type': 'UltraIntelligentIntruderDetector'
                }
            },
            'last_updated': datetime.now().isoformat()
        }
        
        # Add intruder detection stats if available
        if self.intruder_integration:
            try:
                intruder_stats = self.intruder_integration.get_integration_stats()
                status['intruder_integration_stats'] = intruder_stats
            except Exception as e:
                status['intruder_integration_error'] = str(e)
        
        return status

# Create Flask App with Enhanced Features
def create_enhanced_app():
    """สร้าง Flask App พร้อม Enhanced Features"""
    app = Flask(__name__)
    
    # Initialize Enhanced AI Detector
    ai_detector = EnhancedAIDetector()
    
    # Setup Flask Integration
    ai_detector.setup_flask_integration(app)
    
    # Original routes
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/ai-chat')
    def ai_chat():
        return render_template('ai_chat.html')
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        try:
            data = request.get_json()
            message = data.get('message', '')
            
            if not message:
                return jsonify({
                    'success': False,
                    'error': 'No message provided'
                })
            
            response = ai_detector.get_chat_response(message)
            
            return jsonify({
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/statistics')
    def get_statistics():
        try:
            # Enhanced statistics including intruder detection
            enhanced_status = ai_detector.get_enhanced_system_status()
            
            return jsonify({
                'success': True,
                'data': enhanced_status
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/system-health')
    def system_health():
        try:
            health_status = {
                'status': 'healthy' if sum(ai_detector.system_status.values()) >= 2 else 'degraded',
                'components': ai_detector.system_status,
                'uptime': (datetime.now() - ai_detector.performance_metrics['uptime_start']).total_seconds(),
                'performance': ai_detector.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify({
                'success': True,
                'health': health_status
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    return app, ai_detector

# Run the Enhanced Application
if __name__ == '__main__':
    try:
        print("🚀 เริ่มต้น Enhanced Swallow AI System with Intruder Detection...")
        
        app, detector = create_enhanced_app()
        
        print("✅ Enhanced System พร้อมใช้งาน!")
        print("🌐 Web Interface: http://127.0.0.1:5000")
        print("🤖 AI Chat: http://127.0.0.1:5000/ai-chat")
        print("📊 API Endpoints:")
        print("   - /api/chat - AI Chat")
        print("   - /api/statistics - Enhanced Statistics")
        print("   - /api/system-health - System Health")
        print("   - /api/intruder/status - Intruder Detection Status")
        print("   - /api/intruder/alerts - Recent Alerts")
        print("   - /api/detection/unified - Unified Detection API")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
        
    except Exception as e:
        print(f"❌ ไม่สามารถเริ่มต้นระบบ: {e}")
        traceback.print_exc()
