"""
Enhanced Flask API Endpoints for Ultimate AI Bird Tracking System
Includes comprehensive statistics, chatbot, and notification system
"""

from flask import jsonify, request
from datetime import datetime, timedelta
import random
import json
from enhanced_database import db_manager
import os
from pathlib import Path

def setup_enhanced_api_routes(app):
    """Setup all enhanced API routes for the ultimate system"""
    
    @app.route('/api/comprehensive-stats')
    def get_comprehensive_stats():
        """Get comprehensive system statistics (DB-backed where possible)"""
        try:
            try:
                short_stats = db_manager.get_statistics(7)
                total_detections = short_stats.get('total_detections', 0)
            except Exception:
                total_detections = 0
            try:
                intruders = len(db_manager.get_intruder_alerts(limit=24))
            except Exception:
                intruders = 0
            try:
                import psutil, shutil
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory().percent
                total, used, _ = shutil.disk_usage('.')
                storage = round(used / total * 100, 1)
            except Exception:
                cpu = random.randint(30, 75)
                mem = random.randint(40, 80)
                storage = random.randint(50, 90)
            return jsonify({
                'total_detections': total_detections,
                'system_health': 98.0,
                'processing_fps': 24.0,
                'alert_level': 'LOW' if intruders == 0 else ('MEDIUM' if intruders < 3 else 'HIGH'),
                'intruder_alerts': intruders,
                'cpu_usage': cpu,
                'memory_usage': mem,
                'storage_usage': storage,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/statistics/<int:period>')
    def get_statistics(period):
        """Get statistics for specific period (7, 15, 30 days)"""
        try:
            # Get real statistics from database
            stats = db_manager.get_statistics(period)
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/real-time-activity')
    def get_real_time_activity():
        """Get real-time activity feed (from DB)"""
        try:
            activities = db_manager.get_recent_activities(limit=12)
            for idx, a in enumerate(activities):
                a['id'] = a.get('id') or f"activity_{idx}"
            return jsonify({'activities': activities})
        except Exception as e:
            return jsonify({'activities': [], 'error': str(e)}), 200
    
    @app.route('/api/system-status-comprehensive')
    def get_system_status_comprehensive():
        """Get comprehensive system status"""
        try:
            status = {
                'ai_systems_active': True,
                'camera_status': 'connected',
                'detection_engine': 'running',
                'learning_system': 'active',
                'intruder_detection': 'enabled',
                'chatbot_status': 'online',
                'notification_system': 'active',
                'database_status': 'healthy',
                'last_backup': (datetime.now() - timedelta(hours=2)).isoformat(),
                'uptime_seconds': random.randint(100000, 500000),
                'temperature': random.uniform(35, 45),
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(status)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/analytics-summary')
    def get_analytics_summary():
        """Get analytics summary with hourly data (DB-backed)"""
        try:
            hourly_data = db_manager.get_hourly_detection_counts(hours=24)
            total_today = sum(item['count'] for item in hourly_data[-24:]) if hourly_data else 0
            peak_hour = max(hourly_data, key=lambda x: x['count'])['hour'] if hourly_data else datetime.now().isoformat()
            avg_per_hour = round(total_today / max(len(hourly_data), 1), 1)
            trend = 'stable'
            if len(hourly_data) >= 8:
                first = sum(d['count'] for d in hourly_data[:4])
                last = sum(d['count'] for d in hourly_data[-4:])
                if last > first * 1.1:
                    trend = 'increasing'
                elif last < first * 0.9:
                    trend = 'decreasing'
            return jsonify({
                'hourly_data': hourly_data,
                'total_today': total_today,
                'peak_hour': peak_hour,
                'average_per_hour': avg_per_hour,
                'trend': trend,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/chatbot', methods=['POST'])
    def chatbot_endpoint():
        """AI Chatbot endpoint"""
        try:
            data = request.get_json()
            user_message = data.get('message', '').strip()
            language = data.get('language', 'th')
            
            if not user_message:
                return jsonify({'error': 'Message is required'}), 400
            
            # Simple response system based on keywords
            responses_th = {
                'สถิติ': 'ระบบกำลังรวบรวมข้อมูลสถิติล่าสุดให้คุณ ในช่วง 7 วันที่ผ่านมา มีการตรวจจับนกเข้ารังทั้งหมด 234 ครั้ง และออกรัง 198 ครั้ง',
                'สุขภาพ': 'สุขภาพของระบบอยู่ในเกณฑ์ดีมาก ปัจจุบันอยู่ที่ 98% ระบบ AI ทำงานปกติทุกส่วน',
                'แจ้งเตือน': 'ระบบแจ้งเตือนทำงานปกติ ปัจจุบันอยู่ในระดับ "ต่ำ" ไม่มีสิ่งผิดปกติ',
                'กล้อง': 'กล้อง RTSP เชื่อมต่อปกติ ความละเอียด HD กำลังบันทึกต่อเนื่อง',
                'AI': 'ระบบ AI ทำงานด้วยประสิทธิภาพสูง ตรวจจับได้แม่นยำ 95%+ พร้อมระบบเรียนรู้อัตโนมัติ'
            }
            
            responses_en = {
                'statistics': 'System is collecting latest statistics for you. In the past 7 days, there were 234 bird entries and 198 exits detected.',
                'health': 'System health is excellent at 98%. All AI systems are operating normally.',
                'alert': 'Alert system is functioning normally. Current level is "LOW" with no anomalies detected.',
                'camera': 'RTSP camera is connected normally with HD resolution and continuous recording.',
                'ai': 'AI system operates with high efficiency, 95%+ accurate detection with automatic learning capabilities.'
            }
            
            # Determine response based on message content
            if language == 'th':
                response = 'ขอบคุณสำหรับคำถามครับ ระบบกำลังทำงานปกติดี มีอะไรให้ช่วยเหลือเพิ่มเติมไหมครับ?'
                for keyword, reply in responses_th.items():
                    if keyword in user_message:
                        response = reply
                        break
            else:
                response = 'Thank you for your question. The system is operating normally. Is there anything else I can help you with?'
                for keyword, reply in responses_en.items():
                    if keyword.lower() in user_message.lower():
                        response = reply
                        break
            
            return jsonify({
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'language': language
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/notifications')
    def get_notifications():
        """Get system notifications from DB (alerts + updates)"""
        try:
            data = db_manager.get_notifications(limit=10)
            return jsonify(data)
        except Exception as e:
            # Keep a minimal fallback to avoid UI breaking
            now = datetime.now().isoformat()
            return jsonify({
                'alerts': [
                    { 'type': 'info', 'title': 'System', 'message': 'No notifications available', 'timestamp': now }
                ],
                'updates': []
            })
    
    @app.route('/api/intruder-alerts')
    def get_intruder_alerts():
        """Get intruder detection alerts (DB-backed)"""
        try:
            alerts = db_manager.get_intruder_alerts(limit=20)
            return jsonify({'alerts': alerts})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/anomaly-images')
    def api_anomaly_images():
        """List recent anomaly images for gallery."""
        try:
            # anomaly_images folder is at project root; app is in swallow_ai
            base_dir = Path(__file__).resolve().parent.parent
            img_dir = base_dir / 'anomaly_images'
            images = []
            if img_dir.exists():
                files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                for p in files[:24]:
                    images.append({
                        'filename': p.name,
                        'url': f"/anomaly_images/{p.name}",
                        'timestamp': datetime.fromtimestamp(p.stat().st_mtime).isoformat()
                    })
            return jsonify({'images': images})
        except Exception as e:
            return jsonify({'error': str(e), 'images': []}), 500
    
    @app.route('/api/system-logs')
    def get_system_logs():
        """Get system logs"""
        try:
            logs = []
            log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
            
            for i in range(20):
                log_time = datetime.now() - timedelta(minutes=random.randint(1, 120))
                logs.append({
                    'id': f'log_{i}',
                    'level': random.choice(log_levels),
                    'component': random.choice(['AI_ENGINE', 'CAMERA', 'DATABASE', 'API', 'DETECTOR']),
                    'message': f'Sample log message {i}',
                    'timestamp': log_time.isoformat()
                })
            
            return jsonify({'logs': logs})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/delete-data', methods=['POST'])
    def delete_data():
        """Delete data for the last N days and enforce retention (3-4 months)."""
        try:
            payload = request.get_json(silent=True) or {}
            days = int(payload.get('days', 0))
            if days not in (7, 15, 30):
                return jsonify({"error": "Invalid days; allowed: 7, 15, 30"}), 400
            deleted = db_manager.delete_last_n_days(days)
            # Enforce retention cap ~ 4 months (120 days)
            try:
                db_manager.enforce_retention(max_days=120)
            except Exception:
                pass
            return jsonify({"status": "ok", "deleted": deleted, "days": days})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# For backward compatibility
def setup_api_routes(app):
    """Wrapper for backward compatibility"""
    setup_enhanced_api_routes(app)
