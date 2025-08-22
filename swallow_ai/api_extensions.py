# api_extensions.py - Additional API endpoints for management
from flask import jsonify, request
from datetime import datetime, timedelta
import os

def register_management_apis(app, db, performance_monitor, backup_system, error_handler):
    """ลงทะเบียน API endpoints เพิ่มเติมสำหรับการจัดการระบบ"""
    
    @app.route('/api/system/performance')
    def get_performance_stats():
        """API สำหรับดึงสถิติประสิทธิภาพระบบ"""
        try:
            stats = performance_monitor.get_stats()
            health = performance_monitor.get_health_status()
            return jsonify({
                'performance': stats,
                'health': health,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            error_handler.log_error(e, "get_performance_stats")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/system/backup', methods=['POST'])
    def create_backup():
        """API สำหรับสร้าง backup ฐานข้อมูล"""
        try:
            backup_file = backup_system.backup_database()
            export_file = backup_system.export_stats_to_json()
            
            return jsonify({
                'success': True,
                'backup_file': backup_file,
                'export_file': export_file,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            error_handler.log_error(e, "create_backup")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/system/logs')
    def get_recent_logs():
        """API สำหรับดึง log ล่าสุด"""
        try:
            log_lines = request.args.get('lines', 50, type=int)
            log_file = 'app.log'
            
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_lines = lines[-log_lines:] if len(lines) > log_lines else lines
                
                return jsonify({
                    'logs': [line.strip() for line in recent_lines],
                    'total_lines': len(lines),
                    'showing': len(recent_lines)
                })
            else:
                return jsonify({'logs': [], 'total_lines': 0, 'showing': 0})
                
        except Exception as e:
            error_handler.log_error(e, "get_recent_logs")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/statistics/advanced')
    def get_advanced_statistics():
        """API สำหรับสถิติขั้นสูง"""
        try:
            # สถิติรายชั่วโมง
            hourly_stats = db.get_hourly_stats()
            
            # สถิติ anomaly
            anomaly_summary = db.get_anomaly_summary()
            
            # แนวโน้มการเปลี่ยนแปลง
            trend_analysis = db.get_trend_analysis()
            
            return jsonify({
                'hourly_stats': hourly_stats,
                'anomaly_summary': anomaly_summary,
                'trend_analysis': trend_analysis,
                'generated_at': datetime.now().isoformat()
            })
        except Exception as e:
            error_handler.log_error(e, "get_advanced_statistics")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/config')
    def get_current_config():
        """API สำหรับดึงการตั้งค่าปัจจุบัน"""
        try:
            from config import Config
            
            # เฉพาะค่าที่ปลอดภัยที่จะแสดง
            safe_config = {
                'BIRD_CLASS_ID': Config.BIRD_CLASS_ID,
                'DETECTION_THRESHOLD': Config.DETECTION_THRESHOLD,
                'BIRD_CONFIDENCE_THRESHOLD': Config.BIRD_CONFIDENCE_THRESHOLD,
                'COUNTING_LINE_Y': Config.COUNTING_LINE_Y,
                'ANOMALY_COOLDOWN': Config.ANOMALY_COOLDOWN,
                'AUTO_BACKUP_ENABLED': Config.AUTO_BACKUP_ENABLED,
                'BACKUP_RETENTION_DAYS': Config.BACKUP_RETENTION_DAYS
            }
            
            return jsonify(safe_config)
        except Exception as e:
            error_handler.log_error(e, "get_current_config")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/config', methods=['PUT'])
    def update_config():
        """API สำหรับอัปเดตการตั้งค่า"""
        try:
            from config import Config
            
            new_config = request.json
            updated_fields = []
            
            # รายการฟิลด์ที่อนุญาตให้แก้ไข
            allowed_fields = [
                'DETECTION_THRESHOLD',
                'BIRD_CONFIDENCE_THRESHOLD', 
                'COUNTING_LINE_Y',
                'ANOMALY_COOLDOWN'
            ]
            
            for field, value in new_config.items():
                if field in allowed_fields:
                    if hasattr(Config, field):
                        setattr(Config, field, value)
                        updated_fields.append(field)
            
            return jsonify({
                'success': True,
                'updated_fields': updated_fields,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            error_handler.log_error(e, "update_config")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/system/reset_stats', methods=['POST'])
    def reset_statistics():
        """API สำหรับรีเซ็ตสถิติ"""
        try:
            # สำรองข้อมูลก่อนรีเซ็ต
            backup_system.backup_database()
            
            # รีเซ็ตสถิติ global
            global bird_stats
            bird_stats = {
                'birds_in': 0,
                'birds_out': 0,
                'current_inside': 0,
                'last_updated': datetime.now().strftime("%H:%M:%S")
            }
            
            return jsonify({
                'success': True,
                'message': 'Statistics reset successfully',
                'backup_created': True,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            error_handler.log_error(e, "reset_statistics")
            return jsonify({'error': str(e)}), 500

    print("🔗 Management API endpoints registered successfully")

# Extensions สำหรับ Database class
def extend_database_class(db_class):
    """เพิ่มเมธอดใหม่ให้กับ Database class"""
    
    def get_hourly_stats(self):
        """ดึงสถิติรายชั่วโมงย้อนหลัง 24 ชั่วโมง"""
        try:
            from models import SessionLocal, BirdActivity
            session = SessionLocal()
            
            # Query สำหรับรวมข้อมูลตามชั่วโมง
            from sqlalchemy import func
            hourly_data = session.query(
                func.strftime('%H', BirdActivity.timestamp).label('hour'),
                func.sum(BirdActivity.birds_in).label('total_in'),
                func.sum(BirdActivity.birds_out).label('total_out')
            ).filter(
                BirdActivity.timestamp >= datetime.now() - timedelta(hours=24)
            ).group_by(
                func.strftime('%H', BirdActivity.timestamp)
            ).all()
            
            session.close()
            
            return [{
                'hour': int(row.hour),
                'birds_in': row.total_in or 0,
                'birds_out': row.total_out or 0
            } for row in hourly_data]
            
        except Exception as e:
            print(f"Error getting hourly stats: {e}")
            return []
    
    def get_anomaly_summary(self):
        """ดึงสรุปข้อมูล anomaly"""
        try:
            from models import SessionLocal, AnomalyDetection
            session = SessionLocal()
            
            # นับจำนวน anomaly แต่ละประเภท
            from sqlalchemy import func
            anomaly_counts = session.query(
                AnomalyDetection.object_type,
                func.count(AnomalyDetection.id).label('count')
            ).group_by(AnomalyDetection.object_type).all()
            
            # นับ anomaly รายวัน (7 วันล่าสุด)
            daily_anomalies = session.query(
                func.date(AnomalyDetection.timestamp).label('date'),
                func.count(AnomalyDetection.id).label('count')
            ).filter(
                AnomalyDetection.timestamp >= datetime.now() - timedelta(days=7)
            ).group_by(
                func.date(AnomalyDetection.timestamp)
            ).all()
            
            session.close()
            
            return {
                'by_type': [{
                    'object_type': row.object_type,
                    'count': row.count
                } for row in anomaly_counts],
                'daily': [{
                    'date': row.date.isoformat(),
                    'count': row.count
                } for row in daily_anomalies]
            }
            
        except Exception as e:
            print(f"Error getting anomaly summary: {e}")
            return {'by_type': [], 'daily': []}
    
    def get_trend_analysis(self):
        """วิเคราะห์แนวโน้มการเปลี่ยนแปลง"""
        try:
            daily_stats = self.get_daily_stats()
            
            if len(daily_stats) < 2:
                return {'trend': 'insufficient_data', 'change_percentage': 0}
            
            # คำนวณการเปลี่ยนแปลงจากวันล่าสุดกับวันก่อนหน้า
            latest = daily_stats[0]
            previous = daily_stats[1]
            
            latest_total = latest['total_in'] - latest['total_out']
            previous_total = previous['total_in'] - previous['total_out']
            
            if previous_total == 0:
                change_percentage = 100 if latest_total > 0 else 0
            else:
                change_percentage = ((latest_total - previous_total) / previous_total) * 100
            
            if change_percentage > 10:
                trend = 'increasing'
            elif change_percentage < -10:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'change_percentage': round(change_percentage, 2),
                'latest_net': latest_total,
                'previous_net': previous_total
            }
            
        except Exception as e:
            print(f"Error in trend analysis: {e}")
            return {'trend': 'error', 'change_percentage': 0}
    
    # เพิ่มเมธอดใหม่ให้กับคลาส
    db_class.get_hourly_stats = get_hourly_stats
    db_class.get_anomaly_summary = get_anomaly_summary
    db_class.get_trend_analysis = get_trend_analysis
    
    print("📊 Database extensions added successfully")
