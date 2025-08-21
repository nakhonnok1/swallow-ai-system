"""
📊 เครื่องมือติดตามสถิติและการแจ้งเตือนสำหรับสตรีม 24 ชั่วโมง
"""

import sqlite3
import json
import time
import smtplib
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple
import pandas as pd

class StreamMonitor:
    """📊 ระบบติดตามสถิติสตรีม"""
    
    def __init__(self, db_path: str = 'live_stream_data.db'):
        self.db_path = db_path
        self.load_config()
        
    def load_config(self):
        """📂 โหลดการตั้งค่า"""
        try:
            with open('live_stream_config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except:
            self.config = {"notification_settings": {"enable_alerts": False}}
    
    def get_hourly_stats(self, hours: int = 24) -> List[Dict]:
        """📈 ดึงสถิติรายชั่วโมง"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                COUNT(CASE WHEN event_type = 'entry' THEN 1 END) as entries,
                COUNT(CASE WHEN event_type = 'exit' THEN 1 END) as exits
            FROM bird_events 
            WHERE timestamp >= ?
            GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
            ORDER BY hour
        ''', (start_time,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'hour': row[0],
                'entries': row[1],
                'exits': row[2]
            })
        
        conn.close()
        return results
    
    def get_daily_summary(self) -> Dict:
        """📅 สรุปสถิติรายวัน"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN event_type = 'entry' THEN 1 END) as entries,
                COUNT(CASE WHEN event_type = 'exit' THEN 1 END) as exits,
                COUNT(*) as total_events
            FROM bird_events 
            WHERE date(timestamp) = ?
        ''', (today,))
        
        row = cursor.fetchone()
        
        # สถิติรายสัปดาห์
        week_ago = today - timedelta(days=7)
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN event_type = 'entry' THEN 1 END) as entries,
                COUNT(CASE WHEN event_type = 'exit' THEN 1 END) as exits
            FROM bird_events 
            WHERE date(timestamp) >= ?
        ''', (week_ago,))
        
        week_row = cursor.fetchone()
        
        conn.close()
        
        return {
            'today': {
                'entries': row[0] if row else 0,
                'exits': row[1] if row else 0,
                'total_events': row[2] if row else 0
            },
            'this_week': {
                'entries': week_row[0] if week_row else 0,
                'exits': week_row[1] if week_row else 0
            }
        }
    
    def generate_chart(self, hours: int = 24, save_path: str = 'bird_stats.png'):
        """📊 สร้างกราฟสถิติ"""
        
        stats = self.get_hourly_stats(hours)
        
        if not stats:
            print("⚠️ ไม่มีข้อมูลสำหรับสร้างกราฟ")
            return
        
        # แปลงข้อมูล
        hours_list = [datetime.strptime(s['hour'], '%Y-%m-%d %H:%M:%S') for s in stats]
        entries = [s['entries'] for s in stats]
        exits = [s['exits'] for s in stats]
        
        # สร้างกราฟ
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(hours_list, entries, 'g-', label='นกเข้า', linewidth=2, marker='o')
        ax.plot(hours_list, exits, 'r-', label='นกออก', linewidth=2, marker='s')
        
        ax.set_xlabel('เวลา')
        ax.set_ylabel('จำนวนนก')
        ax.set_title(f'สถิติการเข้าออกของนกแอ่น ({hours} ชั่วโมงล่าสุด)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ตั้งค่าแกน X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 บันทึกกราฟที่: {save_path}")
        plt.close()
    
    def check_anomalies(self) -> List[Dict]:
        """🚨 ตรวจสอบความผิดปกติ"""
        
        anomalies = []
        
        # ดึงข้อมูล 1 ชั่วโมงล่าสุด
        recent_stats = self.get_hourly_stats(1)
        
        if recent_stats:
            latest = recent_stats[-1]
            
            # ตรวจสอบการเข้าผิดปกติ (มากกว่า 50 ตัวต่อชั่วโมง)
            if latest['entries'] > 50:
                anomalies.append({
                    'type': 'high_entry',
                    'message': f"🚨 นกเข้าผิดปกติ: {latest['entries']} ตัวใน 1 ชั่วโมง",
                    'value': latest['entries'],
                    'timestamp': latest['hour']
                })
            
            # ตรวจสอบการออกผิดปกติ (มากกว่า 20 ตัวต่อชั่วโมง)
            if latest['exits'] > 20:
                anomalies.append({
                    'type': 'high_exit',
                    'message': f"⚠️ นกออกผิดปกติ: {latest['exits']} ตัวใน 1 ชั่วโมง",
                    'value': latest['exits'],
                    'timestamp': latest['hour']
                })
        
        # ตรวจสอบการไม่มีกิจกรรม (3 ชั่วโมงไม่มีการเข้าออก)
        no_activity_stats = self.get_hourly_stats(3)
        if no_activity_stats:
            total_activity = sum(s['entries'] + s['exits'] for s in no_activity_stats)
            if total_activity == 0:
                anomalies.append({
                    'type': 'no_activity',
                    'message': "⚠️ ไม่มีกิจกรรมเป็นเวลา 3 ชั่วโมง",
                    'value': 0,
                    'timestamp': datetime.now().isoformat()
                })
        
        return anomalies
    
    def send_alert(self, message: str):
        """📧 ส่งการแจ้งเตือน"""
        
        if not self.config.get('notification_settings', {}).get('enable_alerts', False):
            return
        
        # Webhook
        webhook_url = self.config.get('notification_settings', {}).get('webhook_url', '')
        if webhook_url:
            try:
                payload = {
                    'text': f"🐦 การแจ้งเตือนระบบตรวจจับนกแอ่น: {message}",
                    'timestamp': datetime.now().isoformat()
                }
                requests.post(webhook_url, json=payload, timeout=10)
                print(f"📤 ส่งการแจ้งเตือน Webhook: {message}")
            except Exception as e:
                print(f"❌ ส่ง Webhook ไม่สำเร็จ: {e}")
        
        # Email (ถ้าตั้งค่าไว้)
        if self.config.get('notification_settings', {}).get('email_notifications', False):
            self.send_email_alert(message)
    
    def send_email_alert(self, message: str):
        """📧 ส่งอีเมลแจ้งเตือน"""
        
        try:
            # ตั้งค่าอีเมล (ควรเก็บไว้ในไฟล์แยก)
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "your_email@gmail.com"
            sender_password = "your_app_password"
            receiver_email = "alert_receiver@gmail.com"
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = "🐦 การแจ้งเตือนระบบตรวจจับนกแอ่น"
            
            body = f"""
เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
ข้อความ: {message}

สถิติล่าสุด:
{self.get_daily_summary()}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"📧 ส่งอีเมลแจ้งเตือน: {message}")
            
        except Exception as e:
            print(f"❌ ส่งอีเมลไม่สำเร็จ: {e}")
    
    def run_monitoring(self, check_interval: int = 300):
        """🔄 รันการติดตามต่อเนื่อง"""
        
        print("📊 เริ่มระบบติดตามสถิติ")
        print(f"🔄 ตรวจสอบทุก {check_interval} วินาที")
        
        while True:
            try:
                # ตรวจสอบความผิดปกติ
                anomalies = self.check_anomalies()
                
                for anomaly in anomalies:
                    print(f"🚨 {anomaly['message']}")
                    self.send_alert(anomaly['message'])
                
                # สร้างกราฟทุก 1 ชั่วโมง
                current_time = datetime.now()
                if current_time.minute == 0:  # ทุกชั่วโมงตรง
                    self.generate_chart()
                
                # แสดงสถิติ
                summary = self.get_daily_summary()
                print(f"""
📊 === สถิติปัจจุบัน ===
📅 วันนี้: เข้า {summary['today']['entries']} ออก {summary['today']['exits']}
📅 สัปดาห์นี้: เข้า {summary['this_week']['entries']} ออก {summary['this_week']['exits']}
⏰ เวลาตรวจสอบ: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("🛑 หยุดการติดตาม")
                break
            except Exception as e:
                print(f"❌ ข้อผิดพลาดในการติดตาม: {e}")
                time.sleep(10)

def main():
    """🚀 เริ่มต้นระบบติดตาม"""
    
    monitor = StreamMonitor()
    
    print("📊 เครื่องมือติดตามสถิติสตรีม 24 ชั่วโมง")
    print("=" * 50)
    print("1 - รันการติดตามต่อเนื่อง")
    print("2 - สร้างกราฟสถิติ")
    print("3 - แสดงสถิติรายวัน")
    print("4 - ตรวจสอบความผิดปกติ")
    
    choice = input("เลือกฟังก์ชัน (1-4): ").strip()
    
    if choice == '1':
        monitor.run_monitoring()
    elif choice == '2':
        hours = int(input("จำนวนชั่วโมงที่ต้องการ (24): ") or "24")
        monitor.generate_chart(hours)
    elif choice == '3':
        summary = monitor.get_daily_summary()
        print(f"""
📊 === สถิติรายวัน ===
📅 วันนี้:
   🐦 นกเข้า: {summary['today']['entries']} ตัว
   🐦 นกออก: {summary['today']['exits']} ตัว
   📊 เหตุการณ์ทั้งหมด: {summary['today']['total_events']}

📅 สัปดาห์นี้:
   🐦 นกเข้า: {summary['this_week']['entries']} ตัว
   🐦 นกออก: {summary['this_week']['exits']} ตัว
        """)
    elif choice == '4':
        anomalies = monitor.check_anomalies()
        if anomalies:
            print("🚨 พบความผิดปกติ:")
            for anomaly in anomalies:
                print(f"   {anomaly['message']}")
        else:
            print("✅ ไม่พบความผิดปกติ")

if __name__ == "__main__":
    main()
