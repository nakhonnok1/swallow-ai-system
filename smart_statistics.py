"""
🧠 Smart Statistics & AI Assistant System
ระบบคำนวณสถิติอัจฉริยะและ AI ผู้ช่วยสำหรับนกนางแอ่น
"""

import sqlite3
import json
import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import statistics

@dataclass
class BirdStatistics:
    """โครงสร้างข้อมูลสถิติของนก"""
    period: str  # "3days", "7days", "1month"
    total_birds: int
    birds_in: int
    birds_out: int
    net_change: int
    growth_rate: float
    timestamp: datetime.datetime

@dataclass
class AIInsight:
    """ข้อมูลความเข้าใจของ AI"""
    message: str
    insight_type: str  # "growth", "warning", "celebration", "analysis"
    confidence: float
    timestamp: datetime.datetime

class SmartStatistics:
    """ระบบสถิติอัจฉริยะ"""
    
    def __init__(self, db_path: str = "swallow_smart_stats.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """สร้างฐานข้อมูลสถิติ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ตารางข้อมูลรายวัน
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                birds_entering INTEGER DEFAULT 0,
                birds_exiting INTEGER DEFAULT 0,
                net_change INTEGER DEFAULT 0,
                anomalies_detected INTEGER DEFAULT 0,
                video_count INTEGER DEFAULT 0,
                processing_time REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ตารางสถิติรายงวด (3วัน, 7วัน, 1เดือน)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS period_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_type TEXT NOT NULL,  -- '3days', '7days', '1month'
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                total_birds_estimated INTEGER,
                birds_in INTEGER,
                birds_out INTEGER,
                net_change INTEGER,
                growth_rate REAL,
                avg_daily_activity REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ตารางข้อความ AI
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                data_source TEXT,  -- JSON ของข้อมูลที่ใช้วิเคราะห์
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_daily_record(self, birds_in: int, birds_out: int, anomalies: int = 0, 
                        video_count: int = 1, processing_time: float = 0):
        """เพิ่มข้อมูลรายวัน"""
        today = datetime.date.today().isoformat()
        net_change = birds_in - birds_out
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO daily_stats 
            (date, birds_entering, birds_exiting, net_change, anomalies_detected, video_count, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (today, birds_in, birds_out, net_change, anomalies, video_count, processing_time))
        
        conn.commit()
        conn.close()
        
        # คำนวณสถิติใหม่และสร้าง AI insight
        self.calculate_period_statistics()
        self.generate_ai_insights()
    
    def calculate_period_statistics(self):
        """คำนวณสถิติ 3วัน 7วัน 1เดือน"""
        periods = [
            ('3days', 3),
            ('7days', 7),
            ('1month', 30)
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for period_name, days in periods:
            start_date = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
            end_date = datetime.date.today().isoformat()
            
            # ดึงข้อมูลในช่วงเวลา
            cursor.execute("""
                SELECT SUM(birds_entering), SUM(birds_exiting), SUM(net_change), 
                       AVG(birds_entering + birds_exiting), COUNT(*)
                FROM daily_stats 
                WHERE date >= ? AND date <= ?
            """, (start_date, end_date))
            
            result = cursor.fetchone()
            if result and result[0] is not None:
                birds_in, birds_out, net_change, avg_activity, record_count = result
                
                # คำนวณอัตราการเติบโต
                growth_rate = self._calculate_growth_rate(period_name, net_change, days)
                
                # ประมาณการนกทั้งหมดในบ้าน
                total_estimated = self._estimate_total_birds(birds_in, birds_out, days)
                
                # บันทึกสถิติ
                cursor.execute("""
                    INSERT OR REPLACE INTO period_stats 
                    (period_type, start_date, end_date, total_birds_estimated, 
                     birds_in, birds_out, net_change, growth_rate, avg_daily_activity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (period_name, start_date, end_date, total_estimated, 
                      int(birds_in), int(birds_out), int(net_change), 
                      growth_rate, avg_activity))
        
        conn.commit()
        conn.close()
        
        # ลบข้อมูลเก่า (เก็บไว้แค่ 45 วัน)
        self._cleanup_old_data()
    
    def _calculate_growth_rate(self, period_name: str, current_net: int, days: int) -> float:
        """คำนวณอัตราการเติบโต"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # หาช่วงเวลาก่อนหน้า
        prev_start = (datetime.date.today() - datetime.timedelta(days=days*2)).isoformat()
        prev_end = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT SUM(net_change) FROM daily_stats 
            WHERE date >= ? AND date < ?
        """, (prev_start, prev_end))
        
        prev_result = cursor.fetchone()
        prev_net = prev_result[0] if prev_result and prev_result[0] else 0
        
        conn.close()
        
        if prev_net == 0:
            return 100.0 if current_net > 0 else 0.0
        
        return ((current_net - prev_net) / abs(prev_net)) * 100
    
    def _estimate_total_birds(self, birds_in: int, birds_out: int, days: int) -> int:
        """ประมาณการจำนวนนกทั้งหมดในบ้าน"""
        # สูตรประมาณการอย่างง่าย
        # พิจารณาว่านกที่เข้า-ออก = กิจกรรม ประมาณ 20-30% ของประชากรรวม
        activity_ratio = 0.25  # สมมติ 25% ของนกทั้งหมดที่มีกิจกรรม
        
        daily_activity = (birds_in + birds_out) / days
        estimated_total = int(daily_activity / activity_ratio)
        
        # ใส่ขั้นต่ำและสูงสุดที่สมเหตุสมผล
        return max(min(estimated_total, 1000), birds_in)  # อย่างน้อยต้องมีเท่ากับที่เข้ามา
    
    def _cleanup_old_data(self):
        """ลบข้อมูลเก่า เก็บไว้แค่ 45 วัน"""
        cutoff_date = (datetime.date.today() - datetime.timedelta(days=45)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM daily_stats WHERE date < ?", (cutoff_date,))
        cursor.execute("DELETE FROM ai_insights WHERE created_at < datetime('now', '-45 days')")
        
        conn.commit()
        conn.close()
    
    def get_period_statistics(self) -> Dict[str, Dict]:
        """ดึงสถิติทุกช่วงเวลา"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT period_type, total_birds_estimated, birds_in, birds_out, 
                   net_change, growth_rate, avg_daily_activity
            FROM period_stats 
            ORDER BY created_at DESC
        """)
        
        results = {}
        for row in cursor.fetchall():
            period_type, total_est, birds_in, birds_out, net_change, growth_rate, avg_activity = row
            results[period_type] = {
                'total_birds_estimated': total_est,
                'birds_in': birds_in,
                'birds_out': birds_out,
                'net_change': net_change,
                'growth_rate': round(growth_rate, 1),
                'avg_daily_activity': round(avg_activity, 1)
            }
        
        conn.close()
        return results
    
    def generate_ai_insights(self):
        """สร้างข้อความ AI insights"""
        stats = self.get_period_statistics()
        insights = []
        
        # วิเคราะห์การเติบโต
        if '3days' in stats:
            growth_3d = stats['3days']['growth_rate']
            total_birds = stats['3days']['total_birds_estimated']
            net_change = stats['3days']['net_change']
            
            if growth_3d > 20:
                insights.append({
                    'message': f'🎉 ยอดเยี่ยม! ประชากรนกเพิ่มขึ้น {growth_3d:.1f}% ใน 3 วันที่ผ่านมา! ตอนนี้ประมาณว่ามีนกในบ้านประมาณ {total_birds} ตัว การเติบโตนี้แสดงว่าสภาพแวดล้อมดีมาก 🌟',
                    'type': 'celebration',
                    'confidence': 0.9
                })
            elif growth_3d > 0:
                insights.append({
                    'message': f'📈 นกเพิ่มขึ้นอย่างค่อยเป็นค่อยไป {growth_3d:.1f}% ใน 3 วัน ประมาณการนกทั้งหมด {total_birds} ตัว แนวโน้มดี ควรรักษาสภาพแวดล้อมนี้ไว้ 🏠',
                    'type': 'growth',
                    'confidence': 0.8
                })
            elif growth_3d < -10:
                insights.append({
                    'message': f'⚠️ นกลดลง {abs(growth_3d):.1f}% ใน 3 วัน อาจเป็นช่วงอพยพ หรือมีปัจจัยรบกวน ควรตรวจสอบสภาพแวดล้อมรอบบ้านนก 🔍',
                    'type': 'warning',
                    'confidence': 0.7
                })
        
        # วิเคราะห์แนวโน้มรายสัปดาห์
        if '7days' in stats and '3days' in stats:
            weekly_growth = stats['7days']['growth_rate']
            three_day_growth = stats['3days']['growth_rate']
            
            if weekly_growth > three_day_growth:
                insights.append({
                    'message': f'📊 แนวโน้มระยะยาวดีกว่าระยะสั้น! การเติบโต 7 วัน ({weekly_growth:.1f}%) สูงกว่า 3 วัน ({three_day_growth:.1f}%) แสดงว่าพัฒนาการมั่นคง 💪',
                    'type': 'analysis',
                    'confidence': 0.8
                })
        
        # วิเคราะห์รายเดือน
        if '1month' in stats:
            monthly_data = stats['1month']
            if monthly_data['avg_daily_activity'] > 10:
                insights.append({
                    'message': f'🌟 บ้านนกของคุณมีชีวิตชีวา! กิจกรรมเฉลี่ย {monthly_data["avg_daily_activity"]:.1f} ครั้ง/วัน ในเดือนนี้ นกชอบที่นี่มาก! 🐦',
                    'type': 'celebration',
                    'confidence': 0.9
                })
        
        # สุ่มเลือก insight ที่ดีที่สุด
        if insights:
            best_insight = max(insights, key=lambda x: x['confidence'])
            self._save_ai_insight(best_insight)
    
    def _save_ai_insight(self, insight: Dict):
        """บันทึก AI insight ลงฐานข้อมูล"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ai_insights (message, insight_type, confidence, data_source)
            VALUES (?, ?, ?, ?)
        """, (insight['message'], insight['type'], insight['confidence'], 
              json.dumps(self.get_period_statistics())))
        
        conn.commit()
        conn.close()
    
    def get_latest_ai_insights(self, limit: int = 3) -> List[Dict]:
        """ดึงข้อความ AI ล่าสุด"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT message, insight_type, confidence, created_at
            FROM ai_insights 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        insights = []
        for row in cursor.fetchall():
            message, insight_type, confidence, created_at = row
            insights.append({
                'message': message,
                'type': insight_type,
                'confidence': confidence,
                'timestamp': created_at
            })
        
        conn.close()
        return insights
    
    def get_dashboard_data(self) -> Dict:
        """ดึงข้อมูลทั้งหมดสำหรับ dashboard"""
        return {
            'statistics': self.get_period_statistics(),
            'ai_insights': self.get_latest_ai_insights(5),
            'last_updated': datetime.datetime.now().isoformat()
        }

class SmartAIAssistant:
    """AI ผู้ช่วยอัจฉริยะ (ออฟไลน์)"""
    
    def __init__(self):
        self.responses = {
            'morning': [
                '🌅 สวัสดีตอนเช้า! นกๆ เริ่มตื่นแล้วนะ พร้อมเริ่มวันใหม่ไหม?',
                '☀️ อรุณสวัสดิ์! วันนี้อากาศดี นกคงชอบบินออกมาเล่น',
                '🐦 เช้าดี! ได้เวลาดูกิจกรรมของนกแล้ว'
            ],
            'growth_good': [
                '🎉 ว้าว! นกเพิ่มขึ้นเยอะมาก บ้านนกคุณเป็นที่นิยม!',
                '📈 เยี่ยมมาก! การเติบโตนี้แสดงว่าสภาพแวดล้อมดีเยี่ยม',
                '🌟 สุดยอด! นกๆ ชอบบ้านหลังนี้จริงๆ'
            ],
            'growth_concern': [
                '🤔 อืมม... นกลดลงนิดหน่อย อาจเป็นฤดูกาลหรือมีอะไรรบกวน',
                '👀 สังเกตเห็นนกลดลง ลองตรวจสอบดูว่ามีอะไรผิดปกติไหม',
                '🔍 นกน้อยลง อาจต้องดูแลบ้านนกให้ดีขึ้น'
            ],
            'anomaly_detected': [
                '🚨 เอ๊ะ! มีสิ่งแปลกปลอมเข้ามา ควรไปดูหน่อยนะ',
                '⚠️ ระวัง! มีอะไรบางอย่างที่ไม่ใช่นกในพื้นที่',
                '🔔 แจ้งเตือน! ตรวจพบกิจกรรมผิดปกติ'
            ],
            'celebration': [
                '🎊 ปาร์ตี้! นกเยอะมากแล้ว บ้านคุณเป็นชุมชนนกที่ดีที่สุด!',
                '🏆 เป็นเจ้าของบ้านนกยอดเยี่ยม! นกๆ มีความสุขมาก',
                '💝 คุณดูแลนกได้เยี่ยมมาก นกทุกตัวรักที่นี่'
            ]
        }
    
    def get_smart_message(self, context: Dict) -> str:
        """สร้างข้อความอัจฉริยะตามสถานการณ์"""
        current_hour = datetime.datetime.now().hour
        
        # ข้อความตอนเช้า
        if 5 <= current_hour <= 9:
            return np.random.choice(self.responses['morning'])
        
        # วิเคราะห์จากข้อมูลสถิติ
        if 'statistics' in context:
            stats = context['statistics']
            
            # ตรวจสอบการเติบโต
            if '3days' in stats:
                growth = stats['3days'].get('growth_rate', 0)
                total_birds = stats['3days'].get('total_birds_estimated', 0)
                
                if growth > 15 or total_birds > 50:
                    return np.random.choice(self.responses['celebration'])
                elif growth > 5:
                    return np.random.choice(self.responses['growth_good'])
                elif growth < -5:
                    return np.random.choice(self.responses['growth_concern'])
        
        # ข้อความทั่วไป
        general_messages = [
            '💭 ระบบทำงานปกติ กำลังเฝ้าดูนกๆ อย่างใกล้ชิด',
            '🧠 AI พร้อมช่วยวิเคราะห์ข้อมูลนกให้คุณ',
            '📊 กำลังประมวลผลข้อมูลเพื่อให้ข้อมูลที่แม่นยำ',
            '🔄 ระบบอัปเดตตัวเองอย่างต่อเนื่อง',
            '👁️ กำลังติดตามพฤติกรรมนกอย่างระมัดระวัง'
        ]
        
        return np.random.choice(general_messages)

if __name__ == "__main__":
    # ทดสอบระบบ
    stats = SmartStatistics()
    ai = SmartAIAssistant()
    
    # เพิ่มข้อมูลทดสอบ
    stats.add_daily_record(birds_in=5, birds_out=2, anomalies=1)
    
    # ดูผลลัพธ์
    dashboard_data = stats.get_dashboard_data()
    
    print("📊 Smart Statistics Dashboard")
    print("=" * 50)
    print(json.dumps(dashboard_data, indent=2, ensure_ascii=False))
    
    print("\n🧠 AI Assistant Message:")
    print(ai.get_smart_message(dashboard_data))
