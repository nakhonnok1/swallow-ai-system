#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 Database Reset and Management Tool
เครื่องมือจัดการและรีเซ็ตฐานข้อมูล
"""

import sqlite3
import os
import shutil
import json
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    """ระบบจัดการฐานข้อมูล"""
    
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.backup_dir = self.current_dir / "backups" 
        self.backup_dir.mkdir(exist_ok=True)
        
    def backup_database(self, db_path):
        """สำรองฐานข้อมูล"""
        if not os.path.exists(db_path):
            print(f"❌ ไม่พบไฟล์: {db_path}")
            return False
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_name = Path(db_path).stem
            backup_name = f"{db_name}_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(db_path, backup_path)
            print(f"✅ สำรองข้อมูลเรียบร้อย: {backup_name}")
            return True
            
        except Exception as e:
            print(f"❌ ไม่สามารถสำรองข้อมูลได้: {e}")
            return False
    
    def reset_database(self, db_path, create_tables=True):
        """รีเซ็ตฐานข้อมูล"""
        try:
            # สำรองก่อนลบ
            if os.path.exists(db_path):
                self.backup_database(db_path)
                os.remove(db_path)
                print(f"🗑️ ลบฐานข้อมูลเก่า: {db_path}")
            
            if create_tables:
                # สร้างฐานข้อมูลใหม่
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # สร้างตาราง bird_activity
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS bird_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    birds_in INTEGER DEFAULT 0,
                    birds_out INTEGER DEFAULT 0,
                    confidence TEXT,
                    weather_data TEXT,
                    meta_data TEXT
                )
                ''')
                
                # สร้างตาราง anomaly_detection
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_detection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    object_type TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    image_path TEXT,
                    status TEXT DEFAULT 'new'
                )
                ''')
                
                # สร้างตาราง system_stats
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_birds_today INTEGER DEFAULT 0,
                    peak_activity_hour INTEGER DEFAULT 0,
                    anomalies_detected INTEGER DEFAULT 0,
                    system_uptime TEXT,
                    performance_score REAL DEFAULT 0.0
                )
                ''')
                
                conn.commit()
                conn.close()
                print(f"✅ สร้างฐานข้อมูลใหม่: {db_path}")
                
            return True
            
        except Exception as e:
            print(f"❌ ไม่สามารถรีเซ็ตฐานข้อมูลได้: {e}")
            return False
    
    def insert_sample_data(self, db_path):
        """เพิ่มข้อมูลตัวอย่าง"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # ข้อมูลตัวอย่าง bird_activity
            sample_activities = [
                (datetime.now(), 5, 2, "0.85", '{"temperature": 25, "humidity": 60}', '{"source": "live_camera"}'),
                (datetime.now(), 3, 4, "0.92", '{"temperature": 26, "humidity": 58}', '{"source": "live_camera"}'),
                (datetime.now(), 7, 1, "0.78", '{"temperature": 24, "humidity": 62}', '{"source": "live_camera"}')
            ]
            
            cursor.executemany('''
            INSERT INTO bird_activity (timestamp, birds_in, birds_out, confidence, weather_data, meta_data)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', sample_activities)
            
            # ข้อมูลตัวอย่าง system_stats
            cursor.execute('''
            INSERT INTO system_stats (timestamp, total_birds_today, peak_activity_hour, anomalies_detected, system_uptime, performance_score)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), 45, 14, 2, "24:30:15", 0.95))
            
            conn.commit()
            conn.close()
            print(f"✅ เพิ่มข้อมูลตัวอย่างเรียบร้อย")
            return True
            
        except Exception as e:
            print(f"❌ ไม่สามารถเพิ่มข้อมูลตัวอย่างได้: {e}")
            return False
    
    def check_and_repair_database(self, db_path):
        """ตรวจสอบและซ่อมแซมฐานข้อมูล"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # ตรวจสอบความสมบูรณ์
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result[0] == "ok":
                print(f"✅ ฐานข้อมูล {db_path} สมบูรณ์")
                
                # ตรวจสอบตาราง
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                required_tables = ['bird_activity', 'anomaly_detection']
                existing_tables = [table[0] for table in tables]
                
                missing_tables = []
                for required in required_tables:
                    if required not in existing_tables:
                        missing_tables.append(required)
                
                if missing_tables:
                    print(f"⚠️ ตารางที่หายไป: {missing_tables}")
                    # สร้างตารางที่หายไป
                    if 'bird_activity' in missing_tables:
                        cursor.execute('''
                        CREATE TABLE bird_activity (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            birds_in INTEGER DEFAULT 0,
                            birds_out INTEGER DEFAULT 0,
                            confidence TEXT,
                            weather_data TEXT,
                            meta_data TEXT
                        )
                        ''')
                        print("✅ สร้างตาราง bird_activity")
                    
                    if 'anomaly_detection' in missing_tables:
                        cursor.execute('''
                        CREATE TABLE anomaly_detection (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            object_type TEXT NOT NULL,
                            confidence TEXT NOT NULL,
                            image_path TEXT,
                            status TEXT DEFAULT 'new'
                        )
                        ''')
                        print("✅ สร้างตาราง anomaly_detection")
                    
                    conn.commit()
                
                conn.close()
                return True
            else:
                print(f"❌ ฐานข้อมูล {db_path} เสียหาย: {result[0]}")
                conn.close()
                return False
                
        except Exception as e:
            print(f"❌ ไม่สามารถตรวจสอบฐานข้อมูลได้: {e}")
            return False
    
    def reset_all_databases(self):
        """รีเซ็ตฐานข้อมูลทั้งหมด"""
        main_databases = [
            "db.sqlite",
            "ultimate_self_learning_ai.db", 
            "swallow_smart_stats.db",
            "anomaly_alerts.db"
        ]
        
        print("🔄 DATABASE RESET TOOL - รีเซ็ตฐานข้อมูลทั้งหมด")
        print("=" * 60)
        print("⚠️ การดำเนินการนี้จะลบข้อมูลทั้งหมด!")
        print("✅ ระบบจะสำรองข้อมูลเก่าไว้ในโฟลเดอร์ backups")
        
        confirm = input("\n❓ ต้องการดำเนินการต่อหรือไม่? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ ยกเลิกการดำเนินการ")
            return
        
        print(f"\n📂 กำลังรีเซ็ตฐานข้อมูล {len(main_databases)} ไฟล์...")
        
        success_count = 0
        for db_name in main_databases:
            db_path = str(self.current_dir / db_name)
            print(f"\n🔄 กำลังรีเซ็ต: {db_name}")
            
            if self.reset_database(db_path):
                success_count += 1
                # เพิ่มข้อมูลตัวอย่างสำหรับฐานข้อมูลหลัก
                if db_name in ["db.sqlite", "ultimate_self_learning_ai.db"]:
                    self.insert_sample_data(db_path)
        
        print(f"\n🎉 รีเซ็ตเสร็จสิ้น: {success_count}/{len(main_databases)} ไฟล์")

def main():
    """ฟังก์ชันหลัก"""
    manager = DatabaseManager()
    
    print("🛠️ DATABASE MANAGEMENT TOOL")
    print("=" * 40)
    print("1. รีเซ็ตฐานข้อมูลทั้งหมด")
    print("2. ตรวจสอบและซ่อมแซมฐานข้อมูล")
    print("3. สำรองฐานข้อมูลทั้งหมด")
    print("4. ออกจากโปรแกรม")
    
    while True:
        choice = input("\n🔢 เลือกตัวเลือก (1-4): ").strip()
        
        if choice == "1":
            manager.reset_all_databases()
            break
        elif choice == "2":
            # ตรวจสอบและซ่อมแซม
            databases = ["db.sqlite", "ultimate_self_learning_ai.db", "swallow_smart_stats.db"]
            for db in databases:
                db_path = str(manager.current_dir / db)
                if os.path.exists(db_path):
                    manager.check_and_repair_database(db_path)
            break
        elif choice == "3":
            # สำรองทั้งหมด
            databases = list(manager.current_dir.glob("*.db")) + list(manager.current_dir.glob("*.sqlite"))
            for db_path in databases:
                manager.backup_database(str(db_path))
            break
        elif choice == "4":
            print("👋 ออกจากโปรแกรม")
            break
        else:
            print("❌ ตัวเลือกไม่ถูกต้อง")

if __name__ == "__main__":
    main()
