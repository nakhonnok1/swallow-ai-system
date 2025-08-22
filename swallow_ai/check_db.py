#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Database Checker - ตรวจสอบสถานะฐานข้อมูล
ตรวจสอบและแสดงข้อมูลในฐานข้อมูลทั้งหมด
"""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path

class DatabaseChecker:
    """ระบบตรวจสอบฐานข้อมูล"""
    
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.db_files = self.find_all_databases()
        
    def find_all_databases(self):
        """ค้นหาไฟล์ฐานข้อมูลทั้งหมด"""
        db_files = []
        for db_file in self.current_dir.glob("*.db"):
            if db_file.is_file():
                db_files.append(db_file)
        
        # เพิ่ม SQLite files
        for sqlite_file in self.current_dir.glob("*.sqlite"):
            if sqlite_file.is_file():
                db_files.append(sqlite_file)
                
        return sorted(db_files)
    
    def check_database(self, db_path):
        """ตรวจสอบฐานข้อมูลหนึ่งไฟล์"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # ดูรายชื่อตาราง
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"\n📁 ฐานข้อมูล: {db_path.name}")
            print(f"📊 ขนาดไฟล์: {db_path.stat().st_size / 1024:.2f} KB")
            print(f"🗂️ จำนวนตาราง: {len(tables)}")
            
            if tables:
                print("📋 รายชื่อตาราง:")
                for table in tables:
                    table_name = table[0]
                    
                    # นับจำนวนแถว
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    # ดูโครงสร้างตาราง
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    print(f"   📄 {table_name}: {count} แถว")
                    print(f"      คอลัมน์: {[col[1] for col in columns]}")
                    
                    # แสดงข้อมูลตัวอย่าง 3 แถวล่าสุด
                    try:
                        cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 3")
                        samples = cursor.fetchall()
                        if samples:
                            print(f"      ตัวอย่างข้อมูล (3 แถวล่าสุด):")
                            for i, sample in enumerate(samples, 1):
                                print(f"        {i}. {sample}")
                    except Exception as e:
                        print(f"      ⚠️ ไม่สามารถดูตัวอย่างข้อมูลได้: {e}")
            else:
                print("   📄 ไม่มีตาราง")
                
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดกับฐานข้อมูล {db_path.name}: {e}")
            return False
    
    def check_all_databases(self):
        """ตรวจสอบฐานข้อมูลทั้งหมด"""
        print("🔍 DATABASE CHECKER - ตรวจสอบสถานะฐานข้อมูลทั้งหมด")
        print("=" * 70)
        print(f"📅 วันที่ตรวจสอบ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 โฟลเดอร์: {self.current_dir}")
        print(f"🗃️ พบฐานข้อมูล: {len(self.db_files)} ไฟล์")
        
        if not self.db_files:
            print("\n⚠️ ไม่พบไฟล์ฐานข้อมูล")
            return
        
        working_dbs = 0
        error_dbs = 0
        
        for db_file in self.db_files:
            if self.check_database(db_file):
                working_dbs += 1
            else:
                error_dbs += 1
        
        print("\n" + "=" * 70)
        print(f"📊 สรุปผลการตรวจสอบ:")
        print(f"   ✅ ฐานข้อมูลที่ทำงานได้: {working_dbs}")
        print(f"   ❌ ฐานข้อมูลที่มีปัญหา: {error_dbs}")
        print(f"   📈 อัตราความสำเร็จ: {working_dbs/len(self.db_files)*100:.1f}%")
        
        if error_dbs == 0:
            print("\n🎉 ฐานข้อมูลทั้งหมดทำงานได้ปกติ!")
        else:
            print(f"\n⚠️ มีฐานข้อมูล {error_dbs} ไฟล์ที่ต้องตรวจสอบ")
    
    def get_main_database_info(self):
        """ดูข้อมูลฐานข้อมูลหลัก"""
        main_dbs = ['db.sqlite', 'ultimate_self_learning_ai.db', 'swallow_smart_stats.db']
        
        print("\n🎯 ข้อมูลฐานข้อมูลหลัก:")
        print("-" * 50)
        
        for db_name in main_dbs:
            db_path = self.current_dir / db_name
            if db_path.exists():
                self.check_database(db_path)
            else:
                print(f"\n📁 ฐานข้อมูล: {db_name}")
                print("❌ ไม่พบไฟล์")

def main():
    """ฟังก์ชันหลักสำหรับรันโปรแกรม"""
    checker = DatabaseChecker()
    checker.check_all_databases()
    checker.get_main_database_info()

if __name__ == "__main__":
    main()
