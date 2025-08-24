#!/usr/bin/env python3
"""
Script to fix database schema for detections table
"""

import sqlite3
import os

def fix_database_schema():
    """Fix database schema for all relevant databases"""
    
    # Check all database files that might have detections table
    db_files = [
        'enhanced_ai_system.db',
        'object_detection_alerts.db', 
        'intelligent_intruder_detections.db',
        'swallow_smart_stats.db'
    ]

    for db_file in db_files:
        if os.path.exists(db_file):
            print(f'\n🔍 ตรวจสอบ database: {db_file}')
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Check if detections table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
                if cursor.fetchone():
                    print(f'  ✅ Table detections มีอยู่ใน {db_file}')
                    
                    # Get current columns
                    cursor.execute('PRAGMA table_info(detections)')
                    columns = cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    print(f'  📋 Columns: {", ".join(column_names)}')
                    
                    # Add missing columns
                    if 'camera_source' not in column_names:
                        print('  ⚙️ เพิ่ม camera_source column...')
                        cursor.execute('ALTER TABLE detections ADD COLUMN camera_source TEXT DEFAULT "demo"')
                        print('  ✅ เพิ่ม camera_source สำเร็จ')
                    else:
                        print('  ✅ camera_source column มีอยู่แล้ว')
                    
                    if 'ai_model' not in column_names:
                        print('  ⚙️ เพิ่ม ai_model column...')
                        cursor.execute('ALTER TABLE detections ADD COLUMN ai_model TEXT DEFAULT "yolo"')
                        print('  ✅ เพิ่ม ai_model สำเร็จ')
                    else:
                        print('  ✅ ai_model column มีอยู่แล้ว')
                        
                    conn.commit()
                else:
                    print(f'  ❌ Table detections ไม่มีใน {db_file}')
                
                conn.close()
                
            except Exception as e:
                print(f'  ❌ Error กับ {db_file}: {e}')
        else:
            print(f'❌ ไฟล์ {db_file} ไม่มี')

    print('\n✅ Database schema update เสร็จสิ้น!')

if __name__ == "__main__":
    fix_database_schema()
