#!/usr/bin/env python3
"""
Add color_code column to object_detection_alerts.db
"""

import sqlite3

def add_color_code_column():
    """Add color_code column to detections table"""
    
    print('🔧 กำลังแก้ไข object_detection_alerts.db...')
    conn = sqlite3.connect('object_detection_alerts.db')
    cursor = conn.cursor()

    # Get current columns
    cursor.execute('PRAGMA table_info(detections)')
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    print(f'📋 Current columns: {", ".join(column_names)}')

    # Add missing color_code column
    if 'color_code' not in column_names:
        print('⚙️ เพิ่ม color_code column...')
        cursor.execute('ALTER TABLE detections ADD COLUMN color_code TEXT DEFAULT "blue"')
        print('✅ เพิ่ม color_code สำเร็จ')
    else:
        print('✅ color_code column มีอยู่แล้ว')

    conn.commit()
    conn.close()
    print('✅ database update สำเร็จ!')

if __name__ == "__main__":
    add_color_code_column()
