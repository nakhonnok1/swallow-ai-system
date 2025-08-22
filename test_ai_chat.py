#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Enhanced Ultra Smart AI Agent Chat API
"""
import requests
import json

def test_ai_chat():
    """ทดสอบ AI Agent ผ่าน API"""
    url = "http://127.0.0.1:5000/api/chat"
    
    # คำถามทดสอบต่างๆ
    test_questions = [
        "สวัสดี คุณเป็น AI Agent ที่ฉลาดจริงๆหรือ?",
        "บอกข้อมูลเกี่ยวกับนกนางแอ่น",
        "ตอนนี้มีนกกี่ตัวในระบบ?",
        "แอพนี้ทำอะไรได้บ้าง?",
        "คุณสามารถเรียนรู้ได้ไหม?"
    ]
    
    print("🧪 เริ่มทดสอบ Enhanced Ultra Smart AI Agent...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. คำถาม: {question}")
        
        try:
            # ส่งคำถามไป API
            response = requests.post(
                url,
                json={"message": question},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"💬 คำตอบ: {result.get('response', 'ไม่มีคำตอบ')}")
                print(f"🧠 การเรียนรู้: {result.get('learning_status', 'ไม่ทราบ')}")
                print(f"📊 สถิติ: {result.get('stats', {})}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ ข้อผิดพลาด: {e}")
            
        print("-" * 50)
    
    print("\n✅ ทดสอบเสร็จสิ้น!")

if __name__ == "__main__":
    test_ai_chat()
