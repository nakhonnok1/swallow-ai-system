#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Ultra Smart AI Agent Tester
ระบบทดสอบ AI Agent อัจฉริยะ
"""

import sys
import os
import time

# เพิ่ม path สำหรับ import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra_smart_ai_agent import UltraSmartAIAgent

def test_ai_agent():
    """ทดสอบ AI Agent แบบครบถ้วน"""
    print("🚀 เริ่มทดสอบ Ultra Smart AI Agent...")
    print("=" * 60)
    
    # สร้าง AI Agent
    agent = UltraSmartAIAgent()
    
    # รายการคำถามทดสอบ
    test_cases = [
        # การทักทาย
        {
            'category': 'การทักทาย',
            'questions': [
                'สวัสดี',
                'หวัดดี',
                'hello'
            ]
        },
        # เกี่ยวกับนก
        {
            'category': 'ข้อมูลนก',
            'questions': [
                'นกเข้ากี่ตัว',
                'นกออกกี่ตัว',
                'นกตอนนี้กี่ตัว',
                'สถิตินก'
            ]
        },
        # เกี่ยวกับสิ่งแปลกปลอม
        {
            'category': 'สิ่งแปลกปลอม',
            'questions': [
                'มีสิ่งแปลกปลอมไหม',
                'พบคนไหม',
                'มีคนบุกรุกไหม',
                'สิ่งแปลกปลอมวันนี้'
            ]
        },
        # เกี่ยวกับระบบ
        {
            'category': 'สถานะระบบ',
            'questions': [
                'สถานะระบบ',
                'ระบบทำงานไหม',
                'กล้องเชื่อมต่อไหม'
            ]
        },
        # เกี่ยวกับเวลา
        {
            'category': 'เวลา',
            'questions': [
                'เวลาเท่าไหร่',
                'กี่โมงแล้ว',
                'วันนี้วันที่เท่าไหร่'
            ]
        },
        # ความรู้เกี่ยวกับนกนางแอ่น
        {
            'category': 'ความรู้นกนางแอ่น',
            'questions': [
                'นกนางแอ่นคืออะไร',
                'นกนางแอ่นทำรังยังไง',
                'ประโยชน์ของนกนางแอ่น'
            ]
        },
        # การขอความช่วยเหลือ
        {
            'category': 'ความช่วยเหลือ',
            'questions': [
                'ช่วย',
                'help',
                'มีคำสั่งอะไรบ้าง'
            ]
        },
        # คำถามไม่รู้จัก
        {
            'category': 'คำถามไม่รู้จัก',
            'questions': [
                'อากาศวันนี้เป็นยังไง',
                'ข้าวเท่าไหร่',
                'asdfghjkl'
            ]
        }
    ]
    
    # ทดสอบแต่ละหมวด
    for test_category in test_cases:
        print(f"\n🧪 ทดสอบ: {test_category['category']}")
        print("-" * 40)
        
        for question in test_category['questions']:
            print(f"\n👤 คำถาม: {question}")
            
            # วัดเวลาในการตอบ
            start_time = time.time()
            response = agent.get_response(question)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            print(f"🤖 คำตอบ: {response}")
            print(f"⏱️ เวลาตอบ: {response_time:.2f} วินาที")
            
            # หน่วงเวลาเล็กน้อย
            time.sleep(0.5)
    
    # สรุปผลการทดสอบ
    print("\n" + "=" * 60)
    print("📊 สรุปผลการทดสอบ:")
    print(f"🗣️ จำนวนการสนทนา: {agent.conversation_count}")
    print(f"🧠 Learned patterns: {len(agent.learned_patterns)}")
    print(f"⏱️ เวลาทำงาน: {time.time() - agent.session_start.timestamp():.2f} วินาที")
    print("✅ การทดสอบเสร็จสิ้น!")

def interactive_test():
    """ทดสอบแบบโต้ตอบ"""
    print("🎯 โหมดทดสอบแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออก")
    print("-" * 40)
    
    agent = UltraSmartAIAgent()
    
    while True:
        try:
            user_input = input("\n👤 คุณ: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'ออก', 'เลิก']:
                print("👋 ขอบคุณที่ใช้บริการ!")
                break
            
            if not user_input:
                continue
                
            # วัดเวลาตอบ
            start_time = time.time()
            response = agent.get_response(user_input)
            end_time = time.time()
            
            print(f"🤖 AI: {response}")
            print(f"⏱️ เวลาตอบ: {end_time - start_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n👋 ขอบคุณที่ใช้บริการ!")
            break
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    print("🚀 Ultra Smart AI Agent Tester")
    print("เลือกโหมดการทดสอบ:")
    print("1. ทดสอบอัตโนมัติ (Auto Test)")
    print("2. ทดสอบแบบโต้ตอบ (Interactive Test)")
    
    try:
        choice = input("\nเลือก (1 หรือ 2): ").strip()
        
        if choice == "1":
            test_ai_agent()
        elif choice == "2":
            interactive_test()
        else:
            print("ตัวเลือกไม่ถูกต้อง กำลังรันทดสอบอัตโนมัติ...")
            test_ai_agent()
            
    except KeyboardInterrupt:
        print("\n👋 ขอบคุณที่ใช้บริการ!")
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาด: {e}")
