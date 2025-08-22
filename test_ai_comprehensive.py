#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced AI Agent Test - ทดสอบ Enhanced Ultra Smart AI Agent แบบละเอียด
"""
import sys
import os

# Add the swallow_ai directory to the path
sys.path.append(r'C:\Nakhonnok\swallow_ai')

from enhanced_ultra_smart_ai_agent import EnhancedUltraSmartAIAgent

def test_ai_agent_comprehensive():
    """ทดสอบ AI Agent แบบครอบคลุม"""
    print("🧪 ทดสอบ Enhanced Ultra Smart AI Agent แบบครอบคลุม...")
    print("=" * 70)
    
    try:
        # Initialize AI Agent
        ai_agent = EnhancedUltraSmartAIAgent()
        print("✅ AI Agent โหลดสำเร็จ!")
        
        # Test categories with specific questions
        test_categories = {
            "🤝 การทักทาย": [
                "สวัสดี",
                "หวัดดี เป็นอย่างไรบ้าง",
                "Hello AI Agent"
            ],
            "🐦 คำถามเกี่ยวกับนก": [
                "นกเข้ากี่ตัว",
                "นกออกกี่ตัว", 
                "ตอนนี้นกในรังมีกี่ตัว",
                "สถิตินกวันนี้",
                "สรุปข้อมูลนก"
            ],
            "🚨 สิ่งแปลกปลอม": [
                "มีสิ่งแปลกปลอมไหม",
                "พบผู้บุกรุกไหม",
                "การแจ้งเตือนล่าสุด",
                "มีคนแปลกหน้าไหม"
            ],
            "⚙️ สถานะระบบ": [
                "สถานะระบบ",
                "กล้องทำงานไหม",
                "สุขภาพระบบ",
                "AI ทำงานปกติไหม",
                "ประสิทธิภาพระบบ"
            ],
            "🕐 เวลา": [
                "เวลาเท่าไหร่แล้ว",
                "วันนี้วันอะไร",
                "ตอนนี้กี่โมงแล้ว"
            ],
            "📚 ความรู้นกแอ่น": [
                "เกี่ยวกับนกนางแอ่น",
                "นกแอ่นมีประโยชน์อย่างไร",
                "แอพนี้ทำอะไรได้บ้าง",
                "ฟีเจอร์ของระบบ"
            ],
            "🆘 ขอความช่วยเหลือ": [
                "ช่วยเหลือ",
                "help",
                "สอนการใช้งาน",
                "คำสั่งที่ใช้ได้"
            ],
            "🤔 คำถามทั่วไป": [
                "คุณสามารถเรียนรู้ได้ไหม",
                "คุณฉลาดแค่ไหน",
                "ทำไมต้องใช้ AI",
                "อนาคตของ AI เป็นอย่างไร"
            ]
        }
        
        for category, questions in test_categories.items():
            print(f"\n{category}")
            print("-" * 60)
            
            for i, question in enumerate(questions, 1):
                print(f"\n{i}. คำถาม: {question}")
                
                try:
                    response = ai_agent.get_response(question)
                    print(f"💬 คำตอบ: {response}")
                    
                except Exception as e:
                    print(f"❌ ข้อผิดพลาด: {e}")
                    
                print("-" * 50)
        
        # Test learning capability
        print(f"\n🧠 **สรุปความสามารถในการเรียนรู้:**")
        print(f"📊 จำนวนการสนทนาทั้งหมด: {ai_agent.conversation_count}")
        print(f"📚 รูปแบบที่เรียนรู้: {len(ai_agent.learned_patterns)}")
        print(f"🧬 ฐานความรู้: {len(ai_agent.knowledge_base)} หมวดหมู่")
    
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการเริ่ม AI Agent: {e}")
    
    print("\n✅ การทดสอบครอบคลุมเสร็จสิ้น!")

if __name__ == "__main__":
    test_ai_agent_comprehensive()
