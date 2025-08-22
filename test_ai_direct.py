#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct AI Agent Test - Test Enhanced Ultra Smart AI Agent directly
"""
import sys
import os

# Add the swallow_ai directory to the path
sys.path.append(r'C:\Nakhonnok\swallow_ai')

from enhanced_ultra_smart_ai_agent import EnhancedUltraSmartAIAgent

def test_ai_agent_direct():
    """ทดสอบ AI Agent โดยตรง"""
    print("🧪 เริ่มทดสอบ Enhanced Ultra Smart AI Agent โดยตรง...")
    print("=" * 60)
    
    try:
        # Initialize AI Agent
        ai_agent = EnhancedUltraSmartAIAgent()
        print("✅ AI Agent โหลดสำเร็จ!")
        
        # Test questions
        test_questions = [
            "สวัสดี คุณเป็น AI Agent ที่ฉลาดจริงๆหรือ?",
            "บอกข้อมูลเกี่ยวกับนกนางแอ่น",
            "แอพนี้ทำอะไรได้บ้าง?",
            "คุณสามารถเรียนรู้ได้ไหม?",
            "ช่วยแสดงสถิติของระบบ"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. คำถาม: {question}")
            
            try:
                response = ai_agent.get_response(question)
                print(f"💬 คำตอบ: {response}")
                
            except Exception as e:
                print(f"❌ ข้อผิดพลาด: {e}")
                
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการเริ่ม AI Agent: {e}")
    
    print("\n✅ ทดสอบเสร็จสิ้น!")

if __name__ == "__main__":
    test_ai_agent_direct()
