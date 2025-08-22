#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 ระบบทดสอบ AI แชทบอตแบบครบถ้วน
"""

import requests
import json
import time
import sys
import os
from urllib.parse import urljoin

class ComprehensiveAIChatbotTester:
    """ระบบทดสอบ AI แชทบอตแบบครบถ้วน"""
    
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.test_results = {
            'direct_agent': False,
            'flask_server': False,
            'api_chat': False,
            'web_interface': False,
            'system_integration': False,
            'learning_system': False,
            'knowledge_base': False,
            'error_handling': False
        }
        
    def test_direct_agent(self):
        """ทดสอบ AI Agent โดยตรง"""
        print("🔧 ทดสอบ Ultra Smart AI Agent โดยตรง...")
        try:
            sys.path.append('.')
            from ultra_smart_ai_agent import UltraSmartAIAgent
            
            agent = UltraSmartAIAgent()
            
            # ทดสอบการตอบสนอง
            test_messages = [
                "สวัสดี",
                "นกในรังมีกี่ตัว",
                "อธิบายระบบ",
                "เกี่ยวกับนกแอ่น",
                "ขอบคุณ"
            ]
            
            all_passed = True
            for msg in test_messages:
                response = agent.get_response(msg)
                if not response or len(response) < 10:
                    all_passed = False
                    print(f"   ❌ ตอบสั้นเกินไป: {msg}")
                else:
                    print(f"   ✅ {msg}: {response[:50]}...")
                    
            self.test_results['direct_agent'] = all_passed
            print(f"   {'✅ สำเร็จ' if all_passed else '❌ ล้มเหลว'}")
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.test_results['direct_agent'] = False
            return False
    
    def test_flask_server(self):
        """ทดสอบ Flask Server"""
        print("🌐 ทดสอบ Flask Server...")
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                print("   ✅ Flask Server รันได้")
                self.test_results['flask_server'] = True
                return True
            else:
                print(f"   ❌ HTTP {response.status_code}")
                self.test_results['flask_server'] = False
                return False
        except Exception as e:
            print(f"   ❌ ไม่สามารถเชื่อมต่อ: {e}")
            self.test_results['flask_server'] = False
            return False
    
    def test_api_chat(self):
        """ทดสอบ API Chat"""
        print("💬 ทดสอบ API Chat...")
        try:
            test_cases = [
                {"message": "สวัสดี", "expected_keywords": ["สวัสดี", "AI", "Agent"]},
                {"message": "นกมีกี่ตัว", "expected_keywords": ["นก", "ตัว"]},
                {"message": "ระบบทำงานยังไง", "expected_keywords": ["ระบบ", "ทำงาน"]},
            ]
            
            all_passed = True
            for case in test_cases:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json={"message": case["message"]},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        ai_response = data.get('response', '')
                        print(f"   ✅ {case['message']}: {ai_response[:50]}...")
                    else:
                        print(f"   ⚠️ API ตอบแต่ไม่สำเร็จ: {case['message']}")
                        all_passed = False
                else:
                    print(f"   ❌ HTTP {response.status_code}: {case['message']}")
                    all_passed = False
                    
            self.test_results['api_chat'] = all_passed
            print(f"   {'✅ สำเร็จ' if all_passed else '❌ บางส่วนล้มเหลว'}")
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.test_results['api_chat'] = False
            return False
    
    def test_web_interface(self):
        """ทดสอบ Web Interface"""
        print("🖥️ ทดสอบ Web Interface...")
        try:
            # ทดสอบหน้า AI Chat
            response = requests.get(f"{self.base_url}/ai-chat", timeout=5)
            if response.status_code == 200 and 'html' in response.headers.get('content-type', ''):
                print("   ✅ หน้า AI Chat โหลดได้")
                
                # ตรวจสอบ content
                html_content = response.text
                required_elements = ['Ultra Smart AI Agent', 'chat-container', 'chat-input']
                missing_elements = [elem for elem in required_elements if elem not in html_content]
                
                if not missing_elements:
                    print("   ✅ HTML elements ครบถ้วน")
                    self.test_results['web_interface'] = True
                    return True
                else:
                    print(f"   ⚠️ ขาด elements: {missing_elements}")
                    self.test_results['web_interface'] = False
                    return False
            else:
                print(f"   ❌ HTTP {response.status_code} หรือไม่ใช่ HTML")
                self.test_results['web_interface'] = False
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.test_results['web_interface'] = False
            return False
    
    def test_system_integration(self):
        """ทดสอบการเชื่อมต่อระบบ"""
        print("🔗 ทดสอบการเชื่อมต่อระบบ...")
        try:
            # ทดสอบ APIs ต่างๆ
            endpoints = [
                '/api/statistics',
                '/api/system-health',
                '/api/object-detection/status'
            ]
            
            working_endpoints = 0
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                    if response.status_code == 200:
                        working_endpoints += 1
                        print(f"   ✅ {endpoint}")
                    else:
                        print(f"   ⚠️ {endpoint}: HTTP {response.status_code}")
                except:
                    print(f"   ❌ {endpoint}: ไม่ตอบสนอง")
                    
            success_rate = working_endpoints / len(endpoints)
            if success_rate >= 0.7:  # 70% success rate
                print(f"   ✅ ระบบเชื่อมต่อได้ {working_endpoints}/{len(endpoints)} endpoints")
                self.test_results['system_integration'] = True
                return True
            else:
                print(f"   ⚠️ ระบบเชื่อมต่อได้เพียง {working_endpoints}/{len(endpoints)} endpoints")
                self.test_results['system_integration'] = False
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.test_results['system_integration'] = False
            return False
    
    def test_learning_system(self):
        """ทดสอบระบบเรียนรู้"""
        print("🧠 ทดสอบระบบเรียนรู้...")
        try:
            # ตรวจสอบไฟล์ฐานข้อมูลการเรียนรู้
            db_files = ['ai_agent_memory.db', 'swallow_smart_stats.db']
            found_dbs = []
            
            for db_file in db_files:
                if os.path.exists(db_file):
                    found_dbs.append(db_file)
                    print(f"   ✅ {db_file} พบแล้ว")
                else:
                    print(f"   ⚠️ {db_file} ไม่พบ")
                    
            if found_dbs:
                print(f"   ✅ ระบบเรียนรู้พร้อมใช้งาน ({len(found_dbs)} databases)")
                self.test_results['learning_system'] = True
                return True
            else:
                print("   ❌ ไม่พบฐานข้อมูลการเรียนรู้")
                self.test_results['learning_system'] = False
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.test_results['learning_system'] = False
            return False
    
    def test_knowledge_base(self):
        """ทดสอบฐานความรู้"""
        print("📚 ทดสอบฐานความรู้...")
        try:
            # ทดสอบความรู้เฉพาะ
            knowledge_tests = [
                {"message": "นกแอ่นมีประโยชน์อย่างไร", "keywords": ["ประโยชน์", "นก"]},
                {"message": "พฤติกรรมนกแอ่น", "keywords": ["พฤติกรรม", "นก"]},
                {"message": "ระบบตรวจจับทำงานยังไง", "keywords": ["ระบบ", "ตรวจจับ"]}
            ]
            
            correct_answers = 0
            for test in knowledge_tests:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json={"message": test["message"]},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        ai_response = data.get('response', '').lower()
                        if any(keyword.lower() in ai_response for keyword in test["keywords"]):
                            correct_answers += 1
                            print(f"   ✅ {test['message']}")
                        else:
                            print(f"   ⚠️ {test['message']}: ตอบไม่ตรงประเด็น")
                    else:
                        print(f"   ❌ {test['message']}: API ล้มเหลว")
                else:
                    print(f"   ❌ {test['message']}: HTTP Error")
                    
            success_rate = correct_answers / len(knowledge_tests)
            if success_rate >= 0.7:
                print(f"   ✅ ฐานความรู้ทำงานได้ดี ({correct_answers}/{len(knowledge_tests)})")
                self.test_results['knowledge_base'] = True
                return True
            else:
                print(f"   ⚠️ ฐานความรู้ต้องปรับปรุง ({correct_answers}/{len(knowledge_tests)})")
                self.test_results['knowledge_base'] = False
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.test_results['knowledge_base'] = False
            return False
    
    def test_error_handling(self):
        """ทดสอบการจัดการข้อผิดพลาด"""
        print("⚠️ ทดสอบการจัดการข้อผิดพลาด...")
        try:
            error_tests = [
                {"message": "", "description": "ข้อความว่าง"},
                {"message": "ทดสอบข้อความที่ยาวมากๆ " * 100, "description": "ข้อความยาวเกินไป"}
            ]
            
            handled_correctly = 0
            for test in error_tests:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json={"message": test["message"]},
                    timeout=5
                )
                
                if response.status_code in [200, 400]:  # ควรตอบสนองหรือ error อย่างเหมาะสม
                    data = response.json()
                    if 'error' in data or data.get('success') == False:
                        print(f"   ✅ {test['description']}: จัดการ error ได้")
                        handled_correctly += 1
                    elif data.get('success') and data.get('response'):
                        print(f"   ✅ {test['description']}: ตอบสนองได้")
                        handled_correctly += 1
                    else:
                        print(f"   ⚠️ {test['description']}: การตอบสนองไม่ชัดเจน")
                else:
                    print(f"   ❌ {test['description']}: HTTP {response.status_code}")
                    
            success_rate = handled_correctly / len(error_tests)
            if success_rate >= 0.8:
                print(f"   ✅ การจัดการ error ดี ({handled_correctly}/{len(error_tests)})")
                self.test_results['error_handling'] = True
                return True
            else:
                print(f"   ⚠️ การจัดการ error ต้องปรับปรุง ({handled_correctly}/{len(error_tests)})")
                self.test_results['error_handling'] = False
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.test_results['error_handling'] = False
            return False
    
    def run_comprehensive_test(self):
        """รันการทดสอบแบบครบถ้วน"""
        print("🧪 เริ่มการทดสอบ AI แชทบอตแบบครบถ้วน")
        print("=" * 60)
        
        # เรียงลำดับการทดสอบ
        tests = [
            ("ทดสอบ AI Agent โดยตรง", self.test_direct_agent),
            ("ทดสอบ Flask Server", self.test_flask_server),
            ("ทดสอบ API Chat", self.test_api_chat),
            ("ทดสอบ Web Interface", self.test_web_interface),
            ("ทดสอบการเชื่อมต่อระบบ", self.test_system_integration),
            ("ทดสอบระบบเรียนรู้", self.test_learning_system),
            ("ทดสอบฐานความรู้", self.test_knowledge_base),
            ("ทดสอบการจัดการข้อผิดพลาด", self.test_error_handling),
        ]
        
        passed_tests = 0
        for test_name, test_func in tests:
            print(f"\n{test_name}...")
            if test_func():
                passed_tests += 1
            time.sleep(1)  # หน่วงเวลาระหว่างการทดสอบ
            
        # สรุปผล
        print("\n" + "=" * 60)
        print("📊 สรุปผลการทดสอบ")
        print("=" * 60)
        
        total_tests = len(tests)
        success_rate = (passed_tests / total_tests) * 100
        
        for key, status in self.test_results.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {key.replace('_', ' ').title()}")
            
        print(f"\n🎯 ผลรวม: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🏆 ระบบพร้อมใช้งานเต็มประสิทธิภาพ!")
        elif success_rate >= 70:
            print("✅ ระบบทำงานได้ดี มีข้อปรับปรุงเล็กน้อย")
        elif success_rate >= 50:
            print("⚠️ ระบบทำงานได้ แต่ต้องปรับปรุงหลายจุด")
        else:
            print("❌ ระบบมีปัญหาที่ต้องแก้ไขอย่างเร่งด่วน")
            
        return success_rate

if __name__ == "__main__":
    tester = ComprehensiveAIChatbotTester()
    tester.run_comprehensive_test()
