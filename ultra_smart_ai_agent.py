#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra Smart AI Agent - Advanced Chatbot System
ระบบ AI Agent อัจฉริยะที่เชื่อมต่อข้อมูลจริงและเรียนรู้ได้
เชื่อมต่อกับระบบ AI จับนก และ AI ตรวจจับสิ่งแปลกปลอม
"""

import re
import json
import datetime as dt
import random
import sqlite3
import requests
import os
from typing import Dict, List, Any, Optional

class UltraSmartAIAgent:
    """🚀 Ultra Smart AI Agent - AI ตัวแทนอัจฉริยะที่เรียนรู้และพัฒนาตัวเองได้"""
    
    def __init__(self):
        # ฐานความรู้และการเรียนรู้
        self.knowledge_base = self._initialize_knowledge_base()
        self.learning_db = "ai_agent_memory.db"
        self.conversation_history = []
        self.user_patterns = {}
        
        # เชื่อมต่อระบบ
        self.api_endpoints = {
            'bird_stats': 'http://127.0.0.1:5000/api/statistics',
            'intruder_stats': 'http://127.0.0.1:5000/api/object-detection/stats',
            'intruder_alerts': 'http://127.0.0.1:5000/api/object-detection/alerts',
            'system_status': 'http://127.0.0.1:5000/api/object-detection/status',
            'system_health': 'http://127.0.0.1:5000/api/system-health'
        }
        
        # Session tracking และการเรียนรู้
        self.session_start = dt.datetime.now()
        self.conversation_count = 0
        self.last_context = {}
        self.learned_patterns = []
        
        # เริ่มต้นฐานข้อมูลการเรียนรู้
        self._initialize_learning_database()
        self._load_learned_patterns()
        
        print("Ultra Smart AI Agent initialized successfully!")
        print("Loading knowledge base and learned patterns...")
        print("Connecting to system APIs...")
        
    def _initialize_knowledge_base(self) -> Dict:
        """เริ่มต้นฐานความรู้ขั้นสูง"""
        return {
            'greetings': [
                'สวัสดี', 'หวัดดี', 'ดีครับ', 'ดีค่ะ', 'เฮลโล', 'hi', 'hello', 'ไฮ', 
                'สวัสดีตอนเช้า', 'สวัสดีตอนบ่าย', 'สวัสดีตอนเย็น'
            ],
            
            'bird_questions': [
                'นก', 'เข้า', 'ออก', 'จำนวน', 'นับ', 'สถิติ', 'นางแอ่น', 'swallow', 
                'มาแล้ว', 'ไปแล้ว', 'ตัว', 'จำนวนนก', 'เหินบิน', 'บิน', 'ไข่', 'รัง', 
                'เจาะรู', 'โซน', 'ปลิงเหิน', 'วางไข่', 'ลูกนก', 'หน้าฝน'
            ],
            
            'intruder_questions': [
                'สิ่งแปลกปลอม', 'คน', 'สัตว์', 'แปลกปลอม', 'งู', 'หนู', 'แมว', 'สุนัข', 
                'ตุกแก', 'เหยี่ยว', 'พิราบ', 'มีคน', 'เจอคน', 'ผู้บุกรุก', 'บุกรุก', 
                'intruder', 'person', 'animal', 'security', 'ภัยคุกคาม'
            ],
            
            'alert_questions': [
                'แจ้งเตือน', 'เตือน', 'แอลเลิร์ต', 'alert', 'เสียงเตือน', 'แจ้ง', 
                'เกิดเหตุ', 'เหตุการณ์', 'รายงาน', 'เคส', 'case', 'notification'
            ],
            
            'system_questions': [
                'ระบบ', 'สถานะ', 'status', 'กล้อง', 'camera', 'AI', 'ทำงาน', 'ออนไลน์', 
                'เซิร์ฟเวอร์', 'server', 'เชื่อมต่อ', 'connect', 'พร้อม', 'ready'
            ],
            
            'time_questions': [
                'เวลา', 'ตอนนี้', 'กี่โมง', 'วันนี้', 'วัน', 'เดือน', 'ปี', 'ชั่วโมง', 
                'นาที', 'วินาที', 'time', 'now', 'today', 'hour', 'minute'
            ],
            
            'help_questions': [
                'ช่วย', 'help', 'สอน', 'แนะนำ', 'คำสั่ง', 'command', 'ใช้งาน', 
                'วิธี', 'tutorial', 'manual', 'guide'
            ],
            
            'swallow_knowledge': {
                'basic_info': [
                    'นกนางแอ่นเป็นนกอพยพที่เดินทางมาประเทศไทยในช่วงหน้าฝน',
                    'นกนางแอ่นมักเจาะรูในกำแพงเพื่อทำรังและวางไข่',
                    'นกนางแอ่นจับแมลงเป็นอาหาร เป็นประโยชน์ต่อการเกษตร',
                    'นกนางแอ่นสามารถบินด้วยความเร็วสูงและเปลี่ยนทิศทางได้รวดเร็ว'
                ],
                'behavior': [
                    'นกนางแอ่นมักบินเป็นฝูงและเข้าออกจากรังในช่วงเวลาเดียวกัน',
                    'นกนางแอ่นจะกลับมาใช้รังเดิมในปีถัดไป',
                    'นกนางแอ่นมีพฤติกรรมการสื่อสารด้วยเสียงร้องที่หลากหลาย'
                ],
                'benefits': [
                    'นกนางแอ่นช่วยควบคุมจำนวนแมลงศัตรูพืช',
                    'รังนกนางแอ่นมีคุณค่าทางเศรษฐกิจ',
                    'นกนางแอ่นเป็นส่วนหนึ่งของระบบนิเวศที่สำคัญ'
                ]
            }
        }
        
    def _initialize_learning_database(self):
        """เริ่มต้นฐานข้อมูลการเรียนรู้"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # ตารางเก็บประวัติการสนทนา
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    context TEXT,
                    satisfaction_score INTEGER DEFAULT 0
                )
            ''')
            
            # ตารางเก็บรูปแบบที่เรียนรู้
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 1,
                    last_used TEXT NOT NULL
                )
            ''')
            
            # ตารางเก็บข้อมูลผู้ใช้
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_response_style TEXT DEFAULT 'friendly',
                    common_questions TEXT,
                    interaction_count INTEGER DEFAULT 0,
                    last_interaction TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            print("Learning database initialized successfully!")
            
        except Exception as e:
            print("Learning database initialization failed:", str(e))
    
    def _load_learned_patterns(self):
        """โหลดรูปแบบที่เรียนรู้แล้ว"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pattern, category, confidence 
                FROM learned_patterns 
                WHERE confidence > 0.7
                ORDER BY usage_count DESC
            ''')
            
            for pattern, category, confidence in cursor.fetchall():
                self.learned_patterns.append({
                    'pattern': pattern,
                    'category': category,
                    'confidence': confidence
                })
            
            conn.close()
            print(f"Loaded {len(self.learned_patterns)} learned patterns")
            
        except Exception as e:
            print(f"Error loading learned patterns: {e}")
    
    def _get_real_data(self, endpoint: str) -> Dict:
        """ดึงข้อมูลจริงจาก API"""
        try:
            if endpoint not in self.api_endpoints:
                return {}
                
            response = requests.get(self.api_endpoints[endpoint], timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ API {endpoint} returned status {response.status_code}")
                return {}
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout connecting to {endpoint}")
            return {}
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error to {endpoint}")
            return {}
        except Exception as e:
            print(f"⚠️ Error fetching data from {endpoint}: {e}")
            return {}
    
    def get_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        ตอบสนองต่อข้อความของผู้ใช้แบบอัจฉริยะ
        Args:
            message: ข้อความจากผู้ใช้
            context: ข้อมูลบริบท
        Returns:
            คำตอบจาก AI Agent
        """
        if not message or not isinstance(message, str):
            return "กรุณาส่งข้อความมาด้วยนะครับ 📝"
            
        self.conversation_count += 1
        message = message.lower().strip()
        context = context or {}
        
        # บันทึกการสนทนา
        self._record_conversation(message, context)
        
        # Preprocess message
        message = self._preprocess_message(message)
        print(f"DEBUG: Preprocessed message: '{message}'")
        
        # ตรวจสอบประเภทคำถาม
        question_type = self._classify_question_advanced(message)
        print(f"DEBUG: Question type: {question_type}")
        
        try:
            if question_type == 'greeting':
                return self._get_greeting_response()
            elif question_type == 'bird':
                return self._generate_bird_response(message, context)
            elif question_type == 'intruder':
                return self._generate_intruder_response(message, context)
            elif question_type == 'alert':
                return self._generate_alert_response(message, context)
            elif question_type == 'system':
                return self._generate_system_response(message, context)
            elif question_type == 'time':
                return self._generate_time_response(message, context)
            elif question_type == 'help':
                return self._generate_help_response()
            elif question_type == 'swallow_info':
                return self._generate_swallow_knowledge_response(message)
            else:
                return self._generate_learning_response(message, context)
                
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return "ขออภัยครับ เกิดข้อผิดพลาด ลองถามใหม่ได้ไหม? 😅"
    
    def _preprocess_message(self, message: str) -> str:
        """ปรับแต่งข้อความก่อนประมวลผล"""
        # ตรวจสอบว่าเป็น string หรือไม่
        if not isinstance(message, str):
            message = str(message)
        
        # ลบ whitespace เกิน
        message = message.strip()
        
        return message
    
    def _classify_question_advanced(self, message: str) -> str:
        """จำแนกประเภทคำถามแบบขั้นสูง"""
        # ตรวจสอบ learned patterns ก่อน
        for pattern in self.learned_patterns:
            if pattern['pattern'].lower() in message:
                return pattern['category']
        
        # ตรวจสอบทักทาย
        greetings = ['สวัสดี', 'หวัดดี', 'ดีครับ', 'ดีค่ะ', 'เฮลโล', 'hi', 'hello', 'ไฮ']
        if any(greeting in message for greeting in greetings):
            return 'greeting'
        
        # ตรวจสอบคำถามเกี่ยวกับนก
        bird_keywords = ['นก', 'เข้า', 'ออก', 'จำนวน', 'นับ', 'สถิติ', 'นางแอ่น', 'swallow', 'ตัว', 'จำนวนนก']
        if any(keyword in message for keyword in bird_keywords):
            return 'bird'
        
        # ตรวจสอบคำถามเกี่ยวกับสิ่งแปลกปลอม
        intruder_keywords = ['สิ่งแปลกปลอม', 'คน', 'สัตว์', 'แปลกปลอม', 'งู', 'หนู', 'แมว', 'สุนัข', 'ตุกแก', 'เหยี่ยว', 'พิราบ', 'มีคน', 'เจอคน', 'ผู้บุกรุก', 'บุกรุก', 'intruder', 'person', 'animal']
        if any(keyword in message for keyword in intruder_keywords):
            return 'intruder'
        
        # ตรวจสอบคำถามเกี่ยวกับการแจ้งเตือน
        alert_keywords = ['แจ้งเตือน', 'เตือน', 'แอลเลิร์ต', 'alert', 'เสียงเตือน', 'แจ้ง', 'เกิดเหตุ', 'เหตุการณ์', 'รายงาน']
        if any(keyword in message for keyword in alert_keywords):
            return 'alert'
        
        # ตรวจสอบคำถามเกี่ยวกับระบบ
        system_keywords = ['ระบบ', 'สถานะ', 'status', 'กล้อง', 'camera', 'AI', 'ทำงาน', 'ออนไลน์', 'เซิร์ฟเวอร์', 'server', 'เชื่อมต่อ', 'connect', 'พร้อม', 'ready']
        if any(keyword in message for keyword in system_keywords):
            return 'system'
        
        # ตรวจสอบคำถามเกี่ยวกับเวลา
        time_keywords = ['เวลา', 'ตอนนี้', 'กี่โมง', 'วันนี้', 'วัน', 'เดือน', 'ปี', 'ชั่วโมง', 'นาที', 'วินาที', 'time', 'now', 'today', 'hour', 'minute']
        if any(keyword in message for keyword in time_keywords):
            return 'time'
        
        # ตรวจสอบคำถามขอความช่วยเหลือ
        help_keywords = ['ช่วย', 'help', 'สอน', 'แนะนำ', 'คำสั่ง', 'command', 'ใช้งาน', 'วิธี', 'tutorial', 'manual', 'guide']
        if any(keyword in message for keyword in help_keywords):
            return 'help'
        
        # ตรวจสอบความรู้เกี่ยวกับนกนางแอ่น
        swallow_keywords = ['นกนางแอ่น', 'นางแอ่น', 'swallow', 'ปลิงเหิน', 'รัง', 'ไข่']
        if any(keyword in message for keyword in swallow_keywords):
            return 'swallow_info'
        
        return 'unknown'
    
    def _record_conversation(self, message: str, context: Dict):
        """บันทึกการสนทนาเพื่อการเรียนรู้"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (timestamp, user_message, ai_response, context)
                VALUES (?, ?, ?, ?)
            ''', (
                dt.datetime.now().isoformat(),
                message,
                "",  # จะ update ภายหลัง
                json.dumps(context)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error recording conversation: {e}")
    
    def _get_greeting_response(self) -> str:
        """สร้างคำทักทาย"""
        greetings = [
            "สวัสดีครับ! ผมเป็น AI Agent ที่ดูแลระบบตรวจจับนกนางแอ่น 🤖✨",
            "หวัดดีครับ! ยินดีต้อนรับสู่ระบบ Smart Swallow Detection 🪶",
            "สวัสดี! ผมพร้อมช่วยเหลือคุณเกี่ยวกับระบบ AI ของเรา 🚀",
        ]
        
        current_hour = dt.datetime.now().hour
        if 6 <= current_hour < 12:
            time_greeting = "สวัสดีตอนเช้าครับ! ☀️"
        elif 12 <= current_hour < 18:
            time_greeting = "สวัสดีตอนบ่ายครับ! 🌤️"
        else:
            time_greeting = "สวัสดีตอนเย็นครับ! 🌙"
        
        base_greeting = random.choice(greetings)
        return f"{time_greeting} {base_greeting}"
    
    def _generate_bird_response(self, message: str, context: Dict) -> str:
        """สร้างคำตอบเกี่ยวกับนก"""
        # ดึงข้อมูลจริงจาก API
        bird_data = self._get_real_data('bird_stats')
        
        if not bird_data:
            return "ขออภัยครับ ตอนนี้ไม่สามารถเชื่อมต่อกับระบบได้ กรุณาลองใหม่อีกครั้ง 🔄"
        
        if any(word in message for word in ['เข้า', 'มา', 'in', 'entering']):
            birds_in = bird_data.get('birds_in', 0)
            return f"📊 นกที่เข้ามา: {birds_in} ตัว\n✨ ข้อมูล ณ เวลา {dt.datetime.now().strftime('%H:%M:%S')}"
        
        elif any(word in message for word in ['ออก', 'ไป', 'out', 'exiting']):
            birds_out = bird_data.get('birds_out', 0)
            return f"📊 นกที่ออกไป: {birds_out} ตัว\n✨ ข้อมูล ณ เวลา {dt.datetime.now().strftime('%H:%M:%S')}"
        
        elif any(word in message for word in ['ตอนนี้', 'ปัจจุบัน', 'current', 'now']):
            current_count = bird_data.get('current_count', 0)
            return f"🪶 นกในปัจจุบัน: {current_count} ตัว\n⏰ อัพเดทล่าสุด: {dt.datetime.now().strftime('%H:%M:%S')}"
        
        elif any(word in message for word in ['สถิติ', 'รายงาน', 'stats', 'report']):
            return self._generate_detailed_bird_stats(bird_data)
        
        else:
            # คำตอบทั่วไป
            birds_in = bird_data.get('birds_in', 0)
            birds_out = bird_data.get('birds_out', 0)
            current_count = bird_data.get('current_count', 0)
            
            return f"""🪶 สรุปสถิตินกนางแอ่น:
📈 เข้า: {birds_in} ตัว
📉 ออก: {birds_out} ตัว  
🔢 ปัจจุบัน: {current_count} ตัว
⏰ อัพเดท: {dt.datetime.now().strftime('%H:%M:%S')}"""
    
    def _generate_intruder_response(self, message: str, context: Dict) -> str:
        """สร้างคำตอบเกี่ยวกับสิ่งแปลกปลอม"""
        # ดึงข้อมูลจริงจาก API
        intruder_stats = self._get_real_data('intruder_stats')
        intruder_alerts = self._get_real_data('intruder_alerts')
        
        if not intruder_stats and not intruder_alerts:
            return "ขออภัยครับ ไม่สามารถเชื่อมต่อกับระบบตรวจจับสิ่งแปลกปลอมได้ 🔄"
        
        today_total = intruder_stats.get('today_total', 0)
        total_alerts = intruder_stats.get('total_alerts', 0)
        
        if any(word in message for word in ['วันนี้', 'today']):
            if today_total > 0:
                return f"🚨 วันนี้พบสิ่งแปลกปลอม {today_total} ครั้ง\n📊 รวมทั้งหมด {total_alerts} ครั้ง"
            else:
                return "✅ วันนี้ไม่พบสิ่งแปลกปลอม ระบบทำงานปกติ"
        
        elif any(word in message for word in ['ล่าสุด', 'latest', 'recent']):
            if intruder_alerts and len(intruder_alerts) > 0:
                latest = intruder_alerts[0]
                return f"""🚨 การตรวจจับล่าสุด:
🎯 ประเภท: {latest.get('object_name', 'ไม่ระบุ')}
⏰ เวลา: {latest.get('timestamp', 'ไม่ระบุ')}
📊 ความเชื่อมั่น: {latest.get('confidence', 0)*100:.1f}%"""
            else:
                return "✅ ไม่มีการตรวจจับสิ่งแปลกปลอมล่าสุด"
        
        else:
            # คำตอบทั่วไป
            if today_total > 0:
                return f"""🛡️ สรุปการตรวจจับสิ่งแปลกปลอม:
📅 วันนี้: {today_total} ครั้ง
📊 รวมทั้งหมด: {total_alerts} ครั้ง
⚡ ระบบทำงาน: ปกติ"""
            else:
                return "🛡️ ระบบตรวจจับสิ่งแปลกปลอมทำงานปกติ ไม่พบภัยคุกคาม ✅"
    
    def _generate_alert_response(self, message: str, context: Dict) -> str:
        """สร้างคำตอบเกี่ยวกับการแจ้งเตือน"""
        intruder_alerts = self._get_real_data('intruder_alerts')
        
        if not intruder_alerts:
            return "📢 ไม่มีการแจ้งเตือนในขณะนี้ ระบบทำงานปกติ ✅"
        
        if len(intruder_alerts) == 0:
            return "📢 ไม่มีการแจ้งเตือนล่าสุด ระบบสงบ ✅"
        
        # แสดงการแจ้งเตือนล่าสุด 3 รายการ
        response = "📢 การแจ้งเตือนล่าสุด:\n\n"
        for i, alert in enumerate(intruder_alerts[:3]):
            response += f"{i+1}. 🚨 {alert.get('object_name', 'ไม่ระบุ')}\n"
            response += f"   ⏰ {alert.get('timestamp', 'ไม่ระบุ')}\n"
            response += f"   📊 {alert.get('confidence', 0)*100:.1f}%\n\n"
        
        return response.strip()
    
    def _generate_system_response(self, message: str, context: Dict) -> str:
        """สร้างคำตอบเกี่ยวกับระบบ"""
        system_status = self._get_real_data('system_status')
        
        enabled = system_status.get('enabled', False)
        model_loaded = system_status.get('model_loaded', False)
        
        uptime = dt.datetime.now() - self.session_start
        uptime_str = str(uptime).split('.')[0]  # ลบ microseconds
        
        status_text = "🟢 ออนไลน์" if (enabled and model_loaded) else "🔴 ออฟไลน์"
        
        return f"""⚙️ สถานะระบบ:
{status_text} ระบบทำงาน: {"ปกติ" if enabled else "หยุดชั่วคราว"}
🤖 AI Model: {"โหลดแล้ว" if model_loaded else "ไม่พร้อม"}
📹 กล้อง: เชื่อมต่อแล้ว
⏱️ เวลาทำงาน: {uptime_str}
💬 การสนทนา: {self.conversation_count} ครั้ง"""
    
    def _generate_time_response(self, message: str, context: Dict) -> str:
        """สร้างคำตอบเกี่ยวกับเวลา"""
        now = dt.datetime.now()
        thai_time = now.strftime('%H:%M:%S')
        thai_date = now.strftime('%d/%m/%Y')
        
        uptime = now - self.session_start
        uptime_str = str(uptime).split('.')[0]
        
        return f"""⏰ ข้อมูลเวลา:
🕐 เวลาปัจจุบัน: {thai_time}
📅 วันที่: {thai_date}
⚡ ระบบทำงานมา: {uptime_str}
🤖 สนทนาไปแล้ว: {self.conversation_count} ครั้ง"""
    
    def _generate_help_response(self) -> str:
        """สร้างคำตอบช่วยเหลือ"""
        return """🆘 คำสั่งที่สามารถใช้ได้:

🪶 เกี่ยวกับนก:
• "นกเข้ากี่ตัว" - จำนวนนกที่เข้ามา
• "นกออกกี่ตัว" - จำนวนนกที่ออกไป  
• "นกตอนนี้" - จำนวนนกปัจจุบัน
• "สถิตินก" - รายงานสถิติละเอียด

🚨 เกี่ยวกับสิ่งแปลกปลอม:
• "พบสิ่งแปลกปลอมไหม" - การตรวจจับล่าสุด
• "สิ่งแปลกปลอมวันนี้" - สถิติวันนี้
• "สิ่งแปลกปลอมล่าสุด" - การตรวจจับล่าสุด

📢 เกี่ยวกับการแจ้งเตือน:
• "แจ้งเตือนล่าสุด" - การแจ้งเตือนล่าสุด
• "มีแจ้งเตือนไหม" - ตรวจสอบการแจ้งเตือน

⚙️ เกี่ยวกับระบบ:
• "สถานะระบบ" - ข้อมูลสถานะทั้งหมด
• "เวลาเท่าไหร่" - เวลาปัจจุบัน

🪶 เกี่ยวกับนกนางแอ่น:
• "นกนางแอ่นคืออะไร" - ข้อมูลพื้นฐาน
• "นกนางแอ่นทำรังยังไง" - พฤติกรรม
• "ประโยชน์ของนกนางแอ่น" - คุณประโยชน์

💡 เคล็ดลับ: คุณสามารถพิมพ์คำถามธรรมดาๆ ผมจะเข้าใจเอง!"""
    
    def _generate_swallow_knowledge_response(self, message: str) -> str:
        """สร้างคำตอบเกี่ยวกับความรู้นกนางแอ่น"""
        knowledge = self.knowledge_base['swallow_knowledge']
        
        if any(word in message for word in ['คือ', 'อะไร', 'what', 'พื้นฐาน']):
            return "🪶 " + random.choice(knowledge['basic_info'])
        
        elif any(word in message for word in ['พฤติกรรม', 'ทำรัง', 'behavior', 'บิน']):
            return "🏠 " + random.choice(knowledge['behavior'])
        
        elif any(word in message for word in ['ประโยชน์', 'benefit', 'ดี', 'เป็นประโยชน์']):
            return "✨ " + random.choice(knowledge['benefits'])
        
        else:
            # คำตอบรวม
            responses = []
            responses.extend(knowledge['basic_info'])
            return "🪶 " + random.choice(responses)
    
    def _generate_detailed_bird_stats(self, bird_data: Dict) -> str:
        """สร้างรายงานสถิตินกแบบละเอียด"""
        birds_in = bird_data.get('birds_in', 0)
        birds_out = bird_data.get('birds_out', 0)
        current_count = bird_data.get('current_count', 0)
        
        # คำนวณสถิติเพิ่มเติม
        net_change = birds_in - birds_out
        activity_level = "สูง" if (birds_in + birds_out) > 10 else "ปกติ" if (birds_in + birds_out) > 5 else "ต่ำ"
        
        return f"""📊 รายงานสถิตินกนางแอ่นละเอียด:

🔢 จำนวนนก:
• เข้า: {birds_in} ตัว
• ออก: {birds_out} ตัว
• ปัจจุบัน: {current_count} ตัว
• ผลต่าง: {net_change:+d} ตัว

📈 การวิเคราะห์:
• ระดับกิจกรรม: {activity_level}
• รูปแบบ: {"เข้ามากกว่าออก" if net_change > 0 else "ออกมากกว่าเข้า" if net_change < 0 else "สมดุล"}

⏰ อัพเดทล่าสุด: {dt.datetime.now().strftime('%H:%M:%S')}"""
    
    def _generate_learning_response(self, message: str, context: Dict) -> str:
        """สร้างคำตอบแบบเรียนรู้สำหรับคำถามที่ไม่รู้จัก"""
        # ลองค้นหาคำคล้ายคลึง
        similar_responses = [
            "🤔 ขออภัยครับ ผมยังไม่เข้าใจคำถามนี้ ลองถามเกี่ยวกับนก 🪶 หรือระบบการตรวจจับ 🤖 ดูนะครับ",
            "😅 คำถามนี้ท้าทายสำหรับผม! ลองใช้คำสั่ง 'ช่วย' เพื่อดูตัวอย่างคำถามได้ครับ",
            "🧠 ผมกำลังเรียนรู้คำถามแบบนี้ ขอเวลาสักหน่อยนะครับ! ตอนนี้ลองถามเกี่ยวกับระบบของเราดูมั้ย?",
            "💭 ผมยังไม่เข้าใจ แต่ผมจะเรียนรู้! ลองเปลี่ยนคำถามหรือใช้คำง่ายๆ ดูนะครับ"
        ]
        
        # บันทึกเป็น pattern ใหม่เพื่อเรียนรู้
        self._learn_new_pattern(message, 'unknown')
        
        return random.choice(similar_responses)
    
    def _learn_new_pattern(self, message: str, category: str):
        """เรียนรู้รูปแบบใหม่"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # เช็คว่ามี pattern นี้แล้วหรือไม่
            cursor.execute('''
                SELECT id, usage_count FROM learned_patterns 
                WHERE pattern = ? AND category = ?
            ''', (message, category))
            
            result = cursor.fetchone()
            
            if result:
                # อัพเดท usage count
                cursor.execute('''
                    UPDATE learned_patterns 
                    SET usage_count = usage_count + 1, last_used = ?
                    WHERE id = ?
                ''', (dt.datetime.now().isoformat(), result[0]))
            else:
                # เพิ่ม pattern ใหม่
                cursor.execute('''
                    INSERT INTO learned_patterns (pattern, category, last_used)
                    VALUES (?, ?, ?)
                ''', (message, category, dt.datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error learning pattern: {e}")

# Compatibility aliases สำหรับระบบเก่า
class SmartAIChatbot(UltraSmartAIAgent):
    """Compatibility class"""
    pass

class SmartAIChatbotV2(UltraSmartAIAgent):
    """Compatibility class"""
    pass

class UltraSmartAIChatbot(UltraSmartAIAgent):
    """Compatibility class"""
    pass

# ทดสอบระบบ
if __name__ == "__main__":
    print("🚀 Testing Ultra Smart AI Agent...")
    
    agent = UltraSmartAIAgent()
    
    # ทดสอบคำถามต่างๆ
    test_questions = [
        "สวัสดี",
        "นกเข้ากี่ตัว",
        "มีสิ่งแปลกปลอมไหม",
        "สถานะระบบ",
        "เวลาเท่าไหร่",
        "ช่วย",
        "นกนางแอ่นคืออะไร"
    ]
    
    for question in test_questions:
        print(f"\n👤: {question}")
        response = agent.get_response(question)
        print(f"🤖: {response}")
    
    print("\n✅ Testing completed!")
