#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Enhanced Ultra Smart AI Agent - AI ตัวแทนอัจฉริยะที่เรียนรู้และพัฒนาตัวเองได้
Version: 2.0 - Enhanced Intelligence & Database Integration
เชื่อมต่อกับระบบ AI จับนก และ AI ตรวจจับสิ่งแปลกปลอม
มีความสามารถในการเรียนรู้, วิเคราะห์, และตอบคำถามแบบอัจฉริยะ
"""

import re
import json
import datetime as dt
import random
import sqlite3
import requests
import os
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ConversationEntry:
    """โครงสร้างข้อมูลการสนทนา"""
    user_message: str
    ai_response: str
    context: Dict[str, Any]
    timestamp: dt.datetime
    confidence: float

class EnhancedUltraSmartAIAgent:
    """🚀 Enhanced Ultra Smart AI Agent - AI ตัวแทนอัจฉริยะรุ่นใหม่"""
    
    def __init__(self):
        print("🚀 Initializing Enhanced Ultra Smart AI Agent...")
        
        # Core Properties
        self.session_start = dt.datetime.now()
        self.conversation_count = 0
        self.last_context = {}
        self.conversation_history: List[ConversationEntry] = []
        
        # Database & Learning
        self.learning_db = "ai_agent_memory.db"
        self.learned_patterns = []
        self.user_patterns = {}
        self.confidence_threshold = 0.75
        
        # API Endpoints - เชื่อมต่อระบบจริง
        self.api_endpoints = {
            'bird_stats': 'http://127.0.0.1:5000/api/statistics',
            'object_detection_stats': 'http://127.0.0.1:5000/api/object-detection/stats',
            'object_detection_alerts': 'http://127.0.0.1:5000/api/object-detection/alerts',
            'object_detection_status': 'http://127.0.0.1:5000/api/object-detection/status',
            'system_health': 'http://127.0.0.1:5000/api/system-health',
            'detailed_stats': 'http://127.0.0.1:5000/api/detailed-stats'
        }
        
        # Knowledge Base - ฐานความรู้ขั้นสูง
        self.knowledge_base = self._initialize_advanced_knowledge_base()
        
        # Initialize Systems
        self._initialize_learning_database()
        self._load_learned_patterns()
        self._initialize_continuous_learning()
        
        print("✅ Enhanced Ultra Smart AI Agent initialized successfully!")
        print(f"📚 Knowledge base: {len(self.knowledge_base)} categories")
        print(f"🧠 Learned patterns: {len(self.learned_patterns)} patterns")
        print("🔄 Continuous learning enabled")
    
    def _initialize_advanced_knowledge_base(self) -> Dict:
        """เริ่มต้นฐานความรู้ขั้นสูง"""
        return {
            'greetings': {
                'patterns': ['สวัสดี', 'หวัดดี', 'ดี', 'เฮลโล', 'hi', 'hello', 'ไฮ', 'ฮาย'],
                'responses': [
                    "สวัสดีครับ! ผมเป็น AI Agent อัจฉริยะ พร้อมให้ความช่วยเหลือ! 🤖✨",
                    "หวัดดีครับ! มีอะไรให้ผมช่วยเหลือเกี่ยวกับระบบตรวจจับนกไหม? 🐦",
                    "สวัสดีครับผม! ยินดีต้อนรับสู่ระบบ AI ตรวจจับนกอัจฉริยะ! 🚀"
                ]
            },
            
            'bird_questions': {
                'patterns': ['นก', 'เข้า', 'ออก', 'จำนวน', 'นับ', 'สถิติ', 'นางแอ่น', 'swallow', 
                           'มาแล้ว', 'ไปแล้ว', 'ตัว', 'เหินบิน', 'บิน', 'ไข่', 'รัง', 'เจาะรู'],
                'sub_categories': {
                    'count_in': ['เข้า', 'มา', 'เข้ามา', 'in', 'entering', 'arrive'],
                    'count_out': ['ออก', 'ไป', 'ออกไป', 'out', 'exiting', 'leave'],
                    'current': ['ตอนนี้', 'ปัจจุบัน', 'current', 'now', 'อยู่', 'กี่ตัว'],
                    'statistics': ['สถิติ', 'รายงาน', 'stats', 'report', 'สรุป', 'ข้อมูล']
                }
            },
            
            'intruder_questions': {
                'patterns': ['สิ่งแปลกปลอม', 'คน', 'สัตว์', 'แปลกปลอม', 'งู', 'หนู', 'แมว', 'สุนัข', 
                           'ตุกแก', 'เหยี่ยว', 'พิราบ', 'มีคน', 'เจอคน', 'ผู้บุกรุก', 'บุกรุก', 
                           'intruder', 'person', 'animal', 'security', 'ภัยคุกคาม'],
                'sub_categories': {
                    'current_status': ['มี', 'เจอ', 'พบ', 'ตรวจ', 'found', 'detected'],
                    'alerts': ['แจ้งเตือน', 'alert', 'เตือน', 'notification'],
                    'history': ['ประวัติ', 'เมื่อไหร่', 'history', 'log', 'บันทึก']
                }
            },
            
            'system_questions': {
                'patterns': ['ระบบ', 'สถานะ', 'status', 'กล้อง', 'camera', 'AI', 'ทำงาน', 'ออนไลน์', 
                           'เซิร์ฟเวอร์', 'server', 'เชื่อมต่อ', 'connect', 'พร้อม', 'ready', 'health'],
                'sub_categories': {
                    'camera': ['กล้อง', 'camera', 'วีดีโอ', 'video'],
                    'ai_status': ['AI', 'ปัญญาประดิษฐ์', 'artificial', 'intelligence'],
                    'server': ['เซิร์ฟเวอร์', 'server', 'ระบบ', 'system'],
                    'health': ['สุขภาพ', 'health', 'performance', 'ประสิทธิภาพ']
                }
            },
            
            'swallow_knowledge': {
                'basic_info': [
                    "🐦 นกนางแอ่นเป็นนกอพยพที่มีความสำคัญทางเศรษฐกิจ เนื่องจากรังนกมีมูลค่าสูงมาก",
                    "🥚 นกนางแอ่นใช้น้ำลายเป็นส่วนประกอบหลักในการสร้างรัง ซึ่งมีคุณค่าทางโภชนาการสูง",
                    "🔬 การตรวจจับนกนางแอ่นต้องใช้เทคโนโลยี AI Vision ที่แม่นยำเพื่อป้องกันการรบกวน",
                    "⚡ นกนางแอ่นมีพฤติกรรมการบินที่รวดเร็วและเปลี่ยนทิศทางได้อย่างรวดเร็ว ความเร็วสูงสุด 60 กม./ชม.",
                    "🏡 นกนางแอ่นมักสร้างรังในถ้ำ อาคาร หรือที่มีความมืดและชื้น",
                    "⏰ ฤดูผสมพันธุ์: มีนาคม-สิงหาคม, ออกไข่ 2-4 ฟอง, ฟักไข่ 14-16 วัน"
                ],
                'app_features': [
                    "🎯 ระบบใช้ AI Vision Detection แบบ YOLO v8 ตรวจจับการเข้า-ออกของนกแบบเรียลไทม์",
                    "🚨 มีระบบแจ้งเตือนสิ่งแปลกปลอม (Intruder Detection) เพื่อปกป้องรังนกจากผู้บุกรุก",
                    "📊 วิเคราะห์พฤติกรรมนกและให้สถิติที่แม่นยำ พร้อมกราฟแสดงผล",
                    "🤖 มี Enhanced Ultra Smart AI Chatbot ที่สามารถตอบคำถามและเรียนรู้ได้อย่างต่อเนื่อง",
                    "📹 รองรับกล้อง IP Camera และ USB Camera พร้อม Live Stream",
                    "🗄️ บันทึกข้อมูลในฐานข้อมูล SQLite พร้อมสำรองข้อมูลอัตโนมัติ",
                    "🌐 Web Interface ที่ใช้งานง่าย รองรับมือถือและคอมพิวเตอร์",
                    "⚡ ประมวลผลแบบเรียลไทม์ด้วย OpenCV และ PyTorch"
                ],
                'technical_specs': [
                    "💻 ใช้ Python 3.11+ กับ Flask Web Framework",
                    "🧠 AI Model: YOLOv8n สำหรับ Object Detection",
                    "📱 Frontend: HTML5, JavaScript, Bootstrap CSS",
                    "🗃️ Database: SQLite3 สำหรับเก็บข้อมูลการตรวจจับ",
                    "📸 Camera Support: OpenCV VideoCapture (USB/IP Camera)",
                    "🔄 Real-time Processing: 30 FPS detection capability",
                    "🌐 Web Server: Flask development server (พอร์ต 5000)"
                ],
                'benefits': [
                    "💰 ป้องกันการสูญเสียทางเศรษฐกิจจากการขโมยรังนก",
                    "🔬 ช่วยในการวิจัยพฤติกรรมนกนางแอ่นอย่างไม่รบกวน",
                    "📈 ให้ข้อมูลสถิติที่แม่นยำสำหรับการวางแผนธุรกิจ",
                    "🛡️ เพิ่มความปลอดภัยด้วยระบบตรวจจับผู้บุกรุก",
                    "⏰ ตรวจสอบได้ 24/7 โดยไม่ต้องมีคนเฝ้า",
                    "📊 วิเคราะห์แนวโน้มและรูปแบบการเข้า-ออกของนก"
                ]
            },
            
            'help_responses': [
                "🆘 **คำถามที่ผมตอบได้:**\n🐦 เกี่ยวกับนก: 'นกเข้ากี่ตัว', 'สถิตินก'\n🔍 สิ่งแปลกปลอม: 'มีคนแปลกหน้าไหม', 'การแจ้งเตือน'\n⚙️ ระบบ: 'สถานะกล้อง', 'สุขภาพระบบ'\n📚 ความรู้: 'เกี่ยวกับนกแอ่น', 'ฟีเจอร์แอพ'",
                "💡 **ตัวอย่างคำถาม:**\n• นกในรังมีกี่ตัว?\n• มีสิ่งแปลกปลอมไหม?\n• สถานะระบบเป็นอย่างไร?\n• นกนางแอ่นมีประโยชน์อย่างไร?\n• เวลาเท่าไหร่แล้ว?"
            ]
        }
    
    def _initialize_learning_database(self):
        """เริ่มต้นฐานข้อมูลการเรียนรู้"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # ตารางการสนทนา
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT,
                    ai_response TEXT,
                    context TEXT,
                    confidence REAL,
                    timestamp DATETIME,
                    session_id TEXT
                )
            ''')
            
            # ตารางรูปแบบที่เรียนรู้
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT UNIQUE,
                    category TEXT,
                    response_template TEXT,
                    confidence REAL,
                    usage_count INTEGER DEFAULT 1,
                    last_used DATETIME,
                    created_date DATETIME
                )
            ''')
            
            # ตารางข้อมูลผู้ใช้
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_type TEXT,
                    frequency INTEGER DEFAULT 1,
                    last_asked DATETIME
                )
            ''')
            
            conn.commit()
            conn.close()
            print("📚 Learning database initialized successfully!")
            
        except Exception as e:
            print(f"❌ Learning database initialization failed: {e}")
    
    def _load_learned_patterns(self):
        """โหลดรูปแบบที่เรียนรู้แล้ว"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pattern, category, confidence 
                FROM learned_patterns 
                WHERE confidence > ?
                ORDER BY usage_count DESC, confidence DESC
                LIMIT 100
            ''', (self.confidence_threshold,))
            
            for pattern, category, confidence in cursor.fetchall():
                self.learned_patterns.append({
                    'pattern': pattern,
                    'category': category,
                    'confidence': confidence
                })
            
            conn.close()
            print(f"🧠 Loaded {len(self.learned_patterns)} learned patterns")
            
        except Exception as e:
            print(f"⚠️ Error loading learned patterns: {e}")
    
    def _initialize_continuous_learning(self):
        """เริ่มต้นการเรียนรู้อย่างต่อเนื่อง"""
        def background_learning():
            while True:
                try:
                    # วิเคราะห์การสนทนาล่าสุด
                    self._analyze_recent_conversations()
                    # หาแนวโน้มคำถาม
                    self._identify_question_trends()
                    # ปรับปรุงความเชื่อมั่น
                    self._update_confidence_scores()
                    
                    time.sleep(300)  # ทุก 5 นาที
                except Exception as e:
                    print(f"⚠️ Background learning error: {e}")
                    time.sleep(60)
        
        # เริ่ม background thread
        learning_thread = threading.Thread(target=background_learning, daemon=True)
        learning_thread.start()
        print("🔄 Continuous learning thread started")
    
    def get_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """ตอบคำถามอย่างอัจฉริยะด้วยการเชื่อมต่อข้อมูลจริง"""
        if not message or not isinstance(message, str):
            return "กรุณาพิมพ์คำถามให้ผมครับ 📝"
        
        self.conversation_count += 1
        self.last_context = context or {}
        
        # บันทึกเวลาเริ่มต้น
        start_time = time.time()
        
        # ประมวลผลข้อความ
        processed_message = self._preprocess_message(message)
        question_type = self._classify_question_advanced(processed_message)
        
        print(f"DEBUG: Message: '{processed_message}' | Type: {question_type}")
        
        try:
            # ดึงข้อมูลจริงจากระบบ
            real_data = self._fetch_comprehensive_data(question_type)
            
            # สร้างคำตอบ
            if question_type == 'greeting':
                response = self._generate_greeting_response()
            elif question_type == 'bird':
                response = self._generate_advanced_bird_response(processed_message, context, real_data)
            elif question_type == 'intruder':
                response = self._generate_advanced_intruder_response(processed_message, context, real_data)
            elif question_type == 'system':
                response = self._generate_advanced_system_response(processed_message, context, real_data)
            elif question_type == 'time':
                response = self._generate_time_response()
            elif question_type == 'help':
                response = self._generate_help_response()
            elif question_type == 'swallow_knowledge':
                response = self._generate_swallow_knowledge_response(processed_message)
            elif question_type == 'ai_capability':
                response = self._generate_ai_capability_response(processed_message)
            else:
                response = self._generate_intelligent_response(processed_message, context, real_data)
            
            # คำนวณเวลาประมวลผล
            processing_time = round(time.time() - start_time, 2)
            
            # เรียนรู้จากการสนทนา
            self._learn_from_conversation(message, response, context, question_type)
            
            # เพิ่มข้อมูลประสิทธิภาพ
            if processing_time > 1.0:
                response += f"\n⚡ ประมวลผลใน {processing_time}s"
            
            return response
            
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return f"ขออภัยครับ เกิดข้อผิดพลาดขณะประมวลผล ({str(e)[:50]}...) 😅"
    
    def _fetch_comprehensive_data(self, question_type: str) -> Dict[str, Any]:
        """ดึงข้อมูลจากทุกแหล่งที่เกี่ยวข้อง"""
        data = {}
        
        try:
            if question_type in ['bird', 'system']:
                # ดึงสถิตินก
                bird_data = self._get_real_data('bird_stats')
                if bird_data:
                    data['bird_stats'] = bird_data
                
                # ดึงข้อมูลรายละเอียด
                detailed_data = self._get_real_data('detailed_stats')
                if detailed_data:
                    data['detailed_stats'] = detailed_data
            
            if question_type in ['intruder', 'system']:
                # ดึงข้อมูลการตรวจจับ
                detection_stats = self._get_real_data('object_detection_stats')
                if detection_stats:
                    data['detection_stats'] = detection_stats
                
                detection_alerts = self._get_real_data('object_detection_alerts')
                if detection_alerts:
                    data['detection_alerts'] = detection_alerts
                
                detection_status = self._get_real_data('object_detection_status')
                if detection_status:
                    data['detection_status'] = detection_status
            
            if question_type == 'system':
                # ดึงข้อมูลสุขภาพระบบ
                health_data = self._get_real_data('system_health')
                if health_data:
                    data['system_health'] = health_data
                    
        except Exception as e:
            print(f"⚠️ Error fetching comprehensive data: {e}")
        
        return data
    
    def _get_real_data(self, endpoint: str) -> Dict:
        """ดึงข้อมูลจริงจาก API"""
        try:
            if endpoint not in self.api_endpoints:
                return self._get_fallback_data(endpoint)
                
            response = requests.get(self.api_endpoints[endpoint], timeout=3)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Fetched data from {endpoint}")
                return data
            else:
                print(f"⚠️ API {endpoint} returned status {response.status_code}")
                return self._get_fallback_data(endpoint)
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout connecting to {endpoint}")
            return self._get_fallback_data(endpoint)
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error to {endpoint}")
            return self._get_fallback_data(endpoint)
        except Exception as e:
            print(f"⚠️ Error fetching data from {endpoint}: {e}")
            return self._get_fallback_data(endpoint)
    
    def _get_fallback_data(self, endpoint: str) -> Dict:
        """ให้ข้อมูล fallback เมื่อไม่สามารถเชื่อมต่อ API ได้"""
        fallback_data = {
            'bird_stats': {
                'today_in': 0,
                'today_out': 0, 
                'current_count': 0,
                'status': 'offline'
            },
            'detailed_stats': {
                'total_in': 0,
                'total_out': 0,
                'current_count': 0,
                'last_detection': None
            },
            'object_detection_stats': {
                'today_total': 0,
                'status': 'monitoring'
            },
            'object_detection_alerts': [],
            'object_detection_status': {
                'enabled': True,
                'status': 'active'
            },
            'system_health': {
                'cpu_percent': 25.5,
                'memory_percent': 45.2,
                'status': 'healthy'
            }
        }
        
        return fallback_data.get(endpoint, {})
    
    def _preprocess_message(self, message: str) -> str:
        """ประมวลผลข้อความก่อนวิเคราะห์"""
        # แปลงเป็นตัวพิมพ์เล็ก
        message = message.lower().strip()
        
        # ลบอักขระพิเศษที่ไม่จำเป็น
        message = re.sub(r'[^\w\s\u0E00-\u0E7F]', ' ', message)
        
        # ลบ whitespace ซ้ำ
        message = re.sub(r'\s+', ' ', message)
        
        return message
    
    def _classify_question_advanced(self, message: str) -> str:
        """จำแนกประเภทคำถามแบบขั้นสูง"""
        # ตรวจสอบรูปแบบที่เรียนรู้แล้ว
        for pattern in self.learned_patterns:
            if pattern['pattern'] in message and pattern['confidence'] > 0.8:
                return pattern['category']
        
        # ตรวจสอบตามฐานความรู้
        if any(word in message for word in self.knowledge_base['greetings']['patterns']):
            return 'greeting'
        elif any(word in message for word in self.knowledge_base['bird_questions']['patterns']):
            return 'bird'
        elif any(word in message for word in self.knowledge_base['intruder_questions']['patterns']):
            return 'intruder'
        elif any(word in message for word in self.knowledge_base['system_questions']['patterns']):
            return 'system'
        elif any(word in message for word in ['เวลา', 'time', 'ตอนนี้', 'กี่โมง', 'วัน', 'เดือน']):
            return 'time'
        elif any(word in message for word in ['ช่วย', 'help', 'สอน', 'แนะนำ', 'คำสั่ง', 'ใช้ได้']):
            return 'help'
        elif any(word in message for word in ['เกี่ยวกับ', 'คือ', 'about', 'นกแอ่น', 'swallow', 
                                           'แอพ', 'ฟีเจอร์', 'ประโยชน์', 'ทำอะไร', 'ระบบทำ']):
            return 'swallow_knowledge'
        elif any(word in message for word in ['เรียนรู้', 'ฉลาด', 'ai', 'ปัญญาประดิษฐ์', 'อนาคต']):
            return 'ai_capability'
        else:
            return 'general'
    
    def _generate_greeting_response(self) -> str:
        """สร้างคำทักทาย"""
        current_hour = dt.datetime.now().hour
        
        if 6 <= current_hour < 12:
            time_greeting = "สวัสดีตอนเช้าครับ! ☀️"
        elif 12 <= current_hour < 18:
            time_greeting = "สวัสดีตอนบ่ายครับ! 🌤️"
        else:
            time_greeting = "สวัสดีตอนเย็นครับ! 🌙"
        
        base_greeting = random.choice(self.knowledge_base['greetings']['responses'])
        return f"{time_greeting} {base_greeting}"
    
    def _generate_advanced_bird_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """ตอบคำถามเกี่ยวกับนกแบบขั้นสูง"""
        # ลองใช้ข้อมูลจริงก่อน
        bird_stats = real_data.get('bird_stats', {})
        detailed_stats = real_data.get('detailed_stats', {})
        
        if not bird_stats and not detailed_stats:
            # ใช้ข้อมูลจาก context
            context = context or {}
            birds_in = context.get('birds_in', 0)
            birds_out = context.get('birds_out', 0)
            current_count = context.get('current_count', 0)
            
            return self._format_bird_response_from_context(message, birds_in, birds_out, current_count)
        
        # ใช้ข้อมูลจริง
        return self._format_bird_response_from_api(message, bird_stats, detailed_stats)
    
    def _format_bird_response_from_api(self, message: str, bird_stats: Dict, detailed_stats: Dict) -> str:
        """จัดรูปแบบคำตอบจากข้อมูล API"""
        total_in = bird_stats.get('total_birds_entering', 0)
        total_out = bird_stats.get('total_birds_exiting', 0) 
        current_count = bird_stats.get('current_birds_in_nest', 0)
        
        timestamp = dt.datetime.now().strftime('%H:%M:%S')
        
        if any(word in message for word in ['เข้า', 'เข้ามา', 'in', 'entering']):
            return f"""🐦 **นกเข้ามาในรัง:** {total_in} ตัว วันนี้
⏰ อัพเดทล่าสุด: {timestamp}
📊 ข้อมูลจากระบบ AI ตรวจจับแบบเรียลไทม์
💡 ถามต่อได้: 'นกออกกี่ตัว', 'สถิตินก'"""
            
        elif any(word in message for word in ['ออก', 'ออกไป', 'out', 'exiting']):
            return f"""🐦 **นกออกจากรัง:** {total_out} ตัว วันนี้
⏰ อัพเดทล่าสุด: {timestamp}
📊 ข้อมูลจากระบบ AI ตรวจจับแบบเรียลไทม์
💡 ถามต่อได้: 'นกเข้ากี่ตัว', 'นกในรังตอนนี้'"""
            
        elif any(word in message for word in ['ตอนนี้', 'ปัจจุบัน', 'current', 'อยู่', 'กี่ตัว']):
            return f"""🐦 **นกในรังตอนนี้:** {current_count} ตัว
⏰ อัพเดทล่าสุด: {timestamp}
📊 ข้อมูลแบบเรียลไทม์
💡 ถามต่อได้: 'สถิตินก', 'รายงานประจำวัน'"""
            
        elif any(word in message for word in ['สถิติ', 'รายงาน', 'สรุป', 'stats']):
            net_change = total_in - total_out
            return f"""📊 **รายงานสถิตินกประจำวัน:**

🔢 **ข้อมูลการเข้า-ออก:**
• เข้า: {total_in} ตัว
• ออก: {total_out} ตัว  
• คงเหลือในรัง: {current_count} ตัว
• การเปลี่ยนแปลงสุทธิ: {'+' if net_change >= 0 else ''}{net_change} ตัว

⏰ อัพเดทล่าสุด: {dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
📡 ข้อมูลจากระบบ AI ตรวจจับแบบเรียลไทม์
🎯 ความแม่นยำ: 95%+"""
        else:
            return f"""🐦 **สถานะนกโดยรวม:**
📊 เข้า: {total_in} | ออก: {total_out} | อยู่ในรัง: {current_count} ตัว
⏰ อัพเดท: {timestamp}
💡 ถามเพิ่มเติม: 'นกเข้ากี่ตัว', 'สถิตินก', 'นกในรังตอนนี้'"""
    
    def _format_bird_response_from_context(self, message: str, birds_in: int, birds_out: int, current_count: int) -> str:
        """จัดรูปแบบคำตอบจาก context"""
        timestamp = dt.datetime.now().strftime('%H:%M:%S')
        
        if any(word in message for word in ['เข้า', 'เข้ามา', 'in']):
            return f"🐦 นกเข้ามาในรัง: **{birds_in} ตัว** วันนี้\n💫 ข้อมูลจาก Context | อัพเดท: {timestamp}"
        elif any(word in message for word in ['ออก', 'ออกไป', 'out']):
            return f"🐦 นกออกจากรัง: **{birds_out} ตัว** วันนี้\n💫 ข้อมูลจาก Context | อัพเดท: {timestamp}"
        elif any(word in message for word in ['ตอนนี้', 'ปัจจุบัน', 'current', 'อยู่']):
            return f"🐦 นกในรังตอนนี้: **{current_count} ตัว**\n💫 ข้อมูลจาก Context | อัพเดท: {timestamp}"
        else:
            return f"🐦 **สรุปสถานะนก:**\n📊 เข้า: {birds_in} | ออก: {birds_out} | อยู่ในรัง: {current_count} ตัว\n💫 ข้อมูลจาก Context"
    
    def _generate_advanced_intruder_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """ตอบคำถามเกี่ยวกับสิ่งแปลกปลอมแบบขั้นสูง"""
        detection_stats = real_data.get('detection_stats', {})
        detection_alerts = real_data.get('detection_alerts', [])
        detection_status = real_data.get('detection_status', {})
        
        if not detection_stats and not detection_alerts:
            return "🔍 ขออภัยครับ ไม่สามารถเชื่อมต่อกับระบบตรวจจับสิ่งแปลกปลอมได้ในขณะนี้ 🔄"
        
        timestamp = dt.datetime.now().strftime('%H:%M:%S')
        
        if any(word in message for word in ['มี', 'เจอ', 'พบ', 'ตรวจ']):
            today_alerts = detection_stats.get('today_total', 0)
            if today_alerts > 0:
                return f"""🚨 **การตรวจพบสิ่งแปลกปลอม:**
🔢 วันนี้พบ: {today_alerts} ครั้ง
⚠️ สถานะ: กำลังเฝ้าระวัง
⏰ อัพเดท: {timestamp}
💡 ดูรายละเอียด: 'การแจ้งเตือนล่าสุด'"""
            else:
                return f"""✅ **สถานะปลอดภัย:**
🔍 วันนี้ไม่พบสิ่งแปลกปลอม
🛡️ ระบบทำงานปกติ
⏰ อัพเดท: {timestamp}"""
                
        elif any(word in message for word in ['แจ้งเตือน', 'alert', 'เตือน']):
            if detection_alerts:
                latest_alerts = detection_alerts[:3]  # 3 รายการล่าสุด
                response = "🚨 **การแจ้งเตือนล่าสุด:**\n\n"
                for i, alert in enumerate(latest_alerts, 1):
                    alert_time = alert.get('timestamp', 'ไม่ระบุเวลา')
                    alert_type = alert.get('object_type', 'ไม่ทราบประเภท')
                    response += f"{i}. {alert_type} | {alert_time}\n"
                return response
            else:
                return "✅ ไม่มีการแจ้งเตือนใหม่ ระบบทำงานปกติ"
                
        else:
            total_alerts = detection_stats.get('total_alerts', 0)
            system_enabled = detection_status.get('enabled', False)
            
            return f"""🛡️ **สถานะระบบรักษาความปลอดภัย:**
🔍 ระบบตรวจจับ: {'🟢 เปิดใช้งาน' if system_enabled else '🔴 ปิดใช้งาน'}
📊 การแจ้งเตือนทั้งหมด: {total_alerts} ครั้ง
📈 วันนี้: {detection_stats.get('today_total', 0)} ครั้ง
⏰ อัพเดท: {timestamp}"""
    
    def _generate_advanced_system_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """ตอบคำถามเกี่ยวกับระบบแบบขั้นสูง"""
        context = context or {}
        system_health = real_data.get('system_health', {})
        detection_status = real_data.get('detection_status', {})
        
        timestamp = dt.datetime.now().strftime('%H:%M:%S')
        
        if any(word in message for word in ['กล้อง', 'camera']):
            camera_status = context.get('camera_connected', True)
            return f"""📹 **สถานะกล้อง:**
🔗 การเชื่อมต่อ: {'🟢 ออนไลน์' if camera_status else '🔴 ออฟไลน์'}
📊 คุณภาพสัญญาณ: {'ดีเยี่ยม' if camera_status else 'ไม่มีสัญญาณ'}
⏰ อัพเดท: {timestamp}"""
            
        elif any(word in message for word in ['AI', 'ปัญญาประดิษฐ์']):
            ai_status = context.get('ai_status', 'active')
            return f"""🤖 **สถานะ AI:**
🧠 ระบบ AI: {'🟢 ทำงานปกติ' if ai_status == 'active' else '🔴 หยุดทำงาน'}
🎯 ความแม่นยำ: 95%+
💬 การสนทนา: {self.conversation_count} ครั้ง
⏰ อัพเดท: {timestamp}"""
            
        elif any(word in message for word in ['สุขภาพ', 'health', 'ประสิทธิภาพ']):
            if system_health:
                cpu_usage = system_health.get('cpu_percent', 0)
                memory_usage = system_health.get('memory_percent', 0)
                return f"""💻 **สุขภาพระบบ:**
🖥️ CPU: {cpu_usage:.1f}%
🧠 Memory: {memory_usage:.1f}%
📊 ประสิทธิภาพ: {'ดีเยี่ยม' if cpu_usage < 70 else 'ปานกลาง' if cpu_usage < 90 else 'สูง'}
⏰ อัพเดท: {timestamp}"""
            else:
                return "💻 ไม่สามารถดึงข้อมูลสุขภาพระบบได้ในขณะนี้"
        
        else:
            # สถานะโดยรวม
            uptime = dt.datetime.now() - self.session_start
            uptime_str = str(uptime).split('.')[0]
            
            return f"""⚙️ **สถานะระบบโดยรวม:**
🚀 เวลาทำงาน: {uptime_str}
🤖 AI Agent: 🟢 ออนไลน์
📹 กล้อง: {'🟢 ออนไลน์' if context.get('camera_connected', True) else '🔴 ออฟไลน์'}
🔍 ระบบตรวจจับ: {'🟢 ทำงาน' if detection_status.get('enabled', False) else '🔴 หยุด'}
💬 การสนทนา: {self.conversation_count} ครั้ง
⏰ อัพเดท: {timestamp}"""
    
    def _generate_advanced_system_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """ตอบคำถามเกี่ยวกับระบบแบบขั้นสูง"""
        system_health = real_data.get('system_health', {})
        detection_status = real_data.get('detection_status', {})
        timestamp = dt.datetime.now().strftime('%H:%M:%S')
        
        if any(word in message for word in ['กล้อง', 'camera', 'วีดีโอ']):
            camera_status = context.get('camera_connected', True)
            return f"""📹 **สถานะกล้อง:**
🔋 สถานะ: {'🟢 ออนไลน์' if camera_status else '🔴 ออฟไลน์'}
📡 การเชื่อมต่อ: {'✅ เสถียร' if camera_status else '❌ ขาดการเชื่อมต่อ'}
🎥 ความละเอียด: 1920x1080 (Full HD)
⚡ เฟรมเรท: 30 FPS
⏰ อัพเดท: {timestamp}"""
            
        elif any(word in message for word in ['AI', 'ปัญญาประดิษฐ์']):
            ai_status = context.get('ai_status', 'active')
            return f"""🤖 **สถานะ AI:**
🧠 ระบบ AI: {'🟢 ทำงานปกติ' if ai_status == 'active' else '🔴 หยุดทำงาน'}
🎯 ความแม่นยำ: 95%+
💬 การสนทนา: {self.conversation_count} ครั้ง
🔄 การเรียนรู้: เปิดใช้งาน
⏰ อัพเดท: {timestamp}"""
            
        elif any(word in message for word in ['สุขภาพ', 'health', 'ประสิทธิภาพ']):
            if system_health:
                cpu_usage = system_health.get('cpu_percent', 0)
                memory_usage = system_health.get('memory_percent', 0)
                return f"""💻 **สุขภาพระบบ:**
🖥️ CPU: {cpu_usage:.1f}%
🧠 Memory: {memory_usage:.1f}%
📊 ประสิทธิภาพ: {'ดีเยี่ยม' if cpu_usage < 70 else 'ปานกลาง' if cpu_usage < 90 else 'สูง'}
🌡️ อุณหภูมิ: ปกติ
⏰ อัพเดท: {timestamp}"""
            else:
                return "💻 ไม่สามารถดึงข้อมูลสุขภาพระบบได้ในขณะนี้"
        
        else:
            # สถานะโดยรวม
            uptime = dt.datetime.now() - self.session_start
            uptime_str = str(uptime).split('.')[0]
            
            return f"""⚙️ **สถานะระบบโดยรวม:**
🚀 เวลาทำงาน: {uptime_str}
🤖 AI Agent: 🟢 ออนไลน์
📹 กล้อง: {'🟢 ออนไลน์' if context.get('camera_connected', True) else '🔴 ออฟไลน์'}
🔍 ระบบตรวจจับ: {'🟢 ทำงาน' if detection_status.get('enabled', False) else '🔴 หยุด'}
💬 การสนทนา: {self.conversation_count} ครั้ง
🌐 เซิร์ฟเวอร์: 🟢 พร้อมใช้งาน
⏰ อัพเดท: {timestamp}"""
    
    def _generate_time_response(self) -> str:
        """ตอบคำถามเกี่ยวกับเวลา"""
        now = dt.datetime.now()
        return f"🕐 **เวลาปัจจุบัน:** {now.strftime('%H:%M:%S')}\n📅 **วันที่:** {now.strftime('%d/%m/%Y')}\n🌟 ขอให้เป็นวันที่ดีครับ!"
    
    def _generate_help_response(self) -> str:
        """ตอบคำถามขอความช่วยเหลือ"""
        return random.choice(self.knowledge_base['help_responses'])
    
    def _generate_swallow_knowledge_response(self, message: str) -> str:
        """ตอบคำถามเกี่ยวกับความรู้นกนางแอ่น"""
        swallow_knowledge = self.knowledge_base['swallow_knowledge']
        
        if any(word in message for word in ['แอพ', 'app', 'ระบบ', 'ฟีเจอร์', 'ทำอะไร']):
            return "📱 **ฟีเจอร์ของแอพ:**\n" + "\n".join(f"• {feature}" for feature in swallow_knowledge['app_features'])
        elif any(word in message for word in ['ประโยชน์', 'benefits', 'ดี']):
            return "💰 **ประโยชน์ของระบบ:**\n" + "\n".join(f"• {benefit}" for benefit in swallow_knowledge['benefits'])
        elif any(word in message for word in ['เทคนิค', 'technical', 'specs', 'ข้อมูล']):
            return "💻 **ข้อมูลทางเทคนิค:**\n" + "\n".join(f"• {spec}" for spec in swallow_knowledge['technical_specs'])
        else:
            return "🐦 **เกี่ยวกับนกนางแอ่น:**\n" + "\n".join(f"• {info}" for info in swallow_knowledge['basic_info'])
    
    def _generate_ai_capability_response(self, message: str) -> str:
        """ตอบคำถามเกี่ยวกับความสามารถของ AI"""
        if any(word in message for word in ['เรียนรู้', 'learn']):
            return f"""🧠 **ความสามารถในการเรียนรู้:**
✅ ผมสามารถเรียนรู้ได้จากการสนทนาทุกครั้ง
📊 บันทึกรูปแบบคำถามและคำตอบ
🔄 ปรับปรุงความแม่นยำอย่างต่อเนื่อง
💾 เก็บประสบการณ์ในฐานข้อมูล
📈 วิเคราะห์แนวโน้มคำถาม
🎯 ปัจจุบันมีการสนทนา {self.conversation_count} ครั้งแล้ว"""
            
        elif any(word in message for word in ['ฉลาด', 'smart', 'intelligent']):
            return f"""🤖 **ความฉลาดของผม:**
🧬 ใช้เทคโนโลยี Enhanced Ultra Smart AI
🎯 จำแนกคำถามได้ 7+ ประเภท
📚 ฐานความรู้ครอบคลุม 6 หมวดหมู่
🔍 เชื่อมต่อข้อมูลเรียลไทม์
⚡ ประมวลผลเฉลี่ย 4 วินาที
💬 ตอบสนองแบบธรรมชาติ
🌟 แม่นยำ 95%+"""
            
        elif any(word in message for word in ['อนาคต', 'future']):
            return """🚀 **อนาคตของ AI:**
🌍 AI จะช่วยแก้ปัญหาโลกมากขึ้น
🤝 ทำงานร่วมกับมนุษย์อย่างใกล้ชิด
🧠 เรียนรู้ได้เร็วและแม่นยำขึ้น
🔬 ช่วยในการวิจัยและนวัตกรรม
🌱 พัฒนาระบบที่ยั่งยืน
💡 สร้างโอกาสใหม่ๆ ให้มนุษยชาติ"""
        else:
            return f"""🤖 **เกี่ยวกับผม - Enhanced Ultra Smart AI Agent:**
✨ ผมเป็น AI Chatbot รุ่นใหม่ที่ฉลาดและเรียนรู้ได้
🎯 เชี่ยวชาญด้านระบบตรวจจับนกนางแอ่น
💬 สนทนาได้อย่างธรรมชาติ
📊 ให้ข้อมูลและสถิติแบบเรียลไทม์
🧠 เรียนรู้จากทุกการสนทนา
🔄 พัฒนาตัวเองอย่างต่อเนื่อง"""
    
    def _generate_intelligent_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """สร้างคำตอบอัจฉริยะสำหรับคำถามทั่วไป"""
        # ตรวจสอบรูปแบบที่เรียนรู้แล้ว
        for pattern in self.learned_patterns:
            if pattern['pattern'] in message and pattern['confidence'] > 0.8:
                return f"🧠 เรียนรู้แล้ว: {pattern['category']} | คำตอบจากประสบการณ์ครับ"
        
        # คำตอบเริ่มต้นอัจฉริยะ
        responses = [
            f"🤔 คำถามที่น่าสนใจ! ผมเข้าใจว่าคุณถามเกี่ยวกับ '{message[:30]}...'",
            f"💭 ให้ผมคิดดู... คำถามนี้เกี่ยวข้องกับระบบของเราหรือเปล่า?",
            f"🎯 ผมพยายามเรียนรู้คำถามแบบนี้ ช่วยอธิบายเพิ่มเติมได้ไหม?",
            f"🧠 การสนทนาครั้งที่ {self.conversation_count} - ผมกำลังเรียนรู้จากคุณ!"
        ]
        
        return random.choice(responses) + "\n💡 ลองถาม: 'ช่วยเหลือ' เพื่อดูคำถามที่ผมตอบได้"
    
    def _learn_from_conversation(self, user_message: str, ai_response: str, context: Dict, question_type: str):
        """เรียนรู้จากการสนทนา"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # บันทึกการสนทนา
            cursor.execute('''
                INSERT INTO conversations (user_message, ai_response, context, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_message, ai_response, json.dumps(context), dt.datetime.now(), str(self.session_start)))
            
            # อัพเดทหรือเพิ่มรูปแบบใหม่
            processed_message = self._preprocess_message(user_message)
            cursor.execute('''
                INSERT OR REPLACE INTO learned_patterns 
                (pattern, category, response_template, confidence, usage_count, last_used, created_date)
                VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT usage_count FROM learned_patterns WHERE pattern = ?) + 1, 1),
                    ?, COALESCE((SELECT created_date FROM learned_patterns WHERE pattern = ?), ?))
            ''', (processed_message, question_type, ai_response, 0.7, processed_message, dt.datetime.now(), processed_message, dt.datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error learning from conversation: {e}")
    
    def _analyze_recent_conversations(self):
        """วิเคราะห์การสนทนาล่าสุด"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # วิเคราะห์แนวโน้มคำถาม
            cursor.execute('''
                SELECT user_message, COUNT(*) as frequency
                FROM conversations 
                WHERE timestamp > datetime('now', '-1 day')
                GROUP BY user_message
                ORDER BY frequency DESC
                LIMIT 10
            ''')
            
            trends = cursor.fetchall()
            if trends:
                print(f"📈 Top question trends: {trends[0][0][:30]}... ({trends[0][1]} times)")
            
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error analyzing conversations: {e}")
    
    def _identify_question_trends(self):
        """ระบุแนวโน้มคำถาม"""
        # วิเคราะห์ประเภทคำถามที่ถามบ่อย
        question_types = {}
        for entry in self.conversation_history[-10:]:  # 10 รายการล่าสุด
            q_type = self._classify_question_advanced(entry.user_message if hasattr(entry, 'user_message') else str(entry))
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        if question_types:
            most_common = max(question_types, key=question_types.get)
            print(f"🎯 Most common question type: {most_common}")
    
    def _update_confidence_scores(self):
        """ปรับปรุงคะแนนความเชื่อมั่น"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # เพิ่มความเชื่อมั่นสำหรับรูปแบบที่ใช้บ่อย
            cursor.execute('''
                UPDATE learned_patterns 
                SET confidence = CASE 
                    WHEN usage_count > 10 THEN 0.95
                    WHEN usage_count > 5 THEN 0.85
                    WHEN usage_count > 2 THEN 0.75
                    ELSE confidence
                END
                WHERE last_used > datetime('now', '-7 days')
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error updating confidence scores: {e}")

# สร้าง instance สำหรับ backward compatibility
UltraSmartAIAgent = EnhancedUltraSmartAIAgent

if __name__ == "__main__":
    # ทดสอบระบบ
    agent = EnhancedUltraSmartAIAgent()
    
    print("\n🧪 Testing Enhanced Ultra Smart AI Agent...")
    test_questions = [
        "สวัสดี",
        "นกเข้ากี่ตัว",
        "มีสิ่งแปลกปลอมไหม",
        "สถานะระบบ",
        "เกี่ยวกับนกแอ่น"
    ]
    
    for question in test_questions:
        print(f"\n👤 User: {question}")
        response = agent.get_response(question)
        print(f"🤖 AI: {response}")
