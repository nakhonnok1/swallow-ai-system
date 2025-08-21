#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 Ultra Smart AI Agent Web Interface
เว็บอินเตอร์เฟซสำหรับ AI Agent อัจฉริยะ
"""

from flask import Flask, render_template, request, jsonify, Response
import json
import datetime as dt
import logging
import threading
import time
import os
import sys

# เพิ่ม path สำหรับ import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra_smart_ai_agent import UltraSmartAIAgent

# ตั้งค่า Flask
app = Flask(__name__)
app.secret_key = 'ultra_smart_ai_agent_2024'

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# สร้าง AI Agent
ai_agent = UltraSmartAIAgent()

@app.route('/')
def index():
    """หน้าหลัก"""
    return render_template('ai_agent_chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API สำหรับการสนทนา - เชื่อมต่อกับระบบหลัก"""
    try:
        # ส่งต่อไปยัง main system API
        import requests
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'ข้อความไม่ถูกต้อง',
                'success': False
            }), 400
        
        # ส่งไปยัง main system
        response = requests.post(
            'http://127.0.0.1:5000/api/ai-agent/chat',
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            # Fallback ใช้ local AI agent
            user_message = data['message'].strip()
            if not user_message:
                return jsonify({
                    'error': 'กรุณาส่งข้อความมาด้วย',
                    'success': False
                }), 400
            
            # ส่งไปให้ AI Agent ตอบ
            start_time = time.time()
            ai_response = ai_agent.get_response(user_message)
            response_time = time.time() - start_time
            
            return jsonify({
                'response': ai_response,
                'response_time': round(response_time, 2),
                'conversation_count': ai_agent.conversation_count,
                'timestamp': dt.datetime.now().isoformat(),
                'success': True,
                'source': 'local_fallback'
            })
        
    except requests.exceptions.RequestException:
        # Fallback ใช้ local AI agent
        data = request.get_json()
        user_message = data['message'].strip()
        
        start_time = time.time()
        ai_response = ai_agent.get_response(user_message)
        response_time = time.time() - start_time
        
        return jsonify({
            'response': ai_response,
            'response_time': round(response_time, 2),
            'conversation_count': ai_agent.conversation_count,
            'timestamp': dt.datetime.now().isoformat(),
            'success': True,
            'source': 'local_fallback'
        })
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({
            'error': f'เกิดข้อผิดพลาด: {str(e)}',
            'success': False
        }), 500

@app.route('/api/status')
def status():
    """API สถานะระบบ"""
    try:
        uptime = dt.datetime.now() - ai_agent.session_start
        
        return jsonify({
            'status': 'online',
            'uptime': str(uptime).split('.')[0],
            'conversation_count': ai_agent.conversation_count,
            'learned_patterns': len(ai_agent.learned_patterns),
            'knowledge_base_size': len(ai_agent.knowledge_base),
            'timestamp': dt.datetime.now().isoformat(),
            'success': True
        })
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({
            'error': f'เกิดข้อผิดพลาด: {str(e)}',
            'success': False
        }), 500

@app.route('/api/stats')
def stats():
    """API สถิติการใช้งาน"""
    try:
        return jsonify({
            'total_conversations': ai_agent.conversation_count,
            'learned_patterns': len(ai_agent.learned_patterns),
            'session_start': ai_agent.session_start.isoformat(),
            'uptime_seconds': (dt.datetime.now() - ai_agent.session_start).total_seconds(),
            'knowledge_categories': list(ai_agent.knowledge_base.keys()),
            'api_endpoints': list(ai_agent.api_endpoints.keys()),
            'success': True
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'error': f'เกิดข้อผิดพลาด: {str(e)}',
            'success': False
        }), 500

@app.route('/templates/ai_agent_chat.html')
def get_template():
    """ส่งไฟล์ HTML template"""
    html_content = '''<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Ultra Smart AI Agent</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .status-bar {
            background: #f8f9fa;
            padding: 10px 20px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9em;
        }
        
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .ai-message {
            background: white;
            color: #333;
            border: 1px solid #e9ecef;
            margin-right: auto;
        }
        
        .message-time {
            font-size: 0.7em;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e9ecef;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #messageInput:focus {
            border-color: #007bff;
        }
        
        #sendButton {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }
        
        #sendButton:hover {
            background: #0056b3;
        }
        
        #sendButton:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 10px;
            color: #666;
        }
        
        .quick-questions {
            padding: 0 20px 20px;
            background: white;
        }
        
        .quick-questions h4 {
            margin-bottom: 10px;
            color: #666;
        }
        
        .quick-btn {
            display: inline-block;
            margin: 5px;
            padding: 6px 12px;
            background: #e9ecef;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
            transition: background-color 0.3s;
        }
        
        .quick-btn:hover {
            background: #007bff;
            color: white;
        }
        
        .stats {
            display: flex;
            justify-content: space-around;
            text-align: center;
        }
        
        .stat-item {
            font-size: 0.8em;
        }
        
        .stat-value {
            font-weight: bold;
            color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Ultra Smart AI Agent</h1>
            <p>ระบบ AI อัจฉริยะสำหรับนกนางแอ่น</p>
        </div>
        
        <div class="status-bar">
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="statusIndicator">🟢 ออนไลน์</div>
                    <div>สถานะ</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="conversationCount">0</div>
                    <div>การสนทนา</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="uptime">00:00:00</div>
                    <div>เวลาทำงาน</div>
                </div>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message ai-message">
                <div>🤖 สวัสดีครับ! ผมเป็น Ultra Smart AI Agent พร้อมตอบคำถามเกี่ยวกับระบบตรวจจับนกนางแอ่นและสิ่งแปลกปลอม</div>
                <div class="message-time" id="initialTime"></div>
            </div>
        </div>
        
        <div class="quick-questions">
            <h4>🎯 คำถามด่วน:</h4>
            <button class="quick-btn" onclick="sendQuickMessage('นกเข้ากี่ตัว')">นกเข้ากี่ตัว</button>
            <button class="quick-btn" onclick="sendQuickMessage('มีสิ่งแปลกปลอมไหม')">มีสิ่งแปลกปลอมไหม</button>
            <button class="quick-btn" onclick="sendQuickMessage('สถานะระบบ')">สถานะระบบ</button>
            <button class="quick-btn" onclick="sendQuickMessage('สถิตินก')">สถิตินก</button>
            <button class="quick-btn" onclick="sendQuickMessage('เวลาเท่าไหร่')">เวลาเท่าไหร่</button>
            <button class="quick-btn" onclick="sendQuickMessage('ช่วย')">ช่วยเหลือ</button>
        </div>
        
        <div class="loading" id="loading">
            🤖 กำลังประมวลผล...
        </div>
        
        <div class="input-container">
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="พิมพ์คำถามของคุณ..." autocomplete="off">
                <button id="sendButton" onclick="sendMessage()">ส่ง</button>
            </div>
        </div>
    </div>

    <script>
        // ตั้งเวลาเริ่มต้น
        document.getElementById('initialTime').textContent = new Date().toLocaleTimeString('th-TH');
        
        // การจัดการ Enter key
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        function sendQuickMessage(message) {
            document.getElementById('messageInput').value = message;
            sendMessage();
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // แสดงข้อความของผู้ใช้
            addMessage(message, 'user');
            input.value = '';
            
            // แสดง loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('sendButton').disabled = true;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addMessage(data.response, 'ai', data.response_time);
                    updateStats(data.conversation_count);
                } else {
                    addMessage('⚠️ ' + data.error, 'ai');
                }
            } catch (error) {
                addMessage('⚠️ เกิดข้อผิดพลาดในการเชื่อมต่อ: ' + error.message, 'ai');
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('sendButton').disabled = false;
                input.focus();
            }
        }
        
        function addMessage(text, sender, responseTime = null) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            const now = new Date();
            const timeStr = now.toLocaleTimeString('th-TH');
            
            let messageContent = `<div>${text}</div>`;
            
            if (responseTime) {
                messageContent += `<div class="message-time">${timeStr} (${responseTime}s)</div>`;
            } else {
                messageContent += `<div class="message-time">${timeStr}</div>`;
            }
            
            messageDiv.innerHTML = messageContent;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function updateStats(conversationCount) {
            document.getElementById('conversationCount').textContent = conversationCount || 0;
        }
        
        // อัพเดทสถานะแบบ real-time
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('statusIndicator').textContent = '🟢 ออนไลน์';
                    document.getElementById('uptime').textContent = data.uptime;
                    document.getElementById('conversationCount').textContent = data.conversation_count;
                } else {
                    document.getElementById('statusIndicator').textContent = '🔴 ออฟไลน์';
                }
            } catch (error) {
                document.getElementById('statusIndicator').textContent = '🔴 ออฟไลน์';
            }
        }
        
        // อัพเดทสถานะทุก 30 วินาที
        setInterval(updateStatus, 30000);
        updateStatus(); // เรียกครั้งแรก
        
        // Focus ที่ input เมื่อโหลดหน้า
        window.onload = function() {
            document.getElementById('messageInput').focus();
        };
    </script>
</body>
</html>'''
    return Response(html_content, mimetype='text/html')

# สร้างโฟลเดอร์ templates หากไม่มี
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

# สร้างไฟล์ template
template_file = os.path.join(templates_dir, 'ai_agent_chat.html')
if not os.path.exists(template_file):
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write('')  # ไฟล์ว่าง เพราะเราส่ง HTML จาก route

if __name__ == '__main__':
    print("🚀 Starting Ultra Smart AI Agent Web Interface...")
    print("🌐 Open your browser and go to:")
    print("   http://127.0.0.1:8080")
    print("   http://localhost:8080")
    print("-" * 50)
    
    try:
        app.run(
            host='0.0.0.0',
            port=8080,
            debug=True,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n👋 Ultra Smart AI Agent Web Interface stopped!")
    except Exception as e:
        print(f"⚠️ Error starting web interface: {e}")
