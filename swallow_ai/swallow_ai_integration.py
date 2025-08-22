#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 Swallow AI Integration Controller
ระบบควบคุมการเชื่อมต่อและจัดการทุกส่วนของ Swallow AI System
"""

import os
import sys
import time
import json
import subprocess
import signal
import threading
import logging
from typing import Dict, List, Any, Optional

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('swallow_ai_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SwallowAIIntegrationController:
    """🔗 ระบบควบคุมการเชื่อมต่อ Swallow AI ทั้งหมด"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.processes = {}
        self.status = {
            'main_system': False,
            'ai_agent_web': False,
            'integration_status': 'stopped'
        }
        
        # Component files
        self.components = {
            'main_system': 'app_working.py',
            'ai_agent_web': 'ai_agent_web.py',
            'ultra_smart_ai_agent': 'ultra_smart_ai_agent.py',
            'advanced_object_detector': 'advanced_object_detector.py'
        }
        
        # Ports
        self.ports = {
            'main_system': 5000,
            'ai_agent_web': 8080
        }
        
        print("🔗 Swallow AI Integration Controller initialized")
        print(f"📁 Base directory: {self.base_dir}")
        
    def check_dependencies(self) -> Dict[str, bool]:
        """ตรวจสอบไฟล์ที่จำเป็น"""
        results = {}
        
        for component, filename in self.components.items():
            filepath = os.path.join(self.base_dir, filename)
            exists = os.path.exists(filepath)
            results[component] = exists
            
            if exists:
                logger.info(f"✅ {component}: {filename} - พร้อมใช้งาน")
            else:
                logger.warning(f"❌ {component}: {filename} - ไม่พบไฟล์")
        
        return results
    
    def start_component(self, component: str) -> bool:
        """เริ่มต้นส่วนประกอบ"""
        if component not in self.components:
            logger.error(f"❌ ไม่พบ component: {component}")
            return False
        
        filename = self.components[component]
        filepath = os.path.join(self.base_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"❌ ไม่พบไฟล์: {filepath}")
            return False
        
        try:
            logger.info(f"🚀 เริ่มต้น {component}...")
            
            # เปลี่ยน working directory
            os.chdir(self.base_dir)
            
            # สำหรับ Windows
            if os.name == 'nt':
                process = subprocess.Popen(
                    ['python', filename],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.base_dir,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    ['python3', filename],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.base_dir
                )
            
            self.processes[component] = process
            
            # ให้เวลาในการเริ่มต้น
            time.sleep(2)
            
            # ตรวจสอบว่า process ยังทำงานอยู่หรือไม่
            if process.poll() is None:
                self.status[component] = True
                logger.info(f"✅ {component} เริ่มต้นสำเร็จ (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ {component} เริ่มต้นไม่สำเร็จ")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้น {component}: {e}")
            return False
    
    def stop_component(self, component: str) -> bool:
        """หยุดส่วนประกอบ"""
        if component not in self.processes:
            logger.warning(f"⚠️ {component} ไม่ได้ทำงานอยู่")
            return True
        
        try:
            process = self.processes[component]
            
            if os.name == 'nt':
                # Windows
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Unix/Linux
                process.terminate()
            
            # รอให้ process หยุด
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            del self.processes[component]
            self.status[component] = False
            logger.info(f"🛑 {component} หยุดแล้ว")
            return True
            
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการหยุด {component}: {e}")
            return False
    
    def start_all(self) -> bool:
        """เริ่มต้นระบบทั้งหมด"""
        logger.info("🚀 เริ่มต้นระบบ Swallow AI ทั้งหมด...")
        
        # ตรวจสอบ dependencies
        deps = self.check_dependencies()
        
        # เริ่มต้น main system ก่อน
        if deps['main_system']:
            if self.start_component('main_system'):
                logger.info("✅ Main System เริ่มต้นสำเร็จ")
                time.sleep(3)  # รอให้ระบบหลักพร้อม
            else:
                logger.error("❌ Main System เริ่มต้นไม่สำเร็จ")
                return False
        
        # เริ่มต้น AI Agent Web Interface
        if deps['ai_agent_web']:
            if self.start_component('ai_agent_web'):
                logger.info("✅ AI Agent Web Interface เริ่มต้นสำเร็จ")
            else:
                logger.warning("⚠️ AI Agent Web Interface เริ่มต้นไม่สำเร็จ")
        
        self.status['integration_status'] = 'running'
        
        logger.info("🎉 ระบบ Swallow AI เริ่มต้นครบถ้วนแล้ว!")
        self.print_system_info()
        return True
    
    def stop_all(self) -> bool:
        """หยุดระบบทั้งหมด"""
        logger.info("🛑 หยุดระบบ Swallow AI ทั้งหมด...")
        
        # หยุดทุก component
        for component in list(self.processes.keys()):
            self.stop_component(component)
        
        self.status['integration_status'] = 'stopped'
        logger.info("✅ หยุดระบบทั้งหมดแล้ว")
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """ดึงสถานะระบบ"""
        # อัพเดทสถานะ process
        for component, process in self.processes.items():
            if process.poll() is None:
                self.status[component] = True
            else:
                self.status[component] = False
        
        return {
            'status': self.status.copy(),
            'processes': {
                component: {
                    'pid': process.pid,
                    'running': process.poll() is None
                } for component, process in self.processes.items()
            },
            'ports': self.ports,
            'dependencies': self.check_dependencies()
        }
    
    def print_system_info(self):
        """แสดงข้อมูลระบบ"""
        print("\n" + "="*60)
        print("🚀 SWALLOW AI SYSTEM INTEGRATION")
        print("="*60)
        
        status = self.get_system_status()
        
        print("\n📊 สถานะระบบ:")
        for component, running in status['status'].items():
            if component != 'integration_status':
                icon = "🟢" if running else "🔴"
                print(f"  {icon} {component}: {'ทำงาน' if running else 'หยุด'}")
        
        print(f"\n🔗 สถานะการเชื่อมต่อ: {status['status']['integration_status']}")
        
        print("\n🌐 เว็บเซอร์วิส:")
        if status['status'].get('main_system'):
            print(f"  📱 Main System: http://127.0.0.1:{self.ports['main_system']}")
            print(f"  🤖 Bird Detection API: http://127.0.0.1:{self.ports['main_system']}/api/statistics")
            print(f"  🚨 Intruder Detection API: http://127.0.0.1:{self.ports['main_system']}/api/object-detection/stats")
        
        if status['status'].get('ai_agent_web'):
            print(f"  💬 AI Agent Chat: http://127.0.0.1:{self.ports['ai_agent_web']}")
        
        print("\n" + "="*60)
    
    def monitor_system(self):
        """ตรวจสอบระบบแบบต่อเนื่อง"""
        try:
            while self.status['integration_status'] == 'running':
                time.sleep(10)
                
                # ตรวจสอบ process ที่หายไป
                for component in list(self.processes.keys()):
                    process = self.processes[component]
                    if process.poll() is not None:
                        logger.warning(f"⚠️ {component} หยุดทำงานแล้ว")
                        self.status[component] = False
                        del self.processes[component]
                
        except KeyboardInterrupt:
            logger.info("📢 ได้รับสัญญาณหยุด...")
            self.stop_all()

def main():
    """ฟังก์ชันหลัก"""
    controller = SwallowAIIntegrationController()
    
    try:
        print("🚀 Swallow AI Integration Controller")
        print("เลือกการดำเนินการ:")
        print("1. เริ่มต้นระบบทั้งหมด (Start All)")
        print("2. ตรวจสอบสถานะ (Check Status)")
        print("3. หยุดระบบทั้งหมด (Stop All)")
        print("4. ตรวจสอบไฟล์ที่จำเป็น (Check Dependencies)")
        print("5. เริ่มต้นและตรวจสอบต่อเนื่อง (Start & Monitor)")
        
        choice = input("\nเลือก (1-5): ").strip()
        
        if choice == "1":
            controller.start_all()
        elif choice == "2":
            status = controller.get_system_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
        elif choice == "3":
            controller.stop_all()
        elif choice == "4":
            deps = controller.check_dependencies()
            print("\n📁 ไฟล์ที่จำเป็น:")
            for component, exists in deps.items():
                icon = "✅" if exists else "❌"
                print(f"  {icon} {component}: {controller.components[component]}")
        elif choice == "5":
            if controller.start_all():
                controller.monitor_system()
        else:
            print("❌ ตัวเลือกไม่ถูกต้อง")
            
    except KeyboardInterrupt:
        print("\n👋 ขอบคุณที่ใช้บริการ!")
        controller.stop_all()
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาด: {e}")
        controller.stop_all()

if __name__ == "__main__":
    main()
