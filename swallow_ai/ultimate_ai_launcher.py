#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ULTIMATE AI SYSTEM LAUNCHER
ตัวเริ่มต้นระบบ AI แบบครบถ้วน
Version: 1.0 - COMPLETE AI ECOSYSTEM STARTER

🎯 เริ่มต้นระบบ AI ทั้งหมด:
- Ultimate AI Vision System
- AI Helper System  
- AI Performance Booster
- Enhanced Ultra Smart AI Agent
- Real-time Dashboard
- System Monitoring
"""

import sys
import time
import signal
import threading
import logging
from pathlib import Path
from datetime import datetime

# กำหนด logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_ai_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('UltimateAILauncher')

class UltimateAILauncher:
    """ตัวเริ่มต้นระบบ AI แบบครบถ้วน"""
    
    def __init__(self):
        self.running = False
        self.systems = {}
        self.startup_time = None
        self.shutdown_handlers = []
        
        # กำหนด signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """จัดการ signal สำหรับปิดระบบ"""
        logger.info(f"🛑 รับ signal {signum}, เริ่มปิดระบบ...")
        self.shutdown()
        sys.exit(0)
    
    def startup_banner(self):
        """แสดง banner เริ่มต้น"""
        banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    🚀 ULTIMATE AI SYSTEM LAUNCHER v1.0                                       ║
║                                                                               ║
║    🎯 เริ่มต้นระบบ AI ทั้งหมด                                                  ║
║    ⚡ พร้อมการปรับปรุงประสิทธิภาพแบบอัตโนมัติ                                  ║
║    🤖 AI Helper System สำหรับการจัดการระบบ                                    ║
║    📊 Real-time Monitoring & Analytics                                        ║
║                                                                               ║
║    Developed by: Swallow AI Team                                              ║
║    License: MIT                                                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info("🎯 ULTIMATE AI SYSTEM เริ่มต้น...")
    
    def check_prerequisites(self):
        """ตรวจสอบความพร้อมของระบบ"""
        logger.info("🔍 ตรวจสอบความพร้อมของระบบ...")
        
        issues = []
        
        # ตรวจสอบไฟล์ที่จำเป็น
        required_files = [
            'opencv_yolo_detector.py',
            'ai_helper_system.py', 
            'ai_performance_booster.py',
            'ultimate_ai_config.py'
        ]
        
        for file in required_files:
            if not Path(file).exists():
                issues.append(f"❌ ไฟล์ที่จำเป็นไม่พบ: {file}")
        
        # ตรวจสอบ dependencies
        try:
            import cv2
            logger.info(f"✅ OpenCV: {cv2.__version__}")
        except ImportError:
            issues.append("❌ OpenCV ไม่ได้ติดตั้ง")
        
        try:
            import numpy as np
            logger.info(f"✅ NumPy: {np.__version__}")
        except ImportError:
            issues.append("❌ NumPy ไม่ได้ติดตั้ง")
        
        try:
            import sqlite3
            logger.info(f"✅ SQLite3: {sqlite3.sqlite_version}")
        except ImportError:
            issues.append("❌ SQLite3 ไม่ได้ติดตั้ง")
        
        # ตรวจสอบการตั้งค่า
        try:
            from ultimate_ai_config import validate_all_configs
            config_issues = validate_all_configs()
            if config_issues:
                issues.extend(config_issues)
            else:
                logger.info("✅ การตั้งค่าทั้งหมดถูกต้อง")
        except Exception as e:
            issues.append(f"❌ ไม่สามารถตรวจสอบการตั้งค่าได้: {e}")
        
        if issues:
            logger.error("❌ พบปัญหาในการตรวจสอบความพร้อม:")
            for issue in issues:
                logger.error(f"   {issue}")
            return False
        
        logger.info("✅ ระบบพร้อมใช้งาน")
        return True
    
    def start_ai_vision_system(self):
        """เริ่มต้น AI Vision System"""
        logger.info("🎥 เริ่มต้น Ultimate AI Vision System...")
        
        try:
            from opencv_yolo_detector import OpenCVYOLODetector
            ai_detector = OpenCVYOLODetector()
            
            if ai_detector.available:
                self.systems['ai_vision'] = {
                    'instance': ai_detector,
                    'status': 'running',
                    'start_time': time.time()
                }
                logger.info("✅ Ultimate AI Vision System เริ่มต้นสำเร็จ")
                return ai_detector
            else:
                logger.error("❌ Ultimate AI Vision System ไม่พร้อมใช้งาน")
                return None
                
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้น AI Vision: {e}")
            return None
    
    def start_ai_helper_system(self, ai_detector=None):
        """เริ่มต้น AI Helper System"""
        logger.info("🤖 เริ่มต้น AI Helper System...")
        
        try:
            from ai_helper_system import get_ai_helper
            ai_helper = get_ai_helper()
            
            # ลงทะเบียน AI systems
            if ai_detector:
                ai_helper.register_ai_system("main_detector", ai_detector)
                logger.info("🔗 เชื่อมต่อ AI Vision กับ Helper System")
            
            self.systems['ai_helper'] = {
                'instance': ai_helper,
                'status': 'running',
                'start_time': time.time()
            }
            
            logger.info("✅ AI Helper System เริ่มต้นสำเร็จ")
            return ai_helper
            
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้น AI Helper: {e}")
            return None
    
    def start_performance_booster(self, ai_detector=None):
        """เริ่มต้น Performance Booster"""
        logger.info("⚡ เริ่มต้น AI Performance Booster...")
        
        try:
            from ai_performance_booster import get_performance_booster
            performance_booster = get_performance_booster()
            
            # ปรับปรุงประสิทธิภาพ AI systems
            if ai_detector:
                performance_booster.optimize_ai_system(ai_detector)
                logger.info("🚀 ปรับปรุงประสิทธิภาพ AI Vision System")
            
            self.systems['performance_booster'] = {
                'instance': performance_booster,
                'status': 'running',
                'start_time': time.time()
            }
            
            logger.info("✅ AI Performance Booster เริ่มต้นสำเร็จ")
            return performance_booster
            
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้น Performance Booster: {e}")
            return None
    
    def start_ai_chatbot(self):
        """เริ่มต้น AI Chatbot"""
        logger.info("💬 เริ่มต้น Enhanced Ultra Smart AI Agent...")
        
        try:
            from enhanced_ultra_smart_ai_agent import EnhancedUltraSmartAIAgent
            ai_chatbot = EnhancedUltraSmartAIAgent()
            
            self.systems['ai_chatbot'] = {
                'instance': ai_chatbot,
                'status': 'running',
                'start_time': time.time()
            }
            
            logger.info("✅ AI Chatbot เริ่มต้นสำเร็จ")
            return ai_chatbot
            
        except Exception as e:
            logger.error(f"⚠️ AI Chatbot ไม่พร้อม: {e}")
            return None
    
    def start_monitoring_system(self):
        """เริ่มต้นระบบ Monitoring"""
        logger.info("📊 เริ่มต้นระบบ Monitoring...")
        
        def monitoring_loop():
            """วนลูป monitoring"""
            while self.running:
                try:
                    self.update_system_status()
                    time.sleep(10)  # อัพเดททุก 10 วินาที
                except Exception as e:
                    logger.error(f"❌ เกิดข้อผิดพลาดใน monitoring: {e}")
                    time.sleep(5)
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        self.systems['monitoring'] = {
            'instance': monitoring_thread,
            'status': 'running',
            'start_time': time.time()
        }
        
        logger.info("✅ ระบบ Monitoring เริ่มต้นสำเร็จ")
    
    def update_system_status(self):
        """อัพเดทสถานะระบบ"""
        current_time = time.time()
        
        # ตรวจสอบสถานะระบบแต่ละตัว
        for system_name, system_data in self.systems.items():
            if system_name == 'monitoring':
                continue
                
            try:
                instance = system_data['instance']
                uptime = current_time - system_data['start_time']
                
                # อัพเดทข้อมูล
                system_data['uptime'] = uptime
                system_data['last_check'] = current_time
                
                # ตรวจสอบสถานะเฉพาะ
                if system_name == 'ai_vision' and hasattr(instance, 'get_detection_stats'):
                    stats = instance.get_detection_stats()
                    system_data['stats'] = stats
                
            except Exception as e:
                logger.warning(f"⚠️ ไม่สามารถอัพเดทสถานะ {system_name}: {e}")
                system_data['status'] = 'error'
    
    def show_status_dashboard(self):
        """แสดง dashboard สถานะระบบ"""
        print("\n" + "="*80)
        print("📊 ULTIMATE AI SYSTEM STATUS DASHBOARD")
        print("="*80)
        
        current_time = time.time()
        total_uptime = current_time - self.startup_time if self.startup_time else 0
        
        print(f"🕐 เวลาทำงานรวม: {total_uptime/3600:.1f} ชั่วโมง")
        print(f"🖥️ ระบบที่เปิดใช้งาน: {len([s for s in self.systems.values() if s['status'] == 'running'])}/{len(self.systems)}")
        
        print("\n📋 สถานะระบบ:")
        for system_name, system_data in self.systems.items():
            status_emoji = "✅" if system_data['status'] == 'running' else "❌"
            uptime = system_data.get('uptime', 0)
            
            print(f"   {status_emoji} {system_name.upper()}: {system_data['status']}")
            print(f"      เวลาทำงาน: {uptime/3600:.1f} ชั่วโมง")
            
            # แสดงข้อมูลเพิ่มเติม
            if 'stats' in system_data:
                stats = system_data['stats']
                if 'fps' in stats:
                    print(f"      FPS: {stats['fps']:.1f}")
                if 'total_detections' in stats:
                    print(f"      การตรวจจับทั้งหมด: {stats['total_detections']}")
        
        print("="*80)
    
    def start_all_systems(self):
        """เริ่มต้นระบบทั้งหมด"""
        logger.info("🚀 เริ่มต้นระบบ AI ทั้งหมด...")
        
        self.running = True
        self.startup_time = time.time()
        
        # 1. เริ่มต้น AI Vision System
        ai_detector = self.start_ai_vision_system()
        
        # 2. เริ่มต้น AI Helper System
        ai_helper = self.start_ai_helper_system(ai_detector)
        
        # 3. เริ่มต้น Performance Booster
        performance_booster = self.start_performance_booster(ai_detector)
        
        # 4. เริ่มต้น AI Chatbot
        ai_chatbot = self.start_ai_chatbot()
        
        # 5. เริ่มต้นระบบ Monitoring
        self.start_monitoring_system()
        
        # แสดงสถานะเริ่มต้น
        running_systems = len([s for s in self.systems.values() if s['status'] == 'running'])
        total_systems = len(self.systems)
        
        logger.info(f"🎯 เริ่มต้นระบบเสร็จสิ้น: {running_systems}/{total_systems} ระบบทำงาน")
        
        return {
            'ai_detector': ai_detector,
            'ai_helper': ai_helper,
            'performance_booster': performance_booster,
            'ai_chatbot': ai_chatbot
        }
    
    def interactive_mode(self):
        """โหมด Interactive"""
        logger.info("🎮 เข้าสู่โหมด Interactive")
        
        while self.running:
            try:
                print("\n🔧 ULTIMATE AI SYSTEM CONTROL PANEL")
                print("1. 📊 แสดงสถานะระบบ")
                print("2. 🎥 ทดสอบ AI Vision")
                print("3. 🤖 สนทนากับ AI")
                print("4. ⚡ ทดสอบประสิทธิภาพ")
                print("5. 🔄 รีสตาร์ทระบบ")
                print("6. 🛑 ปิดระบบ")
                
                choice = input("\n👉 เลือกตัวเลือก (1-6): ").strip()
                
                if choice == "1":
                    self.show_status_dashboard()
                elif choice == "2":
                    self.test_ai_vision()
                elif choice == "3":
                    self.chat_with_ai()
                elif choice == "4":
                    self.test_performance()
                elif choice == "5":
                    self.restart_systems()
                elif choice == "6":
                    logger.info("🛑 ผู้ใช้เลือกปิดระบบ")
                    break
                else:
                    print("❌ ตัวเลือกไม่ถูกต้อง")
                    
            except KeyboardInterrupt:
                logger.info("🛑 รับ Ctrl+C จากผู้ใช้")
                break
            except Exception as e:
                logger.error(f"❌ เกิดข้อผิดพลาดใน interactive mode: {e}")
                time.sleep(1)
    
    def test_ai_vision(self):
        """ทดสอบ AI Vision"""
        if 'ai_vision' not in self.systems:
            print("❌ AI Vision System ไม่พร้อมใช้งาน")
            return
        
        print("🎥 เรียก AI Vision Demo...")
        try:
            from ultimate_ai_demo import demo_ai_vision
            ai_detector = self.systems['ai_vision']['instance']
            ai_helper = self.systems.get('ai_helper', {}).get('instance')
            demo_ai_vision(ai_detector, ai_helper)
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการทดสอบ AI Vision: {e}")
    
    def chat_with_ai(self):
        """สนทนากับ AI"""
        if 'ai_chatbot' not in self.systems:
            print("❌ AI Chatbot ไม่พร้อมใช้งาน")
            return
        
        print("💬 เรียก AI Chatbot Demo...")
        try:
            from ultimate_ai_demo import demo_ai_chatbot
            ai_chatbot = self.systems['ai_chatbot']['instance']
            demo_ai_chatbot(ai_chatbot)
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการเรียก AI Chatbot: {e}")
    
    def test_performance(self):
        """ทดสอบประสิทธิภาพ"""
        if 'ai_vision' not in self.systems or 'performance_booster' not in self.systems:
            print("❌ ระบบที่จำเป็นไม่พร้อมใช้งาน")
            return
        
        print("⚡ เรียกการทดสอบประสิทธิภาพ...")
        try:
            from ultimate_ai_demo import demo_performance_test
            ai_detector = self.systems['ai_vision']['instance']
            performance_booster = self.systems['performance_booster']['instance']
            demo_performance_test(ai_detector, performance_booster)
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการทดสอบประสิทธิภาพ: {e}")
    
    def restart_systems(self):
        """รีสตาร์ทระบบ"""
        logger.info("🔄 รีสตาร์ทระบบ...")
        
        # ปิดระบบปัจจุบัน
        self.shutdown()
        
        # เริ่มใหม่
        time.sleep(2)
        logger.info("🚀 เริ่มต้นระบบใหม่...")
        self.systems.clear()
        self.start_all_systems()
    
    def shutdown(self):
        """ปิดระบบทั้งหมด"""
        logger.info("🛑 เริ่มปิดระบบ Ultimate AI System...")
        
        self.running = False
        
        # ปิดระบบแต่ละตัว
        for system_name, system_data in self.systems.items():
            try:
                logger.info(f"🔄 ปิด {system_name}...")
                
                instance = system_data['instance']
                
                # ปิดระบบตามประเภท
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
                elif hasattr(instance, 'close'):
                    instance.close()
                elif hasattr(instance, 'shutdown'):
                    instance.shutdown()
                
                system_data['status'] = 'stopped'
                logger.info(f"✅ ปิด {system_name} สำเร็จ")
                
            except Exception as e:
                logger.error(f"❌ เกิดข้อผิดพลาดในการปิด {system_name}: {e}")
        
        # เรียก shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"❌ เกิดข้อผิดพลาดใน shutdown handler: {e}")
        
        total_uptime = time.time() - self.startup_time if self.startup_time else 0
        logger.info(f"🏁 ปิดระบบเสร็จสิ้น (เวลาทำงาน: {total_uptime/3600:.1f} ชั่วโมง)")

def main():
    """ฟังก์ชันหลัก"""
    launcher = UltimateAILauncher()
    
    try:
        # แสดง banner
        launcher.startup_banner()
        
        # ตรวจสอบความพร้อม
        if not launcher.check_prerequisites():
            logger.error("❌ ระบบไม่พร้อมใช้งาน")
            return
        
        # เริ่มต้นระบบทั้งหมด
        systems = launcher.start_all_systems()
        
        # แสดงสถานะเริ่มต้น
        launcher.show_status_dashboard()
        
        # เข้าสู่โหมด Interactive
        launcher.interactive_mode()
        
    except KeyboardInterrupt:
        logger.info("🛑 รับ Ctrl+C จากผู้ใช้")
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")
    finally:
        launcher.shutdown()

if __name__ == "__main__":
    main()
