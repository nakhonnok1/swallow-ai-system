#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ULTIMATE AI DEMO - ตัวอย่างการใช้งาน AI System ที่สมบูรณ์
แสดงการทำงานของระบบ AI ทั้งหมดร่วมกัน
Version: 1.0 - COMPREHENSIVE AI DEMONSTRATION

🚀 ระบบที่รวมอยู่:
- Ultimate AI Vision System (OpenCV YOLO Detector)
- AI Helper System (Smart Monitoring & Optimization)
- AI Performance Booster (Performance Enhancement)
- Enhanced Ultra Smart AI Agent (Chatbot)
- Real-time Statistics & Analytics
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
import json

def main_ai_demo():
    """การสาธิตระบบ AI หลัก"""
    print("🎯 ULTIMATE AI SYSTEM DEMONSTRATION")
    print("="*80)
    
    # เริ่มต้นระบบ AI ทั้งหมด
    print("🚀 กำลังเริ่มต้นระบบ AI ทั้งหมด...")
    
    # 1. เริ่มต้น AI Vision System
    print("\n1️⃣ เริ่มต้น Ultimate AI Vision System...")
    try:
        from opencv_yolo_detector import OpenCVYOLODetector
        ai_detector = OpenCVYOLODetector()
        
        if ai_detector.available:
            print("✅ AI Vision System พร้อมใช้งาน")
        else:
            print("❌ AI Vision System ไม่พร้อมใช้งาน")
            return
            
    except Exception as e:
        print(f"❌ Error loading AI Vision: {e}")
        return
    
    # 2. เริ่มต้น AI Helper System
    print("\n2️⃣ เริ่มต้น AI Helper System...")
    try:
        from ai_helper_system import get_ai_helper
        ai_helper = get_ai_helper()
        ai_helper.register_ai_system("main_detector", ai_detector)
        print("✅ AI Helper System เชื่อมต่อแล้ว")
    except Exception as e:
        print(f"⚠️ AI Helper System ไม่พร้อม: {e}")
        ai_helper = None
    
    # 3. เริ่มต้น Performance Booster
    print("\n3️⃣ เริ่มต้น AI Performance Booster...")
    try:
        from ai_performance_booster import get_performance_booster
        performance_booster = get_performance_booster()
        performance_booster.optimize_ai_system(ai_detector)
        print("✅ Performance Booster ปรับปรุงระบบแล้ว")
    except Exception as e:
        print(f"⚠️ Performance Booster ไม่พร้อม: {e}")
        performance_booster = None
    
    # 4. เริ่มต้น AI Chatbot
    print("\n4️⃣ เริ่มต้น Enhanced Ultra Smart AI Agent...")
    try:
        from enhanced_ultra_smart_ai_agent import EnhancedUltraSmartAIAgent
        ai_chatbot = EnhancedUltraSmartAIAgent()
        print("✅ AI Chatbot พร้อมใช้งาน")
    except Exception as e:
        print(f"⚠️ AI Chatbot ไม่พร้อม: {e}")
        ai_chatbot = None
    
    print("\n" + "="*80)
    print("🎯 เริ่มการสาธิต AI System...")
    
    # การสาธิตแบบ Interactive
    while True:
        print("\n🔧 เลือกการสาธิต:")
        print("1. 🎥 ทดสอบ AI Vision Detection")
        print("2. 🤖 ทดสอบ AI Chatbot")
        print("3. 📊 แสดง Dashboard & Statistics")
        print("4. ⚡ ทดสอบ Performance")
        print("5. 🔄 ปรับปรุงระบบอัตโนมัติ")
        print("6. 💾 บันทึกข้อมูลและออก")
        
        choice = input("\n👉 เลือกตัวเลือก (1-6): ").strip()
        
        if choice == "1":
            demo_ai_vision(ai_detector, ai_helper)
        elif choice == "2":
            demo_ai_chatbot(ai_chatbot)
        elif choice == "3":
            show_comprehensive_dashboard(ai_helper, performance_booster, ai_detector)
        elif choice == "4":
            demo_performance_test(ai_detector, performance_booster)
        elif choice == "5":
            auto_optimize_systems(ai_helper, performance_booster)
        elif choice == "6":
            save_and_exit(ai_helper, ai_detector)
            break
        else:
            print("❌ ตัวเลือกไม่ถูกต้อง")

def demo_ai_vision(ai_detector, ai_helper):
    """สาธิต AI Vision Detection"""
    print("\n🎥 การสาธิต AI Vision Detection")
    print("-" * 50)
    
    # ตัวเลือกแหล่งข้อมูล
    print("เลือกแหล่งข้อมูล:")
    print("1. กล้อง RTSP")
    print("2. กล้อง USB/Webcam")
    print("3. ไฟล์วิดีโอ")
    print("4. ทดสอบแบบอัตโนมัติ")
    
    source_choice = input("👉 เลือก (1-4): ").strip()
    
    cap = None
    
    if source_choice == "1":
        rtsp_url = input("🔗 ใส่ RTSP URL (หรือกด Enter สำหรับค่าเริ่มต้น): ").strip()
        if not rtsp_url:
            rtsp_url = "rtsp://ainok1:ainok123@192.168.1.100:554/stream1"
        cap = cv2.VideoCapture(rtsp_url)
        
    elif source_choice == "2":
        cap = cv2.VideoCapture(0)
        
    elif source_choice == "3":
        video_path = input("📹 ใส่ path ของไฟล์วิดีโอ: ").strip()
        if Path(video_path).exists():
            cap = cv2.VideoCapture(video_path)
        else:
            print("❌ ไฟล์ไม่พบ")
            return
            
    elif source_choice == "4":
        # ทดสอบอัตโนมัติ
        test_auto_detection(ai_detector)
        return
    
    if not cap or not cap.isOpened():
        print("❌ ไม่สามารถเชื่อมต่อแหล่งข้อมูลได้")
        return
    
    print("✅ เชื่อมต่อสำเร็จ!")
    print("📝 กด 'q' เพื่อออก, 's' เพื่อบันทึกภาพ, 'p' เพื่อพักชั่วคราว")
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # ตรวจจับด้วย AI
            detections = ai_detector.detect_objects(frame, conf_threshold=0.4)
            
            if len(detections) > 0:
                detection_count += 1
                
                # วาดผลลัพธ์
                result_frame = ai_detector.draw_detections(frame, detections)
                
                # แสดงข้อมูลบนหน้าจอ
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                info_text = f"Frame: {frame_count} | Detections: {len(detections)} | FPS: {fps:.1f}"
                cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # แสดงรายการ objects
                y_offset = 60
                for i, det in enumerate(detections):
                    det_text = f"{det['class']}: {det['confidence']:.2f}"
                    cv2.putText(result_frame, det_text, (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('AI Vision Detection', result_frame)
                
                # แสดงข้อมูลใน console
                if frame_count % 30 == 0:  # ทุก 30 เฟรม
                    print(f"📊 Frame {frame_count}: {len(detections)} objects, FPS: {fps:.1f}")
                    for det in detections:
                        print(f"   🎯 {det['class']}: {det['confidence']:.2f}")
            else:
                cv2.imshow('AI Vision Detection', frame)
        
        # จัดการ keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # บันทึกภาพ
            timestamp = int(time.time())
            filename = f"ai_detection_{timestamp}.jpg"
            if len(detections) > 0:
                cv2.imwrite(filename, result_frame)
            else:
                cv2.imwrite(filename, frame)
            print(f"💾 บันทึกภาพ: {filename}")
        elif key == ord('p'):
            paused = not paused
            print(f"⏸️ {'พัก' if paused else 'เล่น'}การสาธิต")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # สรุปผล
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
    
    print(f"\n📈 สรุปการสาธิต AI Vision:")
    print(f"   🎬 เฟรมทั้งหมด: {frame_count}")
    print(f"   🎯 เฟรมที่มีการตรวจจับ: {detection_count}")
    print(f"   📊 อัตราการตรวจจับ: {detection_rate:.1f}%")
    print(f"   ⚡ FPS เฉลี่ย: {avg_fps:.1f}")
    print(f"   ⏱️ เวลาทั้งหมด: {total_time:.1f} วินาที")

def demo_ai_chatbot(ai_chatbot):
    """สาธิต AI Chatbot"""
    if not ai_chatbot:
        print("❌ AI Chatbot ไม่พร้อมใช้งาน")
        return
    
    print("\n🤖 การสาธิต AI Chatbot")
    print("-" * 50)
    print("💬 พิมพ์ข้อความเพื่อสนทนากับ AI")
    print("📝 พิมพ์ 'exit' เพื่อออก")
    print("📝 พิมพ์ 'help' เพื่อดูคำสั่งพิเศษ")
    
    while True:
        user_input = input("\n👤 คุณ: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'help':
            print("🔧 คำสั่งพิเศษ:")
            print("   'status' - ตรวจสอบสถานะระบบ")
            print("   'stats' - แสดงสถิติ")
            print("   'analyze' - วิเคราะห์ข้อมูล")
            print("   'clear' - ล้างประวัติการสนทนา")
            continue
        elif user_input.lower() == 'status':
            print("🔍 ตรวจสอบสถานะระบบ...")
            # แสดงสถานะระบบ
            continue
        elif user_input.lower() == 'stats':
            print("📊 แสดงสถิติระบบ...")
            # แสดงสถิติ
            continue
        elif user_input.lower() == 'clear':
            ai_chatbot.conversation_history.clear()
            print("🧹 ล้างประวัติการสนทนาแล้ว")
            continue
        
        if not user_input:
            continue
        
        try:
            # ส่งข้อความไปยัง AI
            print("🤔 AI กำลังคิด...")
            response = ai_chatbot.generate_response(user_input)
            print(f"🤖 AI: {response}")
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")

def show_comprehensive_dashboard(ai_helper, performance_booster, ai_detector):
    """แสดง Dashboard แบบครบถ้วน"""
    print("\n📊 COMPREHENSIVE AI DASHBOARD")
    print("="*80)
    
    # 1. System Overview
    print("🖥️ SYSTEM OVERVIEW")
    print("-"*40)
    
    if ai_detector:
        model_info = ai_detector.get_model_info()
        print(f"🤖 AI Model: {model_info['version']}")
        print(f"⚙️ Backend: {model_info['backend']}")
        print(f"🎯 Target: {model_info['target']}")
        print(f"📏 Input Size: {model_info['input_size']}")
        print(f"📚 Classes: {model_info['classes_count']}")
    
    # 2. Detection Statistics
    if ai_detector:
        print(f"\n🎯 DETECTION STATISTICS")
        print("-"*40)
        stats = ai_detector.get_detection_stats()
        print(f"🔍 Total Detections: {stats['total_detections']}")
        print(f"🐦 Birds: {stats['birds_detected']}")
        print(f"👤 Persons: {stats['persons_detected']}")
        print(f"🐾 Animals: {stats['animals_detected']}")
        print(f"🚗 Vehicles: {stats['vehicles_detected']}")
        print(f"⚡ Current FPS: {stats['fps']:.1f}")
        print(f"⏱️ Processing Time: {stats['processing_time']*1000:.1f}ms")
    
    # 3. AI Helper Dashboard
    if ai_helper:
        print(f"\n🤖 AI HELPER DASHBOARD")
        print("-"*40)
        dashboard = ai_helper.get_system_dashboard()
        
        overview = dashboard['overview']
        print(f"🎛️ Active Systems: {overview['active_systems']}")
        print(f"⏰ Uptime: {overview['total_uptime']/3600:.1f} hours")
        print(f"📊 Status: {overview['status']}")
        
        for system_id, data in dashboard['systems'].items():
            print(f"\n   🔧 {system_id}:")
            print(f"      Status: {data['status']}")
            print(f"      FPS: {data['fps']:.1f}")
            print(f"      Accuracy: {data['accuracy']:.2f}")
            print(f"      CPU: {data['cpu_usage']:.1f}%")
            print(f"      Memory: {data['memory_usage']:.1f}%")
    
    # 4. Performance Report
    if performance_booster:
        print(f"\n⚡ PERFORMANCE REPORT")
        print("-"*40)
        report = performance_booster.get_performance_report()
        
        profile = report['system_profile']
        print(f"💻 CPU Cores: {profile['cpu_cores']}")
        print(f"🧠 Memory: {profile['memory_gb']:.1f} GB")
        print(f"🚀 GPU: {'Available' if profile['gpu_available'] else 'Not Available'}")
        print(f"⚙️ Processing Mode: {profile['processing_mode']}")
        
        metrics = report['performance_metrics']
        print(f"\n📈 Performance Improvements:")
        print(f"   FPS: +{metrics['fps_improvement']:.1f}%")
        print(f"   Memory: +{metrics['memory_saved']:.1f}%")
        print(f"   CPU: +{metrics['cpu_optimization']:.1f}%")
        print(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
        print(f"   Total Speedup: +{metrics['total_speedup']:.1f}%")
    
    # 5. Recommendations
    if ai_helper:
        print(f"\n💡 RECOMMENDATIONS")
        print("-"*40)
        dashboard = ai_helper.get_system_dashboard()
        
        for system_id, recommendations in dashboard['recommendations'].items():
            if recommendations:
                print(f"📝 {system_id}:")
                for rec in recommendations[:3]:  # แสดง 3 ข้อแรก
                    priority_emoji = "🔴" if rec['priority'] >= 4 else "🟡" if rec['priority'] >= 3 else "🟢"
                    print(f"   {priority_emoji} {rec['description']}")
                    print(f"      Expected Improvement: +{rec['improvement']:.1f}%")
    
    print("\n" + "="*80)

def demo_performance_test(ai_detector, performance_booster):
    """สาธิตการทดสอบประสิทธิภาพ"""
    print("\n⚡ การทดสอบประสิทธิภาพ AI")
    print("-" * 50)
    
    if not ai_detector or not ai_detector.available:
        print("❌ AI Detector ไม่พร้อมใช้งาน")
        return
    
    print("🧪 เริ่มการทดสอบประสิทธิภาพ...")
    
    # สร้างภาพทดสอบ
    test_frames = []
    for i in range(10):
        # สร้างภาพสุ่ม
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_frames.append(frame)
    
    print(f"📸 สร้างภาพทดสอบ {len(test_frames)} ภาพ")
    
    # ทดสอบก่อนปรับปรุง
    print("\n1️⃣ ทดสอบก่อนปรับปรุง...")
    times_before = []
    for i, frame in enumerate(test_frames):
        start_time = time.time()
        detections = ai_detector.detect_objects(frame)
        end_time = time.time()
        processing_time = end_time - start_time
        times_before.append(processing_time)
        print(f"   เฟรม {i+1}: {processing_time*1000:.1f}ms ({len(detections)} objects)")
    
    avg_time_before = sum(times_before) / len(times_before)
    avg_fps_before = 1.0 / avg_time_before
    
    # ปรับปรุงประสิทธิภาพ
    print("\n2️⃣ ปรับปรุงประสิทธิภาพ...")
    if performance_booster:
        performance_booster.optimize_ai_system(ai_detector)
        print("✅ ปรับปรุงระบบแล้ว")
    
    # ทดสอบหลังปรับปรุง
    print("\n3️⃣ ทดสอบหลังปรับปรุง...")
    times_after = []
    for i, frame in enumerate(test_frames):
        start_time = time.time()
        detections = ai_detector.detect_objects(frame)
        end_time = time.time()
        processing_time = end_time - start_time
        times_after.append(processing_time)
        print(f"   เฟรม {i+1}: {processing_time*1000:.1f}ms ({len(detections)} objects)")
    
    avg_time_after = sum(times_after) / len(times_after)
    avg_fps_after = 1.0 / avg_time_after
    
    # สรุปผล
    improvement = ((avg_time_before - avg_time_after) / avg_time_before) * 100
    fps_improvement = ((avg_fps_after - avg_fps_before) / avg_fps_before) * 100
    
    print(f"\n📊 สรุปผลการทดสอบประสิทธิภาพ:")
    print(f"   ⏱️ เวลาเฉลี่ยก่อนปรับปรุง: {avg_time_before*1000:.1f}ms")
    print(f"   ⏱️ เวลาเฉลี่ยหลังปรับปรุง: {avg_time_after*1000:.1f}ms")
    print(f"   📈 การปรับปรุงเวลา: {improvement:.1f}%")
    print(f"   ⚡ FPS ก่อนปรับปรุง: {avg_fps_before:.1f}")
    print(f"   ⚡ FPS หลังปรับปรุง: {avg_fps_after:.1f}")
    print(f"   📈 การปรับปรุง FPS: {fps_improvement:.1f}%")
    
    if improvement > 0:
        print("✅ ประสิทธิภาพดีขึ้น!")
    else:
        print("⚠️ ไม่มีการปรับปรุงประสิทธิภาพ")

def auto_optimize_systems(ai_helper, performance_booster):
    """ปรับปรุงระบบอัตโนมัติ"""
    print("\n🔄 การปรับปรุงระบบอัตโนมัติ")
    print("-" * 50)
    
    optimizations_count = 0
    
    if ai_helper:
        print("🤖 ปรับปรุงระบบด้วย AI Helper...")
        for system_id in ai_helper.active_ai_systems:
            if ai_helper.optimize_system(system_id, auto_apply=True):
                optimizations_count += 1
                print(f"✅ ปรับปรุง {system_id} สำเร็จ")
    
    if performance_booster:
        print("⚡ ปรับปรุงประสิทธิภาพ...")
        # ปรับปรุงระบบทั้งหมดที่ลงทะเบียน
        if ai_helper:
            for system_id, system_data in ai_helper.active_ai_systems.items():
                ai_system = system_data['instance']
                if performance_booster.optimize_ai_system(ai_system):
                    optimizations_count += 1
                    print(f"✅ เพิ่มประสิทธิภาพ {system_id} สำเร็จ")
    
    print(f"\n🎯 สรุป: ปรับปรุงระบบ {optimizations_count} ระบบ")

def test_auto_detection(ai_detector):
    """ทดสอบการตรวจจับอัตโนมัติ"""
    print("🤖 การทดสอบอัตโนมัติ")
    
    # สร้างภาพทดสอบ
    test_images = []
    
    # ภาพพื้นหลัง
    background = np.zeros((480, 640, 3), dtype=np.uint8)
    test_images.append(("Background", background))
    
    # ภาพสุ่ม
    random_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_images.append(("Random", random_image))
    
    # ภาพที่มีรูปร่าง
    shape_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(shape_image, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.circle(shape_image, (400, 300), 50, (255, 0, 0), -1)
    test_images.append(("Shapes", shape_image))
    
    total_detections = 0
    
    for name, image in test_images:
        print(f"\n🔍 ทดสอบภาพ: {name}")
        
        detections = ai_detector.detect_objects(image, conf_threshold=0.3)
        total_detections += len(detections)
        
        print(f"   พบ {len(detections)} objects")
        for det in detections:
            print(f"   - {det['class']}: {det['confidence']:.2f}")
        
        # แสดงภาพ (optional)
        if len(detections) > 0:
            result_image = ai_detector.draw_detections(image, detections)
            cv2.imshow(f'Test: {name}', result_image)
            cv2.waitKey(1000)  # แสดง 1 วินาที
    
    cv2.destroyAllWindows()
    print(f"\n📊 สรุป: พบ objects ทั้งหมด {total_detections} รายการ")

def save_and_exit(ai_helper, ai_detector):
    """บันทึกข้อมูลและออก"""
    print("\n💾 บันทึกข้อมูลและออกจากระบบ")
    print("-" * 50)
    
    # สร้างโฟลเดอร์สำหรับบันทึก
    save_dir = Path("ai_demo_results")
    save_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    
    # บันทึกสถิติ AI Detector
    if ai_detector:
        stats = ai_detector.get_detection_stats()
        model_info = ai_detector.get_model_info()
        
        ai_data = {
            'timestamp': timestamp,
            'statistics': stats,
            'model_info': model_info,
            'demo_completed': True
        }
        
        with open(save_dir / f"ai_detector_stats_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(ai_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ บันทึกข้อมูล AI Detector แล้ว")
    
    # บันทึกข้อมูล AI Helper
    if ai_helper:
        dashboard = ai_helper.get_system_dashboard()
        
        with open(save_dir / f"ai_helper_dashboard_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)
        
        print(f"✅ บันทึกข้อมูล AI Helper แล้ว")
        
        # ยกเลิกการลงทะเบียน
        ai_helper.unregister_ai_system("main_detector")
    
    print(f"📁 ข้อมูลถูกบันทึกใน: {save_dir}")
    print("🎯 การสาธิต Ultimate AI System เสร็จสิ้น!")
    print("ขอบคุณที่ใช้งาน Ultimate AI System! 🚀")

if __name__ == "__main__":
    try:
        main_ai_demo()
    except KeyboardInterrupt:
        print("\n⏹️ การสาธิตถูกหยุดโดยผู้ใช้")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    finally:
        print("👋 ขอบคุณที่ใช้งาน Ultimate AI Demo!")
