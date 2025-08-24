#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate AI Bird Intelligence System V2.0 - Launcher
====================================================
สำหรับเปิดระบบใหม่ที่ได้รับการปรับปรุงแล้ว
====================================================
"""

import os
import sys
import subprocess
import time

def main():
    """Launch the new Ultimate AI Bird System"""
    print("🦅" + "="*60 + "🦅")
    print("🚀 ULTIMATE AI BIRD INTELLIGENCE SYSTEM V2.0")
    print("🦅" + "="*60 + "🦅")
    print("")
    print("✨ Features:")
    print("   📊 Real-time Statistics Dashboard")
    print("   🎥 Live Video Feed with AI Detection")
    print("   🌍 Bilingual Support (Thai/English)")
    print("   📈 Advanced Analytics & Charts")
    print("   🔔 Smart Alert System")
    print("   📱 Responsive Design")
    print("")
    print("🌐 Starting web server...")
    print("📍 URL: http://localhost:5000")
    print("💡 Press Ctrl+C to stop")
    print("")
    
    try:
        # Change to the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Run the application
        subprocess.run([sys.executable, "app_working.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running application: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print("\n👋 Thank you for using Ultimate AI Bird Intelligence System!")

if __name__ == "__main__":
    main()
