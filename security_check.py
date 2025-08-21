#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 Security Check Script for Swallow AI
ตรวจสอบความปลอดภัยก่อน Git commit
"""

import os
import re
import json
from typing import List, Dict, Any

class SecurityChecker:
    """🔒 ตรวจสอบความปลอดภัยของไฟล์"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 🚨 Pattern ที่อันตราย
        self.dangerous_patterns = [
            # Credentials & Passwords
            r'password\s*=\s*["\'][^"\']{3,}["\']',
            r'passwd\s*=\s*["\'][^"\']{3,}["\']',
            r'api_key\s*=\s*["\'][^"\']{10,}["\']',
            r'secret\s*=\s*["\'][^"\']{10,}["\']',
            r'token\s*=\s*["\'][^"\']{10,}["\']',
            
            # IP Addresses (specific ones)
            r'192\.168\.1\.\d{1,3}',
            r'10\.0\.12\.\d{1,3}',
            
            # Camera credentials
            r'ainok1',
            r'ainok123',
            r'rtsp://[^/]+:[^@]+@',
            
            # Database paths with sensitive info
            r'C:\\[^\\]+\\[^\\]+\\.*\.db',
            r'C:/[^/]+/[^/]+/.*\.db',
        ]
        
        # 📁 ไฟล์ที่ต้องตรวจสอบ
        self.check_extensions = ['.py', '.js', '.json', '.md', '.txt', '.bat', '.sh']
        
        # 📁 ไฟล์ที่ไม่ควร commit
        self.forbidden_files = [
            'config.json',
            'camera_config.json',
            'live_stream_config.json',
            'entrance_config.json',
            'test_video.mp4',
            'yolov8n.pt',
            '.env'
        ]
    
    def scan_file(self, filepath: str) -> List[Dict[str, Any]]:
        """สแกนไฟล์หาข้อมูลอันตราย"""
        issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for line_no, line in enumerate(content.split('\n'), 1):
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            'file': filepath,
                            'line': line_no,
                            'pattern': pattern,
                            'content': line.strip()[:100],
                            'severity': 'HIGH'
                        })
                        
        except (UnicodeDecodeError, FileNotFoundError):
            pass
            
        return issues
    
    def check_directory(self, directory: str = None) -> Dict[str, Any]:
        """ตรวจสอบทั้งไดเรกทอรี"""
        if directory is None:
            directory = self.base_dir
            
        results = {
            'safe_files': [],
            'issues': [],
            'forbidden_found': [],
            'total_files': 0
        }
        
        for root, dirs, files in os.walk(directory):
            # ข้าม directories ที่ไม่ต้องตรวจ
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'backups', 'temp']]
            
            for file in files:
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, directory)
                
                results['total_files'] += 1
                
                # ตรวจสอบไฟล์ที่ห้าม
                if file in self.forbidden_files:
                    results['forbidden_found'].append(relative_path)
                    continue
                
                # ตรวจสอบเฉพาะไฟล์ที่กำหนด
                if any(file.endswith(ext) for ext in self.check_extensions):
                    file_issues = self.scan_file(filepath)
                    if file_issues:
                        results['issues'].extend(file_issues)
                    else:
                        results['safe_files'].append(relative_path)
        
        return results
    
    def generate_safe_files_list(self) -> List[str]:
        """สร้างรายการไฟล์ที่ปลอดภัยสำหรับ commit"""
        safe_files = [
            # Core Python files
            'app_working.py',
            'ultra_smart_ai_agent.py',
            'ai_agent_web.py',
            'swallow_ai_integration.py',
            'test_ai_agent.py',
            
            # Component files
            'advanced_object_detector.py',
            'camera_detector.py',
            'models.py',
            'schemas.py',
            
            # Scripts
            'start_complete_system.bat',
            'install.bat',
            
            # Documentation
            'README.md',
            'COMPLETE_INTEGRATION_REPORT.md',
            'requirements.txt',
            
            # Config (safe versions)
            '.gitignore',
            'security_check.py'
        ]
        
        # ตรวจสอบว่าไฟล์มีอยู่จริง
        existing_files = []
        for file in safe_files:
            if os.path.exists(os.path.join(self.base_dir, file)):
                existing_files.append(file)
        
        return existing_files

def main():
    """ฟังก์ชันหลัก"""
    checker = SecurityChecker()
    
    print("🔒 SWALLOW AI SECURITY CHECK")
    print("=" * 50)
    
    # ตรวจสอบความปลอดภัย
    results = checker.check_directory()
    
    print(f"📁 ตรวจสอบไฟล์ทั้งหมด: {results['total_files']} ไฟล์")
    
    # แสดงไฟล์ที่ห้าม commit
    if results['forbidden_found']:
        print("\n🚨 ไฟล์ที่ห้าม commit:")
        for file in results['forbidden_found']:
            print(f"  ❌ {file}")
    
    # แสดงปัญหาความปลอดภัย
    if results['issues']:
        print(f"\n⚠️ พบปัญหาความปลอดภัย: {len(results['issues'])} ปัญหา")
        for issue in results['issues']:
            print(f"  🔴 {issue['file']}:{issue['line']}")
            print(f"     {issue['content']}")
    
    # แสดงไฟล์ที่ปลอดภัย
    safe_files = checker.generate_safe_files_list()
    print(f"\n✅ ไฟล์ปลอดภัยสำหรับ commit: {len(safe_files)} ไฟล์")
    for file in safe_files:
        print(f"  ✓ {file}")
    
    # สรุป
    print("\n" + "=" * 50)
    if results['issues'] or results['forbidden_found']:
        print("🔴 SECURITY ALERT: พบปัญหาความปลอดภัย!")
        print("⚠️ กรุณาแก้ไขก่อน commit")
        return False
    else:
        print("🟢 SECURITY OK: ปลอดภัยสำหรับ commit")
        return True

if __name__ == "__main__":
    main()
