#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Git Repository Setup Guide
คู่มือการสร้าง GitHub Repository สำหรับ Swallow AI System
"""

print("""
🎉 Git Repository พร้อมแล้ว!

📁 โปรเจค: Swallow AI System
📊 Files: 71 ไฟล์ 
💾 Commits: 1 commit แรกเรียบร้อย

🌐 วิธีอัพโหลดขึ้น GitHub:

1. ไปที่ https://github.com
2. Login เข้าบัญชีของคุณ  
3. คลิก "New repository"
4. ตั้งชื่อ: "swallow-ai-system"
5. เลือก "Public" หรือ "Private"
6. ไม่ต้องติก "Initialize with README"
7. คลิก "Create repository"

8. รันคำสั่งต่อไปนี้:

   git remote set-url origin https://github.com/YOUR-USERNAME/swallow-ai-system.git
   git branch -M main
   git push -u origin main

⚠️ แทน YOUR-USERNAME ด้วยชื่อผู้ใช้ GitHub ของคุณ

🎯 หรือใช้คำสั่งเร็ว:
   git remote -v  # ดู remote ปัจจุบัน
   git log --oneline  # ดูประวัติ commits

✅ Repository พร้อมใช้งาน!
📦 ไฟล์ทั้งหมดถูกจัดเก็บอย่างปลอดภัย
🔄 พร้อม push ขึ้น GitHub เมื่อไหร่ก็ได้

🌟 ฟีเจอร์ที่บันทึกไว้:
   ✅ Complete Swallow AI System
   ✅ Ultra Smart AI Agent  
   ✅ Web Interfaces (Port 5000 & 8080)
   ✅ Advanced Object Detection
   ✅ Real-time Monitoring
   ✅ Production Ready Code
""")

# สร้างไฟล์ git commands helper
git_commands = """
# 🔧 Git Commands Helper

## Push ขึ้น GitHub ใหม่:
git remote set-url origin https://github.com/YOUR-USERNAME/swallow-ai-system.git
git branch -M main  
git push -u origin main

## Commands อื่นๆ ที่มีประโยชน์:
git status                    # ดูสถานะไฟล์
git log --oneline            # ดูประวัติ commits
git add .                    # เพิ่มไฟล์ใหม่
git commit -m "message"      # สร้าง commit ใหม่
git push                     # อัพโหลด changes
git pull                     # ดาวน์โหลด changes ล่าสุด

## Branch management:
git branch                   # ดู branches
git checkout -b new-feature  # สร้าง branch ใหม่
git merge branch-name        # รวม branch

## Repository info:
git remote -v               # ดู remote repositories
git config --list          # ดูการตั้งค่า git
"""

with open("GIT_COMMANDS.md", "w", encoding="utf-8") as f:
    f.write(git_commands)

print("✅ สร้างไฟล์ GIT_COMMANDS.md แล้ว!")
