
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
