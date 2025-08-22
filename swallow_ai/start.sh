#!/bin/bash
echo "================================================"
echo " AI Bird Tracking System - Quick Start"
echo "================================================"
echo

echo "🔍 Checking Python environment..."
cd "c:\Nakhonnok\swallow_ai"

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv"
    read -p "Press enter to exit..."
    exit 1
fi

echo "✅ Virtual environment found"

echo
echo "📦 Activating virtual environment..."
source venv/Scripts/activate

echo
echo "🚀 Starting AI Bird Tracking System..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo "🔧 System Management: http://localhost:5000/system"
echo "❤️  Health Check: http://localhost:5000/health/detailed"
echo
echo "💡 Press Ctrl+C to stop the system"
echo

python app_master.py

echo
echo "👋 System stopped. Press any key to exit..."
read -p ""
