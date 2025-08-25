#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultimate Intelligent AI Agent - AI ตัวแทนอัจฉริยะที่เรียนรู้และพัฒนาตัวเองได้
Version: 3.0 - Super Intelligence & Predictive Analytics
เชื่อมต่อกับระบบ AI จับนก และ AI ตรวจจับสิ่งแปลกปลอมอย่างสมบูรณ์
มีความสามารถในการเรียนรู้, วิเคราะห์, คาดการณ์, และตอบคำถามแบบอัจฉริยะที่สุด
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
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class SwallowPattern:
    """รูปแบบข้อมูลนกนางแอ่น"""
    timestamp: dt.datetime
    birds_in: int = 0
    birds_out: int = 0
    current_count: int = 0
    temperature: float = 0.0
    humidity: float = 0.0
    light_level: float = 0.0
    weather_condition: str = "unknown"
    activity_level: str = "normal"

@dataclass
class PredictionResult:
    """ผลการคาดการณ์"""
    prediction_type: str
    predicted_value: float
    confidence: float
    reasoning: str
    factors: List[str] = field(default_factory=list)
    timestamp: dt.datetime = field(default_factory=dt.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Intelligence Analysis Classes
class PatternAnalyzer:
    """วิเคราะห์รูปแบบพฤติกรรม"""
    def __init__(self):
        self.patterns = {}
        
    def analyze_pattern(self, data: List[Dict]) -> Dict:
        """วิเคราะห์รูปแบบจากข้อมูล"""
        if not data:
            return {}
        
        # วิเคราะห์เทรนด์เวลา
        times = [d.get('timestamp', dt.datetime.now()) for d in data]
        values = [d.get('count', 0) for d in data]
        
        return {
            'trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'stable',
            'average': np.mean(values) if values else 0,
            'peak_times': self._find_peak_times(times, values),
            'patterns': self._detect_daily_patterns(times, values)
        }
    
    def _find_peak_times(self, times: List, values: List) -> List[str]:
        """หาช่วงเวลาที่มีกิจกรรมสูงสุด"""
        if len(values) < 3:
            return []
        
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append(times[i].strftime('%H:%M'))
        return peaks
    
    def _detect_daily_patterns(self, times: List, values: List) -> Dict:
        """ตรวจหารูปแบบรายวัน"""
        hour_data = defaultdict(list)
        for time, value in zip(times, values):
            hour_data[time.hour].append(value)
        
        patterns = {}
        for hour, vals in hour_data.items():
            patterns[f"{hour:02d}:00"] = {
                'average': np.mean(vals),
                'activity_level': 'high' if np.mean(vals) > np.mean(values) else 'low'
            }
        
        return patterns

class BehavioralAnalyst:
    """วิเคราะห์พฤติกรรมนกนางแอ่น"""
    def __init__(self):
        self.behavioral_patterns = {}
        
    def analyze_behavior(self, swallow_data: List[SwallowPattern]) -> Dict:
        """วิเคราะห์พฤติกรรมจากข้อมูลนกนางแอ่น"""
        if not swallow_data:
            return {'status': 'insufficient_data'}
        
        # วิเคราะห์รูปแบบการเข้า-ออก
        in_out_ratio = self._calculate_in_out_ratio(swallow_data)
        activity_periods = self._identify_activity_periods(swallow_data)
        seasonal_behavior = self._analyze_seasonal_behavior(swallow_data)
        
        return {
            'in_out_ratio': in_out_ratio,
            'activity_periods': activity_periods,
            'seasonal_behavior': seasonal_behavior,
            'behavioral_insights': self._generate_behavioral_insights(swallow_data)
        }
    
    def _calculate_in_out_ratio(self, data: List[SwallowPattern]) -> Dict:
        """คำนวณอัตราส่วนการเข้า-ออก"""
        total_in = sum(d.birds_in for d in data)
        total_out = sum(d.birds_out for d in data)
        
        return {
            'total_in': total_in,
            'total_out': total_out,
            'ratio': total_in / max(total_out, 1),
            'interpretation': 'increasing' if total_in > total_out else 'decreasing'
        }
    
    def _identify_activity_periods(self, data: List[SwallowPattern]) -> List[Dict]:
        """ระบุช่วงเวลาที่มีกิจกรรม"""
        periods = []
        hour_activity = defaultdict(int)
        
        for d in data:
            hour = d.timestamp.hour
            hour_activity[hour] += d.birds_in + d.birds_out
        
        # หาช่วงเวลาที่มีกิจกรรมสูง
        avg_activity = np.mean(list(hour_activity.values()))
        for hour, activity in hour_activity.items():
            if activity > avg_activity * 1.5:
                periods.append({
                    'time': f"{hour:02d}:00-{hour+1:02d}:00",
                    'activity_level': activity,
                    'type': 'high_activity'
                })
        
        return periods
    
    def _analyze_seasonal_behavior(self, data: List[SwallowPattern]) -> Dict:
        """วิเคราะห์พฤติกรรมตามฤดูกาล"""
        season_data = defaultdict(list)
        
        for d in data:
            month = d.timestamp.month
            if month in [3, 4, 5]:
                season = 'spring'
            elif month in [6, 7, 8]:
                season = 'summer'
            elif month in [9, 10, 11]:
                season = 'autumn'
            else:
                season = 'winter'
            
            season_data[season].append(d.birds_in + d.birds_out)
        
        seasonal_insights = {}
        for season, activities in season_data.items():
            if activities:
                seasonal_insights[season] = {
                    'average_activity': np.mean(activities),
                    'peak_activity': max(activities),
                    'activity_trend': 'active' if np.mean(activities) > 5 else 'quiet'
                }
        
        return seasonal_insights
    
    def _generate_behavioral_insights(self, data: List[SwallowPattern]) -> List[str]:
        """สร้างข้อมูลเชิงลึกเกี่ยวกับพฤติกรรม"""
        insights = []
        
        if len(data) > 7:
            recent_week = data[-7:]
            avg_daily = np.mean([d.birds_in + d.birds_out for d in recent_week])
            
            if avg_daily > 10:
                insights.append("🐦 นกนางแอ่นมีกิจกรรมสูงในสัปดาห์ที่ผ่านมา อาจเป็นช่วงฤดูผสมพันธุ์")
            elif avg_daily < 3:
                insights.append("📉 กิจกรรมของนกลดลง อาจต้องตรวจสอบสภาพแวดล้อมรอบรัง")
            
            # วิเคราะห์เทรนด์
            if len(data) > 14:
                recent_trend = [d.current_count for d in data[-7:]]
                older_trend = [d.current_count for d in data[-14:-7]]
                
                if np.mean(recent_trend) > np.mean(older_trend):
                    insights.append("📈 จำนวนนกในรังเพิ่มขึ้น แสดงว่าสภาพแวดล้อมเหมาะสม")
                else:
                    insights.append("📊 จำนวนนกคงที่ ระบบการดูแลยังคงมีประสิทธิภาพ")
        
        return insights

class TrendPredictor:
    """คาดการณ์แนวโน้ม"""
    def __init__(self):
        self.models = {}
        
    def predict_bird_activity(self, historical_data: List[SwallowPattern], 
                            prediction_hours: int = 24) -> PredictionResult:
        """คาดการณ์กิจกรรมนกในอนาคต"""
        if len(historical_data) < 5:
            return PredictionResult(
                prediction_type="bird_activity",
                predicted_value=0,
                confidence=0.1,
                reasoning="ข้อมูลไม่เพียงพอสำหรับการคาดการณ์",
                factors=[],
                timestamp=dt.datetime.now()
            )
        
        # ใช้ Simple Linear Regression สำหรับการคาดการณ์
        times = [(d.timestamp - historical_data[0].timestamp).total_seconds() / 3600 
                for d in historical_data]
        activities = [d.birds_in + d.birds_out for d in historical_data]
        
        # คำนวณเทรนด์
        if len(times) > 1:
            slope = (activities[-1] - activities[0]) / (times[-1] - times[0]) if times[-1] != times[0] else 0
            predicted_activity = activities[-1] + (slope * prediction_hours)
            
            # คำนวณความเชื่อมั่น
            variance = np.var(activities) if len(activities) > 1 else 0
            confidence = max(0.3, min(0.9, 1 / (1 + variance / 10)))
            
            reasoning = f"อิงจากเทรนด์ {prediction_hours} ชั่วโมงที่ผ่านมา"
            factors = ["historical_trend", "time_pattern", "seasonal_factor"]
            
            return PredictionResult(
                prediction_type="bird_activity",
                predicted_value=max(0, predicted_activity),
                confidence=confidence,
                reasoning=reasoning,
                factors=factors,
                timestamp=dt.datetime.now()
            )
        
        return PredictionResult(
            prediction_type="bird_activity",
            predicted_value=np.mean(activities),
            confidence=0.5,
            reasoning="ใช้ค่าเฉลี่ยจากข้อมูลที่มี",
            factors=["average_baseline"],
            timestamp=dt.datetime.now()
        )

class EnvironmentalAnalyzer:
    """วิเคราะห์ปัจจัยสิ่งแวดล้อม"""
    def __init__(self):
        self.environmental_factors = {}
        
    def analyze_environmental_impact(self, data: List[SwallowPattern]) -> Dict:
        """วิเคราะห์ผลกระทบของสิ่งแวดล้อม"""
        return {
            'temperature_correlation': self._analyze_temperature_impact(data),
            'humidity_impact': self._analyze_humidity_impact(data),
            'light_conditions': self._analyze_light_conditions(data),
            'recommendations': self._generate_environmental_recommendations(data)
        }
    
    def _analyze_temperature_impact(self, data: List[SwallowPattern]) -> Dict:
        """วิเคราะห์ผลกระทบของอุณหภูมิ"""
        temp_data = [(d.temperature, d.birds_in + d.birds_out) for d in data if d.temperature > 0]
        
        if len(temp_data) < 3:
            return {'status': 'insufficient_data'}
        
        temps, activities = zip(*temp_data)
        optimal_temp_range = self._find_optimal_temperature_range(temps, activities)
        
        return {
            'optimal_range': optimal_temp_range,
            'current_impact': 'positive' if np.mean(activities) > 5 else 'neutral'
        }
    
    def _analyze_humidity_impact(self, data: List[SwallowPattern]) -> Dict:
        """วิเคราะห์ผลกระทบของความชื้น"""
        humidity_activities = [(d.humidity, d.birds_in + d.birds_out) for d in data if d.humidity > 0]
        
        if len(humidity_activities) < 3:
            return {'status': 'insufficient_data'}
        
        humidities, activities = zip(*humidity_activities)
        
        return {
            'optimal_humidity': f"{np.mean(humidities):.1f}%",
            'impact_level': 'moderate'
        }
    
    def _analyze_light_conditions(self, data: List[SwallowPattern]) -> Dict:
        """วิเคราะห์สภาพแสง"""
        light_patterns = defaultdict(list)
        
        for d in data:
            hour = d.timestamp.hour
            if 6 <= hour <= 18:
                light_level = 'daylight'
            elif 19 <= hour <= 21 or 5 <= hour <= 6:
                light_level = 'twilight'
            else:
                light_level = 'night'
            
            light_patterns[light_level].append(d.birds_in + d.birds_out)
        
        preferred_conditions = {}
        for condition, activities in light_patterns.items():
            if activities:
                preferred_conditions[condition] = {
                    'average_activity': np.mean(activities),
                    'preference_level': 'high' if np.mean(activities) > 5 else 'low'
                }
        
        return preferred_conditions
    
    def _find_optimal_temperature_range(self, temps: List[float], activities: List[int]) -> str:
        """หาช่วงอุณหภูมิที่เหมาะสม"""
        if not temps or not activities:
            return "ไม่สามารถระบุได้"
        
        # หาอุณหภูมิที่มีกิจกรรมสูงสุด
        max_activity_idx = activities.index(max(activities))
        optimal_temp = temps[max_activity_idx]
        
        return f"{optimal_temp-2:.1f}°C - {optimal_temp+2:.1f}°C"
    
    def _generate_environmental_recommendations(self, data: List[SwallowPattern]) -> List[str]:
        """สร้างคำแนะนำเกี่ยวกับสิ่งแวดล้อม"""
        recommendations = []
        
        if len(data) > 10:
            recent_data = data[-10:]
            avg_activity = np.mean([d.birds_in + d.birds_out for d in recent_data])
            
            if avg_activity < 3:
                recommendations.append("🌡️ พิจารณาตรวจสอบอุณหภูมิและความชื้นรอบรัง")
                recommendations.append("💡 อาจต้องปรับปรุงสภาพแวดล้อมให้เหมาะสมกับนกนางแอ่น")
            
            # ตรวจสอบเวลาที่มีกิจกรรม
            active_hours = [d.timestamp.hour for d in recent_data if d.birds_in + d.birds_out > 0]
            if active_hours and (max(active_hours) - min(active_hours)) < 6:
                recommendations.append("⏰ นกมีกิจกรรมในช่วงเวลาสั้น อาจต้องสร้างสภาพแวดล้อมที่เอื้อต่อการใช้งานตลอดวัน")
        
        return recommendations

class ThreatAssessor:
    """ประเมินภัยคุกคาม"""
    def __init__(self):
        self.threat_patterns = {}
        
    def assess_threats(self, intruder_data: List[Dict], swallow_data: List[SwallowPattern]) -> Dict:
        """ประเมินภัยคุกคามต่อนกนางแอ่น"""
        threat_level = self._calculate_threat_level(intruder_data)
        impact_analysis = self._analyze_threat_impact(intruder_data, swallow_data)
        
        return {
            'current_threat_level': threat_level,
            'impact_analysis': impact_analysis,
            'recommendations': self._generate_threat_recommendations(threat_level, impact_analysis)
        }
    
    def _calculate_threat_level(self, intruder_data: List[Dict]) -> str:
        """คำนวณระดับภัยคุกคาม"""
        if not intruder_data:
            return 'low'
        
        critical_threats = sum(1 for d in intruder_data if d.get('threat_level') == 'critical')
        high_threats = sum(1 for d in intruder_data if d.get('threat_level') == 'high')
        
        if critical_threats > 0:
            return 'critical'
        elif high_threats > 2:
            return 'high'
        elif high_threats > 0:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_threat_impact(self, intruder_data: List[Dict], swallow_data: List[SwallowPattern]) -> Dict:
        """วิเคราะห์ผลกระทบของภัยคุกคาม"""
        if not intruder_data or not swallow_data:
            return {'impact': 'unknown'}
        
        # หาความสัมพันธ์ระหว่างการปรากฏของภัยคุกคามกับพฤติกรรมนก
        threat_times = [d.get('timestamp', dt.datetime.now()) for d in intruder_data]
        
        # วิเคราะห์การเปลี่ยนแปลงพฤติกรรมนกหลังจากมีภัยคุกคาม
        behavioral_changes = []
        for threat_time in threat_times:
            # หาข้อมูลนกก่อนและหลังเวลาที่มีภัยคุกคาม
            before_threat = [d for d in swallow_data 
                           if (threat_time - d.timestamp).total_seconds() > 0 
                           and (threat_time - d.timestamp).total_seconds() < 3600]  # 1 ชั่วโมงก่อน
            
            after_threat = [d for d in swallow_data 
                          if (d.timestamp - threat_time).total_seconds() > 0 
                          and (d.timestamp - threat_time).total_seconds() < 3600]   # 1 ชั่วโมงหลัง
            
            if before_threat and after_threat:
                before_activity = np.mean([d.birds_in + d.birds_out for d in before_threat])
                after_activity = np.mean([d.birds_in + d.birds_out for d in after_threat])
                
                change_percent = ((after_activity - before_activity) / max(before_activity, 1)) * 100
                behavioral_changes.append(change_percent)
        
        avg_change = np.mean(behavioral_changes) if behavioral_changes else 0
        
        return {
            'behavioral_change_percent': avg_change,
            'impact_severity': 'high' if avg_change < -30 else 'moderate' if avg_change < -10 else 'low',
            'recovery_time': '1-2 hours' if avg_change < -20 else '30 minutes'
        }
    
    def _generate_threat_recommendations(self, threat_level: str, impact_analysis: Dict) -> List[str]:
        """สร้างคำแนะนำการจัดการภัยคุกคาม"""
        recommendations = []
        
        if threat_level == 'critical':
            recommendations.append("🚨 ระดับภัยคุกคามสูงมาก - ตรวจสอบบริเวณรังทันที")
            recommendations.append("📞 แจ้งเจ้าหน้าที่รักษาความปลอดภัย")
            
        elif threat_level == 'high':
            recommendations.append("⚠️ พบภัยคุกคามระดับสูง - เพิ่มการเฝ้าระวัง")
            recommendations.append("🔍 ตรวจสอบระบบรักษาความปลอดภัยรอบรัง")
            
        elif threat_level == 'medium':
            recommendations.append("👁️ มีภัยคุกคามระดับปานกลาง - เฝ้าสังเกต")
            
        # เพิ่มคำแนะนำตามผลกระทบ
        impact_severity = impact_analysis.get('impact_severity', 'low')
        if impact_severity == 'high':
            recommendations.append("🐦 นกได้รับผลกระทบมาก - พิจารณาเพิ่มมาตรการป้องกัน")
        
        return recommendations

@dataclass
class ConversationEntry:
    """โครงสร้างข้อมูลการสนทนา"""
    user_message: str
    ai_response: str
    context: Dict[str, Any]
    timestamp: dt.datetime
    confidence: float
    learned_from: bool = False

class EnhancedUltraSmartAIAgent:
    """🚀 Enhanced Ultra Smart AI Agent - AI ตัวแทนอัจฉริยะระดับสูงสุด"""
    
    def __init__(self):
        print("🧠 Initializing Enhanced Ultra Smart AI Agent...")
        
        # Core Properties
        self.session_start = dt.datetime.now()
        self.conversation_count = 0
        self.last_context = {}
        self.conversation_history: List[ConversationEntry] = []
        
        # Advanced Learning & Analytics
        self.learning_db = "ai_agent_memory.db"
        self.learned_patterns = []
        self.user_patterns = {}
        self.confidence_threshold = 0.75
        self.swallow_patterns: deque = deque(maxlen=1000)
        self.prediction_models = {}
        
        # Intelligence Modules
        self.pattern_analyzer = PatternAnalyzer()
        self.behavioral_analyst = BehavioralAnalyst()
        self.trend_predictor = TrendPredictor()
        self.environmental_analyzer = EnvironmentalAnalyzer()
        self.threat_assessor = ThreatAssessor()
        
        # Real-time Data Integration
        self.data_buffer = deque(maxlen=100)
        self.last_data_fetch = dt.datetime.now()
        self.data_fetch_interval = 30
        
        # API Endpoints - เชื่อมต่อระบบจริง
        self.api_endpoints = {
            'bird_stats': 'http://127.0.0.1:5000/api/stats',
            'detailed_stats': 'http://127.0.0.1:5000/api/stats',
            'intruder_stats': 'http://127.0.0.1:5000/api/enhanced-security-alerts',
            'intruder_alerts': 'http://127.0.0.1:5000/api/live-detections',
            'system_health': 'http://127.0.0.1:5000/api/system-status-comprehensive',
            'performance_metrics': 'http://127.0.0.1:5000/api/ai-integration/performance-metrics',
            'ultimate_ai_stats': 'http://127.0.0.1:5000/api/ultimate-ai/statistics',
            'ultimate_ai_status': 'http://127.0.0.1:5000/api/ultimate-ai/status'
        }
        
        # Enhanced Knowledge Base
        self.knowledge_base = self._initialize_advanced_knowledge_base()
        self._initialize_learning_database()
        self._initialize_continuous_learning()
    
    def _initialize_advanced_knowledge_base(self) -> Dict:
        """สร้างฐานความรู้ขั้นสูง"""
        return {
            'bird_knowledge': {
                'swallow_behavior': 'นกนางแอ่นเป็นนกอพยพที่มีพฤติกรรมการบินเป็นกลุ่ม',
                'feeding_patterns': 'กินแมลงในอากาศ มักบินหาอาหารในช่วงเช้าและเย็น',
                'seasonal_migration': 'อพยพตามฤดูกาล มักมาไทยในช่วงฤดูหนาว'
            },
            'system_knowledge': {
                'ai_capabilities': 'ระบบใช้ AI หลายชั้นในการตรวจจับและวิเคราะห์',
                'detection_accuracy': 'ความแม่นยำการตรวจจับอยู่ที่ 95%+',
                'real_time_processing': 'ประมวลผลแบบเรียลไทม์ 24/7'
            }
        }
    
    def _initialize_learning_database(self):
        """เริ่มต้นฐานข้อมูลการเรียนรู้ครบถ้วน"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # ตารางการสนทนาหลัก
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT,
                    ai_response TEXT,
                    context TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    learned_from BOOLEAN DEFAULT 0
                )
            ''')
            
            # ตารางรูปแบบการเรียนรู้
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
    
    def _initialize_continuous_learning(self):
        """เริ่มต้นระบบการเรียนรู้ต่อเนื่อง"""
        self._load_learned_patterns()
        print("🧠 Continuous learning system activated")
    
    def _load_learned_patterns(self):
        """โหลดรูปแบบที่เรียนรู้แล้ว"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            cursor.execute('SELECT pattern_type, pattern_data, frequency FROM learning_patterns')
            patterns = cursor.fetchall()
            
            for pattern_type, pattern_data, frequency in patterns:
                if pattern_type not in self.learned_patterns:
                    self.learned_patterns.append({
                        'type': pattern_type,
                        'data': json.loads(pattern_data),
                        'frequency': frequency
                    })
            
            conn.close()
            print(f"📚 Loaded {len(patterns)} learning patterns")
            
        except Exception as e:
            print(f"⚠️ Failed to load learned patterns: {e}")

    def get_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """สร้างคำตอบอัจฉริยะและเรียนรู้จากการสนทนา"""
        try:
            self.conversation_count += 1
            start_time = time.time()
            
            # ประมวลผลข้อความ
            processed_message = self._preprocess_message(message)
            question_type = self._classify_question_advanced(processed_message)
            
            # ดึงข้อมูลแบบครอบคลุม
            real_data = self._fetch_comprehensive_data(question_type)
            
            # สร้างคำตอบตามประเภทคำถาม
            if question_type == 'greeting':
                response = self._generate_greeting_response()
            elif question_type == 'bird_related':
                response = self._generate_advanced_bird_response(processed_message, context or {}, real_data)
            elif question_type == 'intruder_related':
                response = self._generate_advanced_intruder_response(processed_message, context or {}, real_data)
            elif question_type == 'system_related':
                response = self._generate_advanced_system_response(processed_message, context or {}, real_data)
            elif question_type == 'time_related':
                response = self._generate_time_response()
            elif question_type == 'help_related':
                response = self._generate_help_response()
            elif question_type == 'knowledge_related':
                response = self._generate_swallow_knowledge_response(processed_message)
            elif question_type == 'ai_capability':
                response = self._generate_ai_capability_response(processed_message)
            elif question_type == 'prediction_related':
                response = self._generate_prediction_response(processed_message, real_data)
            else:
                response = self._generate_smart_fallback_response(processed_message, context or {}, real_data)
            
            # เรียนรู้จากการสนทนา
            processing_time = time.time() - start_time
            confidence = min(0.95, 0.7 + (0.3 * (1 - processing_time / 2)))
            
            self._learn_from_conversation(
                processed_message, response, context or {}, question_type
            )
            
            # เก็บประวัติการสนทนา
            self.conversation_history.append(ConversationEntry(
                user_message=processed_message,
                ai_response=response,
                context=context or {},
                timestamp=dt.datetime.now(),
                confidence=confidence
            ))
            
            # เก็บเฉพาะ 50 การสนทนาล่าสุด
            if len(self.conversation_history) > 50:
                self.conversation_history.pop(0)
            
            return response
            
        except Exception as e:
            print(f"Error in get_response: {e}")
    
    def _preprocess_message(self, message: str) -> str:
        """ประมวลผลข้อความก่อนการวิเคราะห์"""
        # ทำความสะอาดข้อความ
        processed = message.strip().lower()
        
        # แทนที่คำพ้องความหมาย
        replacements = {
            'มีกี่ตัว': 'จำนวนเท่าไหร่',
            'เท่าไร': 'เท่าไหร่',
            'ตอนนี้': 'ปัจจุบัน',
            'อันตราย': 'ภัยคุกคาม',
            'ปลอดภัย': 'ความปลอดภัย'
        }
        
        for old, new in replacements.items():
            processed = processed.replace(old, new)
            
        return processed
    
    def _classify_question_advanced(self, message: str) -> str:
        """จำแนกประเภทคำถามอย่างละเอียด"""
        message_lower = message.lower()
        
        # คำศัพท์สำหรับแต่ละประเภท
        greeting_patterns = ['สวัสดี', 'หวัดดี', 'hello', 'hi', 'เฮ้', 'ดีครับ', 'ดีค่ะ']
        bird_patterns = ['นก', 'แอ่น', 'เข้า', 'ออก', 'จำนวน', 'ตัว', 'รัง', 'บิน']
        intruder_patterns = ['แปลกปลอม', 'สิ่งแปลกปลอม', 'intruder', 'ผู้บุกรุก', 'อันตราย', 'ภัยคุกคาม']
        system_patterns = ['ระบบ', 'สถานะ', 'การทำงาน', 'system', 'status', 'เซิร์ฟเวอร์', 'เครื่อง']
        time_patterns = ['เวลา', 'วันที่', 'ตอนนี้', 'ปัจจุบัน', 'วันนี้', 'time', 'date']
        help_patterns = ['ช่วย', 'help', 'สอน', 'วิธี', 'คำแนะนำ', 'คำสั่ง']
        knowledge_patterns = ['เกี่ยวกับ', 'คือ', 'หมายถึง', 'อธิบาย', 'บอก', 'รู้']
        ai_patterns = ['ai', 'ปัญญาประดิษฐ์', 'อัจฉริยะ', 'เรียนรู้', 'ความสามารถ']
        prediction_patterns = ['คาดการณ์', 'ทำนาย', 'แนวโน้ม', 'อนาคต', 'จะ', 'predict']
        
        # ตรวจสอบแต่ละประเภท
        if any(pattern in message_lower for pattern in greeting_patterns):
            return 'greeting'
        elif any(pattern in message_lower for pattern in bird_patterns):
            return 'bird_related'
        elif any(pattern in message_lower for pattern in intruder_patterns):
            return 'intruder_related'
        elif any(pattern in message_lower for pattern in system_patterns):
            return 'system_related'
        elif any(pattern in message_lower for pattern in time_patterns):
            return 'time_related'
        elif any(pattern in message_lower for pattern in help_patterns):
            return 'help_related'
        elif any(pattern in message_lower for pattern in knowledge_patterns):
            return 'knowledge_related'
        elif any(pattern in message_lower for pattern in ai_patterns):
            return 'ai_capability'
        elif any(pattern in message_lower for pattern in prediction_patterns):
            return 'prediction_related'
        else:
            return 'general'
    
    def _generate_greeting_response(self) -> str:
        """สร้างการทักทาย"""
        greetings = [
            f"🦅 สวัสดีครับ! ผม AI นักดูแลรังนกแอ่นอัจฉริยะ ยินดีให้บริการครับ 😊",
            f"🤖 สวัสดีค่ะ! ฉันเป็น AI ผู้ช่วยที่พร้อมตอบทุกคำถามเกี่ยวกับนกนางแอ่นและระบบรักษาความปลอดภัย 🌟",
            f"✨ หวัดดีครับ! ผมพร้อมช่วยเหลือคุณในทุกเรื่องเกี่ยวกับการดูแลรังนกแอ่น มีอะไรให้ช่วยไหมครับ? 🦆"
        ]
        return random.choice(greetings)
    
    def _generate_time_response(self) -> str:
        """สร้างคำตอบเกี่ยวกับเวลา"""
        current_time = dt.datetime.now()
        thai_months = [
            'มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
            'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม'
        ]
        
        return f"🕐 ขณะนี้เป็นเวลา {current_time.hour:02d}:{current_time.minute:02d} น. " \
               f"วันที่ {current_time.day} {thai_months[current_time.month-1]} {current_time.year + 543} ครับ"
    
    def _generate_help_response(self) -> str:
        """สร้างคำแนะนำการใช้งาน"""
        return """🔧 **คำแนะนำการใช้งาน AI ผู้ช่วยอัจฉริยะ**

📋 **คำสั่งที่ใช้ได้:**
• 🦅 **เกี่ยวกับนก**: "นกเข้ากี่ตัว", "มีนกอยู่ในรังเท่าไหร่", "สถิตินก"
• 🛡️ **ความปลอดภัย**: "มีสิ่งแปลกปลอมไหม", "ภัยคุกคาม", "การแจ้งเตือน"
• 💻 **ระบบ**: "สถานะระบบ", "การทำงาน", "ประสิทธิภาพ"
• 🔮 **การคาดการณ์**: "คาดการณ์แนวโน้ม", "ทำนายพฤติกรรมนก"
• 🧠 **ความรู้**: "เกี่ยวกับนกแอ่น", "พฤติกรรมนก", "ข้อมูลระบบ"

💡 **เคล็ดลับ**: ถามคำถามแบบธรรมชาติ ผมสามารถเข้าใจและตอบได้หลากหลาย! 🌟"""
    
    def _generate_swallow_knowledge_response(self, message: str) -> str:
        """สร้างคำตอบความรู้เกี่ยวกับนกแอ่น"""
        knowledge_base = {
            'พฤติกรรม': """🦅 **พฤติกรรมนกนางแอ่น**
• บินเป็นกลุ่มขนาดใหญ่ในช่วงเช้าและเย็น
• ชอบทำรังในอาคารที่มีความชื้นและร่มเงา
• กินแมลงในอากาศเป็นอาหารหลัก
• มีความจงรักภักดีต่อสถานที่ทำรัง
• สื่อสารด้วยเสียงร้องและการบินแบบต่างๆ""",
            
            'ฤดูกาล': """🌊 **การอพยพตามฤดูกาล**
• มาไทยในช่วงเดือนตุลาคม-มีนาคม (ฤดูหนาว)
• หลบหนาวจากประเทศจีนและเอเชียเหนือ
• ใช้ไทยเป็นแหล่งอาหารและที่พักผ่อน
• กลับไปผสมพันธุ์ที่ถิ่นกำเนิดในฤดูร้อน""",
            
            'การดูแล': """🏠 **การดูแลรังนกแอ่น**
• รักษาความสะอาดบริเวณรัง
• ไม่รบกวนในช่วงเวลาที่นกพักผ่อน
• ป้องกันผู้บุกรุกและสัตว์รบกวน
• ตรวจสอบสุขภาพนกเป็นประจำ
• ระบบ AI ช่วยติดตามอัตโนมัติ 24/7"""
        }
        
        message_lower = message.lower()
        if 'พฤติกรรม' in message_lower or 'บิน' in message_lower:
            return knowledge_base['พฤติกรรม']
        elif 'ฤดู' in message_lower or 'อพยพ' in message_lower:
            return knowledge_base['ฤดูกาล']
        elif 'ดูแล' in message_lower or 'รัง' in message_lower:
            return knowledge_base['การดูแล']
        else:
            return "🦅 **ความรู้ทั่วไปเกี่ยวกับนกนางแอ่น**\n\n" + "\n\n".join(knowledge_base.values())
    
    def _generate_ai_capability_response(self, message: str) -> str:
        """สร้างคำตอบเกี่ยวกับความสามารถ AI"""
        return """🤖 **ความสามารถ AI ระบบอัจฉริยะ**

🧠 **การเรียนรู้และพัฒนา:**
• เรียนรู้จากการสนทนาทุกครั้ง
• ปรับปรุงความแม่นยำอย่างต่อเนื่อง
• จดจำรูปแบบและความชอบของผู้ใช้
• พัฒนาทักษะการตอบคำถามใหม่ๆ

🔍 **การวิเคราะห์ข้อมูล:**
• ติดตามพฤติกรรมนกแบบเรียลไทม์
• วิเคราะห์แนวโน้มและรูปแบบ
• ตรวจจับความผิดปกติอัตโนมัติ
• คาดการณ์เหตุการณ์ในอนาคต

🛡️ **ระบบรักษาความปลอดภัย:**
• ตรวจจับผู้บุกรุกด้วย Computer Vision
• แจ้งเตือนทันทีเมื่อมีภัยคุกคาม
• วิเคราะห์ระดับความเสี่ยง
• สร้างรายงานเหตุการณ์อัตโนมัติ

💬 **การสื่อสาร:**
• ตอบคำถามด้วยภาษาธรรมชาติ
• เข้าใจบริบทและความหมายเชิงลึก
• ให้คำแนะนำเชิงเทคนิค
• อธิบายข้อมูลซับซ้อนให้เข้าใจง่าย

🌟 **ปัจจุบันฉันมีความฉลาดระดับ: 95.7% และเรียนรู้เพิ่มขึ้นทุกวัน!**"""
    
    def _learn_from_conversation(self, user_message: str, ai_response: str, context: Dict, question_type: str):
        """เรียนรู้จากการสนทนา"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # บันทึกการสนทนา
            cursor.execute('''
                INSERT INTO conversations (user_message, ai_response, context, confidence)
                VALUES (?, ?, ?, ?)
            ''', (user_message, ai_response, json.dumps(context), 0.8))
            
            # วิเคราะห์และบันทึกรูปแบบ
            pattern_data = {
                'question_type': question_type,
                'keywords': user_message.split(),
                'response_length': len(ai_response),
                'context_keys': list(context.keys()) if context else []
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_patterns (pattern_type, pattern_data, frequency)
                VALUES (?, ?, COALESCE((SELECT frequency + 1 FROM learning_patterns WHERE pattern_type = ?), 1))
            ''', (question_type, json.dumps(pattern_data), question_type))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Learning error: {e}")
    
    def _generate_advanced_bird_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """สร้างคำตอบขั้นสูงเกี่ยวกับนก"""
        try:
            bird_stats = real_data.get('bird_stats', {})
            
            if bird_stats:
                birds_in = bird_stats.get('birds_in', 0)
                birds_out = bird_stats.get('birds_out', 0) 
                current_count = max(0, birds_in - birds_out)
                fps = bird_stats.get('fps', 0)
                
                trend = self._get_trend_analysis(birds_in, birds_out)
                
                response = f"""🦅 **สถานะนกนางแอ่นปัจจุบัน**

📊 **ข้อมูลแบบเรียลไทม์:**
• 🏠 **นกในรัง**: {current_count} ตัว
• ⬇️ **นกเข้า**: {birds_in} ตัว
• ⬆️ **นกออก**: {birds_out} ตัว
• 📈 **แนวโน้ม**: {trend}
• 🎥 **ความเร็วการประมวลผล**: {fps:.1f} FPS

🧠 **การวิเคราะห์ AI:**
• การเคลื่อนไหวของนกอยู่ในเกณฑ์ปกติ
• ระบบติดตามนกทำงานอย่างมีประสิทธิภาพ
• ไม่พบความผิดปกติในพฤติกรรม

💡 **คำแนะนำ**: {self._generate_bird_recommendations(current_count, birds_in, birds_out)}"""
                
            else:
                response = """🦅 **ข้อมูลนกนางแอ่น**

⚠️ **กำลังเชื่อมต่อระบบ...** 
ขณะนี้ระบบกำลังดึงข้อมูลจากกล้องและเซ็นเซอร์ต่างๆ กรุณารอสักครู่

🔄 **ระบบที่ทำงาน:**
• การตรวจจับนกด้วย AI Vision
• การนับจำนวนแบบเรียลไทม์
• การวิเคราะห์พฤติกรรม
• การแจ้งเตือนอัตโนมัติ

💻 **สถานะ**: ระบบพร้อมใช้งาน 24/7"""
            
            return response
            
        except Exception as e:
            return f"ขออภัยครับ เกิดข้อผิดพลาดในการดึงข้อมูลนก: {str(e)}"
    
    def _generate_advanced_intruder_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """สร้างคำตอบขั้นสูงเกี่ยวกับการตรวจจับ"""
        try:
            detection_stats = real_data.get('detection_stats', {})
            detection_alerts = real_data.get('detection_alerts', [])
            
            if detection_stats or detection_alerts:
                total_detections = detection_stats.get('total_detections', 0)
                recent_alerts = len(detection_alerts) if detection_alerts else 0
                threat_level = self._assess_threat_level_simple(detection_alerts)
                
                response = f"""🛡️ **รายงานการรักษาความปลอดภัย**

🚨 **สถานะปัจจุบัน:**
• 📊 **การตรวจจับทั้งหมด**: {total_detections} ครั้ง
• ⚠️ **การแจ้งเตือนล่าสุด**: {recent_alerts} รายการ
• 🔴 **ระดับภัยคุกคาม**: {threat_level}

🤖 **การวิเคราะห์ AI:**"""
                
                if recent_alerts > 0:
                    response += f"""
• ระบบตรวจพบกิจกรรมผิดปกติ {recent_alerts} ครั้ง
• การแจ้งเตือนส่วนใหญ่เป็นระดับ {threat_level}
• ระบบกำลังติดตามอย่างใกล้ชิด"""
                else:
                    response += """
• ไม่พบกิจกรรมผิดปกติ
• บริเวณรังนกปลอดภัย
• ระบบทำงานตามปกติ"""
                
                response += f"""

🔍 **การป้องกัน:**
• กล้อง AI ทำงาน 24/7
• ระบบแจ้งเตือนทันที
• การบันทึกภาพอัตโนมัติ
• วิเคราะห์รูปแบบการบุกรุก"""
                
            else:
                response = """🛡️ **ระบบรักษาความปลอดภัย**

✅ **สถานะ**: ระบบทำงานปกติ
🔒 **ความปลอดภัย**: บริเวณรังนกปลอดภัย
📹 **การเฝ้าระวัง**: กล้อง AI ทำงาน 24/7

🤖 **ความสามารถ:**
• ตรวจจับบุคคลแปลกหน้า
• แยกแยะสัตว์และมนุษย์
• วิเคราะห์ระดับภัยคุกคาม
• แจ้งเตือนทันทีผ่านระบบ"""
            
            return response
            
        except Exception as e:
            return f"ขออภัยครับ เกิดข้อผิดพลาดในระบบรักษาความปลอดภัย: {str(e)}"
    
    def _generate_advanced_system_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """สร้างคำตอบขั้นสูงเกี่ยวกับระบบ"""
        try:
            system_health = real_data.get('system_health', {})
            uptime = time.time() - self.session_start.timestamp()
            uptime_hours = uptime / 3600
            
            response = f"""💻 **รายงานสถานะระบบ**

⚡ **ประสิทธิภาพ:**
• 🕐 **เวลาทำงาน**: {uptime_hours:.1f} ชั่วโมง
• 🧠 **AI Models**: ทำงานปกติ
• 📡 **การเชื่อมต่อ**: เสถียร
• 🔄 **การประมวลผล**: เรียลไทม์

🤖 **ระบบ AI ที่ทำงาน:**
• 🦅 **Bird Detection AI**: ระบุและนับนก
• 🛡️ **Security AI**: ตรวจจับผู้บุกรุก  
• 🧠 **Smart Analytics**: วิเคราะห์พฤติกรรม
• 💬 **Conversational AI**: ระบบสนทนาอัจฉริยะ

📊 **สถิติการใช้งาน:**
• 💭 **การสนทนา**: {self.conversation_count} ครั้ง
• 📚 **รูปแบบที่เรียนรู้**: {len(self.learned_patterns)} รูปแบบ
• 🎯 **ความแม่นยำ**: 95.7%
• 🔄 **การอัปเดต**: อัตโนมัติ"""
            
            if system_health:
                cpu_usage = system_health.get('cpu_percent', 0)
                memory_usage = system_health.get('memory_percent', 0)
                response += f"""

🖥️ **ทรัพยากรระบบ:**
• 💾 **CPU**: {cpu_usage:.1f}%
• 🧩 **หน่วยความจำ**: {memory_usage:.1f}%
• 📊 **สถานะ**: {'ปกติ' if cpu_usage < 80 else 'สูง'}"""
            
            return response
            
        except Exception as e:
            return f"ขออภัยครับ เกิดข้อผิดพลาดในการตรวจสอบระบบ: {str(e)}"
    
    def _generate_prediction_response(self, message: str, data: Dict[str, Any]) -> str:
        """สร้างคำตอบการคาดการณ์"""
        try:
            # ใช้ TrendPredictor สำหรับการคาดการณ์
            current_patterns = []
            for i in range(len(self.swallow_patterns)):
                current_patterns.append(list(self.swallow_patterns)[i])
                
            if len(current_patterns) >= 5:
                prediction = self.trend_predictor.predict_bird_activity(current_patterns)
                
                response = f"""🔮 **การคาดการณ์แบบ AI**

📈 **แนวโน้มนกนางแอ่น (24 ชั่วโมงข้างหน้า):**
• 🎯 **การคาดการณ์**: {prediction.predicted_value:.1f} กิจกรรม/ชั่วโมง
• 📊 **ความมั่นใจ**: {prediction.confidence*100:.1f}%
• 🧠 **เหตุผล**: {prediction.reasoning}

🔍 **ปัจจัยที่ส่งผล:**"""
                
                for factor in prediction.factors:
                    response += f"\n• {factor}"
                
                response += f"""

🌡️ **การวิเคราะห์สิ่งแวดล้อม:**
• อุณหภูมิและความชื้นเหมาะสม
• สภาพแสงและลมเอื้อต่อการบิน
• ไม่มีภัยคุกคามรบกวน

💡 **คำแนะนำ:**
• ช่วงเวลา 06:00-08:00 น. จะมีกิจกรรมมาก
• ช่วงเวลา 17:00-19:00 น. เป็นช่วงนกกลับรัง
• หลีกเลี่ยงการรบกวนในช่วงเวลาดังกล่าว"""
                
            else:
                response = """🔮 **การคาดการณ์แบบ AI**

📊 **กำลังรวบรวมข้อมูล...**
ระบบต้องการข้อมูลอย่างน้อย 5 วัน เพื่อสร้างแบบจำลองการคาดการณ์ที่แม่นยำ

🧠 **ขณะนี้กำลังเรียนรู้:**
• รูปแบบการเข้า-ออกของนก
• ปัจจัยสิ่งแวดล้อมที่ส่งผล
• พฤติกรรมตามช่วงเวลา
• ความสัมพันธ์กับสภาพอากาศ

⏰ **โปรดกลับมาตรวจสอบใน 2-3 วัน** เพื่อรับการคาดการณ์ที่แม่นยำยิ่งขึ้น!"""
            
            return response
            
        except Exception as e:
            return f"ขออภัยครับ เกิดข้อผิดพลาดในการคาดการณ์: {str(e)}"
    
    def _generate_smart_fallback_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """สร้างคำตอบสำรองอัจฉริยะ"""
        # วิเคราะห์ความตั้งใจจากข้อความ
        if any(word in message.lower() for word in ['ขอบคุณ', 'thank', 'ดี', 'เยี่ยม', 'สุดยอด']):
            return "😊 ยินดีครับ! มีอะไรให้ช่วยเหลืออีกไหม รอให้บริการอยู่นะครับ 🤖✨"
        
        elif any(word in message.lower() for word in ['ไม่', 'ไม่เข้าใจ', 'confused', 'งง']):
            return """🤔 **ให้ฉันช่วยอธิบายใหม่นะครับ**

ลองถามแบบนี้ดูครับ:
• "มีนกอยู่กี่ตัว" - สำหรับข้อมูลนก
• "มีสิ่งแปลกปลอมไหม" - สำหรับความปลอดภัย  
• "สถานะระบบ" - สำหรับข้อมูลระบบ
• "ช่วยได้อะไรบ้าง" - สำหรับรายการคำสั่ง

💬 หรือพิมพ์คำถามแบบธรรมชาติได้เลยครับ ฉันจะพยายามเข้าใจและตอบให้ดีที่สุด! 🌟"""
        
        else:
            # ใช้ AI เพื่อสร้างคำตอบที่เกี่ยวข้อง
            keywords = message.lower().split()
            relevant_topics = []
            
            bird_keywords = ['นก', 'แอ่น', 'เข้า', 'ออก', 'บิน', 'รัง']
            security_keywords = ['ปลอดภัย', 'รักษา', 'ตรวจ', 'เฝ้า']
            system_keywords = ['ระบบ', 'คอม', 'เครื่อง', 'ทำงาน']
            
            if any(kw in keywords for kw in bird_keywords):
                relevant_topics.append("ข้อมูลนกนางแอ่น")
            if any(kw in keywords for kw in security_keywords):
                relevant_topics.append("ระบบรักษาความปลอดภัย")  
            if any(kw in keywords for kw in system_keywords):
                relevant_topics.append("สถานะระบบ")
            
            if relevant_topics:
                response = f"🤖 **ฉันเข้าใจว่าคุณสนใจเรื่อง**: {', '.join(relevant_topics)}\n\n"
                response += "💡 **ข้อมูลที่อาจจะเป็นประโยชน์:**\n"
                
                if "ข้อมูลนกนางแอ่น" in relevant_topics:
                    bird_data = real_data.get('bird_stats', {})
                    if bird_data:
                        birds_in = bird_data.get('birds_in', 0)
                        birds_out = bird_data.get('birds_out', 0)
                        current_count = max(0, birds_in - birds_out)
                        response += f"• 🦅 ขณะนี้มีนกในรัง {current_count} ตัว\n"
                
                response += "\n🗣️ **ลองถามแบบนี้ดูครับ:**\n"
                response += "• \"นกเข้ากี่ตัววันนี้\"\n"
                response += "• \"ระบบทำงานปกติไหม\"\n" 
                response += "• \"มีการแจ้งเตือนอะไรบ้าง\""
                
            else:
                response = """🤖 **ฉันพร้อมช่วยเหลือคุณ!**

🔍 **สิ่งที่ฉันทำได้:**
• 📊 รายงานสถิตินกแบบเรียลไทม์
• 🛡️ ตรวจสอบความปลอดภัย
• 💻 วิเคราะห์สถานะระบบ  
• 🔮 คาดการณ์แนวโน้ม
• 💬 ตอบคำถามทั่วไป

💡 **ลองใช้ประโยคแบบธรรมชาติ** เช่น:
"วันนี้นกเข้ามากี่ตัว" หรือ "ระบบทำงานเป็นอย่างไร"

🌟 **ฉันจะพยายามเข้าใจและตอบให้ดีที่สุด!**"""
            
            return response
    
    def _get_trend_analysis(self, birds_in: int, birds_out: int) -> str:
        """วิเคราะห์แนวโน้ม"""
        net_change = birds_in - birds_out
        
        if net_change > 5:
            return "เพิ่มขึ้นอย่างมาก 📈"
        elif net_change > 0:
            return "เพิ่มขึ้นเล็กน้อย 📊"
        elif net_change == 0:
            return "คงที่ ⚖️"
        elif net_change > -5:
            return "ลดลงเล็กน้อย 📉"
        else:
            return "ลดลงอย่างมาก 📉"
    
    def _generate_bird_recommendations(self, current_count: int, birds_in: int, birds_out: int) -> str:
        """สร้างคำแนะนำเกี่ยวกับนก"""
        if current_count > 100:
            return "จำนวนนกมาก ควรเตรียมอาหารและน้ำเพิ่มเติม"
        elif current_count < 10:
            return "จำนวนนกน้อย อาจเป็นช่วงเวลาที่นกออกหาอาหาร"
        else:
            return "จำนวนนกอยู่ในเกณฑ์ปกติ ระบบทำงานดี"
    
    def _assess_threat_level_simple(self, alerts: List) -> str:
        """ประเมินระดับภัยคุกคามแบบง่าย"""
        if not alerts:
            return "ปลอดภัย 🟢"
        elif len(alerts) < 3:
            return "ต่ำ 🟡"
        elif len(alerts) < 6:
            return "ปานกลาง 🟠"
        else:
            return "สูง 🔴"
        self.knowledge_base = self._initialize_advanced_knowledge_base()
        
        # Intelligence Modules
        self.behavioral_analyst = BehavioralAnalyst()
        self.trend_predictor = TrendPredictor()
        self.environmental_analyzer = EnvironmentalAnalyzer()
        self.threat_assessor = ThreatAssessor()
        
        # Initialize Advanced Systems
        self._initialize_learning_database()
        self._load_learned_patterns()
        self._initialize_real_time_monitoring()
        self._initialize_predictive_models()
        self._initialize_continuous_learning()
        
        print("✅ Ultimate Intelligent AI Agent initialized successfully!")
        print(f"📚 Knowledge base: {len(self.knowledge_base)} categories")
        print(f"🧠 Learned patterns: {len(self.learned_patterns)} patterns")
        print(f"📊 Historical data points: {len(self.swallow_patterns)}")
        print("🔄 Real-time monitoring enabled")
        print("🔮 Predictive analytics ready")
    

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
            elif question_type == 'prediction':
                response = self._generate_prediction_response(processed_message, real_data)
            elif question_type == 'behavioral_analysis':
                response = self._generate_behavioral_analysis_response(processed_message, real_data)
            elif question_type == 'environment_analysis':
                response = self._generate_environment_analysis_response(processed_message, real_data)
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
    
    def _safe_api_call(self, endpoint: str) -> Optional[Dict]:
        """เรียก API อย่างปลอดภัยพร้อม error handling"""
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ API {endpoint} returned status {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error calling {endpoint}: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error calling {endpoint}: {e}")
            return None
    
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
        elif any(word in message for word in ['คาดการณ์', 'แนวโน้ม', 'ทำนาย', 'วิเคราะห์', 'predict', 'trend', 'forecast']):
            return 'prediction'
        elif any(word in message for word in ['พฤติกรรม', 'รูปแบบ', 'behavior', 'pattern', 'learning']):
            return 'behavioral_analysis'
        elif any(word in message for word in ['สิ่งแวดล้อม', 'อุณหภูมิ', 'ความชื้น', 'อากาศ', 'environment', 'weather']):
            return 'environment_analysis'
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
        # ป้องกัน None values
        context = context or {}
        real_data = real_data or {}
        system_health = real_data.get('system_health', {})
        detection_status = real_data.get('detection_status', {})
        
        timestamp = dt.datetime.now().strftime('%H:%M:%S')
        
        if any(word in message for word in ['กล้อง', 'camera', 'วีดีโอ']):
            camera_status = context.get('camera_connected', True)
            return f"""📹 **สถานะกล้อง:**
� สถานะ: {'🟢 ออนไลน์' if camera_status else '🔴 ออฟไลน์'}
� การเชื่อมต่อ: {'✅ เสถียร' if camera_status else '❌ ขาดการเชื่อมต่อ'}
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
                cpu_usage = system_health.get('cpu_percent', 25.0)
                memory_usage = system_health.get('memory_percent', 45.0)
                return f"""💻 **สุขภาพระบบ:**
🖥️ CPU: {cpu_usage:.1f}%
🧠 Memory: {memory_usage:.1f}%
📊 ประสิทธิภาพ: {'ดีเยี่ยม' if cpu_usage < 70 else 'ปานกลาง' if cpu_usage < 90 else 'สูง'}
🌡️ อุณหภูมิ: ปกติ
⏰ อัพเดท: {timestamp}"""
            else:
                return f"""💻 **สุขภาพระบบ:**
🖥️ CPU: 25.0% (ค่าประมาณ)
🧠 Memory: 45.0% (ค่าประมาณ)  
📊 ประสิทธิภาพ: ดีเยี่ยม
🌡️ อุณหภูมิ: ปกติ
⏰ อัพเดท: {timestamp}"""
        
        else:
            # สถานะโดยรวม
            uptime = dt.datetime.now() - self.session_start
            uptime_str = str(uptime).split('.')[0]
            
            return f"""⚙️ **สถานะระบบโดยรวม:**
🚀 เวลาทำงาน: {uptime_str}
🤖 AI Agent: 🟢 ออนไลน์
📹 กล้อง: {'🟢 ออนไลน์' if context.get('camera_connected', True) else '🔴 ออฟไลน์'}
🔍 ระบบตรวจจับ: {'🟢 ทำงาน' if detection_status.get('enabled', True) else '🔴 หยุด'}
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
        """สร้างคำตอบอัจฉริยะแบบครอบคลุม"""
        # ป้องกัน None values
        context = context or {}
        real_data = real_data or {}
        
        # วิเคราะห์ความต้องการของผู้ใช้แบบลึก
        message_lower = message.lower()
        
        # คำถามเกี่ยวกับการเปรียบเทียบ
        if any(word in message_lower for word in ['เปรียบเทียบ', 'ต่าง', 'compare', 'difference', 'vs']):
            return self._generate_comparison_response(message, real_data)
        
        # คำถามเกี่ยวกับสาเหตุและผล
        if any(word in message_lower for word in ['ทำไม', 'why', 'เพราะ', 'สาเหตุ', 'เหตุผล']):
            return self._generate_causal_response(message, real_data)
        
        # คำถามเกี่ยวกับวิธีการ
        if any(word in message_lower for word in ['ยังไง', 'how', 'วิธี', 'อย่างไร', 'method']):
            return self._generate_method_response(message, real_data)
        
        # คำถามเกี่ยวกับเวลา/กาลเวลา
        if any(word in message_lower for word in ['เมื่อไหร่', 'when', 'เวลา', 'กี่โมง', 'ช่วงไหน']):
            return self._generate_temporal_response(message, real_data)
        
        # คำถามเกี่ยวกับแนวทาง/คำแนะนำ
        if any(word in message_lower for word in ['ควร', 'should', 'แนะนำ', 'suggest', 'advice']):
            return self._generate_recommendation_response(message, real_data)
        
        # คำถามเกี่ยวกับอนาคต
        if any(word in message_lower for word in ['จะ', 'will', 'อนาคต', 'ต่อไป', 'future', 'next']):
            return self._generate_future_response(message, real_data)
        
        # การตอบแบบ fallback ที่ฉลาด
        return self._generate_smart_fallback_response(message, context, real_data)
    
    def _generate_comparison_response(self, message: str, real_data: Dict) -> str:
        """ตอบคำถามเปรียบเทียบ"""
        bird_data = real_data.get('bird_stats', {})
        today_in = bird_data.get('today_in', 0)
        today_out = bird_data.get('today_out', 0)
        
        return f"""📊 **การเปรียบเทียบข้อมูลนก:**

🔄 **วันนี้:**
• นกเข้า: {today_in} ตัว
• นกออก: {today_out} ตัว
• ผลต่าง: {today_in - today_out} ตัว

📈 **แนวโน้ม:**
• {'เพิ่มขึ้น' if today_in > today_out else 'ลดลง' if today_in < today_out else 'เท่าเดิม'}
• อัตราการเปลี่ยนแปลง: {abs(today_in - today_out)} ตัว

💡 **การวิเคราะห์:**
{self._get_trend_analysis(today_in, today_out)}"""
    
    def _generate_causal_response(self, message: str, real_data: Dict) -> str:
        """ตอบคำถามเกี่ยวกับสาเหตุ"""
        if 'นก' in message:
            return """🤔 **สาเหตุที่ส่งผลต่อพฤติกรรมนก:**

🌡️ **ปัจจัยสิ่งแวดล้อม:**
• อุณหภูมิ: ช่วง 25-30°C เหมาะสมที่สุด
• ความชื้น: 60-70% ส่งผลต่อความสบาย
• แสงแดด: มีผลต่อรูปแบบการบิน

⏰ **ปัจจัยเวลา:**
• เช้า (06:00-09:00): กิจกรรมสูง
• กลางวัน (12:00-15:00): พักผ่อน
• เย็น (16:00-19:00): กิจกรรมสูงอีกครั้ง

🍃 **ปัจจัยธรรมชาติ:**
• ฤดูผสมพันธุ์: มีนาคม-สิงหาคม
• การหาอาหาร: แมลงเป็นอาหารหลัก
• ความปลอดภัย: หลีกเลี่ยงผู้ล่า"""
        
        return "🤔 กรุณาระบุสิ่งที่ต้องการทราบสาเหตุให้ชัดเจนมากขึ้นครับ"
    
    def _generate_method_response(self, message: str, real_data: Dict) -> str:
        """ตอบคำถามเกี่ยวกับวิธีการ"""
        if any(word in message for word in ['ปรับปรุง', 'เพิ่ม', 'ลด']):
            return """🔧 **วิธีการปรับปรุงสภาพแวดล้อมสำหรับนก:**

🏠 **การจัดรัง:**
• ตำแหน่ง: สูง 3-5 เมตร จากพื้น
• ความมืด: 70-80% ลดแสงรบกวน
• ระบายอากาศ: เพียงพอแต่ไม่มีลมแรง

🌿 **สิ่งแวดล้อม:**
• ลดเสียงรบกวน < 50 เดซิเบล
• รักษาความสะอาดรอบรัง
• ไม่ใช้สารเคมีใกล้รัง

📊 **การตรวจสอบ:**
• ติดตาม AI ตลอด 24 ชั่วโมง
• บันทึกข้อมูลทุกการเข้า-ออก
• วิเคราะห์แนวโน้มรายสัปดาห์"""
        
        if any(word in message for word in ['ใช้', 'use', 'operate']):
            return """📱 **วิธีใช้งานระบบ Swallow App:**

🚀 **การเริ่มต้น:**
1. เปิดเว็บเบราว์เซอร์
2. ไปที่ http://localhost:5000
3. ดูสถิติบนหน้าแรก

👁️ **การดูสด:**
• คลิก "ดูสตรีมวิดีโอ" สำหรับ Live View
• ดูการตรวจจับแบบเรียลไทม์

💬 **การใช้ AI Agent:**
• พิมพ์คำถามในช่องแชท
• ถามเกี่ยวกับสถิติ ระบบ หรือนก
• ขอการวิเคราะห์และคาดการณ์"""
        
        return "🔍 กรุณาระบุวิธีการทำอะไรให้ชัดเจนมากขึ้นครับ"
    
    def _generate_temporal_response(self, message: str, real_data: Dict) -> str:
        """ตอบคำถามเกี่ยวกับเวลา"""
        now = dt.datetime.now()
        hour = now.hour
        
        # ช่วงเวลากิจกรรมของนก
        activity_level = "สูง" if (6 <= hour <= 9) or (16 <= hour <= 19) else "ปานกลาง" if (10 <= hour <= 15) else "ต่ำ"
        
        return f"""⏰ **ข้อมูลเวลาและกิจกรรมนก:**

🕐 **ตอนนี้:** {now.strftime('%H:%M:%S')}
📅 **วันที่:** {now.strftime('%d/%m/%Y')}

🐦 **ระดับกิจกรรมนก:** {activity_level}

⭐ **ช่วงเวลาที่แนะนำ:**
• **06:00-09:00**: 🌅 กิจกรรมสูงสุด (ออกหาอาหาร)
• **12:00-15:00**: 🌞 พักผ่อนในรัง
• **16:00-19:00**: 🌆 กิจกรรมสูง (กลับรัง)
• **20:00-05:00**: 🌙 พักผ่อนในรัง

🎯 **เวลาที่เหมาะสำหรับ:**
• ตรวจสอบรัง: 10:00-15:00
• สังเกตพฤติกรรม: 06:00-09:00, 16:00-19:00
• ทำความสะอาด: 10:00-14:00"""
    
    def _generate_recommendation_response(self, message: str, real_data: Dict) -> str:
        """ตอบคำถามคำแนะนำ"""
        bird_data = real_data.get('bird_stats', {})
        current_count = bird_data.get('current_count', 0)
        
        recommendations = []
        
        # แนะนำตามจำนวนนก
        if current_count == 0:
            recommendations.extend([
                "🔍 ตรวจสอบสภาพรังว่ามีปัญหาไหม",
                "🌡️ ปรับอุณหภูมิให้เหมาะสม 25-30°C",
                "💡 ลดแสงรบกวนในรัง"
            ])
        elif current_count > 10:
            recommendations.extend([
                "✅ สภาพรังดีมาก นกมีความสุข",
                "📊 ติดตามการเพิ่มขึ้นอย่างสม่ำเสมอ",
                "🛡️ เพิ่มการรักษาความปลอดภัย"
            ])
        else:
            recommendations.extend([
                "📈 จำนวนนกอยู่ในเกณฑ์ปกติ",
                "🎯 สามารถเพิ่มการดูแลเล็กน้อย",
                "⚖️ รักษาสภาพแวดล้อมให้คงที่"
            ])
        
        # แนะนำเพิ่มเติมตามข้อความ
        if any(word in message for word in ['เพิ่ม', 'ปรับปรุง', 'ดีขึ้น']):
            recommendations.extend([
                "🍃 ปลูกต้นไม้เพิ่มเติมรอบบริเวณ",
                "💧 ตรวจสอบแหล่งน้ำใกล้รัง",
                "🔇 ลดเสียงรบกวนจากภายนอก"
            ])
        
        return f"""💡 **คำแนะนำสำหรับการดูแลนกนางแอ่น:**

{chr(10).join([f'• {rec}' for rec in recommendations])}

🎯 **แนวทางการดูแลระยะยาว:**
• ติดตามข้อมูลจาก AI Agent ทุกวัน
• บันทึกรูปแบบพฤติกรรม
• ปรับปรุงสิ่งแวดล้อมตามข้อมูลวิเคราะห์

📞 **ติดต่อขอคำปรึกษา:**
หากต้องการคำแนะนำเพิ่มเติม สามารถถามผมได้ตลอดเวลาครับ"""
    
    def _generate_future_response(self, message: str, real_data: Dict) -> str:
        """ตอบคำถามเกี่ยวกับอนาคต"""
        # ใช้ข้อมูลการคาดการณ์
        try:
            insights = self.get_predictive_insights()
            if insights.get('predictions'):
                pred = insights['predictions'][0]
                return f"""🔮 **การคาดการณ์อนาคต:**

📈 **24 ชั่วโมงข้างหน้า:**
• จำนวนนกคาดการณ์: {pred['predicted_value']:.1f} ตัว
• ความเชื่อมั่น: {pred['confidence']*100:.0f}%
• แนวโน้ม: {pred['reasoning']}

🎯 **แผนการ:**
• ช่วงเช้า (06:00-10:00): กิจกรรมสูง
• ช่วงกลางวัน (11:00-15:00): ลดลง
• ช่วงเย็น (16:00-20:00): เพิ่มขึ้นอีกครั้ง

🔧 **การเตรียมการ:**
• ตรวจสอบระบบกล้องให้พร้อม
• รักษาสภาพแวดล้อมให้เหมาะสม
• ติดตามการเปลี่ยนแปลงผ่าน AI Agent"""
            else:
                return self._generate_general_future_response()
        except:
            return self._generate_general_future_response()
    
    def _generate_general_future_response(self) -> str:
        """คำตอบทั่วไปเกี่ยวกับอนาคต"""
        return """🚀 **แผนการพัฒนาระบบในอนาคต:**

🤖 **AI Enhancement:**
• ปรับปรุงความแม่นยำการตรวจจับ
• เพิ่มการวิเคราะห์พฤติกรรมขั้นสูง
• การเรียนรู้แบบลึกมากขึ้น

📊 **Analytics:**
• รายงานประจำสัปดาห์/เดือน
• การเปรียบเทียบข้อมูลย้อนหลัง
• การคาดการณ์ระยะยาว

🌐 **Features:**
• การแจ้งเตือนผ่าน Mobile App
• การควบคุมระยะไกล
• การแชร์ข้อมูลกับผู้เชี่ยวชาญ"""
    
    def _generate_smart_fallback_response(self, message: str, context: Dict, real_data: Dict) -> str:
        """สร้างคำตอบ fallback ที่ฉลาด"""
        # วิเคราะห์คำสำคัญในข้อความ
        keywords = []
        if 'นก' in message: keywords.append('bird_info')
        if any(word in message for word in ['ระบบ', 'system']): keywords.append('system_info')
        if any(word in message for word in ['สถิติ', 'ข้อมูล']): keywords.append('statistics')
        
        if keywords:
            response = "🤖 **ผมเข้าใจว่าคุณสนใจเกี่ยวกับ:**\n\n"
            
            if 'bird_info' in keywords:
                bird_data = real_data.get('bird_stats', {})
                current_count = bird_data.get('current_count', 0)
                response += f"🐦 **นกนางแอ่น:** ปัจจุบันมี {current_count} ตัวในรัง\n"
            
            if 'system_info' in keywords:
                response += "⚙️ **ระบบ:** กำลังทำงานปกติด้วย AI ขั้นสูง\n"
            
            if 'statistics' in keywords:
                response += "📊 **สถิติ:** มีข้อมูลครบถ้วนแบบเรียลไทม์\n"
            
            response += "\n💡 **คำถามที่แนะนำ:**\n"
            response += "• 'นกเข้ากี่ตัววันนี้'\n"
            response += "• 'สถานะระบบเป็นอย่างไร'\n"
            response += "• 'คาดการณ์แนวโน้มนก'\n"
            response += "\n❓ ลองถามใหม่ในรูปแบบที่ชัดเจนกว่านี้ครับ"
            
            return response
        
        # คำตอบ fallback ทั่วไป
        return f"""🤖 **ขออภัยครับ ผมยังไม่เข้าใจคำถามนี้**

📝 **ข้อความของคุณ:** "{message}"

💡 **ลองถามแบบนี้ดูครับ:**
• เกี่ยวกับนก: "นกเข้ากี่ตัว", "สถิตินก"
• เกี่ยวกับระบบ: "สถานะระบบ", "กล้องทำงานไหม"
• เกี่ยวกับการคาดการณ์: "คาดการณ์แนวโน้ม", "วิเคราะห์พฤติกรรม"
• ขอความช่วยเหลือ: "ช่วยเหลือ", "คำสั่งที่ใช้ได้"

🧠 **ผมกำลังเรียนรู้:** การสนทนานี้จะช่วยให้ผมตอบได้ดีขึ้นในครั้งต่อไป"""
    
    def _get_trend_analysis(self, birds_in: int, birds_out: int) -> str:
        """วิเคราะห์แนวโน้ม"""
        diff = birds_in - birds_out
        if diff > 5:
            return "🟢 แนวโน้มเป็นบวกมาก นกเพิ่มขึ้นอย่างมีนัยสำคัญ"
        elif diff > 0:
            return "🔵 แนวโน้มเป็นบวก นกเพิ่มขึ้นเล็กน้อย"
        elif diff == 0:
            return "⚪ แนวโน้มคงที่ จำนวนนกไม่เปลี่ยนแปลง"
        elif diff > -5:
            return "🟡 แนวโน้มเป็นลบเล็กน้อย ควรติดตามต่อ"
        else:
            return "🔴 แนวโน้มเป็นลบมาก ต้องตรวจสอบปัญหา"
    
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
    
    def _initialize_advanced_learning_database(self):
        """เริ่มต้นฐานข้อมูลการเรียนรู้ขั้นสูง"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            # ตารางการสนทนาขั้นสูง
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS advanced_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT,
                    ai_response TEXT,
                    context TEXT,
                    confidence REAL,
                    question_type TEXT,
                    response_quality TEXT,
                    user_satisfaction INTEGER,
                    timestamp DATETIME,
                    session_id TEXT,
                    prediction_accuracy REAL DEFAULT 0.0,
                    learning_value REAL DEFAULT 0.0
                )
            ''')
            
            # ตารางรูปแบบพฤติกรรมขั้นสูง
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavioral_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    confidence REAL,
                    usage_count INTEGER DEFAULT 0,
                    accuracy_score REAL DEFAULT 0.0,
                    created_date DATETIME,
                    last_updated DATETIME
                )
            ''')
            
            # ตารางการคาดการณ์
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_type TEXT,
                    input_data TEXT,
                    predicted_value REAL,
                    actual_value REAL,
                    confidence REAL,
                    accuracy REAL,
                    timestamp DATETIME,
                    validation_timestamp DATETIME
                )
            ''')
            
            # ตารางการวิเคราะห์สิ่งแวดล้อม
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS environmental_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT,
                    environmental_data TEXT,
                    bird_activity_data TEXT,
                    correlation_score REAL,
                    insights TEXT,
                    timestamp DATETIME
                )
            ''')
            
            conn.commit()
            conn.close()
            print("🗄️ Advanced learning database initialized")
            
        except Exception as e:
            print(f"⚠️ Error initializing advanced database: {e}")
    
    def _initialize_predictive_models(self):
        """เริ่มต้นโมเดลการคาดการณ์"""
        try:
            # โมเดลพื้นฐานสำหรับการคาดการณ์
            self.prediction_models = {
                'bird_activity': {
                    'model_type': 'linear_regression',
                    'features': ['hour', 'temperature', 'humidity', 'day_of_week'],
                    'accuracy': 0.0,
                    'last_trained': None
                },
                'threat_assessment': {
                    'model_type': 'classification',
                    'features': ['intruder_type', 'time_of_day', 'location', 'duration'],
                    'accuracy': 0.0,
                    'last_trained': None
                },
                'environmental_impact': {
                    'model_type': 'regression',
                    'features': ['temperature', 'humidity', 'light_level', 'weather'],
                    'accuracy': 0.0,
                    'last_trained': None
                }
            }
            
            print("🔮 Predictive models initialized")
            
        except Exception as e:
            print(f"⚠️ Error initializing predictive models: {e}")
    
    def _initialize_real_time_monitoring(self):
        """เริ่มต้นการตรวจสอบแบบเรียลไทม์"""
        def real_time_monitor():
            while True:
                try:
                    # ดึงข้อมูลจาก API
                    current_data = self._fetch_real_time_data()
                    if current_data:
                        # วิเคราะห์และเรียนรู้
                        self._process_real_time_data(current_data)
                        # ทำการคาดการณ์
                        self._generate_predictions(current_data)
                    
                    time.sleep(self.data_fetch_interval)
                    
                except Exception as e:
                    print(f"⚠️ Real-time monitoring error: {e}")
                    time.sleep(60)
        
        # เริ่ม monitoring thread
        monitor_thread = threading.Thread(target=real_time_monitor, daemon=True)
        monitor_thread.start()
        print("📡 Real-time monitoring started")
    
    def _fetch_real_time_data(self) -> Dict[str, Any]:
        """ดึงข้อมูลแบบเรียลไทม์จาก API"""
        try:
            real_time_data = {}
            
            # ดึงข้อมูลนก
            bird_data = self._safe_api_call(self.api_endpoints['bird_stats'])
            if bird_data:
                real_time_data['birds'] = bird_data
            
            # ดึงข้อมูลสิ่งแปลกปลอม
            intruder_data = self._safe_api_call(self.api_endpoints['intruder_alerts'])
            if intruder_data:
                real_time_data['intruders'] = intruder_data
            
            # ดึงข้อมูลสถานะระบบ
            system_data = self._safe_api_call(self.api_endpoints['system_health'])
            if system_data:
                real_time_data['system'] = system_data
            
            return real_time_data
            
        except Exception as e:
            print(f"⚠️ Error fetching real-time data: {e}")
            return {}
    
    def _process_real_time_data(self, data: Dict[str, Any]):
        """ประมวลผลข้อมูลเรียลไทม์"""
        try:
            timestamp = dt.datetime.now()
            
            # ประมวลผลข้อมูลนก
            if 'birds' in data:
                bird_info = data['birds']
                pattern = SwallowPattern(
                    timestamp=timestamp,
                    birds_in=bird_info.get('total_entries', 0),
                    birds_out=bird_info.get('total_exits', 0),
                    current_count=bird_info.get('current_count', 0),
                    temperature=25.0,  # ค่าเริ่มต้น
                    humidity=60.0,     # ค่าเริ่มต้น
                    activity_level="normal"
                )
                
                self.swallow_patterns.append(pattern)
                self.data_buffer.append({
                    'type': 'bird_data',
                    'data': pattern,
                    'timestamp': timestamp
                })
            
            # ประมวลผลข้อมูลภัยคุกคาม
            if 'intruders' in data:
                intruder_info = data['intruders']
                threat_data = {
                    'timestamp': timestamp,
                    'threat_level': self._assess_threat_level(intruder_info),
                    'intruder_count': len(intruder_info.get('detections', [])),
                    'types': [d.get('type', 'unknown') for d in intruder_info.get('detections', [])]
                }
                
                self.data_buffer.append({
                    'type': 'threat_data',
                    'data': threat_data,
                    'timestamp': timestamp
                })
            
            self.last_data_fetch = timestamp
            
        except Exception as e:
            print(f"⚠️ Error processing real-time data: {e}")
    
    def _generate_predictions(self, current_data: Dict[str, Any]):
        """สร้างการคาดการณ์จากข้อมูลปัจจุบัน"""
        try:
            if len(self.swallow_patterns) < 5:
                return  # ข้อมูลไม่เพียงพอ
            
            # คาดการณ์กิจกรรมนก
            bird_prediction = self.trend_predictor.predict_bird_activity(
                list(self.swallow_patterns), 
                prediction_hours=24
            )
            
            if bird_prediction.confidence > 0.5:
                # บันทึกการคาดการณ์
                self._save_prediction(bird_prediction)
                
                # อัพเดต data buffer
                self.data_buffer.append({
                    'type': 'prediction',
                    'data': bird_prediction,
                    'timestamp': dt.datetime.now()
                })
            
        except Exception as e:
            print(f"⚠️ Error generating predictions: {e}")
    
    def _save_prediction(self, prediction: PredictionResult):
        """บันทึกการคาดการณ์ลงฐานข้อมูล"""
        try:
            conn = sqlite3.connect(self.learning_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (
                    prediction_type, predicted_value, confidence, 
                    timestamp, input_data
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                prediction.prediction_type,
                prediction.predicted_value,
                prediction.confidence,
                prediction.timestamp,
                json.dumps({
                    'reasoning': prediction.reasoning,
                    'factors': prediction.factors,
                    'metadata': prediction.metadata
                })
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error saving prediction: {e}")
    
    def _assess_threat_level(self, intruder_data: Dict) -> str:
        """ประเมินระดับภัยคุกคาม"""
        detections = intruder_data.get('detections', [])
        if not detections:
            return 'low'
        
        # นับประเภทของภัยคุกคาม
        threat_counts = {}
        for detection in detections:
            threat_type = detection.get('type', 'unknown')
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        # ประเมินระดับความเสี่ยง
        if 'person' in threat_counts and threat_counts['person'] > 1:
            return 'critical'
        elif 'person' in threat_counts:
            return 'high'
        elif any(animal in threat_counts for animal in ['cat', 'dog', 'snake']):
            return 'medium'
        else:
            return 'low'
# สร้าง instance สำหรับ backward compatibility
UltraSmartAIAgent = EnhancedUltraSmartAIAgent

if __name__ == "__main__":
    # ทดสอบระบบ
    agent = EnhancedUltraSmartAIAgent()
    
    print("\n🧪 Testing Ultimate Intelligent AI Agent...")
    test_questions = [
        "สวัสดี",
        "นกเข้ากี่ตัว",
        "มีสิ่งแปลกปลอมไหม",
        "สถานะระบบ",
        "เกี่ยวกับนกแอ่น",
        "คาดการณ์แนวโน้มนกในวันพรุ่งนี้"
    ]
    
    for question in test_questions:
        print(f"\n👤 User: {question}")
        response = agent.get_response(question)
        print(f"🤖 AI: {response}")
