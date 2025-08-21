# models.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import json

# Import configuration and schemas with fallbacks
try:
    try:
        from .config import Config
    except ImportError:
        from config import Config
except ImportError:
    # Fallback configuration
    class Config:
        SQLITE_DATABASE_URI = 'sqlite:///db.sqlite'
        ANOMALY_DB_PATH = 'anomaly_alerts.db'
        BIRD_DB_PATH = 'swallow_smart_stats.db'

try:
    try:
        from .schemas import BirdActivityData, AnomalyData
    except ImportError:
        from schemas import BirdActivityData, AnomalyData
except ImportError:
    # Fallback schemas
    from typing import Dict, Any
    BirdActivityData = Dict[str, Any]
    AnomalyData = Dict[str, Any]

# กำหนด Base สำหรับ Declarative system
Base = declarative_base()

# --- กำหนดโครงสร้างฐานข้อมูล ---
class BirdActivity(Base):
    """
    คลาสสำหรับบันทึกกิจกรรมของนกในแต่ละครั้งที่มีการตรวจจับ
    """
    __tablename__ = 'bird_activity'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    birds_in = Column(Integer, default=0)
    birds_out = Column(Integer, default=0)
    confidence = Column(String, nullable=True)
    weather_data = Column(Text, nullable=True)
    meta_data = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "birds_in": self.birds_in,
            "birds_out": self.birds_out,
            "confidence": self.confidence,
            "weather_data": self.weather_data,
            "meta_data": self.meta_data,
        }

    def to_pydantic(self):
        """Convert to Pydantic model if available, otherwise return dict"""
        try:
            return BirdActivityData(
                birds_in=self.birds_in,
                birds_out=self.birds_out,
                confidence=float(self.confidence) if self.confidence else 0.0,
                weather_data=json.loads(self.weather_data) if self.weather_data else None,
                metadata=json.loads(self.meta_data) if self.meta_data else None
            )
        except:
            return self.to_dict()

class AnomalyDetection(Base):
    """
    คลาสสำหรับบันทึกเหตุการณ์ผิดปกติ (Anomaly) ที่ตรวจพบ
    """
    __tablename__ = 'anomaly_detection'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    object_type = Column(String, nullable=False)
    confidence = Column(String, nullable=False)
    image_path = Column(String, nullable=True)
    status = Column(String, default='new')

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "object_type": self.object_type,
            "confidence": float(self.confidence),
            "image_path": self.image_path,
            "status": self.status,
        }

    def to_pydantic(self):
        """Convert to Pydantic model if available, otherwise return dict"""
        try:
            return AnomalyData(
                object_type=self.object_type,
                confidence=float(self.confidence),
                image_path=self.image_path,
                status=self.status
            )
        except:
            return self.to_dict()

# --- การตั้งค่าฐานข้อมูล ---
engine = create_engine(Config.SQLITE_DATABASE_URI)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(engine)
    return True

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session():
    """Get database session directly (not generator)"""
    return SessionLocal()