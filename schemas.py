# schemas.py
from pydantic import BaseModel, validator, Field
from datetime import datetime
from typing import Optional, Dict
import json

class BirdActivityData(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    birds_in: int = Field(ge=0)
    birds_out: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    weather_data: Optional[Dict] = None
    metadata: Optional[Dict] = None

    @validator('birds_in', 'birds_out')
    def validate_count(cls, v):
        if v < 0:
            raise ValueError("Bird count cannot be negative")
        return v

    @validator('weather_data', 'metadata', pre=True)
    def validate_json(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format")
        return v

class AnomalyData(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    object_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    image_path: Optional[str] = None
    status: str = Field(default='new')

    @validator('status')
    def validate_status(cls, v):
        if v not in ['new', 'viewed', 'archived']:
            raise ValueError("Invalid status value")
        return v

class AnomalyUpdate(BaseModel):
    status: str = Field(..., description="Status to update anomaly to")

    @validator('status')
    def validate_status(cls, v):
        if v not in ['viewed', 'archived']:
            raise ValueError("Invalid status for update")
        return v