

# config.py - Modern, Robust Configuration for Swallow AI System
import os
import json
from typing import Dict, Any, Optional

class Config:
    """
    Central configuration for all AI modules:
    - Object/Anomaly Detection
    - Bird Detection/Counting
    - Chatbot
    พร้อม type hints, robust validation, และโครงสร้างขยายง่าย
    """

    # Database Configuration
    SQLITE_DATABASE_URI: str = os.environ.get('SQLITE_DATABASE_URI', 'sqlite:///db.sqlite')
    ANOMALY_DB_PATH: str = os.environ.get('ANOMALY_DB_PATH', 'anomaly_alerts.db')
    BIRD_DB_PATH: str = os.environ.get('BIRD_DB_PATH', 'swallow_smart_stats.db')

    # Model Paths
    YOLO_MODEL_PATH: str = os.environ.get('YOLO_MODEL_PATH', 'yolov8n.pt')
    ULTRA_SAFE_MODEL_PATH: str = os.environ.get('ULTRA_SAFE_MODEL_PATH', 'ultra_safe_detector.pt')
    CHATBOT_MODEL_PATH: str = os.environ.get('CHATBOT_MODEL_PATH', 'smart_ai_chatbot_v2.pt')

    # Image Folders
    ANOMALY_IMAGE_FOLDER: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'anomaly_images')
    BIRD_IMAGE_FOLDER: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'bird_images')

    # Detection Parameters
    BIRD_CLASS_ID: int = int(os.environ.get('BIRD_CLASS_ID', 14))
    BIRD_CONFIDENCE_THRESHOLD: float = float(os.environ.get('BIRD_CONFIDENCE_THRESHOLD', 0.6))
    ANOMALY_COOLDOWN: int = int(os.environ.get('ANOMALY_COOLDOWN', 10))  # seconds

    # Chatbot Parameters
    CHATBOT_PROVIDER: str = os.environ.get('CHATBOT_PROVIDER', 'ultra')
    CHATBOT_TIMEOUT: int = int(os.environ.get('CHATBOT_TIMEOUT', 10))

    # General Settings
    LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO')
    DEBUG_MODE: bool = os.environ.get('DEBUG_MODE', 'False') == 'True'

    @classmethod
    def validate(cls) -> bool:
        """
        Validate config for all AI modules. Print issues if found.
        Returns True if valid, False otherwise.
        """
        issues = []
        if not os.path.exists(cls.YOLO_MODEL_PATH):
            issues.append(f"YOLO model not found: {cls.YOLO_MODEL_PATH}")
        if not os.path.exists(cls.ULTRA_SAFE_MODEL_PATH):
            issues.append(f"Ultra Safe model not found: {cls.ULTRA_SAFE_MODEL_PATH}")
        if not 0 <= cls.BIRD_CONFIDENCE_THRESHOLD <= 1:
            issues.append("BIRD_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if not os.path.isdir(cls.ANOMALY_IMAGE_FOLDER):
            issues.append(f"Anomaly image folder missing: {cls.ANOMALY_IMAGE_FOLDER}")
        if not os.path.isdir(cls.BIRD_IMAGE_FOLDER):
            issues.append(f"Bird image folder missing: {cls.BIRD_IMAGE_FOLDER}")
        if issues:
            print("⚠️ Configuration Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        print("✅ Configuration validation passed")
        return True

    @classmethod
    def load_from_json(cls, path: str) -> None:
        """
        Load config from JSON file.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cls, k):
                    setattr(cls, k, v)
        except Exception as e:
            print(f"❌ Error loading config from JSON: {e}")

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """
        Return config as dictionary.
        """
        return {
            'SQLITE_DATABASE_URI': cls.SQLITE_DATABASE_URI,
            'YOLO_MODEL_PATH': cls.YOLO_MODEL_PATH,
            'ULTRA_SAFE_MODEL_PATH': cls.ULTRA_SAFE_MODEL_PATH,
            'CHATBOT_MODEL_PATH': cls.CHATBOT_MODEL_PATH,
            'ANOMALY_DB_PATH': cls.ANOMALY_DB_PATH,
            'BIRD_DB_PATH': cls.BIRD_DB_PATH,
            'ANOMALY_IMAGE_FOLDER': cls.ANOMALY_IMAGE_FOLDER,
            'BIRD_IMAGE_FOLDER': cls.BIRD_IMAGE_FOLDER,
            'BIRD_CLASS_ID': cls.BIRD_CLASS_ID,
            'BIRD_CONFIDENCE_THRESHOLD': cls.BIRD_CONFIDENCE_THRESHOLD,
            'ANOMALY_COOLDOWN': cls.ANOMALY_COOLDOWN,
            'CHATBOT_PROVIDER': cls.CHATBOT_PROVIDER,
            'CHATBOT_TIMEOUT': cls.CHATBOT_TIMEOUT,
            'LOG_LEVEL': cls.LOG_LEVEL,
            'DEBUG_MODE': cls.DEBUG_MODE,
        }

    # config.py
    import os
    import json

    class Config:
        """
        Central configuration for all AI modules:
        - Object/Anomaly Detection
        - Bird Detection/Counting
        - Chatbot
        """

        # Model Paths
        YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'yolov8n.pt')
        ULTRA_SAFE_MODEL_PATH = os.environ.get('ULTRA_SAFE_MODEL_PATH', 'ultra_safe_detector.pt')
        CHATBOT_MODEL_PATH = os.environ.get('CHATBOT_MODEL_PATH', 'smart_ai_chatbot_v2.pt')

        # Database Paths
        ANOMALY_DB_PATH = os.environ.get('ANOMALY_DB_PATH', 'anomaly_alerts.db')
        BIRD_DB_PATH = os.environ.get('BIRD_DB_PATH', 'swallow_smart_stats.db')

        # Image Folders
        ANOMALY_IMAGE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'anomaly_images')
        BIRD_IMAGE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'bird_images')

        # Detection Parameters
        BIRD_CLASS_ID = int(os.environ.get('BIRD_CLASS_ID', 14))
        BIRD_CONFIDENCE_THRESHOLD = float(os.environ.get('BIRD_CONFIDENCE_THRESHOLD', 0.6))
        ANOMALY_COOLDOWN = int(os.environ.get('ANOMALY_COOLDOWN', 10))  # seconds

        # Chatbot Parameters
        CHATBOT_PROVIDER = os.environ.get('CHATBOT_PROVIDER', 'ultra')
        CHATBOT_TIMEOUT = int(os.environ.get('CHATBOT_TIMEOUT', 10))

        # General Settings
        LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
        DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False') == 'True'

        @classmethod
        def validate(cls):
            """Validate config for all AI modules."""
            issues = []
            if not os.path.exists(cls.YOLO_MODEL_PATH):
                issues.append(f"YOLO model not found: {cls.YOLO_MODEL_PATH}")
            if not os.path.exists(cls.ULTRA_SAFE_MODEL_PATH):
                issues.append(f"Ultra Safe model not found: {cls.ULTRA_SAFE_MODEL_PATH}")
            if not 0 <= cls.BIRD_CONFIDENCE_THRESHOLD <= 1:
                issues.append("BIRD_CONFIDENCE_THRESHOLD must be between 0 and 1")
            if not os.path.isdir(cls.ANOMALY_IMAGE_FOLDER):
                issues.append(f"Anomaly image folder missing: {cls.ANOMALY_IMAGE_FOLDER}")
            if not os.path.isdir(cls.BIRD_IMAGE_FOLDER):
                issues.append(f"Bird image folder missing: {cls.BIRD_IMAGE_FOLDER}")


        @classmethod
        def load_from_json(cls, path):
            """Load config from JSON file."""
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(cls, k):
                        setattr(cls, k, v)
            except Exception as e:
                print(f"❌ Error loading config from JSON: {e}")

        @classmethod
        def as_dict(cls):
            """Return config as dictionary."""
            return {
                'YOLO_MODEL_PATH': cls.YOLO_MODEL_PATH,
                'ULTRA_SAFE_MODEL_PATH': cls.ULTRA_SAFE_MODEL_PATH,
                'CHATBOT_MODEL_PATH': cls.CHATBOT_MODEL_PATH,
                'ANOMALY_DB_PATH': cls.ANOMALY_DB_PATH,
                'BIRD_DB_PATH': cls.BIRD_DB_PATH,
                'ANOMALY_IMAGE_FOLDER': cls.ANOMALY_IMAGE_FOLDER,
                'BIRD_IMAGE_FOLDER': cls.BIRD_IMAGE_FOLDER,
                'BIRD_CLASS_ID': cls.BIRD_CLASS_ID,
                'BIRD_CONFIDENCE_THRESHOLD': cls.BIRD_CONFIDENCE_THRESHOLD,
                'ANOMALY_COOLDOWN': cls.ANOMALY_COOLDOWN,
                'CHATBOT_PROVIDER': cls.CHATBOT_PROVIDER,
                'CHATBOT_TIMEOUT': cls.CHATBOT_TIMEOUT,
                'LOG_LEVEL': cls.LOG_LEVEL,
                'DEBUG_MODE': cls.DEBUG_MODE,
            }
