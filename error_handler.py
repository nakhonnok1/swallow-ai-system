# error_handler.py - ระบบจัดการ Error ที่แข็งแกร่ง
import logging
import traceback
from datetime import datetime
from functools import wraps

class ErrorHandler:
    def __init__(self, log_file="error.log"):
        self.logger = logging.getLogger("ErrorHandler")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.ERROR)

    def log_error(self, error, context="Unknown"):
        """บันทึก Error พร้อม Context"""
        error_msg = f"Context: {context} | Error: {str(error)} | Traceback: {traceback.format_exc()}"
        self.logger.error(error_msg)
        print(f"🚨 ERROR: {error_msg}")

    def safe_execute(self, func, *args, **kwargs):
        """Execute function safely with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log_error(e, f"Function: {func.__name__}")
            return None

def error_handler_decorator(error_handler_instance):
    """Decorator สำหรับ wrap functions ด้วย error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler_instance.log_error(e, f"Function: {func.__name__}")
                return None
        return wrapper
    return decorator

# สร้าง instance สำหรับใช้งาน
error_handler = ErrorHandler()
safe_execute = error_handler_decorator(error_handler)
