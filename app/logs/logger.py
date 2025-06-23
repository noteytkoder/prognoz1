import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    """Настройка логгера с ротацией файлов"""
    logger = logging.getLogger("TradingApp")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        handler = RotatingFileHandler(
            "logs/app.log",
            maxBytes=10*1024*1024*1024,  # 100 МБ
            backupCount=3
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger