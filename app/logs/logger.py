# app/logs/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    """Настройка основного логгера с ротацией файлов"""
    logger = logging.getLogger("TradingApp")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        handler = RotatingFileHandler(
            "logs/app.log",
            maxBytes=10*1024*1024,  # 10 МБ
            backupCount=3
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def setup_predictions_logger():
    """Настройка логгера для прогнозов"""
    logger = logging.getLogger("PredictionsLogger")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        handler = RotatingFileHandler(
            "logs/predictions.log",
            maxBytes=10*1024*1024,  # 10 МБ
            backupCount=3
        )
        formatter = logging.Formatter("%(asctime)s, прогноз_1мин=%(min_pred)f, прогноз_1час=%(hour_pred)f")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger