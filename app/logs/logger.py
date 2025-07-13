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
            maxBytes=100*10*1024*1024,
            backupCount=5
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def setup_predictions_logger():
    """Настройка логгера для прогнозов на русском языке"""
    logger = logging.getLogger("PredictionsLogger")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        handler = RotatingFileHandler(
            "logs/predictions.log",
            maxBytes=100*10*1024*1024,  
            backupCount=3
        )
        formatter = logging.Formatter(
            "время=%(timestamp)s, цена=%(actual_price).4f, "
            "прогноз_на_1мин=%(min_pred).4f, целевое_время_1мин=%(min_pred_time)s, отклонение_1мин=%(min_change)s, "
            "прогноз_на_1час=%(hour_pred).4f, целевое_время_1час=%(hour_pred_time)s, отклонение_1час=%(hour_change)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
