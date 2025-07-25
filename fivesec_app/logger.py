import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(log_dir="logs"):
    """Настройка основного логгера"""
    logger = logging.getLogger("FiveSecAppLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(log_dir, "fivesec_app.log"),
            maxBytes=100*1024*1024,  # 100 MB
            backupCount=3
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def setup_predictions_logger(log_dir="logs"):
    """Настройка логгера для прогнозов"""
    logger = logging.getLogger("FiveSecPredictionsLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(log_dir, "fivesec_predictions.log"),
            maxBytes=100*1024*1024,  # 100 MB
            backupCount=3
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger