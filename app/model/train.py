import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from app.logs.logger import setup_logger
from app.config.manager import load_config

logger = setup_logger()
config = load_config()

minute_model = None
hourly_model = None

def train_model(df, n_estimators=None, max_depth=None):
    """Обучение модели для минутного прогноза"""
    global minute_model
    try:
        if len(df) < config["model"]["min_candles"]:
            logger.warning(f"Insufficient data for training: {len(df)} candles")
            return
        features = ["close", "rsi", "sma", "volume", "log_volume"]
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        if len(X) < config["model"]["min_candles"]:
            logger.warning(f"Too few valid samples: {len(X)}")
            return
        model = RandomForestRegressor(
            n_estimators=n_estimators or config["model"]["n_estimators"],
            max_depth=max_depth or config["model"]["max_depth"],
            random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        minute_model = model
        logger.info(f"Minute model trained, R^2={r2:.4f}, samples={len(X)}")
    except Exception as e:
        logger.error(f"Error training minute model: {e}")

def train_hourly_model(df):
    """Обучение модели для почасового прогноза"""
    global hourly_model
    try:
        if len(df) < config["model"]["min_hourly_candles"]:
            logger.warning(f"Insufficient data for hourly training: {len(df)} candles")
            return
        features = ["close", "rsi", "sma", "volume", "log_volume"]  # Исправлено
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        if len(X) < config["model"]["min_hourly_candles"]:
            logger.warning(f"Too few valid hourly samples: {len(X)}")
            return
        model = RandomForestRegressor(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        hourly_model = model
        logger.info(f"Hourly model trained, R^2={r2:.4f}, samples={len(X)}")
    except Exception as e:
        logger.error(f"Error training hourly model: {e}")

def predict(features):
    """Прогноз для минутной модели"""
    global minute_model
    try:
        if minute_model is None:
            # logger.warning("Minute model not trained")
            return None
        return minute_model.predict(features)[0]
    except Exception as e:
        logger.error(f"Minute prediction error: {e}")
        return None

def predict_hourly(features):
    """Прогноз для почасовой модели"""
    global hourly_model
    try:
        if hourly_model is None:
            # logger.warning("Hourly model not trained")
            return None
        return hourly_model.predict(features)[0]
    except Exception as e:
        logger.error(f"Hourly prediction error: {e}")
        return None
