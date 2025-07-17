import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from app.logs.logger import setup_logger
from app.config.manager import load_config

logger = setup_logger()
config = load_config()

minute_model = None
hourly_model = None
minute_scaler = None
hourly_scaler = None

def train_model(df, n_estimators=None, max_depth=None):
    """Обучение минутной модели с нормализацией и отладкой"""
    global minute_model, minute_scaler
    try:
        # Логируем входные данные
        logger.debug(f"train_model: Input dataframe shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Ограничение данных по окну обучения
        window_minutes = config["model"]["train_window_minutes"]
        df = df.tail(int(window_minutes))
        logger.debug(f"train_model: After window limit, shape: {df.shape}")
        
        if len(df) < config["model"]["min_candles"]:
            logger.warning(f"Insufficient data for training: {len(df)} candles, required: {config['model']['min_candles']}")
            return
        
        features = ["close", "rsi", "sma", "volume", "log_volume"]
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        
        logger.debug(f"train_model: Features shape: {X.shape}, Target shape: {y.shape}")
        
        if len(X) < config["model"]["min_candles"]:
            logger.warning(f"Too few valid samples: {len(X)}, required: {config['model']['min_candles']}")
            return
        
        # Проверка на NaN и бесконечности
        if X.isna().any().any() or np.any(np.isinf(X.values)):
            logger.error(f"NaN or Inf values found in features: {X.isna().sum()}")
            return
        
        # Проверка стандартного отклонения для нормализации
        if np.any(X.std() == 0):
            logger.warning(f"Zero standard deviation in features: {X.std()}")
            return
        
        # Нормализация данных
        minute_scaler = StandardScaler()
        X_scaled = minute_scaler.fit_transform(X)
        logger.debug(f"train_model: Data normalized, X_scaled shape: {X_scaled.shape}")
        
        # Преобразование max_depth: если 0 или null, то устанавливаем None
        max_depth = max_depth or config["model"]["max_depth"]
        if max_depth in (0, None):
            logger.info("max_depth is 0 or null, setting to None for unlimited tree depth")
            max_depth = None
        elif not isinstance(max_depth, (int, type(None))) or (isinstance(max_depth, int) and max_depth < 1):
            logger.warning(f"Invalid max_depth: {max_depth}, using default value 12")
            max_depth = 12
        
        model = RandomForestRegressor(
            n_estimators=n_estimators or config["model"]["n_estimators"],
            max_depth=max_depth,
            min_samples_split=config["model"].get("min_samples_split", 2),
            min_samples_leaf=config["model"].get("min_samples_leaf", 1),
            max_features=config["model"].get("max_features", "sqrt"),
            random_state=42
        )
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        minute_model = model
        logger.info(f"Minute model trained, R^2={r2:.4f}, samples={len(X)}")
    except Exception as e:
        logger.error(f"Error training minute model: {e}", exc_info=True)

def train_hourly_model(df):
    """Обучение часовой модели с нормализацией и отладкой"""
    global hourly_model, hourly_scaler
    try:
        # Логируем входные данные
        logger.debug(f"train_hourly_model: Input dataframe shape: {df.shape}, columns: {df.columns.tolist()}")
        
        window_minutes = config["model"]["hourly_train_window_minutes"]
        df = df.tail(int(window_minutes / 60))
        logger.debug(f"train_hourly_model: After window limit, shape: {df.shape}")
        
        if len(df) < config["model"]["min_hourly_candles"]:
            logger.warning(f"Insufficient data for hourly training: {len(df)} candles, required: {config['model']['min_hourly_candles']}")
            return
        
        features = ["close", "rsi", "sma", "volume", "log_volume"]
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        
        logger.debug(f"train_hourly_model: Features shape: {X.shape}, Target shape: {y.shape}")
        
        if len(X) < config["model"]["min_hourly_candles"]:
            logger.warning(f"Too few valid hourly samples: {len(X)}, required: {config['model']['min_hourly_candles']}")
            return
        
        if X.isna().any().any() or np.any(np.isinf(X.values)):
            logger.error(f"NaN or Inf values found in features: {X.isna().sum()}")
            return
        
        # Проверка стандартного отклонения для нормализации
        if np.any(X.std() == 0):
            logger.warning(f"Zero standard deviation in features: {X.std()}")
            return
        
        # Нормализация данных
        hourly_scaler = StandardScaler()
        X_scaled = hourly_scaler.fit_transform(X)
        logger.debug(f"train_hourly_model: Data normalized, X_scaled shape: {X_scaled.shape}")
        
        # Преобразование hourly_max_depth: если 0 или null, то устанавливаем None
        max_depth = config["model"]["hourly_max_depth"]
        if max_depth in (0, None):
            logger.info("hourly_max_depth is 0 or null, setting to None for unlimited tree depth")
            max_depth = None
        elif not isinstance(max_depth, (int, type(None))) or (isinstance(max_depth, int) and max_depth < 1):
            logger.warning(f"Invalid hourly_max_depth: {max_depth}, using default value 20")
            max_depth = 20
        
        model = RandomForestRegressor(
            n_estimators=config["model"]["hourly_n_estimators"],
            max_depth=max_depth,
            min_samples_split=config["model"].get("min_samples_split", 2),
            min_samples_leaf=config["model"].get("min_samples_leaf", 1),
            max_features=config["model"].get("max_features", "sqrt"),
            random_state=42
        )
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        minute_model = model
        logger.info(f"Minute model trained, R^2={r2:.4f}, samples={len(X)}")
    except Exception as e:
        logger.error(f"Error training minute model: {e}", exc_info=True)

def train_hourly_model(df):
    """Обучение часовой модели с нормализацией и отладкой"""
    global hourly_model, hourly_scaler
    try:
        # Логируем входные данные
        logger.debug(f"train_hourly_model: Input dataframe shape: {df.shape}, columns: {df.columns.tolist()}")
        
        window_minutes = config["model"]["hourly_train_window_minutes"]
        df = df.tail(int(window_minutes / 60))
        logger.debug(f"train_hourly_model: After window limit, shape: {df.shape}")
        
        if len(df) < config["model"]["min_hourly_candles"]:
            logger.warning(f"Insufficient data for hourly training: {len(df)} candles, required: {config['model']['min_hourly_candles']}")
            return
        
        features = ["close", "rsi", "sma", "volume", "log_volume"]
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        
        logger.debug(f"train_hourly_model: Features shape: {X.shape}, Target shape: {y.shape}")
        
        if len(X) < config["model"]["min_hourly_candles"]:
            logger.warning(f"Too few valid hourly samples: {len(X)}, required: {config['model']['min_hourly_candles']}")
            return
        
        if X.isna().any().any() or np.any(np.isinf(X.values)):
            logger.error(f"NaN or Inf values found in features: {X.isna().sum()}")
            return
        
        # Проверка стандартного отклонения для нормализации
        if np.any(X.std() == 0):
            logger.warning(f"Zero standard deviation in features: {X.std()}")
            return
        
        # Нормализация данных
        hourly_scaler = StandardScaler()
        X_scaled = hourly_scaler.fit_transform(X)
        logger.debug(f"train_hourly_model: Data normalized, X_scaled shape: {X_scaled.shape}")
        
        # Преобразование hourly_max_depth: если 0 или null, то устанавливаем None
        max_depth = config["model"]["hourly_max_depth"]
        if max_depth in (0, None):
            logger.info("hourly_max_depth is 0 or null, setting to None for unlimited tree depth")
            max_depth = None
        elif not isinstance(max_depth, (int, type(None))) or (isinstance(max_depth, int) and max_depth < 1):
            logger.warning(f"Invalid hourly_max_depth: {max_depth}, using default value 20")
            max_depth = 20
        
        model = RandomForestRegressor(
            n_estimators=config["model"]["hourly_n_estimators"],
            max_depth=max_depth,
            min_samples_split=config["model"].get("min_samples_split", 2),
            min_samples_leaf=config["model"].get("min_samples_leaf", 1),
            max_features=config["model"].get("max_features", "sqrt"),
            random_state=42
        )
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        hourly_model = model
        logger.info(f"Hourly model trained, R^2={r2:.4f}, samples={len(X)}")
    except Exception as e:
        logger.error(f"Error training hourly model: {e}", exc_info=True)

def predict(features):
    """Прогноз для минутной модели с нормализацией"""
    global minute_model, minute_scaler
    try:
        if minute_model is None or minute_scaler is None:
            logger.warning("Minute model or scaler not initialized")
            return None
        if features.isna().any().any() or np.any(np.isinf(features.values)):
            logger.error("NaN or Inf values in prediction features")
            return None
        features_scaled = minute_scaler.transform(features)
        return minute_model.predict(features_scaled)[0]
    except Exception as e:
        logger.error(f"Minute prediction error: {e}", exc_info=True)
        return None

def predict_hourly(features):
    """Прогноз для часовой модели с нормализацией"""
    global hourly_model, hourly_scaler
    try:
        if hourly_model is None or hourly_scaler is None:
            logger.warning("Hourly model or scaler not initialized")
            return None
        if features.isna().any().any() or np.any(np.isinf(features.values)):
            logger.error("NaN or Inf values in prediction features")
            return None
        features_scaled = hourly_scaler.transform(features)
        return hourly_model.predict(features_scaled)[0]
    except Exception as e:
        logger.error(f"Hourly prediction error: {e}", exc_info=True)
        return None