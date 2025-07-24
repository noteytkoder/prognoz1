import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from fivesec_app.logger import setup_logger
from fivesec_app.config_manager import load_config

config = load_config()
logger = setup_logger()
fivesec_model = None
fivesec_scaler = None

def train_fivesec_model(df):
    """Обучение 5-секундной модели"""
    global fivesec_model, fivesec_scaler
    try:
        logger.debug(f"train_fivesec_model: Input dataframe shape: {df.shape}, columns: {df.columns.tolist()}")
        
        window_seconds = config["model"]["fivesec_train_window_seconds"]
        df = df.tail(int(window_seconds / 5))
        logger.debug(f"train_fivesec_model: After window limit, shape: {df.shape}")
        
        if len(df) < config["model"]["min_fivesec_candles"]:
            logger.warning(f"Insufficient data for 5-sec training: {len(df)} candles, required: {config['model']['min_fivesec_candles']}")
            return
        
        features = ["close", "rsi", "sma", "volume", "log_volume"]
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        
        logger.debug(f"train_fivesec_model: Features shape: {X.shape}, Target shape: {y.shape}")
        
        if len(X) < config["model"]["min_fivesec_candles"]:
            logger.warning(f"Too few valid 5-sec samples: {len(X)}, required: {config['model']['min_fivesec_candles']}")
            return
        
        if X.isna().any().any() or np.any(np.isinf(X.values)):
            logger.error(f"NaN or Inf values found in features: {X.isna().sum()}")
            return
        
        if np.any(X.std() == 0):
            logger.warning(f"Zero standard deviation in features: {X.std()}")
            return
        
        fivesec_scaler = StandardScaler()
        X_scaled = fivesec_scaler.fit_transform(X)
        logger.debug(f"train_fivesec_model: Data normalized, X_scaled shape: {X_scaled.shape}")
        
        max_depth = config["model"]["fivesec_max_depth"]
        if max_depth in (0, None):
            logger.info("fivesec_max_depth is 0 or null, setting to None for unlimited tree depth")
            max_depth = None
        elif not isinstance(max_depth, (int, type(None))) or (isinstance(max_depth, int) and max_depth < 1):
            logger.warning(f"Invalid fivesec_max_depth: {max_depth}, using default value 15")
            max_depth = 15
        
        model = RandomForestRegressor(
            n_estimators=config["model"]["fivesec_n_estimators"],
            max_depth=max_depth,
            min_samples_split=config["model"].get("min_samples_split", 2),
            min_samples_leaf=config["model"].get("min_samples_leaf", 1),
            max_features=config["model"].get("max_features", "sqrt"),
            random_state=42
        )
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        fivesec_model = model
        logger.info(f"5-second model trained, R^2={r2:.4f}, samples={len(X)}")
    except Exception as e:
        logger.error(f"Error training 5-second model: {e}", exc_info=True)

def predict_fivesec(features):
    """Прогноз для 5-секундной модели"""
    global fivesec_model, fivesec_scaler
    try:
        if fivesec_model is None or fivesec_scaler is None:
            logger.warning("5-second model or scaler not initialized")
            return None
        if features.isna().any().any() or np.any(np.isinf(features.values)):
            logger.error("NaN or Inf values in prediction features")
            return None
        features_scaled = fivesec_scaler.transform(features)
        return fivesec_model.predict(features_scaled)[0]
    except Exception as e:
        logger.error(f"5-second prediction error: {e}", exc_info=True)
        return None