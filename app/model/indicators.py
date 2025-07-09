import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import numpy as np
from app.logs.logger import setup_logger
from app.config.manager import load_config

logger = setup_logger()
config = load_config()

def calculate_indicators(df):
    try:
        df["rsi"] = RSIIndicator(df["close"], window=config["indicators"]["rsi_window"]).rsi()
        df["sma"] = SMAIndicator(df["close"], window=config["indicators"]["sma_window"]).sma_indicator()
        df["log_volume"] = np.log1p(df["volume"])
        
        for col in ["rsi", "sma", "log_volume"]:
            if df[col].isna().all():
                logger.warning(f"All values in {col} are NaN, filling with last known value or 0")
                df[col] = df[col].fillna(method="ffill").fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mean())
        logger.info(f"Indicators calculated, rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df