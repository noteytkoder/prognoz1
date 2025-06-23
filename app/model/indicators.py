import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import numpy as np
from app.logs.logger import setup_logger
from app.config.manager import load_config

logger = setup_logger()
config = load_config()

def calculate_indicators(df):
    """Расчёт технических индикаторов (RSI, SMA, log_volume)"""
    try:
        df["rsi"] = RSIIndicator(df["close"], window=config["indicators"]["rsi_window"]).rsi()
        df["sma"] = SMAIndicator(df["close"], window=config["indicators"]["sma_window"]).sma_indicator()
        df["log_volume"] = np.log1p(df["volume"])
        df["rsi"] = df["rsi"].fillna(df["rsi"].mean())
        df["sma"] = df["sma"].fillna(df["sma"].mean())
        df["log_volume"] = df["log_volume"].fillna(df["log_volume"].mean())
        # logger.info(f"Indicators calculated, rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df