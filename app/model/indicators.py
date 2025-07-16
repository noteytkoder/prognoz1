import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from app.logs.logger import setup_logger
from app.config.manager import load_config

logger = setup_logger()
config = load_config()

def calculate_indicators(df):
    """Рассчет индикаторов RSI, SMA, log_volume как в оффлайн-версии"""
    try:
        df = df.copy()
        df["rsi"] = RSIIndicator(df["close"], window=config["indicators"]["rsi_window"]).rsi()
        df["sma"] = SMAIndicator(df["close"], window=config["indicators"]["sma_window"]).sma_indicator()
        df["log_volume"] = np.log1p(df["volume"].replace(0, 1e-8))

        # Заполнение пропусков средними значениями
        df["rsi"] = df["rsi"].fillna(df["rsi"].mean())
        df["sma"] = df["sma"].fillna(df["sma"].mean())
        df["log_volume"] = df["log_volume"].fillna(df["log_volume"].mean())

        # Проверка на аномалии
        if df.empty or df[["rsi", "sma", "log_volume"]].isnull().any().any() or np.any(np.isinf(df[["rsi", "sma", "log_volume"]].values)):
            logger.error("Invalid indicator values detected: NaN or Inf found")
            return None

        logger.debug(f"Indicators calculated, rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None