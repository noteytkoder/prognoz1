import pandas as pd
import websockets
import json
import asyncio
import time
import requests
import pytz
import threading
import os
from collections import deque
from app.logs.logger import setup_logger, setup_predictions_logger
from app.model.train import train_model, train_hourly_model, predict, predict_hourly
from app.model.indicators import calculate_indicators
from app.config.manager import load_config
from threading import Lock
from pathlib import Path

predictions_logger = setup_predictions_logger()
prediction_file_lock = Lock()
logger = setup_logger()
config = load_config()

# Buffers using deque with maxlen
data_buffer = deque(maxlen=config["data"]["buffer_size"])
trade_buffer = deque(maxlen=config["data"]["buffer_size"])
raw_queue = asyncio.Queue()

last_train_time = 0
predictions = []
last_pred_time = 0
last_kline_time = 0
RESTART_FLAG = Path("restart.flag")

def process_timestamp(timestamp_ms):
    """Преобразование времени в MSK"""
    utc_time = pd.to_datetime(timestamp_ms / 1000, unit="s", utc=True)
    msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))
    return utc_time.astimezone(msk_tz)

async def fetch_historical_data(range_type="1day", start_time=None, end_time=None):
    """Загрузка исторических данных с Binance"""
    global data_buffer
    try:
        ranges = {"1hour": 60*60*1000, "1day": 24*60*60*1000, "1month": 30*24*60*60*1000}
        intervals = config["data"]["websocket_intervals"]
        interval = intervals.get(range_type, "1s")
        if interval not in ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]:
            logger.error(f"Invalid interval: {interval}")
            interval = "1s"
        range_ms = ranges.get(range_type, ranges["1day"])
        interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600, "1d": 86400}
        expected_records = range_ms // (interval_seconds.get(interval, 1) * 1000)
        
        end_time = end_time or int(time.time() * 1000)
        start_time = start_time or (end_time - range_ms)
        limit = 1000
        klines = []
        last_timestamp = None
        
        while start_time < end_time:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}"
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded, sleeping for 60 seconds")
                    await asyncio.sleep(60)
                    continue
                raise
            new_klines = response.json()
            if not new_klines:
                logger.warning(f"No more data for {range_type} at start_time={start_time}")
                break
            for kline in new_klines:
                if last_timestamp is None or kline[0] > last_timestamp:
                    klines.append(kline)
                    last_timestamp = kline[0]
            logger.info(f"Fetched {len(new_klines)} records for {range_type}, total: {len(klines)}")
            start_time = last_timestamp + (interval_seconds.get(interval, 1) * 1000)
            await asyncio.sleep(0.5)
        
        if not klines:
            logger.error(f"No historical data fetched for {range_type}")
            return
        
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = df["timestamp"].apply(process_timestamp)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        
        df = df.drop_duplicates(subset=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        gaps = df.index.to_series().diff().dt.total_seconds()
        if gaps.max() > interval_seconds.get(interval, 1):
            logger.warning(f"Detected gaps in historical data: max gap {gaps.max()} seconds")
        df = df.interpolate(method="linear")
        
        logger.info(f"Fetched {len(df)} historical records for {range_type}, expected: {expected_records}")
        if len(df) < expected_records * 0.8:
            logger.warning(f"Insufficient data: got {len(df)} records, expected {expected_records}")
        
        with buffer_lock:
            data_buffer.clear()
            data_buffer.extend(df.reset_index().to_dict("records"))
            logger.info(f"Data buffer updated with {len(data_buffer)} records")
        
        if len(data_buffer) >= config["data"]["min_records"]:
            with buffer_lock:
                df = pd.DataFrame(data_buffer)
            df = df.drop_duplicates(subset=["timestamp"])
            df.set_index("timestamp", inplace=True)
            df = df.sort_index()
            df = calculate_indicators(df)
            
            df_min = df.resample("1min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).interpolate(method="linear")
            df_min = calculate_indicators(df_min)
            train_model(df_min)
            
            df_hour = df.resample("1h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).interpolate(method="linear")
            df_hour = calculate_indicators(df_hour)
            train_hourly_model(df_hour)
            
            logger.info("Initial models trained")
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")

async def check_network():
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        if response.status_code == 200:
            logger.info("Network connection to Binance API is stable")
            return True
        else:
            logger.warning("Network connection to Binance API failed")
            return False
    except Exception as e:
        logger.error(f"Network check failed: {e}")
        return False

# Producer: listens to WebSocket and puts raw JSON into queue
async def producer_ws(uri, name, raw_queue):
    global last_kline_time
    reconnect_delay = 5
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=30) as ws:
                logger.info(f"[{name}] Connected to {uri}")
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    await raw_queue.put((name, raw))
                    if name == "kline":
                        last_kline_time = time.time()
        except Exception as e:
            logger.error(f"[{name}] Error: {e}. Reconnecting in {reconnect_delay}s")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)


# Consumer: processes queue into buffer
async def consumer_loop(raw_queue):
    message_count = 0
    interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600, "1d": 86400}
    interval = config["data"]["websocket_intervals"].get(config["data"]["download_range"], "1s")
    last_timestamp = None

    while True:
        try:
            name, raw = await raw_queue.get()
            data = json.loads(raw)

            if name == "kline":
                if "k" not in data:
                    continue
                k = data["k"]
                timestamp = process_timestamp(k["t"])
                item = {
                    "timestamp": timestamp,
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"])
                }

                if last_timestamp and (timestamp - last_timestamp).total_seconds() > interval_seconds.get(interval, 1) * 2:
                    logger.warning(f"[consumer] GAP DETECTED: {timestamp} vs {last_timestamp}")

                last_timestamp = timestamp
                data_buffer.append(item)
                message_count += 1

                if message_count % 100 == 0:
                    logger.info(f"[consumer] Klines processed: {message_count}, buffer size: {len(data_buffer)}")

            elif name == "aggTrade":
                timestamp = process_timestamp(data["T"])
                quantity = float(data["q"])
                trade_buffer.append({"timestamp": timestamp, "quantity": quantity})

                # Update volumes in data_buffer
                for candle in reversed(data_buffer):
                    if abs((candle["timestamp"] - timestamp).total_seconds()) < interval_seconds.get(interval, 1):
                        candle["volume"] = quantity
                        break

        except Exception as e:
            logger.error(f"[consumer] Error: {e}")


# Watchdog: monitors if kline data stops arriving
async def watchdog():
    global last_kline_time
    while True:
        await asyncio.sleep(30)
        if last_kline_time and time.time() - last_kline_time > 120:
            logger.error("[watchdog] No Kline data for 2 minutes — creating restart flag!")
            RESTART_FLAG.touch()
            os._exit(0)

async def start_binance_websocket():
    await check_network()

    raw_queue = asyncio.Queue()

    range_type = config["data"]["download_range"]
    interval = config["data"]["websocket_intervals"].get(range_type, "1s")
    kline_uri = f"wss://stream.binance.com:443/ws/btcusdt@kline_{interval}"
    trade_uri = "wss://stream.binance.com:443/ws/btcusdt@aggTrade"

    tasks = [
        producer_ws(kline_uri, "kline", raw_queue),
        producer_ws(trade_uri, "aggTrade", raw_queue),
        consumer_loop(raw_queue),
        watchdog(),
    ]
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        RESTART_FLAG.touch()
        os._exit(0)


def get_latest_features(forecast_range="1min"):
    """Получение последних признаков для прогноза"""
    global predictions, last_pred_time
    with buffer_lock:
        if len(data_buffer) < config["data"]["min_records"]:
            logger.warning(f"Insufficient data: {len(data_buffer)} records")
            return None
        df = pd.DataFrame(data_buffer.copy())
    
    try:
        df = df.drop_duplicates(subset=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        interval = config["data"]["websocket_intervals"].get(config["data"]["download_range"], "1s")
        interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600, "1d": 86400}

        latest_timestamp = df.index.max()
        current_time = pd.Timestamp.now(tz=config.get("timezone", "Europe/Moscow"))
        if (current_time - latest_timestamp).total_seconds() > interval_seconds.get(interval, 1) * 2:
            logger.warning(f"Data is stale: latest_timestamp={latest_timestamp}, current_time={current_time}, data_buffer_last={data_buffer[-5:]}")
            return None

        df = df.resample(f"{interval_seconds.get(interval, 1)}s").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear").dropna()
        logger.debug(f"Resampler shape: {df.shape}, last_timestamp={df.index.max()}")

        df = calculate_indicators(df)
        
        df_min = df.resample("1min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear").dropna()
        df_min = calculate_indicators(df_min)
        
        df_hour = df.resample("1h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear").dropna()
        df_hour = calculate_indicators(df_hour)
        
        if len(df_min) < config["model"]["min_candles"] or len(df_hour) < config["model"]["min_hourly_candles"]:
            logger.warning(f"Too few candles: min={len(df_min)}, hour={len(df_hour)}")
            return None
        
        actual_price = df_min.iloc[-1]["close"]
        current_time = time.time()
        if current_time - last_pred_time >= interval_seconds.get(interval, 1):
            features = df_min.iloc[-1][["close", "rsi", "sma", "volume", "log_volume"]]
            features_df = pd.DataFrame([features])
            
            min_prediction = predict(features_df)
            hour_prediction = predict_hourly(features_df)
            
            pred_timestamp = pd.Timestamp.now(tz=config.get("timezone", "Europe/Moscow"))
            if min_prediction is not None and hour_prediction is not None:
                predictions_logger.info("", extra={"min_pred": min_prediction, "hour_pred": hour_prediction})
                
                predictions.append({
                    "timestamp": pred_timestamp,
                    "actual_price": actual_price,
                    "predicted_price": min_prediction if forecast_range == "1min" else hour_prediction,
                    "error5": abs(actual_price - (min_prediction if forecast_range == "1min" else hour_prediction))
                })
                with prediction_file_lock:
                    pred_file = "predictions_minute.csv" if forecast_range == "1min" else "predictions_hourly.csv"
                    pd.DataFrame(predictions).to_csv(pred_file, index=False)
                logger.info(f"Prediction saved to {pred_file}: actual={actual_price}, predicted={(min_prediction if forecast_range == '1min' else hour_prediction)}")
            last_pred_time = current_time
        
        return df_min.iloc[-1] if forecast_range == "1min" else df_hour.iloc[-1]
    except Exception as e:
        logger.error(f"Error in get_latest_features: {e}", exc_info=True)
        return None
