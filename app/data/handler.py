# handler.py
import pandas as pd
import websockets
import json
import asyncio
import time
import requests
import pytz
import threading
import os
from app.logs.logger import setup_logger
from app.model.train import train_model, train_hourly_model, predict, predict_hourly
from app.model.indicators import calculate_indicators
from app.config.manager import load_config
from threading import Lock
from app.logs.logger import setup_predictions_logger
from pathlib import Path

predictions_logger = setup_predictions_logger()
prediction_file_lock = Lock()
logger = setup_logger()
config = load_config()
data_buffer = []
trade_buffer = []
buffer_lock = threading.Lock()
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

async def monitor_websocket():
    """Периодическая проверка активности WebSocket"""
    global last_kline_time
    while True:
        current_time = time.time()
        diff = current_time - last_kline_time if last_kline_time else 0
        logger.debug(f"Monitor WebSocket: last_kline_time={last_kline_time}, current_time={current_time}, diff={diff}")
        if last_kline_time and diff > 120:  # Уменьшено до 2 минут для быстрого реагирования
            logger.error("No Kline data received for 2 minutes, restarting WebSocket")
            RESTART_FLAG.touch()
            os._exit(0)
        elif not last_kline_time:
            logger.debug("last_kline_time not initialized, waiting for first Kline data")
        await asyncio.sleep(30)

async def start_binance_websocket():
    global data_buffer, trade_buffer, last_train_time, last_kline_time
    await check_network()
    range_type = config["data"]["download_range"]
    interval = config["data"]["websocket_intervals"].get(range_type, "1s")
    if interval not in ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]:
        logger.error(f"Invalid interval: {interval}")
        interval = "1s"
    kline_uri = f"wss://stream.binance.com:443/ws/btcusdt@kline_{interval}"
    trade_uri = "wss://stream.binance.com:443/ws/btcusdt@aggTrade"
    message_count = 0
    interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600, "1d": 86400}
    reconnect_delay = 5
    max_reconnect_delay = 60
    max_gap_seconds = 300

    async def handle_kline():
        global last_kline_time
        nonlocal message_count, reconnect_delay
        last_timestamp = None
        while True:
            try:
                async with websockets.connect(kline_uri, ping_interval=20, ping_timeout=30, max_size=None) as ws:
                    reconnect_delay = 5
                    logger.info(f"Kline WebSocket connected to {kline_uri}")
                    while True:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=60)  # Таймаут 1 минута
                            data = json.loads(message)
                            if "k" not in data:
                                logger.debug("Skipping non-kline message")
                                continue
                            kline = data["k"]
                            timestamp = process_timestamp(kline["t"])
                            close_price = float(kline["c"])
                            
                            with buffer_lock:
                                # Удаляем дубликаты с таким же timestamp
                                data_buffer[:] = [d for d in data_buffer if d["timestamp"] != timestamp]
                                
                                cutoff_time = timestamp - pd.Timedelta(seconds=config["data"]["buffer_size"] * interval_seconds.get(interval, 1))
                                data_buffer[:] = [d for d in data_buffer if d["timestamp"] >= cutoff_time]
                                
                                data_buffer.append({
                                    "timestamp": timestamp,
                                    "open": float(kline["o"]),
                                    "high": float(kline["h"]),
                                    "low": float(kline["l"]),
                                    "close": close_price,
                                    "volume": float(kline["v"])
                                })
                                last_timestamp = timestamp
                                last_kline_time = time.time()
                                logger.debug(f"Added kline to data_buffer: timestamp={timestamp}, close={close_price}, buffer_size={len(data_buffer)}")
                            
                            message_count += 1
                            if message_count % 100 == 0:
                                with buffer_lock:
                                    total_records = len(data_buffer)
                                logger.info(f"Received {total_records} kline records, last_close: {close_price}")
                            
                            with buffer_lock:
                                if len(data_buffer) > config["data"]["buffer_size"] * 1.5:
                                    logger.warning(f"Data buffer size {len(data_buffer)} exceeds limit, clearing old data")
                                    data_buffer[:] = data_buffer[-config["data"]["buffer_size"]:]
                        
                        except asyncio.TimeoutError:
                            logger.error("Kline WebSocket timeout after 60 seconds, reconnecting...")
                            raise
                        except Exception as e:
                            logger.error(f"Kline WebSocket inner error: {e}")
                            raise
            except Exception as e:
                logger.error(f"Kline WebSocket error: {e}, reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def handle_agg_trade():
        nonlocal reconnect_delay
        while True:
            try:
                async with websockets.connect(trade_uri, ping_interval=20, ping_timeout=30, max_size=None) as ws:
                    reconnect_delay = 5
                    logger.info(f"AggTrade WebSocket connected to {trade_uri}")
                    while True:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=60)  # Таймаут 1 минута
                            data = json.loads(message)
                            timestamp = process_timestamp(data["T"])
                            quantity = float(data["q"])
                            with buffer_lock:
                                trade_buffer.append({"timestamp": timestamp, "quantity": quantity})
                                if len(trade_buffer) > config["data"]["buffer_size"]:
                                    trade_buffer.pop(0)
                                trade_df = pd.DataFrame(trade_buffer)
                                trade_df = trade_df.groupby(trade_df["timestamp"].dt.floor("s"))["quantity"].sum().reset_index()
                                for _, row in trade_df.iterrows():
                                    for item in data_buffer:
                                        if abs((item["timestamp"] - row["timestamp"]).total_seconds()) < interval_seconds.get(interval, 1):
                                            item["volume"] = row["quantity"]
                                            # logger.debug(f"Updated volume for timestamp={item['timestamp']}, volume={row['quantity']}")
                                            break
                        except asyncio.TimeoutError:
                            logger.error("AggTrade WebSocket timeout after 60 seconds, reconnecting...")
                            raise
                        except Exception as e:
                            logger.error(f"AggTrade WebSocket inner error: {e}")
                            raise
            except Exception as e:
                logger.error(f"AggTrade WebSocket error: {e}, reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    
    tasks = [handle_kline(), handle_agg_trade(), monitor_websocket()]  # Убедимся, что monitor_websocket включена
    
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
