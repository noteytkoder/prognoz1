# app/data/handler.py
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
            logger.debug(f"Fetched {len(new_klines)} records for {range_type}, total: {len(klines)}")
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
            if len(df_hour) < config["model"]["min_hourly_candles"]:
                logger.warning(f"Too few hourly candles: {len(df_hour)}, required: {config['model']['min_hourly_candles']}")
            else:
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
                if all(key in item for key in ["timestamp", "open", "high", "low", "close", "volume"]):
                    with buffer_lock:
                        data_buffer.append(item)
                        logger.debug(f"Added new kline to data_buffer, timestamp: {timestamp}, buffer size: {len(data_buffer)}")
                else:
                    logger.error(f"Invalid data item: {item}")
                message_count += 1

                if message_count % 100 == 0:
                    logger.info(f"[consumer] Klines processed: {message_count}, buffer size: {len(data_buffer)}")

            elif name == "aggTrade":
                timestamp = process_timestamp(data["T"])
                quantity = float(data["q"])
                trade_buffer.append({"timestamp": timestamp, "quantity": quantity})

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
        if last_kline_time and time.time() - last_kline_time > 180:
            logger.error("[watchdog] No Kline data for 2 minutes — creating restart flag!")
            RESTART_FLAG.touch()
            os._exit(0)

cached_df = None
last_buffer_update = 0

def get_latest_features():
    global cached_df, last_buffer_update
    with buffer_lock:
        if len(data_buffer) < config["data"]["min_records"]:
            logger.warning(f"Insufficient data: {len(data_buffer)} records")
            return None
        
        # Проверяем, обновился ли буфер
        buffer_timestamp = data_buffer[-1]["timestamp"] if data_buffer else 0
        if buffer_timestamp == last_buffer_update and cached_df is not None:
            return cached_df["min"].iloc[-1], cached_df["hour"].iloc[-1], cached_df["min"]
        df = pd.DataFrame(data_buffer.copy())
        last_buffer_update = buffer_timestamp

    try:
        df = df.drop_duplicates(subset=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()

        interval = config["data"]["websocket_intervals"].get(config["data"]["download_range"], "1s")
        interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600, "1d": 86400}

        latest_timestamp = df.index.max()
        current_time = pd.Timestamp.now(tz=config.get("timezone", "Europe/Moscow"))
        if (current_time - latest_timestamp).total_seconds() > interval_seconds.get(interval, 1) * 20:
            logger.warning(f"Data is stale: latest_timestamp={latest_timestamp}, current_time={current_time}")
            return None

        df = df.resample(f"{interval_seconds.get(interval, 1)}s").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear").dropna()
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

        cached_df = {"min": df_min, "hour": df_hour}
        return df_min.iloc[-1], df_hour.iloc[-1], df_min

    except Exception as e:
        logger.error(f"Error in get_latest_features: {e}", exc_info=True)
        return None
  
async def update_errors_loop():
    logger.info("update_errors_loop started")
    msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))
    tolerance_seconds = {"min": 120, "hour": 600}

    global predictions
    last_data_buffer = None
    data_df = None

    while True:
        try:
            # Проверка данных в памяти
            if not predictions:
                await asyncio.sleep(1)
                continue

            # Загружаем данные из буфера
            with buffer_lock:
                current_data_buffer = list(data_buffer)
                if current_data_buffer != last_data_buffer:
                    data_df = pd.DataFrame(current_data_buffer)
                    last_data_buffer = current_data_buffer

            if data_df is None or data_df.empty:
                await asyncio.sleep(1)
                continue

            data_df["timestamp"] = pd.to_datetime(data_df["timestamp"])
            now = pd.Timestamp.now(tz=msk_tz)

            updated_count = 0

            # Обход списка предсказаний в памяти
            with prediction_file_lock:
                for prediction in predictions:
                    for pred_type in ["min", "hour"]:
                        actual_col = f"{pred_type}_actual_price"
                        error_col = f"{pred_type}_error"
                        pred_time_col = f"{pred_type}_pred_time"
                        pred_value_col = f"{pred_type}_pred"

                        # Уже посчитано
                        if prediction.get(actual_col) is not None:
                            continue

                        # Время предсказания
                        pred_time = prediction.get(pred_time_col)
                        if pred_time is None:
                            continue

                        pred_time = pd.to_datetime(pred_time).tz_convert(msk_tz)
                        if pred_time > now:
                            continue

                        # Находим ближайшее значение
                        time_diff = (data_df["timestamp"] - pred_time).abs()
                        if time_diff.empty or time_diff.isnull().all():
                            continue

                        min_diff = time_diff.min()
                        if pd.isna(min_diff):
                            continue

                        if min_diff.total_seconds() <= tolerance_seconds[pred_type]:
                            closest_idx = time_diff.idxmin()
                            actual_price = data_df.loc[closest_idx, "close"]

                            if prediction.get(actual_col) is None:
                                prediction[actual_col] = actual_price
                                prediction[error_col] = abs(actual_price - prediction[pred_value_col])
                                updated_count += 1

            if updated_count > 0:
                logger.debug(f"update_errors_loop: updated {updated_count} predictions in memory")

        except Exception as e:
            logger.error(f"Error in update_errors_loop: {e}", exc_info=True)

        await asyncio.sleep(1)

async def retrain_loop():
    """Периодическое переобучение моделей"""
    global last_train_time
    logger.info("retrain_loop started")
    train_interval = config["data"]["train_interval"]
    hourly_train_interval = hourly_train_interval = config["data"]["hourly_train_interval"]

    while True:
        try:
            current_time = time.time()
            with buffer_lock:
                if len(data_buffer) < config["data"]["min_records"]:
                    logger.warning(f"Insufficient data for retraining: {len(data_buffer)} records")
                    await asyncio.sleep(train_interval)
                    continue

                df = pd.DataFrame(data_buffer)
                df = df.drop_duplicates(subset=["timestamp"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = df.sort_index()
                df = calculate_indicators(df)

                # Минутная модель
                if current_time - last_train_time >= train_interval:
                    df_min = df.resample("1min").agg({
                        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                    }).interpolate(method="linear").dropna()
                    df_min = calculate_indicators(df_min)
                    if len(df_min) >= config["model"]["min_candles"]:
                        train_model(df_min)
                        last_train_time = current_time
                        logger.info("Minutely model retrained")
                    else:
                        logger.warning(f"Too few minute candles for retraining: {len(df_min)}")

                # Часовая модель
                if current_time - last_train_time >= hourly_train_interval:
                    df_hour = df.resample("1h").agg({
                        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                    }).interpolate(method="linear").dropna()
                    df_hour = calculate_indicators(df_hour)
                    if len(df_hour) >= config["model"]["min_hourly_candles"]:
                        train_hourly_model(df_hour)
                        logger.info("Hourly model retrained")
                    else:
                        logger.warning(f"Too few hourly candles for retraining: {len(df_hour)}")

        except Exception as e:
            logger.error(f"Error in retrain_loop: {e}", exc_info=True)

        await asyncio.sleep(train_interval)

async def prediction_loop():
    logger.info("prediction_loop started")
    global last_pred_time, predictions
    interval = config["data"]["websocket_intervals"].get(config["data"]["download_range"], "1s")
    interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600, "1d": 86400}
    wait_seconds = interval_seconds.get(interval, 1)
    max_predictions = 10000
    csv_file_path = "logs/predictions.csv"
    msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))

    logger.info(f"Starting prediction_loop with interval: {wait_seconds} seconds")

    # Инициализация файла с заголовками, если он не существует
    if not os.path.exists(csv_file_path):
        with prediction_file_lock:
            pd.DataFrame(columns=[
                "timestamp", "actual_price", "min_pred", "hour_pred", "min_error", "hour_error",
                "min_pred_time", "hour_pred_time", "min_change_pct", "hour_change_pct",
                "min_actual_price", "hour_actual_price"
            ]).to_csv(csv_file_path, index=False)
            logger.info(f"Initialized empty predictions.csv with headers")

    while True:
        start = time.time()
        try:
            result = get_latest_features()
            if result is None:
                logger.debug("prediction_loop: no features available this cycle")
            else:
                df_min_row, df_hour_row, df_min_df = result
                actual_price = df_min_row["close"]

                features = df_min_row[["close", "rsi", "sma", "volume", "log_volume"]]
                features_df = pd.DataFrame([features])

                min_prediction = predict(features_df)
                hour_prediction = predict_hourly(features_df)

                pred_timestamp = pd.Timestamp.now(tz=msk_tz)
                min_pred_time = pred_timestamp + pd.Timedelta(minutes=1)
                hour_pred_time = pred_timestamp + pd.Timedelta(hours=1)

                # Вычисляем проценты изменения
                min_change_pct = ((min_prediction - actual_price) / actual_price * 100) if actual_price > 0 else 0
                hour_change_pct = ((hour_prediction - actual_price) / actual_price * 100) if actual_price > 0 else 0
                min_change_str = f"{min_change_pct:+.2f}%"
                hour_change_str = f"{hour_change_pct:+.2f}%"

                if min_prediction is not None and hour_prediction is not None:
                    # Логируем в predictions.log
                    predictions_logger.info(
                        "",
                        extra={
                            "timestamp": str(pred_timestamp),
                            "actual_price": actual_price,
                            "min_pred": min_prediction,
                            "min_pred_time": min_pred_time.strftime('%Y-%m-%d %H:%M:%S%z'),
                            "min_change": min_change_str,
                            "hour_pred": hour_prediction,
                            "hour_pred_time": hour_pred_time.strftime('%Y-%m-%d %H:%M:%S%z'),
                            "hour_change": hour_change_str,
                        }
                    )

                    prediction_record = {
                        "timestamp": pred_timestamp,
                        "actual_price": actual_price,
                        "min_pred": min_prediction,
                        "hour_pred": hour_prediction,
                        "min_error": None,
                        "hour_error": None,
                        "min_pred_time": min_pred_time,
                        "hour_pred_time": hour_pred_time,
                        "min_change_pct": min_change_pct,
                        "hour_change_pct": hour_change_pct,
                        "min_actual_price": None,
                        "hour_actual_price": None
                    }
                    predictions.append(prediction_record)
                    if len(predictions) > max_predictions:
                        predictions = predictions[-max_predictions:]

                    # Атомарная запись с повторными попытками
                    retries = 3
                    for attempt in range(retries):
                        try:
                            with prediction_file_lock:
                                tmp_path = csv_file_path + ".tmp"
                                pd.DataFrame(predictions).to_csv(tmp_path, index=False, encoding='utf-8')
                                os.replace(tmp_path, csv_file_path)
                                logger.debug(f"Predictions saved to {csv_file_path}, size: {os.path.getsize(csv_file_path)} bytes")
                            break
                        except PermissionError as e:
                            logger.warning(f"PermissionError on attempt {attempt+1} in prediction_loop: {e}")
                            if attempt < retries - 1:
                                time.sleep(0.1)
                            else:
                                logger.error(f"Failed to save predictions after {retries} attempts: {e}")
                                raise

                last_pred_time = time.time()

        except Exception as e:
            logger.error(f"Error in prediction_loop: {e}", exc_info=True)
            
        elapsed = time.time() - start
        sleep_time = max(0, wait_seconds - elapsed)
        await asyncio.sleep(sleep_time)

async def start_binance_websocket():
    await check_network()
    raw_queue = asyncio.Queue(maxsize=10000)
    range_type = config["data"]["download_range"]
    interval = config["data"]["websocket_intervals"].get(range_type, "1s")
    kline_uri = f"wss://stream.binance.com:443/ws/btcusdt@kline_{interval}"
    trade_uri = "wss://stream.binance.com:443/ws/btcusdt@aggTrade"

    # Добавляем задачу retrain_loop
    tasks = [
        asyncio.create_task(producer_ws(kline_uri, "kline", raw_queue)),
        asyncio.create_task(producer_ws(trade_uri, "aggTrade", raw_queue)),
        asyncio.create_task(consumer_loop(raw_queue)),
        asyncio.create_task(watchdog()),
        asyncio.create_task(prediction_loop()),
        asyncio.create_task(update_errors_loop()),
        asyncio.create_task(retrain_loop())  # Новая задача для переобучения
    ]

    logger.info("All websocket tasks started")
    # Ждём все задачи и логируем ошибки, не падая всем циклом
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Task {idx} raised: {res}", exc_info=True)
    logger.error("start_binance_websocket exited, creating restart flag")
    RESTART_FLAG.touch()
    os._exit(0)