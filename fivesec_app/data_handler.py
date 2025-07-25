import pandas as pd
import numpy as np
import websockets
import json
import asyncio
import time
import requests
import pytz
from collections import deque
from threading import Lock
from fivesec_app.logger import setup_logger, setup_predictions_logger
from fivesec_app.config_manager import load_config
from pathlib import Path
import os

config = load_config()
logger = setup_logger(log_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"))
buffer_lock = Lock()
fivesec_buffer = deque(maxlen=config["data"]["buffer_size"])
fivesec_predictions = []
fivesec_prediction_file_lock = Lock()
last_fivesec_train_time = time.time()

def process_timestamp(ms_timestamp):
    """Преобразование миллисекундного таймстемпа в datetime"""
    return pd.to_datetime(ms_timestamp, unit="ms", utc=True).tz_convert(config.get("timezone", "Europe/Moscow"))

def calculate_indicators(df):
    """Расчет индикаторов RSI, SMA, log_volume"""
    try:
        df["rsi"] = compute_rsi(df["close"], config["model"].get("rsi_window", 7))
        df["sma"] = df["close"].rolling(window=config["model"].get("sma_window", 3)).mean()
        df["log_volume"] = np.log1p(df["volume"])
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def compute_rsi(data, periods=7):
    """Расчет RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_data_for_model(df, interval="5s"):
    """Ресэмплинг данных до указанного интервала для модели"""
    try:
        # Убедимся, что индекс — DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            else:
                logger.error("No 'timestamp' column found in DataFrame")
                return None
        df = df.resample(interval).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear").dropna()
        df = calculate_indicators(df)
        return df
    except Exception as e:
        logger.error(f"Error processing data for interval {interval}: {e}", exc_info=True)
        return None

async def fetch_fivesec_historical_data():
    """Загрузка исторических данных (1 час, интервал 1s)"""
    global fivesec_buffer
    try:
        range_type = "1hour"
        ranges = {"1hour": 60*60*1000}
        interval = "1s"
        range_ms = ranges[range_type]
        interval_seconds = {"1s": 1}
        expected_records = range_ms // (interval_seconds[interval] * 1000)
        
        end_time = int(time.time() * 1000)
        start_time = end_time - range_ms
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
            start_time = last_timestamp + (interval_seconds[interval] * 1000)
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
        
        df = df.interpolate(method="linear")
        if df[["open", "high", "low", "close", "volume"]].isna().any().any() or \
           np.any(np.isinf(df[["open", "high", "low", "close", "volume"]].values)) or \
           (df[["open", "high", "low", "close", "volume"]] < 0).any().any():
            logger.error("Invalid data detected in 5-second historical data")
            return
        
        logger.info(f"Fetched {len(df)} historical records for {range_type}, expected: {expected_records}")
        if len(df) < expected_records * 0.8:
            logger.warning(f"Insufficient data: got {len(df)} records, expected {expected_records}")
        
        with buffer_lock:
            fivesec_buffer.clear()
            fivesec_buffer.extend(df.reset_index().to_dict("records"))
            logger.info(f"5-second buffer updated with {len(fivesec_buffer)} records")
        
        if len(fivesec_buffer) >= config["data"]["min_records"]:
            with buffer_lock:
                df = pd.DataFrame(fivesec_buffer)
            df = df.drop_duplicates(subset=["timestamp"])
            df.set_index("timestamp", inplace=True)
            df = df.sort_index()
            df = process_data_for_model(df, interval="5s")
            if df is not None:
                from .model import train_fivesec_model
                train_fivesec_model(df)
                logger.info("Initial 5-second model trained")
    except Exception as e:
        logger.error(f"Error fetching 5-second historical data: {e}", exc_info=True)

async def producer_ws(uri, name, queue):
    """WebSocket-продюсер для Binance"""
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                logger.info(f"WebSocket {name} connected")
                while True:
                    message = await websocket.recv()
                    await queue.put((name, message))
        except Exception as e:
            logger.error(f"WebSocket {name} error: {e}")
            await asyncio.sleep(5)

async def consumer_loop(raw_queue):
    """Обработка данных из WebSocket"""
    global fivesec_buffer
    message_count = 0
    interval_seconds = {"1s": 1}
    last_fivesec_timestamp = None

    while True:
        try:
            name, raw = await raw_queue.get()
            data = json.loads(raw)
            if name == "fivesec_kline" and "k" in data:
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

                if last_fivesec_timestamp and (timestamp - last_fivesec_timestamp).total_seconds() > interval_seconds["1s"] * 2:
                    logger.warning(f"[consumer] GAP DETECTED in fivesec_kline: {timestamp} vs {last_fivesec_timestamp}")

                last_fivesec_timestamp = timestamp
                if all(key in item for key in ["timestamp", "open", "high", "low", "close", "volume"]):
                    with buffer_lock:
                        fivesec_buffer.append(item)
                        logger.debug(f"Added new kline to fivesec_buffer, timestamp: {timestamp}, buffer size: {len(fivesec_buffer)}")
                else:
                    logger.error(f"Invalid 5-sec data item: {item}")
                message_count += 1

                if message_count % 100 == 0:
                    logger.info(f"[consumer] Klines processed: {message_count}, buffer size: {len(fivesec_buffer)}")
        except Exception as e:
            logger.error(f"[consumer] Error: {e}")

async def fivesec_prediction_loop(root_dir):
    """Цикл предсказаний для 5-секундной модели"""
    from .model import predict_fivesec
    logger.info("fivesec_prediction_loop started")
    global fivesec_predictions
    predictions_logger = setup_predictions_logger(log_dir=os.path.join(root_dir, "logs"))
    interval = "1s"
    interval_seconds = {"1s": 1}
    wait_seconds = interval_seconds[interval]
    max_predictions = 10000
    csv_file_path = os.path.join(root_dir, "logs", "fivesec_predictions.csv")
    msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))

    # Удаляем существующий файл прогнозов и создаём новый
    os.makedirs(os.path.join(root_dir, "logs"), exist_ok=True)
    try:
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
            logger.info(f"Removed existing fivesec_predictions.csv at {csv_file_path}")
        with fivesec_prediction_file_lock:
            pd.DataFrame(columns=[
                "timestamp", "actual_price", "fivesec_pred", "fivesec_error",
                "fivesec_pred_time", "fivesec_change_pct", "fivesec_actual_price"
            ]).to_csv(csv_file_path, index=False, encoding='utf-8')
            logger.info(f"Initialized new fivesec_predictions.csv at {csv_file_path}")
    except Exception as e:
        logger.error(f"Failed to initialize fivesec_predictions.csv: {e}", exc_info=True)
        return

    while True:
        start = time.time()
        try:
            with buffer_lock:
                if len(fivesec_buffer) < config["data"]["min_records"]:
                    logger.debug(f"fivesec_prediction_loop: insufficient data, buffer size={len(fivesec_buffer)}")
                    await asyncio.sleep(wait_seconds)
                    continue
                df = pd.DataFrame(fivesec_buffer)
                df = df.drop_duplicates(subset=["timestamp"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = df.sort_index()
                df = process_data_for_model(df, interval="5s")
                if df is None or df.empty:
                    logger.debug("fivesec_prediction_loop: failed to process data or empty dataframe")
                    await asyncio.sleep(wait_seconds)
                    continue
                latest_row = df.iloc[-1]
                actual_price = latest_row["close"]
                features = latest_row[["close", "rsi", "sma", "volume", "log_volume"]]
                features_df = pd.DataFrame([features])

            fivesec_prediction = predict_fivesec(features_df)
            if fivesec_prediction is None:
                logger.warning("fivesec_prediction_loop: prediction is None")
                await asyncio.sleep(wait_seconds)
                continue

            pred_timestamp = pd.Timestamp.now(tz=msk_tz)
            fivesec_pred_time = pred_timestamp + pd.Timedelta(seconds=5)
            fivesec_change_pct = ((fivesec_prediction - actual_price) / actual_price * 100) if actual_price > 0 else 0
            fivesec_change_str = f"{fivesec_change_pct:+.2f}%"

            predictions_logger.info(
                f"время={pred_timestamp}, цена={actual_price:.4f}, "
                f"прогноз_на_5сек={fivesec_prediction:.4f}, целевое_время_5сек={fivesec_pred_time.strftime('%Y-%m-%d %H:%M:%S%z')}, "
                f"отклонение_5сек={fivesec_change_str}"
            )

            prediction_record = {
                "timestamp": pred_timestamp,
                "actual_price": actual_price,
                "fivesec_pred": fivesec_prediction,
                "fivesec_error": None,
                "fivesec_pred_time": fivesec_pred_time,
                "fivesec_change_pct": fivesec_change_pct,
                "fivesec_actual_price": None
            }
            fivesec_predictions.append(prediction_record)
            if len(fivesec_predictions) > max_predictions:
                fivesec_predictions = fivesec_predictions[-max_predictions:]

            retries = 3
            for attempt in range(retries):
                try:
                    with fivesec_prediction_file_lock:
                        pd.DataFrame([prediction_record]).to_csv(
                            csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False, encoding='utf-8'
                        )
                        logger.debug(f"5-second prediction appended to {csv_file_path}, size: {os.path.getsize(csv_file_path)} bytes")
                    break
                except PermissionError as e:
                    logger.warning(f"PermissionError on attempt {attempt+1} in fivesec_prediction_loop: {e}")
                    if attempt < retries - 1:
                        time.sleep(0.1)
                    else:
                        logger.error(f"Failed to append prediction to {csv_file_path} after {retries} attempts: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error while saving to {csv_file_path}: {e}", exc_info=True)
                    break

        except Exception as e:
            logger.error(f"Error in fivesec_prediction_loop: {e}", exc_info=True)

        elapsed = time.time() - start
        sleep_time = max(0, wait_seconds - elapsed)
        await asyncio.sleep(sleep_time)

async def fivesec_retrain_loop():
    """Периодическое переобучение 5-секундной модели"""
    from .model import train_fivesec_model
    global last_fivesec_train_time
    logger.info("fivesec_retrain_loop started")
    train_interval = config["data"].get("fivesec_train_interval", 60)

    while True:
        try:
            current_time = time.time()
            with buffer_lock:
                if len(fivesec_buffer) < config["data"]["min_records"]:
                    logger.warning(f"Insufficient data for 5-sec retraining: {len(fivesec_buffer)} records")
                    await asyncio.sleep(train_interval)
                    continue
                df = pd.DataFrame(fivesec_buffer)
                df = df.drop_duplicates(subset=["timestamp"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = df.sort_index()
                df = process_data_for_model(df, interval="5s")
                if df is None or df.empty:
                    logger.error("Failed to process data in fivesec_retrain_loop")
                    await asyncio.sleep(train_interval)
                    continue

                if current_time - last_fivesec_train_time >= train_interval:
                    if len(df) >= config["model"].get("min_fivesec_candles", 1):
                        train_fivesec_model(df)
                        last_fivesec_train_time = current_time
                        logger.info(f"5-second model retrained, samples={len(df)}")
                    else:
                        logger.warning(f"Too few 5-sec candles for retraining: {len(df)}")

            await asyncio.sleep(train_interval)
        except Exception as e:
            logger.error(f"Error in fivesec_retrain_loop: {e}", exc_info=True)
            await asyncio.sleep(train_interval)

async def update_fivesec_errors_loop(root_dir):
    """Обновление ошибок для 5-секундных прогнозов"""
    logger.info("update_fivesec_errors_loop started")
    msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))
    tolerance_seconds = {"fivesec": 10}
    global fivesec_predictions
    last_data_buffer = None
    data_df = None
    csv_file_path = os.path.join(root_dir, "logs", "fivesec_predictions.csv")

    while True:
        try:
            if not fivesec_predictions:
                await asyncio.sleep(1)
                continue

            with buffer_lock:
                current_data_buffer = list(fivesec_buffer)
                if current_data_buffer != last_data_buffer:
                    data_df = pd.DataFrame(current_data_buffer)
                    last_data_buffer = current_data_buffer

            if data_df is None or data_df.empty:
                await asyncio.sleep(1)
                continue

            data_df["timestamp"] = pd.to_datetime(data_df["timestamp"])
            now = pd.Timestamp.now(tz=msk_tz)
            updated_count = 0

            with fivesec_prediction_file_lock:
                for prediction in fivesec_predictions:
                    for pred_type in ["fivesec"]:
                        actual_col = f"{pred_type}_actual_price"
                        error_col = f"{pred_type}_error"
                        pred_time_col = f"{pred_type}_pred_time"
                        pred_value_col = f"{pred_type}_pred"

                        if prediction.get(actual_col) is not None:
                            continue

                        pred_time = prediction.get(pred_time_col)
                        if pred_time is None:
                            continue

                        pred_time = pd.to_datetime(pred_time).tz_convert(msk_tz)
                        if pred_time > now:
                            continue

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

                # Сохраняем обновлённые прогнозы в CSV
                if updated_count > 0:
                    try:
                        pd.DataFrame(fivesec_predictions).to_csv(
                            csv_file_path, index=False, encoding='utf-8'
                        )
                        logger.debug(f"Updated {updated_count} predictions in {csv_file_path}")
                    except Exception as e:
                        logger.error(f"Failed to save updated predictions to {csv_file_path}: {e}", exc_info=True)

            if updated_count > 0:
                logger.debug(f"update_fivesec_errors_loop: updated {updated_count} predictions in memory")

        except Exception as e:
            logger.error(f"Error in update_fivesec_errors_loop: {e}", exc_info=True)

        await asyncio.sleep(1)

async def start_binance_websocket(root_dir):
    """Запуск WebSocket и всех циклов"""
    raw_queue = asyncio.Queue(maxsize=10000)
    fivesec_kline_uri = f"wss://stream.binance.com:443/ws/btcusdt@kline_1s"

    tasks = [
        asyncio.create_task(producer_ws(fivesec_kline_uri, "fivesec_kline", raw_queue)),
        asyncio.create_task(consumer_loop(raw_queue)),
        asyncio.create_task(fivesec_prediction_loop(root_dir)),
        asyncio.create_task(fivesec_retrain_loop()),
        asyncio.create_task(update_fivesec_errors_loop(root_dir)),
    ]

    logger.info("All websocket tasks started")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Task {idx} raised: {res}", exc_info=True)
    logger.error("start_binance_websocket exited, creating restart flag")
    Path(os.path.join(root_dir, "fivesec_restart.flag")).touch()
    os._exit(0)