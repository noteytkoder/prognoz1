import pandas as pd
import requests
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import logging
import time
import os
from threading import Thread, Lock
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from app.config.manager import load_config, load_environment_config


config = load_config()
env_name = config["app_env"]
env_config = load_environment_config()

# Настройка логирования
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "offline.log"),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Конфигурация
CONFIG = {
    "indicators": {
        "rsi_window": 7,
        "sma_window": 3
    },
    "model": {
        "n_estimators": 100,
        "n_estimators_min": 100,
        "n_estimators_hour": 100,
        "max_depth": None,
        "max_depth_min": None,
        "max_depth_hour": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "min_candles": 100,
        "min_hourly_candles": 24,
        "training_window_minutes_min": 60,    # Окно обучения для минутной модели (1 час)
        "retrain_every_n_steps_min": 1,      
        "training_window_minutes_hour": 1440, # Окно обучения для часовой модели (1 день)
        "retrain_every_n_steps_hour": 60
    },
    "visual": {
        "real_price_color": "blue",
        "predicted_price_color": "red",
        "error_band_color": "rgba(255,165,0,0.2)",
        "error_band_min": 50,
        "error_band_multiplier": 1.5,
        "update_interval": 1000
    },
    "data": {
        "buffer_size": 1000,
        "min_records": 100,
        "download_range": "1day",
        "websocket_intervals": {
            "1hour": "1m",
            "1day": "1m",
            "1month": "1h"
        }
    },
    "timezone": "Europe/Moscow"
}

# Глобальные переменные
data_buffer = []
training_buffer = []  # Новый буфер для дозагруженных данных
predictions = []
minute_model = None
hourly_model = None
minute_scaler = None
hourly_scaler = None
data_buffer_lock = Lock()
training_buffer_lock = Lock()
predictions_lock = Lock()
simulation_running = False

def calculate_indicators(df):
    """Рассчет индикаторов RSI, SMA, log_volume"""
    try:
        logger.debug("Calculating indicators")
        df = df.copy()
        df["rsi"] = RSIIndicator(df["close"], window=CONFIG["indicators"]["rsi_window"]).rsi()
        df["sma"] = SMAIndicator(df["close"], window=CONFIG["indicators"]["sma_window"]).sma_indicator()
        df["log_volume"] = np.log1p(df["volume"].replace(0, 1e-8))
        df["rsi"] = df["rsi"].fillna(df["rsi"].mean())
        df["sma"] = df["sma"].fillna(df["sma"].mean())
        df["log_volume"] = df["log_volume"].fillna(df["log_volume"].mean())
        if df.empty or df["rsi"].isnull().any() or np.any(np.isinf(df.values)):
            logger.error("Invalid indicator values detected")
            return None
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def train_model(df):
    """Обучение модели на минутных данных"""
    global minute_model, minute_scaler
    try:
        logger.debug("Training minute model")
        if len(df) < CONFIG["model"]["min_candles"]:
            logger.warning(f"Insufficient data for training: {len(df)} candles")
            return False
        features = ["close", "rsi", "sma", "volume", "log_volume"]
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        if len(X) < CONFIG["model"]["min_candles"]:
            logger.warning(f"Too few valid samples: {len(X)}")
            return False
        minute_scaler = StandardScaler()
        X_scaled = minute_scaler.fit_transform(X)
        minute_model = RandomForestRegressor(
            n_estimators=CONFIG["model"]["n_estimators_min"],
            max_depth=CONFIG["model"]["max_depth_min"],
            min_samples_split=CONFIG["model"]["min_samples_split"],
            min_samples_leaf=CONFIG["model"]["min_samples_leaf"],
            max_features=CONFIG["model"]["max_features"],
            random_state=42
        )
        minute_model.fit(X_scaled, y)
        y_pred = minute_model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        logger.info(f"Minute model trained, MSE={mse:.2f}, MAE={mae:.2f}, samples={len(X)}")
        return True
    except Exception as e:
        logger.error(f"Error training minute model: {e}")
        return False

def train_hourly_model(df):
    """Обучение модели на часовых данных"""
    global hourly_model, hourly_scaler
    try:
        logger.debug("Training hourly model")
        if len(df) < CONFIG["model"]["min_hourly_candles"]:
            logger.warning(f"Insufficient data for hourly training: {len(df)} candles")
            return False
        features = ["close", "rsi", "sma", "volume", "log_volume"]
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        if len(X) < CONFIG["model"]["min_hourly_candles"]:
            logger.warning(f"Too few valid hourly samples: {len(X)}")
            return False
        hourly_scaler = StandardScaler()
        X_scaled = hourly_scaler.fit_transform(X)
        hourly_model = RandomForestRegressor(
            n_estimators=CONFIG["model"]["n_estimators_hour"],
            max_depth=CONFIG["model"]["max_depth_hour"],
            min_samples_split=CONFIG["model"]["min_samples_split"],
            min_samples_leaf=CONFIG["model"]["min_samples_leaf"],
            max_features=CONFIG["model"]["max_features"],
            random_state=42
        )
        hourly_model.fit(X_scaled, y)
        y_pred = hourly_model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        logger.info(f"Hourly model trained, MSE={mse:.2f}, MAE={mae:.2f}, samples={len(X)}")
        return True
    except Exception as e:
        logger.error(f"Error training hourly model: {e}")
        return False

def predict(features_df):
    """Предсказание для минутной модели"""
    global minute_model, minute_scaler
    try:
        if minute_model is None or minute_scaler is None:
            logger.warning("Minute model or scaler not initialized")
            return None
        features_scaled = minute_scaler.transform(features_df)
        return minute_model.predict(features_scaled)[0]
    except Exception as e:
        logger.error(f"Minute prediction error: {e}")
        return None

def predict_hourly(features_df):
    """Предсказание для часовой модели"""
    global hourly_model, hourly_scaler
    try:
        if hourly_model is None or hourly_scaler is None:
            logger.warning("Hourly model or scaler not initialized")
            return None
        features_scaled = hourly_scaler.transform(features_df)
        return hourly_model.predict(features_scaled)[0]
    except Exception as e:
        logger.error(f"Hourly prediction error: {e}")
        return None

def fetch_historical_data(start_date, end_date, interval="1m"):
    """Загрузка исторических данных с Binance"""
    logger.debug(f"Fetching data: start_date={start_date}, end_date={end_date}, interval={interval}")
    try:
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        if start_ts >= end_ts:
            logger.error(f"Invalid date range: start_ts={start_ts}, end_ts={end_ts}")
            return None
        limit = 1000
        klines = []
        last_timestamp = None
        while start_ts < end_ts:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&startTime={start_ts}&endTime={end_ts}&limit={limit}"
            logger.debug(f"Sending request to {url}")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            new_klines = response.json()
            logger.debug(f"Received {len(new_klines)} klines")
            if not new_klines:
                logger.warning(f"No more data at start_time={start_ts}")
                break
            for kline in new_klines:
                if last_timestamp is None or kline[0] > last_timestamp:
                    klines.append(kline)
                    last_timestamp = kline[0]
            start_ts = last_timestamp + 1
            time.sleep(0.2)
        if not klines:
            logger.error("No historical data fetched")
            return None
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(CONFIG["timezone"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
        logger.debug(f"DataFrame shape before indicators: {df.shape}")
        df = calculate_indicators(df)
        if df is None or df.empty:
            logger.error("Failed to process indicators")
            return None
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)
        return None

def simulate_realtime(start_date, end_date, interval="1m", speed=60.0):
    """Симуляция реального времени"""
    global simulation_running
    logger.debug(f"Starting simulation: start_date={start_date}, end_date={end_date}, interval={interval}, speed={speed}")
    try:
        # Дозагрузка данных для обучения
        max_window = max(CONFIG["model"]["training_window_minutes_min"], CONFIG["model"]["training_window_minutes_hour"])
        training_start_date = (pd.to_datetime(start_date) - pd.Timedelta(minutes=max_window)).strftime("%Y-%m-%d")
        logger.info(f"Pre-fetching training data from {training_start_date} to {start_date}")
        training_df = fetch_historical_data(training_start_date, start_date, interval)
        if training_df is not None:
            with training_buffer_lock:
                training_buffer.clear()
                training_buffer.extend(training_df.reset_index().to_dict("records"))
                logger.info(f"Training buffer updated with {len(training_buffer)} records")

        # Загрузка данных для симуляции
        if not data_buffer:
            sim_df = fetch_historical_data(start_date, end_date, interval)
            if sim_df is None:
                logger.error("Failed to load data for simulation")
                return
            with data_buffer_lock:
                data_buffer.clear()
                data_buffer.extend(sim_df.reset_index().to_dict("records"))
                logger.info(f"Data buffer updated with {len(data_buffer)} records")

        with data_buffer_lock:
            df = pd.DataFrame(data_buffer)
        if df.empty:
            logger.error("Data buffer is empty")
            return
        df = df.sort_values("timestamp")
        interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600}
        sleep_time = interval_seconds.get(interval, 60) / float(speed)

        # Инициализация моделей на дозагруженных данных
        with training_buffer_lock:
            training_df = pd.DataFrame(training_buffer)
        if not training_df.empty:
            # Минутные данные
            df_min = training_df.set_index("timestamp").resample("1min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).interpolate(method="linear").dropna()
            df_min = calculate_indicators(df_min)
            if df_min is not None:
                train_model(df_min)
            # Часовые данные
            df_hour = training_df.set_index("timestamp").resample("1h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).interpolate(method="linear").dropna()
            df_hour = calculate_indicators(df_hour)
            if df_hour is not None:
                train_hourly_model(df_hour)

        start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        sim_id = int(time.time())
        pred_file_min = f"data/offline_results/offline_predictions_min_{start_date_str}_{sim_id}.csv"
        pred_file_hour = f"data/offline_results/offline_predictions_hour_{start_date_str}_{sim_id}.csv"
        metrics_file = "data/offline_results/offline_metrics.csv"
        os.makedirs("data/offline_results", exist_ok=True)

        logger.debug(f"Starting simulation loop with {len(df)} points")
        for i in range(len(df) - 1):
            if not simulation_running:
                logger.info("Simulation stopped")
                break
            with data_buffer_lock:
                current_data = df.iloc[i]
                features = current_data[["close", "rsi", "sma", "volume", "log_volume"]]
                features_df = pd.DataFrame([features])

            # Переобучение минутной модели
            if CONFIG["model"]["retrain_every_n_steps_min"] > 0 and i % CONFIG["model"]["retrain_every_n_steps_min"] == 0:
                logger.info(f"Retraining minute model at step {i}...")
                cutoff_time = current_data["timestamp"] - pd.Timedelta(minutes=CONFIG["model"]["training_window_minutes_min"])
                with training_buffer_lock:
                    training_df = pd.DataFrame(training_buffer)
                with data_buffer_lock:
                    sim_df = pd.DataFrame(data_buffer)
                df_window = pd.concat([training_df, sim_df])
                df_window = df_window[df_window["timestamp"] <= current_data["timestamp"]]
                df_window = df_window[df_window["timestamp"] >= cutoff_time]
                df_window_min = df_window.set_index("timestamp").resample("1min").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).interpolate(method="linear").dropna()
                df_window_min = calculate_indicators(df_window_min)
                if df_window_min is not None and not df_window_min.empty:
                    train_model(df_window_min)

            # Переобучение часовой модели
            if CONFIG["model"]["retrain_every_n_steps_hour"] > 0 and i % CONFIG["model"]["retrain_every_n_steps_hour"] == 0:
                logger.info(f"Retraining hourly model at step {i}...")
                cutoff_time = current_data["timestamp"] - pd.Timedelta(minutes=CONFIG["model"]["training_window_minutes_hour"])
                with training_buffer_lock:
                    training_df = pd.DataFrame(training_buffer)
                with data_buffer_lock:
                    sim_df = pd.DataFrame(data_buffer)
                df_window = pd.concat([training_df, sim_df])
                df_window = df_window[df_window["timestamp"] <= current_data["timestamp"]]
                df_window = df_window[df_window["timestamp"] >= cutoff_time]
                df_window_hour = df_window.set_index("timestamp").resample("1h").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).interpolate(method="linear").dropna()
                df_window_hour = calculate_indicators(df_window_hour)
                if df_window_hour is not None and not df_window_hour.empty:
                    train_hourly_model(df_window_hour)

            # Прогноз для минутной модели
            prediction_min = predict(features_df)
            if prediction_min is not None:
                new_pred_min = [{
                    "timestamp": current_data["timestamp"],
                    "actual_price": current_data["close"],
                    "predicted_price": prediction_min,
                    "error": abs(current_data["close"] - prediction_min)
                }]
                with predictions_lock:
                    predictions.append(new_pred_min[0] | {"forecast_range": "1min"})
                    pred_df_min = pd.DataFrame(new_pred_min)
                    header = not os.path.exists(pred_file_min)
                    pred_df_min.to_csv(pred_file_min, index=False, encoding="utf-8", mode="a", header=header)

            # Прогноз для часовой модели (на начало часа)
            if current_data["timestamp"].minute == 0:
                prediction_hour = predict_hourly(features_df)
                if prediction_hour is not None:
                    new_pred_hour = [{
                        "timestamp": current_data["timestamp"],
                        "actual_price": current_data["close"],
                        "predicted_price": prediction_hour,
                        "error": abs(current_data["close"] - prediction_hour)
                    }]
                    with predictions_lock:
                        predictions.append(new_pred_hour[0] | {"forecast_range": "1hour"})
                        pred_df_hour = pd.DataFrame(new_pred_hour)
                        header = not os.path.exists(pred_file_hour)
                        pred_df_hour.to_csv(pred_file_hour, index=False, encoding="utf-8", mode="a", header=header)

            time.sleep(sleep_time)
        logger.info("Simulation completed")
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)

# Dash приложение
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Оффлайн-симуляция BTC/USDT"),
    html.Div([
        html.Label("Диапазон дат:"),
        dcc.DatePickerRange(
            id="date-range",
            start_date="2025-01-01",
            end_date="2025-01-02",
            display_format="YYYY-MM-DD",
            style={"margin": "10px"}
        ),
    ]),
    html.Div([
        html.Label("Интервал данных:"),
        dcc.Dropdown(
            id="data-interval",
            options=[
                {"label": "1 секунда", "value": "1s"},
                {"label": "1 минута", "value": "1m"},
                {"label": "1 час", "value": "1h"}
            ],
            value="1m",
            style={"width": "200px", "margin": "10px"}
        ),
    ]),
    html.Div([
        html.Label("Скорость симуляции (в секундах):"),
        dcc.Input(
            id="simulation-speed",
            type="number",
            value=600.0,
            step=0.1,
            style={"width": "100px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Скорость симуляции в секундах. Указывает, во сколько раз ускорить или замедлить воспроизведение (например, 1 = реальное время, 60 = ускорение в 60 раз, то есть 1 сек реального времени = 1 минута симуляции)."),
    ]),
    html.H3("Настройки минутной модели", style={"margin-top": "20px"}),
    html.Div([
        html.Label("Количество деревьев:"),
        dcc.Input(
            id="n-estimators-min",
            type="number",
            value=CONFIG["model"]["n_estimators_min"],
            step=10,
            style={"width": "80px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Число деревьев в случайном лесу. Большее значение увеличивает точность, но замедляет обучение (рекомендуется 50-200)."),
    ], style={"display": "flex", "align-items": "center"}),
    html.Div([
        html.Label("Максимальная глубина:"),
        dcc.Input(
            id="max-depth-min",
            type="number",
            value=CONFIG["model"]["max_depth_min"] or 0,
            step=1,
            style={"width": "80px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Максимальная глубина дерева. 0 = без ограничений."),
    ], style={"display": "flex", "align-items": "center"}),
    html.Div([
        html.Label("Окно обучения (минуты):"),
        dcc.Input(
            id="training-window-minutes-min",
            type="number",
            value=CONFIG["model"]["training_window_minutes_min"],
            step=60,
            style={"width": "100px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Количество минут данных для обучения минутной модели (например, 60 = 1 час)."),
    ], style={"display": "flex", "align-items": "center"}),
    html.Div([
        html.Label("Переобучать каждые N шагов:"),
        dcc.Input(
            id="retrain-every-n-steps-min",
            type="number",
            value=CONFIG["model"]["retrain_every_n_steps_min"],
            step=1,
            style={"width": "100px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Шаг симуляции соответствует одной свече с выбранным интервалом данных (например, 1 минута при интервале 1m). Указывает, через сколько свечей переобучать минутную модель."),
    ], style={"display": "flex", "align-items": "center"}),

    html.H3("Настройки часовой модели", style={"margin-top": "20px"}),
    html.Div([
        html.Label("Количество деревьев:"),
        dcc.Input(
            id="n-estimators-hour",
            type="number",
            value=CONFIG["model"]["n_estimators_hour"],
            step=10,
            style={"width": "80px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Число деревьев в случайном лесу для часовой модели."),
    ], style={"display": "flex", "align-items": "center"}),
    html.Div([
        html.Label("Максимальная глубина:"),
        dcc.Input(
            id="max-depth-hour",
            type="number",
            value=CONFIG["model"]["max_depth_hour"] or 0,
            step=1,
            style={"width": "80px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Максимальная глубина дерева для часовой модели. 0 = без ограничений."),
    ], style={"display": "flex", "align-items": "center"}),
    html.Div([
        html.Label("Окно обучения (минуты):"),
        dcc.Input(
            id="training-window-minutes-hour",
            type="number",
            value=CONFIG["model"]["training_window_minutes_hour"],
            step=60,
            style={"width": "100px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Количество минут данных для обучения часовой модели (например, 1440 = 1 день)."),
    ], style={"display": "flex", "align-items": "center"}),
    html.Div([
        html.Label("Переобучать каждые N шагов:"),
        dcc.Input(
            id="retrain-every-n-steps-hour",
            type="number",
            value=CONFIG["model"]["retrain_every_n_steps_hour"],
            step=1,
            style={"width": "100px", "margin": "10px"}
        ),
        html.Span("?", className="tooltip", title="Шаг симуляции соответствует одной свече с выбранным интервалом данных (например, 1 минута при интервале 1m). Указывает, через сколько свечей переобучать часовую модель."),
    ], style={"display": "flex", "align-items": "center"}),

    html.Div([
        dcc.Checklist(
            id="show-candles",
            options=[{"label": "Показать свечи", "value": "candles"}],
            value=["candles"],
            style={"margin": "10px"}
        ),
    ]),

    html.Button("Запустить симуляцию", id="start-simulation", n_clicks=0),
    html.Button("Остановить симуляцию", id="stop-simulation", n_clicks=0),

    dcc.Graph(id="main-graph", config={"displayModeBar": True, "scrollZoom": True}),
    dcc.Graph(id="predictions-min-graph", config={"displayModeBar": True, "scrollZoom": True}),
    dcc.Graph(id="predictions-hour-graph", config={"displayModeBar": True, "scrollZoom": True}),

    dcc.Interval(id="interval-component", interval=CONFIG["visual"]["update_interval"], n_intervals=0),

    dcc.Store(id="main-graph-state", data={}),
    dcc.Store(id="predictions-min-graph-state", data={}),
    dcc.Store(id="predictions-hour-graph-state", data={})
])

# Колбэки сохранения состояния (фиксированные)
@app.callback(
    Output("main-graph-state", "data"),
    Input("main-graph", "relayoutData"),
    State("main-graph-state", "data"),
    prevent_initial_call=True
)
def save_main_graph_state(relayoutData, existing_data):
    logger.debug(f"Saving main graph state: relayoutData={relayoutData}")
    if relayoutData is None:
        return existing_data
    if "autosize" in relayoutData and relayoutData["autosize"]:
        logger.debug("Reset detected: clearing stored state")
        return {}
    new_data = existing_data.copy() if existing_data else {}
    new_data.update(relayoutData)
    logger.debug(f"Updated main graph state: {new_data}")
    return new_data

@app.callback(
    Output("predictions-min-graph-state", "data"),
    Input("predictions-min-graph", "relayoutData"),
    State("predictions-min-graph-state", "data"),
    prevent_initial_call=True
)
def save_predictions_min_graph_state(relayoutData, existing_data):
    logger.debug(f"Saving predictions-min graph state: relayoutData={relayoutData}")
    if relayoutData is None:
        return existing_data
    if "autosize" in relayoutData and relayoutData["autosize"]:
        logger.debug("Reset detected: clearing stored state")
        return {}
    new_data = existing_data.copy() if existing_data else {}
    new_data.update(relayoutData)
    logger.debug(f"Updated predictions-min graph state: {new_data}")
    return new_data

@app.callback(
    Output("predictions-hour-graph-state", "data"),
    Input("predictions-hour-graph", "relayoutData"),
    State("predictions-hour-graph-state", "data"),
    prevent_initial_call=True
)
def save_predictions_hour_graph_state(relayoutData, existing_data):
    logger.debug(f"Saving predictions-hour graph state: relayoutData={relayoutData}")
    if relayoutData is None:
        return existing_data
    if "autosize" in relayoutData and relayoutData["autosize"]:
        logger.debug("Reset detected: clearing stored state")
        return {}
    new_data = existing_data.copy() if existing_data else {}
    new_data.update(relayoutData)
    logger.debug(f"Updated predictions-hour graph state: {new_data}")
    return new_data

@app.callback(
    Output("main-graph", "figure"),
    Output("predictions-min-graph", "figure"),
    Output("predictions-hour-graph", "figure"),
    Input("interval-component", "n_intervals"),
    Input("show-candles", "value"),
    State("main-graph-state", "data"),
    State("predictions-min-graph-state", "data"),
    State("predictions-hour-graph-state", "data"),
    prevent_initial_call=False
)
def update_graph(n, show_candles, main_graph_state, predictions_min_graph_state, predictions_hour_graph_state):
    logger.debug(f"Updating graph: n_intervals={n}, show_candles={show_candles}, "
                 f"main_graph_state={main_graph_state}, "
                 f"predictions_min_graph_state={predictions_min_graph_state}, "
                 f"predictions_hour_graph_state={predictions_hour_graph_state}")

    # Основной график
    main_dragmode = main_graph_state.get("dragmode", "zoom") if main_graph_state else "zoom"
    fig = go.Figure()
    fig.update_layout(
        title="BTC/USDT: Цены",
        xaxis_title="Время (MSK)",
        yaxis_title="Цена (USDT)",
        height=700,
        template="plotly_dark",
        dragmode=main_dragmode,
        xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45),
        showlegend=True
    )

    # Минутный график
    min_dragmode = predictions_min_graph_state.get("dragmode", "zoom") if predictions_min_graph_state else "zoom"
    pred_fig_min = go.Figure()
    pred_fig_min.update_layout(
        title="BTC/USDT: Предсказания (1 минута)",
        xaxis_title="Время (MSK)",
        yaxis_title="Цена (USDT)",
        height=400,
        template="plotly_dark",
        dragmode=min_dragmode,
        xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45),
        showlegend=True
    )

    # Часовой график
    hour_dragmode = predictions_hour_graph_state.get("dragmode", "zoom") if predictions_hour_graph_state else "zoom"
    pred_fig_hour = go.Figure()
    pred_fig_hour.update_layout(
        title="BTC/USDT: Предсказания (1 час)",
        xaxis_title="Время (MSK)",
        yaxis_title="Цена (USDT)",
        height=400,
        template="plotly_dark",
        dragmode=hour_dragmode,
        xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45),
        showlegend=True
    )

    # Чтение данных
    with data_buffer_lock:
        if not data_buffer:
            logger.debug("Data buffer empty")
            return fig, pred_fig_min, pred_fig_hour
        df = pd.DataFrame(data_buffer)

    if df.empty:
        logger.debug("DataFrame empty")
        return fig, pred_fig_min, pred_fig_hour

    # Отображение свечей или линии
    show_candles = show_candles and "candles" in show_candles
    if show_candles:
        fig.add_trace(go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Свечи",
            increasing_line_color="green",
            decreasing_line_color="red"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["close"],
            mode="lines",
            name="Цена закрытия",
            line=dict(color=CONFIG["visual"]["real_price_color"])
        ))

    # Прогнозы
    with predictions_lock:
        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df = pred_df.sort_values("timestamp")
            pred_df["time_diff"] = pred_df["timestamp"].diff().dt.total_seconds()
            pred_df = pred_df[pred_df["time_diff"].fillna(60) <= 3600]

            # Минутные прогнозы
            pred_df_min = pred_df[pred_df["forecast_range"] == "1min"]
            if not pred_df_min.empty:
                pred_fig_min.add_trace(go.Scatter(
                    x=pred_df_min["timestamp"],
                    y=pred_df_min["actual_price"],
                    mode="lines",
                    name="Фактическая цена",
                    line=dict(color=CONFIG["visual"]["real_price_color"])
                ))
                pred_fig_min.add_trace(go.Scatter(
                    x=pred_df_min["timestamp"],
                    y=pred_df_min["predicted_price"],
                    mode="lines",
                    name="Предсказанная цена",
                    line=dict(color=CONFIG["visual"]["predicted_price_color"])
                ))
                mse_min = mean_squared_error(pred_df_min["actual_price"], pred_df_min["predicted_price"])
                mae_min = mean_absolute_error(pred_df_min["actual_price"], pred_df_min["predicted_price"])
                pred_fig_min.add_annotation(
                    xref="paper", yref="paper", x=0.05, y=0.95,
                    text=f"MSE: {mse_min:.2f}, MAE: {mae_min:.2f}, Количество: {len(pred_df_min)}",
                    showarrow=False, font=dict(size=12, color="white")
                )

            # Часовые прогнозы
            pred_df_hour = pred_df[pred_df["forecast_range"] == "1hour"]
            if not pred_df_hour.empty:
                pred_fig_hour.add_trace(go.Scatter(
                    x=pred_df_hour["timestamp"],
                    y=pred_df_hour["actual_price"],
                    mode="lines+markers",
                    name="Фактическая цена",
                    line=dict(color=CONFIG["visual"]["real_price_color"])
                ))
                pred_fig_hour.add_trace(go.Scatter(
                    x=pred_df_hour["timestamp"],
                    y=pred_df_hour["predicted_price"],
                    mode="lines+markers",
                    name="Предсказанная цена",
                    line=dict(color=CONFIG["visual"]["predicted_price_color"])
                ))
                mse_hour = mean_squared_error(pred_df_hour["actual_price"], pred_df_hour["predicted_price"])
                mae_hour = mean_absolute_error(pred_df_hour["actual_price"], pred_df_hour["predicted_price"])
                pred_fig_hour.add_annotation(
                    xref="paper", yref="paper", x=0.05, y=0.95,
                    text=f"MSE: {mse_hour:.2f}, MAE: {mae_hour:.2f}, Количество: {len(pred_df_hour)}",
                    showarrow=False, font=dict(size=12, color="white")
                )

    # Применение сохранённых zoom и панорамирования
    try:
        if main_graph_state:
            x0 = main_graph_state.get("xaxis.range[0]")
            x1 = main_graph_state.get("xaxis.range[1]")
            y0 = main_graph_state.get("yaxis.range[0]")
            y1 = main_graph_state.get("yaxis.range[1]")

            if x0 and x1:
                fig.update_xaxes(range=[pd.to_datetime(x0), pd.to_datetime(x1)])
            if y0 is not None and y1 is not None:
                fig.update_yaxes(range=[y0, y1])
            if "dragmode" in main_graph_state:
                fig.update_layout(dragmode=main_graph_state["dragmode"])
    except Exception as e:
        logger.error(f"Error applying main graph zoom: {e}")

    try:
        if predictions_min_graph_state:
            x0 = predictions_min_graph_state.get("xaxis.range[0]")
            x1 = predictions_min_graph_state.get("xaxis.range[1]")
            y0 = predictions_min_graph_state.get("yaxis.range[0]")
            y1 = predictions_min_graph_state.get("yaxis.range[1]")

            if x0 and x1:
                pred_fig_min.update_xaxes(range=[pd.to_datetime(x0), pd.to_datetime(x1)])
            if y0 is not None and y1 is not None:
                pred_fig_min.update_yaxes(range=[y0, y1])
            if "dragmode" in predictions_min_graph_state:
                pred_fig_min.update_layout(dragmode=predictions_min_graph_state["dragmode"])
    except Exception as e:
        logger.error(f"Error applying min graph zoom: {e}")

    try:
        if predictions_hour_graph_state:
            x0 = predictions_hour_graph_state.get("xaxis.range[0]")
            x1 = predictions_hour_graph_state.get("xaxis.range[1]")
            y0 = predictions_hour_graph_state.get("yaxis.range[0]")
            y1 = predictions_hour_graph_state.get("yaxis.range[1]")

            if x0 and x1:
                pred_fig_hour.update_xaxes(range=[pd.to_datetime(x0), pd.to_datetime(x1)])
            if y0 is not None and y1 is not None:
                pred_fig_hour.update_yaxes(range=[y0, y1])
            if "dragmode" in predictions_hour_graph_state:
                pred_fig_hour.update_layout(dragmode=predictions_hour_graph_state["dragmode"])
    except Exception as e:
        logger.error(f"Error applying hour graph zoom: {e}")

    return fig, pred_fig_min, pred_fig_hour

# Колбэк запуска симуляции
@app.callback(
    Output("start-simulation", "n_clicks"),
    Input("start-simulation", "n_clicks"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("data-interval", "value"),
    State("simulation-speed", "value"),
    State("n-estimators-min", "value"),
    State("max-depth-min", "value"),
    State("n-estimators-hour", "value"),
    State("max-depth-hour", "value"),
    State("training-window-minutes-min", "value"),
    State("retrain-every-n-steps-min", "value"),
    State("training-window-minutes-hour", "value"),
    State("retrain-every-n-steps-hour", "value"),
    prevent_initial_call=True
)
def start_simulation(n_clicks, start_date, end_date, data_interval, simulation_speed,
                     n_estimators_min, max_depth_min, n_estimators_hour, max_depth_hour,
                     training_window_minutes_min, retrain_every_n_steps_min,
                     training_window_minutes_hour, retrain_every_n_steps_hour):
    logger.debug(f"Starting simulation: n_clicks={n_clicks}, start_date={start_date}, end_date={end_date}, "
                 f"interval={data_interval}, speed={simulation_speed}")
    if n_clicks:
        try:
            global simulation_running, CONFIG
            simulation_running = True
            with data_buffer_lock:
                data_buffer.clear()
            with training_buffer_lock:
                training_buffer.clear()
            with predictions_lock:
                predictions.clear()

            CONFIG["model"]["n_estimators_min"] = int(n_estimators_min) if n_estimators_min else CONFIG["model"]["n_estimators"]
            CONFIG["model"]["max_depth_min"] = int(max_depth_min) if max_depth_min else None
            CONFIG["model"]["n_estimators_hour"] = int(n_estimators_hour) if n_estimators_hour else CONFIG["model"]["n_estimators"]
            CONFIG["model"]["max_depth_hour"] = int(max_depth_hour) if max_depth_hour else None
            CONFIG["model"]["training_window_minutes_min"] = int(training_window_minutes_min) if training_window_minutes_min else CONFIG["model"]["training_window_minutes_min"]
            CONFIG["model"]["retrain_every_n_steps_min"] = int(retrain_every_n_steps_min) if retrain_every_n_steps_min else CONFIG["model"]["retrain_every_n_steps_min"]
            CONFIG["model"]["training_window_minutes_hour"] = int(training_window_minutes_hour) if training_window_minutes_hour else CONFIG["model"]["training_window_minutes_hour"]
            CONFIG["model"]["retrain_every_n_steps_hour"] = int(retrain_every_n_steps_hour) if retrain_every_n_steps_hour else CONFIG["model"]["retrain_every_n_steps_hour"]

            data_interval = data_interval or "1m"
            simulation_speed = float(simulation_speed or 60.0)

            Thread(target=simulate_realtime, args=(start_date, end_date, data_interval, simulation_speed), daemon=True).start()
            logger.info(f"Simulation started: {start_date} to {end_date}, interval={data_interval}, speed={simulation_speed}")
        except Exception as e:
            logger.error(f"Error starting simulation: {e}", exc_info=True)
    return n_clicks

# Колбэк остановки симуляции
@app.callback(
    Output("stop-simulation", "n_clicks"),
    Output("main-graph-state", "data", allow_duplicate=True),
    Output("predictions-min-graph-state", "data", allow_duplicate=True),
    Output("predictions-hour-graph-state", "data", allow_duplicate=True),
    Input("stop-simulation", "n_clicks"),
    prevent_initial_call=True
)
def stop_simulation(n_clicks):
    logger.debug(f"Stopping simulation: n_clicks={n_clicks}")
    if n_clicks:
        try:
            global simulation_running
            simulation_running = False
            logger.info("Simulation stop requested")
            return n_clicks, {}, {}, {}
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}", exc_info=True)
    return n_clicks, {}, {}, {}

# Запуск приложения
if __name__ == "__main__":
    logger.info("Starting offline Dash application")
    app.run(host="0.0.0.0", port=env_config[env_name]["port_offline"], debug=True)
