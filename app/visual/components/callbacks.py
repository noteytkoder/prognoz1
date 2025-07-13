import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, State
from .utils import (
    logger, config, last_stats_time, last_mse, last_mae, current_layout,
    data_buffer, get_latest_features, buffer_lock,
    predict, predict_hourly, load_config, save_config
)
import numpy as np
import time
import os
import sys
import subprocess
import platform
from pathlib import Path
from app.logs.logger import setup_logger
from app.data.handler import buffer_lock
import pytz

RESTART_FLAG = Path("restart.flag")
logger = setup_logger()

# Cache for resampled dataframe and predictions
cached_df = None
cached_timestamp = None
cached_pred_df = None
cached_pred_timestamp = None

def prepare_data(data_copy, interval, msk_tz):
    """Подготовка данных: ресэмплинг и фильтрация"""
    global cached_df, cached_timestamp
    latest_timestamp = data_copy[-1]["timestamp"] if data_copy else None

    if cached_df is not None and cached_timestamp == latest_timestamp:
        return cached_df, latest_timestamp

    df = pd.DataFrame(data_copy)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.empty:
        return None, latest_timestamp

    interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600, "1d": 86400}
    df = df.resample(f"{interval_seconds.get(interval, 1)}s").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).interpolate(method="linear")
    cached_df = df
    cached_timestamp = latest_timestamp
    return df, latest_timestamp

def prepare_predictions(msk_tz, last_time, time_delta):
    """Подготовка данных предсказаний"""
    global cached_pred_df, cached_pred_timestamp
    mse_min, mae_min, mse_hour, mae_hour, pred_count = None, None, None, None, 0
    pred_df = pd.DataFrame()

    try:
        if os.path.exists("logs/predictions.csv") and os.path.getsize("logs/predictions.csv") > 0:
            with buffer_lock:
                pred_df = pd.read_csv("logs/predictions.csv", encoding='utf-8')
            pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True).dt.tz_convert(msk_tz)
            pred_df["min_pred_time"] = pd.to_datetime(pred_df["min_pred_time"], utc=True).dt.tz_convert(msk_tz)
            pred_df["hour_pred_time"] = pd.to_datetime(pred_df["hour_pred_time"], utc=True).dt.tz_convert(msk_tz)
            pred_df = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
            if len(pred_df) > 1:
                min_valid = pred_df[pred_df["min_error"].notna()]
                hour_valid = pred_df[pred_df["hour_error"].notna()]
                if not min_valid.empty:
                    mse_min = np.mean(min_valid["min_error"] ** 2)
                    mae_min = np.mean(min_valid["min_error"])
                if not hour_valid.empty:
                    mse_hour = np.mean(hour_valid["hour_error"] ** 2)
                    mae_hour = np.mean(hour_valid["hour_error"])
                pred_count = len(pred_df)
            cached_pred_df = pred_df
            cached_pred_timestamp = last_time
    except Exception as e:
        logger.error(f"Failed to read logs/predictions.csv: {e}")
    
    return pred_df, mse_min, mae_min, mse_hour, mae_hour, pred_count

def create_main_figure(df, show_candles, show_error_band, forecast_range, last_time, features, error_band_width):
    """Создание основного графика"""
    fig = go.Figure()
    if show_candles:
        fig.add_trace(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="Свечи", increasing_line_color="green", decreasing_line_color="red"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["close"], mode="lines", name="Цена закрытия",
            line=dict(color="blue")
        ))

    if features is not None:
        df_min_row, _, _ = features
        features_df = pd.DataFrame([df_min_row[["close", "rsi", "sma", "volume", "log_volume"]]])
        prediction = predict(features_df) if forecast_range == "1min" else predict_hourly(features_df)
        pred_time = last_time + (pd.Timedelta(minutes=1) if forecast_range == "1min" else pd.Timedelta(hours=1))
        if prediction is not None:
            fig.add_trace(go.Scatter(
                x=[last_time, pred_time], y=[df["close"].iloc[-1], prediction],
                mode="lines", name=f"Прогноз ({'1 минута' if forecast_range == '1min' else '1 час'})",
                line=dict(color=config["visual"]["predicted_price_color"])
            ))
            if show_error_band:
                fig.add_trace(go.Scatter(
                    x=[last_time, pred_time, pred_time, last_time],
                    y=[df["close"].iloc[-1], prediction + error_band_width, prediction - error_band_width, df["close"].iloc[-1]],
                    fill="toself", fillcolor=config["visual"]["error_band_color"],
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"Зона погрешности (±{error_band_width:.2f} USDT)"
                ))

    return fig

def create_prediction_figures(pred_df, mse_min, mae_min, mse_hour, mae_hour, pred_count, last_time, time_delta, x_range_pred_min, x_range_pred_hour, y_range):
    """Создание графиков предсказаний"""
    pred_fig_min = go.Figure()
    pred_fig_hour = go.Figure()
    pred_style_min = {"display": "block" if not pred_df.empty else "none"}
    pred_style_hour = {"display": "block" if not pred_df.empty else "none"}

    if not pred_df.empty:
        filtered_pred_df = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
        if not filtered_pred_df.empty:
            pred_fig_min.add_trace(go.Scatter(
                x=filtered_pred_df["timestamp"], y=filtered_pred_df["actual_price"],
                mode="lines", name="Фактическая цена", line=dict(color=config["visual"]["real_price_color"])
            ))
            pred_fig_min.add_trace(go.Scatter(
                x=filtered_pred_df["min_pred_time"], y=filtered_pred_df["min_pred"],
                mode="lines", name="Предсказанная цена (1 мин)", line=dict(color=config["visual"]["predicted_price_color"])
            ))
            annotation_text = (f"MSE: {mse_min:.2f}, MAE: {mae_min:.2f}, Количество: {pred_count}"
                              if mse_min is not None and mae_min is not None
                              else "Ожидание данных для расчета метрик")
            pred_fig_min.add_annotation(
                xref="paper", yref="paper", x=0.05, y=0.95,
                text=annotation_text,
                showarrow=False, font=dict(size=12, color="white")
            )
            pred_fig_min.update_layout(
                title="BTC/USDT: Фактические и предсказанные цены (1 минута)",
                xaxis_title="Время (MSK)",
                yaxis_title="Цена (USDT)",
                xaxis_range=x_range_pred_min,
                yaxis_range=y_range,
                showlegend=True,
                height=400,
                template="plotly_dark",
                dragmode="zoom",
                uirevision="predictions-graph-min",
                xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45)
            )

            pred_fig_hour.add_trace(go.Scatter(
                x=filtered_pred_df["timestamp"], y=filtered_pred_df["actual_price"],
                mode="lines", name="Фактическая цена", line=dict(color=config["visual"]["real_price_color"])
            ))
            pred_fig_hour.add_trace(go.Scatter(
                x=filtered_pred_df["hour_pred_time"], y=filtered_pred_df["hour_pred"],
                mode="lines", name="Предсказанная цена (1 час)", line=dict(color=config["visual"]["predicted_price_color"])
            ))
            annotation_text = (f"MSE: {mse_hour:.2f}, MAE: {mae_hour:.2f}, Количество: {pred_count}"
                              if mse_hour is not None and mae_hour is not None
                              else "Ожидание данных для расчета метрик")
            pred_fig_hour.add_annotation(
                xref="paper", yref="paper", x=0.05, y=0.95,
                text=annotation_text,
                showarrow=False, font=dict(size=12, color="white")
            )
            pred_fig_hour.update_layout(
                title="BTC/USDT: Фактические и предсказанные цены (1 час)",
                xaxis_title="Время (MSK)",
                yaxis_title="Цена (USDT)",
                xaxis_range=x_range_pred_hour,
                yaxis_range=y_range,
                showlegend=True,
                height=400,
                template="plotly_dark",
                dragmode="zoom",
                uirevision="predictions-graph-hour",
                xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45)
            )

    return pred_fig_min, pred_fig_hour, pred_style_min, pred_style_hour

@callback(
    Output("interval-component", "interval"),
    Output("interval-update", "data"),
    Input("download-range", "value"),
    Input("websocket-interval-1hour", "value"),
    Input("websocket-interval-1day", "value"),
    Input("websocket-interval-1month", "value"),
)
def update_interval(download_range, ws_interval_1hour, ws_interval_1day, ws_interval_1month):
    interval_map = {"1s": 1000, "1m": 60000, "15m": 15*60000, "1h": 60*60000}
    new_config = load_config()
    
    if download_range == "1hour" and ws_interval_1hour:
        interval = interval_map.get(ws_interval_1hour, 1000)
        new_config["data"]["websocket_intervals"]["1hour"] = ws_interval_1hour
    elif download_range == "1day" and ws_interval_1day:
        interval = interval_map.get(ws_interval_1day, 60000)
        new_config["data"]["websocket_intervals"]["1day"] = ws_interval_1day
    elif download_range == "1month" and ws_interval_1month:
        interval = interval_map.get(ws_interval_1month, 60000)
        new_config["data"]["websocket_intervals"]["1month"] = ws_interval_1month
    else:
        interval = config["visual"]["update_interval"]
    
    save_config(new_config)
    logger.info(f"Graph update interval set to {interval} ms for {download_range}")
    return interval, interval

@callback(
    Output("main-graph", "figure"),
    Output("predictions-graph-min", "figure"),
    Output("predictions-graph-min", "style"),
    Output("predictions-graph-hour", "figure"),
    Output("predictions-graph-hour", "style"),
    Output("graph-layout", "data"),
    Output("pred-min-layout", "data"),
    Output("pred-hour-layout", "data"),
    Input("interval-component", "n_intervals"),
    Input("train-window-minutes", "value"),  # Заменили train-period на train-window-minutes
    Input("show-candles", "value"),
    Input("show-error-band", "value"),
    Input("forecast-range", "value"),
    Input("autoscale-range", "value"),
    Input("main-graph", "relayoutData"),
    Input("predictions-graph-min", "relayoutData"),
    Input("predictions-graph-hour", "relayoutData"),
    State("graph-layout", "data"),
    State("pred-min-layout", "data"),
    State("pred-hour-layout", "data"),
    prevent_initial_call=True
)
def update_graph(n, train_period, show_candles, show_error_band, forecast_range, autoscale_range, 
                 main_relayout_data, pred_min_relayout_data, pred_hour_relayout_data,
                 main_stored_layout, pred_min_stored_layout, pred_hour_stored_layout):
    global last_stats_time, last_mse, last_mae, current_layout
    logger.debug(f"Starting update_graph, n_intervals={n}, forecast_range={forecast_range}, autoscale_range={autoscale_range}")
    msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))

    try:
        # Обновление периода обучения
        if train_period and train_period != config["model"]["train_window_minutes"]:
            new_config = load_config()
            new_config["model"]["train_window_minutes"] = int(train_period)
            save_config(new_config)
            logger.info(f"Train period updated to {train_period} minutes")

        # Обновление макетов графиков
        if main_relayout_data and "xaxis.range[0]" in main_relayout_data:
            main_stored_layout = main_relayout_data
        if pred_min_relayout_data and "xaxis.range[0]" in pred_min_relayout_data:
            pred_min_stored_layout = pred_min_relayout_data
        if pred_hour_relayout_data and "xaxis.range[0]" in pred_hour_relayout_data:
            pred_hour_stored_layout = pred_hour_relayout_data
        current_layout = main_stored_layout

        # Получение данных из буфера
        with buffer_lock:
            if not data_buffer:
                logger.warning("Data buffer is empty")
                return (go.Figure(), go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, 
                        main_stored_layout, pred_min_stored_layout, pred_hour_stored_layout)
            data_copy = list(data_buffer)

        # Подготовка данных
        interval = config["data"]["websocket_intervals"].get(config["data"]["download_range"], "1s")
        df, latest_timestamp = prepare_data(data_copy, interval, msk_tz)
        if df is None:
            return (go.Figure(), go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, 
                    main_stored_layout, pred_min_stored_layout, pred_hour_stored_layout)

        # Определение диапазона времени
        last_time = df.index.max()
        ranges = {"1hour": pd.Timedelta(hours=1), "1day": pd.Timedelta(days=1), "1month": pd.Timedelta(days=30)}
        default_time_delta = ranges.get(config["data"]["download_range"], pd.Timedelta(days=1))

        if autoscale_range == "10min":
            time_delta = pd.Timedelta(minutes=10)
            default_x_range = [last_time - time_delta, last_time + pd.Timedelta(minutes=1)]
            default_x_range_pred = [last_time - time_delta, last_time + pd.Timedelta(hours=1)]
        elif autoscale_range == "1hour":
            time_delta = pd.Timedelta(hours=1)
            default_x_range = [last_time - time_delta, last_time + pd.Timedelta(minutes=1)]
            default_x_range_pred = [last_time - time_delta, last_time + pd.Timedelta(hours=1)]
        else:
            time_delta = default_time_delta
            default_x_range = [last_time - time_delta, last_time]
            default_x_range_pred = [last_time - time_delta, last_time + pd.Timedelta(hours=1)]

        x_range = ([pd.to_datetime(main_stored_layout["xaxis.range[0]"], utc=True).tz_convert(msk_tz), 
                    pd.to_datetime(main_stored_layout["xaxis.range[1]"], utc=True).tz_convert(msk_tz)] 
                   if main_stored_layout and "xaxis.range[0]" in main_stored_layout 
                   else default_x_range)
        x_range_pred_min = ([pd.to_datetime(pred_min_stored_layout["xaxis.range[0]"], utc=True).tz_convert(msk_tz), 
                            pd.to_datetime(pred_min_stored_layout["xaxis.range[1]"], utc=True).tz_convert(msk_tz)] 
                           if pred_min_stored_layout and "xaxis.range[0]" in pred_min_stored_layout 
                           else default_x_range_pred)
        x_range_pred_hour = ([pd.to_datetime(pred_hour_stored_layout["xaxis.range[0]"], utc=True).tz_convert(msk_tz), 
                             pd.to_datetime(pred_hour_stored_layout["xaxis.range[1]"], utc=True).tz_convert(msk_tz)] 
                            if pred_hour_stored_layout and "xaxis.range[0]" in pred_hour_stored_layout 
                            else default_x_range_pred)

        # Фильтрация данных по видимому диапазону
        df = df.loc[x_range[0]:x_range[1]].reset_index()

        # Проверка устаревания данных
        current_time = pd.Timestamp.now(tz=msk_tz)
        interval_seconds = {"1s": 1, "1m": 60, "3m": 180, "15m": 900, "1h": 3600, "1d": 86400}
        if (current_time - last_time).total_seconds() > interval_seconds.get(interval, 1) * 60:
            fig = go.Figure()
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.5,
                text="Данные устарели. Пожалуйста, проверьте соединение.",
                showarrow=False, font=dict(size=16, color="red")
            )
            return (fig, go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, 
                    main_stored_layout, pred_min_stored_layout, pred_hour_stored_layout)

        if df.empty:
            return (go.Figure(), go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, 
                    main_stored_layout, pred_min_stored_layout, pred_hour_stored_layout)

        # Подготовка предсказаний
        show_candles = "candles" in (show_candles or [])
        show_band = "show" in (show_error_band or [])
        pred_df, mse_min, mae_min, mse_hour, mae_hour, pred_count = prepare_predictions(msk_tz, last_time, time_delta)

        # Вычисление зоны погрешности
        error_band_width = config["visual"]["error_band_min"]
        if forecast_range == "1min" and mae_min is not None:
            error_band_width = max(mae_min * config["visual"]["error_band_multiplier"], config["visual"]["error_band_min"])
        elif forecast_range == "1hour" and mae_hour is not None:
            error_band_width = max(mae_hour * config["visual"]["error_band_multiplier"], config["visual"]["error_band_min"])

        # Создание основного графика
        min_price = df["close"].min()
        max_price = df["close"].max()
        y_range = [min_price * 0.995, max_price * 1.005]
        features = get_latest_features()
        fig = create_main_figure(df, show_candles, show_band, forecast_range, last_time, features, error_band_width)
        
        # Обновление макета основного графика
        fig.update_layout(
            title="BTC/USDT: Цены и прогноз",
            xaxis_title="Время (MSK)",
            yaxis_title="Цена (USDT)",
            xaxis_range=x_range,
            yaxis_range=y_range,
            showlegend=True,
            height=700,
            template="plotly_dark",
            dragmode="zoom",
            uirevision="main-graph",
            xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45),
            margin=dict(b=100)
        )

        # Создание графиков предсказаний
        pred_fig_min, pred_fig_hour, pred_style_min, pred_style_hour = create_prediction_figures(
            pred_df, mse_min, mae_min, mse_hour, mae_hour, pred_count, last_time, time_delta, x_range_pred_min, x_range_pred_hour, y_range
        )

        logger.debug("Graph updated successfully")
        return (fig, pred_fig_min, pred_style_min, pred_fig_hour, pred_style_hour, 
                main_stored_layout, pred_min_stored_layout, pred_hour_stored_layout)

    except Exception as e:
        logger.error(f"Error in update_graph: {e}", exc_info=True)
        return (go.Figure(), go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, 
                main_stored_layout, pred_min_stored_layout, pred_hour_stored_layout)

@callback(
    Output("download-btn", "n_clicks"),
    Input("download-btn", "n_clicks")
)
def download_data(n_clicks):
    """Скачивание данных"""
    if n_clicks:
        with buffer_lock:
            df = pd.DataFrame(data_buffer)
        df.to_csv(f"data/downloads/{config['data']['download_range']}_{int(time.time())}.csv", index=False)
        logger.info(f"Data downloaded for {config['data']['download_range']}")
    return n_clicks

@callback(
    Output("apply-settings", "n_clicks"),
    Input("apply-settings", "n_clicks"),
    Input("buffer-size", "value"),
    Input("rsi-window", "value"),
    Input("sma-window", "value"),
    Input("update-interval", "value"),
    Input("download-range", "value"),
    Input("websocket-interval-1hour", "value"),
    Input("websocket-interval-1day", "value"),
    Input("websocket-interval-1month", "value"),
    Input("min-records", "value"),
    Input("train-interval", "value"),
    Input("max-depth", "value"),
    Input("n_estimators", "value"),
    Input("hourly-max-depth", "value"),
    Input("hourly-n_estimators", "value"),
    Input("train-window-minutes", "value"),
    Input("hourly-train-window-minutes", "value")
)
def update_settings(n_clicks, buffer_size, rsi_window, sma_window, update_interval, download_range,
                    websocket_interval_1hour, websocket_interval_1day, websocket_interval_1month,
                    min_records, train_interval, max_depth, n_estimators, hourly_max_depth,
                    hourly_n_estimators, train_window_minutes, hourly_train_window_minutes):
    """Обновление настроек"""
    if n_clicks:
        try:
            new_config = load_config()
            new_config["data"]["buffer_size"] = int(buffer_size) if buffer_size else new_config["data"]["buffer_size"]
            new_config["indicators"]["rsi_window"] = int(rsi_window) if rsi_window else new_config["indicators"]["rsi_window"]
            new_config["indicators"]["sma_window"] = int(sma_window) if sma_window else new_config["indicators"]["sma_window"]
            new_config["visual"]["update_interval"] = int(update_interval) if update_interval else new_config["visual"]["update_interval"]
            new_config["data"]["download_range"] = download_range if download_range else new_config["data"]["download_range"]
            new_config["data"]["websocket_intervals"]["1hour"] = websocket_interval_1hour if websocket_interval_1hour else new_config["data"]["websocket_intervals"]["1hour"]
            new_config["data"]["websocket_intervals"]["1day"] = websocket_interval_1day if websocket_interval_1day else new_config["data"]["websocket_intervals"]["1day"]
            new_config["data"]["websocket_intervals"]["1month"] = websocket_interval_1month if websocket_interval_1month else new_config["data"]["websocket_intervals"]["1month"]
            new_config["data"]["min_records"] = int(min_records) if min_records else new_config["data"]["min_records"]
            new_config["data"]["train_interval"] = int(train_interval) if train_interval else new_config["data"]["train_interval"]
            new_config["model"]["max_depth"] = int(max_depth) if max_depth else new_config["model"]["max_depth"]
            new_config["model"]["n_estimators"] = int(n_estimators) if n_estimators else new_config["model"]["n_estimators"]
            new_config["model"]["hourly_max_depth"] = int(hourly_max_depth) if hourly_max_depth else new_config["model"]["hourly_max_depth"]
            new_config["model"]["hourly_n_estimators"] = int(hourly_n_estimators) if hourly_n_estimators else new_config["model"]["hourly_n_estimators"]
            new_config["model"]["train_window_minutes"] = int(train_window_minutes) if train_window_minutes else new_config["model"]["train_window_minutes"]
            new_config["model"]["hourly_train_window_minutes"] = int(hourly_train_window_minutes) if hourly_train_window_minutes else new_config["model"]["hourly_train_window_minutes"]
            save_config(new_config)
            logger.info("Settings saved to config.yaml")

        except Exception as e:
            logger.error(f"Error in update_settings: {e}", exc_info=True)
            raise  # Для отладки, чтобы увидеть ошибку в логах
    return n_clicks

@callback(
    Output("restart-btn", "n_clicks"),
    Input("restart-btn", "n_clicks"),
    prevent_initial_call=True
)
def restart_application(n_clicks):
    """Сигнализирует через файл о необходимости перезапуска"""
    try:
        logger.info("Пользователь инициировал перезапуск приложения.")
        RESTART_FLAG.touch()
        os._exit(0)
    except Exception as e:
        logger.error(f"Ошибка при попытке перезапуска: {e}")
    return n_clicks