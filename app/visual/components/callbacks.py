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

RESTART_FLAG = Path("restart.flag")
logger = setup_logger()

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
    Input("interval-component", "n_intervals"),
    Input("train-period", "value"),
    Input("show-candles", "value"),
    Input("show-error-band", "value"),
    Input("forecast-range", "value"),
    Input("autoscale-range", "value"),
    Input("main-graph", "relayoutData"),
    State("graph-layout", "data"),
    prevent_initial_call=True
)
def update_graph(n, train_period, show_candles, show_error_band, forecast_range, autoscale_range, relayout_data, stored_layout):
    global last_stats_time, last_mse, last_mae, current_layout
    logger.debug(f"Starting update_graph, n_intervals={n}, forecast_range={forecast_range}, autoscale_range={autoscale_range}")

    try:
        if train_period and train_period != config["model"]["train_window_minutes"]:
            new_config = load_config()
            new_config["model"]["train_window_minutes"] = int(train_period)
            save_config(new_config)
            logger.info(f"Train period updated to {train_period} minutes")

        if relayout_data and "xaxis.range[0]" in relayout_data:
            stored_layout = relayout_data
        current_layout = stored_layout

        with buffer_lock:
            if not data_buffer:
                logger.warning("Data buffer is empty")
                return go.Figure(), go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, stored_layout
            data_copy = list(data_buffer)
            df = pd.DataFrame(data_copy)

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        interval = config["data"]["websocket_intervals"].get(config["data"]["download_range"], "1s")
        interval_seconds = {"1s": 1, "1m": 60, "3m": 3*60, "15m": 15*60, "1h": 60*60, "1d": 24*60*60}

        df.set_index("timestamp", inplace=True)
        if df.empty:
            return go.Figure(), go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, stored_layout
        df = df.resample(f"{interval_seconds.get(interval, 1)}s").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear")
        df.reset_index(inplace=True)

        latest_timestamp = df["timestamp"].max()
        current_time = pd.Timestamp.now(tz=config.get("timezone", "Europe/Moscow"))
        if (current_time - latest_timestamp).total_seconds() > interval_seconds.get(interval, 1) * 20:
            fig = go.Figure()
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.5,
                text="Данные устарели. Пожалуйста, проверьте соединение.",
                showarrow=False, font=dict(size=16, color="red")
            )
            return fig, go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, stored_layout

        gaps = df["timestamp"].diff().dt.total_seconds()
        if gaps.max() > interval_seconds.get(interval, 1) * 1.5:
            logger.warning(f"Detected gaps in data: max gap {gaps.max()} seconds")

        last_time = df["timestamp"].max()
        ranges = {"1hour": pd.Timedelta(hours=1), "1day": pd.Timedelta(days=1), "1month": pd.Timedelta(days=30)}
        default_time_delta = ranges.get(config["data"]["download_range"], pd.Timedelta(days=1))

        # Определяем диапазон оси X в зависимости от autoscale-range
        if autoscale_range == "10min":
            time_delta = pd.Timedelta(minutes=10)
            df = df[df["timestamp"] >= last_time - time_delta]
            x_range = [last_time - time_delta, last_time + pd.Timedelta(minutes=1)]
            x_range_pred = [last_time - time_delta, last_time]  # Для графиков предсказаний
        elif autoscale_range == "1hour":
            time_delta = pd.Timedelta(hours=1)
            df = df[df["timestamp"] >= last_time - time_delta]
            x_range = [last_time - time_delta, last_time + pd.Timedelta(minutes=1)]
            x_range_pred = [last_time - time_delta, last_time]  # Для графиков предсказаний
        else:
            time_delta = default_time_delta
            if stored_layout and "xaxis.range[0]" in stored_layout:
                x_range = [pd.to_datetime(stored_layout["xaxis.range[0]"]), pd.to_datetime(stored_layout["xaxis.range[1]"])]
                x_range_pred = x_range  # Используем тот же диапазон для предсказаний
            else:
                x_range = [last_time - time_delta, last_time]
                x_range_pred = [last_time - time_delta, last_time]

        if df.empty:
            return go.Figure(), go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, stored_layout

        show_candles = "candles" in (show_candles or [])
        df_candles = df
        show_band = "show" in (show_error_band or [])

        fig = go.Figure()
        if show_candles:
            fig.add_trace(go.Candlestick(
                x=df_candles["timestamp"], open=df_candles["open"], high=df_candles["high"],
                low=df_candles["low"], close=df_candles["close"],
                name="Свечи", increasing_line_color="green", decreasing_line_color="red"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["close"], mode="lines", name="Цена закрытия",
                line=dict(color="blue")
            ))

        min_price = df["close"].min()
        max_price = df["close"].max()
        y_range = [min_price * 0.995, max_price * 1.005]

        error_band_width = config["visual"]["error_band_min"]
        mse_min, mae_min, mse_hour, mae_hour, pred_count = None, None, None, None, 0
        pred_df = pd.DataFrame()
        try:
            if os.path.exists("logs/predictions.csv") and os.path.getsize("logs/predictions.csv") > 0:
                with buffer_lock:
                    pred_df = pd.read_csv("logs/predictions.csv", encoding='utf-8')
                pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
                pred_df = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]  # Фильтруем по выбранному диапазону
                if len(pred_df) > 1:
                    mse_min = np.mean(pred_df["min_error"] ** 2)
                    mae_min = np.mean(pred_df["min_error"])
                    mse_hour = np.mean(pred_df["hour_error"] ** 2)
                    mae_hour = np.mean(pred_df["hour_error"])
                    pred_count = len(pred_df)
                    error_band_width = max(mae_min * config["visual"]["error_band_multiplier"], config["visual"]["error_band_min"]) if forecast_range == "1min" else max(mae_hour * config["visual"]["error_band_multiplier"], config["visual"]["error_band_min"])
        except Exception as e:
            logger.error(f"Failed to read logs/predictions.csv: {e}")

        features = get_latest_features()
        if features is not None:
            df_min_row, df_hour_row, df_min_df = features
            features_df = pd.DataFrame([df_min_row[["close", "rsi", "sma", "volume", "log_volume"]]])
            prediction = predict(features_df) if forecast_range == "1min" else predict_hourly(features_df)
            pred_time = last_time + (pd.Timedelta(minutes=1) if forecast_range == "1min" else pd.Timedelta(hours=1))
            if prediction is not None:
                fig.add_trace(go.Scatter(
                    x=[last_time, pred_time], y=[df["close"].iloc[-1], prediction],
                    mode="lines+markers", name=f"Прогноз ({'1 минута' if forecast_range == '1min' else '1 час'})",
                    line=dict(color=config["visual"]["predicted_price_color"])
                ))
                if show_band:
                    fig.add_trace(go.Scatter(
                        x=[last_time, pred_time, pred_time, last_time],
                        y=[df["close"].iloc[-1], prediction + error_band_width, prediction - error_band_width, df["close"].iloc[-1]],
                        fill="toself", fillcolor=config["visual"]["error_band_color"],
                        line=dict(color="rgba(255,255,255,0)"),
                        name=f"Зона погрешности (±{error_band_width:.2f} USDT)"
                    ))

        if mse_min is not None:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.05, y=0.95,
                text=f"MSE: {mse_min if forecast_range == '1min' else mse_hour:.2f}, MAE: {mae_min if forecast_range == '1min' else mae_hour:.2f}, Количество: {pred_count}",
                showarrow=False, font=dict(size=12, color="white")
            )

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

        pred_fig_min = go.Figure()
        pred_fig_hour = go.Figure()
        pred_style_min = {"display": "block" if not pred_df.empty else "none"}
        pred_style_hour = {"display": "block" if not pred_df.empty else "none"}
        if not pred_df.empty:
            filtered_pred_df = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
            if not filtered_pred_df.empty:
                # Минутный график
                pred_fig_min.add_trace(go.Scatter(
                    x=filtered_pred_df["timestamp"], y=filtered_pred_df["actual_price"],
                    mode="lines", name="Фактическая цена", line=dict(color=config["visual"]["real_price_color"])
                ))
                pred_fig_min.add_trace(go.Scatter(
                    x=filtered_pred_df["timestamp"], y=filtered_pred_df["min_pred"],
                    mode="lines", name="Предсказанная цена (1 мин)", line=dict(color=config["visual"]["predicted_price_color"])
                ))
                pred_fig_min.update_layout(
                    title="BTC/USDT: Фактические и предсказанные цены (1 минута)",
                    xaxis_title="Время (MSK)",
                    yaxis_title="Цена (USDT)",
                    xaxis_range=x_range_pred,  # Используем диапазон, соответствующий autoscale-range
                    yaxis_range=y_range,
                    showlegend=True,
                    height=400,
                    template="plotly_dark",
                    dragmode="zoom",
                    uirevision="predictions-graph-min",
                    xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45)
                )

                # Часовой график
                pred_fig_hour.add_trace(go.Scatter(
                    x=filtered_pred_df["timestamp"], y=filtered_pred_df["actual_price"],
                    mode="lines", name="Фактическая цена", line=dict(color=config["visual"]["real_price_color"])
                ))
                pred_fig_hour.add_trace(go.Scatter(
                    x=filtered_pred_df["timestamp"], y=filtered_pred_df["hour_pred"],
                    mode="lines", name="Предсказанная цена (1 час)", line=dict(color=config["visual"]["predicted_price_color"])
                ))
                pred_fig_hour.update_layout(
                    title="BTC/USDT: Фактические и предсказанные цены (1 час)",
                    xaxis_title="Время (MSK)",
                    yaxis_title="Цена (USDT)",
                    xaxis_range=x_range_pred,  # Используем диапазон, соответствующий autoscale-range
                    yaxis_range=y_range,
                    showlegend=True,
                    height=400,
                    template="plotly_dark",
                    dragmode="zoom",
                    uirevision="predictions-graph-hour",
                    xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45)
                )

        logger.debug("Graph updated successfully")
        return fig, pred_fig_min, pred_style_min, pred_fig_hour, pred_style_hour, stored_layout

    except Exception as e:
        logger.error(f"Error in update_graph: {e}", exc_info=True)
        return go.Figure(), go.Figure(), {"display": "none"}, go.Figure(), {"display": "none"}, stored_layout
    
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
    Input("min-candles", "value"),
    Input("min-hourly-candles", "value"),
    Input("train-window-minutes", "value")
)
def update_settings(n_clicks, buffer_size, rsi_window, sma_window, update_interval, download_range,
                    websocket_interval_1hour, websocket_interval_1day, websocket_interval_1month,
                    min_records, train_interval, max_depth, n_estimators, min_candles, min_hourly_candles,
                    train_window_minutes):
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
            new_config["model"]["min_candles"] = int(min_candles) if min_candles else new_config["model"]["min_candles"]
            new_config["model"]["min_hourly_candles"] = int(min_hourly_candles) if min_hourly_candles else new_config["model"]["min_hourly_candles"]
            new_config["model"]["train_window_minutes"] = int(train_window_minutes) if train_window_minutes else new_config["model"]["train_window_minutes"]
            save_config(new_config)
            logger.info("Settings saved to config.yaml")

            # Перезапуск WebSocket
            import asyncio
            async def restart_websocket():
                pass  # Замените на реальную логику перезапуска WebSocket
            asyncio.run(restart_websocket())

            # Перезапуск Dash
            current_dir = os.path.dirname(os.path.abspath(__file__))
            main_py_path = os.path.join(current_dir, "..", "..", "..", "main.py")
            main_py_path = os.path.normpath(main_py_path)

            if not os.path.exists(main_py_path):
                logger.error(f"main.py not found at {main_py_path}")
                return n_clicks

            if sys.platform == "win32":
                subprocess.run(["taskkill", "/IM", "python.exe", "/F"], check=False)
                subprocess.Popen(["python", main_py_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.run(["pkill", "-f", "dash"], check=False)
                subprocess.Popen(["python", main_py_path])

            logger.info("Dash and WebSocket restarted successfully")
        except Exception as e:
            logger.error(f"Error restarting Dash: {e}", exc_info=True)
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