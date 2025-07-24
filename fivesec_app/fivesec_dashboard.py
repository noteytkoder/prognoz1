from dash import Dash, dcc, html, Input, Output, callback, State
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import os
import psutil
from pathlib import Path
import pytz
from dash_auth import BasicAuth
import secrets
from fivesec_app.data_handler import fivesec_buffer, buffer_lock, process_fivesec_data
from fivesec_app.model import predict_fivesec
from fivesec_app.config_manager import load_config, load_environment_config, save_config
from fivesec_app.logger import setup_logger

config = load_config()
env_name = config["app_env"]
env_config = load_environment_config()
logger = setup_logger()
RESTART_FLAG = Path("restart.flag")
APP_START_TIME = time.time()
cached_df = None
cached_timestamp = None
cached_pred_df = None
cached_pred_timestamp = None

DASH_AUTH_CREDENTIALS = {
    config["auth"]["username"]: config["auth"]["password"]
}

SECRET_KEY = secrets.token_hex(16)
dash_app = Dash(__name__, assets_folder="static")
dash_app.server.secret_key = SECRET_KEY
BasicAuth(dash_app, DASH_AUTH_CREDENTIALS)

def create_layout():
    """Создание макета дашборда"""
    return html.Div([
        html.Div(id="main-content", children=create_fivesec_layout()),
        html.Div(id="server-status", children=create_server_status_panel()),
        dcc.Interval(id="interval-component", interval=config["visual"]["update_interval"], n_intervals=0),
        dcc.Interval(id="server-status-interval", interval=5000, n_intervals=0),
        dcc.Store(id="graph-layout", data={}),
        dcc.Store(id="interval-update", data=config["visual"]["update_interval"]),
        dcc.Store(id="pred-fivesec-layout", data={}),
    ])

def create_fivesec_layout():
    """Макет для 5-секундного режима"""
    return html.Div([
        html.Div([
            dcc.Tabs(id="control-tabs", value="controls", children=[
                dcc.Tab(label="Управление", value="controls", children=[
                    dcc.Checklist(id="show-candles", options=[{"label": "Показать свечи", "value": "candles"}],
                                  value=[]),
                    dcc.Checklist(id="show-error-band", options=[{"label": "Зона погрешности", "value": "show"}],
                                  value=["show"] if config["visual"]["show_error_band"] else []),
                    html.Label("Диапазон автоскейлинга:"),
                    dcc.Dropdown(id="autoscale-range", options=[
                        {"label": "10 минут", "value": "10min"},
                        {"label": "1 час", "value": "1hour"},
                    ], value="10min"),
                    html.Button("Скачать данные", id="download-btn"),
                    html.Button("Перезапустить приложение", id="restart-btn", n_clicks=0),
                ]),
                dcc.Tab(label="Настройки", value="settings", children=create_settings_panel()),
            ]),
        ], style={"width": "20%", "display": "inline-block", "vertical-align": "top"}),
        html.Div([
            dcc.Graph(id="main-graph", config={"displayModeBar": True, "scrollZoom": True, "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d"]}),
            dcc.Graph(id="predictions-graph-fivesec", style={"display": "none"}, config={"displayModeBar": True, "scrollZoom": True, "modeBarButtonsToAdd": ["zoom2d", "pan2d"]}),
        ], style={"width": "80%", "display": "inline-block"}),
    ])

def create_server_status_panel():
    """Создание панели статуса сервера"""
    return html.Div([
        html.H3("Статус сервера", style={"margin-top": "20px"}),
        html.Div(id="server-status-content", children=[
            html.P("Загрузка CPU: ...%", id="cpu-usage"),
            html.P("Использование памяти: ...%", id="memory-usage"),
            html.P("Время работы сервера: ...", id="uptime"),
            html.P("Статус: ...", id="server-health")
        ], style={"border": "1px solid #444", "padding": "10px", "margin": "10px"}),
    ])

def create_settings_panel():
    """Создание панели настроек"""
    return html.Div([
        html.H3("Настройки", style={"margin-top": "20px"}),
        html.Div([
            html.Label("Размер буфера:"),
            dcc.Input(id="buffer-size", type="number", value=config["data"]["buffer_size"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Максимальное количество записей в буфере данных (например, 50000)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Интервал переобучения (сек):"),
            dcc.Input(id="fivesec-train-interval", type="number", value=config["data"]["fivesec_train_interval"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Интервал в секундах между переобучением модели (например, 60)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Максимальная глубина модели:"),
            dcc.Input(id="fivesec-max-depth", type="number", value=config["model"]["fivesec_max_depth"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Максимальная глубина деревьев в случайном лесу (0 = без ограничений)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Количество деревьев:"),
            dcc.Input(id="fivesec-n_estimators", type="number", value=config["model"]["fivesec_n_estimators"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Число деревьев в случайном лесу (рекомендуется 50-200)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Период обучения (сек):"),
            dcc.Input(id="fivesec-train-window-seconds", type="number", value=config["model"]["fivesec_train_window_seconds"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Количество секунд данных для обучения модели (например, 3600)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Button("Применить", id="apply-settings"),
    ])

dash_app.layout = create_layout()

def prepare_data(data_copy, msk_tz):
    """Подготовка данных: ресэмплинг до 5 секунд"""
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

    df = process_fivesec_data(df)
    cached_df = df
    cached_timestamp = latest_timestamp
    return df, latest_timestamp

def prepare_predictions(msk_tz, last_time, time_delta):
    """Подготовка данных предсказаний"""
    global cached_pred_df, cached_pred_timestamp
    mse_fivesec, mae_fivesec, pred_count = None, None, 0
    pred_df = pd.DataFrame()

    try:
        csv_file_path = "logs/fivesec_predictions.csv"
        if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
            with buffer_lock:
                pred_df = pd.read_csv(csv_file_path, encoding='utf-8')
            pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True).dt.tz_convert(msk_tz)
            pred_df["fivesec_pred_time"] = pd.to_datetime(pred_df["fivesec_pred_time"], utc=True).dt.tz_convert(msk_tz)
            pred_df = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
            if len(pred_df) > 1:
                fivesec_valid = pred_df[pred_df["fivesec_error"].notna()]
                if not fivesec_valid.empty:
                    mse_fivesec = np.mean(fivesec_valid["fivesec_error"] ** 2)
                    mae_fivesec = np.mean(fivesec_valid["fivesec_error"])
                pred_count = len(pred_df)
            cached_pred_df = pred_df
            cached_pred_timestamp = last_time

    except Exception as e:
        logger.error(f"Failed to read fivesec_predictions.csv: {e}")

    return pred_df, mse_fivesec, mae_fivesec, pred_count

def create_main_figure(df, show_candles, show_error_band, last_time, error_band_width):
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

    features = df.iloc[-1][["close", "rsi", "sma", "volume", "log_volume"]]
    features_df = pd.DataFrame([features])
    prediction = predict_fivesec(features_df)
    pred_time = last_time + pd.Timedelta(seconds=5)
    if prediction is not None:
        fig.add_trace(go.Scatter(
            x=[last_time, pred_time], y=[df["close"].iloc[-1], prediction],
            mode="lines", name="Прогноз (5 сек)",
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

def create_prediction_figure(pred_df, mse_fivesec, mae_fivesec, pred_count, last_time, time_delta, x_range_pred, y_range):
    """Создание графика предсказаний"""
    pred_fig = go.Figure()
    pred_style = {"display": "block" if not pred_df.empty else "none"}

    if not pred_df.empty:
        filtered_pred_df = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
        if not filtered_pred_df.empty:
            pred_fig.add_trace(go.Scatter(
                x=filtered_pred_df["timestamp"], y=filtered_pred_df["actual_price"],
                mode="lines", name="Фактическая цена", line=dict(color=config["visual"]["real_price_color"])
            ))
            pred_fig.add_trace(go.Scatter(
                x=filtered_pred_df["fivesec_pred_time"], y=filtered_pred_df["fivesec_pred"],
                mode="lines", name="Предсказанная цена (5 сек)", line=dict(color=config["visual"]["predicted_price_color"])
            ))
            annotation_text = (f"MSE: {mse_fivesec:.2f}, MAE: {mae_fivesec:.2f}, Количество: {pred_count}"
                              if mse_fivesec is not None and mae_fivesec is not None
                              else "Ожидание данных для расчета метрик")
            pred_fig.add_annotation(
                xref="paper", yref="paper", x=0.05, y=0.95,
                text=annotation_text,
                showarrow=False, font=dict(size=12, color="white")
            )
            pred_fig.update_layout(
                title="BTC/USDT: Фактические и предсказанные цены (5 секунд)",
                xaxis_title="Время (MSK)",
                yaxis_title="Цена (USDT)",
                xaxis_range=x_range_pred,
                yaxis_range=y_range,
                showlegend=True,
                height=400,
                template="plotly_dark",
                dragmode="zoom",
                uirevision="predictions-graph-fivesec",
                xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45)
            )

    return pred_fig, pred_style

@callback(
    Output("main-graph", "figure"),
    Output("predictions-graph-fivesec", "figure"),
    Output("predictions-graph-fivesec", "style"),
    Output("graph-layout", "data"),
    Output("pred-fivesec-layout", "data"),
    Input("interval-component", "n_intervals"),
    Input("show-candles", "value"),
    Input("show-error-band", "value"),
    Input("autoscale-range", "value"),
    Input("main-graph", "relayoutData"),
    Input("predictions-graph-fivesec", "relayoutData"),
    State("graph-layout", "data"),
    State("pred-fivesec-layout", "data"),
    prevent_initial_call=True
)
def update_graph(n, show_candles, show_error_band, autoscale_range, main_relayout_data, pred_fivesec_relayout_data,
                main_stored_layout, pred_fivesec_stored_layout):
    logger.debug(f"Starting update_graph, n_intervals={n}, autoscale_range={autoscale_range}")
    msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))

    try:
        if main_relayout_data and "xaxis.range[0]" in main_relayout_data:
            main_stored_layout = main_relayout_data
        if pred_fivesec_relayout_data and "xaxis.range[0]" in pred_fivesec_relayout_data:
            pred_fivesec_stored_layout = pred_fivesec_relayout_data

        with buffer_lock:
            if not fivesec_buffer:
                logger.warning("5-second buffer is empty")
                return go.Figure(), go.Figure(), {"display": "none"}, main_stored_layout, pred_fivesec_stored_layout
            data_copy = list(fivesec_buffer)

        df, latest_timestamp = prepare_data(data_copy, msk_tz)
        if df is None:
            return go.Figure(), go.Figure(), {"display": "none"}, main_stored_layout, pred_fivesec_stored_layout

        last_time = df.index.max()
        ranges = {"1hour": pd.Timedelta(hours=1)}
        default_time_delta = ranges.get("1hour")

        if autoscale_range == "10min":
            time_delta = pd.Timedelta(minutes=10)
            default_x_range = [last_time - time_delta, last_time + pd.Timedelta(seconds=5)]
            default_x_range_pred = [last_time - time_delta, last_time + pd.Timedelta(seconds=5)]
        else:
            time_delta = default_time_delta
            default_x_range = [last_time - time_delta, last_time]
            default_x_range_pred = [last_time - time_delta, last_time + pd.Timedelta(seconds=5)]

        x_range = ([pd.to_datetime(main_stored_layout["xaxis.range[0]"], utc=True).tz_convert(msk_tz),
                    pd.to_datetime(main_stored_layout["xaxis.range[1]"], utc=True).tz_convert(msk_tz)]
                   if main_stored_layout and "xaxis.range[0]" in main_stored_layout
                   else default_x_range)
        x_range_pred = ([pd.to_datetime(pred_fivesec_stored_layout["xaxis.range[0]"], utc=True).tz_convert(msk_tz),
                         pd.to_datetime(pred_fivesec_stored_layout["xaxis.range[1]"], utc=True).tz_convert(msk_tz)]
                        if pred_fivesec_stored_layout and "xaxis.range[0]" in pred_fivesec_stored_layout
                        else default_x_range_pred)

        df = df.loc[x_range[0]:x_range[1]].reset_index()

        current_time = pd.Timestamp.now(tz=msk_tz)
        if (current_time - last_time).total_seconds() > 60:
            fig = go.Figure()
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.5,
                text="Данные устарели. Пожалуйста, проверьте соединение.",
                showarrow=False, font=dict(size=16, color="red")
            )
            return fig, go.Figure(), {"display": "none"}, main_stored_layout, pred_fivesec_stored_layout

        if df.empty:
            return go.Figure(), go.Figure(), {"display": "none"}, main_stored_layout, pred_fivesec_stored_layout

        show_candles = "candles" in (show_candles or [])
        show_band = "show" in (show_error_band or [])
        pred_df, mse_fivesec, mae_fivesec, pred_count = prepare_predictions(msk_tz, last_time, time_delta)

        error_band_width = config["visual"]["error_band_min"]
        if mae_fivesec is not None:
            error_band_width = max(mae_fivesec * config["visual"]["error_band_multiplier"], config["visual"]["error_band_min"])

        min_price = df["close"].min()
        max_price = df["close"].max()
        y_range = [min_price * 0.995, max_price * 1.005]
        fig = create_main_figure(df, show_candles, show_band, last_time, error_band_width)

        fig.update_layout(
            title="BTC/USDT: Цены и прогноз (5 секунд)",
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

        pred_fig, pred_style = create_prediction_figure(pred_df, mse_fivesec, mae_fivesec, pred_count, last_time, time_delta, x_range_pred, y_range)

        logger.debug("Graph updated successfully")
        return fig, pred_fig, pred_style, main_stored_layout, pred_fivesec_stored_layout

    except Exception as e:
        logger.error(f"Error in update_graph: {e}", exc_info=True)
        return go.Figure(), go.Figure(), {"display": "none"}, main_stored_layout, pred_fivesec_stored_layout

@callback(
    Output("download-btn", "n_clicks"),
    Input("download-btn", "n_clicks")
)
def download_data(n_clicks):
    """Скачивание данных"""
    if n_clicks:
        with buffer_lock:
            df = pd.DataFrame(fivesec_buffer)
        os.makedirs("data/downloads", exist_ok=True)
        df.to_csv(f"data/downloads/fivesec_{int(time.time())}.csv", index=False)
        logger.info("5-second data downloaded")
    return n_clicks

@callback(
    Output("apply-settings", "n_clicks"),
    Input("apply-settings", "n_clicks"),
    Input("buffer-size", "value"),
    Input("fivesec-train-interval", "value"),
    Input("fivesec-max-depth", "value"),
    Input("fivesec-n_estimators", "value"),
    Input("fivesec-train-window-seconds", "value")
)
def update_settings(n_clicks, buffer_size, fivesec_train_interval, fivesec_max_depth, fivesec_n_estimators, fivesec_train_window_seconds):
    """Обновление настроек"""
    if n_clicks:
        try:
            new_config = load_config()
            new_config["data"]["buffer_size"] = int(buffer_size) if buffer_size else new_config["data"]["buffer_size"]
            new_config["data"]["fivesec_train_interval"] = int(fivesec_train_interval) if fivesec_train_interval else new_config["data"]["fivesec_train_interval"]
            new_config["model"]["fivesec_max_depth"] = int(fivesec_max_depth) if fivesec_max_depth else new_config["model"]["fivesec_max_depth"]
            new_config["model"]["fivesec_n_estimators"] = int(fivesec_n_estimators) if fivesec_n_estimators else new_config["model"]["fivesec_n_estimators"]
            new_config["model"]["fivesec_train_window_seconds"] = int(fivesec_train_window_seconds) if fivesec_train_window_seconds else new_config["model"]["fivesec_train_window_seconds"]
            save_config(new_config)
            logger.info("Settings saved to config.yaml")
        except Exception as e:
            logger.error(f"Error in update_settings: {e}", exc_info=True)
            raise
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

@callback(
    Output("cpu-usage", "children"),
    Output("memory-usage", "children"),
    Output("uptime", "children"),
    Output("server-health", "children"),
    Input("server-status-interval", "n_intervals"),
    prevent_initial_call=True
)
def update_server_status(n_intervals):
    """Обновление статуса сервера"""
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        uptime_seconds = time.time() - APP_START_TIME
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(days)}д {int(hours)}ч {int(minutes)}м {int(seconds)}с"
        health_status = "ОК"
        if cpu_usage > 90 or memory_usage > 90:
            health_status = "Высокая нагрузка"
            logger.warning(f"High server load detected: CPU={cpu_usage}%, Memory={memory_usage}%")
        elif cpu_usage > 75 or memory_usage > 75:
            health_status = "Повышенная нагрузка"
            logger.debug(f"Elevated server load: CPU={cpu_usage}%, Memory={memory_usage}%")
        logger.debug(f"Server status updated: CPU={cpu_usage}%, Memory={memory_usage}%, Uptime={uptime_str}, Health={health_status}")
        return (
            f"Загрузка CPU: {cpu_usage:.1f}%",
            f"Использование памяти: {memory_usage:.1f}%",
            f"Время работы сервера: {uptime_str}",
            f"Статус: {health_status}"
        )
    except Exception as e:
        logger.error(f"Error in update_server_status: {e}", exc_info=True)
        return (
            "Загрузка CPU: Ошибка",
            "Использование памяти: Ошибка",
            f"Время работы сервера: Ошибка",
            "Статус: Ошибка"
        )

def start_fivesec_dash():
    """Запуск сервера Dash"""
    try:
        logger.info("Starting 5-second Dash server")
        dash_app.run(port=env_config[env_name]["port_dash"], host="0.0.0.0")
    except Exception as e:
        logger.error(f"5-second Dash server error: {e}")
        raise