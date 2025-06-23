from dash import dcc, html
from .utils import config

def create_layout():
    """Создание макета дашборда"""
    return html.Div([
        html.Div(id="main-content", children=create_online_layout()),
        dcc.Interval(id="interval-component", interval=config["visual"]["update_interval"], n_intervals=0),
        dcc.Store(id="graph-layout", data={}),
        dcc.Store(id="interval-update", data=config["visual"]["update_interval"])
    ])

def create_online_layout():
    """Макет для онлайн-режима"""
    return html.Div([
        html.Div([
            dcc.Tabs(id="control-tabs", value="controls", children=[
                dcc.Tab(label="Управление", value="controls", children=[
                    html.Label("Период обучения (минуты):"),
                    dcc.Input(id="train-period", type="number", value=config["model"]["train_window_minutes"]),
                    dcc.Checklist(id="show-candles", options=[{"label": "Показать свечи", "value": "candles"}],
                                  value=[]),
                    dcc.Checklist(id="show-error-band", options=[{"label": "Зона погрешности", "value": "show"}],
                                  value=["show"] if config["visual"]["show_error_band"] else []),
                    dcc.Dropdown(id="forecast-range", options=[
                        {"label": "1 минута", "value": "1min"},
                        {"label": "1 час", "value": "1hour"},
                    ], value="1min"),
                    html.Label("Диапазон автоскейлинга:"),
                    dcc.Dropdown(id="autoscale-range", options=[
                        {"label": "10 минут", "value": "10min"},
                        {"label": "1 час", "value": "1hour"},
                        {"label": "Без ограничений", "value": "full"},
                    ], value="full"),
                    html.Button("Скачать данные", id="download-btn"),
                    html.Button("Перезапустить приложение", id="restart-btn", n_clicks=0),
                ]),
                dcc.Tab(label="Настройки", value="settings", children=create_settings_panel()),
            ]),
        ], style={"width": "20%", "display": "inline-block", "vertical-align": "top"}),
        html.Div([
            dcc.Graph(id="main-graph", config={"displayModeBar": True, "scrollZoom": True, "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d"]}),
            dcc.Graph(id="predictions-graph", style={"display": "none"}, config={"displayModeBar": True}),
        ], style={"width": "80%", "display": "inline-block"}),
    ])

def create_settings_panel():
    """Создание панели настроек"""
    return html.Div([
        html.Label("Размер буфера:"),
        dcc.Input(id="buffer-size", type="number", value=config["data"]["buffer_size"]),
        html.Label("Окно RSI:"),
        dcc.Input(id="rsi-window", type="number", value=config["indicators"]["rsi_window"]),
        html.Label("Окно SMA:"),
        dcc.Input(id="sma-window", type="number", value=config["indicators"]["sma_window"]),
        html.Label("Интервал обновления графика (мс):"),
        dcc.Input(id="update-interval", type="number", value=config["visual"]["update_interval"]),
        html.Label("Диапазон загрузки данных:"),
        dcc.Dropdown(id="download-range", options=[
            {"label": "1 час", "value": "1hour"},
            {"label": "1 день", "value": "1day"},
            {"label": "1 месяц", "value": "1month"},
        ], value=config["data"]["download_range"]),
        html.Label("WebSocket интервал для 1 часа:"),
        dcc.Dropdown(id="websocket-interval-1hour", options=[
            {"label": "1 секунда", "value": "1s"},
            {"label": "1 минута", "value": "1m"},
            {"label": "15 минут", "value": "15m"},
            {"label": "1 час", "value": "1h"},
        ], value=config["data"]["websocket_intervals"]["1hour"]),
        html.Label("WebSocket интервал для 1 дня:"),
        dcc.Dropdown(id="websocket-interval-1day", options=[
            {"label": "1 секунда", "value": "1s"},
            {"label": "1 минута", "value": "1m"},
            {"label": "15 минут", "value": "15m"},
            {"label": "1 час", "value": "1h"},
        ], value=config["data"]["websocket_intervals"]["1day"]),
        html.Label("WebSocket интервал для 1 месяца:"),
        dcc.Dropdown(id="websocket-interval-1month", options=[
            {"label": "1 секунда", "value": "1s"},
            {"label": "1 минута", "value": "1m"},
            {"label": "15 минут", "value": "15m"},
            {"label": "1 час", "value": "1h"},
        ], value=config["data"]["websocket_intervals"]["1month"]),
        html.Label("Минимальное количество записей:"),
        dcc.Input(id="min-records", type="number", value=config["data"]["min_records"]),
        html.Label("Период обучения модели (сек):"),
        dcc.Input(id="train-interval", type="number", value=config["data"]["train_interval"]),
        html.Button("Применить", id="apply-settings"),
    ])