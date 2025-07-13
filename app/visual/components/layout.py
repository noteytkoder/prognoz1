from dash import dcc, html
from .utils import config

def create_layout():
    """Создание макета дашборда"""
    return html.Div([
        html.Div(id="main-content", children=create_online_layout()),
        dcc.Interval(id="interval-component", interval=config["visual"]["update_interval"], n_intervals=0),
        dcc.Store(id="graph-layout", data={}),
        dcc.Store(id="interval-update", data=config["visual"]["update_interval"]),
        dcc.Store(id="pred-min-layout", data={}),
        dcc.Store(id="pred-hour-layout", data={})
    ])

def create_online_layout():
    """Макет для онлайн-режима"""
    return html.Div([
        html.Div([
            dcc.Tabs(id="control-tabs", value="controls", children=[
                dcc.Tab(label="Управление", value="controls", children=[
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
            dcc.Graph(id="predictions-graph-min", style={"display": "none"}, config={"displayModeBar": True, "scrollZoom": True, "modeBarButtonsToAdd": ["zoom2d", "pan2d"]}),
            dcc.Graph(id="predictions-graph-hour", style={"display": "none"}, config={"displayModeBar": True, "scrollZoom": True, "modeBarButtonsToAdd": ["zoom2d", "pan2d"]}),
        ], style={"width": "80%", "display": "inline-block"}),
    ])

def create_settings_panel():
    """Создание панели настроек с всплывающими подсказками"""
    return html.Div([
        html.H3("Настройки", style={"margin-top": "20px"}),
        html.Div([
            html.Label("Размер буфера:"),
            dcc.Input(id="buffer-size", type="number", value=config["data"]["buffer_size"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Максимальное количество записей в буфере данных (например, 20000)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Окно RSI:"),
            dcc.Input(id="rsi-window", type="number", value=config["indicators"]["rsi_window"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Период для расчета индикатора RSI (например, 7 свечей)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Окно SMA:"),
            dcc.Input(id="sma-window", type="number", value=config["indicators"]["sma_window"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Период для расчета скользящей средней SMA (например, 3 свечи)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Интервал обновления графика (мс):"),
            dcc.Input(id="update-interval", type="number", value=config["visual"]["update_interval"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Частота обновления графиков в миллисекундах (например, 1000 = 1 секунда)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Диапазон загрузки данных:"),
            dcc.Dropdown(id="download-range", options=[
                {"label": "1 час", "value": "1hour"},
                {"label": "1 день", "value": "1day"},
                {"label": "1 месяц", "value": "1month"},
            ], value=config["data"]["download_range"], style={"width": "200px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Период времени для загрузки исторических данных (например, 1 день)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("WebSocket интервал для 1 часа:"),
            dcc.Dropdown(id="websocket-interval-1hour", options=[
                {"label": "1 секунда", "value": "1s"},
                {"label": "1 минута", "value": "1m"},
                {"label": "15 минут", "value": "15m"},
                {"label": "1 час", "value": "1h"},
            ], value=config["data"]["websocket_intervals"]["1hour"], style={"width": "200px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Интервал обновления данных WebSocket для диапазона 1 час."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("WebSocket интервал для 1 дня:"),
            dcc.Dropdown(id="websocket-interval-1day", options=[
                {"label": "1 секунда", "value": "1s"},
                {"label": "1 минута", "value": "1m"},
                {"label": "15 минут", "value": "15m"},
                {"label": "1 час", "value": "1h"},
            ], value=config["data"]["websocket_intervals"]["1day"], style={"width": "200px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Интервал обновления данных WebSocket для диапазона 1 день."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("WebSocket интервал для 1 месяца:"),
            dcc.Dropdown(id="websocket-interval-1month", options=[
                {"label": "1 секунда", "value": "1s"},
                {"label": "1 минута", "value": "1m"},
                {"label": "15 минут", "value": "15m"},
                {"label": "1 час", "value": "1h"},
            ], value=config["data"]["websocket_intervals"]["1month"], style={"width": "200px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Интервал обновления данных WebSocket для диапазона 1 месяц."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Минимальное количество записей:"),
            dcc.Input(id="min-records", type="number", value=config["data"]["min_records"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Минимальное количество записей для обучения модели (например, 60)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Период обучения модели (сек):"),
            dcc.Input(id="train-interval", type="number", value=config["data"]["train_interval"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Интервал в секундах между переобучением модели (например, 30)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Максимальная глубина модели (минутная):"),
            dcc.Input(id="max-depth", type="number", value=config["model"]["max_depth"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Максимальная глубина деревьев в случайном лесу для минутной модели. 0 = без ограничений."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Количество деревьев (минутная):"),
            dcc.Input(id="n_estimators", type="number", value=config["model"]["n_estimators"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Число деревьев в случайном лесу для минутной модели. Большее значение увеличивает точность, но замедляет обучение (рекомендуется 50-200)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Максимальная глубина модели (часовая):"),
            dcc.Input(id="hourly-max-depth", type="number", value=config["model"]["hourly_max_depth"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Максимальная глубина деревьев в случайном лесу для часовой модели. 0 = без ограничений."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Количество деревьев (часовая):"),
            dcc.Input(id="hourly-n_estimators", type="number", value=config["model"]["hourly_n_estimators"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Число деревьев в случайном лесу для часовой модели. Большее значение увеличивает точность, но замедляет обучение (рекомендуется 50-200)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Период обучения модели (минутная, минуты):"),
            dcc.Input(id="train-window-minutes", type="number", value=config["model"]["train_window_minutes"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Количество минут данных для обучения минутной модели (например, 60)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Div([
            html.Label("Период обучения модели (часовая, минуты):"),
            dcc.Input(id="hourly-train-window-minutes", type="number", value=config["model"]["hourly_train_window_minutes"], style={"width": "100px", "margin": "10px"}),
            html.Span("?", className="tooltip", title="Количество минут данных для обучения часовой модели (например, 1440)."),
        ], style={"display": "flex", "align-items": "center"}),
        html.Button("Применить", id="apply-settings"),
    ])
