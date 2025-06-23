import pandas as pd
import pytz
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.logs.logger import setup_logger
from app.model.train import train_model, train_hourly_model, predict, predict_hourly
from app.model.indicators import calculate_indicators
from app.config.manager import load_config
import os

logger = setup_logger()
config = load_config()

def process_offline_file(df, model_type="1min", n_estimators=None, max_depth=None):
    """Обработка загруженного CSV-файла с данными Binance"""
    try:
        # Создаём папку offline_results, если её нет
        os.makedirs("data/offline_results", exist_ok=True)
        
        # Предполагаем, что CSV имеет формат Binance с 12 столбцами
        df.columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ]
        
        # Преобразуем timestamp из микросекунд в datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(config.get("timezone", "Europe/Moscow"))
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        
        # Очистка данных
        df = df.dropna()
        df["volume"] = df["volume"].replace(0, 1e-8)
        if df.empty or df["timestamp"].isnull().any() or df["close"].isnull().any():
            logger.error("Invalid data in CSV file")
            return False
        
        df = df.drop_duplicates(subset=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = calculate_indicators(df)
        
        if df.empty or df["rsi"].isnull().any() or np.any(np.isinf(df.values)):
            logger.error("Failed to calculate indicators or invalid values detected")
            return False
        
        # Разделение на обучение и тест (80% на обучение, 20% на тест)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Обучение модели
        if model_type == "1min":
            train_model(
                train_df,
                n_estimators=n_estimators if n_estimators is not None else config["model"]["n_estimators"],
                max_depth=max_depth if max_depth is not None else config["model"]["max_depth"]
            )
        else:
            train_hourly_model(
                train_df,
                n_estimators=n_estimators if n_estimators is not None else config["model"]["n_estimators"],
                max_depth=max_depth if max_depth is not None else config["model"]["max_depth"]
            )
        
        # Создание предсказаний
        predictions = []
        for i in range(len(train_df) - 1):
            features = train_df.iloc[i][["close", "rsi", "sma", "volume", "log_volume"]]
            prediction = predict(pd.DataFrame([features])) if model_type == "1min" else predict_hourly(pd.DataFrame([features]))
            if prediction is not None:
                predictions.append({
                    "timestamp": train_df.index[i + 1],
                    "actual_price": train_df.iloc[i + 1]["close"],
                    "predicted_price": prediction,
                    "error": abs(train_df.iloc[i + 1]["close"] - prediction)
                })
        
        for i in range(len(test_df) - 1):
            features = test_df.iloc[i][["close", "rsi", "sma", "volume", "log_volume"]]
            prediction = predict(pd.DataFrame([features])) if model_type == "1min" else predict_hourly(pd.DataFrame([features]))
            if prediction is not None:
                predictions.append({
                    "timestamp": test_df.index[i + 1],
                    "actual_price": test_df.iloc[i + 1]["close"],
                    "predicted_price": prediction,
                    "error": abs(test_df.iloc[i + 1]["close"] - prediction)
                })
        
        if not predictions:
            logger.error("No predictions generated")
            return False
        
        # Сохранение предсказаний
        pred_df = pd.DataFrame(predictions)
        output_path = f"data/offline_results/predictions_{int(pd.Timestamp.now().timestamp())}.csv"
        pred_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Predictions saved to {output_path}")
        
        # Расчёт метрик
        mse = np.mean((pred_df["actual_price"] - pred_df["predicted_price"]) ** 2)
        mae = np.mean(np.abs(pred_df["actual_price"] - pred_df["predicted_price"]))
        metrics = [{"timestamp": pd.Timestamp.now(), "mse": mse, "mae": mae, "count": len(pred_df)}]
        metrics_path = f"data/offline_results/metrics_{int(pd.Timestamp.now().timestamp())}.csv"
        pd.DataFrame(metrics).to_csv(metrics_path, index=False, encoding='utf-8')
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Создание графиков
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.8, 0.2], vertical_spacing=0.05)
        
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Свечи",
            increasing_line_color="green",
            decreasing_line_color="red"
        ), row=1, col=1)
        
        fig.add_trace(go.Histogram(
            x=df.index,
            y=df["volume"],
            name="Объём",
            opacity=0.3,
            marker_color="grey",
            xbins=dict(size=60*1000)  # 1 минута в миллисекундах
        ), row=2, col=1)
        
        # Линия разделения train/test
        train_end = df.index[train_size]
        fig.add_vline(x=train_end, line_dash="dash", line_color="red", annotation_text="Разделение")
        
        fig.update_layout(
            title="BTC/USDT: Оффлайн данные",
            xaxis_title="Время (MSK)",
            yaxis_title="Цена (USDT)",
            yaxis2_title="Объём",
            showlegend=True,
            height=700,
            template="plotly_dark",
            dragmode="zoom",
            xaxis=dict(
                tickformat="%Y-%m-%d %H:%M:%S",
                tickangle=45
            )
        )
        
        # График предсказаний
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(
            x=pred_df["timestamp"],
            y=pred_df["actual_price"],
            mode="lines",
            name="Фактическая цена",
            line=dict(color=config["visual"]["real_price_color"])
        ))
        pred_fig.add_trace(go.Scatter(
            x=pred_df["timestamp"],
            y=pred_df["predicted_price"],
            mode="lines",
            name="Предсказанная цена",
            line=dict(color=config["visual"]["predicted_price_color"])
        ))
        pred_fig.add_vline(x=train_end, line_dash="dash", line_color="red", annotation_text="Разделение")
        
        pred_fig.update_layout(
            title="BTC/USDT: Предсказания",
            xaxis_title="Время (MSK)",
            yaxis_title="Цена (USDT)",
            showlegend=True,
            height=400,
            template="plotly_dark",
            dragmode="zoom",
            xaxis=dict(
                tickformat="%Y-%m-%d %H:%M:%S",
                tickangle=45
            )
        )
        
        # Сохранение графиков
        fig.write_image(f"data/offline_results/main_graph_{int(pd.Timestamp.now().timestamp())}.png")
        pred_fig.write_image(f"data/offline_results/predictions_graph_{int(pd.Timestamp.now().timestamp())}.png")
        logger.info("Graphs saved to offline_results")
        
        return True
    except Exception as e:
        logger.error(f"Error processing offline file: {str(e)}")
        return False