from dash import Dash
from app.logs.logger import setup_logger
from dash_auth import BasicAuth
from app.visual.components import create_layout
import secrets
import yaml
import os
from flask import Response, request
from flask import send_from_directory
import requests
import pandas as pd
from app.data.handler import buffer_lock

# Загружаем конфиг
with open("app/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DASH_AUTH_CREDENTIALS = {
    config["auth"]["username"]: config["auth"]["password"]
}

SECRET_KEY = secrets.token_hex(16)
logger = setup_logger()

dash_app = Dash(__name__, assets_folder="../../static")
dash_app.server.secret_key = SECRET_KEY
BasicAuth(dash_app, DASH_AUTH_CREDENTIALS)

dash_app.layout = create_layout()

def get_file_reversed(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return ''.join(reversed(lines))


@dash_app.server.route('/logs/predictions.log', methods=['GET'])
def serve_predictions_log_text():
    """Возвращает содержимое файла predictions.log в виде текста"""
    try:
        log_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'predictions.log')
        )
        logger.info(f"Serving predictions log file: {log_file_path}")

        if not os.path.exists(log_file_path):
            logger.error(f"Predictions log file not found at {log_file_path}")
            return Response("Лог предсказаний отсутствует", status=404, mimetype='text/plain')

        # with open(log_file_path, 'r', encoding='utf-8') as f:
        #     log_content = f.read()
        log_content = get_file_reversed(log_file_path)


        return Response(log_content, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error serving predictions.log: {e}")
        return Response(f"Ошибка: {str(e)}", status=500, mimetype='text/plain')
    
@dash_app.server.route('/logs/predictions.csv', methods=['GET'])
def serve_predictions_csv():
    """Возвращает содержимое файла predictions.csv в виде текста"""
    try:
        csv_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'predictions.csv')
        )
        logger.info(f"Serving predictions CSV file: {csv_file_path}")

        if not os.path.exists(csv_file_path):
            logger.error(f"Predictions CSV file not found at {csv_file_path}")
            return Response("Файл предсказаний отсутствует", status=404, mimetype='text/plain')

        # with open(csv_file_path, 'r', encoding='utf-8') as f:
        #     csv_content = f.read()
        csv_content = get_file_reversed(csv_file_path)


        return Response(csv_content, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error serving predictions.csv: {e}")
        return Response(f"Ошибка: {str(e)}", status=500, mimetype='text/plain')


@dash_app.server.route('/logs/logtotal', methods=['GET'])
def serve_logtotal():
    try:
        log_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'app.log')
        )
        # logger.info(f"Serving full log file: {log_file_path}")

        if not os.path.exists(log_file_path):
            logger.error(f"Log file not found at {log_file_path}")
            return Response("Лог отсутствует", status=404, mimetype='text/plain')

        # with open(log_file_path, 'r', encoding='utf-8') as f:
        #     log_content = f.read()
        log_content = get_file_reversed(log_file_path)


        return Response(log_content, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error serving logtotal: {e}")
        return Response(f"Ошибка: {str(e)}", status=500, mimetype='text/plain')

@dash_app.server.route('/logs/predictions.table', methods=['GET'])
def serve_predictions_table():
    """Возвращает HTML-таблицу с последней записью из predictions.csv с расчетом времени прогнозов"""
    try:
        csv_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'predictions.csv')
        )
        logger.info(f"Serving predictions CSV file: {csv_file_path}")

        if not os.path.exists(csv_file_path):
            logger.error(f"Predictions file not found at {csv_file_path}")
            return Response("Логи отсутствуют", status=404, mimetype='text/plain')

        with buffer_lock:
            if os.path.getsize(csv_file_path) == 0:
                logger.error(f"Predictions file is empty at {csv_file_path}")
                return Response("Логи отсутствуют (файл пуст)", status=404, mimetype='text/plain')

            pred_df = pd.read_csv(csv_file_path, encoding='utf-8')
            if pred_df.empty:
                logger.error(f"Predictions file is empty after reading at {csv_file_path}")
                return Response("Логи отсутствуют", status=404, mimetype='text/plain')

        last_pred = pred_df.iloc[-1]

        # Парсим timestamp
        timestamp_raw = pd.to_datetime(last_pred['timestamp'])
        timestamp = timestamp_raw.strftime('%Y-%m-%d %H:%M:%S')

        # Считаем прогнозные времена
        min_pred_timestamp = (timestamp_raw + pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
        hour_pred_timestamp = (timestamp_raw + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

        # Форматируем значения цен
        actual_price = round(float(last_pred['actual_price']), 4)
        min_pred = round(float(last_pred['min_pred']), 4)
        hour_pred = round(float(last_pred['hour_pred']), 4)

        # Вычисляем проценты
        if actual_price > 0:
            min_change = ((min_pred - actual_price) / actual_price) * 100
            hour_change = ((hour_pred - actual_price) / actual_price) * 100
            min_change_str = f"{min_change:+.2f}%"
            hour_change_str = f"{hour_change:+.2f}%"
        else:
            min_change_str = hour_change_str = "N/A"

        # Формируем строку таблицы
        table_rows = f"""
            <tr>
                <td>{timestamp}</td>
                <td>{actual_price:.4f}</td>
                <td>{min_pred:.4f} ({min_change_str}) <br><small>{min_pred_timestamp}</small></td>
                <td>{hour_pred:.4f} ({hour_change_str}) <br><small>{hour_pred_timestamp}</small></td>
            </tr>
        """



        # Загрузка шаблона
        TEMPLATE_PATH = os.path.join(
            os.path.dirname(__file__), '..', '..', 'app', 'templates', 'logs_template.html'
        )
        with open(TEMPLATE_PATH, encoding='utf-8') as f:
            template = f.read()

        html_content = template.replace('{{TABLE_ROWS}}', table_rows)

        return Response(html_content, mimetype='text/html')

    except pd.errors.EmptyDataError:
        logger.error(f"Predictions file is empty or corrupted at {csv_file_path}")
        return Response("Логи отсутствуют (файл пуст или поврежден)", status=404, mimetype='text/plain')
    except Exception as e:
        logger.error(f"Error serving predictions.csv: {e}", exc_info=True)
        return Response(f"Ошибка: {str(e)}", status=500, mimetype='text/plain')



def start_dash():
    """Запуск сервера Dash"""
    try:
        dash_app.run(port=8050, host="0.0.0.0")
        logger.info("Dash server started")
    except Exception as e:
        logger.error(f"Dash server error: {e}")
        raise
