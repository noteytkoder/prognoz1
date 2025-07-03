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



@dash_app.server.route('/logs/predictions_raw', methods=['GET'])
def serve_predictions_raw():
    """Возвращает содержимое файла predictions.log в виде текста"""
    try:
        log_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'predictions.log')
        )
        logger.info(f"Serving predictions log file: {log_file_path}")

        if not os.path.exists(log_file_path):
            logger.error(f"Predictions log file not found at {log_file_path}")
            return Response("Лог предсказаний отсутствует", status=404, mimetype='text/plain')

        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        return Response(log_content, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error serving predictions.log: {e}")
        return Response(f"Ошибка: {str(e)}", status=500, mimetype='text/plain')

@dash_app.server.route('/logs/predictions', methods=['GET'])
def serve_predictions_log():
    try:
        # Чтение параметра refresh из query-параметров
        user_refresh = request.args.get('refresh')
        if user_refresh and user_refresh.isdigit():
            refresh_interval = int(user_refresh)
        else:
            refresh_interval = config["visual"]["log_refresh_interval"]

        log_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'predictions.log')
        )

        if not os.path.exists(log_file_path):
            return Response("Логи отсутствуют", status=404, mimetype='text/plain')

        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.readlines()

        log_content = log_content[-1:]

        # Получение текущей цены с Binance
        try:
            binance_data = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5
            )
            binance_price = round(float(binance_data.json().get("price", 0)), 2)
        except Exception as e:
            logger.error(f"Ошибка запроса к Binance: {e}")
            binance_price = 0

        # Построение таблицы
        table_rows = ""
        for line in log_content:
            try:
                parts = line.strip().split(", ")
                if len(parts) != 3:
                    continue
                timestamp = parts[0]
                min_pred = float(parts[1].split("=")[1])
                hour_pred = float(parts[2].split("=")[1])

                # Проценты
                if binance_price > 0:
                    min_change = ((min_pred - binance_price) / binance_price) * 100
                    hour_change = ((hour_pred - binance_price) / binance_price) * 100
                    min_change_str = f"{min_change:+.2f}%"
                    hour_change_str = f"{hour_change:+.2f}%"
                else:
                    min_change_str = hour_change_str = "N/A"

                table_rows += f"""
                  <tr>
                    <td>{timestamp}</td>
                    <td>{binance_price:.2f}</td>
                    <td>{min_pred:.2f} ({min_change_str})</td>
                    <td>{hour_pred:.2f} ({hour_change_str})</td>
                  </tr>
                """
            except Exception as e:
                logger.warning(f"Failed to parse log line: {line.strip()}, error: {e}")

        # Загрузка шаблона
        TEMPLATE_PATH = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'app', 'templates', 'logs_template.html'
        )
        with open(TEMPLATE_PATH, encoding='utf-8') as f:
            template = f.read()

        html_content = template \
            .replace('{{TABLE_ROWS}}', table_rows) \
            .replace('{{REFRESH_INTERVAL}}', str(config["visual"]["log_refresh_interval"])) \
            .replace('{{REFRESH_INTERVAL_SEC}}', str(int(config["visual"]["log_refresh_interval"] // 1000))) \
            .replace('{{CURRENT_INTERVAL_SEC}}', str(int(config["visual"]["log_refresh_interval"] // 1000)))


        return Response(html_content, mimetype='text/html')

    except Exception as e:
        logger.error(f"Error serving predictions.log: {e}")
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

        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        return Response(log_content, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error serving logtotal: {e}")
        return Response(f"Ошибка: {str(e)}", status=500, mimetype='text/plain')


def start_dash():
    """Запуск сервера Dash"""
    try:
        dash_app.run(port=8050, host="0.0.0.0")
        logger.info("Dash server started")
    except Exception as e:
        logger.error(f"Dash server error: {e}")
        raise
