from dash import Dash
from app.logs.logger import setup_logger
from dash_auth import BasicAuth
from app.visual.components import create_layout
from dash_auth import BasicAuth
import secrets
import yaml
with open("app/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
DASH_AUTH_CREDENTIALS = {
    config["auth"]["username"]: config["auth"]["password"]
}


SECRET_KEY = secrets.token_hex(16)  # Генерируем случайный ключ
logger = setup_logger()
dash_app = Dash(__name__, assets_folder="../static")
dash_app.server.secret_key = SECRET_KEY  # Устанавливаем секретный ключ
BasicAuth(dash_app, DASH_AUTH_CREDENTIALS)

dash_app.layout = create_layout()

def start_dash():
    print(dash_app.callback_map.keys())
    """Запуск сервера Dash"""
    try:
        dash_app.run(port=8050, host="0.0.0.0")
        logger.info("Dash server started")
    except Exception as e:
        logger.error(f"Dash server error: {e}")
        raise
