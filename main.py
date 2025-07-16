# main.py
import asyncio
import uvicorn
from fastapi import FastAPI
from app.data.handler import start_binance_websocket, fetch_historical_data
from app.visual.dashboard import start_dash
from app.logs.logger import setup_logger
import threading
import signal
import sys
from pathlib import Path
from app.config.manager import load_config, load_environment_config


logger = setup_logger()
app = FastAPI()
config = load_config()
env_name = config["app_env"]
env_config = load_environment_config()


RESTART_FLAG = Path("restart.flag")


def run_websocket():
    """
    Запускаем WebSocket в отдельном потоке с отдельным event loop.
    """
    try:
        logger.info("Starting Binance WebSocket thread with its own event loop")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_binance_websocket())
    except Exception as e:
        logger.error(f"WebSocket thread error: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)


def run_dash():
    """
    Запуск Dash сервера. Тут нет asyncio, просто обычный запуск.
    """
    try:
        logger.info("Starting Dash server")
        start_dash()
    except Exception as e:
        logger.error(f"Dash thread error: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)


def signal_handler(sig, frame):
    logger.info("Shutdown signal received, exiting")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting application")

    # Загрузка исторических данных до старта потоков
    try:
        logger.info("Fetching historical data before starting services")
        asyncio.run(fetch_historical_data(config["data"]["download_range"]))
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}", exc_info=True)

    # Потоки для WebSocket и Dash
    websocket_thread = threading.Thread(target=run_websocket, daemon=True, name="WebSocketThread")
    dash_thread = threading.Thread(target=run_dash, daemon=True, name="DashThread")
    websocket_thread.start()
    dash_thread.start()

    # Uvicorn (FastAPI) на главном потоке
    try:
        logger.info("Starting FastAPI server with Uvicorn")
        uvicorn.run(app, host="0.0.0.0", port=env_config[env_name]["port_unicorn"], access_log=False)
    except Exception as e:
        logger.error(f"Uvicorn error: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)


if __name__ == "__main__":
    main()
