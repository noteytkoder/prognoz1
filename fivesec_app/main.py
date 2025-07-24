import asyncio
import threading
import signal
import sys
import os
from pathlib import Path
from fivesec_app.data_handler import start_binance_websocket, fetch_fivesec_historical_data
from fivesec_app.fivesec_dashboard import start_fivesec_dash
from fivesec_app.logger import setup_logger
from fivesec_app.config_manager import load_config, load_environment_config

# Установить текущую рабочую директорию в папку fivesec_app
os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = load_config()
env_config = load_environment_config()
logger = setup_logger()
RESTART_FLAG = Path("restart.flag")

# Остальной код без изменений
def signal_handler(sig, frame):
    """Обработчик сигналов завершения"""
    logger.info(f"Received signal {sig}, shutting down")
    RESTART_FLAG.touch()
    sys.exit(0)

def run_websocket():
    """Запуск WebSocket в отдельном потоке"""
    try:
        logger.info("Starting WebSocket")
        asyncio.run(start_binance_websocket())
    except Exception as e:
        logger.error(f"WebSocket thread error: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)

def run_fivesec_dash():
    """Запуск Dash сервера"""
    try:
        logger.info("Starting 5-second Dash server")
        start_fivesec_dash()
    except Exception as e:
        logger.error(f"5-second Dash thread error: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting 5-second application")

    # Загрузка исторических данных
    try:
        logger.info("Fetching historical data")
        asyncio.run(fetch_fivesec_historical_data())
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}", exc_info=True)

    # Запуск потоков
    websocket_thread = threading.Thread(target=run_websocket, daemon=True, name="WebSocketThread")
    dash_thread = threading.Thread(target=run_fivesec_dash, daemon=True, name="DashThread")
    websocket_thread.start()
    dash_thread.start()

    # Главный поток ожидает завершения
    try:
        websocket_thread.join()
        dash_thread.join()
    except KeyboardInterrupt:
        logger.info("Main thread received KeyboardInterrupt")
        RESTART_FLAG.touch()
        sys.exit(0)

if __name__ == "__main__":
    main()