# ЗАПУСКАТЬ ЧЕРЕЗ watcher.py
import asyncio
import uvicorn
from fastapi import FastAPI
from app.data.handler import start_binance_websocket, fetch_historical_data
from app.visual.dashboard import start_dash
from app.logs.logger import setup_logger
import threading
import signal
import sys
from app.config.manager import load_config
from pathlib import Path

logger = setup_logger()
app = FastAPI()
config = load_config()

RESTART_FLAG = Path("restart.flag")

def run_websocket():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_binance_websocket())
        loop.close()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        sys.exit(1)

def run_dash():
    try:
        start_dash()
    except Exception as e:
        logger.error(f"Dash error: {e}")
        sys.exit(1)

def signal_handler(sig, frame):
    logger.info("Shutdown signal received, exiting")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Starting application")

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(fetch_historical_data(config["data"]["download_range"]))
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")

    websocket_thread = threading.Thread(target=run_websocket, daemon=True)
    dash_thread = threading.Thread(target=run_dash, daemon=True)
    websocket_thread.start()
    dash_thread.start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Uvicorn error: {e}")
        sys.exit(1)
