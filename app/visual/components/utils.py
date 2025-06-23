import pandas as pd
import plotly.graph_objects as go
from app.data.handler import data_buffer, get_latest_features, buffer_lock
from app.model.train import predict, predict_hourly
from app.config.manager import load_config, save_config
from app.logs.logger import setup_logger
import numpy as np
import time

logger = setup_logger()
config = load_config()
last_stats_time = 0
last_mse = 0
last_mae = 0
current_layout = None