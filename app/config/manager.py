import os
import yaml

CONFIG_PATH = "app/config/config.yaml"
ENVIRONMENT_PATH = "app/config/environment.yaml"


def load_config():
    """Чтение основного системного конфига"""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config_data):
    """Сохранение системного конфига"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)


def load_environment_config():
    """
    Чтение конфигурации окружения (prod или dev)
    """
    with open(ENVIRONMENT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)