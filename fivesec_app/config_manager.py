import yaml
import os

def load_config():
    """Загрузка конфигурации из config.yaml"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_environment_config():
    """Загрузка конфигурации окружения из environment.yaml"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", "environment.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config):
    """Сохранение конфигурации в config.yaml"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "config"), exist_ok=True)
    config_path = os.path.join(base_dir, "config", "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, allow_unicode=True)