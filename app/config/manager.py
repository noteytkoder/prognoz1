import yaml

def load_config():
    """Чтение конфигурации"""
    with open("app/config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config_data):
    """Сохранение конфигурации"""
    with open("app/config/config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)