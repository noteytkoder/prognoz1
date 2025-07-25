import subprocess
import time
import os
from pathlib import Path
from fivesec_app.logger import setup_logger

# Настройка логгера
logger = setup_logger(log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))

# Путь к флагу перезапуска (в корне проекта)
RESTART_FLAG = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fivesec_restart.flag"))

def run():
    """Основной цикл watcher'а для fivesec_app"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        # Удаляем флаг перезапуска, если он существует
        if RESTART_FLAG.exists():
            logger.info("Removing existing fivesec_restart.flag")
            RESTART_FLAG.unlink()

        logger.info("Starting fivesec_app.main")
        print("Starting fivesec_app.main")
        
        # Запускаем main.py как модуль fivesec_app
        process = subprocess.Popen(
            ["python", "-m", "fivesec_app.main"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Ждём завершения процесса и фильтруем вывод
        stdout, stderr = process.communicate()
        logger.debug(f"fivesec_app.main stdout: {stdout}")
        if stderr:
            # Фильтруем сообщения Dash/Flask
            filtered_stderr = "\n".join(
                line for line in stderr.splitlines()
                if not any(keyword in line for keyword in [
                    "dash-component-suites",
                    "GET /",
                    "POST /",
                    "Running on http",
                    "This is a development server"
                ])
            )
            if filtered_stderr.strip():
                logger.error(f"fivesec_app.main stderr: {filtered_stderr}")

        # Проверяем, был ли запрошен перезапуск
        if RESTART_FLAG.exists():
            logger.info("Restart request detected for fivesec_app")
            print("Restart request detected for fivesec_app")
        else:
            logger.info("fivesec_app.main completed without requesting a restart. Ending watcher.")
            print("fivesec_app.main completed without requesting a restart. Ending watcher.")
            break

        # Задержка перед перезапуском
        logger.debug("Waiting 2 seconds before restarting")
        time.sleep(2)

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logger.error(f"Watcher error: {e}", exc_info=True)
        print(f"Watcher error: {e}")