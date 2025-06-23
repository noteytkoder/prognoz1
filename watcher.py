import subprocess
import time
import os
from pathlib import Path

RESTART_FLAG = Path("restart.flag")

def run():
    while True:
        if RESTART_FLAG.exists():
            RESTART_FLAG.unlink()

        print("Запускаем main.py")
        process = subprocess.Popen(["python", "main.py"])
        process.wait()

        if RESTART_FLAG.exists():
            print("Обнаружен запрос на перезапуск.")
        else:
            print("main.py завершён без запроса перезапуска. Завершаем watcher.")
            break

        time.sleep(2)

if __name__ == "__main__":
    run()
