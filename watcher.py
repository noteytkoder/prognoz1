import subprocess
import time
import os
from pathlib import Path

RESTART_FLAG = Path("restart.flag")

def run():
    while True:
        if RESTART_FLAG.exists():
            RESTART_FLAG.unlink()

        print("start main.py")
        process = subprocess.Popen(["python", "main.py"])
        process.wait()

        if RESTART_FLAG.exists():
            print("restart request detected.")
        else:
            print("main.py completed without requesting a restart. Ending watcherwatcher.")
            break

        time.sleep(2)

if __name__ == "__main__":
    run()
