#!/bin/bash

echo "Checking for Python..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python3 is not installed. Install Python 3.9 or higher."
    exit 1
fi

pyver=$(python3 --version | awk '{print $2}')
py_major=$(echo "$pyver" | cut -d. -f1)
py_minor=$(echo "$pyver" | cut -d. -f2)

if [ "$py_major" -lt 3 ]; then
    echo "[ERROR] Python 3.9 or higher is required. Found: $pyver"
    exit 1
fi

if [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 9 ]; then
    echo "[ERROR] Python 3.9 or higher is required. Found: $pyver"
    exit 1
fi

echo "Python found: version $pyver"
echo

echo "Updating pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to update pip, setuptools, or wheel."
    exit 1
fi

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies. Check requirements.txt or internet connection."
    exit 1
fi

echo "Dependencies installed successfully."
echo "Freezing dependency versions for stability..."
pip freeze > requirements_freeze.txt
echo "Dependency versions saved to requirements_freeze.txt."
echo

echo "Launching application..."
python3 watcher.py &
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to launch application. Check logs/app.log for details."
    exit 1
fi

echo "Waiting for server to start..."
sleep 5

echo "Application launched. Check http://95.81.114.29:8050 in your browser."
echo "Logs saved to logs/app.log. If errors occur, check logs/app.log and requirements_freeze.txt."
exit 0