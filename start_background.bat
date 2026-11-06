@echo off
cd /d "%~dp0"
echo Starting AI Proxy using Virtual Environment...

:: Use the python executable specifically from the venv folder
start "EchoText Brain" cmd /k "venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000"
start "EchoText Listener" cmd /k "venv\Scripts\python.exe telegram_listener.py

pause