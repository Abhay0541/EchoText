@echo off
echo Starting AI Proxy in the background...
start /b pythonw -m uvicorn server:app --host 127.0.0.1 --port 8000
start /b pythonw telegram_listener.py
exit