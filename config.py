# config.py
# One place for every path and setting, so server, listener and dashboard
# always point at the same files no matter which folder you start them from.
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# You can override these with environment variables if you ever need to
MODEL_PATH = Path(os.getenv("ECHOTEXT_MODEL_PATH", BASE_DIR / "Llama-3.2-3B-Instruct.Q4_K_M.gguf"))
DB_PATH = Path(os.getenv("ECHOTEXT_DB", BASE_DIR / "proxy_memory.db"))

STATUS_FILE = BASE_DIR / "proxy_status.txt"   # contains BUSY or AVAILABLE
RAG_FOLDER = BASE_DIR / "rag_memory"          # notes / facts uploaded from the dashboard
SESSION_PATH = BASE_DIR / "my_proxy"          # Telethon adds .session itself

BRAIN_URL = os.getenv("ECHOTEXT_BRAIN_URL", "http://127.0.0.1:8000/chat")