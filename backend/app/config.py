"""
Central config, read from environment variables (with sane local defaults).
Keeping this in one file means swapping the model backend, adapter path, or
enabling an integration is a one-line env change — never a code change.
"""
import os

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "template")  # "template" | "phi_lora"
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./adapter")
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/Phi-3.5-mini-instruct")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
GMAIL_CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", "./credentials.json")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
