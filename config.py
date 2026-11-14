"""Centralized configuration module for EchoText.

All filesystem paths are resolved as absolute ``pathlib.Path`` objects anchored to
``BASE_DIR`` so starting the server, listener, or Streamlit dashboard from any
working directory behaves identically. Includes hardware profiles for 3B (phone/CPU)
and 8B (PC), mandatory local API key provisioning, environment validation, and
rotating log setup.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import secrets
from typing import Any, Literal

try:
    from dotenv import load_dotenv as _dotenv_load
except ImportError:  # pragma: no cover
    _dotenv_load = None  # type: ignore[assignment]


def _load_env_fallback(env_path: Path) -> None:
    """Fallback .env parser if python-dotenv is not installed in current environment."""
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


BASE_DIR: Path = Path(__file__).resolve().parent
if _dotenv_load is not None:
    _dotenv_load(BASE_DIR / ".env")
else:
    _load_env_fallback(BASE_DIR / ".env")

HardwareProfile = Literal["pc_8b", "pc_3b", "phone_3b"]
PlatformType = Literal["telegram", "whatsapp", "cli", "test"]
ContactMode = Literal["off", "shadow", "draft", "auto"]
RelationshipTier = Literal[
    "friend", "family", "colleague", "professor", "acquaintance", "unknown"
]
ToneStyle = Literal[
    "casual_hinglish",
    "polite_hinglish",
    "formal_english",
    "warm_family",
    "concise",
]
DraftStatus = Literal[
    "pending_approval",
    "pending_send",
    "approved",
    "sending",
    "edited",
    "sent",
    "failed",
    "rejected",
    "shadow",
    "cancelled",
    "deleted_sent",
]

VALID_PLATFORMS: tuple[PlatformType, ...] = ("telegram", "whatsapp", "cli", "test")
VALID_MODES: tuple[ContactMode, ...] = ("off", "shadow", "draft", "auto")
VALID_TIERS: tuple[RelationshipTier, ...] = (
    "friend",
    "family",
    "colleague",
    "professor",
    "acquaintance",
    "unknown",
)
VALID_TONES: tuple[ToneStyle, ...] = (
    "casual_hinglish",
    "polite_hinglish",
    "formal_english",
    "warm_family",
    "concise",
)
VALID_DRAFT_STATUSES: tuple[DraftStatus, ...] = (
    "pending_approval",
    "pending_send",
    "approved",
    "sending",
    "edited",
    "sent",
    "failed",
    "rejected",
    "shadow",
    "cancelled",
    "deleted_sent",
)

INSECURE_DEFAULT_PASSWORDS: frozenset[str] = frozenset(
    {
        "change_me_before_first_run",
        "password",
        "admin",
        "123456",
        "12345678",
        "echotext",
    }
)

# Hardware profiles for 8B on PC vs 3B on PC vs 3B on Phone (Termux / Android)
MODEL_PROFILES: dict[HardwareProfile, dict[str, Any]] = {
    "pc_8b": {
        "model_filename": "echotext-qwen3-8b-Q4_K_M.gguf",
        "n_ctx": 4096,
        "n_threads": max(4, (os.cpu_count() or 4) - 1),
        "n_batch": 512,
        "n_gpu_layers": int(os.getenv("ECHOTEXT_GPU_LAYERS", "0")),
        "max_new_tokens": 128,
        "description": "PC 8B profile (Qwen 3 8B / Llama 8B, high quality reasoning & Hinglish nuance)",
    },
    "pc_3b": {
        "model_filename": "Llama-3.2-3B-Instruct.Q8_0.gguf",
        "n_ctx": 2048,
        "n_threads": max(2, (os.cpu_count() or 4) - 1),
        "n_batch": 256,
        "n_gpu_layers": int(os.getenv("ECHOTEXT_GPU_LAYERS", "0")),
        "max_new_tokens": 96,
        "description": "PC 3B profile (fast CPU-only inference on i3/i5 laptops)",
    },
    "phone_3b": {
        "model_filename": "Llama-3.2-3B-Instruct.Q4_K_M.gguf",
        "n_ctx": 2048,
        "n_threads": 4,
        "n_batch": 128,
        "n_gpu_layers": 0,
        "max_new_tokens": 64,
        "description": "Phone 3B profile (low-memory arm64 / Termux configuration)",
    },
}


def _resolve_path(env_key: str, default_relative: str) -> Path:
    raw = os.getenv(env_key, "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (BASE_DIR / p).resolve()
    return (BASE_DIR / default_relative).resolve()


def _parse_int_or_float(env_key: str, default: int) -> int:
    """Parse an integer environment variable, tolerating float strings gracefully."""
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default
    try:
        return int(round(float(raw)))
    except ValueError:
        return default


_raw_profile = os.getenv("ECHOTEXT_HARDWARE_PROFILE", "pc_3b").strip().lower()
HARDWARE_PROFILE: HardwareProfile = (
    _raw_profile  # type: ignore[assignment]
    if _raw_profile in MODEL_PROFILES
    else "pc_3b"
)
ACTIVE_PROFILE: dict[str, Any] = MODEL_PROFILES[HARDWARE_PROFILE]

# Absolute filesystem paths (never depend on current working directory)
MODELS_DIR: Path = _resolve_path("ECHOTEXT_MODELS_DIR", "models")
MODEL_PATH: Path = _resolve_path(
    "ECHOTEXT_MODEL_PATH", str(MODELS_DIR / str(ACTIVE_PROFILE["model_filename"]))
)
if not MODEL_PATH.exists():
    if (BASE_DIR / str(ACTIVE_PROFILE["model_filename"])).exists():
        MODEL_PATH = (BASE_DIR / str(ACTIVE_PROFILE["model_filename"])).resolve()
    elif (MODELS_DIR / "echotext-qwen3-8b-Q4_K_M.gguf").exists():
        MODEL_PATH = (MODELS_DIR / "echotext-qwen3-8b-Q4_K_M.gguf").resolve()
    else:
        # Fallback to any available .gguf file
        existing_ggufs = list(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.exists() else []
        if existing_ggufs:
            MODEL_PATH = existing_ggufs[0].resolve()


def get_available_models(models_dir: Path | None = None) -> list[dict[str, Any]]:
    """Scan models directory for available .gguf files and return detailed metadata."""
    target_dir = MODELS_DIR if models_dir is None else models_dir
    found: list[dict[str, Any]] = []
    if target_dir.exists():
        for f in target_dir.glob("*.gguf"):
            try:
                size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                size_gb = round(size_mb / 1024, 2)
            except OSError:
                size_mb, size_gb = 0.0, 0.0
            fname = f.name
            fname_lower = fname.lower()
            if "qwen3" in fname_lower:
                model_type = "Qwen 3 8B (Fine-tuned)"
            elif "qwen" in fname_lower:
                model_type = "Qwen 2.5 (Fine-tuned)"
            elif "llama-3.2-3b" in fname_lower:
                model_type = "Llama 3.2 3B Instruct"
            elif "llama" in fname_lower:
                model_type = "Llama"
            else:
                model_type = "GGUF Model"
            label = f"{fname} ({size_gb:.1f} GB — {model_type})"
            found.append({
                "filename": fname,
                "path": str(f.resolve()),
                "size_mb": size_mb,
                "size_gb": size_gb,
                "model_type": model_type,
                "label": label,
                "is_qwen": "qwen" in fname_lower,
            })
    # Sort: Qwen models first, then by size descending
    found.sort(key=lambda x: (not x["is_qwen"], -x["size_mb"]))
    return found

DB_PATH: Path = _resolve_path("ECHOTEXT_DB_PATH", "proxy_memory.db")
STATUS_FILE_PATH: Path = _resolve_path("ECHOTEXT_STATUS_FILE", "proxy_status.txt")
RAG_DIR: Path = _resolve_path("ECHOTEXT_RAG_DIR", "rag_memory")
LOG_DIR: Path = _resolve_path("ECHOTEXT_LOG_DIR", "logs")
LOG_FILE_PATH: Path = LOG_DIR / "echotext.log"
PID_FILE_PATH: Path = LOG_DIR / "echotext_pids.json"
API_KEY_FILE_PATH: Path = LOG_DIR / ".echotext_api_key"
SESSION_PATH: Path = _resolve_path("ECHOTEXT_SESSION_PATH", "my_proxy.session")
WHISPER_MODEL_PATH: Path = _resolve_path("ECHOTEXT_WHISPER_MODEL_PATH", "models/ggml-base.bin")
WHISPER_BINARY_PATH: str = os.getenv("ECHOTEXT_WHISPER_BIN", "whisper-cli")


def get_or_create_api_key(key_file: Path | None = None) -> str:
    """Return the configured ``ECHOTEXT_API_KEY`` or provision a persistent local secret key."""
    env_key = os.getenv("ECHOTEXT_API_KEY", "").strip()
    if env_key:
        return env_key

    target = API_KEY_FILE_PATH if key_file is None else key_file
    try:
        if target.exists():
            stored = target.read_text(encoding="utf-8").strip()
            if len(stored) >= 16:
                return stored
        target.parent.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_hex(32)
        tmp_path = target.with_suffix(f".tmp.{os.getpid()}")
        tmp_path.write_text(generated + "\n", encoding="utf-8")
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(tmp_path, target)
        return generated
    except OSError:
        return secrets.token_hex(32)


# Network & server settings
SERVER_HOST: str = os.getenv("ECHOTEXT_SERVER_HOST", "127.0.0.1")
SERVER_PORT: int = _parse_int_or_float("ECHOTEXT_SERVER_PORT", 8000)
SERVER_URL: str = os.getenv("ECHOTEXT_SERVER_URL", f"http://{SERVER_HOST}:{SERVER_PORT}")
DASHBOARD_HOST: str = "127.0.0.1"
DASHBOARD_PORT: int = _parse_int_or_float("ECHOTEXT_DASHBOARD_PORT", 8501)
API_SECRET_KEY: str = get_or_create_api_key()
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "127.0.0.1",
        f"127.0.0.1:{SERVER_PORT}",
        "localhost",
        f"localhost:{SERVER_PORT}",
        "[::1]",
        f"[::1]:{SERVER_PORT}",
        "testserver",
        SERVER_HOST.lower(),
        f"{SERVER_HOST.lower()}:{SERVER_PORT}",
    }
)

# Token, batching & multi-part splitting budgets
CONTEXT_WINDOW: int = _parse_int_or_float(
    "ECHOTEXT_CONTEXT_WINDOW", int(ACTIVE_PROFILE["n_ctx"])
)
MAX_NEW_TOKENS: int = _parse_int_or_float(
    "ECHOTEXT_MAX_NEW_TOKENS", int(ACTIVE_PROFILE["max_new_tokens"])
)
TOKEN_SAFETY_MARGIN: int = _parse_int_or_float("ECHOTEXT_TOKEN_SAFETY_MARGIN", 120)
MAX_USER_INPUT_CHARS: int = _parse_int_or_float("ECHOTEXT_MAX_USER_INPUT_CHARS", 1500)
DEFAULT_HISTORY_LIMIT: int = _parse_int_or_float("ECHOTEXT_HISTORY_LIMIT", 6)
NUM_DRAFT_OPTIONS: int = _parse_int_or_float("ECHOTEXT_NUM_DRAFT_OPTIONS", 3)
MAX_SINGLE_REPLY_CHARS: int = _parse_int_or_float("ECHOTEXT_MAX_SINGLE_REPLY_CHARS", 140)
PARTIAL_BATCH_THRESHOLD: int = _parse_int_or_float("ECHOTEXT_PARTIAL_BATCH_THRESHOLD", 4)

# Safety, rate limits & autonomy defaults
OWNER_NAME: str = os.getenv("ECHOTEXT_OWNER_NAME", "Abhay").strip() or "Abhay"
TIMEZONE_NAME: str = os.getenv("ECHOTEXT_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"
DEFAULT_FOOTER_TEXT: str = os.getenv(
    "ECHOTEXT_FOOTER_TEXT", f"— sent by {OWNER_NAME}'s assistant"
)
SENSITIVITY_THRESHOLD: int = _parse_int_or_float("ECHOTEXT_SENSITIVITY_THRESHOLD", 2)
MAX_AUTO_REPLIES_PER_HOUR: int = _parse_int_or_float("ECHOTEXT_MAX_REPLIES_PER_HOUR", 5)
COOLDOWN_SECONDS: int = _parse_int_or_float("ECHOTEXT_COOLDOWN_SECONDS", 60)
KILL_SWITCH_PAUSE_SECONDS: int = (
    _parse_int_or_float("ECHOTEXT_KILL_SWITCH_SECONDS", 0)
    or int(round(float(os.getenv("ECHOTEXT_KILL_SWITCH_HOURS", "2.0")) * 3600))
)
KILL_SWITCH_PAUSE_HOURS: float = KILL_SWITCH_PAUSE_SECONDS / 3600.0
AUTO_SEND_DELAY_SECONDS: int = _parse_int_or_float("ECHOTEXT_AUTO_SEND_DELAY_SECONDS", 10)
GRADUATED_AUTONOMY_THRESHOLD: int = _parse_int_or_float(
    "ECHOTEXT_GRADUATED_AUTONOMY_COUNT", 10
)
MAX_SEND_ATTEMPTS: int = _parse_int_or_float("ECHOTEXT_MAX_SEND_ATTEMPTS", 3)

# Active hours & privacy retention (message text redacted by default for privacy)
ACTIVE_HOURS_ENABLED: bool = (
    os.getenv("ECHOTEXT_ACTIVE_HOURS_ENABLED", "false").strip().lower() == "true"
)
ACTIVE_HOURS_START: int = _parse_int_or_float("ECHOTEXT_ACTIVE_HOURS_START", 8)
ACTIVE_HOURS_END: int = _parse_int_or_float("ECHOTEXT_ACTIVE_HOURS_END", 23)
RETENTION_DAYS: int = _parse_int_or_float("ECHOTEXT_RETENTION_DAYS", 30)
HIDE_MESSAGE_TEXT: bool = (
    os.getenv("ECHOTEXT_HIDE_MESSAGE_TEXT", "true").strip().lower() != "false"
)

# Upload hardening limits
ALLOWED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset({".txt", ".pdf"})
MAX_UPLOAD_BYTES: int = _parse_int_or_float(
    "ECHOTEXT_MAX_UPLOAD_BYTES", 5 * 1024 * 1024
)

# Dashboard authentication settings
DASHBOARD_PASSWORD: str = os.getenv("ECHOTEXT_DASHBOARD_PASSWORD", "").strip()
DASHBOARD_PASSWORD_HASH: str = os.getenv("ECHOTEXT_DASHBOARD_PASSWORD_HASH", "").strip()
DASHBOARD_LOCKOUT_ATTEMPTS: int = _parse_int_or_float("ECHOTEXT_LOCKOUT_ATTEMPTS", 5)
DASHBOARD_LOCKOUT_SECONDS: int = _parse_int_or_float("ECHOTEXT_LOCKOUT_SECONDS", 300)
DASHBOARD_SESSION_TIMEOUT_SECONDS: int = _parse_int_or_float(
    "ECHOTEXT_SESSION_TIMEOUT_SECONDS", 1800
)

# Cloud-sync folder warning names
CLOUD_SYNC_MARKERS: tuple[str, ...] = (
    "onedrive",
    "dropbox",
    "google drive",
    "googledrive",
    "icloud",
)

_MD_SPECIAL_RE = re.compile(r"([_*`\[\]()~>#+=|{}.!\\-])")


def escape_markdown(text: str) -> str:
    """Escape Markdown special characters in untrusted user/contact strings."""
    if not text:
        return ""
    return _MD_SPECIAL_RE.sub(r"\\\1", str(text))


def is_path_in_cloud_sync(path: Path) -> bool:
    """Return True if ``path`` appears to reside inside a cloud-synced folder."""
    lowered = str(path.resolve()).lower()
    return any(marker in lowered for marker in CLOUD_SYNC_MARKERS)


def validate_env(
    require_telegram: bool = False,
    require_dashboard_auth: bool = False,
    require_phone_bot: bool = False,
    env_override: dict[str, str] | None = None,
) -> list[str]:
    """Validate required environment variables and return a list of clear error strings."""
    env = os.environ if env_override is None else env_override
    errors: list[str] = []

    if require_telegram:
        api_id_raw = env.get("TELEGRAM_API_ID", "").strip()
        api_hash = env.get("TELEGRAM_API_HASH", "").strip()
        if not api_id_raw:
            errors.append(
                "Missing TELEGRAM_API_ID in .env (get one from https://my.telegram.org)."
            )
        elif not api_id_raw.isdigit():
            errors.append(
                f"Invalid TELEGRAM_API_ID='{api_id_raw}' in .env: must be an integer."
            )
        if not api_hash or api_hash == "your_telegram_api_hash_here":
            errors.append(
                "Missing TELEGRAM_API_HASH in .env (get one from https://my.telegram.org)."
            )

    if require_dashboard_auth:
        pwd = env.get("ECHOTEXT_DASHBOARD_PASSWORD", "").strip()
        pwd_hash = env.get("ECHOTEXT_DASHBOARD_PASSWORD_HASH", "").strip()
        if not pwd and not pwd_hash:
            errors.append(
                "Missing ECHOTEXT_DASHBOARD_PASSWORD or ECHOTEXT_DASHBOARD_PASSWORD_HASH in .env."
            )
        elif pwd.lower() in INSECURE_DEFAULT_PASSWORDS:
            errors.append(
                "Insecure default ECHOTEXT_DASHBOARD_PASSWORD detected. Choose a strong password."
            )

    if require_phone_bot:
        bot_token = env.get("ECHOTEXT_APPROVAL_BOT_TOKEN", "").strip()
        owner_chat_id = env.get("ECHOTEXT_OWNER_TELEGRAM_ID", "").strip()
        if not bot_token:
            errors.append(
                "Missing ECHOTEXT_APPROVAL_BOT_TOKEN in .env for the private phone approval bot."
            )
        if not owner_chat_id or not owner_chat_id.lstrip("-").isdigit():
            errors.append(
                "Missing or non-numeric ECHOTEXT_OWNER_TELEGRAM_ID in .env for phone bot."
            )

    if is_path_in_cloud_sync(SESSION_PATH):
        errors.append(
            f"WARNING: SESSION_PATH ({SESSION_PATH}) is inside a cloud-synced folder. "
            "Move my_proxy.session outside OneDrive/Dropbox/Google Drive/iCloud."
        )

    return errors


def redact_text(text: str, hide: bool | None = None) -> str:
    """Mask message text in logs/console when ``HIDE_MESSAGE_TEXT`` is enabled."""
    should_hide = HIDE_MESSAGE_TEXT if hide is None else hide
    if not should_hide:
        return text
    return f"[REDACTED len={len(text)} chars]"


def setup_logging(name: str = "echotext") -> logging.Logger:
    """Configure a RotatingFileHandler + console logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def get_proxy_status(status_file: Path | None = None) -> Literal["BUSY", "AVAILABLE"]:
    """Fail-closed status gate: returns ``'BUSY'`` ONLY when the file explicitly reads ``BUSY``."""
    target = STATUS_FILE_PATH if status_file is None else status_file
    try:
        if not target.exists():
            return "AVAILABLE"
        content = target.read_text(encoding="utf-8").strip().upper()
        return "BUSY" if content == "BUSY" else "AVAILABLE"
    except OSError:
        return "AVAILABLE"


def set_proxy_status(
    status: Literal["BUSY", "AVAILABLE"], status_file: Path | None = None
) -> Literal["BUSY", "AVAILABLE"]:
    """Atomically write ``BUSY`` or ``AVAILABLE`` using a temp file + ``os.replace``."""
    normalized: Literal["BUSY", "AVAILABLE"] = (
        "BUSY" if str(status).strip().upper() == "BUSY" else "AVAILABLE"
    )
    target = STATUS_FILE_PATH if status_file is None else status_file
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f".tmp.{os.getpid()}")
    tmp_path.write_text(normalized + "\n", encoding="utf-8")
    os.replace(tmp_path, target)
    return normalized
