"""Unified SQLite data access layer for EchoText.

All database interactions go through this module. Features:
- Automatic schema creation and non-destructive column migrations on startup (``init_db``)
- Dynamic SQL CHECK constraints generated from ``config.py`` tuples (single source of truth)
- Immutable ``platform_user_id`` binding so reassigned Telegram usernames never inherit
  a trusted contact's autonomy mode, tier, chat history, or scoped notes
- Per-contact ambiguity freeze (``flagged_for_review``, ``flag_reason``) that halts
  auto-replies for a contact until the owner reviews, edits/sends, or deletes the draft
- Multi-part reply storage (``reply_parts_json``, ``sent_msg_ids_json``) for natural
  "first half, second half" split delivery and multi-message deletion
- Durable ``outbox_events`` with separate ``processed`` (listener) and ``ui_read``
  (dashboard notification bell) flags
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from config import (
    AUTO_SEND_DELAY_SECONDS,
    COOLDOWN_SECONDS,
    DB_PATH,
    DEFAULT_FOOTER_TEXT,
    DEFAULT_HISTORY_LIMIT,
    GRADUATED_AUTONOMY_THRESHOLD,
    HARDWARE_PROFILE,
    KILL_SWITCH_PAUSE_HOURS,
    KILL_SWITCH_PAUSE_SECONDS,
    MAX_AUTO_REPLIES_PER_HOUR,
    MAX_SEND_ATTEMPTS,
    OWNER_NAME,
    RAG_DIR,
    RETENTION_DAYS,
    SENSITIVITY_THRESHOLD,
    TIMEZONE_NAME,
    VALID_DRAFT_STATUSES,
    VALID_MODES,
    VALID_PLATFORMS,
    VALID_TIERS,
    VALID_TONES,
    setup_logging,
)

logger = setup_logging("echotext.database")

_DB_LOCK = threading.RLock()
_INITIALIZED_DBS: set[str] = set()

# Guarded state machine transitions for drafts
ALLOWED_DRAFT_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending_approval": frozenset({"approved", "rejected"}),
    "pending_send": frozenset({"approved", "pending_approval", "rejected", "cancelled"}),
    "approved": frozenset(
        {"approved", "sending", "sent", "failed", "rejected", "deleted_sent"}
    ),
    "sending": frozenset({"sent", "failed", "approved", "deleted_sent"}),
    "failed": frozenset({"approved", "rejected"}),
    "sent": frozenset({"deleted_sent"}),
    "edited": frozenset({"approved", "sending", "sent", "rejected", "deleted_sent"}),
    "shadow": frozenset(),
    "rejected": frozenset(),
    "cancelled": frozenset(),
    "deleted_sent": frozenset(),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    target = _utc_now() if dt is None else dt
    return target.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _sql_enum_literals(values: tuple[str, ...]) -> str:
    """Format a tuple of trusted string literals from config.py for SQL CHECK clauses."""
    return ", ".join(f"'{v}'" for v in values)


def normalize_handle(handle: str) -> str:
    """Normalize a contact handle or username to lowercase without leading '@'."""
    cleaned = str(handle).strip().lstrip("@").lower()
    if not cleaned:
        raise ValueError("Contact handle cannot be empty.")
    return cleaned


def validate_platform(app_source: str) -> str:
    """Validate that ``app_source`` is in ``VALID_PLATFORMS``."""
    cleaned = str(app_source).strip().lower()
    if cleaned not in VALID_PLATFORMS:
        raise ValueError(
            f"Unsupported platform '{app_source}'. "
            f"Must be one of: {', '.join(VALID_PLATFORMS)}."
        )
    return cleaned


def _validate_draft_transition(
    current_status: str, target_status: str, *, allow_same: bool = False
) -> None:
    """Enforce valid draft state transitions and raise ``ValueError`` on illegal jumps."""
    if current_status == target_status and not allow_same:
        raise ValueError(f"Draft is already in status '{current_status}'.")
    allowed = ALLOWED_DRAFT_TRANSITIONS.get(current_status, frozenset())
    if target_status not in allowed:
        raise ValueError(
            f"Invalid draft state transition from '{current_status}' to '{target_status}'."
        )


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a short-lived, thread-safe SQLite connection with foreign keys enforced."""
    target = DB_PATH if db_path is None else db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with _DB_LOCK:
        conn = sqlite3.connect(str(target), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
            except sqlite3.OperationalError:
                conn.execute("PRAGMA journal_mode = DELETE;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _run_schema_migrations(conn: sqlite3.Connection) -> None:
    """Apply idempotent column additions for existing databases."""
    contact_cols = {
        str(r["name"]) for r in conn.execute("PRAGMA table_info(contacts)").fetchall()
    }
    if contact_cols:
        if "platform_user_id" not in contact_cols:
            conn.execute(
                "ALTER TABLE contacts ADD COLUMN platform_user_id TEXT DEFAULT NULL;"
            )
        if "flagged_for_review" not in contact_cols:
            conn.execute(
                "ALTER TABLE contacts ADD COLUMN flagged_for_review INTEGER NOT NULL DEFAULT 0;"
            )
        if "flag_reason" not in contact_cols:
            conn.execute(
                "ALTER TABLE contacts ADD COLUMN flag_reason TEXT NOT NULL DEFAULT '';"
            )

    draft_cols = {
        str(r["name"]) for r in conn.execute("PRAGMA table_info(drafts)").fetchall()
    }
    if draft_cols:
        if "send_attempts" not in draft_cols:
            conn.execute(
                "ALTER TABLE drafts ADD COLUMN send_attempts INTEGER NOT NULL DEFAULT 0;"
            )
        if "last_error" not in draft_cols:
            conn.execute(
                "ALTER TABLE drafts ADD COLUMN last_error TEXT NOT NULL DEFAULT '';"
            )
        if "reply_parts_json" not in draft_cols:
            conn.execute(
                "ALTER TABLE drafts ADD COLUMN reply_parts_json TEXT NOT NULL DEFAULT '[]';"
            )
        if "sent_msg_ids_json" not in draft_cols:
            conn.execute(
                "ALTER TABLE drafts ADD COLUMN sent_msg_ids_json TEXT NOT NULL DEFAULT '[]';"
            )

    outbox_cols = {
        str(r["name"])
        for r in conn.execute("PRAGMA table_info(outbox_events)").fetchall()
    }
    if outbox_cols and "ui_read" not in outbox_cols:
        conn.execute(
            "ALTER TABLE outbox_events ADD COLUMN ui_read INTEGER NOT NULL DEFAULT 0;"
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'admin',
            created_at TEXT NOT NULL,
            last_login_at TEXT
        );
        """
    )


def init_db(db_path: Path | None = None) -> None:
    """Create all required tables, indexes, and migrations if not yet initialized."""
    target = (DB_PATH if db_path is None else db_path).resolve()
    key = str(target)
    with _DB_LOCK:
        if key in _INITIALIZED_DBS and target.exists():
            return

        tiers_sql = _sql_enum_literals(VALID_TIERS)
        tones_sql = _sql_enum_literals(VALID_TONES)
        modes_sql = _sql_enum_literals(VALID_MODES)
        platforms_sql = _sql_enum_literals(VALID_PLATFORMS)
        statuses_sql = _sql_enum_literals(VALID_DRAFT_STATUSES)

        ddl = f"""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL UNIQUE,
            platform_user_id TEXT DEFAULT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            relationship_tier TEXT NOT NULL DEFAULT 'unknown'
                CHECK(relationship_tier IN ({tiers_sql})),
            tone TEXT NOT NULL DEFAULT 'casual_hinglish'
                CHECK(tone IN ({tones_sql})),
            custom_persona TEXT NOT NULL DEFAULT '',
            mode TEXT NOT NULL DEFAULT 'off'
                CHECK(mode IN ({modes_sql})),
            footer_enabled INTEGER NOT NULL DEFAULT 1,
            footer_text TEXT NOT NULL DEFAULT '',
            last_auto_reply_at TEXT DEFAULT NULL,
            paused_until TEXT DEFAULT NULL,
            flagged_for_review INTEGER NOT NULL DEFAULT 0,
            flag_reason TEXT NOT NULL DEFAULT '',
            unchanged_approvals_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS contact_handles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            platform TEXT NOT NULL
                CHECK(platform IN ({platforms_sql})),
            handle TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(platform, handle),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            platform TEXT NOT NULL DEFAULT 'telegram',
            sender_role TEXT NOT NULL CHECK(sender_role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            telegram_msg_id INTEGER DEFAULT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_contact_id_desc
            ON messages(contact_id, id DESC);

        CREATE TABLE IF NOT EXISTS drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            platform TEXT NOT NULL DEFAULT 'telegram',
            incoming_text TEXT NOT NULL,
            incoming_msg_id INTEGER DEFAULT NULL,
            draft_text TEXT NOT NULL,
            options_json TEXT NOT NULL DEFAULT '[]',
            reply_parts_json TEXT NOT NULL DEFAULT '[]',
            selected_option_index INTEGER NOT NULL DEFAULT 0,
            chunks_json TEXT NOT NULL DEFAULT '[]',
            sensitivity_score INTEGER NOT NULL DEFAULT 0,
            triage_label TEXT NOT NULL DEFAULT 'chitchat',
            status TEXT NOT NULL
                CHECK(status IN ({statuses_sql})),
            scheduled_send_at TEXT DEFAULT NULL,
            sent_telegram_msg_id INTEGER DEFAULT NULL,
            sent_msg_ids_json TEXT NOT NULL DEFAULT '[]',
            send_attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            actual_owner_reply TEXT DEFAULT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_drafts_status_created
            ON drafts(status, created_at DESC);

        CREATE TABLE IF NOT EXISTS injected_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_text TEXT NOT NULL,
            scope_contact_id INTEGER DEFAULT NULL,
            scope_tier TEXT NOT NULL DEFAULT 'all_trusted',
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(scope_contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER DEFAULT NULL,
            platform TEXT NOT NULL DEFAULT 'telegram',
            event_type TEXT NOT NULL,
            mode_used TEXT NOT NULL DEFAULT '',
            triage_label TEXT NOT NULL DEFAULT '',
            sensitivity_score INTEGER NOT NULL DEFAULT 0,
            chunks_used TEXT NOT NULL DEFAULT '[]',
            outcome TEXT NOT NULL DEFAULT '',
            details TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS app_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'admin',
            created_at TEXT NOT NULL,
            last_login_at TEXT
        );

        CREATE TABLE IF NOT EXISTS outbox_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL
                CHECK(category IN ('notification', 'delete_request')),
            payload_json TEXT NOT NULL,
            processed INTEGER NOT NULL DEFAULT 0,
            ui_read INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_outbox_unprocessed
            ON outbox_events(category, processed, id ASC);
        """
        with get_connection(target) as conn:
            conn.executescript(ddl)
            _run_schema_migrations(conn)
        _INITIALIZED_DBS.add(key)


def get_setting(key: str, default: str = "", db_path: Path | None = None) -> str:
    """Read a persistent setting from ``app_settings``."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ).fetchone()
    return str(row["value"]) if row is not None else default


def set_setting(key: str, value: str, db_path: Path | None = None) -> None:
    """Write a persistent setting to ``app_settings``."""
    init_db(db_path)
    now = _iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE
                SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, str(value), now),
        )


DEFAULT_SYSTEM_SETTINGS: dict[str, Any] = {
    # 1. Timing, Cadence & Auto-Reply Rules
    "poll_interval_seconds": 2.0,
    "auto_send_delay_seconds": 10,
    "cooldown_seconds": 60,
    "multipart_gap_seconds": 0.35,
    "kill_switch_seconds": 7200,
    "kill_switch_hours": 2.0,
    "max_replies_per_hour": 5,
    # 2. AI Persona & Responses
    "owner_name": OWNER_NAME,
    "timezone_name": TIMEZONE_NAME,
    "default_footer_text": DEFAULT_FOOTER_TEXT,
    "num_draft_options": 3,
    "default_tone": "casual_hinglish",
    # 3. Privacy, Safety & Schedule
    "sensitivity_threshold": SENSITIVITY_THRESHOLD,
    "active_hours_enabled": False,
    "active_hours_start": 8,
    "active_hours_end": 23,
    "retention_days": RETENTION_DAYS,
    "hide_message_text": True,
    # 4. Hardware & Local AI Model Profile
    "hardware_profile": HARDWARE_PROFILE,
    "gpu_layers": 0,
    "model_path": "",
    # 5. Mobile Approval Bot (Telegram Bot API)
    "approval_bot_token": "",
    "owner_telegram_id": "",
    # 6. UI Theme Appearance
    "theme": "dark",
}


def get_all_system_settings(db_path: Path | None = None) -> dict[str, Any]:
    """Return all system parameters, merged with stored settings and cast to their correct types."""
    init_db(db_path)
    result = dict(DEFAULT_SYSTEM_SETTINGS)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        stored = {str(r["key"]): str(r["value"]) for r in rows}

    for k, default_val in DEFAULT_SYSTEM_SETTINGS.items():
        if k in stored:
            raw = stored[k]
            if isinstance(default_val, bool):
                result[k] = raw.strip().lower() in ("true", "1", "yes")
            elif isinstance(default_val, int):
                try:
                    result[k] = int(round(float(raw)))
                except ValueError:
                    result[k] = default_val
            elif isinstance(default_val, float):
                try:
                    result[k] = float(raw)
                except ValueError:
                    result[k] = default_val
            else:
                result[k] = raw
    return result


def save_system_settings(
    settings: dict[str, Any], db_path: Path | None = None
) -> dict[str, Any]:
    """Persist system parameters into ``app_settings`` and return the updated configuration."""
    init_db(db_path)
    for k, v in settings.items():
        if k in DEFAULT_SYSTEM_SETTINGS:
            set_setting(k, str(v), db_path=db_path)
    # Maintain bidirectional sync between kill_switch_seconds and kill_switch_hours
    if "kill_switch_seconds" in settings:
        try:
            sec = float(settings["kill_switch_seconds"])
            set_setting("kill_switch_hours", str(round(sec / 3600.0, 4)), db_path=db_path)
        except (ValueError, TypeError):
            pass
    elif "kill_switch_hours" in settings:
        try:
            hrs = float(settings["kill_switch_hours"])
            set_setting("kill_switch_seconds", str(int(round(hrs * 3600))), db_path=db_path)
        except (ValueError, TypeError):
            pass
    return get_all_system_settings(db_path=db_path)


def reset_system_settings_to_defaults(
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Restore all system parameters to their safe recommended default values."""
    init_db(db_path)
    for k, v in DEFAULT_SYSTEM_SETTINGS.items():
        set_setting(k, str(v), db_path=db_path)
    return get_all_system_settings(db_path=db_path)


# -----------------------------------------------------------------------------
# User Account & Authentication Management
# -----------------------------------------------------------------------------


def count_registered_users(db_path: Path | None = None) -> int:
    """Return total number of registered dashboard users."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM app_users").fetchone()
        return int(row["cnt"]) if row else 0


def get_user_by_username(
    username: str, db_path: Path | None = None
) -> dict[str, Any] | None:
    """Lookup a registered user by username (case-insensitive)."""
    init_db(db_path)
    norm = (username or "").strip().lower()
    if not norm:
        return None
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM app_users WHERE lower(username) = ?", (norm,)
        ).fetchone()
        return dict(row) if row else None


def list_app_users(db_path: Path | None = None) -> list[dict[str, Any]]:
    """List all registered users without exposing password hashes."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login_at FROM app_users ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def register_app_user(
    username: str,
    password: str,
    display_name: str = "",
    role: str = "admin",
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Register a new dashboard user account with PBKDF2-HMAC-SHA256 password hashing."""
    from auth import hash_password

    init_db(db_path)
    clean_user = (username or "").strip().lower()
    clean_disp = (display_name or "").strip() or clean_user
    if len(clean_user) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    existing = get_user_by_username(clean_user, db_path=db_path)
    if existing is not None:
        raise ValueError(f"Username '{clean_user}' is already registered.")

    pw_hash = hash_password(password)
    now = _iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO app_users (username, password_hash, display_name, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (clean_user, pw_hash, clean_disp, role, now),
        )
        uid = int(cur.lastrowid or 0)

    # Automatically set owner_name in settings if this is the first registered user
    if clean_disp and count_registered_users(db_path) == 1:
        set_setting("owner_name", clean_disp, db_path=db_path)
        set_setting(
            "default_footer_text",
            f"— sent by {clean_disp}'s assistant",
            db_path=db_path,
        )

    return {
        "id": uid,
        "username": clean_user,
        "display_name": clean_disp,
        "role": role,
        "created_at": now,
    }


def authenticate_app_user(
    username: str,
    password: str,
    db_path: Path | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Verify username and password against registered database users."""
    from auth import verify_password

    user = get_user_by_username(username, db_path=db_path)
    if user is None:
        return False, "User not found.", None

    if not verify_password(password, user["password_hash"]):
        return False, "Invalid password.", None

    now = _iso()
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE app_users SET last_login_at = ? WHERE id = ?",
            (now, user["id"]),
        )
    user["last_login_at"] = now
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return True, "Authentication successful.", safe_user


def update_user_password(
    username: str,
    new_password: str,
    db_path: Path | None = None,
) -> bool:
    """Update password for an existing registered user."""
    from auth import hash_password

    if len(new_password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    user = get_user_by_username(username, db_path=db_path)
    if user is None:
        raise ValueError(f"User '{username}' not found.")
    new_hash = hash_password(new_password)
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE app_users SET password_hash = ? WHERE id = ?",
            (new_hash, user["id"]),
        )
    return True


def get_sensitivity_threshold(db_path: Path | None = None) -> int:
    """Return the persisted sensitivity threshold (1–3)."""
    raw = get_setting(
        "sensitivity_threshold", str(SENSITIVITY_THRESHOLD), db_path=db_path
    )
    try:
        val = int(raw)
        return max(1, min(3, val))
    except ValueError:
        return SENSITIVITY_THRESHOLD


def set_sensitivity_threshold(threshold: int, db_path: Path | None = None) -> int:
    """Persist the user's sensitivity threshold (1–3) in ``app_settings``."""
    clamped = max(1, min(3, int(threshold)))
    set_setting("sensitivity_threshold", str(clamped), db_path=db_path)
    return clamped


def enqueue_outbox_event(
    category: str,
    payload: dict[str, Any],
    db_path: Path | None = None,
) -> int:
    """Persist a notification or delete request in ``outbox_events``."""
    init_db(db_path)
    if category not in ("notification", "delete_request"):
        raise ValueError(f"Invalid outbox category '{category}'.")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO outbox_events (category, payload_json, processed, ui_read, created_at)
            VALUES (?, ?, 0, 0, ?)
            """,
            (category, json.dumps(payload, ensure_ascii=False), _iso()),
        )
        return int(cur.lastrowid or 0)


def pop_outbox_events(
    category: str,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Atomically fetch and mark processed up to ``limit`` pending outbox events."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, payload_json
            FROM outbox_events
            WHERE category = ? AND processed = 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (category, limit),
        ).fetchall()
        if not rows:
            return []
        ids = [int(r["id"]) for r in rows]
        placeholders = ",".join("?" for _ in ids)
        update_sql = (
            "UPDATE outbox_events SET processed = 1 "
            f"WHERE id IN ({placeholders})"
        )  # nosec B608  # noqa: S608
        conn.execute(update_sql, ids)
    events: list[dict[str, Any]] = []
    for r in rows:
        try:
            events.append(json.loads(str(r["payload_json"])))
        except json.JSONDecodeError as exc:
            logger.debug("Skipping malformed outbox payload: %s", exc)
    return events


def peek_outbox_events(
    category: str,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return unprocessed outbox events without marking them processed."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM outbox_events
            WHERE category = ? AND processed = 0
            ORDER BY id ASC
            """,
            (category,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            out.append(json.loads(str(r["payload_json"])))
        except json.JSONDecodeError as exc:
            logger.debug("Skipping malformed outbox payload: %s", exc)
    return out


def list_ui_notifications(
    unread_only: bool = False,
    limit: int = 40,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return notifications for the Dashboard UI Notification Bell (`🔔`)."""
    init_db(db_path)
    where_clause = (
        "WHERE category = 'notification' AND ui_read = 0"
        if unread_only
        else "WHERE category = 'notification'"
    )
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, payload_json, ui_read, created_at
            FROM outbox_events
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,  # nosec B608  # noqa: S608
            (limit,),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for r in rows:
        try:
            payload = json.loads(str(r["payload_json"]))
        except json.JSONDecodeError:
            continue
        payload["notification_id"] = int(r["id"])
        payload["ui_read"] = bool(r["ui_read"])
        payload["created_at"] = str(r["created_at"])
        items.append(payload)
    return items


def mark_ui_notifications_read(
    notification_id: int | None = None,
    db_path: Path | None = None,
) -> int:
    """Mark one or all UI notifications as read in the Dashboard Notification Bell."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if notification_id is not None:
            cur = conn.execute(
                "UPDATE outbox_events SET ui_read = 1 WHERE id = ?",
                (int(notification_id),),
            )
        else:
            cur = conn.execute(
                "UPDATE outbox_events SET ui_read = 1 WHERE category = 'notification'"
            )
        return int(cur.rowcount or 0)


def clear_outbox_events(
    category: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Clear unprocessed outbox events (used in tests or maintenance)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if category is None:
            conn.execute("DELETE FROM outbox_events")
        else:
            conn.execute("DELETE FROM outbox_events WHERE category = ?", (category,))


def flag_contact_for_review(
    contact_id: int,
    reason: str,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Flag a contact so auto-replies are blocked until the owner reviews & confirms."""
    init_db(db_path)
    now = _iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE contacts
            SET flagged_for_review = 1,
                flag_reason = ?,
                unchanged_approvals_count = 0,
                updated_at = ?
            WHERE id = ?
            """,
            (reason.strip()[:300], now, contact_id),
        )
    log_audit_event(
        event_type="contact_flagged_for_review",
        contact_id=contact_id,
        outcome="flagged",
        details=f"Flagged until owner confirmation: {reason[:200]}",
        db_path=db_path,
    )
    return get_contact_by_id(contact_id, db_path=db_path) or {}


def clear_contact_review_flag(
    contact_id: int,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Clear the ambiguity/review flag on a contact after owner confirmation."""
    init_db(db_path)
    now = _iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE contacts
            SET flagged_for_review = 0,
                flag_reason = '',
                updated_at = ?
            WHERE id = ?
            """,
            (now, contact_id),
        )
    return get_contact_by_id(contact_id, db_path=db_path) or {}


def get_or_create_contact(
    handle: str,
    app_source: str = "telegram",
    display_name: str | None = None,
    platform_user_id: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Look up or create a contact by immutable ``platform_user_id`` or ``(platform, handle)``."""
    init_db(db_path)
    norm_handle = normalize_handle(handle)
    platform = validate_platform(app_source)
    clean_uid = str(platform_user_id).strip() if platform_user_id else None
    now = _iso()

    with get_connection(db_path) as conn:
        if clean_uid:
            uid_row = conn.execute(
                "SELECT * FROM contacts WHERE platform_user_id = ?",
                (f"{platform}:{clean_uid}",),
            ).fetchone()
            if uid_row is not None:
                cid = int(uid_row["id"])
                conn.execute(
                    "DELETE FROM contact_handles "
                    "WHERE platform = ? AND handle = ? AND contact_id != ?",
                    (platform, norm_handle, cid),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO contact_handles (
                        contact_id, platform, handle, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (cid, platform, norm_handle, now),
                )
                return dict(uid_row)

        row = conn.execute(
            """
            SELECT c.*
            FROM contacts c
            JOIN contact_handles ch ON ch.contact_id = c.id
            WHERE ch.platform = ? AND ch.handle = ?
            """,
            (platform, norm_handle),
        ).fetchone()

        if row is not None:
            existing_uid = row["platform_user_id"]
            cid = int(row["id"])
            if clean_uid:
                expected_uid = f"{platform}:{clean_uid}"
                if existing_uid is None:
                    conn.execute(
                        "UPDATE contacts SET platform_user_id = ?, updated_at = ? WHERE id = ?",
                        (expected_uid, now, cid),
                    )
                    updated_row = conn.execute(
                        "SELECT * FROM contacts WHERE id = ?", (cid,)
                    ).fetchone()
                    return dict(updated_row) if updated_row else dict(row)
                if existing_uid != expected_uid:
                    conn.execute(
                        "DELETE FROM contact_handles WHERE platform = ? AND handle = ?",
                        (platform, norm_handle),
                    )
                    row = None
            else:
                return dict(row)

        existing_contact = conn.execute(
            "SELECT * FROM contacts WHERE canonical_name = ?",
            (norm_handle,),
        ).fetchone()

        if existing_contact is not None and (
            not clean_uid
            or existing_contact["platform_user_id"] in (None, f"{platform}:{clean_uid}")
        ):
            contact_id = int(existing_contact["id"])
            if clean_uid and existing_contact["platform_user_id"] is None:
                conn.execute(
                    "UPDATE contacts SET platform_user_id = ?, updated_at = ? WHERE id = ?",
                    (f"{platform}:{clean_uid}", now, contact_id),
                )
        else:
            canonical = (
                f"{norm_handle}_{clean_uid}"
                if existing_contact is not None and clean_uid
                else norm_handle
            )
            disp = (display_name or norm_handle).strip()
            uid_val = f"{platform}:{clean_uid}" if clean_uid else None
            # Honor user-configured default_tone and default_footer_text from app_settings
            tone_row = conn.execute(
                "SELECT value FROM app_settings WHERE key = 'default_tone'"
            ).fetchone()
            def_tone = (
                str(tone_row["value"]).strip().lower()
                if tone_row and str(tone_row["value"]).strip().lower() in VALID_TONES
                else "casual_hinglish"
            )
            footer_row = conn.execute(
                "SELECT value FROM app_settings WHERE key = 'default_footer_text'"
            ).fetchone()
            def_footer = (
                str(footer_row["value"]).strip()
                if footer_row and str(footer_row["value"]).strip()
                else DEFAULT_FOOTER_TEXT
            )
            cursor = conn.execute(
                """
                INSERT INTO contacts (
                    canonical_name, platform_user_id, display_name, relationship_tier, tone,
                    custom_persona, mode, footer_enabled, footer_text,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'unknown', ?, '', 'off', 1, ?, ?, ?)
                """,
                (canonical, uid_val, disp, def_tone, def_footer, now, now),
            )
            contact_id = int(cursor.lastrowid or 0)

        conn.execute(
            """
            INSERT OR REPLACE INTO contact_handles (contact_id, platform, handle, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (contact_id, platform, norm_handle, now),
        )

        created = conn.execute(
            "SELECT * FROM contacts WHERE id = ?", (contact_id,)
        ).fetchone()
        return dict(created) if created else {}


def link_contact_handle(
    contact_id: int,
    platform: str,
    handle: str,
    db_path: Path | None = None,
) -> None:
    """Link an additional platform handle to an existing contact."""
    init_db(db_path)
    valid_plat = validate_platform(platform)
    norm_handle = normalize_handle(handle)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO contact_handles (contact_id, platform, handle, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (contact_id, valid_plat, norm_handle, _iso()),
        )


def list_contacts(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return all contacts with their linked platform handles."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        contacts = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM contacts ORDER BY canonical_name ASC"
            ).fetchall()
        ]
        handles = conn.execute(
            "SELECT contact_id, platform, handle FROM contact_handles ORDER BY platform ASC"
        ).fetchall()

    handle_map: dict[int, list[dict[str, str]]] = {}
    for h in handles:
        cid = int(h["contact_id"])
        handle_map.setdefault(cid, []).append(
            {"platform": str(h["platform"]), "handle": str(h["handle"])}
        )

    for c in contacts:
        c["handles"] = handle_map.get(int(c["id"]), [])
    return contacts


def get_contact_by_id(
    contact_id: int, db_path: Path | None = None
) -> dict[str, Any] | None:
    """Fetch a single contact by its primary key."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM contacts WHERE id = ?", (contact_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        handles = conn.execute(
            "SELECT platform, handle FROM contact_handles WHERE contact_id = ?",
            (contact_id,),
        ).fetchall()
        data["handles"] = [dict(h) for h in handles]
        return data


def update_contact_settings(
    contact_id: int,
    *,
    display_name: str | None = None,
    mode: str | None = None,
    relationship_tier: str | None = None,
    tone: str | None = None,
    custom_persona: str | None = None,
    footer_enabled: bool | None = None,
    footer_text: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Update a contact's autonomy mode, persona, relationship tier, tone, or footer settings."""
    init_db(db_path)
    existing = get_contact_by_id(contact_id, db_path=db_path)
    if existing is None:
        raise ValueError(f"Contact id={contact_id} not found.")

    new_mode = existing["mode"] if mode is None else mode.strip().lower()
    if new_mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{new_mode}'. Expected one of {VALID_MODES}.")

    new_tier = (
        existing["relationship_tier"]
        if relationship_tier is None
        else relationship_tier.strip().lower()
    )
    if new_tier not in VALID_TIERS:
        raise ValueError(f"Invalid tier '{new_tier}'. Expected one of {VALID_TIERS}.")

    new_tone = existing["tone"] if tone is None else tone.strip().lower()
    if new_tone not in VALID_TONES:
        raise ValueError(f"Invalid tone '{new_tone}'. Expected one of {VALID_TONES}.")

    new_disp = existing["display_name"] if display_name is None else display_name.strip()
    new_persona = (
        existing["custom_persona"]
        if custom_persona is None
        else custom_persona.strip()
    )
    new_footer_enabled = (
        int(existing["footer_enabled"])
        if footer_enabled is None
        else (1 if footer_enabled else 0)
    )
    new_footer_text = (
        existing["footer_text"] if footer_text is None else footer_text.strip()
    )

    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE contacts
            SET display_name = ?,
                mode = ?,
                relationship_tier = ?,
                tone = ?,
                custom_persona = ?,
                footer_enabled = ?,
                footer_text = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                new_disp,
                new_mode,
                new_tier,
                new_tone,
                new_persona,
                new_footer_enabled,
                new_footer_text,
                _iso(),
                contact_id,
            ),
        )
    updated = get_contact_by_id(contact_id, db_path=db_path)
    return updated or {}


def save_message(
    contact_id: int,
    sender_role: str,
    content: str,
    platform: str = "telegram",
    telegram_msg_id: int | None = None,
    db_path: Path | None = None,
) -> int:
    """Persist a user or assistant message in the ``messages`` table."""
    init_db(db_path)
    if sender_role not in ("user", "assistant"):
        raise ValueError("sender_role must be 'user' or 'assistant'.")
    valid_plat = validate_platform(platform)
    now = _iso()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (
                contact_id, platform, sender_role, content, telegram_msg_id, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (contact_id, valid_plat, sender_role, content, telegram_msg_id, now),
        )
        return int(cursor.lastrowid or 0)


def get_chat_history(
    contact_id: int,
    limit: int = 4,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the NEWEST ``limit`` messages for ``contact_id``, ordered oldest-to-newest."""
    init_db(db_path)
    if limit <= 0:
        return []
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, contact_id, platform, sender_role, content, telegram_msg_id, timestamp
            FROM messages
            WHERE contact_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (contact_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def record_auto_reply_sent(
    contact_id: int,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> None:
    """Update ``last_auto_reply_at`` on a contact when an auto-reply is queued or sent."""
    init_db(db_path)
    ts = _iso(now)
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE contacts SET last_auto_reply_at = ?, updated_at = ? WHERE id = ?",
            (ts, ts, contact_id),
        )


def check_rate_limit_and_cooldown(
    contact_id: int,
    cooldown_seconds: int = COOLDOWN_SECONDS,
    max_per_hour: int = MAX_AUTO_REPLIES_PER_HOUR,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> tuple[bool, str]:
    """Verify whether an auto-reply is permitted under cooldown, hourly cap, and kill-switch rules."""
    init_db(db_path)
    # Dynamically resolve from persisted app_settings when default arguments are used
    if cooldown_seconds == COOLDOWN_SECONDS:
        try:
            cooldown_seconds = int(round(float(get_setting("cooldown_seconds", str(COOLDOWN_SECONDS), db_path=db_path))))
        except ValueError:
            pass
    if max_per_hour == MAX_AUTO_REPLIES_PER_HOUR:
        try:
            max_per_hour = int(round(float(get_setting("max_replies_per_hour", str(MAX_AUTO_REPLIES_PER_HOUR), db_path=db_path))))
        except ValueError:
            pass

    current = _utc_now() if now is None else now
    contact = get_contact_by_id(contact_id, db_path=db_path)
    if contact is None:
        return False, "contact_not_found"

    paused_until_dt = _parse_iso(contact.get("paused_until"))
    if paused_until_dt is not None and paused_until_dt > current:
        return False, f"kill_switch_paused_until_{paused_until_dt.isoformat()}"

    last_reply_dt = _parse_iso(contact.get("last_auto_reply_at"))
    with get_connection(db_path) as conn:
        latest_draft_row = conn.execute(
            """
            SELECT created_at FROM drafts
            WHERE contact_id = ?
              AND status IN ('pending_send', 'approved', 'sending', 'sent')
            ORDER BY id DESC
            LIMIT 1
            """,
            (contact_id,),
        ).fetchone()
    if latest_draft_row is not None:
        draft_dt = _parse_iso(str(latest_draft_row["created_at"]))
        if draft_dt is not None and (last_reply_dt is None or draft_dt > last_reply_dt):
            last_reply_dt = draft_dt

    if last_reply_dt is not None:
        elapsed = (current - last_reply_dt).total_seconds()
        if elapsed < cooldown_seconds:
            return False, f"cooldown_active_{int(cooldown_seconds - elapsed)}s_remaining"

    one_hour_ago = _iso(current - timedelta(hours=1))
    with get_connection(db_path) as conn:
        msg_row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM messages
            WHERE contact_id = ?
              AND sender_role = 'assistant'
              AND timestamp >= ?
            """,
            (contact_id, one_hour_ago),
        ).fetchone()
        draft_row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM drafts
            WHERE contact_id = ?
              AND status IN ('pending_send', 'approved', 'sending', 'sent')
              AND created_at >= ?
            """,
            (contact_id, one_hour_ago),
        ).fetchone()
        msg_cnt = int(msg_row["cnt"]) if msg_row else 0
        draft_cnt = int(draft_row["cnt"]) if draft_row else 0
        recent_count = max(msg_cnt, draft_cnt)

    if recent_count >= max_per_hour:
        return False, f"hourly_rate_cap_reached_{recent_count}_of_{max_per_hour}"

    return True, "ok"


def trigger_kill_switch(
    contact_id: int,
    pause_hours: float | None = None,
    pause_seconds: float | None = None,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> str:
    """Pause auto-replies in a chat for ``pause_seconds`` (or ``pause_hours``) when the owner types manually."""
    init_db(db_path)
    current = _utc_now() if now is None else now
    if pause_seconds is not None:
        delta = timedelta(seconds=float(pause_seconds))
    elif pause_hours is not None:
        delta = timedelta(hours=float(pause_hours))
    else:
        live_settings = get_all_system_settings(db_path=db_path)
        sec = live_settings.get("kill_switch_seconds")
        if sec is not None:
            delta = timedelta(seconds=float(sec))
        elif live_settings.get("kill_switch_hours") is not None:
            delta = timedelta(hours=float(live_settings["kill_switch_hours"]))
        else:
            delta = timedelta(seconds=KILL_SWITCH_PAUSE_SECONDS)
    paused_until = _iso(current + delta)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE contacts
            SET paused_until = ?,
                unchanged_approvals_count = 0,
                updated_at = ?
            WHERE id = ?
            """,
            (paused_until, _iso(current), contact_id),
        )
    log_audit_event(
        event_type="kill_switch_triggered",
        contact_id=contact_id,
        outcome="paused",
        details=f"Owner typed manually; auto-replies paused until {paused_until}",
        db_path=db_path,
    )
    return paused_until


def clear_kill_switch(contact_id: int, db_path: Path | None = None) -> None:
    """Clear the manual kill-switch pause on a contact."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE contacts SET paused_until = NULL, updated_at = ? WHERE id = ?",
            (_iso(), contact_id),
        )


def create_draft(
    contact_id: int,
    incoming_text: str,
    draft_options: list[str],
    *,
    reply_parts: list[str] | None = None,
    platform: str = "telegram",
    incoming_msg_id: int | None = None,
    chunks_used: list[dict[str, Any]] | None = None,
    sensitivity_score: int = 0,
    triage_label: str = "chitchat",
    status: str = "pending_approval",
    scheduled_send_at: str | None = None,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Create a draft entry with candidate reply cards, split parts, and RAG chunk attribution."""
    init_db(db_path)
    valid_plat = validate_platform(platform)
    if status not in VALID_DRAFT_STATUSES:
        raise ValueError(f"Invalid draft status '{status}'.")
    cleaned_options = [opt.strip() for opt in draft_options if opt.strip()]
    if not cleaned_options:
        cleaned_options = ["I'm currently busy and will get back to you soon."]
    primary_draft = cleaned_options[0]
    parts_list = [p.strip() for p in (reply_parts or [primary_draft]) if p.strip()]
    created_iso = _iso(now)

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO drafts (
                contact_id, platform, incoming_text, incoming_msg_id,
                draft_text, options_json, reply_parts_json, selected_option_index,
                chunks_json, sensitivity_score, triage_label, status,
                scheduled_send_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contact_id,
                valid_plat,
                incoming_text,
                incoming_msg_id,
                primary_draft,
                json.dumps(cleaned_options, ensure_ascii=False),
                json.dumps(parts_list, ensure_ascii=False),
                json.dumps(chunks_used or [], ensure_ascii=False),
                int(sensitivity_score),
                triage_label,
                status,
                scheduled_send_at,
                created_iso,
                created_iso,
            ),
        )
        draft_id = int(cursor.lastrowid or 0)
        if status in ("pending_send", "approved"):
            conn.execute(
                "UPDATE contacts SET last_auto_reply_at = ?, updated_at = ? WHERE id = ?",
                (created_iso, created_iso, contact_id),
            )
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def _hydrate_draft(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    try:
        data["options"] = json.loads(data.get("options_json") or "[]")
    except json.JSONDecodeError:
        data["options"] = [data.get("draft_text", "")]
    try:
        parts = json.loads(data.get("reply_parts_json") or "[]")
        data["reply_parts"] = parts if parts else [data.get("draft_text", "")]
    except json.JSONDecodeError:
        data["reply_parts"] = [data.get("draft_text", "")]
    try:
        data["sent_msg_ids"] = json.loads(data.get("sent_msg_ids_json") or "[]")
    except json.JSONDecodeError:
        data["sent_msg_ids"] = []
    try:
        data["chunks"] = json.loads(data.get("chunks_json") or "[]")
    except json.JSONDecodeError:
        data["chunks"] = []
    return data


def get_draft_by_id(
    draft_id: int, db_path: Path | None = None
) -> dict[str, Any] | None:
    """Return a single hydrated draft record joined with contact metadata."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT d.*, c.canonical_name, c.display_name, c.relationship_tier,
                   c.tone, c.mode AS contact_mode, c.footer_enabled, c.footer_text,
                   c.flagged_for_review, c.flag_reason
            FROM drafts d
            JOIN contacts c ON c.id = d.contact_id
            WHERE d.id = ?
            """,
            (draft_id,),
        ).fetchone()
    return _hydrate_draft(row) if row is not None else None


def update_draft_options(
    draft_id: int,
    draft_options: list[str],
    reply_parts: list[str] | None = None,
    *,
    triage_label: str | None = None,
    chunks_used: list[dict[str, Any]] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Replace candidate options on a pending draft after owner provides context."""
    init_db(db_path)
    existing = get_draft_by_id(draft_id, db_path=db_path)
    if existing is None:
        raise ValueError(f"Draft id={draft_id} not found.")
    clean_opts = [o.strip() for o in draft_options if o and o.strip()] or [
        existing.get("draft_text", "")
    ]
    primary = clean_opts[0]
    parts = reply_parts if reply_parts else [primary]
    now = _iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE drafts
            SET draft_text = ?,
                options_json = ?,
                reply_parts_json = ?,
                triage_label = COALESCE(?, triage_label),
                chunks_json = COALESCE(?, chunks_json),
                updated_at = ?
            WHERE id = ?
            """,
            (
                primary,
                json.dumps(clean_opts, ensure_ascii=False),
                json.dumps(parts, ensure_ascii=False),
                triage_label,
                json.dumps(chunks_used, ensure_ascii=False)
                if chunks_used is not None
                else None,
                now,
                draft_id,
            ),
        )
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def list_drafts(
    statuses: list[str] | None = None,
    contact_id: int | None = None,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """List drafts filtered by status and/or contact_id, newest first."""
    init_db(db_path)
    clauses: list[str] = []
    params: list[Any] = []

    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        clauses.append(f"d.status IN ({placeholders})")
        params.extend(statuses)
    if contact_id is not None:
        clauses.append("d.contact_id = ?")
        params.append(contact_id)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT d.*, c.canonical_name, c.display_name, c.relationship_tier,
                   c.tone, c.mode AS contact_mode, c.footer_enabled, c.footer_text,
                   c.flagged_for_review, c.flag_reason
            FROM drafts d
            JOIN contacts c ON c.id = d.contact_id
            {where_sql}
            ORDER BY d.id DESC
            LIMIT ?
            """,  # nosec B608  # noqa: S608
            params,
        ).fetchall()
    return [_hydrate_draft(r) for r in rows]


def approve_draft(
    draft_id: int,
    option_index: int | None = None,
    reply_parts: list[str] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Mark a pending/failed draft as ``approved`` and clear any ambiguity flag on the contact."""
    init_db(db_path)
    draft = get_draft_by_id(draft_id, db_path=db_path)
    if draft is None:
        raise ValueError(f"Draft id={draft_id} not found.")

    current_status = str(draft.get("status") or "")
    _validate_draft_transition(current_status, "approved", allow_same=False)

    options: list[str] = draft.get("options") or [draft["draft_text"]]
    idx = draft["selected_option_index"] if option_index is None else int(option_index)
    if 0 <= idx < len(options):
        chosen_text = options[idx]
    else:
        idx = 0
        chosen_text = draft["draft_text"]

    parts_json = json.dumps(
        reply_parts if reply_parts else [chosen_text], ensure_ascii=False
    )
    now = _iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE drafts
            SET draft_text = ?,
                reply_parts_json = ?,
                selected_option_index = ?,
                status = 'approved',
                updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (chosen_text, parts_json, idx, now, draft_id, current_status),
        )
        if (cur.rowcount or 0) == 0:
            raise ValueError(f"Draft id={draft_id} state changed concurrently.")
        conn.execute(
            """
            UPDATE contacts
            SET unchanged_approvals_count = unchanged_approvals_count + 1,
                flagged_for_review = 0,
                flag_reason = '',
                updated_at = ?
            WHERE id = ?
            """,
            (now, int(draft["contact_id"])),
        )

    log_audit_event(
        event_type="draft_approved",
        contact_id=int(draft["contact_id"]),
        platform=str(draft["platform"]),
        mode_used=str(draft.get("contact_mode", "draft")),
        triage_label=str(draft.get("triage_label", "")),
        sensitivity_score=int(draft.get("sensitivity_score", 0)),
        chunks_used=draft.get("chunks", []),
        outcome="approved",
        details=f"Approved option #{idx + 1} (cleared any contact review flag)",
        db_path=db_path,
    )
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def edit_and_approve_draft(
    draft_id: int,
    new_text: str,
    reply_parts: list[str] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Update a draft's text with human edits, mark ``approved``, and clear contact review flag."""
    init_db(db_path)
    cleaned = new_text.strip()
    if not cleaned:
        raise ValueError("Edited draft text cannot be empty.")
    draft = get_draft_by_id(draft_id, db_path=db_path)
    if draft is None:
        raise ValueError(f"Draft id={draft_id} not found.")

    current_status = str(draft.get("status") or "")
    _validate_draft_transition(current_status, "approved", allow_same=True)

    parts_json = json.dumps(
        reply_parts if reply_parts else [cleaned], ensure_ascii=False
    )
    now = _iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE drafts
            SET draft_text = ?,
                reply_parts_json = ?,
                status = 'approved',
                updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (cleaned, parts_json, now, draft_id, current_status),
        )
        if (cur.rowcount or 0) == 0:
            raise ValueError(f"Draft id={draft_id} state changed concurrently.")
        conn.execute(
            """
            UPDATE contacts
            SET unchanged_approvals_count = 0,
                flagged_for_review = 0,
                flag_reason = '',
                updated_at = ?
            WHERE id = ?
            """,
            (now, int(draft["contact_id"])),
        )

    log_audit_event(
        event_type="draft_edited_and_approved",
        contact_id=int(draft["contact_id"]),
        platform=str(draft["platform"]),
        mode_used=str(draft.get("contact_mode", "draft")),
        triage_label=str(draft.get("triage_label", "")),
        sensitivity_score=int(draft.get("sensitivity_score", 0)),
        chunks_used=draft.get("chunks", []),
        outcome="edited",
        details="Edited before approval (cleared any contact review flag)",
        db_path=db_path,
    )
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def reject_draft(draft_id: int, db_path: Path | None = None) -> dict[str, Any]:
    """Mark a draft as ``rejected`` and clear any ambiguity review flag on the contact."""
    init_db(db_path)
    draft = get_draft_by_id(draft_id, db_path=db_path)
    if draft is None:
        raise ValueError(f"Draft id={draft_id} not found.")

    current_status = str(draft.get("status") or "")
    _validate_draft_transition(current_status, "rejected", allow_same=False)

    now = _iso()
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE drafts SET status = 'rejected', updated_at = ? WHERE id = ?",
            (now, draft_id),
        )
        conn.execute(
            """
            UPDATE contacts
            SET unchanged_approvals_count = 0,
                flagged_for_review = 0,
                flag_reason = '',
                updated_at = ?
            WHERE id = ?
            """,
            (now, int(draft["contact_id"])),
        )
    log_audit_event(
        event_type="draft_rejected",
        contact_id=int(draft["contact_id"]),
        platform=str(draft["platform"]),
        mode_used=str(draft.get("contact_mode", "draft")),
        triage_label=str(draft.get("triage_label", "")),
        sensitivity_score=int(draft.get("sensitivity_score", 0)),
        chunks_used=draft.get("chunks", []),
        outcome="rejected",
        details="Rejected by owner (cleared any contact review flag)",
        db_path=db_path,
    )
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def cancel_pending_auto_send(
    draft_id: int, db_path: Path | None = None
) -> dict[str, Any]:
    """Cancel a ``pending_send`` auto-reply during its 10s delay window."""
    init_db(db_path)
    draft = get_draft_by_id(draft_id, db_path=db_path)
    if draft is None:
        raise ValueError(f"Draft id={draft_id} not found.")

    current_status = str(draft.get("status") or "")
    if current_status != "pending_send":
        raise ValueError(
            f"Cannot cancel auto-send for draft #{draft_id} in status '{current_status}'."
        )

    now = _iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE drafts
            SET status = 'pending_approval',
                scheduled_send_at = NULL,
                updated_at = ?
            WHERE id = ? AND status = 'pending_send'
            """,
            (now, draft_id),
        )
        conn.execute(
            """
            UPDATE contacts
            SET unchanged_approvals_count = 0,
                updated_at = ?
            WHERE id = ?
            """,
            (now, int(draft["contact_id"])),
        )
    log_audit_event(
        event_type="auto_reply_cancelled_in_window",
        contact_id=int(draft["contact_id"]),
        platform=str(draft["platform"]),
        mode_used="auto",
        outcome="converted_to_draft",
        details=f"Draft #{draft_id} cancelled before auto-send timer expired",
        db_path=db_path,
    )
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def promote_due_pending_sends(
    now: datetime | None = None, db_path: Path | None = None
) -> list[dict[str, Any]]:
    """Promote ``pending_send`` drafts whose 10-second delay window has elapsed."""
    init_db(db_path)
    current_iso = _iso(now)
    with get_connection(db_path) as conn:
        due_rows = conn.execute(
            """
            SELECT id FROM drafts
            WHERE status = 'pending_send'
              AND scheduled_send_at IS NOT NULL
              AND scheduled_send_at <= ?
            """,
            (current_iso,),
        ).fetchall()
        due_ids = [int(r["id"]) for r in due_rows]
        if due_ids:
            placeholders = ",".join("?" for _ in due_ids)
            promote_sql = (
                f"UPDATE drafts SET status = 'approved', updated_at = ? "  # nosec B608  # noqa: S608
                f"WHERE id IN ({placeholders})"
            )
            conn.execute(promote_sql, [current_iso, *due_ids])
    return [d for d in (get_draft_by_id(did, db_path=db_path) for did in due_ids) if d]


def claim_approved_drafts_for_sending(
    limit: int = 20,
    max_attempts: int = MAX_SEND_ATTEMPTS,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Atomically claim ``approved`` drafts by transitioning them to ``sending``."""
    init_db(db_path)
    current = _utc_now() if now is None else now
    current_iso = _iso(current)
    stale_cutoff = _iso(current - timedelta(seconds=120))

    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE drafts
            SET status = 'failed',
                last_error = CASE
                    WHEN last_error = '' THEN 'Max send attempts exceeded'
                    ELSE last_error
                END,
                updated_at = ?
            WHERE status IN ('approved', 'sending') AND send_attempts >= ?
            """,
            (current_iso, max_attempts),
        )
        conn.execute(
            """
            UPDATE drafts
            SET status = 'approved',
                updated_at = ?
            WHERE status = 'sending' AND updated_at <= ? AND send_attempts < ?
            """,
            (current_iso, stale_cutoff, max_attempts),
        )

        rows = conn.execute(
            """
            SELECT id FROM drafts
            WHERE status = 'approved' AND send_attempts < ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (max_attempts, limit),
        ).fetchall()
        claim_ids = [int(r["id"]) for r in rows]
        if claim_ids:
            placeholders = ",".join("?" for _ in claim_ids)
            conn.execute(
                f"""
                UPDATE drafts
                SET status = 'sending',
                    send_attempts = send_attempts + 1,
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,  # nosec B608  # noqa: S608
                [current_iso, *claim_ids],
            )
    return [
        d for d in (get_draft_by_id(did, db_path=db_path) for did in claim_ids) if d
    ]


def mark_draft_sent(
    draft_id: int,
    sent_telegram_msg_id: int | None = None,
    sent_msg_ids: list[int] | None = None,
    final_sent_text: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Record that an approved/sending draft was delivered on Telegram (including multi-part IDs)."""
    init_db(db_path)
    draft = get_draft_by_id(draft_id, db_path=db_path)
    if draft is None:
        raise ValueError(f"Draft id={draft_id} not found.")

    current_status = str(draft.get("status") or "")
    _validate_draft_transition(current_status, "sent", allow_same=False)

    text_to_save = (final_sent_text or draft["draft_text"]).strip()
    all_ids = [int(x) for x in (sent_msg_ids or []) if x]
    if sent_telegram_msg_id and int(sent_telegram_msg_id) not in all_ids:
        all_ids.insert(0, int(sent_telegram_msg_id))
    primary_msg_id = all_ids[0] if all_ids else sent_telegram_msg_id

    now = _iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE drafts
            SET status = 'sent',
                draft_text = ?,
                sent_telegram_msg_id = ?,
                sent_msg_ids_json = ?,
                last_error = '',
                updated_at = ?
            WHERE id = ?
            """,
            (
                text_to_save,
                primary_msg_id,
                json.dumps(all_ids),
                now,
                draft_id,
            ),
        )
    save_message(
        contact_id=int(draft["contact_id"]),
        sender_role="assistant",
        content=text_to_save,
        platform=str(draft["platform"]),
        telegram_msg_id=primary_msg_id,
        db_path=db_path,
    )
    record_auto_reply_sent(int(draft["contact_id"]), db_path=db_path)
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def mark_draft_failed(
    draft_id: int,
    error_message: str = "Telegram delivery failed",
    max_attempts: int = MAX_SEND_ATTEMPTS,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Record a failed delivery attempt; retry up to ``max_attempts`` then mark ``failed``."""
    init_db(db_path)
    draft = get_draft_by_id(draft_id, db_path=db_path)
    if draft is None:
        raise ValueError(f"Draft id={draft_id} not found.")

    current_status = str(draft.get("status") or "")
    _validate_draft_transition(current_status, "failed", allow_same=False)

    attempts = int(draft.get("send_attempts") or 0)
    if current_status == "approved":
        attempts += 1
    next_status = "failed" if attempts >= max_attempts else "approved"
    now = _iso()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE drafts
            SET status = ?,
                send_attempts = ?,
                last_error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (next_status, attempts, error_message[:300], now, draft_id),
        )
    log_audit_event(
        event_type="draft_send_failed",
        contact_id=int(draft["contact_id"]),
        platform=str(draft["platform"]),
        outcome=next_status,
        details=f"Attempt {attempts}/{max_attempts}: {error_message[:200]}",
        db_path=db_path,
    )
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def request_delete_sent_reply(
    draft_id: int, db_path: Path | None = None
) -> dict[str, Any]:
    """Mark a sent reply for deletion on Telegram and enqueue outbox deletion for all parts."""
    init_db(db_path)
    draft = get_draft_by_id(draft_id, db_path=db_path)
    if draft is None:
        raise ValueError(f"Draft id={draft_id} not found.")

    current_status = str(draft.get("status") or "")
    _validate_draft_transition(current_status, "deleted_sent", allow_same=False)

    now = _iso()
    sent_msg_id = draft.get("sent_telegram_msg_id")
    all_ids: list[int] = [int(x) for x in (draft.get("sent_msg_ids") or []) if x]
    if sent_msg_id is not None and int(sent_msg_id) not in all_ids:
        all_ids.append(int(sent_msg_id))

    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE drafts SET status = 'deleted_sent', updated_at = ? WHERE id = ?",
            (now, draft_id),
        )
        for mid in all_ids:
            conn.execute(
                "DELETE FROM messages WHERE contact_id = ? AND telegram_msg_id = ?",
                (int(draft["contact_id"]), int(mid)),
            )

    if all_ids:
        enqueue_outbox_event(
            "delete_request",
            {
                "draft_id": draft_id,
                "contact_handle": draft["canonical_name"],
                "platform": draft["platform"],
                "sent_telegram_msg_id": all_ids[0],
                "sent_msg_ids": all_ids,
            },
            db_path=db_path,
        )

    log_audit_event(
        event_type="sent_reply_deleted",
        contact_id=int(draft["contact_id"]),
        platform=str(draft["platform"]),
        outcome="deleted_sent",
        details=f"Deleted sent reply draft_id={draft_id} (msg_ids={all_ids})",
        db_path=db_path,
    )
    return get_draft_by_id(draft_id, db_path=db_path) or {}


def add_injected_fact(
    fact_text: str,
    ttl_minutes: int = 120,
    scope_contact_id: int | None = None,
    scope_tier: str = "all_trusted",
    now: datetime | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Insert a temporary fact with an explicit expiry timestamp and contact/tier scope."""
    init_db(db_path)
    cleaned = fact_text.strip()
    if not cleaned:
        raise ValueError("Injected fact cannot be empty.")
    current = _utc_now() if now is None else now
    expires_at = _iso(current + timedelta(minutes=max(1, ttl_minutes)))
    created_at = _iso(current)

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO injected_facts (
                fact_text, scope_contact_id, scope_tier, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (cleaned, scope_contact_id, scope_tier, expires_at, created_at),
        )
        fact_id = int(cursor.lastrowid or 0)
        row = conn.execute(
            "SELECT * FROM injected_facts WHERE id = ?", (fact_id,)
        ).fetchone()
    return dict(row) if row else {}


def get_active_injected_facts(
    contact_id: int | None = None,
    relationship_tier: str = "unknown",
    now: datetime | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return non-expired injected facts scoped to ``contact_id`` or ``relationship_tier``."""
    init_db(db_path)
    current_iso = _iso(now)
    with get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM injected_facts WHERE expires_at <= ?", (current_iso,)
        )
        rows = conn.execute(
            "SELECT * FROM injected_facts WHERE expires_at > ? ORDER BY id DESC",
            (current_iso,),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        sc_id = item.get("scope_contact_id")
        sc_tier = str(item.get("scope_tier") or "all_trusted")

        if sc_id is not None:
            if contact_id is not None and int(sc_id) == int(contact_id):
                results.append(item)
            continue

        if sc_tier == "public":
            results.append(item)
        elif sc_tier == "all_trusted" and relationship_tier != "unknown":
            results.append(item)
        elif sc_tier == relationship_tier:
            results.append(item)

    return results


def delete_injected_fact(fact_id: int, db_path: Path | None = None) -> None:
    """Remove a specific injected fact immediately."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM injected_facts WHERE id = ?", (fact_id,))


def forget_contact(
    contact_id: int,
    rag_dir: Path | None = None,
    db_path: Path | None = None,
) -> bool:
    """Completely erase a contact, handles, messages, drafts, and contact-scoped RAG notes."""
    init_db(db_path)
    contact = get_contact_by_id(contact_id, db_path=db_path)
    if contact is None:
        return False

    canonical = str(contact["canonical_name"])
    handles = [str(h["handle"]) for h in contact.get("handles", [])]
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM messages WHERE contact_id = ?", (contact_id,))
        conn.execute("DELETE FROM drafts WHERE contact_id = ?", (contact_id,))
        conn.execute("DELETE FROM contact_handles WHERE contact_id = ?", (contact_id,))
        conn.execute(
            "DELETE FROM injected_facts WHERE scope_contact_id = ?", (contact_id,)
        )
        conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))

    target_rag = RAG_DIR if rag_dir is None else rag_dir
    if target_rag.exists():
        for folder_name in {canonical, *handles}:
            contact_folder = target_rag / "contacts" / folder_name
            if contact_folder.exists() and contact_folder.is_dir():
                for child in contact_folder.iterdir():
                    if child.is_file():
                        child.unlink(missing_ok=True)
                try:
                    contact_folder.rmdir()
                except OSError as exc:
                    logger.debug("Could not remove contact RAG dir: %s", exc)

    log_audit_event(
        event_type="contact_forgotten",
        contact_id=None,
        outcome="erased",
        details=f"Erased all data and scoped notes for contact '{canonical}' (id={contact_id})",
        db_path=db_path,
    )
    return True


def purge_old_records(
    retention_days: int = RETENTION_DAYS,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> dict[str, int]:
    """Delete messages, drafts, audit logs, and processed outbox events older than retention."""
    init_db(db_path)
    if retention_days <= 0:
        return {"messages_deleted": 0, "drafts_deleted": 0, "audit_logs_deleted": 0}
    current = _utc_now() if now is None else now
    cutoff = _iso(current - timedelta(days=retention_days))
    with get_connection(db_path) as conn:
        m_cur = conn.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
        d_cur = conn.execute("DELETE FROM drafts WHERE created_at < ?", (cutoff,))
        a_cur = conn.execute("DELETE FROM audit_log WHERE created_at < ?", (cutoff,))
        conn.execute(
            "DELETE FROM outbox_events WHERE processed = 1 AND ui_read = 1 AND created_at < ?",
            (cutoff,),
        )
        return {
            "messages_deleted": int(m_cur.rowcount or 0),
            "drafts_deleted": int(d_cur.rowcount or 0),
            "audit_logs_deleted": int(a_cur.rowcount or 0),
        }


def log_audit_event(
    event_type: str,
    *,
    contact_id: int | None = None,
    platform: str = "telegram",
    mode_used: str = "",
    triage_label: str = "",
    sensitivity_score: int = 0,
    chunks_used: list[Any] | None = None,
    outcome: str = "",
    details: str = "",
    db_path: Path | None = None,
) -> None:
    """Append a structured entry to ``audit_log``."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (
                contact_id, platform, event_type, mode_used, triage_label,
                sensitivity_score, chunks_used, outcome, details, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contact_id,
                platform,
                event_type,
                mode_used,
                triage_label,
                int(sensitivity_score),
                json.dumps(chunks_used or [], ensure_ascii=False),
                outcome,
                details,
                _iso(),
            ),
        )


def list_audit_logs(
    limit: int = 50, db_path: Path | None = None
) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` audit log entries."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.*, c.canonical_name, c.display_name
            FROM audit_log a
            LEFT JOIN contacts c ON c.id = a.contact_id
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_contacts_eligible_for_graduated_autonomy(
    threshold: int = GRADUATED_AUTONOMY_THRESHOLD,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return contacts in ``draft`` mode who have >= ``threshold`` unchanged approvals."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM contacts
            WHERE mode = 'draft'
              AND flagged_for_review = 0
              AND unchanged_approvals_count >= ?
            ORDER BY unchanged_approvals_count DESC
            """,
            (threshold,),
        ).fetchall()
    return [dict(r) for r in rows]
