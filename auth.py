"""Single-password authentication with PBKDF2-HMAC-SHA256 hashing, global lockout, and auto-logout.

Used by the localhost Streamlit dashboard (``app.py``) and optional API routes to protect
message feeds, drafts, and RAG uploads against unauthorized local or LAN access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import secrets
import threading
import time

from config import (
    DASHBOARD_LOCKOUT_ATTEMPTS,
    DASHBOARD_LOCKOUT_SECONDS,
    DASHBOARD_PASSWORD,
    DASHBOARD_PASSWORD_HASH,
    DASHBOARD_SESSION_TIMEOUT_SECONDS,
    INSECURE_DEFAULT_PASSWORDS,
    setup_logging,
)

logger = setup_logging("echotext.auth")

PBKDF2_ITERATIONS: int = 200_000

_LOCKOUT_GUARD = threading.Lock()
# Shared process-wide brute-force lockout counters keyed by password hash so opening
# a new Streamlit browser tab or session cannot reset failed login attempts.
_GLOBAL_LOCKOUT_STATE: dict[str, tuple[int, float]] = {}
_ACTIVE_SESSION_TOKENS: dict[str, dict[str, Any]] = {}
_DEFAULT_DASHBOARD_HASH: str = ""


def _get_default_dashboard_hash() -> str:
    """Compute or return the configured dashboard password hash once per process."""
    global _DEFAULT_DASHBOARD_HASH
    if _DEFAULT_DASHBOARD_HASH:
        return _DEFAULT_DASHBOARD_HASH
    if DASHBOARD_PASSWORD_HASH:
        _DEFAULT_DASHBOARD_HASH = DASHBOARD_PASSWORD_HASH
    elif DASHBOARD_PASSWORD and DASHBOARD_PASSWORD.lower() not in INSECURE_DEFAULT_PASSWORDS:
        _DEFAULT_DASHBOARD_HASH = hash_password(DASHBOARD_PASSWORD)
    return _DEFAULT_DASHBOARD_HASH


def reset_global_lockout() -> None:
    """Clear global brute-force lockout state (used by unit tests)."""
    with _LOCKOUT_GUARD:
        _GLOBAL_LOCKOUT_STATE.clear()


def hash_password(password: str, salt: str | None = None) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256.

    Returns a formatted string ``pbkdf2_sha256$<iterations>$<salt>$<hex_digest>``.
    """
    if not password:
        raise ValueError("Password cannot be empty.")
    actual_salt = salt if salt is not None else secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        actual_salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${actual_salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time verification of ``password`` against ``stored_hash``."""
    if not password or not stored_hash:
        return False
    parts = stored_hash.strip().split("$")
    if len(parts) == 4 and parts[0] == "pbkdf2_sha256":
        try:
            iterations = int(parts[1])
        except ValueError:
            return False
        salt = parts[2]
        expected_hex = parts[3]
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return hmac.compare_digest(dk.hex(), expected_hex)

    if stored_hash.startswith("sha256:"):
        expected = stored_hash.split(":", 1)[1]
        actual = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(actual, expected)

    return False


@dataclass
class AuthTracker:
    """Tracks failed login attempts (shared globally per password hash) and session timeouts."""

    password_hash: str = ""  # noqa: S105
    max_attempts: int = DASHBOARD_LOCKOUT_ATTEMPTS
    lockout_seconds: int = DASHBOARD_LOCKOUT_SECONDS
    session_timeout_seconds: int = DASHBOARD_SESSION_TIMEOUT_SECONDS
    failed_attempts: int = 0
    locked_until: float = 0.0
    authenticated: bool = False
    last_activity_at: float = field(default_factory=lambda: 0.0)
    current_username: str = ""
    current_display_name: str = ""
    session_token: str = ""

    def __post_init__(self) -> None:
        if not self.password_hash:
            self.password_hash = _get_default_dashboard_hash()
        self._sync_from_global()

    def _state_key(self) -> str:
        return self.password_hash or "__default__"

    def _sync_from_global(self) -> None:
        with _LOCKOUT_GUARD:
            state = _GLOBAL_LOCKOUT_STATE.get(self._state_key())
            if state is not None:
                self.failed_attempts, self.locked_until = state

    def _sync_to_global(self) -> None:
        with _LOCKOUT_GUARD:
            _GLOBAL_LOCKOUT_STATE[self._state_key()] = (
                self.failed_attempts,
                self.locked_until,
            )

    def is_configured(self) -> bool:
        """Return True if a valid non-placeholder password or hash has been configured, or user registered."""
        if bool(self.password_hash):
            return True
        try:
            from database import count_registered_users
            return count_registered_users() > 0
        except Exception:
            return False

    def has_registered_users(self) -> bool:
        """Return True if at least one user is registered in the database."""
        try:
            from database import count_registered_users
            return count_registered_users() > 0
        except Exception:
            return False

    def is_locked_out(self, now: float | None = None) -> bool:
        """Return True if login is currently locked due to repeated failed attempts."""
        self._sync_from_global()
        current = time.time() if now is None else now
        if self.locked_until > current:
            return True
        if self.locked_until > 0 and current >= self.locked_until:
            self.locked_until = 0.0
            self.failed_attempts = 0
            self._sync_to_global()
        return False

    def remaining_lockout_seconds(self, now: float | None = None) -> int:
        """Return remaining seconds in the current lockout window."""
        self._sync_from_global()
        current = time.time() if now is None else now
        if self.locked_until <= current:
            return 0
        return int(round(self.locked_until - current))

    def check_session_active(self, now: float | None = None) -> bool:
        """Verify if the user is authenticated and has not exceeded the auto-logout window."""
        current = time.time() if now is None else now
        if not self.authenticated:
            return False
        if (
            self.last_activity_at > 0
            and (current - self.last_activity_at) > self.session_timeout_seconds
        ):
            self.logout()
            return False
        self.last_activity_at = current
        return True

    def attempt_login(
        self,
        candidate_password: str,
        username: str = "",
        now: float | None = None,
        db_path: Path | None = None,
    ) -> tuple[bool, str]:
        """Attempt to authenticate with username and password, or candidate_password directly."""
        current = time.time() if now is None else now

        if self.is_locked_out(current):
            rem = self.remaining_lockout_seconds(current)
            return False, f"Account locked due to failed attempts. Try again in {rem}s."

        # Case A: Explicit username provided
        if username.strip():
            try:
                from database import authenticate_app_user
                ok, msg, user = authenticate_app_user(username.strip(), candidate_password, db_path=db_path)
                if ok and user:
                    self.failed_attempts = 0
                    self.locked_until = 0.0
                    self._sync_to_global()
                    self.authenticated = True
                    self.current_username = str(user.get("username") or username.strip())
                    self.current_display_name = str(user.get("display_name") or self.current_username)
                    self.last_activity_at = current
                    return True, f"Welcome back, {self.current_display_name}!"
            except Exception as exc:
                logger.debug("Database user auth check failed: %s", exc)

        # Case B: Direct password verification against configured password hash
        if self.password_hash and verify_password(candidate_password, self.password_hash):
            self.failed_attempts = 0
            self.locked_until = 0.0
            self._sync_to_global()
            self.authenticated = True
            self.current_username = username.strip() or "admin"
            self.current_display_name = self.current_username
            self.last_activity_at = current
            return True, "Authenticated."

        # Case C: Single password provided, check if it matches any registered DB user
        if not username.strip():
            try:
                from database import list_app_users, get_user_by_username
                users = list_app_users(db_path=db_path)
                for u in users:
                    full_u = get_user_by_username(u["username"], db_path=db_path)
                    if full_u and verify_password(candidate_password, full_u.get("password_hash", "")):
                        self.failed_attempts = 0
                        self.locked_until = 0.0
                        self._sync_to_global()
                        self.authenticated = True
                        self.current_username = full_u["username"]
                        self.current_display_name = full_u.get("display_name") or full_u["username"]
                        self.last_activity_at = current
                        return True, f"Welcome back, {self.current_display_name}!"
            except Exception:
                pass

        if not self.is_configured():
            return (
                False,
                "No dashboard account is registered yet. Please register your owner account.",
            )

        self.failed_attempts += 1
        if self.failed_attempts >= self.max_attempts:
            self.locked_until = current + self.lockout_seconds
            self._sync_to_global()
            return (
                False,
                f"Too many failed attempts ({self.failed_attempts}). "
                f"Locked out for {self.lockout_seconds}s.",
            )

        self._sync_to_global()
        attempts_left = self.max_attempts - self.failed_attempts
        return (
            False,
            f"Invalid credentials ({attempts_left} attempt(s) remaining before lockout).",
        )

    def attempt_register(
        self,
        username: str,
        password: str,
        display_name: str = "",
        db_path: Path | None = None,
    ) -> tuple[bool, str]:
        """Register a new dashboard user account and log in."""
        try:
            from database import register_app_user
            user = register_app_user(username, password, display_name=display_name, db_path=db_path)
            self.failed_attempts = 0
            self.locked_until = 0.0
            self._sync_to_global()
            self.authenticated = True
            self.current_username = user["username"]
            self.current_display_name = user["display_name"]
            self.last_activity_at = time.time()
            return True, f"Account created! Welcome, {self.current_display_name}."
        except Exception as exc:
            return False, str(exc)

    def create_session(self) -> str:
        """Create a secure session token valid for session_timeout_seconds or 24h."""
        token = secrets.token_hex(24)
        with _LOCKOUT_GUARD:
            _ACTIVE_SESSION_TOKENS[token] = {
                "username": self.current_username,
                "display_name": self.current_display_name,
                "expires_at": time.time() + max(86400, self.session_timeout_seconds),
            }
        self.session_token = token
        return token

    def restore_session(self, token: str, now: float | None = None) -> bool:
        """Validate and restore authentication from a persistent session token."""
        if not token:
            return False
        current = time.time() if now is None else now
        with _LOCKOUT_GUARD:
            info = _ACTIVE_SESSION_TOKENS.get(token)
            if info is None or current > info["expires_at"]:
                _ACTIVE_SESSION_TOKENS.pop(token, None)
                return False
            info["expires_at"] = current + max(86400, self.session_timeout_seconds)
            self.authenticated = True
            self.current_username = str(info["username"])
            self.current_display_name = str(info["display_name"])
            self.last_activity_at = current
            self.session_token = token
            return True

    def revoke_session(self, token: str | None = None) -> None:
        """Revoke a session token upon explicit logout."""
        tok = token or self.session_token
        if tok:
            with _LOCKOUT_GUARD:
                _ACTIVE_SESSION_TOKENS.pop(tok, None)
        self.session_token = ""

    def logout(self) -> None:
        """Immediately invalidate the current authenticated session."""
        self.revoke_session()
        self.authenticated = False
        self.current_username = ""
        self.current_display_name = ""
        self.last_activity_at = 0.0
