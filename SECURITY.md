# Security Policy (`SECURITY.md`)

## Threat Model & Enforced Safeguards

EchoText is a **single-user, self-hosted** local Telegram assistant. Because it connects to your personal Telegram account and reads local notes, security and privacy controls are enforced in code:

1. **Mandatory Local API Authentication & Host Allowlist**:
   - The FastAPI Brain server (`server.py`) requires a valid `X-API-Key` header verified with `hmac.compare_digest` on all operational endpoints and rejects requests with foreign `Host` headers (`ALLOWED_HOSTS` middleware).
   - If `ECHOTEXT_API_KEY` is not explicitly set in `.env`, `config.py` automatically generates and persists a 256-bit random key in `logs/.echotext_api_key` (`0600` permissions) shared across local EchoText processes.

2. **Localhost Dashboard + Global Brute-Force Lockout**:
   - The Streamlit dashboard (`app.py`) binds strictly to `127.0.0.1` via `.streamlit/config.toml` and `start_echotext.py`.
   - Dashboard access requires single-password authentication (`auth.py`) verified with `PBKDF2-HMAC-SHA256` (200,000 iterations), rejects known placeholder passwords, enforces a **global process-wide lockout** after 5 failed attempts (so opening a new browser session cannot bypass lockout), and auto-logs out after 30 minutes of inactivity.

3. **Immutable Contact Identity & Allowlist by Default**:
   - Every new contact starts at `mode='off'` and receives zero replies until explicitly enabled.
   - Telegram contacts are bound to their immutable numeric `platform_user_id`. If a Telegram username is reassigned to a different account, the new account is isolated as a fresh `mode='off'`, `tier='unknown'` contact and never inherits prior permissions, history, or notes.

4. **Guarded Draft State Machine, Enqueue Rate Caps & Fail-Closed `BUSY` Gate**:
   - EchoText acts **only** when `proxy_status.txt` explicitly reads `BUSY`.
   - Cooldown (60s) and hourly auto-reply caps (5/hr) are counted at draft creation time so message bursts cannot bypass rate limits.
   - Draft transitions are strictly guarded (`pending_approval`, `pending_send`, `approved`, `sending`, `sent`, `failed`, `rejected`, `shadow`, `cancelled`, `deleted_sent`) with atomic send claims and a 3-attempt retry ceiling.

5. **Multilingual Sensitivity Gate & Prompt-Injection Defense**:
   - Sensitivity scoring (0–3) covers English, Roman Hinglish, and Devanagari Hindi (`ओटीपी`, `पासवर्ड`, `कार्ड नंबर`, `खाता संख्या`, `सीवीवी`, `आधार`, `पैन कार्ड`, 6+ digit strings) and fails safe to score `3` on any error.
   - Sender display names and custom persona fields are sanitized against newline/tag/instruction injection before prompt construction.
   - Earlier chat messages are passed as separate `user`/`assistant` turns, retrieved notes sit inside `<reference_notes readonly="true">`, and raw note text is never pasted into canned fallback replies.
   - Message text is redacted in logs by default (`ECHOTEXT_HIDE_MESSAGE_TEXT=true`).

## Reporting a Vulnerability

If you discover a security issue in EchoText, please open a private security advisory on [https://github.com/Abhay0541/EchoText/security/advisories](https://github.com/Abhay0541/EchoText/security/advisories).
