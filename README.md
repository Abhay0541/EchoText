# EchoText v2.0 — Local Privacy-First Telegram Assistant

EchoText is a self-hosted, single-user Telegram assistant that runs a local GGUF language model (`Llama-3.2-3B-Instruct` on laptop/phone CPU or `Meta-Llama-3.1-8B-Instruct` on PC) to draft or send in-voice replies while you are studying, in class, or away — without sending your private messages or notes to a cloud LLM API.

> **Transparency on Data Flow**: Model inference, BM25 note search, and SQLite storage run 100% locally on your machine. Incoming and outgoing chat messages travel through Telegram's servers via your Telethon session.

## Core Architecture & Safety Controls

### 1. Access Control & Transport Hardening
- **Mandatory Brain API Key + Host Allowlist**: `server.py` enforces constant-time `X-API-Key` verification on all state-reading and mutating endpoints (auto-provisioned in `logs/.echotext_api_key` with `0600` permissions if `ECHOTEXT_API_KEY` is not set in `.env`) and rejects non-local `Host` headers to prevent browser CSRF and DNS-rebinding attacks.
- **Localhost Dashboard + Global Lockout**: `app.py` binds strictly to `127.0.0.1`, verifies passwords with `PBKDF2-HMAC-SHA256` (200,000 iterations), rejects weak placeholder passwords, enforces a global cross-session brute-force lockout (5 attempts -> 300s lock), and auto-logs out after 30 minutes of inactivity.
- **Fail-Closed `BUSY` Gate**: EchoText acts **only** when `proxy_status.txt` explicitly reads `BUSY` (written atomically via temp-file rename). Missing or empty status files fail closed to `AVAILABLE` (complete silence).
- **Immutable Telegram User ID Binding**: Contacts are keyed on immutable numeric Telegram user IDs (`platform_user_id`) so a reassigned `@username` never inherits an existing contact's autonomy mode, tier, chat history, or scoped notes.

### 2. Per-Contact Autonomy, State Machine & Rate Enforcement
- **Autonomy Modes (`off` / `shadow` / `draft` / `auto`)**:
  - `off` *(default for all new contacts — allowlist by default)*: Ignores messages completely.
  - `shadow`: Generates drafts and logs them (`status='shadow'`) without sending.
  - `draft`: Generates **3 candidate reply cards** per message and holds them in the Approval Inbox.
  - `auto`: Generates **1 focused reply card** (when the LLM succeeds) and holds it in a **10-second Delay-and-Cancel** queue (`pending_send`) before sending. If the local LLM is missing or errors, `auto` mode safely holds the message as a draft instead of auto-sending canned fallback text.
- **Guarded Draft State Machine & Bounded Send Retries**: Enforces strict state transitions (`pending_approval`, `pending_send`, `approved`, `sending`, `sent`, `failed`, `rejected`, `shadow`, `cancelled`, `deleted_sent`). Approved drafts are atomically claimed (`sending`) before Telegram delivery and capped at 3 send attempts so failed deliveries never loop infinitely or double-send.
- **Enqueue-Time Rate Cap, Cooldown & Kill Switch**: Enforces the 60s per-contact cooldown and 5/hr auto-reply cap at draft creation time (blocking rapid message bursts), and pauses auto-replies for 2 hours when you manually type in a chat (with an in-flight guard so EchoText's own outgoing replies never trip the kill switch).
- **Single-Writer API + Durable Outbox**: Dashboard actions route through the Brain REST API and SQLite `outbox_events` / `app_settings` tables so deleting sent replies from Telegram, pushing Catch-Up Digests, persisting the sensitivity threshold slider, and erasing contact notes on "Forget Contact" work reliably across processes.

### 3. Multilingual Sensitivity, Prompt Defense & Scoped BM25 RAG
- **Multilingual Sensitivity Gate (0–3) & Fail-Safe**: Detects passwords, OTPs, PINs, CVVs, card numbers, account numbers, Aadhaar/PAN, and 6+ digit sequences across English, Roman Hinglish, and Devanagari Hindi (`ओटीपी`, `पासवर्ड`, `कार्ड नंबर`, `खाता संख्या`), holding sensitive messages as drafts.
- **Prompt-Injection Defense**: Sanitizes sender display names and persona fields, passes earlier chat history as separate `user`/`assistant` turns, wraps retrieved notes inside `<reference_notes readonly="true">`, and never embeds raw RAG note chunks into canned fallback templates.
- **Scoped Okapi BM25 Notes & Expiring Facts**: Accepts `.txt` and `.pdf` uploads (up to 5 MB), scopes notes per contact or relationship tier, and automatically purges expired temporary status facts.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

1. Set `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and a strong `ECHOTEXT_DASHBOARD_PASSWORD` (or `ECHOTEXT_DASHBOARD_PASSWORD_HASH`) in `.env`.
2. Place your GGUF model in `models/Llama-3.2-3B-Instruct.Q4_K_M.gguf` (or `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` for `pc_8b`).
3. Start and stop services:
```bash
python start_echotext.py
python stop_echotext.py
```

## Quality & Verification Gates

```bash
pip install -r requirements-dev.txt
ruff check .
bandit -r config.py auth.py database.py rag.py intelligence.py voice_transcriber.py server.py telegram_listener.py phone_bot.py app.py start_echotext.py stop_echotext.py -ll
mypy config.py auth.py database.py rag.py intelligence.py voice_transcriber.py server.py telegram_listener.py phone_bot.py start_echotext.py stop_echotext.py
pytest -v
```

## Minimal Repository Structure

| File | Responsibility |
| :--- | :--- |
| `config.py` | Absolute paths, mandatory API key provisioning, hardware profiles, log redaction, fail-closed `BUSY` status gate |
| `auth.py` | `PBKDF2-HMAC-SHA256` dashboard authentication, global cross-session brute-force lockout, inactivity auto-logout |
| `database.py` | SQLite schema & migrations, dynamic CHECK constraints, immutable `platform_user_id`, guarded draft state machine, enqueue rate limiting, durable `outbox_events` & `app_settings` |
| `rag.py` | Upload hardening (`.txt`/`.pdf`, 5 MB cap, path sanitization), `pypdf` extraction, chunking, scoped notes, Okapi BM25 |
| `intelligence.py` | English/Hinglish/Devanagari sensitivity scoring (0–3), display-name sanitization, token-budgeted multi-turn prompt builder, safe fallbacks, gibberish/uncertainty guard |
| `voice_transcriber.py` | Local offline voice-note transcription via `faster-whisper` or `whisper.cpp` |
| `server.py` | FastAPI Brain API with mandatory `X-API-Key`, `Host` allowlist, decomposed chat pipeline, and single-writer CRUD endpoints |
| `telegram_listener.py` | Telethon userbot listener with immutable user IDs, atomic send claim handling, bounded retry reporting, and outgoing race guard |
| `phone_bot.py` | Optional private Telegram approval bot with Markdown escaping and plain-text fallback |
| `app.py` | Localhost Streamlit Control Center dashboard wired to the Brain API and persistent settings |
| `start_echotext.py` / `stop_echotext.py` | Ordered process launcher and clean shutdown stopper |
| `tests/test_echotext.py` | End-to-end unit and integration test suite |
