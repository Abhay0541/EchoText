"""
Gmail integration.

    pip install google-api-python-client google-auth-oauthlib
    Put your OAuth `credentials.json` (from Google Cloud Console) at the path
    in GMAIL_CREDENTIALS_PATH, then:
    python -m app.integrations.gmail_client --session_id <id>

Deliberate difference from Telegram/Discord: this NEVER auto-sends. Email is
higher-stakes and lower-frequency than chat, so the right default is
"create a draft the user reviews and hits send on" rather than full
auto-reply. This is a genuine design decision, not a missing feature —
the router's route/reasoning are still fully computed and shown, just gated
one step further before anything leaves your account.
"""
from __future__ import annotations

import argparse
import base64
import logging
import time
from email.mime.text import MIMEText

from app import config
from app.main import SESSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echotext.gmail")

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def _get_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    try:
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    except FileNotFoundError:
        pass

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(config.GMAIL_CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _create_draft(service, to_addr: str, subject: str, body: str, thread_id: str):
    message = MIMEText(body)
    message["to"] = to_addr
    message["subject"] = f"Re: {subject}"
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().drafts().create(
        userId="me",
        body={"message": {"raw": raw, "threadId": thread_id}},
    ).execute()


def poll_loop(session_id: str, poll_interval_seconds: int = 60):
    service = _get_service()
    seen_ids: set[str] = set()

    logger.info("EchoText Gmail watcher started (polling every %ss)", poll_interval_seconds)
    while True:
        session = SESSIONS.get(session_id)
        if session is None:
            time.sleep(poll_interval_seconds)
            continue

        results = service.users().messages().list(userId="me", labelIds=["UNREAD", "INBOX"]).execute()
        for msg_meta in results.get("messages", []):
            if msg_meta["id"] in seen_ids:
                continue
            seen_ids.add(msg_meta["id"])

            msg = service.users().messages().get(userId="me", id=msg_meta["id"], format="full").execute()
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            snippet = msg.get("snippet", "")

            examples = session["index"].query(snippet, top_k=config.RAG_TOP_K)
            decision = session["router"].handle(snippet, examples)

            if decision.reply is not None:
                _create_draft(
                    service,
                    to_addr=headers.get("From", ""),
                    subject=headers.get("Subject", ""),
                    body=decision.reply,
                    thread_id=msg["threadId"],
                )
                logger.info("Draft created (route=%s) for subject: %s", decision.route.value, headers.get("Subject"))
            else:
                logger.info("Held back (route=%s): %s", decision.route.value, headers.get("Subject"))

        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--poll_interval", type=int, default=60)
    args = parser.parse_args()
    poll_loop(args.session_id, args.poll_interval)
