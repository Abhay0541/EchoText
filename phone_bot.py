"""Private Telegram Approval Bot for Phone Control over Tailscale / Localhost.

Sends each newly created ``pending_approval`` draft to the owner's private Telegram
account with inline callback buttons:
- ``[✅ Send #1]`` ``[✅ Send #2]`` ``[✅ Send #3]``
- ``[❌ Reject]``

Includes Markdown escaping and automatic plain-text fallback so special characters in
incoming messages never cause Telegram Bot API parse errors or dropped notifications.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from config import API_SECRET_KEY, SERVER_URL, escape_markdown, setup_logging, validate_env

logger = setup_logging("echotext.phone_bot")

HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_SECRET_KEY:
        h["X-API-Key"] = API_SECRET_KEY
    return h


def build_draft_notification_card(
    draft: dict[str, Any],
) -> tuple[str, list[list[dict[str, str]]]]:
    """Format a draft into a Markdown-safe Telegram message + inline keyboard markup."""
    draft_id = int(draft["id"])
    sender = escape_markdown(
        str(draft.get("display_name") or draft.get("canonical_name") or "Unknown")
    )
    handle = escape_markdown(str(draft.get("canonical_name") or ""))
    platform = escape_markdown(str(draft.get("platform") or "telegram"))
    incoming = escape_markdown(str(draft.get("incoming_text") or ""))
    created = str(draft.get("created_at") or "")
    sens = draft.get("sensitivity_score", 0)
    triage = str(draft.get("triage_label") or "chitchat")
    options: list[str] = draft.get("options") or [draft.get("draft_text", "")]
    chunks: list[dict[str, Any]] = draft.get("chunks") or []

    chunk_ids = (
        ", ".join(str(c.get("chunk_id", "")) for c in chunks) if chunks else "None"
    )

    lines = [
        f"📬 *New Draft #{draft_id}* ({platform})",
        f"👤 *From:* {sender} (`@{handle}`)",
        f"🕒 *Received:* `{created}`",
        f"🏷 *Triage:* `{triage}` | 🔒 *Sensitivity:* `{sens}/3`",
        f"📚 *RAG Chunks:* `{chunk_ids}`",
        "",
        f"💬 *Incoming:*\n{incoming}",
        "",
        "🤖 *Candidate Replies:*",
    ]
    for idx, opt in enumerate(options):
        lines.append(f"*[{idx + 1}]* {escape_markdown(str(opt))}")

    lines.append(
        f"\n⚠️ _Tap a card below to approve & send, or `/edit {draft_id} <text>`._"
    )

    approve_row = [
        {"text": f"✅ Send #{idx + 1}", "callback_data": f"approve:{draft_id}:{idx}"}
        for idx in range(len(options))
    ]
    action_row = [
        {"text": "❌ Reject", "callback_data": f"reject:{draft_id}"},
    ]
    return "\n".join(lines), [approve_row, action_row]


async def send_telegram_bot_message(
    http_client: httpx.AsyncClient,
    tg_base: str,
    chat_id: int,
    text: str,
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Send a Bot API message with automatic plain-text fallback if Markdown fails."""
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    resp = await http_client.post(f"{tg_base}/sendMessage", json=payload)
    if resp.status_code == 200:
        return True

    # Fallback without parse_mode so a formatting issue never drops a notification
    payload.pop("parse_mode", None)
    fallback_resp = await http_client.post(f"{tg_base}/sendMessage", json=payload)
    return fallback_resp.status_code == 200


async def parse_callback_action(
    callback_data: str,
    http_client: httpx.AsyncClient,
    server_url: str = SERVER_URL,
) -> str:
    """Execute a button press callback against the Brain server Drafts API."""
    parts = callback_data.strip().split(":")
    action = parts[0] if parts else ""

    try:
        if action == "approve" and len(parts) >= 3:
            draft_id = int(parts[1])
            opt_idx = int(parts[2])
            resp = await http_client.post(
                f"{server_url.rstrip('/')}/drafts/{draft_id}/approve",
                json={"option_index": opt_idx},
                headers=_headers(),
            )
            if resp.status_code == 409:
                return f"ℹ️ Draft #{draft_id} was already processed."
            resp.raise_for_status()
            return f"✅ Approved Draft #{draft_id} (Card #{opt_idx + 1}). Queued for sending!"

        if action == "reject" and len(parts) >= 2:
            draft_id = int(parts[1])
            resp = await http_client.post(
                f"{server_url.rstrip('/')}/drafts/{draft_id}/reject",
                headers=_headers(),
            )
            if resp.status_code == 409:
                return f"ℹ️ Draft #{draft_id} was already processed."
            resp.raise_for_status()
            return f"❌ Rejected Draft #{draft_id}."
    except httpx.HTTPError as exc:
        logger.warning("Callback action '%s' failed: %s", callback_data, exc)
        return f"⚠️ Action failed: {exc}"

    return "Unknown callback action."


async def run_phone_approval_bot() -> None:
    """Poll Telegram Bot API for owner commands/callbacks and push new drafts."""
    errors = validate_env(require_phone_bot=True)
    fatal = [e for e in errors if not e.startswith("WARNING:")]
    if fatal:
        for err in fatal:
            logger.error(err)
            print(f"[Phone Bot Config Error] {err}")
        raise SystemExit(1)

    bot_token = os.environ["ECHOTEXT_APPROVAL_BOT_TOKEN"].strip()
    owner_id = int(os.environ["ECHOTEXT_OWNER_TELEGRAM_ID"].strip())
    tg_base = f"https://api.telegram.org/bot{bot_token}"

    notified_draft_ids: set[int] = set()
    last_update_id = 0

    logger.info("Starting private Telegram phone approval bot for owner_id=%d", owner_id)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        while True:
            try:
                d_resp = await client.get(
                    f"{SERVER_URL.rstrip('/')}/drafts",
                    params={"status": "pending_approval", "limit": 20},
                    headers=_headers(),
                )
                if d_resp.status_code == 200:
                    for draft in reversed(d_resp.json().get("drafts", [])):
                        did = int(draft["id"])
                        if did in notified_draft_ids:
                            continue
                        text, inline_kb = build_draft_notification_card(draft)
                        ok = await send_telegram_bot_message(
                            client,
                            tg_base,
                            owner_id,
                            text,
                            reply_markup={"inline_keyboard": inline_kb},
                        )
                        if ok:
                            notified_draft_ids.add(did)

                u_resp = await client.get(
                    f"{tg_base}/getUpdates",
                    params={"offset": last_update_id + 1, "timeout": 5},
                )
                if u_resp.status_code == 200:
                    for upd in u_resp.json().get("result", []):
                        last_update_id = int(upd["update_id"])

                        cb = upd.get("callback_query")
                        if cb:
                            from_id = int((cb.get("from") or {}).get("id", 0))
                            if from_id != owner_id:
                                continue
                            outcome_msg = await parse_callback_action(
                                str(cb.get("data") or ""), client, SERVER_URL
                            )
                            await client.post(
                                f"{tg_base}/answerCallbackQuery",
                                json={"callback_query_id": cb["id"], "text": outcome_msg},
                            )
                            await client.post(
                                f"{tg_base}/sendMessage",
                                json={"chat_id": owner_id, "text": outcome_msg},
                            )
                            continue

                        msg = upd.get("message") or {}
                        from_id = int((msg.get("from") or {}).get("id", 0))
                        if from_id != owner_id:
                            continue
                        text_cmd = str(msg.get("text") or "").strip()

                        if text_cmd.startswith("/busy"):
                            await client.post(
                                f"{SERVER_URL.rstrip('/')}/status",
                                json={"status": "BUSY"},
                                headers=_headers(),
                            )
                            await client.post(
                                f"{tg_base}/sendMessage",
                                json={
                                    "chat_id": owner_id,
                                    "text": "🔴 EchoText status set to BUSY.",
                                },
                            )
                        elif text_cmd.startswith("/available") or text_cmd.startswith("/pause"):
                            s_res = await client.post(
                                f"{SERVER_URL.rstrip('/')}/status",
                                json={"status": "AVAILABLE"},
                                headers=_headers(),
                            )
                            digest = s_res.json().get("catchup_digest") or ""
                            reply_txt = (
                                f"🟢 EchoText status set to AVAILABLE.\n\n{digest}".strip()
                            )
                            await client.post(
                                f"{tg_base}/sendMessage",
                                json={"chat_id": owner_id, "text": reply_txt},
                            )
                        elif text_cmd.startswith("/status"):
                            st_data = (
                                await client.get(
                                    f"{SERVER_URL.rstrip('/')}/status", headers=_headers()
                                )
                            ).json()
                            await client.post(
                                f"{tg_base}/sendMessage",
                                json={
                                    "chat_id": owner_id,
                                    "text": (
                                        f"ℹ️ Status: {st_data.get('proxy_status')}\n"
                                        f"Active hours: {st_data.get('active_hours_window')}"
                                    ),
                                },
                            )
                        elif text_cmd.startswith("/edit "):
                            parts = text_cmd.split(maxsplit=2)
                            if len(parts) == 3 and parts[1].isdigit():
                                did = int(parts[1])
                                edited = parts[2].strip()
                                await client.post(
                                    f"{SERVER_URL.rstrip('/')}/drafts/{did}/edit",
                                    json={"new_text": edited},
                                    headers=_headers(),
                                )
                                await client.post(
                                    f"{tg_base}/sendMessage",
                                    json={
                                        "chat_id": owner_id,
                                        "text": f"✏️ Edited & approved Draft #{did}!",
                                    },
                                )
            except Exception:
                logger.exception("Error in phone approval bot loop")

            await asyncio.sleep(2.0)


if __name__ == "__main__":
    asyncio.run(run_phone_approval_bot())
