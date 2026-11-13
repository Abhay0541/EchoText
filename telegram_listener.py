"""Telethon Userbot Listener for EchoText.

Key safety and reliability guarantees:
- Does NOT connect or start the Telegram client at module import time (testable in isolation).
- Validates `.env` variables upfront with clear human-readable errors.
- Checks ``event.is_private`` FIRST before calling ``await event.get_sender()``.
- Extracts quoted "swipe-to-reply" message context via ``await event.get_reply_message()``
  so vague replies to older messages include the exact quoted text.
- Marks chats as read via ``send_read_acknowledge`` and dispatches multi-part replies
  ("first half, second half") sequentially with natural typing simulation.
- Guards against outgoing-message kill-switch race conditions and reports failed sends
  to ``/drafts/{id}/mark_failed``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import httpx

from config import (
    API_SECRET_KEY,
    SERVER_URL,
    SESSION_PATH,
    escape_markdown,
    redact_text,
    setup_logging,
    validate_env,
)
from intelligence import split_reply_into_parts
from voice_transcriber import transcribe_voice_note

logger = setup_logging("echotext.telegram")

_ECHOTEXT_SENT_MSG_IDS: set[int] = set()
_IN_FLIGHT_SEND_UNTIL: dict[str, float] = {}

HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)


def _api_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if API_SECRET_KEY:
        headers["X-API-Key"] = API_SECRET_KEY
    return headers


def should_ignore_incoming_event(
    *,
    is_private: bool,
    is_outgoing: bool,
    sender_is_bot: bool,
    sender_is_self: bool = False,
    raw_text: str = "",
    has_sticker: bool = False,
    has_voice: bool = False,
    has_other_media: bool = False,
) -> tuple[bool, str, str | None]:
    """Pure, testable filter deciding whether an incoming Telegram event should be ignored."""
    if not is_private:
        return True, "not_private_chat", None
    if is_outgoing or sender_is_self:
        return True, "outgoing_or_self_message", None
    if sender_is_bot:
        return True, "sender_is_bot", None

    cleaned = (raw_text or "").strip()
    if has_sticker:
        return False, "sticker_media", "sticker"
    if has_voice:
        return False, "voice_media", "voice"
    if not cleaned:
        if has_other_media:
            return True, "unsupported_empty_media", None
        return True, "empty_message_text", None

    return False, "ok", None


async def _extract_quoted_reply_prefix(event: Any) -> str:
    """If the incoming event is a swipe-to-reply, fetch and format the quoted message."""
    if not bool(getattr(event, "is_reply", False)):
        return ""
    get_reply_fn = getattr(event, "get_reply_message", None)
    if not callable(get_reply_fn):
        return ""
    try:
        quoted = await get_reply_fn()
        if quoted is None:
            return ""
        q_text = str(
            getattr(quoted, "raw_text", None) or getattr(quoted, "message", "") or ""
        ).strip()
        if not q_text:
            return ""
        trimmed_quote = q_text.replace("\n", " ")[:220]
        who = (
            "your earlier message"
            if bool(getattr(quoted, "out", False))
            else "their earlier message"
        )
        return f'[Replying to {who}: "{trimmed_quote}"]\n'
    except Exception as exc:
        logger.debug("Could not fetch quoted reply message: %s", exc)
        return ""


async def handle_incoming_telegram_event(
    event: Any,
    http_client: httpx.AsyncClient,
    server_url: str = SERVER_URL,
) -> dict[str, Any] | None:
    """Process a single incoming Telethon ``NewMessage`` event safely and asynchronously."""
    try:
        if not bool(getattr(event, "is_private", False)):
            return None
        if bool(getattr(event, "out", False)):
            return None

        sender = await event.get_sender()
        if sender is None:
            return None
        if bool(getattr(sender, "bot", False)) or bool(getattr(sender, "is_self", False)):
            return None

        raw_text = str(getattr(event, "raw_text", "") or "").strip()
        has_sticker = bool(getattr(event.message, "sticker", None))
        has_voice = bool(getattr(event.message, "voice", None))
        has_media = bool(getattr(event.message, "media", None))

        ignore, reason, media_type = should_ignore_incoming_event(
            is_private=True,
            is_outgoing=False,
            sender_is_bot=False,
            raw_text=raw_text,
            has_sticker=has_sticker,
            has_voice=has_voice,
            has_other_media=has_media,
        )
        if ignore:
            logger.debug("Ignored incoming event: %s", reason)
            return None

        if media_type == "voice" and not raw_text:
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_f:
                    tmp_path = Path(tmp_f.name)
                await event.message.download_media(file=str(tmp_path))
                transcribed = await asyncio.to_thread(transcribe_voice_note, tmp_path)
                if transcribed:
                    raw_text = f"[Voice note transcript]: {transcribed}"
                    media_type = None
            except Exception:
                logger.exception(
                    "Voice-note transcription failed; falling back to canned reply."
                )
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

        # Prepend quoted swipe-to-reply context if the sender replied to a specific message
        if raw_text:
            quote_prefix = await _extract_quoted_reply_prefix(event)
            if quote_prefix:
                raw_text = f"{quote_prefix}{raw_text}"

        sender_id_raw = getattr(sender, "id", None)
        platform_user_id = str(sender_id_raw) if sender_id_raw is not None else None
        sender_handle = (
            getattr(sender, "username", None)
            or getattr(sender, "phone", None)
            or str(sender_id_raw or "unknown")
        )
        first_name = str(getattr(sender, "first_name", "") or "").strip()
        last_name = str(getattr(sender, "last_name", "") or "").strip()
        display_name = f"{first_name} {last_name}".strip() or str(sender_handle)

        payload = {
            "sender_handle": str(sender_handle).lower(),
            "platform_user_id": platform_user_id,
            "display_name": display_name,
            "message": raw_text,
            "app_source": "telegram",
            "incoming_msg_id": getattr(event.message, "id", None),
            "media_type": media_type,
        }

        resp = await http_client.post(
            f"{server_url.rstrip('/')}/chat",
            json=payload,
            headers=_api_headers(),
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data
    except httpx.TimeoutException:
        logger.error("Timeout calling Brain server at %s/chat", server_url)
        return None
    except Exception:
        logger.exception("Error handling incoming Telegram message")
        return None


async def handle_outgoing_manual_event(
    event: Any,
    http_client: httpx.AsyncClient,
    server_url: str = SERVER_URL,
) -> None:
    """Pause auto-replies (Kill Switch) when the owner manually types in a private chat."""
    try:
        if not bool(getattr(event, "is_private", False)):
            return
        msg_id = int(getattr(event.message, "id", 0) or 0)
        if msg_id in _ECHOTEXT_SENT_MSG_IDS:
            _ECHOTEXT_SENT_MSG_IDS.discard(msg_id)
            return

        chat = await event.get_chat()
        if (
            chat is None
            or bool(getattr(chat, "bot", False))
            or bool(getattr(chat, "is_self", False))
        ):
            return

        handle = (
            getattr(chat, "username", None)
            or getattr(chat, "phone", None)
            or str(getattr(chat, "id", ""))
        )
        if not handle:
            return

        norm_handle = str(handle).lower().lstrip("@")
        in_flight_until = _IN_FLIGHT_SEND_UNTIL.get(norm_handle, 0.0)
        if time.monotonic() < in_flight_until:
            return

        logger.info("Manual outgoing message detected to %s — triggering Kill Switch.", handle)
        await http_client.post(
            f"{server_url.rstrip('/')}/kill-switch",
            json={"sender_handle": norm_handle, "app_source": "telegram"},
            headers=_api_headers(),
        )
    except Exception:
        logger.exception("Failed to process outgoing manual message for kill switch")


def _format_owner_notification(note: dict[str, Any]) -> str:
    """Format an owner notification with escaped Markdown characters."""
    ntype = note.get("type")
    disp = escape_markdown(str(note.get("display_name") or ""))
    handle = escape_markdown(str(note.get("contact_handle") or ""))
    incoming = escape_markdown(str(note.get("incoming_text") or ""))

    if ntype == "urgent_bypass":
        return (
            f"🚨 **EchoText Urgent Bypass**\n"
            f"From: {disp} (@{handle})\n"
            f"Message: {incoming}\n"
            f"*(Contact flagged; no auto-reply sent. Confirm in your Approval Inbox.)*"
        )
    if ntype == "sensitive_hold":
        reasons = escape_markdown(", ".join(note.get("reasons") or []))
        return (
            f"🔒 **EchoText Sensitive Message Held "
            f"(Score {note.get('sensitivity_score')}/3)**\n"
            f"From: {disp} (@{handle})\n"
            f"Reasons: {reasons}\n"
            f"*(Auto-reply blocked; saved to your Approval Inbox.)*"
        )
    if ntype == "ai_unsure_hold":
        reason = escape_markdown(str(note.get("uncertainty_reason") or ""))
        return (
            f"🟡 **EchoText Needs Context / Owner Reply — Message Held**\n"
            f"From: {disp} (@{handle})\n"
            f"Message: {incoming}\n"
            f"Alert: {reason}\n"
            f"👉 *Please reply manually in this chat OR open EchoText Dashboard → Message Approvals to give context and regenerate a reply.*"
        )
    if ntype == "model_unavailable_hold":
        err = escape_markdown(str(note.get("error") or "Model unavailable"))
        return (
            f"⚠️ **EchoText Held Reply — Local Model Unavailable**\n"
            f"From: {disp} (@{handle})\n"
            f"Error: {err}\n"
            f"*(Auto-reply held as draft instead of sending canned text.)*"
        )
    if ntype == "catchup_digest":
        return str(note.get("digest_text") or "")
    return ""


async def _send_saved_messages_note(tg_client: Any, msg_text: str) -> None:
    """Send a notification to Telegram Saved Messages ('me') and record its ID."""
    if not msg_text.strip():
        return
    sent_me = await tg_client.send_message("me", msg_text)
    if getattr(sent_me, "id", None):
        _ECHOTEXT_SENT_MSG_IDS.add(int(sent_me.id))


async def _dispatch_single_draft(
    draft: dict[str, Any],
    tg_client: Any,
    http_client: httpx.AsyncClient,
    server_url: str,
) -> None:
    """Send a claimed draft on Telegram (splitting into Part 1 / Part 2 if long) and mark read."""
    if draft.get("platform") != "telegram":
        return
    target_handle = str(draft.get("canonical_name") or "").strip()
    reply_text = str(draft.get("draft_text") or "").strip()
    draft_id = int(draft["id"])
    if not target_handle or not reply_text:
        return

    raw_parts: list[str] = draft.get("reply_parts") or split_reply_into_parts(reply_text)
    parts = [p.strip() for p in raw_parts if p and p.strip()] or [reply_text]

    norm_handle = target_handle.lower().lstrip("@")
    try:
        entity_target: int | str = (
            int(target_handle) if target_handle.isdigit() else target_handle
        )

        # Mark incoming messages in this chat as read on Telegram (double ticks)
        read_ack_fn = getattr(tg_client, "send_read_acknowledge", None)
        if callable(read_ack_fn):
            try:
                await read_ack_fn(entity_target)
            except Exception as exc:
                logger.debug("send_read_acknowledge skipped: %s", exc)

        sent_ids: list[int] = []
        for idx, part_text in enumerate(parts):
            delay = min(
                2.5,
                max(0.6, len(part_text) * 0.025 + random.uniform(0.15, 0.6)),  # nosec B311  # noqa: S311
            )
            _IN_FLIGHT_SEND_UNTIL[norm_handle] = time.monotonic() + 12.0
            async with tg_client.action(entity_target, "typing"):
                await asyncio.sleep(delay)

            sent_msg = await tg_client.send_message(entity_target, part_text)
            sent_msg_id = int(getattr(sent_msg, "id", 0) or 0)
            if sent_msg_id:
                _ECHOTEXT_SENT_MSG_IDS.add(sent_msg_id)
                sent_ids.append(sent_msg_id)
            if idx < len(parts) - 1:
                try:
                    from database import get_setting
                    gap = float(get_setting("multipart_gap_seconds", "0.35"))
                except Exception:
                    gap = 0.35
                await asyncio.sleep(gap)

        primary_id = sent_ids[0] if sent_ids else None
        await http_client.post(
            f"{server_url.rstrip('/')}/drafts/{draft_id}/mark_sent",
            params={"sent_telegram_msg_id": primary_id},
            json={"sent_telegram_msg_id": primary_id, "sent_msg_ids": sent_ids},
            headers=_api_headers(),
        )
        logger.info(
            "Sent approved draft #%d (%d part(s)) to %s: %s",
            draft_id,
            len(parts),
            target_handle,
            redact_text(reply_text),
        )
    except Exception as exc:
        logger.exception("Failed to send draft #%d to %s", draft_id, target_handle)
        try:
            await http_client.post(
                f"{server_url.rstrip('/')}/drafts/{draft_id}/mark_failed",
                json={"error_message": str(exc)},
                headers=_api_headers(),
            )
        except Exception:
            logger.exception("Failed to report mark_failed for draft #%d", draft_id)
    finally:
        _IN_FLIGHT_SEND_UNTIL.pop(norm_handle, None)


async def poll_and_dispatch_loop(
    tg_client: Any,
    http_client: httpx.AsyncClient,
    server_url: str = SERVER_URL,
    poll_interval: float = 2.0,
) -> None:
    """Background task that sends approved drafts, notifications, and message deletions."""
    while True:
        try:
            resp = await http_client.get(
                f"{server_url.rstrip('/')}/listener/poll",
                headers=_api_headers(),
            )
            if resp.status_code == 200:
                payload = resp.json()

                for note in payload.get("notifications", []):
                    msg_text = _format_owner_notification(note)
                    if msg_text:
                        await _send_saved_messages_note(tg_client, msg_text)

                for draft in payload.get("approved_drafts", []):
                    await _dispatch_single_draft(draft, tg_client, http_client, server_url)

                for del_req in payload.get("delete_requests", []):
                    target_handle = del_req.get("contact_handle")
                    ids_to_del = del_req.get("sent_msg_ids") or [
                        del_req.get("sent_telegram_msg_id")
                    ]
                    clean_ids = [int(x) for x in ids_to_del if x]
                    if target_handle and clean_ids:
                        try:
                            entity_target = (
                                int(target_handle)
                                if str(target_handle).isdigit()
                                else str(target_handle)
                            )
                            await tg_client.delete_messages(entity_target, clean_ids)
                            logger.info(
                                "Deleted sent Telegram message id(s)=%s in chat %s",
                                clean_ids,
                                target_handle,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to delete Telegram message id(s)=%s in chat %s",
                                clean_ids,
                                target_handle,
                            )
        except Exception:
            logger.exception("Error in listener poll_and_dispatch_loop")

        try:
            from database import get_setting
            dynamic_interval = float(get_setting("poll_interval_seconds", str(poll_interval)))
        except Exception:
            dynamic_interval = poll_interval
        await asyncio.sleep(dynamic_interval)


async def run_listener() -> None:
    """Validate `.env`, connect the Telethon client, and start listening + polling."""
    import os

    errors = validate_env(require_telegram=True)
    fatal_errors = [e for e in errors if not e.startswith("WARNING:")]
    for warn in [e for e in errors if e.startswith("WARNING:")]:
        logger.warning(warn)

    if fatal_errors:
        for err in fatal_errors:
            logger.error(err)
            print(f"[EchoText Config Error] {err}")
        raise SystemExit(1)

    from telethon import TelegramClient, events  # type: ignore[import-not-found]

    api_id = int(os.environ["TELEGRAM_API_ID"].strip())
    api_hash = os.environ["TELEGRAM_API_HASH"].strip()

    tg_client = TelegramClient(str(SESSION_PATH), api_id, api_hash)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as http_client:
        @tg_client.on(events.NewMessage(incoming=True))
        async def _on_incoming(event: Any) -> None:
            await handle_incoming_telegram_event(event, http_client, SERVER_URL)

        @tg_client.on(events.NewMessage(outgoing=True))
        async def _on_outgoing(event: Any) -> None:
            await handle_outgoing_manual_event(event, http_client, SERVER_URL)

        await tg_client.start()
        logger.info("Telegram listener started using session %s", SESSION_PATH)
        poll_task = asyncio.create_task(
            poll_and_dispatch_loop(tg_client, http_client, SERVER_URL)
        )
        try:
            await tg_client.run_until_disconnected()
        finally:
            poll_task.cancel()


if __name__ == "__main__":
    asyncio.run(run_listener())
