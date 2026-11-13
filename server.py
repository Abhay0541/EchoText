"""FastAPI Brain Server for EchoText.

Key architectural & security guarantees:
- Mandatory ``X-API-Key`` header authentication (auto-provisioned in ``config.py``) and
  strict ``Host`` header allowlist middleware to block cross-origin / DNS-rebinding abuse.
- In-flight per-contact message coalescing: if a sender sends additional messages while the
  model is generating a reply, those messages are merged into the active batch and the model
  re-generates a unified reply (using partial-batch map-reduce + grammar/coherence synthesis
  when many messages arrive).
- Multi-part reply splitting ("first half, second half") for long replies.
- Per-contact ambiguity flagging: ambiguous messages or replies hold for review, notify the
  owner, and flag the contact so no auto-replies are sent until the owner reviews, edits/sends,
  or deletes the held message.
"""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hmac
from pathlib import Path
import threading
from typing import Any, AsyncIterator, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import (
    ACTIVE_HOURS_ENABLED,
    ACTIVE_HOURS_END,
    ACTIVE_HOURS_START,
    ACTIVE_PROFILE,
    ALLOWED_HOSTS,
    API_SECRET_KEY,
    AUTO_SEND_DELAY_SECONDS,
    CONTEXT_WINDOW,
    DB_PATH,
    DEFAULT_HISTORY_LIMIT,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    NUM_DRAFT_OPTIONS,
    OWNER_NAME,
    RAG_DIR,
    STATUS_FILE_PATH,
    get_proxy_status,
    redact_text,
    set_proxy_status,
    setup_logging,
)
from database import (
    add_injected_fact,
    approve_draft,
    cancel_pending_auto_send,
    check_rate_limit_and_cooldown,
    claim_approved_drafts_for_sending,
    clear_contact_review_flag,
    clear_kill_switch,
    create_draft,
    delete_injected_fact,
    edit_and_approve_draft,
    enqueue_outbox_event,
    flag_contact_for_review,
    forget_contact,
    get_active_injected_facts,
    get_all_system_settings,
    get_chat_history,
    get_draft_by_id,
    get_or_create_contact,
    get_sensitivity_threshold,
    init_db,
    link_contact_handle,
    list_audit_logs,
    list_contacts,
    list_drafts,
    list_ui_notifications,
    log_audit_event,
    mark_draft_failed,
    mark_draft_sent,
    mark_ui_notifications_read,
    pop_outbox_events,
    promote_due_pending_sends,
    purge_old_records,
    reject_draft,
    request_delete_sent_reply,
    reset_system_settings_to_defaults,
    save_message,
    save_system_settings,
    set_sensitivity_threshold,
    trigger_kill_switch,
    update_contact_settings,
    update_draft_options,
)
from intelligence import (
    CANNED_FALLBACKS,
    ReplyGenerationResult,
    TriageLabel,
    build_Safe_chat_messages,
    build_unsure_holding_cards,
    check_ai_uncertainty,
    classify_triage,
    combine_and_refine_partial_replies,
    format_batch_group_prompt,
    format_reply_with_footer as append_footer,
    generate_catchup_digest,
    generate_reply_cards_detailed,
    is_urgent_message,
    is_within_active_hours,
    partition_message_batch,
    score_sensitivity,
    split_reply_into_parts,
)
from rag import NoteChunk, retrieve_top_chunks

logger = setup_logging("echotext.server")

_MODEL_LOCK = threading.Lock()
_LLM_INSTANCE: Any = None
_LLM_LOAD_ERROR: str | None = None

_NOTIFICATIONS_LOCK = threading.Lock()
_OWNER_NOTIFICATIONS: deque[dict[str, Any]] = deque(maxlen=200)
_DELETE_MSG_REQUESTS: deque[dict[str, Any]] = deque(maxlen=200)


@dataclass
class _InFlightContactBatch:
    """Tracks messages arriving for a contact while an LLM reply is actively generating."""

    messages: list[str] = field(default_factory=list)
    version: int = 1


_IN_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT_BATCHES: dict[tuple[str, int], _InFlightContactBatch] = {}


def _record_owner_notification(
    payload: dict[str, Any], db_path: Path | None = None
) -> None:
    """Persist an owner notification to both SQLite outbox_events and in-memory queue."""
    with _NOTIFICATIONS_LOCK:
        _OWNER_NOTIFICATIONS.append(payload)
    enqueue_outbox_event("notification", payload, db_path=db_path)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Mandatory local API key authentication using constant-time comparison."""
    if not API_SECRET_KEY or not x_api_key:
        raise HTTPException(
            status_code=401, detail="Missing X-API-Key authentication header."
        )
    if not hmac.compare_digest(x_api_key.strip(), API_SECRET_KEY):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key header.")


def get_llm() -> Any:
    """Lazy-load the Llama GGUF model under a lock on first inference request."""
    global _LLM_INSTANCE, _LLM_LOAD_ERROR
    with _MODEL_LOCK:
        if _LLM_INSTANCE is not None:
            return _LLM_INSTANCE
        if not MODEL_PATH.exists():
            _LLM_LOAD_ERROR = f"Model file not found at {MODEL_PATH}"
            return None
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]

            logger.info("Loading GGUF model from %s ...", MODEL_PATH)
            _LLM_INSTANCE = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=CONTEXT_WINDOW,
                n_threads=int(ACTIVE_PROFILE["n_threads"]),
                n_batch=int(ACTIVE_PROFILE["n_batch"]),
                n_gpu_layers=int(ACTIVE_PROFILE["n_gpu_layers"]),
                verbose=False,
            )
            _LLM_LOAD_ERROR = None
            logger.info("GGUF model loaded successfully.")
            return _LLM_INSTANCE
        except Exception as exc:
            _LLM_LOAD_ERROR = str(exc)
            logger.exception("Failed to load GGUF model from %s", MODEL_PATH)
            return None


def _run_llm_chat(messages: list[dict[str, str]], temperature: float = 0.5) -> str:
    """Execute one serialized chat completion call against the local GGUF model."""
    llm = get_llm()
    if llm is None:
        raise RuntimeError(_LLM_LOAD_ERROR or "LLM is not loaded.")
    with _MODEL_LOCK:
        response: dict[str, Any] = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            stop=["<|eot_id|>", "<|end_of_text|>", "\nUser:", "\nSender:"],
        )
    choices = response.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "").strip()


class ChatRequest(BaseModel):
    """Incoming message payload from ``telegram_listener.py`` or other interceptors."""

    sender_handle: str = Field(..., min_length=1, description="Sender username or handle")
    message: str = Field(default="", description="Incoming message text")
    extra_messages: list[str] | None = Field(
        default=None,
        description="Optional additional messages coalesced while generating",
    )
    app_source: str = Field(default="telegram", description="Platform identifier")
    display_name: str | None = Field(default=None, description="Optional sender display name")
    platform_user_id: str | None = Field(
        default=None, description="Immutable platform user ID (e.g. Telegram numeric ID)"
    )
    incoming_msg_id: int | None = Field(default=None, description="Platform message ID")
    media_type: str | None = Field(
        default=None, description="Optional media type: sticker, voice, photo, etc."
    )


class StatusUpdateRequest(BaseModel):
    status: str = Field(..., description="BUSY or AVAILABLE")


class DraftApproveRequest(BaseModel):
    option_index: int | None = Field(default=None, ge=0)


class DraftEditRequest(BaseModel):
    new_text: str = Field(..., min_length=1)


class DraftRegenerateRequest(BaseModel):
    context_text: str = Field(..., min_length=1)
    save_as_fact: bool = Field(default=True)
    ttl_minutes: int = Field(default=120, ge=5, le=10080)


class DraftMarkSentRequest(BaseModel):
    sent_telegram_msg_id: int | None = Field(default=None)
    sent_msg_ids: list[int] | None = Field(default=None)


class DraftFailRequest(BaseModel):
    error_message: str = Field(default="Telegram delivery failed")


class KillSwitchRequest(BaseModel):
    sender_handle: str = Field(..., min_length=1)
    app_source: str = Field(default="telegram")
    pause_seconds: float | None = Field(default=None, gt=0)
    pause_hours: float | None = Field(default=None, gt=0)


class ContactUpsertRequest(BaseModel):
    handle: str = Field(..., min_length=1)
    app_source: str = Field(default="telegram")
    display_name: str | None = Field(default=None)
    platform_user_id: str | None = Field(default=None)
    mode: str | None = Field(default=None)
    relationship_tier: str | None = Field(default=None)
    tone: str | None = Field(default=None)
    custom_persona: str | None = Field(default=None)
    footer_enabled: bool | None = Field(default=None)
    footer_text: str | None = Field(default=None)


class ContactUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None)
    mode: str | None = Field(default=None)
    relationship_tier: str | None = Field(default=None)
    tone: str | None = Field(default=None)
    custom_persona: str | None = Field(default=None)
    footer_enabled: bool | None = Field(default=None)
    footer_text: str | None = Field(default=None)


class HandleLinkRequest(BaseModel):
    platform: str = Field(..., min_length=1)
    handle: str = Field(..., min_length=1)


class FactCreateRequest(BaseModel):
    fact_text: str = Field(..., min_length=1)
    ttl_minutes: int = Field(default=120, ge=1, le=10080)
    scope_contact_id: int | None = Field(default=None)
    scope_tier: str = Field(default="all_trusted")


class SensitivitySettingRequest(BaseModel):
    sensitivity_threshold: int = Field(..., ge=1, le=3)


class SystemSettingsUpdateRequest(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)


class PurgeRequest(BaseModel):
    retention_days: int = Field(default=30, ge=1, le=365)


class NotificationReadRequest(BaseModel):
    notification_id: int | None = Field(default=None)


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved filesystem paths and clock for a single chat pipeline execution."""

    db_path: Path
    status_file: Path
    rag_dir: Path
    now: datetime | None


@dataclass
class TriageDecision:
    """Intermediate triage and mode decision for an incoming message."""

    sens_score: int
    sens_reasons: list[str]
    triage_label: TriageLabel
    effective_mode: str
    held_for_sensitivity: bool = False
    held_for_contact_flag: bool = False


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize SQLite schema and warm the local GGUF model at server startup."""
    init_db(DB_PATH)
    if MODEL_PATH.exists():
        threading.Thread(
            target=get_llm, name="echotext-model-warmup", daemon=True
        ).start()
    logger.info(
        "EchoText server initialized (DB=%s, StatusFile=%s)", DB_PATH, STATUS_FILE_PATH
    )
    yield


app = FastAPI(
    title="EchoText Brain API",
    version="2.0.0",
    description="Local privacy-first auto-reply and draft approval server.",
    lifespan=lifespan,
)


@app.middleware("http")
async def enforce_host_allowlist(request: Request, call_next: Any) -> Any:
    """Reject requests with non-local Host headers to prevent DNS rebinding attacks."""
    host = (request.headers.get("host") or "").strip().lower()
    if host and host not in ALLOWED_HOSTS:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Disallowed Host header '{host}'."},
        )
    return await call_next(request)


def _check_gate_and_schedule(
    paths: PipelinePaths, active_hours_enabled: bool
) -> dict[str, Any] | None:
    """Stage 1 & 2: Enforce fail-closed BUSY gate and active-hours schedule."""
    current_status = get_proxy_status(paths.status_file)
    if current_status != "BUSY":
        return {
            "status": "skipped",
            "reason": "proxy_not_busy",
            "proxy_status": current_status,
            "reply": None,
        }
    if not is_within_active_hours(
        now=paths.now,
        enabled=active_hours_enabled,
        start_hour=ACTIVE_HOURS_START,
        end_hour=ACTIVE_HOURS_END,
    ):
        return {
            "status": "skipped",
            "reason": "outside_active_hours",
            "reply": None,
        }
    return None


def _resolve_contact_or_skip(
    req: ChatRequest, paths: PipelinePaths
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Stage 3: Look up or create contact and check if their autonomy mode is 'off'."""
    try:
        contact = get_or_create_contact(
            handle=req.sender_handle,
            app_source=req.app_source,
            display_name=req.display_name,
            platform_user_id=req.platform_user_id,
            db_path=paths.db_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    contact_id = int(contact["id"])
    contact_mode = str(contact.get("mode") or "off").lower()
    if contact_mode == "off":
        log_audit_event(
            event_type="message_ignored_mode_off",
            contact_id=contact_id,
            platform=req.app_source,
            mode_used="off",
            outcome="skipped",
            details="Contact autonomy mode is 'off' (allowlist gate).",
            db_path=paths.db_path,
        )
        return None, {
            "status": "skipped",
            "reason": "contact_mode_off",
            "contact_id": contact_id,
            "reply": None,
        }
    return contact, None


def _handle_media_only_request(
    req: ChatRequest, contact: dict[str, Any], paths: PipelinePaths
) -> dict[str, Any]:
    """Stage 4: Handle sticker or untranscribed voice notes without invoking the LLM."""
    contact_id = int(contact["id"])
    contact_mode = str(contact.get("mode") or "off").lower()
    if bool(contact.get("flagged_for_review")):
        contact_mode = "draft"

    canned_key = "media_sticker" if req.media_type == "sticker" else "media_voice_note"
    canned_reply = append_footer(
        CANNED_FALLBACKS[canned_key],
        footer_enabled=bool(contact.get("footer_enabled")),
        footer_text=str(contact.get("footer_text") or ""),
    )
    if contact_mode == "auto":
        allowed, rate_reason = check_rate_limit_and_cooldown(
            contact_id, now=paths.now, db_path=paths.db_path
        )
        if not allowed:
            return {"status": "skipped", "reason": rate_reason, "reply": None}
        save_message(
            contact_id,
            "user",
            f"[{req.media_type}]",
            platform=req.app_source,
            telegram_msg_id=req.incoming_msg_id,
            db_path=paths.db_path,
        )
        draft = create_draft(
            contact_id=contact_id,
            incoming_text=f"[{req.media_type}]",
            draft_options=[canned_reply],
            reply_parts=[canned_reply],
            platform=req.app_source,
            incoming_msg_id=req.incoming_msg_id,
            triage_label="chitchat",
            status="approved",
            now=paths.now,
            db_path=paths.db_path,
        )
        return {
            "status": "auto_canned",
            "reason": canned_key,
            "draft_id": draft["id"],
            "reply": canned_reply,
            "reply_parts": [canned_reply],
        }
    return {
        "status": "skipped",
        "reason": f"media_{req.media_type}_skipped",
        "reply": None,
    }


def _handle_urgent_bypass(
    req: ChatRequest,
    raw_text: str,
    contact: dict[str, Any],
    sens_score: int,
    paths: PipelinePaths,
) -> dict[str, Any]:
    """Stage 6: Hold urgent message as draft, flag contact, and immediately notify owner."""
    contact_id = int(contact["id"])
    contact_mode = str(contact.get("mode") or "off").lower()
    flag_contact_for_review(
        contact_id,
        "Urgent message requires your review and confirmation before AI resumes.",
        db_path=paths.db_path,
    )
    _record_owner_notification(
        {
            "type": "urgent_bypass",
            "severity": "critical",
            "contact_id": contact_id,
            "contact_handle": contact["canonical_name"],
            "display_name": contact["display_name"],
            "platform": req.app_source,
            "incoming_text": raw_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        db_path=paths.db_path,
    )
    save_message(
        contact_id,
        "user",
        raw_text,
        platform=req.app_source,
        telegram_msg_id=req.incoming_msg_id,
        db_path=paths.db_path,
    )
    draft = create_draft(
        contact_id=contact_id,
        incoming_text=raw_text,
        draft_options=[CANNED_FALLBACKS["urgent"]],
        reply_parts=[CANNED_FALLBACKS["urgent"]],
        platform=req.app_source,
        incoming_msg_id=req.incoming_msg_id,
        sensitivity_score=max(2, sens_score),
        triage_label="urgent",
        status="pending_approval",
        now=paths.now,
        db_path=paths.db_path,
    )
    log_audit_event(
        event_type="urgent_bypass_triggered",
        contact_id=contact_id,
        platform=req.app_source,
        mode_used=contact_mode,
        triage_label="urgent",
        sensitivity_score=sens_score,
        outcome="held_and_pinged",
        details="Urgent message bypassed auto-reply, flagged contact, and notified owner.",
        db_path=paths.db_path,
    )
    return {
        "status": "held_urgent",
        "reason": "urgent_bypass",
        "draft_id": draft["id"],
        "sensitivity_score": sens_score,
        "triage_label": "urgent",
        "reply": None,
    }


def _evaluate_sensitivity_and_mode(
    req: ChatRequest,
    raw_text: str,
    contact: dict[str, Any],
    threshold: int,
    paths: PipelinePaths,
) -> TriageDecision:
    """Stage 5 & 7: Compute sensitivity score, triage label, and effective autonomy mode."""
    sens_score, sens_reasons = score_sensitivity(
        raw_text, display_name=req.display_name or str(contact.get("display_name") or "")
    )
    triage_label = classify_triage(raw_text, sensitivity_score=sens_score)
    contact_mode = str(contact.get("mode") or "off").lower()
    effective_mode = contact_mode
    held_for_sensitivity = False
    held_for_contact_flag = False

    if bool(contact.get("flagged_for_review")) and effective_mode == "auto":
        effective_mode = "draft"
        held_for_contact_flag = True

    if sens_score >= threshold and effective_mode == "auto":
        effective_mode = "draft"
        held_for_sensitivity = True
        _record_owner_notification(
            {
                "type": "sensitive_hold",
                "severity": "warning",
                "contact_id": int(contact["id"]),
                "contact_handle": contact["canonical_name"],
                "display_name": contact["display_name"],
                "platform": req.app_source,
                "incoming_text": raw_text,
                "sensitivity_score": sens_score,
                "reasons": sens_reasons,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            db_path=paths.db_path,
        )

    return TriageDecision(
        sens_score=sens_score,
        sens_reasons=sens_reasons,
        triage_label=triage_label,
        effective_mode=effective_mode,
        held_for_sensitivity=held_for_sensitivity,
        held_for_contact_flag=held_for_contact_flag,
    )


def _generate_for_batch_groups(
    batch_messages: list[str],
    *,
    contact: dict[str, Any],
    tier: str,
    owner_name: str = OWNER_NAME,
    history: list[dict[str, Any]],
    top_chunks: list[NoteChunk],
    active_facts: list[dict[str, Any]],
    triage_label: TriageLabel,
    target_num_cards: int,
    active_llm_fn: Callable[[list[dict[str, str]], float], str] | None,
    now: datetime | None,
) -> tuple[ReplyGenerationResult, dict[str, Any]]:
    """Generate reply cards for 1 group or partial halves/thirds with grammar synthesis."""
    groups = partition_message_batch(batch_messages)
    total_groups = max(1, len(groups))

    if total_groups == 1:
        combined_input = format_batch_group_prompt(groups[0], 1, 1)
        prompt_messages, prompt_meta = build_Safe_chat_messages(
            combined_input,
            contact_name=str(contact.get("display_name") or contact["canonical_name"]),
            relationship_tier=tier,
            tone=str(contact.get("tone") or "casual_hinglish"),
            custom_persona=str(contact.get("custom_persona") or ""),
            owner_name=owner_name,
            history=history,
            retrieved_chunks=top_chunks,
            injected_facts=active_facts,
            now=now,
        )
        contact_tone = str(contact.get("tone") or "casual_hinglish")
        gen_res = generate_reply_cards_detailed(
            prompt_messages,
            llm_callable=active_llm_fn,
            num_cards=target_num_cards,
            triage_label=triage_label,
            language_style=prompt_meta["language_style"],
            tone=contact_tone,
            relationship_tier=tier,
            owner_name=owner_name,
            injected_facts=active_facts,
            retrieved_chunks=top_chunks,
        )
        prompt_meta["batch_groups_count"] = 1
        prompt_meta["batch_messages_count"] = len(batch_messages)
        return gen_res, prompt_meta

    # Multi-group partial generation (first half / second half, or first/second/third thirds)
    partial_primary_replies: list[str] = []
    any_llm_used = False
    first_error: str | None = None
    last_meta: dict[str, Any] = {}

    for idx, grp in enumerate(groups, start=1):
        group_prompt = format_batch_group_prompt(grp, idx, total_groups)
        p_msgs, p_meta = build_Safe_chat_messages(
            group_prompt,
            contact_name=str(contact.get("display_name") or contact["canonical_name"]),
            relationship_tier=tier,
            tone=str(contact.get("tone") or "casual_hinglish"),
            custom_persona=str(contact.get("custom_persona") or ""),
            owner_name=owner_name,
            history=history,
            retrieved_chunks=top_chunks,
            injected_facts=active_facts,
            now=now,
        )
        last_meta = p_meta
        contact_tone = str(contact.get("tone") or "casual_hinglish")
        part_res = generate_reply_cards_detailed(
            p_msgs,
            llm_callable=active_llm_fn,
            num_cards=1,
            triage_label=triage_label,
            language_style=p_meta["language_style"],
            tone=contact_tone,
            relationship_tier=tier,
            owner_name=owner_name,
            injected_facts=active_facts,
            retrieved_chunks=top_chunks,
        )
        if part_res.used_llm:
            any_llm_used = True
        if part_res.llm_error and first_error is None:
            first_error = part_res.llm_error
        if part_res.cards:
            partial_primary_replies.append(part_res.cards[0])

    lang_style = last_meta.get("language_style", "roman_hinglish")
    combined_primary = combine_and_refine_partial_replies(
        partial_primary_replies,
        batch_messages,
        llm_callable=active_llm_fn if first_error is None else None,
        language_style=lang_style,
    )
    cards = [combined_primary]
    if target_num_cards > 1 and len(partial_primary_replies) > 1:
        joined_alt = " ".join(partial_primary_replies)
        if joined_alt not in cards:
            cards.append(joined_alt)
        while len(cards) < target_num_cards:
            cards.append(combined_primary)

    last_meta["batch_groups_count"] = total_groups
    last_meta["batch_messages_count"] = len(batch_messages)
    last_meta["partial_replies"] = partial_primary_replies
    return (
        ReplyGenerationResult(
            cards=cards[:target_num_cards],
            used_llm=any_llm_used,
            llm_error=first_error,
        ),
        last_meta,
    )


def process_chat_request(
    req: ChatRequest,
    *,
    db_path: Path | None = None,
    status_file: Path | None = None,
    rag_dir: Path | None = None,
    sensitivity_threshold: int | None = None,
    active_hours_enabled: bool = ACTIVE_HOURS_ENABLED,
    now: datetime | None = None,
    llm_callable: Callable[[list[dict[str, str]], float], str] | None = None,
    require_llm_for_auto: bool = False,
) -> dict[str, Any]:
    """Core synchronous chat processing pipeline with in-flight message coalescing."""
    paths = PipelinePaths(
        db_path=(DB_PATH if db_path is None else db_path).resolve(),
        status_file=STATUS_FILE_PATH if status_file is None else status_file,
        rag_dir=RAG_DIR if rag_dir is None else rag_dir,
        now=now,
    )
    init_db(paths.db_path)

    gate_skip = _check_gate_and_schedule(paths, active_hours_enabled)
    if gate_skip is not None:
        return gate_skip

    contact, contact_skip = _resolve_contact_or_skip(req, paths)
    if contact_skip is not None or contact is None:
        return contact_skip or {
            "status": "skipped",
            "reason": "contact_mode_off",
            "reply": None,
        }

    raw_text = (req.message or "").strip()
    if req.media_type in ("sticker", "voice", "voice_note") and not raw_text:
        return _handle_media_only_request(req, contact, paths)
    if not raw_text:
        return {"status": "skipped", "reason": "empty_message", "reply": None}

    contact_id = int(contact["id"])
    batch_key = (str(paths.db_path), contact_id)

    # Check if this contact already has an active LLM generation in progress.
    # If so, coalesce this new message into the active batch so the generator restarts
    # with both/all messages combined instead of dropping or double-replying!
    with _IN_FLIGHT_LOCK:
        existing_batch = _IN_FLIGHT_BATCHES.get(batch_key)
        if existing_batch is not None:
            existing_batch.messages.append(raw_text)
            if req.extra_messages:
                existing_batch.messages.extend(
                    m.strip() for m in req.extra_messages if m and m.strip()
                )
            existing_batch.version += 1
            save_message(
                contact_id=contact_id,
                sender_role="user",
                content=raw_text,
                platform=req.app_source,
                telegram_msg_id=req.incoming_msg_id,
                db_path=paths.db_path,
            )
            return {
                "status": "coalesced_in_flight",
                "reason": "merged_into_active_generation",
                "contact_id": contact_id,
                "coalesced_count": len(existing_batch.messages),
                "reply": None,
            }
        initial_msgs = [raw_text]
        if req.extra_messages:
            initial_msgs.extend(
                m.strip() for m in req.extra_messages if m and m.strip()
            )
        active_batch = _InFlightContactBatch(messages=initial_msgs, version=1)
        _IN_FLIGHT_BATCHES[batch_key] = active_batch

    try:
        logger.info(
            "Incoming message from %s (%s): %s",
            contact["canonical_name"],
            req.app_source,
            redact_text(raw_text),
        )

        threshold = (
            get_sensitivity_threshold(paths.db_path)
            if sensitivity_threshold is None
            else int(sensitivity_threshold)
        )
        combined_text = "\n".join(active_batch.messages)
        decision = _evaluate_sensitivity_and_mode(
            req, combined_text, contact, threshold, paths
        )

        if is_urgent_message(combined_text, triage_label=decision.triage_label):
            return _handle_urgent_bypass(
                req, combined_text, contact, decision.sens_score, paths
            )

        if decision.effective_mode == "auto":
            allowed, rate_reason = check_rate_limit_and_cooldown(
                contact_id, now=paths.now, db_path=paths.db_path
            )
            if not allowed:
                # Always save the user's message to history even when auto-reply is rate-limited
                save_message(
                    contact_id=contact_id,
                    sender_role="user",
                    content=raw_text,
                    platform=req.app_source,
                    telegram_msg_id=req.incoming_msg_id,
                    db_path=paths.db_path,
                )
                log_audit_event(
                    event_type="auto_reply_rate_limited",
                    contact_id=contact_id,
                    platform=req.app_source,
                    mode_used="auto",
                    outcome="rate_limited",
                    details=rate_reason,
                    db_path=paths.db_path,
                )
                return {
                    "status": "skipped",
                    "reason": rate_reason,
                    "contact_id": contact_id,
                    "reply": None,
                }

        return _execute_rag_and_persist_draft(
            req=req,
            raw_text=raw_text,
            active_batch=active_batch,
            contact=contact,
            decision=decision,
            threshold=threshold,
            paths=paths,
            llm_callable=llm_callable,
            require_llm_for_auto=require_llm_for_auto,
        )
    finally:
        with _IN_FLIGHT_LOCK:
            _IN_FLIGHT_BATCHES.pop(batch_key, None)


def _execute_rag_and_persist_draft(
    *,
    req: ChatRequest,
    raw_text: str,
    active_batch: _InFlightContactBatch,
    contact: dict[str, Any],
    decision: TriageDecision,
    threshold: int,
    paths: PipelinePaths,
    llm_callable: Callable[[list[dict[str, str]], float], str] | None,
    require_llm_for_auto: bool,
) -> dict[str, Any]:
    """Stage 8-10: Loop if new messages arrive in-flight, run RAG/LLM, split parts, and save."""
    contact_id = int(contact["id"])
    tier = str(contact.get("relationship_tier") or "unknown")
    live_settings = get_all_system_settings(db_path=paths.db_path)
    owner_name = str(live_settings.get("owner_name") or OWNER_NAME)
    history = get_chat_history(
        contact_id, limit=DEFAULT_HISTORY_LIMIT, db_path=paths.db_path
    )
    active_facts = get_active_injected_facts(
        contact_id=contact_id,
        relationship_tier=tier,
        now=paths.now,
        db_path=paths.db_path,
    )

    active_llm_fn = (
        llm_callable
        if llm_callable is not None
        else (_run_llm_chat if get_llm() is not None else None)
    )

    # Re-generate if additional messages arrive from the same sender while generating
    while True:
        with _IN_FLIGHT_LOCK:
            snapshot_version = active_batch.version
            snapshot_messages = list(active_batch.messages)

        combined_query = "\n".join(snapshot_messages)
        if len(snapshot_messages) > 1:
            decision = _evaluate_sensitivity_and_mode(
                req, combined_query, contact, threshold, paths
            )

        top_chunks = retrieve_top_chunks(
            query=combined_query,
            contact_handle=str(contact["canonical_name"]),
            relationship_tier=tier,
            top_k=3,
            rag_dir=paths.rag_dir,
        )
        effective_mode = decision.effective_mode
        target_num_cards = 1 if effective_mode == "auto" else NUM_DRAFT_OPTIONS

        gen_result, prompt_meta = _generate_for_batch_groups(
            snapshot_messages,
            contact=contact,
            tier=tier,
            owner_name=owner_name,
            history=history,
            top_chunks=top_chunks,
            active_facts=active_facts,
            triage_label=decision.triage_label,
            target_num_cards=target_num_cards,
            active_llm_fn=active_llm_fn,
            now=paths.now,
        )

        with _IN_FLIGHT_LOCK:
            if active_batch.version == snapshot_version:
                break

    raw_cards = gen_result.cards
    final_incoming_text = (
        "\n".join(snapshot_messages) if len(snapshot_messages) > 1 else raw_text
    )

    held_for_model_error = False
    if effective_mode == "auto" and (
        gen_result.llm_error is not None
        or (require_llm_for_auto and not gen_result.used_llm)
    ):
        effective_mode = "draft"
        held_for_model_error = True
        _record_owner_notification(
            {
                "type": "model_unavailable_hold",
                "severity": "warning",
                "contact_id": contact_id,
                "contact_handle": contact["canonical_name"],
                "display_name": contact["display_name"],
                "platform": req.app_source,
                "incoming_text": final_incoming_text,
                "error": gen_result.llm_error or "Model not loaded",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            db_path=paths.db_path,
        )

    triage_label: TriageLabel = decision.triage_label
    is_unsure, uncertainty_reason = check_ai_uncertainty(
        final_incoming_text,
        triage_label=triage_label,
        sensitivity_score=decision.sens_score,
        history=history,
        retrieved_chunks=top_chunks,
        injected_facts=active_facts,
        generated_cards=raw_cards,
        strict_grounding=(llm_callable is None),
    )
    if is_unsure:
        effective_mode = "draft"
        triage_label = (
            "ambiguous"
            if "ambiguous" in uncertainty_reason.lower()
            else "unsure"
        )
        # Never surface ungrounded LLM guesses in draft cards when context is missing
        raw_cards = build_unsure_holding_cards(
            tone=str(contact.get("tone") or "casual_hinglish"),
            relationship_tier=tier,
            owner_name=owner_name,
        )
        # Flag this contact so future messages will NOT auto-reply until the owner
        # completes Review -> Edit / Delete / Send (or supplies context) in the dashboard!
        flag_contact_for_review(
            contact_id, uncertainty_reason, db_path=paths.db_path
        )
        _record_owner_notification(
            {
                "type": "ai_unsure_hold",
                "severity": "warning",
                "contact_id": contact_id,
                "contact_handle": contact["canonical_name"],
                "display_name": contact["display_name"],
                "platform": req.app_source,
                "incoming_text": final_incoming_text,
                "uncertainty_reason": uncertainty_reason,
                "contact_flagged": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            db_path=paths.db_path,
        )

    formatted_cards = [
        append_footer(
            c,
            footer_enabled=bool(contact.get("footer_enabled")),
            footer_text=str(contact.get("footer_text") or ""),
        )
        for c in raw_cards
    ]
    reply_parts = split_reply_into_parts(formatted_cards[0])

    save_message(
        contact_id=contact_id,
        sender_role="user",
        content=raw_text,
        platform=req.app_source,
        telegram_msg_id=req.incoming_msg_id,
        db_path=paths.db_path,
    )

    current_dt = datetime.now(timezone.utc) if paths.now is None else paths.now
    if effective_mode == "shadow":
        draft_status = "shadow"
        sched_at = None
    elif effective_mode == "draft":
        draft_status = "pending_approval"
        sched_at = None
    else:
        live_settings = get_all_system_settings(db_path=paths.db_path)
        effective_delay = int(live_settings.get("auto_send_delay_seconds", AUTO_SEND_DELAY_SECONDS))
        if effective_delay <= 0:
            draft_status = "approved"
            sched_at = None
        else:
            draft_status = "pending_send"
            sched_at = (
                current_dt + timedelta(seconds=effective_delay)
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    draft = create_draft(
        contact_id=contact_id,
        incoming_text=final_incoming_text,
        draft_options=formatted_cards,
        reply_parts=reply_parts,
        platform=req.app_source,
        incoming_msg_id=req.incoming_msg_id,
        chunks_used=prompt_meta["chunks_included"],
        sensitivity_score=decision.sens_score,
        triage_label=triage_label,
        status=draft_status,
        scheduled_send_at=sched_at,
        now=paths.now,
        db_path=paths.db_path,
    )

    log_audit_event(
        event_type="message_processed",
        contact_id=contact_id,
        platform=req.app_source,
        mode_used=effective_mode,
        triage_label=triage_label,
        sensitivity_score=decision.sens_score,
        chunks_used=[c["chunk_id"] for c in prompt_meta["chunks_included"]],
        outcome=draft_status,
        details=(
            f"Held due to sensitivity={decision.sens_score} ({','.join(decision.sens_reasons)})"
            if decision.held_for_sensitivity
            else (
                "Held because contact is flagged for owner review"
                if decision.held_for_contact_flag
                else (
                    f"Held due to LLM error/unavailable ({gen_result.llm_error or 'missing model'})"
                    if held_for_model_error
                    else f"Created {len(formatted_cards)} card(s) ({len(reply_parts)} part(s))"
                )
            )
        ),
        db_path=paths.db_path,
    )

    status_label = (
        "held_unsure"
        if is_unsure
        else (
            "held_sensitive"
            if decision.held_for_sensitivity
            else (
                "held_flagged_contact"
                if decision.held_for_contact_flag
                else (
                    "held_model_error"
                    if held_for_model_error
                    else (
                        "queued_auto"
                        if effective_mode == "auto"
                        else f"saved_{effective_mode}"
                    )
                )
            )
        )
    )

    return {
        "status": status_label,
        "uncertainty_reason": uncertainty_reason if is_unsure else None,
        "needs_context": bool(is_unsure),
        "contact_flagged": bool(is_unsure or decision.held_for_contact_flag),
        "mode_used": effective_mode,
        "draft_id": draft["id"],
        "reply": formatted_cards[0] if effective_mode == "auto" else None,
        "reply_parts": reply_parts,
        "options": formatted_cards,
        "chunks_used": prompt_meta["chunks_included"],
        "sensitivity_score": decision.sens_score,
        "triage_label": triage_label,
        "scheduled_send_at": sched_at,
        "prompt_metadata": prompt_meta,
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Liveness & readiness check used by ``start_echotext.py`` launcher."""
    init_db(DB_PATH)
    return {
        "status": "ok",
        "proxy_status": get_proxy_status(),
        "model_file_exists": MODEL_PATH.exists(),
        "model_loaded": _LLM_INSTANCE is not None,
        "hardware_profile": ACTIVE_PROFILE["description"],
    }


@app.get("/status", dependencies=[Depends(verify_api_key)])
def read_status() -> dict[str, Any]:
    """Return current BUSY/AVAILABLE status, active-hours state, and sensitivity threshold."""
    return {
        "proxy_status": get_proxy_status(),
        "within_active_hours": is_within_active_hours(),
        "active_hours_enabled": ACTIVE_HOURS_ENABLED,
        "active_hours_window": f"{ACTIVE_HOURS_START:02d}:00 - {ACTIVE_HOURS_END:02d}:00",
        "sensitivity_threshold": get_sensitivity_threshold(),
    }


@app.post("/status", dependencies=[Depends(verify_api_key)])
def update_status(req: StatusUpdateRequest) -> dict[str, Any]:
    """Atomically flip ``BUSY`` / ``AVAILABLE``. Generates a Catch-Up Digest when BUSY ends."""
    prev_status = get_proxy_status()
    target: Any = "BUSY" if req.status.strip().upper() == "BUSY" else "AVAILABLE"
    new_status = set_proxy_status(target)

    digest_text: str | None = None
    if prev_status == "BUSY" and new_status == "AVAILABLE":
        recent_drafts = list_drafts(limit=30)
        digest_text = generate_catchup_digest(recent_drafts)
        _record_owner_notification(
            {
                "type": "catchup_digest",
                "severity": "info",
                "digest_text": digest_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return {
        "previous_status": prev_status,
        "proxy_status": new_status,
        "catchup_digest": digest_text,
    }


@app.post("/chat", dependencies=[Depends(verify_api_key)])
def receive_message(req: ChatRequest) -> dict[str, Any]:
    """Handle an incoming message from Telegram or another platform listener."""
    return process_chat_request(req)


@app.post("/kill-switch", dependencies=[Depends(verify_api_key)])
def activate_kill_switch(req: KillSwitchRequest) -> dict[str, Any]:
    """Pause auto-replies in a chat when the owner manually types an outgoing message."""
    contact = get_or_create_contact(handle=req.sender_handle, app_source=req.app_source)
    paused_until = trigger_kill_switch(
        int(contact["id"]),
        pause_hours=req.pause_hours,
        pause_seconds=req.pause_seconds,
    )
    return {
        "status": "paused",
        "contact_id": contact["id"],
        "canonical_name": contact["canonical_name"],
        "paused_until": paused_until,
    }


@app.get("/drafts", dependencies=[Depends(verify_api_key)])
def api_list_drafts(
    status: str | None = None,
    contact_id: int | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List drafts for the dashboard, phone approval bot, or listener."""
    promote_due_pending_sends()
    statuses = [s.strip() for s in status.split(",") if s.strip()] if status else None
    items = list_drafts(statuses=statuses, contact_id=contact_id, limit=limit)
    return {"count": len(items), "drafts": items}


@app.post("/drafts/{draft_id}/approve", dependencies=[Depends(verify_api_key)])
def api_approve_draft(
    draft_id: int, req: DraftApproveRequest | None = None
) -> dict[str, Any]:
    """Approve a draft (optionally selecting card index 0, 1, or 2) and clear review flag."""
    try:
        opt_idx = req.option_index if req is not None else None
        existing = get_draft_by_id(draft_id)
        parts: list[str] | None = None
        if existing is not None:
            opts = existing.get("options") or [existing.get("draft_text", "")]
            chosen = (
                opts[opt_idx]
                if opt_idx is not None and 0 <= opt_idx < len(opts)
                else opts[0]
            )
            parts = split_reply_into_parts(str(chosen))
        updated = approve_draft(draft_id, option_index=opt_idx, reply_parts=parts)
        return {"status": "approved", "draft": updated}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@app.post("/drafts/{draft_id}/edit", dependencies=[Depends(verify_api_key)])
def api_edit_draft(draft_id: int, req: DraftEditRequest) -> dict[str, Any]:
    """Edit a draft's text, split into grammatical parts if long, and mark approved."""
    try:
        parts = split_reply_into_parts(req.new_text)
        updated = edit_and_approve_draft(
            draft_id, new_text=req.new_text, reply_parts=parts
        )
        return {"status": "approved", "draft": updated}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@app.post("/drafts/{draft_id}/regenerate", dependencies=[Depends(verify_api_key)])
def api_regenerate_draft(draft_id: int, req: DraftRegenerateRequest) -> dict[str, Any]:
    """Supply missing owner context and regenerate grounded reply options for a pending draft."""
    draft = get_draft_by_id(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"Draft #{draft_id} not found.")
    ctx = req.context_text.strip()
    if not ctx:
        raise HTTPException(status_code=400, detail="Context text cannot be empty.")

    contact_id = int(draft["contact_id"])
    tier = str(draft.get("relationship_tier") or "unknown")
    contact = get_or_create_contact(
        handle=str(draft.get("canonical_name") or ""),
        app_source=str(draft.get("platform") or "telegram"),
    )
    if req.save_as_fact:
        add_injected_fact(
            ctx,
            ttl_minutes=req.ttl_minutes,
            scope_contact_id=contact_id,
            scope_tier=tier if tier != "unknown" else "all_trusted",
        )
    active_facts = get_active_injected_facts(contact_id=contact_id, relationship_tier=tier)
    if not any(ctx in str(f.get("fact_text", "")) for f in active_facts):
        active_facts.insert(0, {"fact_text": ctx, "expires_at": "owner_supplied"})

    history = get_chat_history(contact_id, limit=DEFAULT_HISTORY_LIMIT)
    top_chunks = retrieve_top_chunks(
        query=str(draft.get("incoming_text") or ""),
        contact_handle=str(contact["canonical_name"]),
        relationship_tier=tier,
        top_k=3,
    )
    live_settings = get_all_system_settings()
    owner_name = str(live_settings.get("owner_name") or OWNER_NAME)
    active_llm_fn = _run_llm_chat if get_llm() is not None else None

    gen_res, prompt_meta = _generate_for_batch_groups(
        [str(draft.get("incoming_text") or "")],
        contact=contact,
        tier=tier,
        owner_name=owner_name,
        history=history,
        top_chunks=top_chunks,
        active_facts=active_facts,
        triage_label="facts",
        target_num_cards=NUM_DRAFT_OPTIONS,
        active_llm_fn=active_llm_fn,
        now=None,
    )
    usable_cards = [
        c for c in gen_res.cards if "[UNSURE]" not in c.upper() and "[AMBIGUOUS]" not in c.upper()
    ]
    if not usable_cards:
        usable_cards = generate_reply_cards_detailed(
            [],
            llm_callable=None,
            num_cards=NUM_DRAFT_OPTIONS,
            triage_label="facts",
            language_style=prompt_meta.get("language_style", "roman_hinglish"),
            tone=str(contact.get("tone") or "casual_hinglish"),
            relationship_tier=tier,
            owner_name=owner_name,
            injected_facts=active_facts,
        ).cards
    formatted_cards = [
        append_footer(
            c,
            footer_enabled=bool(contact.get("footer_enabled")),
            footer_text=str(contact.get("footer_text") or ""),
        )
        for c in usable_cards
    ]
    reply_parts = split_reply_into_parts(formatted_cards[0]) if formatted_cards else []
    updated = update_draft_options(
        draft_id,
        draft_options=formatted_cards,
        reply_parts=reply_parts,
        triage_label="facts",
        chunks_used=prompt_meta.get("chunks_included", []),
    )
    return {"status": "regenerated", "draft": updated}


@app.post("/drafts/{draft_id}/reject", dependencies=[Depends(verify_api_key)])
def api_reject_draft(draft_id: int) -> dict[str, Any]:
    """Reject a draft so it is never sent, and clear any ambiguity flag on the contact."""
    try:
        updated = reject_draft(draft_id)
        return {"status": "rejected", "draft": updated}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@app.post("/drafts/{draft_id}/cancel_auto", dependencies=[Depends(verify_api_key)])
def api_cancel_auto_send(draft_id: int) -> dict[str, Any]:
    """Cancel an auto-reply during its delay window, turning it into a draft."""
    try:
        updated = cancel_pending_auto_send(draft_id)
        return {"status": "cancelled_to_draft", "draft": updated}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@app.post("/drafts/{draft_id}/mark_sent", dependencies=[Depends(verify_api_key)])
def api_mark_draft_sent(
    draft_id: int,
    sent_telegram_msg_id: int | None = None,
    req: DraftMarkSentRequest | None = None,
) -> dict[str, Any]:
    """Called by ``telegram_listener.py`` once an approved/sending draft has been delivered."""
    try:
        msg_id = (
            sent_telegram_msg_id
            if sent_telegram_msg_id is not None
            else (req.sent_telegram_msg_id if req is not None else None)
        )
        msg_ids = req.sent_msg_ids if req is not None else None
        updated = mark_draft_sent(
            draft_id, sent_telegram_msg_id=msg_id, sent_msg_ids=msg_ids
        )
        return {"status": "sent", "draft": updated}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@app.post("/drafts/{draft_id}/mark_failed", dependencies=[Depends(verify_api_key)])
def api_mark_draft_failed(
    draft_id: int,
    req: DraftFailRequest | None = None,
) -> dict[str, Any]:
    """Called by ``telegram_listener.py`` when sending a draft on Telegram raises an error."""
    try:
        err_msg = req.error_message if req is not None else "Telegram delivery failed"
        updated = mark_draft_failed(draft_id, error_message=err_msg)
        return {"status": updated.get("status", "failed"), "draft": updated}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@app.post("/drafts/{draft_id}/delete_sent", dependencies=[Depends(verify_api_key)])
def api_delete_sent_reply(draft_id: int) -> dict[str, Any]:
    """Delete a previously sent auto-reply from Telegram and remove it from chat history."""
    try:
        updated = request_delete_sent_reply(draft_id)
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    return {"status": "deleted_sent", "draft": updated}


@app.get("/listener/poll", dependencies=[Depends(verify_api_key)])
def listener_poll() -> dict[str, Any]:
    """Endpoint polled by ``telegram_listener.py`` to claim approved drafts and pop outbox events."""
    promote_due_pending_sends()
    claimed_drafts = claim_approved_drafts_for_sending(limit=20)
    notifications = pop_outbox_events("notification", limit=100)
    delete_requests = pop_outbox_events("delete_request", limit=100)
    with _NOTIFICATIONS_LOCK:
        _OWNER_NOTIFICATIONS.clear()
        _DELETE_MSG_REQUESTS.clear()
    return {
        "approved_drafts": claimed_drafts,
        "notifications": notifications,
        "delete_requests": delete_requests,
    }


@app.get("/notifications", dependencies=[Depends(verify_api_key)])
def api_get_ui_notifications(unread_only: bool = False, limit: int = 40) -> dict[str, Any]:
    """Return persistent notifications for the Dashboard UI Notification Bell (`🔔`)."""
    items = list_ui_notifications(unread_only=unread_only, limit=limit)
    unread_cnt = sum(1 for x in items if not x.get("ui_read"))
    return {"count": len(items), "unread_count": unread_cnt, "notifications": items}


@app.post("/notifications/read", dependencies=[Depends(verify_api_key)])
def api_mark_notifications_read(
    req: NotificationReadRequest | None = None,
) -> dict[str, Any]:
    """Mark one or all UI notifications as read."""
    nid = req.notification_id if req is not None else None
    updated = mark_ui_notifications_read(notification_id=nid)
    return {"status": "ok", "marked_read": updated}


@app.get("/contacts", dependencies=[Depends(verify_api_key)])
def api_list_contacts() -> dict[str, Any]:
    """List all contacts and their per-contact autonomy modes, personas, and review flags."""
    items = list_contacts()
    return {"count": len(items), "contacts": items}


@app.post("/contacts", dependencies=[Depends(verify_api_key)])
def api_create_or_update_contact(req: ContactUpsertRequest) -> dict[str, Any]:
    """Create or update a contact via the Brain REST API."""
    try:
        contact = get_or_create_contact(
            handle=req.handle,
            app_source=req.app_source,
            display_name=req.display_name,
            platform_user_id=req.platform_user_id,
        )
        updated = update_contact_settings(
            int(contact["id"]),
            display_name=req.display_name,
            mode=req.mode,
            relationship_tier=req.relationship_tier,
            tone=req.tone,
            custom_persona=req.custom_persona,
            footer_enabled=req.footer_enabled,
            footer_text=req.footer_text,
        )
        return {"status": "ok", "contact": updated}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/contacts/{contact_id}", dependencies=[Depends(verify_api_key)])
def api_patch_contact(contact_id: int, req: ContactUpdateRequest) -> dict[str, Any]:
    """Update a contact's settings via the Brain REST API."""
    try:
        updated = update_contact_settings(
            contact_id,
            display_name=req.display_name,
            mode=req.mode,
            relationship_tier=req.relationship_tier,
            tone=req.tone,
            custom_persona=req.custom_persona,
            footer_enabled=req.footer_enabled,
            footer_text=req.footer_text,
        )
        return {"status": "ok", "contact": updated}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/contacts/{contact_id}/handles", dependencies=[Depends(verify_api_key)])
def api_link_handle(contact_id: int, req: HandleLinkRequest) -> dict[str, Any]:
    """Link an additional platform handle to a contact."""
    try:
        link_contact_handle(contact_id, req.platform, req.handle)
        return {"status": "linked", "contact_id": contact_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/contacts/{contact_id}/resume", dependencies=[Depends(verify_api_key)])
def api_resume_contact(contact_id: int) -> dict[str, Any]:
    """Clear the manual kill-switch pause on a contact."""
    clear_kill_switch(contact_id)
    return {"status": "resumed", "contact_id": contact_id}


@app.post("/contacts/{contact_id}/unflag", dependencies=[Depends(verify_api_key)])
def api_unflag_contact(contact_id: int) -> dict[str, Any]:
    """Explicitly clear the ambiguity/review flag on a contact."""
    updated = clear_contact_review_flag(contact_id)
    return {"status": "unflagged", "contact": updated}


@app.delete("/contacts/{contact_id}", dependencies=[Depends(verify_api_key)])
def api_forget_contact(contact_id: int) -> dict[str, Any]:
    """Completely erase a contact and their scoped RAG notes."""
    ok = forget_contact(contact_id, rag_dir=RAG_DIR)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Contact {contact_id} not found.")
    return {"status": "forgotten", "contact_id": contact_id}


@app.get("/facts", dependencies=[Depends(verify_api_key)])
def api_list_facts(relationship_tier: str = "friend") -> dict[str, Any]:
    """List active non-expired injected facts."""
    items = get_active_injected_facts(relationship_tier=relationship_tier)
    return {"count": len(items), "facts": items}


@app.post("/facts", dependencies=[Depends(verify_api_key)])
def api_create_fact(req: FactCreateRequest) -> dict[str, Any]:
    """Insert a temporary expiring fact."""
    try:
        fact = add_injected_fact(
            req.fact_text,
            ttl_minutes=req.ttl_minutes,
            scope_contact_id=req.scope_contact_id,
            scope_tier=req.scope_tier,
        )
        return {"status": "created", "fact": fact}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/facts/{fact_id}", dependencies=[Depends(verify_api_key)])
def api_delete_fact(fact_id: int) -> dict[str, Any]:
    """Delete an injected fact immediately."""
    delete_injected_fact(fact_id)
    return {"status": "deleted", "fact_id": fact_id}


@app.get("/settings", dependencies=[Depends(verify_api_key)])
def api_get_settings() -> dict[str, Any]:
    """Return persisted server and system timing settings."""
    all_settings = get_all_system_settings()
    all_settings["sensitivity_threshold"] = get_sensitivity_threshold()
    all_settings["proxy_status"] = get_proxy_status()
    return {"status": "ok", "settings": all_settings}


@app.post("/settings", dependencies=[Depends(verify_api_key)])
def api_update_settings(req: SystemSettingsUpdateRequest) -> dict[str, Any]:
    """Update system timing, persona, and autonomy parameters dynamically."""
    updated = save_system_settings(req.settings)
    return {"status": "updated", "settings": updated}


@app.post("/settings/reset", dependencies=[Depends(verify_api_key)])
def api_reset_settings() -> dict[str, Any]:
    """Restore all system timing and operational parameters to recommended defaults."""
    res = reset_system_settings_to_defaults()
    return {"status": "reset", "settings": res}


@app.post("/settings/sensitivity", dependencies=[Depends(verify_api_key)])
def api_set_sensitivity_threshold(req: SensitivitySettingRequest) -> dict[str, Any]:
    """Persist the sensitivity threshold in SQLite so the server enforces it immediately."""
    saved = set_sensitivity_threshold(req.sensitivity_threshold)
    return {"status": "updated", "sensitivity_threshold": saved}


@app.post("/maintenance/purge", dependencies=[Depends(verify_api_key)])
def api_purge_old_records(req: PurgeRequest) -> dict[str, Any]:
    """Purge messages, drafts, and audit logs older than retention_days."""
    return purge_old_records(retention_days=req.retention_days)


@app.get("/audit", dependencies=[Depends(verify_api_key)])
def api_list_audit_logs(limit: int = 50) -> dict[str, Any]:
    """Return recent audit log entries."""
    items = list_audit_logs(limit=limit)
    return {"count": len(items), "logs": items}
