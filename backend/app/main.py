"""
EchoText backend entrypoint.

    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /sessions/upload       upload a WhatsApp export, build the RAG index
    POST /sessions/{id}/reply   route + (maybe) generate a reply to a message
    GET  /health                liveness check

One design choice worth flagging: sessions are in-memory (a plain dict). For
a personal, single-user, offline-first tool that's genuinely the right
amount of infrastructure — a database here would be solving a problem this
project doesn't have. If EchoText ever needs multiple concurrent users or
persistence across restarts, swapping `SESSIONS` for a real store (SQLite is
the natural next step, not Postgres) touches only this file.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config
from app.data.whatsapp_parser import build_training_pairs, parse_chat_export
from app.models.model_backend import get_model_backend
from app.rag.retriever import ChatIndex
from app.routing.router import MessageRouter

app = FastAPI(title="EchoText", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> {"index": ChatIndex, "router": MessageRouter, "target_user": str, "pair_count": int}
SESSIONS: dict[str, dict] = {}


class ReplyRequest(BaseModel):
    incoming_message: str


class ReplyResponse(BaseModel):
    route: str
    reasoning: str
    reply: str | None
    retrieved_examples: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "model_backend": config.MODEL_BACKEND}


@app.post("/sessions/upload")
async def upload_chat(file: UploadFile, target_user: str):
    """Upload a WhatsApp .txt export. Builds a per-session RAG index over
    the (context, reply) pairs so /reply can retrieve similar past exchanges."""
    raw_bytes = await file.read()
    raw_text = raw_bytes.decode("utf-8", errors="ignore")

    messages = parse_chat_export(raw_text)
    if not messages:
        raise HTTPException(400, "No parseable messages found — is this a WhatsApp .txt export?")

    pairs = build_training_pairs(messages, target_user=target_user)
    if not pairs:
        raise HTTPException(
            400,
            f"Found {len(messages)} messages but none sent by '{target_user}'. "
            "Check the name matches your display name in the chat exactly.",
        )

    index = ChatIndex()
    index.build(pairs)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "index": index,
        "router": MessageRouter(model=get_model_backend(config.MODEL_BACKEND)),
        "target_user": target_user,
        "pair_count": len(pairs),
    }
    return {
        "session_id": session_id,
        "messages_parsed": len(messages),
        "training_pairs_built": len(pairs),
    }


@app.post("/sessions/{session_id}/reply", response_model=ReplyResponse)
def reply(session_id: str, body: ReplyRequest):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session_id — upload a chat export first.")

    index: ChatIndex = session["index"]
    router: MessageRouter = session["router"]

    examples = index.query(body.incoming_message, top_k=config.RAG_TOP_K)
    decision = router.handle(body.incoming_message, examples)

    return ReplyResponse(
        route=decision.route.value,
        reasoning=decision.reasoning,
        reply=decision.reply,
        retrieved_examples=[
            {"context": e.context, "reply": e.reply, "score": round(e.score, 3)}
            for e in examples
        ],
    )


@app.get("/sessions/{session_id}")
def session_info(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session_id.")
    return {
        "target_user": session["target_user"],
        "training_pairs": session["pair_count"],
    }
