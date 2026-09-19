"""
WhatsApp chat log parser.

WhatsApp exports chats as a .txt file where every line looks like:
    12/08/24, 9:41 pm - Abhay: On my way, 5 mins
    12/08/24, 9:42 pm - Riya: ok cool

This module turns that raw export into two things EchoText needs downstream:

1. A clean list of `Message` objects (sender, timestamp, text) — used to build
   the RAG index (so the model can retrieve "how did I reply last time
   someone said X").
2. A list of (context, reply) training *pairs* for the LoRA fine-tuning step —
   every message sent by `target_user` becomes a "reply", and the block of
   messages immediately before it (from other people) becomes the "context".

Design note: this is intentionally regex-based rather than a heavier NLP
parser. WhatsApp's export format is stable and well-documented, so a robust
regex is more maintainable than pulling in a parsing library for one job.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable


# Matches the two common WhatsApp export date/time stamps:
#   "12/08/24, 9:41 pm - "   (12-hour, most Android/iOS exports)
#   "12/08/2024, 21:41 - "   (24-hour, common in EU locales)
_LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*"
    r"(?P<time>\d{1,2}:\d{2}(?:\s?[apAP][mM])?)\s*-\s*"
    r"(?P<sender>[^:]+):\s*(?P<text>.*)$"
)

# System messages we never want treated as a "reply" (nothing to learn from).
_SYSTEM_MSG_MARKERS = (
    "Messages and calls are end-to-end encrypted",
    "<Media omitted>",
    "This message was deleted",
    "You deleted this message",
    "changed the subject",
    "changed this group's icon",
)


@dataclass
class Message:
    sender: str
    text: str
    timestamp: datetime | None = None


@dataclass
class TrainingPair:
    context: list[Message] = field(default_factory=list)
    reply: str = ""

    def as_prompt_completion(self) -> dict:
        """Format as a single instruction-style example for LoRA training."""
        context_str = "\n".join(f"{m.sender}: {m.text}" for m in self.context)
        return {
            "prompt": (
                "You are replying to the conversation below in your own "
                "texting style.\n\n" + context_str + "\n\nYour reply:"
            ),
            "completion": self.reply,
        }


def _parse_timestamp(date_str: str, time_str: str) -> datetime | None:
    formats = (
        "%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p",
        "%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p",
        "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M",
        "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M",
    )
    raw = f"{date_str} {time_str}".replace("\u202f", " ").strip()
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def parse_chat_export(raw_text: str) -> list[Message]:
    """Parse a raw WhatsApp .txt export into a flat list of Message objects.

    Handles multi-line messages (WhatsApp doesn't re-print the timestamp for
    line-wrapped text) by appending continuation lines to the previous
    message.
    """
    messages: list[Message] = []
    for line in raw_text.splitlines():
        line = line.strip("\ufeff \r")
        if not line:
            continue
        match = _LINE_RE.match(line)
        if match:
            text = match.group("text").strip()
            if any(marker in text for marker in _SYSTEM_MSG_MARKERS):
                continue
            messages.append(Message(
                sender=match.group("sender").strip(),
                text=text,
                timestamp=_parse_timestamp(match.group("date"), match.group("time")),
            ))
        elif messages:
            # Continuation of the previous (multi-line) message.
            messages[-1].text += "\n" + line
    return messages


def build_training_pairs(
    messages: Iterable[Message],
    target_user: str,
    max_context_messages: int = 6,
) -> list[TrainingPair]:
    """Turn a message stream into (context, reply) pairs for `target_user`.

    Consecutive messages from `target_user` are merged into one "turn" (people
    often send 3 short texts instead of 1 long one) so the model learns to
    produce a single coherent reply rather than fragmenting it.
    """
    messages = list(messages)
    pairs: list[TrainingPair] = []
    buffer: list[Message] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.sender == target_user:
            # Merge consecutive messages from the target user into one reply.
            reply_parts = [msg.text]
            j = i + 1
            while j < len(messages) and messages[j].sender == target_user:
                reply_parts.append(messages[j].text)
                j += 1
            if buffer:  # only a valid training pair if there's context to learn from
                pairs.append(TrainingPair(
                    context=buffer[-max_context_messages:],
                    reply="\n".join(reply_parts),
                ))
            buffer = []
            i = j
        else:
            buffer.append(msg)
            i += 1

    return pairs
