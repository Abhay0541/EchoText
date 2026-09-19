"""
Two-stage routing, using ONE model called twice.

Why one model twice instead of two separate models (a small classifier +
a generator)?
- One fine-tuned checkpoint to manage, version, and deploy on-device instead
  of two — meaningfully simpler for an offline/on-device target.
- Phi-3.5-mini is already small enough that the latency cost of a second
  short call is negligible compared to loading a second model into memory.
- The classification call and the generation call can share the exact same
  "voice" context (your fine-tuned style), which matters for stage 2 below.

Call 1 - CLASSIFY: given the incoming message + retrieved similar past
replies, decide the *route*:
    - "quick_reply"   -> short, low-stakes, safe to auto-send
    - "needs_context" -> ambiguous / needs more chat history before replying
    - "sensitive"      -> emotionally loaded, financial, or a topic the user
                           said should never be auto-replied to
    - "unknown"        -> no good match, fall back to a safe generic reply

Call 2 - GENERATE: only for "quick_reply" and "needs_context" routes, the
model is called again with the same message + the RAG examples + the route
decision, and produces the actual reply text. "sensitive" and "unknown"
routes skip generation entirely and are flagged for the human to answer
personally — this is the main safety valve of the whole system.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models.model_backend import ModelBackend
from app.rag.retriever import RetrievedExample

# Keyword-based guardrail: even before the model gets a vote, these topics are
# force-routed to "sensitive". A regex/keyword pre-filter here is intentional
# — it can't be fine-tuned away by accident, unlike a purely model-driven
# decision, so it acts as a hard floor under the model's own judgment.
_SENSITIVE_MARKERS = (
    "money", "loan", "password", "otp", "account number",
    "hospital", "emergency", "died", "death", "breakup", "propose",
)


class Route(str, Enum):
    QUICK_REPLY = "quick_reply"
    NEEDS_CONTEXT = "needs_context"
    SENSITIVE = "sensitive"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    route: Route
    reasoning: str
    reply: str | None = None  # populated only if the route allows auto-generation


class MessageRouter:
    def __init__(self, model: ModelBackend):
        self._model = model

    def handle(
        self,
        incoming_message: str,
        examples: list[RetrievedExample],
    ) -> RoutingDecision:
        lowered = incoming_message.lower()
        if any(marker in lowered for marker in _SENSITIVE_MARKERS):
            return RoutingDecision(
                route=Route.SENSITIVE,
                reasoning="Matched a hard-coded sensitive-topic marker; "
                          "auto-reply is intentionally disabled for these.",
            )

        # --- Call 1: classify ---
        classification = self._model.classify(incoming_message, examples)
        route = Route(classification) if classification in Route._value2member_map_ else Route.UNKNOWN

        if route in (Route.SENSITIVE, Route.UNKNOWN):
            return RoutingDecision(
                route=route,
                reasoning=f"Model classified this as '{route.value}'; "
                          "skipping auto-generation.",
            )

        # --- Call 2: generate (only reached for quick_reply / needs_context) ---
        reply = self._model.generate(incoming_message, examples, route=route.value)
        return RoutingDecision(
            route=route,
            reasoning=f"Model classified this as '{route.value}' and generated a reply.",
            reply=reply,
        )
