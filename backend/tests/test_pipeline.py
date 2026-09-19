"""
Run with: pytest -v  (from backend/)
"""
from app.data.whatsapp_parser import build_training_pairs, parse_chat_export
from app.models.model_backend import TemplateBackend
from app.rag.retriever import ChatIndex
from app.routing.router import MessageRouter, Route

SAMPLE_CHAT = """\
12/08/24, 9:41 pm - Riya: hey are we still on for tonight?
12/08/24, 9:42 pm - Abhay: yeah for sure, 8pm?
12/08/24, 9:43 pm - Riya: perfect see you then
13/08/24, 8:00 am - Riya: what time works for lunch today?
13/08/24, 8:01 am - Abhay: 1pm works for me, usual place?
"""


def test_parse_chat_export_counts_messages():
    messages = parse_chat_export(SAMPLE_CHAT)
    assert len(messages) == 5
    assert messages[0].sender == "Riya"
    assert messages[1].sender == "Abhay"


def test_build_training_pairs_only_for_target_user():
    messages = parse_chat_export(SAMPLE_CHAT)
    pairs = build_training_pairs(messages, target_user="Abhay")
    assert len(pairs) == 2
    assert pairs[0].reply == "yeah for sure, 8pm?"
    assert pairs[0].context[0].sender == "Riya"


def test_chat_index_retrieves_closest_match():
    messages = parse_chat_export(SAMPLE_CHAT)
    pairs = build_training_pairs(messages, target_user="Abhay")
    index = ChatIndex()
    index.build(pairs)

    results = index.query("are we still meeting tonight?", top_k=1)
    assert len(results) == 1
    assert results[0].reply == "yeah for sure, 8pm?"


def test_router_holds_back_sensitive_topics():
    router = MessageRouter(model=TemplateBackend())
    decision = router.handle("can you send me your bank account number", examples=[])
    assert decision.route == Route.SENSITIVE
    assert decision.reply is None


def test_router_returns_unknown_with_no_examples():
    router = MessageRouter(model=TemplateBackend())
    decision = router.handle("totally novel message about hiking", examples=[])
    assert decision.route == Route.UNKNOWN
    assert decision.reply is None


def test_router_generates_quick_reply_for_strong_match():
    messages = parse_chat_export(SAMPLE_CHAT)
    pairs = build_training_pairs(messages, target_user="Abhay")
    index = ChatIndex()
    index.build(pairs)
    router = MessageRouter(model=TemplateBackend())

    examples = index.query("are we still on for tonight?", top_k=3)
    decision = router.handle("are we still on for tonight?", examples)
    assert decision.route == Route.QUICK_REPLY
    assert decision.reply is not None
