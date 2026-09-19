"""
Telegram integration.

    pip install python-telegram-bot
    export TELEGRAM_BOT_TOKEN=...
    python -m app.integrations.telegram_bot --session_id <id from /sessions/upload>

Deliberately a thin adapter: it does zero reply logic itself. It just takes
an incoming Telegram message, calls the *same* MessageRouter used by the API
and the frontend, and either sends the auto-reply or, for "sensitive"/
"unknown" routes, notifies you to reply personally instead. Keeping all
platforms behind one router is what makes "add Discord/Gmail later" a small
diff instead of a rewrite.
"""
from __future__ import annotations

import argparse
import logging

from app import config
from app.main import SESSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echotext.telegram")


async def handle_message(update, context):  # noqa: ANN001 - types come from python-telegram-bot
    session = SESSIONS.get(context.bot_data["session_id"])
    if session is None:
        await update.message.reply_text("(EchoText session not loaded — check server logs)")
        return

    incoming = update.message.text
    examples = session["index"].query(incoming, top_k=config.RAG_TOP_K)
    decision = session["router"].handle(incoming, examples)

    if decision.reply is not None:
        await update.message.reply_text(decision.reply)
        logger.info("Auto-replied (route=%s) to: %s", decision.route.value, incoming)
    else:
        logger.info("Held back (route=%s), needs a human reply: %s", decision.route.value, incoming)
        # Deliberately does NOT send anything to the other person here —
        # silence is the safe default for sensitive/unclear messages.


def main():
    from telegram.ext import ApplicationBuilder, MessageHandler, filters

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    args = parser.parse_args()

    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in your environment first.")

    app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.bot_data["session_id"] = args.session_id
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("EchoText Telegram bot running with session %s", args.session_id)
    app.run_polling()


if __name__ == "__main__":
    main()
