"""
Discord integration — same thin-adapter pattern as telegram_bot.py.

    pip install discord.py
    export DISCORD_BOT_TOKEN=...
    python -m app.integrations.discord_bot --session_id <id>
"""
from __future__ import annotations

import argparse
import logging

from app import config
from app.main import SESSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echotext.discord")


def build_client(session_id: str):
    import discord

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        logger.info("EchoText Discord bot logged in as %s", client.user)

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        session = SESSIONS.get(session_id)
        if session is None:
            return

        examples = session["index"].query(message.content, top_k=config.RAG_TOP_K)
        decision = session["router"].handle(message.content, examples)

        if decision.reply is not None:
            await message.channel.send(decision.reply)
            logger.info("Auto-replied (route=%s)", decision.route.value)
        else:
            logger.info("Held back (route=%s), needs a human reply", decision.route.value)

    return client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    args = parser.parse_args()

    if not config.DISCORD_BOT_TOKEN:
        raise SystemExit("Set DISCORD_BOT_TOKEN in your environment first.")

    client = build_client(args.session_id)
    client.run(config.DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
