# telegram_listener.py
import os
import asyncio
import logging

import requests
from dotenv import load_dotenv
from telethon import TelegramClient, events

from config import BRAIN_URL, SESSION_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("echotext.listener")

# Load credentials from .env file
load_dotenv()


def require_env(name: str) -> str:
    """Stops with a clear message that names the missing variable."""
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name}. Add it to your .env file.")
    return value


try:
    API_ID = int(require_env("TELEGRAM_API_ID"))
except ValueError:
    raise SystemExit("TELEGRAM_API_ID must be a number. Check your .env file.")
API_HASH = require_env("TELEGRAM_API_HASH")

# Create the Telegram client (creates my_proxy.session next to this file)
client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)


@client.on(events.NewMessage(incoming=True))
async def handle_new_message(event):
    # 1. Private chats only. Checked FIRST, before any other work.
    if not event.is_private:
        return

    # 2. Nothing to answer (stickers, photos without text, empty messages)
    text = event.raw_text
    if not text or not text.strip():
        return

    # 3. Never answer other bots (two bots would reply to each other forever)
    sender = await event.get_sender()
    if getattr(sender, "bot", False):
        return

    # Fallback to ID if the user doesn't have a public username
    username = sender.username if sender.username else str(sender.id)

    print(f"\n[Incoming] {username}: {text}")

    # Show "typing..." status in Telegram while the model thinks
    async with client.action(event.chat_id, 'typing'):
        payload = {
            "app_source": "telegram",
            "sender_id": username,
            "text": text
        }

        try:
            # Run the blocking HTTP call in a worker thread so Telethon keeps
            # working, and give up if the brain does not answer in time.
            response = await asyncio.to_thread(
                requests.post, BRAIN_URL, json=payload, timeout=(5, 120)
            )
            response.raise_for_status()
            data = response.json()

            # The brain skips the reply when status is not BUSY (or text is empty)
            if data.get("status") != "success":
                return

            reply_text = data["reply"]

            # Simulate natural typing speed (capped at 4 seconds) without blocking
            await asyncio.sleep(min(len(reply_text) * 0.05, 4))

            # Send the final AI reply back to the chat
            await event.reply(reply_text)
            print(f"[Sent] to {username}: {reply_text}")

        except requests.exceptions.ConnectionError:
            log.error("Brain is not reachable at %s. Is server.py running?", BRAIN_URL)
        except requests.exceptions.Timeout:
            log.error("Brain took too long to answer. No reply sent to %s.", username)
        except Exception:
            log.exception("Failed to handle a message from %s", username)


print("Telegram Interceptor is starting...")
client.start()
print("Listening for messages... (Press Ctrl+C to stop)")
client.run_until_disconnected()