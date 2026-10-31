import os
import time
import requests
from dotenv import load_dotenv
from telethon import TelegramClient, events

# Load credentials from .env file
load_dotenv()
API_ID = int(os.getenv('TELEGRAM_API_ID'))
API_HASH = os.getenv('TELEGRAM_API_HASH')
BRAIN_URL = "http://127.0.0.1:8000/chat"

# Create the Telegram client (creates my_proxy.session)
client = TelegramClient('my_proxy', API_ID, API_HASH)

@client.on(events.NewMessage(incoming=True))
async def handle_new_message(event):
    sender = await event.get_sender()
    # Fallback to ID if the user doesn't have a public username
    username = sender.username if sender.username else str(sender.id)
    
    # SAFETY TOGGLE: Ignore group chats, only reply to direct private messages
    if not event.is_private:
        return

    print(f"\n[Incoming] {username}: {event.raw_text}")
    
    # Show "typing..." status in Telegram while the GPU brain thinks
    async with client.action(event.chat_id, 'typing'):
        payload = {
            "app_source": "telegram",
            "sender_id": username,
            "text": event.raw_text
        }
        
        try:
            # Send the message to your local FastAPI server
            response = requests.post(BRAIN_URL, json=payload).json()
            
            # If proxy_status.txt is set to AVAILABLE, the brain skips the auto-reply
            if response.get('status') == 'skipped':
                return
                
            reply_text = response['reply']
            
            # Simulate natural human typing speed (faster for short texts, capped at 4 seconds)
            typing_delay = len(reply_text) * 0.05
            time.sleep(min(typing_delay, 4))
            
            # Send the final AI reply back to the chat
            await event.reply(reply_text)
            print(f"[Sent] to {username}: {reply_text}")
            
        except Exception as e:
            print(f"Error connecting to brain: {e}")

print("Telegram Interceptor is starting...")
client.start()
print("Listening for messages... (Press Ctrl+C to stop)")
client.run_until_disconnected()