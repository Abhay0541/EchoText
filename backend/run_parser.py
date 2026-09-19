import json
from app.data.whatsapp_parser import parse_chat_export, build_training_pairs

# 1. Read your exported chat
with open("chat.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. Parse the raw text into Message objects
messages = parse_chat_export(raw_text)

# 3. Build training pairs (Change "Your Exact WhatsApp Name" to your name in the chat)
pairs = build_training_pairs(messages, target_user="♕♡⚘𝐴ₘᵅᶰ𝑟ᤂ𝔧ῥข𝐭⚘♡♕")

# 4. Format them for LoRA training
dataset = [pair.as_prompt_completion() for pair in pairs]

# 5. Save to dataset.json
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"Success! Generated {len(dataset)} training pairs and saved to dataset.json.")