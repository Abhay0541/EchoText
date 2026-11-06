# server.py
import logging
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

from config import MODEL_PATH, STATUS_FILE, RAG_FOLDER
from database import init_db, get_or_create_contact, save_message, get_chat_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("echotext.server")

app = FastAPI(title="Project EchoText Proxy")

# Make sure the tables exist before the first message arrives
init_db()

# A fresh copy starts SILENT: create the status file as AVAILABLE if it is missing
if not STATUS_FILE.exists():
    STATUS_FILE.write_text("AVAILABLE", encoding="utf-8")

if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model file not found: {MODEL_PATH}\n"
        "Put the .gguf file there, or set ECHOTEXT_MODEL_PATH."
    )

print("Booting up... Loading 3B GGUF into memory...")
# n_gpu_layers=0 keeps the 3B model on the CPU
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False
)
print("Brain is online and listening!")


class IncomingMessage(BaseModel):
    app_source: Literal["telegram", "whatsapp"]
    sender_id: str
    text: str


def is_busy() -> bool:
    """Reply ONLY when the status file clearly says BUSY. Anything else = stay silent."""
    try:
        return STATUS_FILE.read_text(encoding="utf-8").strip().upper() == "BUSY"
    except FileNotFoundError:
        return False
    except OSError:
        log.exception("Could not read the status file; staying silent")
        return False


def read_rag_memory():
    """Scans the folder managed by the Streamlit UI and extracts facts."""
    if not RAG_FOLDER.is_dir():
        return ""

    rag_context = ""
    for file_path in sorted(RAG_FOLDER.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8").strip()
            rag_context += f"\n[Fact from {file_path.name}]: {text}\n"
        except Exception:
            log.exception("Could not read %s", file_path.name)

    return rag_context


@app.post("/chat")
async def receive_message(msg: IncomingMessage):
    log.info("[%s] %s: %s", msg.app_source, msg.sender_id, msg.text)

    # 0. Nothing to answer (stickers, empty text)
    if not msg.text.strip():
        return {"status": "skipped"}

    # 1. Global state check (fail closed: only an explicit BUSY allows replies)
    if not is_busy():
        log.info("Status is not BUSY. Skipping auto-reply.")
        return {"status": "skipped"}

    # 2. Database memory (newest messages)
    contact_id = get_or_create_contact(msg.app_source, msg.sender_id)
    history_text = get_chat_history(contact_id, limit=4)

    # 3. Dynamic persona (sender_id is compared in lowercase)
    if msg.sender_id.strip().lower() == "aman99":
        relationship = "Sender is Aman, your MITS college classmate. Use heavy Gwalior/college slang."
    else:
        relationship = "Sender is a general contact. Keep it casual and short."

    system_instruction = f"You are Abhay. Reply in very short, natural Hinglish. Answer directly. {relationship}"

    # Inject notes / facts into the prompt
    current_facts = read_rag_memory()
    if current_facts:
        system_instruction += f"\n\nCRITICAL CURRENT FACTS ABOUT YOU:\n{current_facts}"

    # 4. Setup the messages
    messages = [
        {"role": "system", "content": f"{system_instruction}\n\nRecent context:\n{history_text}"},
        {"role": "user", "content": msg.text}
    ]

    # 5. Generation (single pass)
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=64
    )

    reply = response["choices"][0]["message"]["content"].strip()
    log.info("Reply: %s", reply)

    # 6. Save both messages ONLY after a reply was generated successfully
    save_message(contact_id, "user", msg.text, msg.app_source)
    save_message(contact_id, "assistant", reply, msg.app_source)
    return {"status": "success", "reply": reply}