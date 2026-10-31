import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from database import get_or_create_contact, save_message, get_chat_history

app = FastAPI(title="Project EchoText Proxy")

print("Booting up... Loading 3B GGUF into memory...")
# Keeping n_gpu_layers=0 for the 3B model on your standard hardware for now
llm = Llama(
    model_path="./Llama-3.2-3B-Instruct.Q4_K_M.gguf", 
    n_ctx=2048,
    n_gpu_layers=0, 
    verbose=False
)
print("Brain is online and listening!")

class IncomingMessage(BaseModel):
    app_source: str
    sender_id: str
    text: str

# ---> NEW: UI Memory Reader <---
def read_rag_memory():
    """Scans the exclusive folder managed by the Streamlit UI and extracts facts."""
    rag_folder = "rag_memory"
    rag_context = ""
    
    if not os.path.exists(rag_folder):
        return ""
        
    for filename in os.listdir(rag_folder):
        file_path = os.path.join(rag_folder, filename)
        # Read .txt files (like your manual facts or notes uploaded via UI)
        if filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    rag_context += f"\n[Fact from {filename}]: {f.read().strip()}\n"
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                
    return rag_context

@app.post("/chat")
async def receive_message(msg: IncomingMessage):
    print(f"\n[{msg.app_source}] {msg.sender_id}: {msg.text}")
    
    # 1. Global State Check (Reads your Streamlit Toggle)
    try:
        with open("proxy_status.txt", "r") as f:
            if f.read().strip().upper() == "AVAILABLE":
                print("Status is AVAILABLE. Skipping auto-reply.")
                return {"status": "skipped"}
    except FileNotFoundError:
        pass # Default to BUSY if the file doesn't exist

    # 2. Database Memory
    contact_id = get_or_create_contact(msg.app_source, msg.sender_id)
    history_text = get_chat_history(contact_id, limit=4)
    save_message(contact_id, "user", msg.text, msg.app_source)
    
    # 3. Dynamic Persona
    if msg.sender_id.lower() == "aman99":
        relationship = "Sender is Aman, your MITS college classmate. Use heavy Gwalior/college slang."
    else:
        relationship = "Sender is a general contact. Keep it casual and short."
        
    system_instruction = f"You are Abhay. Reply in very short, natural Hinglish. Answer directly. {relationship}"
    
    # ---> NEW: Inject RAG Facts into the AI Brain <---
    current_facts = read_rag_memory()
    if current_facts:
        system_instruction += f"\n\nCRITICAL CURRENT FACTS ABOUT YOU:\n{current_facts}"
    
    # 4. Setup the Agent's Messages (No JSON tool schemas to prevent hallucinations)
    messages = [
        {"role": "system", "content": f"{system_instruction}\n\nRecent context:\n{history_text}"},
        {"role": "user", "content": msg.text}
    ]
    
    # 5. Lightning Generation (Single Pass)
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=64
    )
    
    reply = response["choices"][0]["message"]["content"].strip()
    print(f"Reply: {reply}")
    
    # 6. Save Assistant Reply and Return
    save_message(contact_id, "assistant", reply, msg.app_source)
    return {"status": "success", "reply": reply}