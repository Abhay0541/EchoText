# EchoText

**Integration of AI in Personalized Messaging** — an on-device auto-reply
system that learns your texting style from your own chat history and
replies the way you would.

Supervisor: Dr. Manojeet Roy

## Why this exists

Generic AI auto-reply tools (Gmail's Smart Reply, etc.) generate replies in
a bland, one-size-fits-all voice. EchoText instead fine-tunes a small
language model (Phi-3.5-mini-instruct) on *your* actual message history via
LoRA, so replies sound like you specifically — while keeping the whole
pipeline light enough to eventually run fully offline, on-device.

## How it actually works (the pipeline)

```
WhatsApp export (.txt)
        │
        ▼
 whatsapp_parser.py  ──►  (context, reply) pairs
        │                         │
        ▼                         ▼
   ChatIndex (RAG)         train_lora.py (offline, needs GPU)
   TF-IDF similarity              │
   search over your                ▼
   past replies              LoRA adapter (~10–50MB)
        │                         │
        └───────────┬─────────────┘
                     ▼
             MessageRouter
     Call 1: classify (quick_reply / needs_context / sensitive / unknown)
     Call 2: generate (only for quick_reply / needs_context)
                     │
                     ▼
        FastAPI  ──►  React chat UI
                 ──►  Telegram / Discord bots
                 ──►  Gmail draft creator
```

Two design decisions worth calling out (both explained in more depth as
comments at the point they're implemented):

1. **RAG + LoRA are doing different jobs, on purpose.** LoRA fine-tuning
   teaches *style* (word choice, tone, message length). It's bad at
   memorizing specific facts or one-off replies. RAG retrieval pulls the most
   similar real past exchange and hands it to the model as grounding — style
   from the fine-tune, specifics from retrieval. See
   `backend/app/rag/retriever.py`.
2. **One model, called twice, not two separate models.** A classify call
   decides the route; a generate call (only reached for safe routes)
   produces the reply. This halves the number of checkpoints to manage
   on-device and keeps both calls sharing the same fine-tuned "voice". See
   `backend/app/routing/router.py`.

## What's real vs. what's a stand-in right now

Everything **runs today, with zero downloads or GPU**, using a
`TemplateBackend` that does retrieval-based reply generation (reuses your
closest past reply) instead of a real LLM — this keeps the full
parse → retrieve → route → generate → serve pipeline demoable on a laptop.

The **real** Phi-3.5 + LoRA path is fully implemented, not mocked:
`app/training/train_lora.py` is a genuine PEFT/LoRA fine-tuning script, and
`PhiLoRABackend` in `app/models/model_backend.py` is genuine inference code.
Both are just not wired up as the *default*, because they need a GPU and a
trained adapter you don't have until you run the training step yourself.
Swapping to it is a one-line env change (`MODEL_BACKEND=phi_lora`) — nothing
in routing, RAG, or the API changes.

## Running it

### Backend
```bash
cd backend
pip install -r requirements.txt   # core deps only; see requirements.txt for optional extras
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Then open http://localhost:5173, upload a WhatsApp `.txt` export (Chat →
More → Export chat → Without media), enter your display name exactly as it
appears in the chat, and start sending test messages.

### Or with Docker
```bash
docker compose up --build
```

### Running the test suite
```bash
cd backend
pip install pytest
pytest -v
```

## Project layout

```
backend/
  app/
    data/whatsapp_parser.py     parses WhatsApp exports into training pairs
    rag/retriever.py            TF-IDF retrieval index (swappable to embeddings)
    routing/router.py           two-call routing + safety gate
    models/model_backend.py     TemplateBackend (default) + PhiLoRABackend (real)
    training/train_lora.py      real LoRA fine-tuning script (run separately, needs GPU)
    integrations/               Telegram bot, Discord bot, Gmail draft creator
    main.py                     FastAPI app
  tests/test_pipeline.py        pytest suite for parser/RAG/routing
frontend/
  src/                          React + Vite chat UI with routing transparency
```

## Roadmap (pipeline stages ahead, per the project plan)

- [x] Data collection & WhatsApp parsing
- [x] Hosted-API prototype proving style-matching (earlier phase, via Gemini)
- [x] RAG retrieval layer
- [x] Routing logic (single model, two calls)
- [ ] Run `train_lora.py` on real chat data to produce the first adapter
- [ ] Swap `MODEL_BACKEND` to `phi_lora` and benchmark against the
      retrieval-only baseline
- [ ] Package for fully offline, on-device inference (e.g. via GGUF /
      llama.cpp export of the merged LoRA weights)

## A note on privacy

This project processes your own private message history. `SESSIONS` are
in-memory only (nothing is persisted to disk beyond what you explicitly
save via `train_lora.py`'s dataset output), and the Gmail integration
deliberately only ever creates *drafts* — it never sends on your behalf.
