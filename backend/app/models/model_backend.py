"""
Model backend abstraction.

Both the classify() and generate() calls in the router go through this
interface. That's what lets the whole project run and be demoed on a laptop
with no GPU (via TemplateBackend) while being a drop-in swap to the real
fine-tuned Phi-3.5-mini once you have LoRA weights trained (via
PhiLoRABackend) — nothing in routing/, rag/, or api/ needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.retriever import RetrievedExample


class ModelBackend(ABC):
    @abstractmethod
    def classify(self, message: str, examples: list[RetrievedExample]) -> str:
        """Return one of: quick_reply | needs_context | sensitive | unknown"""

    @abstractmethod
    def generate(self, message: str, examples: list[RetrievedExample], route: str) -> str:
        """Return the actual reply text."""


PROMPT_TEMPLATE = """You are texting as the user, matching their tone exactly.

Similar past exchanges:
{examples}

New message: "{message}"

Classify this message's route (quick_reply / needs_context / sensitive), then reply in the user's voice."""


class TemplateBackend(ModelBackend):
    """
    Zero-dependency, zero-download fallback backend.

    This is NOT meant to imitate an LLM's reasoning — it's a honest,
    inspectable stand-in so the full pipeline (parsing -> RAG -> routing ->
    generation -> API -> frontend) is runnable and demoable today, before
    the LoRA fine-tune exists. It classifies using simple heuristics and
    generates by adapting the closest retrieved past reply, which is
    actually a reasonable retrieval-only baseline to benchmark the fine-tuned
    model against later.
    """

    _QUESTION_WORDS = ("?", "when", "where", "why", "how", "what time")

    def classify(self, message: str, examples: list[RetrievedExample]) -> str:
        if not examples:
            return "unknown"
        best_score = examples[0].score
        if best_score < 0.15:
            return "unknown"
        if any(w in message.lower() for w in self._QUESTION_WORDS) and best_score < 0.4:
            return "needs_context"
        return "quick_reply"

    def generate(self, message: str, examples: list[RetrievedExample], route: str) -> str:
        if not examples:
            return "(no confident reply — needs a human)"
        # Retrieval-only baseline: reuse the closest past reply verbatim.
        # A real fine-tuned model would paraphrase/adapt this instead of
        # copying it outright — that gap is exactly what LoRA training buys you.
        return examples[0].reply


class PhiLoRABackend(ModelBackend):  # pragma: no cover - requires GPU + trained weights
    """
    Real backend: Microsoft Phi-3.5-mini-instruct + your trained LoRA adapter.

    Not wired up by default because it needs (a) the base model downloaded
    from Hugging Face and (b) `adapter/` produced by
    backend/app/training/train_lora.py — neither of which exists until you
    run the training step on your own machine/Colab with a GPU.

    To activate: pip install torch transformers peft accelerate, then set
    MODEL_BACKEND=phi_lora in .env (see app/config.py) and point
    ADAPTER_PATH at your trained weights.
    """

    def __init__(self, base_model: str = "microsoft/Phi-3.5-mini-instruct", adapter_path: str = "./adapter"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self._model = PeftModel.from_pretrained(base, adapter_path)
        self._model.eval()

    def _run(self, prompt: str, max_new_tokens: int) -> str:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    def _format_examples(self, examples: list[RetrievedExample]) -> str:
        return "\n---\n".join(f"{e.context}\nReply: {e.reply}" for e in examples) or "(none found)"

    def classify(self, message: str, examples: list[RetrievedExample]) -> str:
        prompt = PROMPT_TEMPLATE.format(examples=self._format_examples(examples), message=message)
        prompt += "\n\nRoute:"
        result = self._run(prompt, max_new_tokens=8).lower()
        for route in ("quick_reply", "needs_context", "sensitive"):
            if route.replace("_", " ") in result or route in result:
                return route
        return "unknown"

    def generate(self, message: str, examples: list[RetrievedExample], route: str) -> str:
        prompt = PROMPT_TEMPLATE.format(examples=self._format_examples(examples), message=message)
        prompt += "\n\nReply:"
        return self._run(prompt, max_new_tokens=80)


def get_model_backend(name: str = "template") -> ModelBackend:
    """Factory used by the API layer — reads MODEL_BACKEND from config/env."""
    if name == "phi_lora":
        return PhiLoRABackend()
    return TemplateBackend()
