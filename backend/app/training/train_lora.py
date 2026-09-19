"""
LoRA fine-tuning of microsoft/Phi-3.5-mini-instruct on your own chat style.

This is meant to be run standalone (not imported by the API) on a machine
with a GPU — a free Colab T4 is enough for a mini model at this LoRA rank.

    pip install torch transformers peft datasets trl accelerate bitsandbytes
    python train_lora.py --chat_export ./my_chat.txt --target_user "Abhay"

What LoRA is actually buying you here, briefly: full fine-tuning of a 3.8B
model means updating every weight (~15GB+ of gradients/optimizer state, needs
a serious GPU). LoRA freezes the base model and injects small trainable
rank-decomposition matrices into the attention layers instead — typically
<1% of the parameters. You get most of the style adaptation at a fraction of
the memory/compute, and the output is a small (~10-50MB) adapter file rather
than a full re-saved model, which is also exactly what you want for
"download this to a phone for offline inference" later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.data.whatsapp_parser import build_training_pairs, parse_chat_export


def build_dataset_jsonl(chat_export_path: str, target_user: str, out_path: str) -> int:
    """Parse the raw export and write LoRA-ready examples to a .jsonl file."""
    raw = Path(chat_export_path).read_text(encoding="utf-8", errors="ignore")
    messages = parse_chat_export(raw)
    pairs = build_training_pairs(messages, target_user=target_user)

    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.as_prompt_completion()) + "\n")
    return len(pairs)


def run_training(dataset_path: str, base_model: str, output_dir: str, epochs: int, rank: int):
    # Heavy imports deferred so `--help` and dataset-building work without
    # torch/transformers installed.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, torch_dtype=torch.float16)

    # rank (r) trades adapter expressiveness for size/speed; 8-16 is the
    # standard sweet spot for a "learn my texting style" task, which is a
    # narrow style-transfer problem, not new knowledge — you don't need a
    # high rank for that.
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(example):
        text = example["prompt"] + " " + example["completion"] + tokenizer.eos_token
        return tokenizer(text, truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir} — point ADAPTER_PATH here in app/config.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset from a WhatsApp export and LoRA-finetune Phi-3.5-mini.")
    parser.add_argument("--chat_export", required=True, help="Path to WhatsApp _chat.txt export")
    parser.add_argument("--target_user", required=True, help="Exact display name you appear as in the export")
    parser.add_argument("--base_model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--output_dir", default="./adapter")
    parser.add_argument("--dataset_out", default="./training_pairs.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dataset_only", action="store_true", help="Just build the .jsonl, skip training")
    args = parser.parse_args()

    n = build_dataset_jsonl(args.chat_export, args.target_user, args.dataset_out)
    print(f"Built {n} training pairs -> {args.dataset_out}")

    if not args.dataset_only:
        run_training(args.dataset_out, args.base_model, args.output_dir, args.epochs, args.rank)
