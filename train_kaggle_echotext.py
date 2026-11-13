# ==============================================================================
# ECHOTEXT QWEN2.5-7B LORA FINE-TUNING — KAGGLE NOTEBOOK
# ==============================================================================
# HOW TO USE THIS NOTEBOOK:
#   1. Paste CELL 1 into a new Kaggle code cell and RUN it.
#   2. After Cell 1 finishes, go to: Session ▸ Restart Session  (or use Ctrl+M .)
#   3. THEN run CELL 2 (the main training cell).
#
# This two-cell approach is REQUIRED because Kaggle caches stale .pyc bytecode
# in memory and the only reliable fix is a kernel restart after pip install.
# ==============================================================================

# ==============================================================================
# ██████████████████████  CELL 1 — INSTALL & RESTART  ██████████████████████████
# ==============================================================================
# Run this cell, then do: Session ▸ Restart Session, THEN run Cell 2.

import subprocess, sys

print("=" * 70)
print("CELL 1: Installing packages...")
print("=" * 70)

packages = [
    "transformers==4.47.1",
    "peft>=0.13.0",
    "bitsandbytes>=0.44.0",
    "accelerate>=0.34.0",
]

for pkg in packages:
    print(f"  Installing {pkg}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts",
         "--force-reinstall", pkg],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [WARN] {pkg}: {result.stderr[-200:]}")
    else:
        print(f"  [OK] {pkg}")

# Clone llama.cpp for GGUF conversion
print("\nCloning llama.cpp for GGUF conversion...")
subprocess.run(
    ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"],
    capture_output=True
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "llama.cpp/requirements.txt"],
    capture_output=True
)

print("\n" + "=" * 70)
print("✅ CELL 1 DONE!")
print("")
print("⚠️  NOW DO THIS:")
print("   Session  ▸  Restart Session   (top menu bar)")
print("   Then run CELL 2.")
print("=" * 70)


# ==============================================================================
# ██████████████████████  CELL 2 — FULL TRAINING  ██████████████████████████████
# ==============================================================================
# Only run this AFTER restarting the session from Cell 1!

import os
import glob
import sys
import gc
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from IPython.display import FileLink, display

# ------------------------------------------------------------------------------
# 1. Config
# ------------------------------------------------------------------------------
MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/kaggle/working/echotext_lora_output"
MERGED_DIR = "/kaggle/working/echotext_merged_model"
GGUF_DIR   = "/kaggle/working/echotext_gguf"

for d in [OUTPUT_DIR, MERGED_DIR, GGUF_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 80)
print(f"EchoText Fine-Tuning  |  Base: {MODEL_ID}")
print("=" * 80)
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  ({props.total_memory/1e9:.1f} GB)")

# ------------------------------------------------------------------------------
# 2. Find dataset
# ------------------------------------------------------------------------------
print("\n[1/6] Locating dataset...")
dataset_files = (
    glob.glob("/kaggle/input/**/echotext_1000_training_dataset.jsonl", recursive=True) or
    glob.glob("/kaggle/input/**/*.jsonl", recursive=True) or
    glob.glob("*.jsonl")
)
if not dataset_files:
    raise FileNotFoundError(
        "Cannot find dataset!\n"
        "Please add it via '+ Add Input' → Dataset → search 'knowrajput/chatsdata-1000'."
    )

DATASET_PATH = dataset_files[0]
print(f"  Found: {DATASET_PATH}")

# ------------------------------------------------------------------------------
# 3. Tokenizer + 4-bit Model
# ------------------------------------------------------------------------------
print(f"\n[2/6] Loading tokenizer & 4-bit model ({MODEL_ID})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ------------------------------------------------------------------------------
# 4. Pure PyTorch Dataset (no `datasets` library needed)
# ------------------------------------------------------------------------------
print("\n[3/6] Building dataset...")

class EchoTextDataset(Dataset):
    def __init__(self, jsonl_path: str, tok, max_len: int = 1024):
        self.tok = tok
        self.max_len = max_len
        self.samples: list[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages = obj.get("messages", [])
                    if messages:
                        text = tok.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        self.samples.append(text)
                except Exception:
                    continue
        print(f"  Loaded {len(self.samples)} training examples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tok(
            self.samples[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        ids = enc["input_ids"]
        return {
            "input_ids": ids,
            "attention_mask": enc["attention_mask"],
            "labels": list(ids),
        }

train_dataset = EchoTextDataset(DATASET_PATH, tokenizer)

# ------------------------------------------------------------------------------
# 5. Train
# ------------------------------------------------------------------------------
print("\n[4/6] Starting training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",
    report_to="none",
    dataloader_pin_memory=False,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
print("[OK] Training complete! Saving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ------------------------------------------------------------------------------
# 6. Merge LoRA → 16-bit
# ------------------------------------------------------------------------------
print("\n[5/6] Merging LoRA into base model (16-bit, CPU)...")

del model, trainer
gc.collect()
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print(f"  Merged model saved to: {MERGED_DIR}")

del base_model, merged_model
gc.collect()

# ------------------------------------------------------------------------------
# 7. GGUF conversion
# ------------------------------------------------------------------------------
print("\n[6/6] Converting to GGUF...")

import subprocess

fp16_gguf  = f"{GGUF_DIR}/echotext-qwen2.5-7b-fp16.gguf"
q4_gguf    = f"{GGUF_DIR}/echotext-qwen2.5-7b-Q4_K_M.gguf"

# Convert merged model → FP16 GGUF
print("  Step 6a: HF → FP16 GGUF...")
r = subprocess.run(
    [sys.executable, "llama.cpp/convert_hf_to_gguf.py",
     MERGED_DIR, "--outfile", fp16_gguf, "--outtype", "f16"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print(f"  [WARN] convert_hf_to_gguf.py stderr:\n{r.stderr[-500:]}")
else:
    print(f"  FP16 GGUF saved: {fp16_gguf}")

# Compile llama-quantize
print("  Step 6b: Compiling llama-quantize...")
subprocess.run(
    ["cmake", "-B", "llama.cpp/build", "llama.cpp",
     "-DCMAKE_BUILD_TYPE=Release"],
    capture_output=True
)
subprocess.run(
    ["cmake", "--build", "llama.cpp/build",
     "--config", "Release", "--target", "llama-quantize", "-j4"],
    capture_output=True
)

# Find binary
quantize_bin = next(
    (p for p in [
        "llama.cpp/build/bin/llama-quantize",
        "llama.cpp/build/llama-quantize",
        "llama.cpp/llama-quantize",
    ] if os.path.exists(p)),
    None
)

if quantize_bin and os.path.exists(fp16_gguf):
    print("  Step 6c: Quantising to Q4_K_M...")
    r = subprocess.run(
        [quantize_bin, fp16_gguf, q4_gguf, "Q4_K_M"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        os.remove(fp16_gguf)
        target = q4_gguf
        print(f"  [OK] Q4_K_M GGUF: {target}")
    else:
        print(f"  [WARN] Quantise failed; keeping FP16. stderr:\n{r.stderr[-300:]}")
        target = fp16_gguf
else:
    print("  [NOTICE] llama-quantize not found; keeping FP16 GGUF.")
    target = fp16_gguf if os.path.exists(fp16_gguf) else None

# ------------------------------------------------------------------------------
# 8. Done — download link
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎉 TRAINING & EXPORT COMPLETE!")
print("=" * 80)
if target and os.path.exists(target):
    size_mb = os.path.getsize(target) / 1e6
    print(f"  File : {target}")
    print(f"  Size : {size_mb:.1f} MB")
    display(FileLink(target))
else:
    print("  [WARN] Target GGUF not found — check logs above.")
