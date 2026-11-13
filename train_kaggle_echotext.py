# ==============================================================================
# CELL 1: DEPENDENCY INSTALLATION (PURE PYTORCH + TRANSFORMERS + PEFT)
# ==============================================================================
# This setup avoids all conflicts between trl, datasets, and huggingface-hub
# by using native PyTorch Dataset + standard transformers.Trainer!
# ==============================================================================

!pip install -q --no-warn-conflicts "peft>=0.13.0" "bitsandbytes>=0.44.0" "accelerate>=0.34.0"
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git > /dev/null 2>&1
!pip install -q -r llama.cpp/requirements.txt --no-warn-conflicts > /dev/null 2>&1

print("[OK] Clean environment ready! No conflicting packages needed.")


# ==============================================================================
# CELL 2: COMPLETE TRAINING & GGUF EXPORT (ZERO 'DATASETS' OR 'TRL' DEPENDENCY)
# ==============================================================================
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
# 1. Base Model & Directories
# ------------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

OUTPUT_DIR = "/kaggle/working/echotext_lora_output"
MERGED_DIR = "/kaggle/working/echotext_merged_model"
GGUF_DIR = "/kaggle/working/echotext_gguf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(GGUF_DIR, exist_ok=True)

print("=" * 80)
print(f"EchoText Native Fine-Tuning: {MODEL_ID}")
print("=" * 80)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ------------------------------------------------------------------------------
# 2. Locate Dataset File
# ------------------------------------------------------------------------------
print("\n[1/6] Searching for echotext_1000_training_dataset.jsonl...")
dataset_files = glob.glob("/kaggle/input/**/echotext_1000_training_dataset.jsonl", recursive=True)
if not dataset_files:
    dataset_files = glob.glob("/kaggle/input/**/*.jsonl", recursive=True)
if not dataset_files:
    dataset_files = glob.glob("*.jsonl")

if not dataset_files:
    raise FileNotFoundError("Could not find dataset! Please upload 'echotext_1000_training_dataset.jsonl' via '+ Add Input'.")

DATASET_PATH = dataset_files[0]
print(f"Found dataset at: {DATASET_PATH}")

# ------------------------------------------------------------------------------
# 3. Load Tokenizer & 4-Bit QLoRA Model
# ------------------------------------------------------------------------------
print(f"\n[2/6] Loading tokenizer and 4-bit base model ({MODEL_ID})...")

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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ------------------------------------------------------------------------------
# 4. Pure PyTorch Dataset (Bypasses 'datasets' library completely)
# ------------------------------------------------------------------------------
print("\n[3/6] Building PyTorch dataset from JSONL...")

class EchoTextDataset(Dataset):
    def __init__(self, jsonl_path, tok, max_len=1024):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages = obj.get("messages", [])
                    if messages:
                        formatted_text = tok.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        self.samples.append(formatted_text)
                except Exception:
                    continue
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = list(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

train_dataset = EchoTextDataset(DATASET_PATH, tokenizer)
print(f"Dataset successfully compiled: {len(train_dataset)} examples.")

# ------------------------------------------------------------------------------
# 5. Training with Standard Hugging Face Trainer
# ------------------------------------------------------------------------------
print("\n[4/6] Starting Trainer execution...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
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
print("\n[OK] Training completed! Saving LoRA adapter weights...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ------------------------------------------------------------------------------
# 6. Merge LoRA Weights into 16-Bit Base Model
# ------------------------------------------------------------------------------
print("\n[5/6] Merging LoRA adapter with base model into 16-bit...")

del model
del trainer
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

print(f"Saving merged 16-bit model to: {MERGED_DIR}")
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

del base_model
del merged_model
gc.collect()

# ------------------------------------------------------------------------------
# 7. Convert to GGUF (FP16 & Q4_K_M)
# ------------------------------------------------------------------------------
print("\n[6/6] Converting merged model to GGUF format...")

fp16_gguf = f"{GGUF_DIR}/echotext-qwen2.5-7b-fp16.gguf"
final_q4_gguf = f"{GGUF_DIR}/echotext-qwen2.5-7b-Q4_K_M.gguf"

import subprocess

# Convert to FP16 GGUF
subprocess.run([
    sys.executable, "llama.cpp/convert_hf_to_gguf.py",
    MERGED_DIR,
    "--outfile", fp16_gguf,
    "--outtype", "f16"
], check=True)
print(f"FP16 GGUF generated at: {fp16_gguf}")

# Compile llama-quantize binary
print("Compiling llama-quantize...")
subprocess.run(["cmake", "-B", "llama.cpp/build", "llama.cpp"], check=False)
subprocess.run(["cmake", "--build", "llama.cpp/build", "--config", "Release", "--target", "llama-quantize"], check=False)

quantize_bin = "llama.cpp/build/bin/llama-quantize"
if not os.path.exists(quantize_bin):
    quantize_bin = "llama.cpp/llama-quantize"

if os.path.exists(quantize_bin):
    print("Quantizing GGUF to Q4_K_M...")
    subprocess.run([quantize_bin, fp16_gguf, final_q4_gguf, "Q4_K_M"], check=True)
    if os.path.exists(fp16_gguf):
        os.remove(fp16_gguf)  # Clean up FP16 to save disk
    print(f"\n[SUCCESS] Final Quantized GGUF saved to: {final_q4_gguf}")
    target_download_path = final_q4_gguf
else:
    print(f"\n[NOTICE] Quantizer binary skipped; using FP16 GGUF: {fp16_gguf}")
    target_download_path = fp16_gguf

# ------------------------------------------------------------------------------
# 8. Interactive Download Link
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TRAINING & EXPORT FINISHED SUCCESSFULLY!")
print("Click the link below to download your model file:")
print("=" * 80)
display(FileLink(target_download_path))
