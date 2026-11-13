"""
================================================================================
EchoText LoRA Fine-Tuning Script for Kaggle (GPU: T4 x2 or P100)
================================================================================
This script fine-tunes Qwen2.5-7B-Instruct (or Llama-3.1-8B-Instruct) on your
echotext_1000_training_dataset.jsonl using 4-bit QLoRA and exports a GGUF model.

Prerequisites on Kaggle:
1. In your Kaggle Notebook settings (right sidebar):
   - Accelerator: GPU T4 x2 (or GPU P100)
   - Internet: Enabled (Turn ON)
2. Add your dataset:
   - Click "+ Add Input" -> Upload "echotext_1000_training_dataset.jsonl"
================================================================================
"""

# ==========================================
# CELL 1: INSTALL DEPENDENCIES
# ==========================================
import subprocess
import sys

print("[1/7] Installing training dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers>=4.45.0", "datasets", "accelerate>=0.34.0",
    "peft>=0.13.0", "trl>=0.11.0", "bitsandbytes>=0.44.0"
], check=True)

# Clone llama.cpp for GGUF conversion
print("Cloning llama.cpp for GGUF export...")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "llama.cpp/requirements.txt"], check=False)
print("Dependencies installed successfully!")


# ==========================================
# CELL 2: CONFIGURATION & HARDWARE CHECK
# ==========================================
import os
import glob
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Base Model Selection
# Recommended: "Qwen/Qwen2.5-7B-Instruct" (Open access, state-of-the-art multilingual & Hindi/English)
# Alternative: "meta-llama/Meta-Llama-3.1-8B-Instruct" (Requires Hugging Face Token)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

OUTPUT_DIR = "/kaggle/working/echotext_lora_output"
MERGED_DIR = "/kaggle/working/echotext_merged_model"
GGUF_DIR = "/kaggle/working/echotext_gguf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(GGUF_DIR, exist_ok=True)

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ==========================================
# CELL 3: LOCATE & LOAD DATASET
# ==========================================
print("\n[2/7] Locating dataset...")
# Search for the jsonl file in Kaggle input or current directory
dataset_files = glob.glob("/kaggle/input/**/echotext_1000_training_dataset.jsonl", recursive=True)
if not dataset_files:
    dataset_files = glob.glob("/kaggle/input/**/*.jsonl", recursive=True)
if not dataset_files:
    dataset_files = glob.glob("*.jsonl")

if not dataset_files:
    raise FileNotFoundError("Could not find echotext_1000_training_dataset.jsonl! Please upload it via '+ Add Input'.")

DATASET_PATH = dataset_files[0]
print(f"Using dataset from: {DATASET_PATH}")

raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Dataset successfully loaded: {len(raw_dataset)} examples.")


# ==========================================
# CELL 4: LOAD TOKENIZER & MODEL (4-BIT QLORA)
# ==========================================
print(f"\n[3/7] Loading tokenizer & model ({MODEL_ID})...")

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

# LoRA Configuration
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


# ==========================================
# CELL 5: DATASET PREPARATION WITH CHAT TEMPLATE
# ==========================================
print("\n[4/7] Applying chat template to training data...")

def format_conversations(batch):
    formatted_texts = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)
    return {"text": formatted_texts}

train_dataset = raw_dataset.map(format_conversations, batched=True, remove_columns=raw_dataset.column_names)
print(f"Formatted dataset sample length: {len(train_dataset[0]['text'])} characters.")


# ==========================================
# CELL 6: TRAINING ARGUMENTS & SFTTRAINER
# ==========================================
print("\n[5/7] Starting LoRA Fine-Tuning...")

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

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()
print("Training completed successfully! Saving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# ==========================================
# CELL 7: MERGE LORA WEIGHTS INTO BASE MODEL
# ==========================================
print("\n[6/7] Merging LoRA adapter with base model for GGUF export...")

# Free up 4-bit model from GPU VRAM
del model
del trainer
torch.cuda.empty_cache()

# Load base model in 16-bit to cleanly merge weights
from peft import PeftModel

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


# ==========================================
# CELL 8: CONVERT TO GGUF FORMAT
# ==========================================
print("\n[7/7] Converting merged model to GGUF (FP16 & Q4_K_M)...")

# Step 1: Convert to FP16 GGUF
fp16_gguf = f"{GGUF_DIR}/echotext-qwen2.5-7b-fp16.gguf"
subprocess.run([
    sys.executable, "llama.cpp/convert_hf_to_gguf.py",
    MERGED_DIR,
    "--outfile", fp16_gguf,
    "--outtype", "f16"
], check=True)

print(f"FP16 GGUF generated at: {fp16_gguf}")

# Step 2: Build llama.cpp quantizer
print("Building llama-quantize binary...")
subprocess.run(["cmake", "-B", "llama.cpp/build", "llama.cpp"], check=False)
subprocess.run(["cmake", "--build", "llama.cpp/build", "--config", "Release", "--target", "llama-quantize"], check=False)

quantize_bin = "llama.cpp/build/bin/llama-quantize"
if not os.path.exists(quantize_bin):
    quantize_bin = "llama.cpp/llama-quantize"

final_q4_gguf = f"{GGUF_DIR}/echotext-qwen2.5-7b-Q4_K_M.gguf"

if os.path.exists(quantize_bin):
    print(f"Quantizing to Q4_K_M...")
    subprocess.run([quantize_bin, fp16_gguf, final_q4_gguf, "Q4_K_M"], check=True)
    # Remove large FP16 file to save disk space
    if os.path.exists(fp16_gguf):
        os.remove(fp16_gguf)
    print(f"\nSUCCESS! Quantized GGUF saved to: {final_q4_gguf}")
else:
    print(f"\nNotice: Quantizer binary build skipped. You can directly download the FP16 GGUF ({fp16_gguf}) or quantize locally.")

print("\n" + "=" * 80)
print("ALL DONE! Check your /kaggle/working/ directory to download your GGUF model.")
print("=" * 80)
