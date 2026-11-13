# ==============================================================================
# CELL 1: RUN THIS CELL FIRST TO INSTALL DEPENDENCIES & RESTART KERNEL
# ==============================================================================
# In Kaggle Notebook Settings (right sidebar):
#   - Accelerator: GPU T4 x2 (or GPU P100)
#   - Internet: Enabled (Turn ON)
# ==============================================================================

!pip install -q -U --force-reinstall "transformers>=4.48.0" "accelerate>=0.34.0" "peft>=0.13.0" "trl>=0.11.0" "bitsandbytes>=0.44.0" "datasets"
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
!pip install -q -r llama.cpp/requirements.txt

import os
print("\n[OK] Dependencies installed! Restarting kernel to clear import cache...")
os._exit(0)  # Automatically restarts kernel; session restart notice is normal!


# ==============================================================================
# CELL 2: COMPLETE END-TO-END TRAINING & GGUF EXPORT (RUN AFTER CELL 1)
# ==============================================================================
import os
import glob
import sys
import gc
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
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
print(f"EchoText LoRA Fine-Tuning: {MODEL_ID}")
print("=" * 80)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ------------------------------------------------------------------------------
# 2. Locate & Load Dataset
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

raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Loaded {len(raw_dataset)} training examples.")

# ------------------------------------------------------------------------------
# 3. Load Tokenizer & 4-Bit Base Model (QLoRA)
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

# ------------------------------------------------------------------------------
# 4. Format Dataset with Chat Template
# ------------------------------------------------------------------------------
print("\n[3/6] Applying tokenizer chat template...")

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
print("Sample formatted conversation preview:\n" + train_dataset[0]["text"][:280] + "...\n")

# ------------------------------------------------------------------------------
# 5. Training with SFTTrainer
# ------------------------------------------------------------------------------
print("\n[4/6] Starting SFTTrainer execution...")

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

# Step 1: Convert to FP16 GGUF
import subprocess

subprocess.run([
    sys.executable, "llama.cpp/convert_hf_to_gguf.py",
    MERGED_DIR,
    "--outfile", fp16_gguf,
    "--outtype", "f16"
], check=True)
print(f"FP16 GGUF generated at: {fp16_gguf}")

# Step 2: Build llama.cpp quantizer binary
print("Compiling llama-quantize binary...")
subprocess.run(["cmake", "-B", "llama.cpp/build", "llama.cpp"], check=False)
subprocess.run(["cmake", "--build", "llama.cpp/build", "--config", "Release", "--target", "llama-quantize"], check=False)

quantize_bin = "llama.cpp/build/bin/llama-quantize"
if not os.path.exists(quantize_bin):
    quantize_bin = "llama.cpp/llama-quantize"

if os.path.exists(quantize_bin):
    print("Quantizing GGUF to Q4_K_M (optimal speed/size balance)...")
    subprocess.run([quantize_bin, fp16_gguf, final_q4_gguf, "Q4_K_M"], check=True)
    if os.path.exists(fp16_gguf):
        os.remove(fp16_gguf)  # Free up disk space
    print(f"\n[SUCCESS] Final Quantized GGUF saved to: {final_q4_gguf}")
    target_download_path = final_q4_gguf
else:
    print(f"\n[NOTICE] Quantizer binary skipped; using FP16 GGUF: {fp16_gguf}")
    target_download_path = fp16_gguf

# ------------------------------------------------------------------------------
# 8. Download Link
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TRAINING & EXPORT FINISHED SUCCESSFULLY!")
print("Click the link below to download your model file:")
print("=" * 80)
display(FileLink(target_download_path))
