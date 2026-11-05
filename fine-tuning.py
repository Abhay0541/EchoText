"""
Project EchoText - Kaggle QLoRA Fine-Tuning Script
This script uses the unsloth framework to fine-tune Llama-3.2-3B-Instruct 
and exports it as a 4-bit GGUF file for local inference.

Note: If running in a Kaggle/Jupyter Notebook, run these pip installs in a separate cell first:
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
max_seq_length = 2048
dtype = None # None auto-detects based on hardware
load_in_4bit = True # Use 4-bit quantization to fit in GPU memory

print("Loading Base Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ==========================================
# 2. Attach LoRA Adapters
# ==========================================
print("Attaching LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ==========================================
# 3. Dataset Formatting
# ==========================================
print("Formatting Dataset...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
)

def formatting_prompts_func(examples):
    conversations = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in conversations]
    return { "text" : texts, }

# NOTE: Ensure "your_chat_data.jsonl" is uploaded to your Kaggle environment
dataset_path = "your_chat_data.jsonl" 
try:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
except FileNotFoundError:
    print(f"ERROR: Dataset {dataset_path} not found. Please upload your JSONL file.")
    exit()

# ==========================================
# 4. Training Configuration
# ==========================================
print("Initializing Trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Increase this (e.g., 300-500) for a full training run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# ==========================================
# 5. Execute Training
# ==========================================
print("Starting Training...")
trainer_stats = trainer.train()

# ==========================================
# 6. Save & Export to GGUF
# ==========================================
print("Training Complete. Saving LoRA Adapters...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("Exporting Model to 4-bit GGUF format for local edge inference...")
# This will output a file like 'echo_text_llama3.2_3B-unsloth-Q4_K_M.gguf'
model.save_pretrained_gguf("echo_text_llama3.2_3B", tokenizer, quantization_method = "q4_k_m")

print("Export complete! Download your .gguf file to use with llama.cpp.")
