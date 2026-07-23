"""
05_train_dpo.py

Fine-tune Qwen2.5-0.5B-Instruct using DPO + QLoRA.

Input:
    data/final/dpo_dataset.jsonl

Output:
    models/dpo/
"""

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig

from trl import (
    DPOConfig,
    DPOTrainer,
)

# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_PATH = "data/final/dpo_dataset.jsonl"

OUTPUT_DIR = "models/dpo"

# =====================================================
# Tokenizer
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# Quantization
# =====================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# =====================================================
# Model
# =====================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False

# =====================================================
# Dataset
# =====================================================

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

# =====================================================
# LoRA
# =====================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
)

# =====================================================
# Training
# =====================================================

training_args = DPOConfig(

    output_dir=OUTPUT_DIR,

    num_train_epochs=2,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=5e-5,

    logging_steps=10,

    save_strategy="epoch",

    beta=0.1,

    fp16=True,

    report_to="none",
)

# =====================================================
# Trainer
# =====================================================

trainer = DPOTrainer(

    model=model,

    ref_model=None,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=peft_config,
)

# =====================================================

trainer.train()

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining Complete!")
print("Model saved to:", OUTPUT_DIR)