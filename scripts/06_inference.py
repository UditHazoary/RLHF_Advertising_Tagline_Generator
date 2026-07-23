import csv
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================================
# Configuration
# ==========================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DPO_ADAPTER = "models/dpo"

NUM_GENERATIONS = 5

MAX_NEW_TOKENS = 20
TEMPERATURE = 0.9
TOP_P = 0.9
REPETITION_PENALTY = 1.1

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Tokenizer
# ==========================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================================
# Original Base Model
# ==========================================================

print("Loading ORIGINAL model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)

base_model.eval()

# ==========================================================
# DPO Model (Separate Instance)
# ==========================================================

print("Loading DPO model...")

dpo_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)

dpo_model = PeftModel.from_pretrained(
    dpo_base,
    DPO_ADAPTER,
)

dpo_model.eval()

print("✓ Models Loaded Successfully!")

# ==========================================================
# Generation Function
# ==========================================================

def generate(model, product):

    prompt = f"""
Create ONE short and catchy advertising tagline.

Rules:
- Maximum 10 words
- Memorable
- Creative
- Do NOT explain
- Return ONLY the tagline

Product:
{product}
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()

# ==========================================================
# CSV Initialization
# ==========================================================

csv_file = "comparison_results.csv"

try:
    open(csv_file, "r").close()

except FileNotFoundError:

    with open(csv_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Timestamp",
            "Product",
            "Base_1",
            "Base_2",
            "Base_3",
            "Base_4",
            "Base_5",
            "DPO_1",
            "DPO_2",
            "DPO_3",
            "DPO_4",
            "DPO_5"
        ])

# ==========================================================
# Interactive Loop
# ==========================================================

while True:

    print("\n" + "=" * 80)

    product = input("Enter Product Description (q to quit): ")

    if product.lower() == "q":
        break

    print("\nGenerating using ORIGINAL model...")

    base_outputs = [
        generate(base_model, product)
        for _ in range(NUM_GENERATIONS)
    ]

    print("Generating using DPO model...")

    dpo_outputs = [
        generate(dpo_model, product)
        for _ in range(NUM_GENERATIONS)
    ]

    print("\n" + "=" * 80)
    print("PRODUCT")
    print("=" * 80)
    print(product)

    print("\n" + "=" * 80)
    print("ORIGINAL MODEL")
    print("=" * 80)

    for i, tagline in enumerate(base_outputs, 1):
        print(f"{i}. {tagline}")

    print("\n" + "=" * 80)
    print("DPO MODEL")
    print("=" * 80)

    for i, tagline in enumerate(dpo_outputs, 1):
        print(f"{i}. {tagline}")

    with open(csv_file, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            product,
            *base_outputs,
            *dpo_outputs,
        ])

    print("\n✓ Results saved to comparison_results.csv")