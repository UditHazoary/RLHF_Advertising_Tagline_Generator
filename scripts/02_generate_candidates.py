"""
02_generate_candidates.py

Generate 5 unique advertising slogans for every product.

Input:
data/prompts/products.json

Output:
data/generated/candidates.json
"""

import json
import os
import random
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_FILE = "data/prompts/products.json"

OUTPUT_DIR = "data/generated"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "candidates.json"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

# =====================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

generic_phrases = {
    "Experience the Future",
    "Innovation Starts Here",
    "The Future is Here",
    "Think Different",
    "Empowering Tomorrow",
    "Next Generation",
    "Unlock the Future",
}

# =====================================================

for idx, sample in enumerate(dataset):

    description = sample["description"]

    slogans = set()

    attempts = 0

    while len(slogans) < 5 and attempts < 25:

        prompt = f"""
You are a senior advertising copywriter.

Write ONE memorable advertising slogan for the product below.

Requirements:
- Maximum 8 words.
- Catchy and memorable.
- Professional and brandable.
- Emotionally engaging.
- Highlight the product's main benefit.
- Avoid clichés.
- No quotation marks.
- No hashtags.
- No emojis.
- Return ONLY the slogan.

Product:
{description}
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

        outputs = model.generate(
            **inputs,
            max_new_tokens=18,
            do_sample=True,
            temperature=random.uniform(0.75, 1.0),
            top_p=0.9,
            repetition_penalty=1.2
        )

        slogan = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        slogan = slogan.strip()
        slogan = slogan.replace('"', "")
        slogan = slogan.replace("'", "")
        slogan = slogan.replace("\n", " ")
        slogan = slogan.strip(" .")

        if (
            2 <= len(slogan.split()) <= 8
            and slogan not in generic_phrases
            and len(slogan) > 5
        ):
            slogans.add(slogan)

        attempts += 1

    slogans = list(slogans)

    while len(slogans) < 5:
        slogans.append("")

    results.append(
        {
            "id": sample["id"],
            "description": description,
            "candidates": slogans
        }
    )

    print(f"[{idx+1:03d}/{len(dataset)}] {len(slogans)} slogans generated")

# =====================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\nFinished!")
print("Saved to:", OUTPUT_FILE)