"""
04_build_dpo_dataset.py

Convert human rankings into a DPO dataset.

Input:
    data/rankings/rankings.json

Output:
    data/final/dpo_dataset.jsonl
"""

import json
import os

INPUT_FILE = "data/rankings/rankings.json"

OUTPUT_DIR = "data/final"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dpo_dataset.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    rankings = json.load(f)

num_pairs = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for sample in rankings:

        prompt = sample["description"]

        candidates = sample["candidates"]

        # ranking example:
        # [2,1,5,3,4]
        ranking = sample["ranking"]

        # Convert to zero-based indices
        ranking = [r - 1 for r in ranking]

        # Produce all pairwise preferences
        for i in range(len(ranking)):
            for j in range(i + 1, len(ranking)):

                chosen = candidates[ranking[i]]
                rejected = candidates[ranking[j]]

                example = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                }

                fout.write(json.dumps(example, ensure_ascii=False) + "\n")

                num_pairs += 1

print("=" * 60)
print("Finished building DPO dataset")
print(f"Rankings processed : {len(rankings)}")
print(f"Preference pairs   : {num_pairs}")
print(f"Saved to           : {OUTPUT_FILE}")
print("=" * 60)