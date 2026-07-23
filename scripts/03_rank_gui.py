"""
03_rank_gui.py

Human ranking interface for DPO dataset creation.

Input:
data/generated/candidates.json

Output:
data/rankings/rankings.json

Type rankings like:

2 1 5 3 4

Meaning:
Best -> Worst
Candidate 2 > Candidate 1 > Candidate 5 > Candidate 3 > Candidate 4
"""

import json
import os

INPUT_FILE = "data/generated/candidates.json"

OUTPUT_DIR = "data/rankings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rankings.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Resume if ranking file already exists
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        rankings = json.load(f)
else:
    rankings = []

completed_ids = {item["id"] for item in rankings}

print("=" * 70)
print("Advertising Tagline Ranking Tool")
print("=" * 70)
print("Enter ranking as:")
print("2 1 5 3 4")
print("(Best ----------> Worst)")
print("Type 'q' anytime to quit.\n")

# -------------------------------------------------------------

for sample in dataset:

    if sample["id"] in completed_ids:
        continue

    print("\n" + "=" * 70)
    print(f"Product #{sample['id']}")
    print("=" * 70)

    print("\nDescription:\n")
    print(sample["description"])

    print("\nCandidates:\n")

    for i, slogan in enumerate(sample["candidates"], start=1):
        print(f"{i}. {slogan}")

    while True:

        ranking = input("\nRanking: ").strip()

        if ranking.lower() == "q":

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(rankings, f, indent=4, ensure_ascii=False)

            print("\nProgress saved.")
            print("Exiting...")
            exit()

        try:

            order = list(map(int, ranking.split()))

            if sorted(order) != [1, 2, 3, 4, 5]:
                raise ValueError

            rankings.append(
                {
                    "id": sample["id"],
                    "description": sample["description"],
                    "candidates": sample["candidates"],
                    "ranking": order
                }
            )

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(rankings, f, indent=4, ensure_ascii=False)

            print("✓ Saved")

            break

        except:
            print("\nInvalid ranking.")
            print("Example:")
            print("2 1 5 3 4")

print("\n" + "=" * 70)
print("All products ranked!")
print(f"Saved to {OUTPUT_FILE}")
print("=" * 70)