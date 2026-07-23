"""
01_create_dataset.py

Creates a synthetic dataset of product/service descriptions
for marketing punchline generation.

Output:
data/prompts/products.json
"""

import json
import os
import random

random.seed(42)

NUM_PRODUCTS = 100

OUTPUT_DIR = "data/prompts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "products.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

industries = [
    "Healthcare",
    "Finance",
    "Education",
    "Fitness",
    "Travel",
    "Food",
    "Fashion",
    "Automotive",
    "Real Estate",
    "Gaming",
    "E-commerce",
    "Cybersecurity",
    "Agriculture",
    "Pet Care",
    "Music",
    "Sports",
    "Productivity",
    "Smart Home",
    "Environment",
    "Retail"
]

products = [
    "mobile app",
    "web platform",
    "AI assistant",
    "smart device",
    "subscription service",
    "marketplace",
    "analytics dashboard",
    "chatbot",
    "wearable device",
    "cloud platform"
]

features = [
    "uses artificial intelligence",
    "provides real-time analytics",
    "automates repetitive tasks",
    "offers personalized recommendations",
    "works completely offline",
    "integrates with popular apps",
    "uses computer vision",
    "leverages blockchain technology",
    "supports voice commands",
    "provides predictive insights"
]

benefits = [
    "saving users time",
    "reducing operational costs",
    "improving productivity",
    "enhancing customer experience",
    "making better decisions",
    "increasing engagement",
    "simplifying daily tasks",
    "improving accessibility",
    "boosting business growth",
    "streamlining workflows"
]

audiences = [
    "students",
    "small businesses",
    "teachers",
    "developers",
    "parents",
    "healthcare professionals",
    "travelers",
    "retail stores",
    "content creators",
    "fitness enthusiasts"
]

dataset = []
seen = set()

while len(dataset) < NUM_PRODUCTS:

    industry = random.choice(industries)
    product = random.choice(products)
    feature = random.choice(features)
    benefit = random.choice(benefits)
    audience = random.choice(audiences)

    description = (
        f"A {industry.lower()} {product} that {feature}, "
        f"helping {audience} by {benefit}."
    )

    if description in seen:
        continue

    seen.add(description)

    dataset.append({
        "id": len(dataset) + 1,
        "industry": industry,
        "description": description
    })

    print(f"[{len(dataset):03d}] {description}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("\nDone!")
print(f"Saved {len(dataset)} product descriptions.")
print(f"Location: {OUTPUT_FILE}")