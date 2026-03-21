#!/usr/bin/env python3
"""
analyze_dataset.py — Assess dataset balance by framing category.
Reads project.conf for paths automatically.

Usage:
    python analyze_dataset.py
"""

import os
import sys
from collections import defaultdict
from project_config import conf

DATASET_PATH = conf["DATASET_PATH"]

CATEGORY_TAGS = {
    "face_closeup": [
        "close-up", "close_up", "closeup", "face_focus", "face focus",
        "extreme_close-up", "extreme_closeup", "head_focus",
    ],
    "head_shoulders": [
        "portrait", "head_and_shoulders", "head and shoulders",
        "bust", "headshot",
    ],
    "upper_body": [
        "upper_body", "upper body", "cowboy_shot", "cowboy shot",
        "waist_up", "waist up", "from_chest", "from chest",
    ],
    "full_body": [
        "full_body", "full body", "feet", "standing", "walking",
        "full_shot", "full shot", "wide_shot", "wide shot",
        "legs", "shoes", "sneakers", "boots", "sandals", "heels",
        "feet_visible",
    ],
}

IDEAL_RATIO = {
    "face_closeup": 0.10,
    "head_shoulders": 0.10,
    "upper_body": 0.15,
    "full_body": 0.65,
    "uncategorized": 0.00,
}


def categorize_caption(caption_text):
    text_lower = caption_text.lower()
    for tag in CATEGORY_TAGS["face_closeup"]:
        if tag in text_lower:
            return "face_closeup"
    for tag in CATEGORY_TAGS["upper_body"]:
        if tag in text_lower:
            return "upper_body"
    for tag in CATEGORY_TAGS["full_body"]:
        if tag in text_lower:
            return "full_body"
    for tag in CATEGORY_TAGS["head_shoulders"]:
        if tag in text_lower:
            return "head_shoulders"
    return "uncategorized"


def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH

    if not os.path.isdir(dataset_dir):
        print(f"Error: '{dataset_dir}' not found.")
        sys.exit(1)

    txt_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.txt')])
    if not txt_files:
        print(f"No caption files in '{dataset_dir}/'")
        sys.exit(1)

    categories = defaultdict(list)
    for txt_file in txt_files:
        with open(os.path.join(dataset_dir, txt_file), 'r') as f:
            caption = f.read().strip()
        category = categorize_caption(caption)
        categories[category].append((txt_file.replace('.txt', '.png'), caption))

    total = len(txt_files)

    print(f"Dataset Analysis: {total} images in {dataset_dir}/")
    print(f"Target: Full-body priority (65% full body)")
    print("=" * 70)
    print(f"\n{'Category':<20} {'Count':>6} {'Current %':>10} {'Ideal %':>10} {'Ideal Count':>12}")
    print("-" * 70)

    recommendations = {}
    for cat in ["face_closeup", "head_shoulders", "upper_body", "full_body", "uncategorized"]:
        count = len(categories[cat])
        current_pct = (count / total * 100) if total > 0 else 0
        ideal_pct = IDEAL_RATIO.get(cat, 0) * 100
        ideal_count = round(IDEAL_RATIO.get(cat, 0) * total)
        deficit = max(0, ideal_count - count)
        recommendations[cat] = deficit
        marker = " (heavy)" if current_pct > ideal_pct + 5 else " (light)" if current_pct < ideal_pct - 5 else ""
        print(f"{cat:<20} {count:>6} {current_pct:>9.1f}% {ideal_pct:>9.1f}% {ideal_count:>12}{marker}")

    facing_tags = ["looking_at_viewer", "looking at viewer", "facing viewer", "facing_viewer"]
    fb_facing = sum(1 for _, c in categories["full_body"] if any(t in c.lower() for t in facing_tags))
    fb_total = len(categories["full_body"])

    if fb_total > 0:
        print(f"\nFull body facing camera: {fb_facing}/{fb_total} ({fb_facing/fb_total*100:.0f}%)")
        target = round(fb_total * 0.7)
        if fb_facing < target:
            print(f"  Need ~{target - fb_facing} more front-facing full body")

    needs = {k: v for k, v in recommendations.items() if v > 0 and k != "uncategorized"}
    print(f"\n{'=' * 70}\nRECOMMENDATIONS\n{'=' * 70}\n")
    if needs:
        total_add = sum(needs.values())
        for cat, deficit in needs.items():
            print(f"  Add ~{deficit} more {cat.replace('_', ' ').title()} images")
        new_total = total + total_add
        reps = 7 if new_total > 120 else 8 if new_total > 100 else 10 if new_total > 60 else 15
        print(f"\n  Total to generate: ~{total_add}")
        print(f"  New dataset size: ~{new_total}")
        print(f"  Suggested NUM_REPEATS in project.conf: {reps}  (~{new_total * reps} steps/epoch)")
    else:
        print("  Dataset looks balanced.")

    if categories["uncategorized"]:
        print(f"\n{'=' * 70}")
        print(f"UNCATEGORIZED ({len(categories['uncategorized'])}) — review manually:")
        print("=" * 70)
        for img, caption in categories["uncategorized"]:
            tags = ", ".join(t.strip() for t in caption.split(",")[5:12])
            print(f"  {img}: {tags}")


if __name__ == "__main__":
    main()
