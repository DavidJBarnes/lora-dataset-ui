#!/usr/bin/env python3
"""
rename_pairs.py — Rename image + caption pairs to sequential numbering.

Usage:
    python rename_pairs.py                  # Uses dataset path from project.conf
    python rename_pairs.py /custom/path     # Override path
    python rename_pairs.py --dry-run        # Preview without renaming
"""

import argparse
import os
import sys
from project_config import conf

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')


def main():
    parser = argparse.ArgumentParser(description="Rename image/caption pairs sequentially")
    parser.add_argument("directory", nargs="?", default=conf["DATASET_PATH"],
                        help=f"Directory (default: {conf['DATASET_PATH']})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' not found.")
        sys.exit(1)

    images = sorted([f for f in os.listdir(args.directory) if f.lower().endswith(EXTENSIONS)])
    if not images:
        print(f"No images in '{args.directory}/'")
        sys.exit(1)

    pairs = []
    for img in images:
        base = os.path.splitext(img)[0]
        txt = base + ".txt"
        if os.path.exists(os.path.join(args.directory, txt)):
            pairs.append((img, txt))
        else:
            print(f"Warning: {img} has no caption file")

    if not pairs:
        print("No image/caption pairs found.")
        sys.exit(1)

    prefix = "DRY RUN: " if args.dry_run else ""
    print(f"{prefix}Renaming {len(pairs)} pairs in {args.directory}/\n")

    # Temp rename to avoid collisions
    temp_names = []
    for i, (img, txt) in enumerate(pairs):
        ext = os.path.splitext(img)[1]
        temp_img = f"__temp_{i:04d}{ext}"
        temp_txt = f"__temp_{i:04d}.txt"
        temp_names.append((temp_img, temp_txt, ext))
        if not args.dry_run:
            os.rename(os.path.join(args.directory, img), os.path.join(args.directory, temp_img))
            os.rename(os.path.join(args.directory, txt), os.path.join(args.directory, temp_txt))

    for i, (temp_img, temp_txt, ext) in enumerate(temp_names):
        final_img = f"{i + 1:03d}{ext}"
        final_txt = f"{i + 1:03d}.txt"
        if not args.dry_run:
            os.rename(os.path.join(args.directory, temp_img), os.path.join(args.directory, final_img))
            os.rename(os.path.join(args.directory, temp_txt), os.path.join(args.directory, final_txt))
        print(f"  {pairs[i][0]} -> {final_img}")

    print(f"\n{prefix}Done. {len(pairs)} pairs renamed.")


if __name__ == "__main__":
    main()
