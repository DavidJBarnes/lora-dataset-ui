#!/usr/bin/env python3
"""
face_crop.py — Auto-detect faces and crop to 1024x1024 centered on the face.

Usage:
    python face_crop.py photo1.jpg photo2.png ...
    python face_crop.py /path/to/photos/*.jpg
    python face_crop.py --output-dir dataset/ photo1.jpg photo2.png

Options:
    --output-dir DIR    Output directory (default: current directory)
    --size SIZE         Output size in pixels (default: 1024)
    --padding FLOAT     Padding around face as fraction of face size (default: 1.5)
"""

import argparse
import os
import sys

import cv2
import numpy as np


def find_face(image):
    """Detect the largest face in an image. Returns (x, y, w, h) or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try DNN face detector first (more accurate)
    model_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(model_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        # Try with more lenient settings
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))

    if len(faces) == 0:
        return None

    # Return the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(faces[0])


def crop_face(image, face_rect, output_size=1024, padding=1.5):
    """Crop a square region centered on the face with padding."""
    h, w = image.shape[:2]
    fx, fy, fw, fh = face_rect

    # Center of face
    cx = fx + fw // 2
    cy = fy + fh // 2

    # Square crop size: face size * padding factor
    crop_size = int(max(fw, fh) * padding)
    # Ensure crop is at least output_size (before resize)
    crop_size = max(crop_size, min(w, h) // 2)

    # Crop bounds (keep within image)
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)

    # Adjust to keep square
    actual_w = x2 - x1
    actual_h = y2 - y1
    size = min(actual_w, actual_h)
    # Re-center
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    if x1 + size > w:
        x1 = w - size
    if y1 + size > h:
        y1 = h - size

    cropped = image[y1:y1+size, x1:x1+size]
    resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
    return resized


def process_image(input_path, output_dir, output_size=1024, padding=1.5):
    """Process a single image: detect face, crop, save."""
    image = cv2.imread(input_path)
    if image is None:
        print(f"  SKIP: Could not read {input_path}")
        return False

    face = find_face(image)
    if face is None:
        print(f"  SKIP: No face found in {os.path.basename(input_path)}")
        return False

    cropped = crop_face(image, face, output_size, padding)

    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_face.png")
    cv2.imwrite(output_path, cropped)
    print(f"  OK: {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({face[2]}x{face[3]} face detected)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Auto-crop faces from photos for LoRA training")
    parser.add_argument("images", nargs="+", help="Input image files")
    parser.add_argument("--output-dir", default=".", help="Output directory (default: current)")
    parser.add_argument("--size", type=int, default=1024, help="Output size in pixels (default: 1024)")
    parser.add_argument("--padding", type=float, default=1.5, help="Padding around face (default: 1.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    success = 0
    total = 0
    for path in args.images:
        total += 1
        if process_image(path, args.output_dir, args.size, args.padding):
            success += 1

    print(f"\nProcessed {success}/{total} images")


if __name__ == "__main__":
    main()
