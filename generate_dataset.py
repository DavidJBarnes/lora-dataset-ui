#!/usr/bin/env python3
"""
generate_dataset.py — Automate dataset generation via A1111 API + FaceSwapLab.
Reads character definition, trigger, and paths from project.conf.

Usage:
    python generate_dataset.py --model pony
    python generate_dataset.py --model lustify --category full_body
    python generate_dataset.py --model pony --no-faceswap
    python generate_dataset.py --model lustify --dry-run
"""

import argparse
import base64
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("Error: pip install requests")
    sys.exit(1)

from project_config import conf

CHARACTER_DESC = conf["CHARACTER_DESC"]
TRIGGER = conf["TRIGGER"]
CLASS = conf["CLASS"]
REFERENCE_FACE = conf["REFERENCE_FACE"]
A1111_URL = conf.get("A1111_URL", "http://localhost:7860")

MODEL_PROFILES = {
    "pony": {
        "name": "CyberRealistic Pony",
        "sampler_name": "DPM++ SDE Karras",
        "steps": 30,
        "cfg_scale": 5,
        "clip_skip": 2,
        "prefix": "score_9, score_8_up, score_7_up, source_realistic",
        "negative": (
            "score_6, score_5, score_4, (worst quality:1.2), (low quality:1.2), "
            "(normal quality:1.2), lowres, bad anatomy, bad hands, signature, "
            "watermarks, ugly, imperfect eyes, skewed eyes, unnatural face, "
            "unnatural body, error, extra limb, missing limbs, multiple people"
        ),
        "negative_front": (
            "score_6, score_5, score_4, (worst quality:1.2), (low quality:1.2), "
            "(normal quality:1.2), lowres, bad anatomy, bad hands, signature, "
            "watermarks, ugly, imperfect eyes, skewed eyes, unnatural face, "
            "unnatural body, error, extra limb, missing limbs, multiple people, "
            "from side, profile, three-quarter view, turned head, asymmetrical framing"
        ),
    },
    "lustify": {
        "name": "Lustify-SDXL",
        "sampler_name": "DPM++ 2M SDE Karras",
        "steps": 30,
        "cfg_scale": 5,
        "clip_skip": 1,
        "prefix": "",
        "negative": (
            "worst quality, low quality, blurry, bad anatomy, bad hands, "
            "deformed, ugly, watermark, signature, extra limb, missing limbs, "
            "multiple people, deformed face, asymmetric eyes"
        ),
        "negative_front": (
            "worst quality, low quality, blurry, bad anatomy, bad hands, "
            "deformed, ugly, watermark, signature, extra limb, missing limbs, "
            "multiple people, deformed face, asymmetric eyes, "
            "from side, profile, three-quarter view, turned head"
        ),
    },
}


def build_positive(profile, scene):
    parts = []
    if profile["prefix"]:
        parts.append(profile["prefix"])
    parts.append(f"1girl, ({CHARACTER_DESC})")
    parts.append(scene)
    return ", ".join(parts)


def build_prompts(profile):
    neg = profile["negative"]
    neg_front = profile["negative_front"]

    return {
        "face_closeup": {
            "count": 15, "width": 1024, "height": 1024,
            "prompts": [
                {"name": "1a_neutral_front", "negative": neg_front,
                 "positive": build_positive(profile,
                    "extreme close-up, face focus, looking at viewer, "
                    "front view, symmetrical face, both ears visible, facing viewer, head centered, "
                    "neutral expression, sharp focus on eyes, studio lighting, "
                    "detailed skin texture, pores, plain background")},
                {"name": "1b_soft_natural", "negative": neg_front,
                 "positive": build_positive(profile,
                    "close-up, face focus, looking at viewer, "
                    "front view, symmetrical face, both ears visible, facing viewer, head centered, "
                    "soft smile, natural window lighting, shallow depth of field, "
                    "detailed skin texture, indoor")},
                {"name": "1c_dramatic", "negative": neg,
                 "positive": build_positive(profile,
                    "close-up, face focus, looking at viewer, "
                    "serious expression, rembrandt lighting, dramatic shadows, "
                    "detailed skin texture, dark background")},
            ],
        },
        "head_shoulders": {
            "count": 15, "width": 1024, "height": 1024,
            "prompts": [
                {"name": "2a_classic", "negative": neg,
                 "positive": build_positive(profile,
                    "portrait, head and shoulders, looking at viewer, "
                    "__cyberRealisticPony/clothing__, gentle smile, __cyberRealisticPony/lighting__, "
                    "__cyberRealisticPony/locations__, detailed skin, hair detail")},
                {"name": "2b_outdoor", "negative": neg,
                 "positive": build_positive(profile,
                    "portrait, head and shoulders, looking at viewer, "
                    "__cyberRealisticPony/clothing__, __cyberRealisticPony/locations__, bokeh, "
                    "__cyberRealisticPony/lighting__, natural skin")},
                {"name": "2c_three_quarter", "negative": neg,
                 "positive": build_positive(profile,
                    "portrait, head and shoulders, three-quarter view, "
                    "__cyberRealisticPony/clothing__, slight smile, __cyberRealisticPony/lighting__, "
                    "__cyberRealisticPony/locations__, detailed hair strands")},
            ],
        },
        "upper_body": {
            "count": 20, "width": 896, "height": 1152,
            "prompts": [
                {"name": "3a_viewer", "negative": neg,
                 "positive": build_positive(profile,
                    "upper body, __cyberRealisticPony/clothing__, __cyberRealisticPony/posing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, natural skin")},
                {"name": "3b_candid", "negative": neg,
                 "positive": build_positive(profile,
                    "upper body, __cyberRealisticPony/clothing__, __cyberRealisticPony/posing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking away from viewer, candid feel")},
                {"name": "3c_hands", "negative": neg,
                 "positive": build_positive(profile,
                    "upper body, __cyberRealisticPony/clothing__, __cyberRealisticPony/posing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "hands visible, detailed skin texture")},
            ],
        },
        "full_body": {
            "count": 90, "width": 1024, "height": 1536,
            "prompts": [
                {"name": "4a_standing", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, feet visible, __cyberRealisticPony/clothing__, __cyberRealisticPony/posing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, facing viewer")},
                {"name": "4b_walking", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, walking, mid-stride, __cyberRealisticPony/clothing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, hair in motion")},
                {"name": "4c_seated", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, sitting, legs crossed, __cyberRealisticPony/clothing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, relaxed pose")},
                {"name": "4d_leaning", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, leaning against wall, __cyberRealisticPony/clothing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, arms crossed, casual")},
                {"name": "4e_dynamic", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, dynamic pose, __cyberRealisticPony/clothing__, __cyberRealisticPony/posing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, action shot, wind")},
                {"name": "4f_tight", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, standing, feet visible, __cyberRealisticPony/clothing__, "
                    "__cyberRealisticPony/lighting__, simple background, "
                    "looking at viewer, facing viewer, centered")},
                {"name": "4g_environmental", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, wide shot, __cyberRealisticPony/clothing__, __cyberRealisticPony/posing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, environment, scenic")},
                {"name": "4h_low_angle", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, from below, __cyberRealisticPony/clothing__, standing, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking down at viewer, confident")},
                {"name": "4i_ground", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, sitting on ground, knees up, __cyberRealisticPony/clothing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, relaxed")},
                {"name": "4j_expressive", "negative": neg,
                 "positive": build_positive(profile,
                    "full body, standing, __cyberRealisticPony/clothing__, __cyberRealisticPony/posing__, "
                    "__cyberRealisticPony/locations__, __cyberRealisticPony/lighting__, "
                    "looking at viewer, laughing, joyful, candid")},
            ],
        },
    }


def build_faceswaplab_args(face_b64=None, face_checkpoint=None):
    """
    Build FaceSwapLab args matching the actual API schema (v1.2.7).
    37 args per face unit × 3 units + 14 global post-processing = 125 args total.

    Per-unit layout (37 args):
      [0]  Reference (base64 or null)
      [1]  Face Checkpoint (string or null — takes precedence)
      [2]  Batch Sources Images
      [3]  Blend Faces
      [4]  Enable
      [5]  Same Gender
      [6]  Sort by size
      [7]  Check similarity
      [8]  Compute similarity — MUST be False or draws bounding boxes
      [9]  Min similarity
      [10] Min reference similarity
      [11] Target face
      [12] Reference source face
      [13] Swap in source image
      [14] Swap in generated image
      --- Inpainting (face) ---
      [15] Denoising strength
      [16] Inpainting prompt
      [17] Inpainting negative prompt
      [18] Inpainting steps
      [19] Inpainting Sampler
      [20] sd model (experimental)
      [21] Inpainting seed
      --- Per-unit post-processing ---
      [22] Restore Face (None/CodeFormer/GFPGAN)
      [23] Restore visibility
      [24] codeformer weight
      [25] Upscaler
      [26] Use improved segmented mask (pastenet)
      [27] Use color corrections
      [28] sharpen face
      [29] Upscaled swapper mask erosion factor
      --- Inpainting (upscaled) ---
      [30] Denoising strength
      [31] Inpainting prompt
      [32] Inpainting negative prompt
      [33] Inpainting steps
      [34] Inpainting Sampler
      [35] sd model
      [36] Inpainting seed

    Global post-processing (14 args, after all 3 units):
      [111] Restore Face
      [112] Restore visibility
      [113] codeformer weight
      [114] Upscaler
      [115] Upscaler scale
      [116] Upscaler visibility (if scale = 1)
      [117] Enable/When
      [118-124] Inpainting settings
    """

    # Unit 1 — active
    unit1 = [
        face_b64,               # [0]  Reference image
        face_checkpoint,        # [1]  Face Checkpoint
        None,                   # [2]  Batch Sources
        True,                   # [3]  Blend Faces
        True,                   # [4]  Enable
        False,                  # [5]  Same Gender
        True,                   # [6]  Sort by size
        False,                  # [7]  Check similarity
        False,                  # [8]  Compute similarity — False to prevent bounding boxes
        0.5,                    # [9]  Min similarity
        0.2,                    # [10] Min reference similarity
        "0",                    # [11] Target face
        0,                      # [12] Reference source face
        False,                  # [13] Swap in source image
        True,                   # [14] Swap in generated image
        # --- Inpainting (face) ---
        0.35,                   # [15] Denoising strength
        "Portrait of a [gender], detailed face, natural skin",
        "blurry, deformed, ugly",
        20,                     # [18] Inpainting steps
        "DPM++ 2M",             # [19] Inpainting Sampler
        None,                   # [20] sd model
        0,                      # [21] Inpainting seed
        # --- Per-unit post-processing ---
        "CodeFormer",           # [22] Restore Face
        1,                      # [23] Restore visibility
        1,                      # [24] codeformer weight
        None,                   # [25] Upscaler
        False,                  # [26] Use segmented mask
        False,                  # [27] Use color corrections
        False,                  # [28] sharpen face
        1,                      # [29] Erosion factor
        # --- Inpainting (upscaled) ---
        0,                      # [30] Denoising strength
        "Portrait of a [gender]",
        "blurry",
        20,                     # [33] Inpainting steps
        "DPM++ 2M",             # [34] Inpainting Sampler
        None,                   # [35] sd model
        0,                      # [36] Inpainting seed
    ]

    # Units 2 & 3 — disabled (37 args each)
    unit_disabled = [
        None, None, None,       # [0-2]  Reference, Checkpoint, Batch
        True, False, False,     # [3-5]  Blend, Enable=OFF, Same Gender
        False, False, False,    # [6-8]  Sort, Check sim, Compute sim
        0, 0,                   # [9-10] Min similarity
        "0", 0,                 # [11-12] Target, Reference source
        False, True,            # [13-14] Swap source, Swap generated
        0, "Portrait of a [gender]", "blurry", 20, "DPM++ 2M", None, 0,  # [15-21] Inpainting
        None, 1, 1, None,       # [22-25] Restore, visibility, codeformer, upscaler
        False, False, False, 1, # [26-29] mask, color, sharpen, erosion
        0, "Portrait of a [gender]", "blurry", 20, "DPM++ 2M", None, 0,  # [30-36] Inpainting 2
    ]

    # Global post-processing (14 args)
    global_pp = [
        "CodeFormer",           # [111] Restore Face
        1,                      # [112] Restore visibility
        1,                      # [113] codeformer weight
        None,                   # [114] Upscaler (None = no upscaling)
        2,                      # [115] Upscaler scale
        1,                      # [116] Upscaler visibility
        "After Upscaling/Before Restore Face",  # [117] Enable/When
        0,                      # [118] Denoising strength
        "Portrait of a [gender]",
        "blurry",
        20,                     # [121] Inpainting steps
        "DPM++ 2M",             # [122] Inpainting Sampler
        None,                   # [123] sd model
        0,                      # [124] Inpainting seed
    ]

    return {"faceswaplab": {"args": unit1 + unit_disabled + unit_disabled + global_pp}}


def load_face(path):
    """Load reference face image as base64. Returns None if path is None."""
    if path is None or path.lower() == "none":
        return None
    if not os.path.isfile(path):
        print(f"Error: Reference face not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate(url, positive, negative, w, h, profile, faceswap=None):
    payload = {
        "prompt": positive, "negative_prompt": negative,
        "width": w, "height": h,
        "sampler_name": profile["sampler_name"],
        "steps": profile["steps"], "cfg_scale": profile["cfg_scale"],
        "seed": -1,
        "override_settings": {"CLIP_stop_at_last_layers": profile["clip_skip"]},
        "override_settings_restore_afterwards": True,
    }
    if faceswap:
        payload["alwayson_scripts"] = faceswap
    r = requests.post(f"{url}/sdapi/v1/txt2img", json=payload, timeout=600)
    r.raise_for_status()
    imgs = r.json().get("images")
    return base64.b64decode(imgs[0]) if imgs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_PROFILES.keys()))
    parser.add_argument("--face", default=REFERENCE_FACE,
                        help="Reference face image for FaceSwapLab (base64 encoded)")
    parser.add_argument("--checkpoint", default=conf.get("FACE_CHECKPOINT"),
                        help="FaceSwapLab face checkpoint name (e.g. 'Kelly_20251124.safetensors'). Takes precedence over --face.")
    parser.add_argument("--lora", action="append", default=[],
                        help="LoRA to apply. Format: name:weight (e.g. 'klb_y1_pony_v1:0.7'). Can specify multiple times. Stacks with LORAS in project.conf.")
    parser.add_argument("--url", default=A1111_URL)
    parser.add_argument("--category", choices=["face_closeup", "head_shoulders", "upper_body", "full_body"])
    parser.add_argument("--output", default=conf.get("DATASET_PATH", "raw_images"))
    parser.add_argument("--add", type=int, default=0,
                        help="Generate N additional images per category (appends, doesn't overwrite)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-faceswap", action="store_true")
    args = parser.parse_args()

    # Build LoRA prompt tags (conf defaults + CLI additions stack)
    lora_tags = ""
    lora_list = []

    # Start with project.conf defaults
    conf_loras = conf.get("LORAS", "None")
    if conf_loras and conf_loras.lower() != "none":
        lora_list = [l.strip() for l in conf_loras.split(",") if l.strip()]

    # CLI --lora adds on top
    if args.lora:
        lora_list.extend(args.lora)

    if lora_list:
        lora_parts = []
        for lora in lora_list:
            if ":" in lora:
                name, weight = lora.rsplit(":", 1)
                lora_parts.append(f"<lora:{name}:{weight}>")
            else:
                lora_parts.append(f"<lora:{lora}:0.7>")
        lora_tags = " ".join(lora_parts)

    profile = MODEL_PROFILES[args.model]

    # Append extra negatives from project.conf
    neg_extra = conf.get("NEGATIVE_EXTRA", "None")
    if neg_extra and neg_extra.lower() != "none":
        profile = dict(profile)  # Don't mutate the original
        profile["negative"] = f"{profile['negative']}, {neg_extra}"
        profile["negative_front"] = f"{profile['negative_front']}, {neg_extra}"

    print(f"Model: {profile['name']}  |  Trigger: {TRIGGER} {CLASS}")
    if lora_tags:
        print(f"LoRAs: {lora_tags}")

    faceswap = None
    if not args.no_faceswap:
        if args.checkpoint:
            # Use a pre-built FaceSwapLab checkpoint (best quality, no base64 needed)
            faceswap = build_faceswaplab_args(face_b64=None, face_checkpoint=args.checkpoint)
            print(f"FaceSwap: ENABLED (checkpoint: {args.checkpoint})")
        elif args.face:
            # Fall back to base64 reference image
            face_b64 = load_face(args.face)
            faceswap = build_faceswaplab_args(face_b64=face_b64, face_checkpoint=None)
            print(f"FaceSwap: ENABLED (reference image: {args.face})")
        else:
            print("Warning: No --checkpoint or --face provided. FaceSwap disabled.")
    else:
        print(f"FaceSwap: DISABLED")

    all_prompts = build_prompts(profile)
    cats = {args.category: all_prompts[args.category]} if args.category else all_prompts

    # In --add mode, override counts and report existing
    if args.add > 0:
        for cat in cats.values():
            cat["count"] = args.add

    total = sum(c["count"] for c in cats.values())

    mode = f"ADD {args.add} per category" if args.add > 0 else "FULL"
    print(f"\n{'=' * 60}")
    print(f"  Mode: {mode}")
    for name, cat in cats.items():
        cat_dir = os.path.join(args.output, name)
        existing = len([f for f in os.listdir(cat_dir) if f.endswith(".png")]) if os.path.isdir(cat_dir) else 0
        label = f"(+{cat['count']} new, {existing} existing)" if args.add > 0 else f"({len(cat['prompts'])} prompts)"
        print(f"  {name:<20} {cat['count']:>3} images at {cat['width']}×{cat['height']}  {label}")
    print(f"  Total to generate: {total} images")

    if args.dry_run:
        print("\n--dry-run: No images generated.")
        return

    try:
        requests.get(f"{args.url}/sdapi/v1/sd-models", timeout=5).raise_for_status()
    except Exception as e:
        print(f"\nError: Cannot reach A1111 at {args.url}\n  {e}")
        sys.exit(1)

    generated = failed = 0
    start = time.time()

    for cat_name, cat in cats.items():
        cat_dir = os.path.join(args.output, cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        # Find highest existing file number to continue from
        existing_nums = []
        for f in os.listdir(cat_dir):
            if f.endswith(".png"):
                # Extract number from end of filename (before .png)
                parts = f.replace(".png", "").rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        existing_nums.append(int(parts[1]))
                    except ValueError:
                        pass
        start_idx = max(existing_nums) if existing_nums else 0

        n_prompts = len(cat["prompts"])
        per = cat["count"] // n_prompts
        rem = cat["count"] % n_prompts
        count = 0

        print(f"\n--- {cat_name} ({cat['count']} images, starting at idx {start_idx + 1}) ---")

        for pi, p in enumerate(cat["prompts"]):
            n = per + (1 if pi < rem else 0)
            for ii in range(n):
                count += 1
                file_idx = start_idx + count
                fn = f"{cat_name}_{p['name']}_{file_idx:03d}.png"
                fp = os.path.join(cat_dir, fn)
                print(f"  [{count}/{cat['count']}] {fn} ... ", end="", flush=True)
                try:
                    positive = p["positive"]
                    if lora_tags:
                        positive = f"{positive} {lora_tags}"
                    img = generate(args.url, positive, p["negative"],
                                   cat["width"], cat["height"], profile, faceswap)
                    if img:
                        with open(fp, "wb") as f:
                            f.write(img)
                        generated += 1
                        print("OK")
                    else:
                        failed += 1
                        print("FAILED")
                except Exception as e:
                    failed += 1
                    print(f"FAILED ({e})")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Done: {generated} generated, {failed} failed, {elapsed/60:.1f}min")
    print(f"\nNext: review {args.output}/, move keepers, then:")
    print(f"  bash tagger.sh")
    print(f"  bash tagger_cleanup.sh {args.model}")
    print(f"  python analyze_dataset.py")
    print(f"  bash train_character.sh {args.model}")


if __name__ == "__main__":
    main()
