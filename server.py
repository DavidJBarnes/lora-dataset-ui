#!/usr/bin/env python3
"""
server.py — Dataset preparation SPA for LoRA training.

Category-based image browser with integrated WD14 tagging, caption editing,
and dataset balance analysis. Single-page app served from one file.

Discovers sibling project directories (those with project.conf) and lets
you switch between them from a dropdown in the header.

Usage:
    python server.py                        # Auto-discover projects
    python server.py --port 9000            # Custom port
    python server.py --loras-dir /path/to   # Explicit parent directory

Controls:
    Click          = select/deselect for deletion
    Shift+Click    = full-size preview + caption editor
    Double-click   = full-size preview + caption editor
    Arrow keys     = navigate in preview
    D key          = mark for delete + next (in preview)
    Ctrl+S         = save caption (in preview)
    Escape         = close preview
"""

import argparse
import datetime
import json
import mimetypes
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from io import BytesIO
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote, urlparse, parse_qs

from project_config import load_conf

# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------

def detect_model(project_dir, conf):
    """Determine model type from conf or directory name."""
    model = conf.get("MODEL", "").lower()
    if model in ("pony", "lustify"):
        return model
    dirname = os.path.basename(project_dir).lower()
    if "pony" in dirname:
        return "pony"
    if "lustify" in dirname:
        return "lustify"
    return "lustify"


SERVER_CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.conf")


def load_server_conf():
    """Load server.conf — returns list of project directory paths."""
    dirs = []
    if not os.path.isfile(SERVER_CONF):
        return dirs
    home = os.path.expanduser("~")
    with open(SERVER_CONF, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = line.replace("$HOME", home)
            if os.path.isabs(line):
                dirs.append(line)
            else:
                # Relative to the server.conf directory
                dirs.append(os.path.join(os.path.dirname(SERVER_CONF), line))
    return dirs


def _load_project(project_dir):
    """Load a single project from a directory. Returns dict or None."""
    project_dir = os.path.abspath(project_dir)
    if not os.path.isdir(project_dir):
        return None
    conf_path = os.path.join(project_dir, "project.conf")
    if not os.path.isfile(conf_path):
        return None
    try:
        pconf = load_conf(conf_path)
    except (SystemExit, Exception):
        return None
    model = detect_model(project_dir, pconf)
    dataset_dir = os.path.join(project_dir, "dataset")
    return {
        "name": os.path.basename(project_dir),
        "dir": project_dir,
        "model": model,
        "trigger": pconf.get("TRIGGER", ""),
        "conf_path": conf_path,
        "dataset_dir": dataset_dir,
    }


def discover_projects(loras_dir):
    """Load projects from server.conf, or auto-discover from loras_dir."""
    # Explicit list from server.conf takes priority, with auto-discovery fallback
    explicit_dirs = load_server_conf()
    if explicit_dirs:
        projects = []
        for d in explicit_dirs:
            proj = _load_project(d)
            if proj:
                projects.append(proj)
            else:
                print(f"  Warning: skipping {d} (no project.conf or parse error)")
        if projects:
            return projects
        print("  server.conf yielded no valid projects, falling back to auto-discovery")

    # Auto-discover: sibling dirs with project.conf
    projects = []
    server_dir = os.path.abspath(os.path.dirname(__file__))
    for entry in sorted(os.listdir(loras_dir)):
        project_dir = os.path.join(loras_dir, entry)
        if not os.path.isdir(project_dir):
            continue
        if os.path.abspath(project_dir) == server_dir:
            continue
        proj = _load_project(project_dir)
        if proj:
            projects.append(proj)
    return projects


class ServerState:
    """Mutable server state that allows switching between projects."""

    def __init__(self, projects):
        self.projects = {p["name"]: p for p in projects}
        self.project_list = projects
        self.current = None
        self.conf = {}
        self.conf_path = ""
        self.base_dir = ""
        self.model = ""
        if projects:
            self.switch_to(projects[0]["name"])

    def switch_to(self, name):
        if name not in self.projects:
            return False
        proj = self.projects[name]
        self.current = name
        self.model = proj["model"]
        self.conf_path = proj["conf_path"]
        self.conf = load_conf(self.conf_path)
        self.base_dir = proj["dataset_dir"]
        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
        return True


# ---------------------------------------------------------------------------
# Category detection (from analyze_dataset.py)
# ---------------------------------------------------------------------------

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

# The representative tag to INSERT when setting a category via the UI
CATEGORY_PRIMARY_TAG = {
    "face_closeup": "close-up",
    "head_shoulders": "portrait",
    "upper_body": "upper_body",
    "full_body": "full_body",
}

IDEAL_RATIO = {
    "face_closeup": 0.10,
    "head_shoulders": 0.10,
    "upper_body": 0.15,
    "full_body": 0.65,
    "uncategorized": 0.00,
}

CATEGORY_ORDER = ["face_closeup", "head_shoulders", "upper_body", "full_body", "uncategorized"]


def categorize_caption(caption_text):
    """Return category string based on tags found in caption."""
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


# ---------------------------------------------------------------------------
# Caption cleanup (ported from tagger_cleanup.sh)
# ---------------------------------------------------------------------------

REMOVE_TAGS = {
    "blonde_hair", "blonde hair",
    "dirty_blonde", "dirty blonde",
    "light_brown_hair", "light brown hair",
    "brown_hair",
    "blue_eyes", "blue eyes",
    "grey_eyes", "grey eyes",
    "green_eyes", "green eyes",
    "freckles",
    "oval_face", "oval face",
    "slim", "thin",
    "pale_skin", "pale skin",
    "light_skin", "light skin",
    "1girl", "solo",
    "breasts", "small_breasts", "medium_breasts", "large_breasts",
}

REMOVE_TAGS_LOWER = {t.lower() for t in REMOVE_TAGS}


def make_prefix(model, conf):
    """Build the caption prefix for a given model."""
    trigger = conf.get("TRIGGER", "trigger")
    cls = conf.get("CLASS", "woman")
    if model == "pony":
        return f"score_9, score_8_up, score_7_up, source_realistic, {trigger} {cls}"
    return f"{trigger} {cls}"


def strip_prefix(caption, model, conf):
    """Remove any existing model prefix from the start of a caption."""
    trigger = conf.get("TRIGGER", "trigger")
    cls = conf.get("CLASS", "woman")
    pony_prefix = f"score_9, score_8_up, score_7_up, source_realistic, {trigger} {cls}"
    lustify_prefix = f"{trigger} {cls}"
    for pfx in [pony_prefix, lustify_prefix]:
        if caption.startswith(pfx):
            caption = caption[len(pfx):].lstrip(", ")
            break
    return caption


def dedupe_tags(tags):
    """Remove duplicate tags (case-insensitive), preserving first occurrence and order."""
    seen = set()
    result = []
    for tag in tags:
        key = tag.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(tag)
    return result


def cleanup_caption(raw_caption, model, conf):
    """Clean a raw WD14 caption: remove character tags, dedupe, add model prefix."""
    text = strip_prefix(raw_caption.strip(), model, conf)
    tags = [t.strip() for t in text.split(",")]
    tags = [t for t in tags if t and t.lower() not in REMOVE_TAGS_LOWER]
    tags = dedupe_tags(tags)
    prefix = make_prefix(model, conf)
    cleaned = ", ".join(tags)
    return f"{prefix}, {cleaned}" if cleaned else prefix


# ---------------------------------------------------------------------------
# Background task system
# ---------------------------------------------------------------------------

_tasks = {}
_tasks_lock = threading.Lock()


def create_task(name):
    """Create a background task entry, return task_id."""
    task_id = str(uuid.uuid4())[:8]
    with _tasks_lock:
        _tasks[task_id] = {
            "name": name, "status": "running",
            "progress": "", "message": "Starting...", "error": None,
        }
    return task_id


def update_task(task_id, **kwargs):
    with _tasks_lock:
        if task_id in _tasks:
            _tasks[task_id].update(kwargs)


def get_task(task_id):
    with _tasks_lock:
        return dict(_tasks.get(task_id, {}))


# ---------------------------------------------------------------------------
# WD14 Tagger integration
# ---------------------------------------------------------------------------

ALL_CATEGORY_TAGS_LOWER = set()
for _tag_list in CATEGORY_TAGS.values():
    ALL_CATEGORY_TAGS_LOWER.update(t.lower() for t in _tag_list)


def merge_captions(existing_caption, wd14_raw, model, conf):
    """Merge WD14 tags into an existing caption, preserving user tags.

    Rules:
    - Never overwrite existing category — skip WD14 category tags if one exists
    - Never add duplicates (case-insensitive)
    - Preserve existing tag order, append new tags at end
    """
    existing_stripped = strip_prefix(existing_caption.strip(), model, conf)
    existing_tags = [t.strip() for t in existing_stripped.split(",") if t.strip()]
    existing_lower = {t.lower() for t in existing_tags}

    # Check if existing caption already has a category
    has_category = any(t in ALL_CATEGORY_TAGS_LOWER for t in existing_lower)

    wd14_tags = [t.strip() for t in wd14_raw.split(",") if t.strip()]
    wd14_tags = [t for t in wd14_tags if t and t.lower() not in REMOVE_TAGS_LOWER]

    for tag in wd14_tags:
        tag_lower = tag.lower()
        # Skip duplicates
        if tag_lower in existing_lower:
            continue
        # Skip category tags if existing caption already has a category
        if has_category and tag_lower in ALL_CATEGORY_TAGS_LOWER:
            continue
        existing_tags.append(tag)
        existing_lower.add(tag_lower)

    tags = dedupe_tags(existing_tags)
    prefix = make_prefix(model, conf)
    cleaned = ", ".join(t for t in tags if t)
    return f"{prefix}, {cleaned}" if cleaned else prefix


def run_tagger_task(task_id, dataset_dir, model, conf):
    """Background task: run WD14 tagger on all images, merge with existing captions."""
    existing_captions = {}
    try:
        sd_scripts = conf.get("SD_SCRIPTS", "")
        if not sd_scripts or not os.path.isdir(sd_scripts):
            update_task(task_id, status="failed", error="SD_SCRIPTS path not found")
            return

        venv_python = os.path.join(sd_scripts, "venv", "bin", "python")
        tagger_script = os.path.join(sd_scripts, "finetune", "tag_images_by_wd14_tagger.py")

        if not os.path.isfile(venv_python):
            update_task(task_id, status="failed", error=f"venv python not found: {venv_python}")
            return
        if not os.path.isfile(tagger_script):
            update_task(task_id, status="failed", error=f"Tagger script not found: {tagger_script}")
            return

        image_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        image_files = [f for f in os.listdir(dataset_dir)
                       if os.path.splitext(f)[1].lower() in image_exts]

        for img_file in image_files:
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            txt_path = os.path.join(dataset_dir, txt_file)
            if os.path.isfile(txt_path):
                with open(txt_path, 'r') as f:
                    existing_captions[txt_file] = f.read().strip()
                os.remove(txt_path)

        gpu = conf.get("TAGGER_GPU", "false").lower() == "true"
        batch_size = "8" if gpu else "1"
        update_task(task_id, progress=f"0/{len(image_files)}",
                    message=f"Running WD14 tagger ({'GPU' if gpu else 'CPU'}, batch={batch_size}) on {len(image_files)} images...")

        model_dir = os.path.join(sd_scripts, "wd14_models")
        cmd = [
            venv_python, tagger_script,
            "--onnx", "--batch_size", batch_size,
            "--thresh", "0.35", "--caption_extension", ".txt",
            "--model_dir", model_dir, dataset_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            for txt_file, content in existing_captions.items():
                with open(os.path.join(dataset_dir, txt_file), 'w') as f:
                    f.write(content)
            update_task(task_id, status="failed",
                        error=f"Auto-caption failed: {result.stderr[:500]}")
            return

        processed = 0
        for img_file in image_files:
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            txt_path = os.path.join(dataset_dir, txt_file)
            if not os.path.isfile(txt_path):
                continue
            with open(txt_path, 'r') as f:
                wd14_raw = f.read().strip()
            if txt_file in existing_captions:
                merged = merge_captions(existing_captions[txt_file], wd14_raw, model, conf)
            else:
                merged = cleanup_caption(wd14_raw, model, conf)
            with open(txt_path, 'w') as f:
                f.write(merged)
            processed += 1
            update_task(task_id, progress=f"{processed}/{len(image_files)}")

        update_task(task_id, status="complete",
                    message=f"Auto-captioned {processed} images")

    except (subprocess.TimeoutExpired, Exception) as e:
        for txt_file, content in existing_captions.items():
            with open(os.path.join(dataset_dir, txt_file), 'w') as f:
                f.write(content)
        update_task(task_id, status="failed", error=str(e))


# ---------------------------------------------------------------------------
# Training system
# ---------------------------------------------------------------------------

_active_training = None  # {"task_id", "process", "run_id", "project_name"}
_training_lock = threading.Lock()

# Regex patterns for parsing kohya training output
_RE_STEP = re.compile(r'(\d+)/(\d+)\s*\[.*?(?:avr_loss[=:]?\s*)([\d.]+)')
_RE_EPOCH = re.compile(r'epoch\s+(\d+)/(\d+)')
_RE_LOSS_SIMPLE = re.compile(r'avr_loss[=:]\s*([\d.]+)')


def get_active_training():
    with _training_lock:
        if _active_training and _active_training.get("task_id"):
            task = get_task(_active_training["task_id"])
            if task and task.get("status") == "running":
                return dict(_active_training)
    return None


def load_training_runs(project_dir):
    """Load training_runs.json for a project."""
    path = os.path.join(project_dir, "training_runs.json")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {"runs": []}


def save_training_runs(project_dir, data):
    """Save training_runs.json for a project."""
    path = os.path.join(project_dir, "training_runs.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def list_lora_files(project_dir):
    """List .safetensors files in project's outputs/ directory."""
    outputs_dir = os.path.join(project_dir, "outputs")
    if not os.path.isdir(outputs_dir):
        return []
    files = []
    for f in sorted(os.listdir(outputs_dir)):
        if f.endswith('.safetensors'):
            fpath = os.path.join(outputs_dir, f)
            stat = os.stat(fpath)
            base = f.rsplit('.', 1)[0]
            has_json = os.path.isfile(os.path.join(outputs_dir, base + '.json'))
            has_preview = (os.path.isfile(os.path.join(outputs_dir, base + '.preview.png')) or
                           os.path.isfile(os.path.join(outputs_dir, base + '.png')))
            files.append({
                "filename": f,
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "has_json": has_json,
                "has_preview": has_preview,
            })
    return files


def get_lora_metadata(project_dir, lora_filename, conf):
    """Get or auto-generate metadata JSON for a LoRA file."""
    outputs_dir = os.path.join(project_dir, "outputs")
    base = lora_filename.rsplit('.', 1)[0]
    json_path = os.path.join(outputs_dir, base + '.json')
    if os.path.isfile(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    # Auto-generate from project config
    trigger = conf.get("TRIGGER", "")
    cls = conf.get("CLASS", "woman")
    model = detect_model(project_dir, conf)
    sd_version = "SDXL"
    return {
        "description": f"{trigger} {cls} LoRA",
        "sd version": sd_version,
        "activation text": f"{trigger} {cls}",
        "preferred weight": 0.75,
        "notes": "",
    }


def save_lora_metadata(project_dir, lora_filename, metadata):
    """Save metadata JSON for a LoRA file."""
    outputs_dir = os.path.join(project_dir, "outputs")
    base = lora_filename.rsplit('.', 1)[0]
    json_path = os.path.join(outputs_dir, base + '.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_lora_preview_path(project_dir, lora_filename):
    """Return preview image path if it exists, else None."""
    outputs_dir = os.path.join(project_dir, "outputs")
    base = lora_filename.rsplit('.', 1)[0]
    for ext in ['.preview.png', '.png']:
        path = os.path.join(outputs_dir, base + ext)
        if os.path.isfile(path):
            return path
    return None


def list_sample_images(project_dir):
    """List sample images from project's outputs/sample/ directory."""
    samples_dir = os.path.join(project_dir, "outputs", "sample")
    if not os.path.isdir(samples_dir):
        return []
    exts = {'.png', '.jpg', '.jpeg', '.webp'}
    return sorted(f for f in os.listdir(samples_dir)
                  if os.path.splitext(f)[1].lower() in exts)


def run_training_task(task_id, run_id, project_dir, model_type, conf):
    """Background task: launch training and parse output in real time."""
    global _active_training

    loss_history = []
    checkpoints = []
    current_epoch = 0
    total_epochs = 0
    total_steps = 0
    last_loss_step = -1

    try:
        sd_scripts = conf.get("SD_SCRIPTS", "")
        if not sd_scripts or not os.path.isdir(sd_scripts):
            update_task(task_id, status="failed", error="SD_SCRIPTS path not found")
            return

        train_script = os.path.join(project_dir, "train_character.sh")
        if not os.path.isfile(train_script):
            update_task(task_id, status="failed", error="train_character.sh not found")
            return

        update_task(task_id, message="Launching training...",
                    training={"run_id": run_id, "model": model_type,
                              "epoch": 0, "total_epochs": 0,
                              "step": 0, "total_steps": 0,
                              "avg_loss": None, "loss_history": [],
                              "checkpoints": [], "samples": [],
                              "elapsed": "", "eta": ""})

        # Launch training subprocess
        env = os.environ.copy()
        proc = subprocess.Popen(
            ["bash", train_script, model_type],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )

        with _training_lock:
            if _active_training:
                _active_training["process"] = proc

        # Read output character by character to handle tqdm \r overwrites
        line_buf = ""
        log_tail = []  # Last 20 lines for error reporting
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            ch = chunk.decode('utf-8', errors='replace')

            if ch == '\r' or ch == '\n':
                line = line_buf.strip()
                line_buf = ""
                if not line:
                    continue

                log_tail.append(line)
                if len(log_tail) > 20:
                    log_tail.pop(0)

                # Parse epoch marker
                m = _RE_EPOCH.search(line)
                if m:
                    current_epoch = int(m.group(1))
                    total_epochs = int(m.group(2))

                # Parse step progress + loss
                m = _RE_STEP.search(line)
                if m:
                    step = int(m.group(1))
                    total_steps = int(m.group(2))
                    avg_loss = float(m.group(3))

                    # Sample loss history every ~10 steps
                    if step - last_loss_step >= 10:
                        loss_history.append({"step": step, "loss": avg_loss})
                        last_loss_step = step

                    # Parse elapsed/eta from tqdm bracket
                    elapsed = ""
                    eta = ""
                    bracket = re.search(r'\[(\d+:\d+)<(\d+:\d+)', line)
                    if bracket:
                        elapsed = bracket.group(1)
                        eta = bracket.group(2)

                    update_task(task_id,
                                progress=f"{step}/{total_steps}",
                                message=f"Epoch {current_epoch}/{total_epochs} — Step {step}/{total_steps} — Loss: {avg_loss:.4f}",
                                training={"run_id": run_id, "model": model_type,
                                          "epoch": current_epoch, "total_epochs": total_epochs,
                                          "step": step, "total_steps": total_steps,
                                          "avg_loss": avg_loss,
                                          "loss_history": loss_history[-200:],
                                          "checkpoints": checkpoints,
                                          "samples": list_sample_images(project_dir),
                                          "elapsed": elapsed, "eta": eta})
                else:
                    # Check for loss in simpler format
                    m2 = _RE_LOSS_SIMPLE.search(line)
                    if m2:
                        avg_loss = float(m2.group(1))

                # Detect checkpoint saves
                if 'model saved' in line.lower() or 'saving model' in line.lower():
                    new_files = list_lora_files(project_dir)
                    checkpoints = new_files

            else:
                line_buf += ch

        proc.wait()
        exit_code = proc.returncode

        # Gather final outputs
        final_checkpoints = list_lora_files(project_dir)
        final_samples = list_sample_images(project_dir)
        final_loss = loss_history[-1]["loss"] if loss_history else None

        if exit_code == 0:
            update_task(task_id, status="complete",
                        message=f"Training complete — {total_epochs} epochs, final loss: {final_loss:.4f}" if final_loss else "Training complete",
                        training={"run_id": run_id, "model": model_type,
                                  "epoch": total_epochs, "total_epochs": total_epochs,
                                  "step": total_steps, "total_steps": total_steps,
                                  "avg_loss": final_loss,
                                  "loss_history": loss_history,
                                  "checkpoints": final_checkpoints,
                                  "samples": final_samples,
                                  "elapsed": "", "eta": ""})
        else:
            tail_text = '\n'.join(log_tail[-10:])
            update_task(task_id, status="failed",
                        error=f"Training exited with code {exit_code}\n{tail_text}")

        # Save to run history
        run_data = {
            "run_id": run_id,
            "model_type": model_type,
            "started_at": datetime.datetime.now().isoformat(),
            "status": "complete" if exit_code == 0 else "failed",
            "final_loss": final_loss,
            "loss_history": loss_history,
            "total_epochs": total_epochs,
            "total_steps": total_steps,
            "checkpoints": final_checkpoints,
            "samples": final_samples,
            "config": {
                "learning_rate": conf.get("LEARNING_RATE", "1e-4"),
                "network_dim": conf.get("NETWORK_DIM", "32"),
                "network_alpha": conf.get("NETWORK_ALPHA", "16"),
                "num_repeats": conf.get("NUM_REPEATS", "10"),
            },
        }
        history = load_training_runs(project_dir)
        history["runs"].append(run_data)
        save_training_runs(project_dir, history)

    except Exception as e:
        update_task(task_id, status="failed", error=str(e))

    finally:
        with _training_lock:
            _active_training = None


# ---------------------------------------------------------------------------
# Image listing and stats
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def get_images_with_categories(base_dir):
    """Return flat list of images with category metadata."""
    images = []
    if not os.path.isdir(base_dir):
        return images
    for filename in sorted(os.listdir(base_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        txt_path = os.path.join(base_dir, os.path.splitext(filename)[0] + '.txt')
        has_caption = os.path.isfile(txt_path)
        caption = ""
        category = "uncategorized"
        if has_caption:
            with open(txt_path, 'r') as f:
                caption = f.read().strip()
            category = categorize_caption(caption)
        images.append({
            "filename": filename, "has_caption": has_caption,
            "caption": caption, "category": category,
        })
    return images


MAX_RECOMMENDED = 50  # Character LoRAs: 30-50 images is the sweet spot


def compute_stats(images):
    """Compute category balance stats from image list."""
    counts = Counter(img["category"] for img in images)
    total = len(images)

    # Target is based on current total, capped at the recommended max.
    # Always target MAX_RECOMMENDED so the table shows what to aim for.
    target_total = MAX_RECOMMENDED

    stats = {}
    for cat in CATEGORY_ORDER:
        count = counts.get(cat, 0)
        current_pct = (count / total * 100) if total > 0 else 0
        ideal_pct = IDEAL_RATIO.get(cat, 0) * 100
        ideal_count = round(IDEAL_RATIO.get(cat, 0) * target_total)
        deficit = max(0, ideal_count - count)
        surplus = max(0, count - ideal_count) if ideal_count > 0 else 0
        stats[cat] = {
            "count": count, "current_pct": round(current_pct, 1),
            "ideal_pct": ideal_pct, "ideal_count": ideal_count,
            "deficit": deficit, "surplus": surplus,
        }

    facing_tags = ["looking_at_viewer", "looking at viewer", "facing viewer", "facing_viewer"]
    fb_images = [img for img in images if img["category"] == "full_body"]
    fb_facing = sum(1 for img in fb_images if any(t in img["caption"].lower() for t in facing_tags))

    # Repeats: aim for ~200-400 steps/epoch
    reps = max(1, round(300 / total)) if total > 0 else 10
    oversized = total > MAX_RECOMMENDED

    return {
        "categories": stats, "total": total, "target_total": target_total,
        "max_recommended": MAX_RECOMMENDED,
        "oversized": oversized,
        "full_body_facing": fb_facing, "full_body_total": len(fb_images),
        "suggested_repeats": reps,
    }


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

def make_handler(state):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            if args and '404' in str(args[0]):
                super().log_message(format, *args)

        def _read_body(self):
            length = int(self.headers.get('Content-Length', 0))
            if length > 1_000_000:
                return {}
            return json.loads(self.rfile.read(length).decode('utf-8'))

        def _json_response(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == '/' or path == '':
                self._serve_html()
            elif path == '/api/images':
                images = get_images_with_categories(state.base_dir)
                stats = compute_stats(images)
                stats["current_repeats"] = int(state.conf.get("NUM_REPEATS", 10))
                self._json_response({"images": images, "stats": stats})
            elif path.startswith('/api/caption/'):
                self._get_caption(path[13:])
            elif path.startswith('/api/tasks/'):
                task_id = path[11:]
                task = get_task(task_id)
                if task:
                    self._json_response(task)
                else:
                    self._json_response({"error": "Task not found"}, 404)
            elif path == '/api/stats':
                images = get_images_with_categories(state.base_dir)
                stats = compute_stats(images)
                stats["current_repeats"] = int(state.conf.get("NUM_REPEATS", 10))
                self._json_response(stats)
            elif path == '/api/config':
                self._get_config()
            elif path == '/api/projects':
                self._get_projects()
            elif path == '/api/dashboard':
                self._get_dashboard()
            elif path == '/api/training/active':
                self._get_active_training()
            elif path == '/api/training/runs':
                self._get_training_runs()
            elif path == '/api/training/loras':
                self._get_lora_files()
            elif path.startswith('/api/training/loras/download/'):
                self._download_lora(path.split('/')[-1])
            elif path == '/api/training/samples':
                self._get_training_samples()
            elif path.startswith('/api/training/sample/'):
                self._serve_sample_image(path[len('/api/training/sample/'):])
            elif path.startswith('/api/training/loras/metadata/'):
                self._get_lora_metadata(path.split('/')[-1])
            elif path.startswith('/api/training/loras/preview/'):
                self._serve_lora_preview(path.split('/')[-1])
            elif path.startswith('/api/training/loras/bundle/'):
                self._download_lora_bundle(path.split('/')[-1])
            elif path.startswith('/img/'):
                self._serve_image(path[5:])
            elif path == '/favicon.ico':
                self.send_response(204)
                self.send_header('Content-Length', '0')
                self.end_headers()
            else:
                self.send_error(404)

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            if path == '/api/delete':
                self._handle_delete(self._read_body())
            elif path.startswith('/api/caption/'):
                self._save_caption(path[13:], self._read_body())
            elif path == '/api/tag':
                self._handle_tag(self._read_body())
            elif path == '/api/set-category':
                self._handle_set_category(self._read_body())
            elif path == '/api/config':
                self._save_config(self._read_body())
            elif path == '/api/setup':
                self._handle_setup()
            elif path == '/api/switch-project':
                self._switch_project(self._read_body())
            elif path == '/api/train':
                self._start_training()
            elif path == '/api/train/cancel':
                self._cancel_training()
            elif path == '/api/training/loras/delete':
                self._delete_lora(self._read_body())
            elif path == '/api/training/loras/rename':
                self._rename_lora(self._read_body())
            elif path == '/api/training/samples/delete':
                self._delete_samples(self._read_body())
            elif path == '/api/training/loras/metadata':
                self._save_lora_metadata(self._read_body())
            elif path == '/api/config/update':
                self._update_config_field(self._read_body())
            elif path == '/api/training/loras/preview':
                self._set_lora_preview(self._read_body())
            else:
                self.send_error(404)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def _serve_html(self):
            html = SPA_HTML
            html = html.replace('{{DATASET_DIR}}', os.path.abspath(state.base_dir))
            html = html.replace('{{MODEL}}', state.model)
            html = html.replace('{{CATEGORY_TAGS_JSON}}', json.dumps(CATEGORY_TAGS))
            html = html.replace('{{CATEGORY_PRIMARY_TAG_JSON}}', json.dumps(CATEGORY_PRIMARY_TAG))
            html = html.replace('{{PROJECTS_JSON}}', json.dumps([
                {"name": p["name"], "model": p["model"], "trigger": p["trigger"],
                 "active": p["name"] == state.current}
                for p in state.project_list
            ]))
            body = html.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_image(self, rel_path):
            filepath = os.path.abspath(os.path.join(state.base_dir, rel_path))
            if not filepath.startswith(os.path.abspath(state.base_dir)):
                self.send_error(403)
                return
            if not os.path.isfile(filepath):
                self.send_error(404)
                return
            self._serve_file_cached(filepath)

        def _serve_file_cached(self, filepath):
            """Serve a file with ETag/Last-Modified caching."""
            stat = os.stat(filepath)
            mtime = stat.st_mtime
            etag = f'"{state.current}-{int(mtime)}-{stat.st_size}"'
            last_modified = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(mtime))

            # Check If-None-Match (ETag)
            if_none_match = self.headers.get('If-None-Match', '')
            if if_none_match == etag:
                self.send_response(304)
                self.end_headers()
                return

            # Only check If-Modified-Since if ETag also matches (avoids
            # cross-project cache collisions where mtime happens to match)
            # Skip — ETag is the authoritative cache validator

            mime = mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(stat.st_size))
            self.send_header('Cache-Control', 'max-age=300, must-revalidate')
            self.send_header('ETag', etag)
            self.send_header('Last-Modified', last_modified)
            self.end_headers()
            try:
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            except BrokenPipeError:
                pass

        def _get_caption(self, rel_path):
            img_path = os.path.abspath(os.path.join(state.base_dir, rel_path))
            if not img_path.startswith(os.path.abspath(state.base_dir)):
                self.send_error(403)
                return
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            caption = ''
            if os.path.isfile(txt_path):
                with open(txt_path, 'r') as f:
                    caption = f.read().strip()
            self._json_response({'caption': caption, 'file': rel_path})

        def _save_caption(self, rel_path, data):
            img_path = os.path.abspath(os.path.join(state.base_dir, rel_path))
            if not img_path.startswith(os.path.abspath(state.base_dir)):
                self.send_error(403)
                return
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            caption = data.get('caption', '')
            tags = dedupe_tags([t.strip() for t in caption.split(',')])
            caption = ', '.join(t for t in tags if t)
            with open(txt_path, 'w') as f:
                f.write(caption)
            category = categorize_caption(caption)
            print(f"  Saved caption: {os.path.basename(txt_path)} [{category}]")
            self._json_response({'saved': True, 'file': rel_path, 'category': category})

        def _handle_delete(self, data):
            files = data.get('files', [])
            deleted = []
            for rel_path in files:
                filepath = os.path.abspath(os.path.join(state.base_dir, rel_path))
                if not filepath.startswith(os.path.abspath(state.base_dir)):
                    continue
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    deleted.append(rel_path)
                    print(f"  Deleted: {rel_path}")
                    txt_path = os.path.splitext(filepath)[0] + '.txt'
                    if os.path.isfile(txt_path):
                        os.remove(txt_path)
            self._json_response({'deleted': deleted})

        def _handle_tag(self, data):
            task_id = create_task("WD14 Tagger")
            # Capture current state for the background thread
            thread = threading.Thread(
                target=run_tagger_task,
                args=(task_id, os.path.abspath(state.base_dir), state.model, dict(state.conf)),
                daemon=True,
            )
            thread.start()
            self._json_response({"task_id": task_id})

        def _handle_set_category(self, data):
            filename = data.get('filename', '')
            new_category = data.get('category', '')
            if new_category not in CATEGORY_PRIMARY_TAG:
                self._json_response({"error": "Invalid category"}, 400)
                return

            filepath = os.path.abspath(os.path.join(state.base_dir, filename))
            if not filepath.startswith(os.path.abspath(state.base_dir)):
                self.send_error(403)
                return

            txt_path = os.path.splitext(filepath)[0] + '.txt'
            caption = ''
            if os.path.isfile(txt_path):
                with open(txt_path, 'r') as f:
                    caption = f.read().strip()

            tags = [t.strip() for t in caption.split(',')]
            all_cat_tags = set()
            for tag_list in CATEGORY_TAGS.values():
                all_cat_tags.update(t.lower() for t in tag_list)
            cleaned_tags = [t for t in tags if t.lower() not in all_cat_tags]

            prefix = make_prefix(state.model, state.conf)
            prefix_tags = [t.strip() for t in prefix.split(',')]
            prefix_len = len(prefix_tags)

            new_tag = CATEGORY_PRIMARY_TAG[new_category]
            result_tags = cleaned_tags[:prefix_len] + [new_tag] + cleaned_tags[prefix_len:]
            result_tags = dedupe_tags(result_tags)
            new_caption = ', '.join(t for t in result_tags if t)

            with open(txt_path, 'w') as f:
                f.write(new_caption)

            print(f"  Set category: {filename} -> {new_category}")
            self._json_response({
                'saved': True, 'filename': filename,
                'category': new_category, 'caption': new_caption,
            })

        def _get_config(self):
            raw = ''
            if os.path.isfile(state.conf_path):
                with open(state.conf_path, 'r') as f:
                    raw = f.read()
            self._json_response({
                'raw': raw, 'path': state.conf_path, 'model': state.model,
            })

        def _save_config(self, data):
            raw = data.get('raw', '')
            if not raw.strip():
                self._json_response({'error': 'Empty config'}, 400)
                return
            try:
                with open(state.conf_path, 'w') as f:
                    f.write(raw)
                state.conf = load_conf(state.conf_path)
                print(f"  Saved project.conf ({len(raw)} bytes)")
                self._json_response({'saved': True})
            except SystemExit:
                self._json_response({'error': 'Config parse error'}, 400)
            except Exception as e:
                self._json_response({'error': str(e)}, 500)

        def _update_config_field(self, data):
            """Update a single field in project.conf."""
            key = data.get('key', '')
            value = data.get('value', '')
            if not key:
                self._json_response({'error': 'Missing key'}, 400)
                return
            try:
                with open(state.conf_path, 'r') as f:
                    lines = f.readlines()
                found = False
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith(key + '=') or stripped.startswith(key + ' ='):
                        lines[i] = f'{key}={value}\n'
                        found = True
                        break
                if not found:
                    lines.append(f'{key}={value}\n')
                with open(state.conf_path, 'w') as f:
                    f.writelines(lines)
                state.conf = load_conf(state.conf_path)
                print(f"  Updated {key}={value} in project.conf")
                self._json_response({'saved': True})
            except Exception as e:
                self._json_response({'error': str(e)}, 500)

        def _handle_setup(self):
            proj = state.projects[state.current]
            project_dir = proj["dir"]
            dirs = [
                os.path.join(project_dir, "dataset"),
                os.path.join(project_dir, "outputs"),
                os.path.join(project_dir, "logs"),
                os.path.join(project_dir, "samples"),
            ]
            created = []
            for d in dirs:
                if not os.path.isdir(d):
                    os.makedirs(d, exist_ok=True)
                    created.append(os.path.relpath(d, project_dir))
            if created:
                print(f"  Setup created: {', '.join(created)}")
            self._json_response({
                'created': created,
                'message': f"Created {len(created)} directories" if created else "All directories already exist",
            })

        def _get_projects(self):
            self._json_response({
                'projects': [
                    {"name": p["name"], "model": p["model"], "trigger": p["trigger"],
                     "active": p["name"] == state.current}
                    for p in state.project_list
                ],
                'current': state.current,
            })

        def _get_dashboard(self):
            """Aggregate data across all projects for the dashboard."""
            active = get_active_training()
            active_info = None
            if active:
                task = get_task(active["task_id"])
                active_info = {
                    "task_id": active["task_id"],
                    "project": active.get("project_name", ""),
                    "model": active.get("model", ""),
                    "task": task,
                }

            all_runs = []
            all_loras = []
            project_summaries = []
            for p in state.project_list:
                pdir = p["dir"]
                # Runs
                history = load_training_runs(pdir)
                for r in history.get("runs", []):
                    r["project"] = p["name"]
                    all_runs.append(r)
                # LoRA files
                for lf in list_lora_files(pdir):
                    lf["project"] = p["name"]
                    all_loras.append(lf)
                # Image count
                dataset_dir = p["dataset_dir"]
                img_count = sum(1 for f in os.listdir(dataset_dir)
                                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS) if os.path.isdir(dataset_dir) else 0
                project_summaries.append({
                    "name": p["name"], "model": p["model"],
                    "trigger": p["trigger"], "images": img_count,
                    "runs": len(history.get("runs", [])),
                    "loras": sum(1 for lf in list_lora_files(pdir)),
                })

            # Sort runs by date, newest first
            all_runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)

            self._json_response({
                "active_training": active_info,
                "projects": project_summaries,
                "runs": all_runs,
                "loras": all_loras,
            })

        def _switch_project(self, data):
            name = data.get('name', '')
            if state.switch_to(name):
                img_count = sum(1 for f in os.listdir(state.base_dir)
                                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS) if os.path.isdir(state.base_dir) else 0
                print(f"  Switched to: {name} ({state.model}, {img_count} images)")
                self._json_response({
                    'switched': True, 'name': name,
                    'model': state.model,
                    'dataset_dir': os.path.abspath(state.base_dir),
                })
            else:
                self._json_response({'error': f'Unknown project: {name}'}, 400)

        # --- Training endpoints ---

        def _start_training(self):
            global _active_training
            active = get_active_training()
            if active:
                self._json_response({"error": "Training already running", "task_id": active["task_id"]}, 409)
                return

            # Count images to validate
            img_count = sum(1 for f in os.listdir(state.base_dir)
                           if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS) if os.path.isdir(state.base_dir) else 0
            if img_count == 0:
                self._json_response({"error": "No images in dataset"}, 400)
                return

            task_id = create_task(f"Training {state.current} ({state.model})")
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{state.model}"

            proj = state.projects[state.current]
            project_dir = proj["dir"]

            with _training_lock:
                _active_training = {
                    "task_id": task_id,
                    "run_id": run_id,
                    "project_name": state.current,
                    "model": state.model,
                    "process": None,
                }

            thread = threading.Thread(
                target=run_training_task,
                args=(task_id, run_id, project_dir, state.model, dict(state.conf)),
                daemon=True,
            )
            thread.start()
            print(f"  Training started: {state.current} ({state.model}), task={task_id}")
            self._json_response({"task_id": task_id, "run_id": run_id})

        def _cancel_training(self):
            global _active_training
            with _training_lock:
                if not _active_training:
                    self._json_response({"error": "No training running"}, 400)
                    return
                proc = _active_training.get("process")
                task_id = _active_training["task_id"]
                if proc:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        proc.terminate()
                update_task(task_id, status="failed", error="Cancelled by user")
                _active_training = None
            print("  Training cancelled")
            self._json_response({"cancelled": True})

        def _get_active_training(self):
            active = get_active_training()
            if active:
                task = get_task(active["task_id"])
                self._json_response({"active": True, "task_id": active["task_id"],
                                     "project": active.get("project_name", ""),
                                     "model": active.get("model", ""), "task": task})
            else:
                self._json_response({"active": False})

        def _get_training_runs(self):
            proj = state.projects[state.current]
            runs = load_training_runs(proj["dir"])
            self._json_response(runs)

        def _get_lora_files(self):
            proj = state.projects[state.current]
            files = list_lora_files(proj["dir"])
            self._json_response({"files": files})

        def _download_lora(self, filename):
            proj = state.projects[state.current]
            outputs_dir = os.path.join(proj["dir"], "outputs")
            filepath = os.path.abspath(os.path.join(outputs_dir, filename))
            if not filepath.startswith(os.path.abspath(outputs_dir)):
                self.send_error(403)
                return
            if not os.path.isfile(filepath):
                self.send_error(404)
                return
            file_size = os.path.getsize(filepath)
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.send_header('Content-Length', str(file_size))
            self.end_headers()
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except BrokenPipeError:
                        break

        def _get_training_samples(self):
            proj = state.projects[state.current]
            samples = list_sample_images(proj["dir"])
            self._json_response({"samples": samples})

        def _serve_sample_image(self, rel_path):
            proj = state.projects[state.current]
            samples_dir = os.path.join(proj["dir"], "outputs", "sample")
            filepath = os.path.abspath(os.path.join(samples_dir, rel_path))
            if not filepath.startswith(os.path.abspath(samples_dir)):
                self.send_error(403)
                return
            if not os.path.isfile(filepath):
                self.send_error(404)
                return
            self._serve_file_cached(filepath)

        def _delete_lora(self, data):
            filename = data.get('filename', '')
            if not filename or '/' in filename or '\\' in filename:
                self._json_response({"error": "Invalid filename"}, 400)
                return
            proj = state.projects[state.current]
            outputs_dir = os.path.join(proj["dir"], "outputs")
            filepath = os.path.abspath(os.path.join(outputs_dir, filename))
            if not filepath.startswith(os.path.abspath(outputs_dir)):
                self.send_error(403)
                return
            if not os.path.isfile(filepath):
                self._json_response({"error": "File not found"}, 404)
                return
            os.remove(filepath)
            print(f"  Deleted LoRA: {filename}")
            self._json_response({"deleted": True, "filename": filename})

        def _rename_lora(self, data):
            old_name = data.get('old_name', '')
            new_name = data.get('new_name', '')
            if not old_name or not new_name:
                self._json_response({"error": "Missing filenames"}, 400)
                return
            for name in [old_name, new_name]:
                if '/' in name or '\\' in name:
                    self._json_response({"error": "Invalid filename"}, 400)
                    return
            if not new_name.endswith('.safetensors'):
                new_name += '.safetensors'
            proj = state.projects[state.current]
            outputs_dir = os.path.join(proj["dir"], "outputs")
            old_path = os.path.abspath(os.path.join(outputs_dir, old_name))
            new_path = os.path.abspath(os.path.join(outputs_dir, new_name))
            if not old_path.startswith(os.path.abspath(outputs_dir)) or \
               not new_path.startswith(os.path.abspath(outputs_dir)):
                self.send_error(403)
                return
            if not os.path.isfile(old_path):
                self._json_response({"error": "File not found"}, 404)
                return
            if os.path.exists(new_path):
                self._json_response({"error": "Target name already exists"}, 409)
                return
            os.rename(old_path, new_path)
            print(f"  Renamed LoRA: {old_name} -> {new_name}")
            self._json_response({"renamed": True, "old_name": old_name, "new_name": new_name})

        def _delete_samples(self, data):
            files = data.get('files', [])
            proj = state.projects[state.current]
            samples_dir = os.path.join(proj["dir"], "outputs", "sample")
            deleted = []
            for f in files:
                if '/' in f or '\\' in f:
                    continue
                filepath = os.path.abspath(os.path.join(samples_dir, f))
                if not filepath.startswith(os.path.abspath(samples_dir)):
                    continue
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    deleted.append(f)
            if deleted:
                print(f"  Deleted {len(deleted)} sample(s)")
            self._json_response({"deleted": deleted})

        def _get_lora_metadata(self, lora_filename):
            proj = state.projects[state.current]
            metadata = get_lora_metadata(proj["dir"], lora_filename, state.conf)
            self._json_response(metadata)

        def _save_lora_metadata(self, data):
            lora_filename = data.get('filename', '')
            metadata = data.get('metadata', {})
            if not lora_filename:
                self._json_response({"error": "Missing filename"}, 400)
                return
            proj = state.projects[state.current]
            save_lora_metadata(proj["dir"], lora_filename, metadata)
            print(f"  Saved metadata for {lora_filename}")
            self._json_response({"saved": True})

        def _serve_lora_preview(self, lora_filename):
            proj = state.projects[state.current]
            preview_path = get_lora_preview_path(proj["dir"], lora_filename)
            if not preview_path:
                self.send_error(404)
                return
            self._serve_file_cached(preview_path)

        def _set_lora_preview(self, data):
            """Copy a dataset image as the preview for a LoRA file."""
            lora_filename = data.get('filename', '')
            image_filename = data.get('image', '')
            if not lora_filename or not image_filename:
                self._json_response({"error": "Missing filename or image"}, 400)
                return
            # Source: dataset image
            src = os.path.abspath(os.path.join(state.base_dir, image_filename))
            if not src.startswith(os.path.abspath(state.base_dir)) or not os.path.isfile(src):
                self._json_response({"error": "Image not found"}, 404)
                return
            # Destination: outputs/base.preview.png
            proj = state.projects[state.current]
            outputs_dir = os.path.join(proj["dir"], "outputs")
            base = lora_filename.rsplit('.', 1)[0]
            dst = os.path.join(outputs_dir, base + '.preview.png')
            shutil.copy2(src, dst)
            print(f"  Set preview for {lora_filename}: {image_filename}")
            self._json_response({"saved": True, "preview": base + '.preview.png'})

        def _download_lora_bundle(self, lora_filename):
            """Download zip bundle: .safetensors + .json + .preview.png."""
            proj = state.projects[state.current]
            outputs_dir = os.path.join(proj["dir"], "outputs")
            safetensors_path = os.path.abspath(os.path.join(outputs_dir, lora_filename))
            if not safetensors_path.startswith(os.path.abspath(outputs_dir)):
                self.send_error(403)
                return
            if not os.path.isfile(safetensors_path):
                self.send_error(404)
                return

            base = lora_filename.rsplit('.', 1)[0]

            # Auto-generate JSON if missing
            json_path = os.path.join(outputs_dir, base + '.json')
            if not os.path.isfile(json_path):
                metadata = get_lora_metadata(proj["dir"], lora_filename, state.conf)
                save_lora_metadata(proj["dir"], lora_filename, metadata)

            # Build zip in memory
            buf = BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:
                zf.write(safetensors_path, lora_filename)
                if os.path.isfile(json_path):
                    zf.write(json_path, base + '.json')
                preview_path = get_lora_preview_path(proj["dir"], lora_filename)
                if preview_path:
                    zf.write(preview_path, base + '.preview.png')

            zip_data = buf.getvalue()
            zip_name = base + '.zip'
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Disposition', f'attachment; filename="{zip_name}"')
            self.send_header('Content-Length', str(len(zip_data)))
            self.end_headers()
            self.wfile.write(zip_data)

    return Handler


# ---------------------------------------------------------------------------
# SPA HTML/JS/CSS
# ---------------------------------------------------------------------------

SPA_HTML = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Prep</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #1a1a2e; color: #eee; }

  /* Header */
  .header { padding: 12px 20px; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 15px; }
  .header h1 { font-size: 1.2em; white-space: nowrap; }
  .header .meta { color: #888; font-size: 0.8em; }
  .header select { background: #0f3460; color: #f39c12; border: 1px solid #555; border-radius: 6px;
                   padding: 6px 12px; font-size: 0.85em; font-weight: 600; cursor: pointer; }
  .header select:focus { outline: none; border-color: #f39c12; }

  /* Tabs */
  .tabs { display: flex; border-bottom: 2px solid #333; padding: 0 10px; background: #16213e;
          position: sticky; top: 0; z-index: 100; }
  .tab { padding: 10px 18px; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px;
         font-size: 0.85em; color: #aaa; transition: all 0.15s; white-space: nowrap; user-select: none; }
  .tab:hover { color: #eee; background: rgba(255,255,255,0.05); }
  .tab.active { color: #f39c12; border-bottom-color: #f39c12; }
  .tab .badge { background: #333; color: #aaa; padding: 1px 7px; border-radius: 10px; font-size: 0.8em; margin-left: 5px; }
  .tab.active .badge { background: #f39c12; color: #1a1a2e; }

  /* Toolbar */
  .toolbar { padding: 10px 20px; border-bottom: 1px solid #333; display: flex; gap: 10px; align-items: center;
             flex-wrap: wrap; background: #1a1a2e; position: sticky; top: 42px; z-index: 99; }
  .toolbar button { padding: 7px 14px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85em; font-weight: 600; }
  .btn-tag { background: #9b59b6; color: #fff; }
  .btn-tag:hover { background: #8e44ad; }
  .btn-tag:disabled { background: #555; cursor: not-allowed; }
  .btn-delete { background: #e74c3c; color: #fff; }
  .btn-delete:hover { background: #c0392b; }
  .btn-delete:disabled { background: #555; cursor: not-allowed; }
  .btn-select { background: #3498db; color: #fff; }
  .btn-select:hover { background: #2980b9; }
  .btn-refresh { background: #2ecc71; color: #fff; }
  .btn-refresh:hover { background: #27ae60; }
  .count { color: #f39c12; font-weight: bold; font-size: 0.85em; margin-left: auto; }

  /* Gallery grid */
  .gallery { padding: 15px 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
  .card { position: relative; border-radius: 8px; overflow: hidden; cursor: pointer; border: 3px solid transparent;
          transition: border-color 0.15s, transform 0.1s; background: #16213e; }
  .card:hover { transform: scale(1.02); }
  .card.selected { border-color: #e74c3c; }
  .card img { width: 100%; display: block; aspect-ratio: auto; }
  .card .name { padding: 5px 8px; font-size: 0.7em; color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card .check { position: absolute; top: 8px; right: 8px; width: 22px; height: 22px; border-radius: 50%;
                 background: rgba(0,0,0,0.6); border: 2px solid #fff; display: flex; align-items: center; justify-content: center; }
  .card.selected .check { background: #e74c3c; }
  .card.selected .check::after { content: "\\2715"; color: #fff; font-weight: bold; font-size: 12px; }
  .card .caption-dot { position: absolute; top: 8px; left: 8px; width: 10px; height: 10px; border-radius: 50%;
                       border: 1px solid rgba(0,0,0,0.3); }
  .caption-dot.has { background: #2ecc71; }
  .caption-dot.none { background: #e74c3c; }
  .card .cat-label { position: absolute; bottom: 28px; left: 0; padding: 2px 8px; font-size: 0.65em;
                     background: rgba(0,0,0,0.7); color: #f39c12; border-radius: 0 4px 4px 0; }

  /* Modal */
  .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.95);
           z-index: 200; }
  .modal.active { display: flex; }
  .modal-layout { display: flex; width: 100%; height: 100%; }
  .modal-image { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; min-width: 0; }
  .modal-image img { max-width: 100%; max-height: 100vh; object-fit: contain; }
  .modal-sidebar { width: 420px; background: #16213e; display: flex; flex-direction: column; border-left: 1px solid #333; }
  .modal-sidebar h3 { padding: 12px 15px; border-bottom: 1px solid #333; font-size: 0.9em; color: #f39c12; }
  .modal-info { padding: 8px 15px; color: #888; font-size: 0.78em; border-bottom: 1px solid #333; }

  /* Category quick-set buttons */
  .category-buttons { padding: 10px 15px; border-bottom: 1px solid #333; display: flex; gap: 6px; flex-wrap: wrap; }
  .category-buttons label { font-size: 0.7em; color: #888; margin-right: auto; width: 100%; margin-bottom: 2px; }
  .cat-btn { padding: 6px 12px; border: 2px solid #555; border-radius: 6px; background: transparent;
             color: #aaa; font-size: 0.78em; cursor: pointer; font-weight: 600; transition: all 0.15s; }
  .cat-btn:hover { border-color: #f39c12; color: #f39c12; }
  .cat-btn.active { border-color: #f39c12; background: #f39c12; color: #1a1a2e; }

  /* Quick tag buttons */
  .quick-tags { padding: 8px 15px; border-bottom: 1px solid #333; display: flex; flex-wrap: wrap; gap: 5px; }
  .quick-tags label { font-size: 0.7em; color: #888; margin-right: auto; width: 100%; margin-bottom: 2px; }
  .tag-btn { padding: 3px 9px; border: 1px solid #555; border-radius: 4px; background: #0f3460;
             color: #aaa; font-size: 0.72em; cursor: pointer; transition: all 0.15s; }
  .tag-btn:hover { border-color: #3498db; color: #fff; }
  .tag-btn.active { background: #3498db; color: #fff; border-color: #3498db; }

  /* Caption area */
  .caption-area { flex: 1; padding: 12px 15px; display: flex; flex-direction: column; }
  .caption-area textarea { flex: 1; background: #0f3460; color: #eee; border: 1px solid #333; border-radius: 6px;
                           padding: 10px; font-family: monospace; font-size: 0.82em; resize: none; line-height: 1.5; }
  .caption-area textarea:focus { outline: none; border-color: #3498db; }
  .btn-row { display: flex; gap: 8px; padding-top: 10px; }
  .btn-row button { flex: 1; padding: 10px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85em; }
  .btn-save { background: #2ecc71; color: #fff; }
  .btn-save:hover { background: #27ae60; }
  .btn-save.dirty { background: #f39c12; }
  .btn-modal-delete { background: #e74c3c; color: #fff; }
  .btn-modal-delete:hover { background: #c0392b; }

  .modal .close { position: absolute; top: 10px; right: 15px; color: #fff; font-size: 1.8em; cursor: pointer; z-index: 10; }
  .modal .nav { position: absolute; top: 50%; color: #fff; font-size: 3em; cursor: pointer; padding: 20px;
                user-select: none; transform: translateY(-50%); z-index: 10; }
  .modal .nav.prev { left: 10px; }
  .modal .nav.next { right: 10px; }

  /* Stats view */
  .stats-view { padding: 30px 40px; max-width: 800px; }
  .stats-view h2 { margin-bottom: 20px; font-size: 1.2em; color: #f39c12; }
  .stats-table { width: 100%; border-collapse: collapse; margin-bottom: 25px; }
  .stats-table th { text-align: left; padding: 8px 12px; border-bottom: 2px solid #555; color: #aaa; font-size: 0.82em; }
  .stats-table td { padding: 8px 12px; border-bottom: 1px solid #333; font-size: 0.85em; }
  .stats-bar { height: 16px; border-radius: 3px; background: #333; overflow: hidden; min-width: 100px; }
  .stats-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .stats-bar-fill.ok { background: #2ecc71; }
  .stats-bar-fill.low { background: #f39c12; }
  .stats-bar-fill.over { background: #3498db; }
  .stats-bar-fill.warn { background: #e74c3c; }
  .stats-summary { color: #aaa; font-size: 0.9em; line-height: 1.8; }
  .stats-summary strong { color: #eee; }

  /* Config view */
  .config-view { padding: 30px 40px; max-width: 900px; display: flex; flex-direction: column; height: calc(100vh - 120px); }
  .config-view h2 { margin-bottom: 8px; font-size: 1.2em; color: #f39c12; }
  .config-path { color: #666; font-size: 0.78em; margin-bottom: 12px; font-family: monospace; }
  .config-view textarea { flex: 1; background: #0f3460; color: #eee; border: 1px solid #333; border-radius: 6px;
                          padding: 15px; font-family: monospace; font-size: 0.85em; resize: none; line-height: 1.7;
                          tab-size: 4; }
  .config-view textarea:focus { outline: none; border-color: #3498db; }
  .config-actions { display: flex; gap: 10px; padding-top: 12px; align-items: center; }
  .config-actions button { padding: 8px 20px; }
  .config-status { font-size: 0.82em; color: #888; margin-left: 10px; }
  .config-actions .btn-save.dirty { background: #f39c12; }

  /* Dashboard view */
  .dashboard-view { padding: 30px 40px; max-width: 1000px; }
  .dashboard-view h2 { margin-bottom: 15px; }
  .dash-active { background: #1a2744; border: 2px solid #9b59b6; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; }
  .dash-active .status { color: #9b59b6; font-weight: 600; font-size: 1.1em; }
  .dash-projects { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
  .dash-project { background: #16213e; border-radius: 8px; padding: 15px; cursor: pointer; border: 2px solid transparent; transition: border-color 0.15s; }
  .dash-project:hover { border-color: #f39c12; }
  .dash-project .name { color: #f39c12; font-weight: 600; font-size: 0.95em; margin-bottom: 5px; }
  .dash-project .meta { color: #888; font-size: 0.8em; line-height: 1.5; }

  /* Training view */
  .training-view { padding: 30px 40px; max-width: 900px; }
  .training-view h2 { margin-bottom: 15px; font-size: 1.2em; color: #f39c12; }
  .training-view h3 { margin: 25px 0 10px; font-size: 1em; color: #f39c12; border-bottom: 1px solid #333; padding-bottom: 5px; }
  .train-summary { color: #aaa; font-size: 0.9em; margin-bottom: 15px; line-height: 1.6; }
  .train-actions { margin: 10px 0; }
  .train-monitor { margin-bottom: 20px; }
  .train-info { color: #aaa; font-size: 0.9em; margin-bottom: 10px; }
  .train-progress { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
  .train-progress-bar { flex: 1; height: 12px; background: #333; border-radius: 6px; overflow: hidden; }
  .train-progress-fill { height: 100%; background: #9b59b6; border-radius: 6px; transition: width 0.5s; }
  .train-progress-text { font-size: 0.85em; color: #aaa; white-space: nowrap; }
  .train-loss { font-size: 1.4em; color: #2ecc71; font-weight: bold; margin: 8px 0; }
  .train-section { margin-bottom: 20px; }
  .lora-files { display: flex; flex-direction: column; gap: 6px; }
  .lora-file { display: flex; align-items: center; gap: 12px; padding: 8px 12px; background: #16213e; border-radius: 6px; }
  .lora-file .name { flex: 1; font-family: monospace; font-size: 0.85em; }
  .lora-file .size { color: #888; font-size: 0.8em; }
  .lora-file .date { color: #666; font-size: 0.75em; }
  .lora-file a { color: #3498db; text-decoration: none; font-size: 0.85em; font-weight: 600; }
  .lora-file a:hover { color: #2980b9; }
  .run-row { padding: 10px 12px; background: #16213e; border-radius: 6px; margin-bottom: 6px; cursor: pointer; }
  .run-row:hover { background: #1a2744; }
  .run-header { display: flex; gap: 15px; align-items: center; font-size: 0.85em; }
  .run-header .run-date { color: #aaa; }
  .run-header .run-model { color: #f39c12; font-weight: 600; }
  .run-header .run-loss { color: #2ecc71; }
  .run-header .run-status { font-weight: 600; }
  .run-header .run-status.complete { color: #2ecc71; }
  .run-header .run-status.failed { color: #e74c3c; }
  .run-detail { padding: 10px 0; display: none; }
  .run-detail.open { display: block; }
  .train-samples { display: flex; flex-wrap: wrap; gap: 8px; }
  .train-samples img { height: 120px; border-radius: 4px; cursor: pointer; }
  .train-samples img:hover { opacity: 0.8; }

  /* Task progress bar */
  .task-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #16213e; border-top: 1px solid #333;
              padding: 10px 20px; display: none; z-index: 300; align-items: center; gap: 15px; }
  .task-bar.active { display: flex; }
  .task-bar .task-label { font-size: 0.85em; white-space: nowrap; }
  .task-bar .task-progress { flex: 1; height: 8px; background: #333; border-radius: 4px; overflow: hidden; }
  .task-bar .task-fill { height: 100%; background: #9b59b6; border-radius: 4px; transition: width 0.3s; }
  .task-bar .task-status { font-size: 0.8em; color: #aaa; }
  .task-bar.complete .task-fill { background: #2ecc71; }
  .task-bar.failed .task-fill { background: #e74c3c; }

  /* Toast */
  .toast { position: fixed; bottom: 60px; right: 20px; background: #2ecc71; color: #fff; padding: 12px 20px;
           border-radius: 8px; font-weight: 600; z-index: 400; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
  .toast.show { opacity: 1; }
  .toast.error { background: #e74c3c; }

  /* Empty state */
  .empty { text-align: center; padding: 60px 20px; color: #555; }
  .empty p { font-size: 1.1em; margin-bottom: 10px; }
</style>
</head>
<body>

<div class="header">
  <h1>Dataset Prep</h1>
  <select id="projectSwitcher" onchange="switchProject(this.value)"></select>
  <span class="meta" id="headerMeta">{{DATASET_DIR}} | {{MODEL}}</span>
</div>

<div class="tabs" id="tabBar">
  <div class="tab" data-tab="dashboard" onclick="switchTab('dashboard')">Dashboard</div>
  <div class="tab active" data-tab="uncategorized" onclick="switchTab('uncategorized')">Uncategorized <span class="badge" id="badge-uncategorized">0</span></div>
  <div class="tab" data-tab="face_closeup" onclick="switchTab('face_closeup')">Face Closeup <span class="badge" id="badge-face_closeup">0</span></div>
  <div class="tab" data-tab="head_shoulders" onclick="switchTab('head_shoulders')">Head/Shoulders <span class="badge" id="badge-head_shoulders">0</span></div>
  <div class="tab" data-tab="upper_body" onclick="switchTab('upper_body')">Upper Body <span class="badge" id="badge-upper_body">0</span></div>
  <div class="tab" data-tab="full_body" onclick="switchTab('full_body')">Full Body <span class="badge" id="badge-full_body">0</span></div>
  <div class="tab" data-tab="all" onclick="switchTab('all')">All <span class="badge" id="badge-all">0</span></div>
  <div class="tab" data-tab="stats" onclick="switchTab('stats')">Stats</div>
  <div class="tab" data-tab="config" onclick="switchTab('config')">Config</div>
  <div class="tab" data-tab="training" onclick="switchTab('training')">Training</div>
</div>

<div class="toolbar" id="toolbar">
  <button class="btn-tag" id="tagBtn" onclick="runTagger()">Auto Caption</button>
  <button class="btn-delete" id="deleteBtn" disabled onclick="deleteSelected()">Delete (0)</button>
  <button class="btn-select" onclick="selectAllVisible()">Select All</button>
  <button class="btn-select" onclick="selectNone()">Select None</button>
  <button class="btn-select" onclick="invertVisible()">Invert</button>
  <button class="btn-refresh" onclick="loadImages()">Refresh</button>
  <span class="count" id="totalCount"></span>
</div>

<div class="gallery" id="gallery">
  <div class="grid" id="grid"></div>
</div>

<div class="dashboard-view" id="dashboardView" style="display:none;">
  <h2 style="color:#f39c12;margin-bottom:20px;">Dashboard</h2>
  <div id="dashActiveTraining"></div>
  <div class="train-section">
    <h3>Projects</h3>
    <div id="dashProjects"></div>
  </div>
  <div class="train-section">
    <h3>All Training Runs</h3>
    <div id="dashRuns"></div>
  </div>
  <div class="train-section">
    <h3>All LoRA Files</h3>
    <div id="dashLoras" class="lora-files"></div>
  </div>
</div>

<div class="stats-view" id="statsView" style="display:none;"></div>

<div class="config-view" id="configView" style="display:none;">
  <h2>project.conf</h2>
  <div class="config-path" id="configPath"></div>
  <textarea id="configText" spellcheck="false"></textarea>
  <div class="config-actions">
    <button class="btn-save" id="configSaveBtn" onclick="saveConfig()">Save (Ctrl+S)</button>
    <button class="btn-refresh" onclick="loadConfig()">Reload</button>
    <button class="btn-tag" onclick="runSetup()">Run Setup</button>
    <span class="config-status" id="configStatus"></span>
  </div>
</div>

<div class="training-view" id="trainingView" style="display:none;">
  <div class="train-launch" id="trainLaunch">
    <h2>Training</h2>
    <div class="train-summary" id="trainSummary"></div>
    <div class="train-actions">
      <button class="btn-tag" id="trainStartBtn" onclick="startTraining()" style="font-size:1em;padding:12px 30px;">Start Training</button>
    </div>
  </div>
  <div class="train-monitor" id="trainMonitor" style="display:none;">
    <h2>Training in Progress</h2>
    <div class="train-info" id="trainInfo"></div>
    <div class="train-progress">
      <div class="train-progress-bar"><div class="train-progress-fill" id="trainFill" style="width:0%"></div></div>
      <span class="train-progress-text" id="trainProgressText"></span>
    </div>
    <div class="train-loss" id="trainLoss"></div>
    <svg id="lossChart" viewBox="0 0 600 200" style="width:100%;max-width:600px;height:200px;background:#0f3460;border-radius:8px;margin:10px 0;"></svg>
    <div class="train-actions">
      <button class="btn-delete" onclick="cancelTraining()" style="padding:8px 20px;">Cancel Training</button>
    </div>
  </div>
  <div class="train-section">
    <h3>LoRA Files</h3>
    <div id="loraFiles" class="lora-files"></div>
  </div>
  <div class="train-section">
    <h3>Run History</h3>
    <div id="runHistory"></div>
  </div>
  <div class="train-section">
    <h3>Training Samples</h3>
    <div id="trainSamples" class="train-samples"></div>
  </div>
</div>

<div class="modal" id="modal">
  <div class="modal-layout">
    <div class="modal-image">
      <span class="close" onclick="closeModal()">&times;</span>
      <span class="nav prev" onclick="navModal(-1); event.stopPropagation();">&#8249;</span>
      <img id="modalImg" src="">
      <span class="nav next" onclick="navModal(1); event.stopPropagation();">&#8250;</span>
    </div>
    <div class="modal-sidebar">
      <h3>Caption Editor</h3>
      <div class="modal-info" id="modalInfo"></div>
      <div class="category-buttons">
        <label>Category</label>
        <button class="cat-btn" data-cat="face_closeup" onclick="setCategory('face_closeup')">Face Closeup</button>
        <button class="cat-btn" data-cat="head_shoulders" onclick="setCategory('head_shoulders')">Head/Shoulders</button>
        <button class="cat-btn" data-cat="upper_body" onclick="setCategory('upper_body')">Upper Body</button>
        <button class="cat-btn" data-cat="full_body" onclick="setCategory('full_body')">Full Body</button>
      </div>
      <div class="quick-tags" id="quickTags">
        <label>Tags</label>
      </div>
      <div class="caption-area">
        <textarea id="captionText" placeholder="No caption yet. Type tags and save to create one."
                  oninput="onCaptionInput()"></textarea>
        <div class="btn-row">
          <button class="btn-save" id="saveBtn" onclick="saveCaption()">Save (Ctrl+S)</button>
          <button class="btn-modal-delete" onclick="deleteCurrentInModal()">Delete</button>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="task-bar" id="taskBar">
  <span class="task-label" id="taskLabel">Captioning...</span>
  <div class="task-progress"><div class="task-fill" id="taskFill" style="width:0%"></div></div>
  <span class="task-status" id="taskStatus"></span>
</div>

<div class="toast" id="toast"></div>

<script>
// --- Config injected from server ---
const CATEGORY_TAGS = JSON.parse('{{CATEGORY_TAGS_JSON}}');
const CATEGORY_PRIMARY_TAG = JSON.parse('{{CATEGORY_PRIMARY_TAG_JSON}}');
const CATEGORY_ORDER = ['face_closeup', 'head_shoulders', 'upper_body', 'full_body', 'uncategorized'];
const QUICK_TAGS = [
  'looking_at_viewer', 'smile', 'standing', 'sitting', 'leaning',
  'from_above', 'from_below', 'from_side', 'from_behind',
  'outdoors', 'indoors', 'simple_background',
];
let PROJECTS = JSON.parse('{{PROJECTS_JSON}}');

// --- State ---
let allImages = [];
let filteredImages = [];
let selected = new Set();
let activeTab = 'uncategorized';
let modalIdx = -1;
let isDirty = false;
let originalCaption = '';
let currentStats = null;
let activeTaskId = null;

// --- Project switcher ---
function initProjectSwitcher() {
  const sel = document.getElementById('projectSwitcher');
  sel.innerHTML = '';
  PROJECTS.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.name;
    opt.textContent = p.name + ' (' + p.model + ')';
    opt.selected = p.active;
    sel.appendChild(opt);
  });
}

async function switchProject(name) {
  try {
    const resp = await fetch('/api/switch-project', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name})
    });
    const data = await resp.json();
    if (data.switched) {
      document.getElementById('headerMeta').textContent = data.dataset_dir + ' | ' + data.model;
      selected.clear();
      configDirty = false;
      PROJECTS.forEach(p => p.active = p.name === name);
      showToast('Switched to ' + name);
      loadImages();
      if (activeTab === 'config') loadConfig();
      else if (activeTab === 'training') loadTraining();
      else if (activeTab === 'stats') loadImages().then(() => renderStats());
    } else {
      showToast('Switch failed: ' + (data.error || 'unknown'), true);
    }
  } catch (e) { showToast('Switch failed: ' + e.message, true); }
}

// --- Category detection (mirrors Python) ---
function categorizeCaption(text) {
  const lower = text.toLowerCase();
  const checkOrder = ['face_closeup', 'upper_body', 'full_body', 'head_shoulders'];
  for (const cat of checkOrder) {
    for (const tag of CATEGORY_TAGS[cat]) {
      if (lower.includes(tag)) return cat;
    }
  }
  return 'uncategorized';
}

// --- Tab management ---
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));

  const isImageTab = !['stats', 'config', 'training', 'dashboard'].includes(tab);
  document.getElementById('gallery').style.display = isImageTab ? '' : 'none';
  document.getElementById('toolbar').style.display = isImageTab ? '' : 'none';
  document.getElementById('statsView').style.display = tab === 'stats' ? '' : 'none';
  document.getElementById('configView').style.display = tab === 'config' ? '' : 'none';
  document.getElementById('trainingView').style.display = tab === 'training' ? '' : 'none';
  document.getElementById('dashboardView').style.display = tab === 'dashboard' ? '' : 'none';

  if (tab === 'dashboard') loadDashboard();
  else if (tab === 'stats') renderStats();
  else if (tab === 'config') loadConfig();
  else if (tab === 'training') loadTraining();
  else renderGrid();
}

function updateBadges() {
  const counts = {};
  let total = 0;
  for (const cat of CATEGORY_ORDER) counts[cat] = 0;
  for (const img of allImages) {
    counts[img.category] = (counts[img.category] || 0) + 1;
    total++;
  }
  for (const cat of CATEGORY_ORDER) {
    const badge = document.getElementById('badge-' + cat);
    if (badge) badge.textContent = counts[cat] || 0;
  }
  document.getElementById('badge-all').textContent = total;
  document.getElementById('totalCount').textContent = total + ' images';
}

// --- Data loading ---
async function loadImages() {
  try {
    const resp = await fetch('/api/images');
    const data = await resp.json();
    allImages = data.images;
    currentStats = data.stats;
    updateBadges();
    if (activeTab === 'stats') renderStats();
    else if (activeTab !== 'config') renderGrid();
  } catch (e) {
    showToast('Failed to load images: ' + e.message, true);
  }
}

// --- Grid rendering ---
function renderGrid() {
  if (activeTab === 'all') {
    filteredImages = [...allImages];
  } else {
    filteredImages = allImages.filter(img => img.category === activeTab);
  }

  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  if (filteredImages.length === 0) {
    grid.innerHTML = '<div class="empty"><p>No images in this category</p></div>';
    return;
  }

  for (let i = 0; i < filteredImages.length; i++) {
    const img = filteredImages[i];
    const card = document.createElement('div');
    card.className = 'card' + (selected.has(img.filename) ? ' selected' : '');
    card.dataset.idx = i;
    card.dataset.filename = img.filename;
    card.onclick = (e) => { if (e.shiftKey) openModal(i); else toggleSelect(img.filename); };
    card.ondblclick = () => openModal(i);

    const imgEl = document.createElement('img');
    imgEl.loading = 'lazy';
    imgEl.src = '/img/' + encodeURIComponent(img.filename);

    const check = document.createElement('div');
    check.className = 'check';

    const dot = document.createElement('div');
    dot.className = 'caption-dot ' + (img.has_caption ? 'has' : 'none');

    if (activeTab === 'all' && img.category !== 'uncategorized') {
      const catLabel = document.createElement('div');
      catLabel.className = 'cat-label';
      catLabel.textContent = img.category.replace('_', ' ');
      card.appendChild(catLabel);
    }

    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = img.filename;

    card.appendChild(imgEl);
    card.appendChild(check);
    card.appendChild(dot);
    card.appendChild(name);
    grid.appendChild(card);
  }
  updateToolbar();
}

// --- Selection ---
function toggleSelect(filename) {
  if (selected.has(filename)) selected.delete(filename); else selected.add(filename);
  updateCardSelection(filename);
  updateToolbar();
}

function selectAllVisible() {
  filteredImages.forEach(img => selected.add(img.filename));
  updateAllCardSelections();
  updateToolbar();
}

function selectNone() {
  selected.clear();
  updateAllCardSelections();
  updateToolbar();
}

function invertVisible() {
  filteredImages.forEach(img => {
    if (selected.has(img.filename)) selected.delete(img.filename);
    else selected.add(img.filename);
  });
  updateAllCardSelections();
  updateToolbar();
}

function updateCardSelection(filename) {
  const card = document.querySelector(`.card[data-filename="${CSS.escape(filename)}"]`);
  if (card) card.classList.toggle('selected', selected.has(filename));
}

function updateAllCardSelections() {
  document.querySelectorAll('.card').forEach(card => {
    card.classList.toggle('selected', selected.has(card.dataset.filename));
  });
}

function updateToolbar() {
  const btn = document.getElementById('deleteBtn');
  btn.disabled = selected.size === 0;
  btn.textContent = `Delete (${selected.size})`;
}

// --- Delete ---
async function deleteSelected() {
  if (selected.size === 0) return;
  if (!confirm(`Delete ${selected.size} image(s)? Cannot undo.`)) return;
  const files = Array.from(selected);
  try {
    const resp = await fetch('/api/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({files})
    });
    const result = await resp.json();
    showToast(`Deleted ${result.deleted.length} images`);
    selected.clear();
    loadImages();
  } catch (e) { showToast('Delete failed: ' + e.message, true); }
}

// --- Modal ---
async function openModal(idx) {
  if (isDirty && !confirm('Unsaved caption changes. Discard?')) return;
  isDirty = false;

  modalIdx = idx;
  const img = filteredImages[idx];
  if (!img) return;

  document.getElementById('modalImg').src = '/img/' + encodeURIComponent(img.filename);
  document.getElementById('modalInfo').textContent = img.filename + ' (' + (idx + 1) + '/' + filteredImages.length + ')';
  document.getElementById('modal').classList.add('active');

  try {
    const resp = await fetch('/api/caption/' + encodeURIComponent(img.filename));
    const data = await resp.json();
    document.getElementById('captionText').value = data.caption || '';
    originalCaption = data.caption || '';
  } catch {
    document.getElementById('captionText').value = '';
    originalCaption = '';
  }

  updateModalCategoryButtons();
  updateModalTagButtons();
  document.getElementById('saveBtn').className = 'btn-save';
  document.getElementById('saveBtn').textContent = 'Save (Ctrl+S)';
  document.getElementById('captionText').focus();
}

function closeModal() {
  if (isDirty && !confirm('Unsaved caption changes. Discard?')) return;
  isDirty = false;
  document.getElementById('modal').classList.remove('active');
}

function navModal(dir) {
  if (isDirty && !confirm('Unsaved caption changes. Discard?')) return;
  isDirty = false;
  const len = filteredImages.length;
  if (len === 0) return;
  modalIdx = (modalIdx + dir + len) % len;
  openModal(modalIdx);
}

function onCaptionInput() {
  const current = document.getElementById('captionText').value;
  isDirty = current !== originalCaption;
  document.getElementById('saveBtn').className = isDirty ? 'btn-save dirty' : 'btn-save';
  document.getElementById('saveBtn').textContent = isDirty ? 'Save * (Ctrl+S)' : 'Save (Ctrl+S)';
  updateModalCategoryButtons();
  updateModalTagButtons();
}

// --- Category buttons ---
function updateModalCategoryButtons() {
  const caption = document.getElementById('captionText').value;
  const currentCat = categorizeCaption(caption);
  document.querySelectorAll('.cat-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.cat === currentCat);
  });
}

async function setCategory(cat) {
  const img = filteredImages[modalIdx];
  if (!img) return;

  try {
    const resp = await fetch('/api/set-category', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: img.filename, category: cat})
    });
    const result = await resp.json();
    if (result.saved) {
      document.getElementById('captionText').value = result.caption;
      originalCaption = result.caption;
      isDirty = false;
      document.getElementById('saveBtn').className = 'btn-save';
      document.getElementById('saveBtn').textContent = 'Save (Ctrl+S)';

      img.caption = result.caption;
      img.category = result.category;
      img.has_caption = true;
      const allImg = allImages.find(i => i.filename === img.filename);
      if (allImg) { allImg.caption = result.caption; allImg.category = result.category; allImg.has_caption = true; }

      updateBadges();
      updateModalCategoryButtons();
      updateModalTagButtons();
      showToast('Category: ' + cat.replace('_', ' '));
    }
  } catch (e) { showToast('Failed: ' + e.message, true); }
}

// --- Quick tag buttons ---
function updateModalTagButtons() {
  const caption = document.getElementById('captionText').value.toLowerCase();
  const container = document.getElementById('quickTags');
  container.innerHTML = '<label>Tags</label>';

  for (const tag of QUICK_TAGS) {
    const btn = document.createElement('button');
    btn.className = 'tag-btn' + (caption.includes(tag.toLowerCase()) ? ' active' : '');
    btn.textContent = tag;
    btn.onclick = () => toggleTag(tag);
    container.appendChild(btn);
  }
}

function toggleTag(tag) {
  const textarea = document.getElementById('captionText');
  let text = textarea.value;
  const tags = text.split(',').map(t => t.trim()).filter(t => t);

  const idx = tags.findIndex(t => t.toLowerCase() === tag.toLowerCase());
  if (idx >= 0) {
    tags.splice(idx, 1);
  } else {
    tags.push(tag);
  }

  textarea.value = tags.join(', ');
  onCaptionInput();
  textarea.focus();
}

// --- Save caption ---
async function saveCaption() {
  const caption = document.getElementById('captionText').value;
  const img = filteredImages[modalIdx];
  if (!img) return;

  try {
    const resp = await fetch('/api/caption/' + encodeURIComponent(img.filename), {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({caption})
    });
    const result = await resp.json();
    originalCaption = caption;
    isDirty = false;
    document.getElementById('saveBtn').className = 'btn-save';
    document.getElementById('saveBtn').textContent = 'Save (Ctrl+S)';

    const newCat = result.category || categorizeCaption(caption);
    img.caption = caption;
    img.category = newCat;
    img.has_caption = true;
    const allImg = allImages.find(i => i.filename === img.filename);
    if (allImg) { allImg.caption = caption; allImg.category = newCat; allImg.has_caption = true; }

    updateBadges();
    updateModalCategoryButtons();
    showToast('Saved');
  } catch (e) { showToast('Save failed: ' + e.message, true); }
}

// --- Delete in modal ---
async function deleteCurrentInModal() {
  const img = filteredImages[modalIdx];
  if (!img) return;
  if (!confirm('Delete this image? Cannot undo.')) return;

  try {
    await fetch('/api/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({files: [img.filename]})
    });
    isDirty = false;
    showToast('Deleted');

    allImages = allImages.filter(i => i.filename !== img.filename);
    filteredImages.splice(modalIdx, 1);
    selected.delete(img.filename);

    if (filteredImages.length === 0) { closeModal(); renderGrid(); updateBadges(); return; }
    if (modalIdx >= filteredImages.length) modalIdx = filteredImages.length - 1;
    openModal(modalIdx);
    updateBadges();
  } catch (e) { showToast('Delete failed', true); }
}

// --- WD14 Tagger ---
async function runTagger() {
  if (activeTaskId) { showToast('A task is already running', true); return; }
  if (allImages.length === 0) { showToast('No images to caption'); return; }
  if (!confirm(`Run auto-caption on ${allImages.length} image(s)? Existing tags will be preserved.`)) return;

  const btn = document.getElementById('tagBtn');
  btn.disabled = true;
  btn.textContent = 'Captioning...';

  try {
    const resp = await fetch('/api/tag', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
    const data = await resp.json();
    activeTaskId = data.task_id;
    pollTask(activeTaskId);
  } catch (e) {
    showToast('Failed to start tagger: ' + e.message, true);
    btn.disabled = false;
    btn.textContent = 'Auto Caption';
  }
}

function pollTask(taskId) {
  const bar = document.getElementById('taskBar');
  bar.className = 'task-bar active';

  const poll = async () => {
    try {
      const resp = await fetch('/api/tasks/' + taskId);
      const task = await resp.json();

      document.getElementById('taskLabel').textContent = task.name || 'Task';
      document.getElementById('taskStatus').textContent = task.message || '';

      const match = (task.progress || '').match(new RegExp('^([0-9]+)/([0-9]+)$'));
      if (match) {
        const pct = Math.round((parseInt(match[1]) / parseInt(match[2])) * 100);
        document.getElementById('taskFill').style.width = pct + '%';
      }

      if (task.status === 'running') {
        setTimeout(poll, 2000);
      } else {
        bar.className = 'task-bar active ' + (task.status === 'complete' ? 'complete' : 'failed');
        document.getElementById('taskFill').style.width = '100%';

        if (task.error) {
          document.getElementById('taskStatus').textContent = 'Error: ' + task.error;
          showToast('Auto-caption failed: ' + task.error, true);
        } else {
          showToast(task.message || 'Done');
        }

        activeTaskId = null;
        document.getElementById('tagBtn').disabled = false;
        document.getElementById('tagBtn').textContent = 'Auto Caption';
        loadImages();
        setTimeout(() => { bar.className = 'task-bar'; }, 4000);
      }
    } catch (e) {
      activeTaskId = null;
      bar.className = 'task-bar';
      document.getElementById('tagBtn').disabled = false;
      document.getElementById('tagBtn').textContent = 'Auto Caption';
    }
  };
  poll();
}

// --- Stats rendering ---
function renderStats() {
  if (!currentStats) return;
  const s = currentStats;
  const view = document.getElementById('statsView');

  let html = '<h2>Dataset Balance</h2>';
  html += '<table class="stats-table"><tr><th>Category</th><th>Count</th><th>Target</th><th>Current</th><th>Ideal</th><th></th><th>Status</th></tr>';

  const catLabels = {
    face_closeup: 'Face Closeup', head_shoulders: 'Head/Shoulders',
    upper_body: 'Upper Body', full_body: 'Full Body', uncategorized: 'Uncategorized'
  };
  const catDesc = {
    face_closeup: 'Face fills frame. Cropped at chin/forehead. Eyes, nose, lips, skin texture.',
    head_shoulders: 'Head to mid-chest. Classic portrait framing. Hair and neck visible.',
    upper_body: 'Waist up. Arms, hands, torso. Clothing and body proportions.',
    full_body: 'Head to feet. Full silhouette, proportions, legs, stance. Hardest for LoRAs \u2014 needs the most examples.',
    uncategorized: 'No framing tag detected. Assign a category before training.'
  };

  for (const cat of CATEGORY_ORDER) {
    const c = s.categories[cat];
    if (!c) continue;
    const pct = c.current_pct;
    const ideal = c.ideal_pct;
    let barClass = 'ok';
    if (cat === 'uncategorized' && c.count > 0) barClass = 'warn';
    else if (c.surplus > 0) barClass = 'over';
    else if (c.deficit > 0) barClass = 'low';

    const barWidth = Math.min(100, Math.round((c.count / Math.max(c.ideal_count, 1)) * 100));
    let status = '';
    if (cat === 'uncategorized' && c.count > 0) status = 'tag these';
    else if (c.surplus > 0) status = '<span style="color:#3498db">over +' + c.surplus + '</span>';
    else if (c.deficit > 0) status = '<span style="color:#f39c12">need +' + c.deficit + '</span>';
    else status = '<span style="color:#2ecc71">ok</span>';

    html += `<tr>
      <td><strong>${catLabels[cat] || cat}</strong><br><span style="color:#888;font-size:0.78em">${catDesc[cat] || ''}</span></td>
      <td>${c.count}</td>
      <td>${c.ideal_count}</td>
      <td>${pct}%</td>
      <td>${ideal}%</td>
      <td><div class="stats-bar"><div class="stats-bar-fill ${barClass}" style="width:${barWidth}%"></div></div></td>
      <td>${status}</td>
    </tr>`;
  }
  html += '</table>';

  html += '<div class="stats-summary">';
  html += `<strong>Current images:</strong> ${s.total}`;
  if (s.oversized) {
    html += ` <span style="color:#e74c3c">(over ${s.max_recommended} recommended max — risk of overfitting)</span>`;
  }
  html += '<br>';
  html += `<strong>Recommended:</strong> 30\u201350 images for character LoRAs<br>`;
  const curReps = s.current_repeats || '?';
  const curSteps = s.current_repeats ? s.total * s.current_repeats : '?';
  const sugSteps = s.total * s.suggested_repeats;
  html += `<strong>NUM_REPEATS in project.conf:</strong> ${curReps} (~${curSteps} steps/epoch)`;
  if (s.current_repeats && s.current_repeats !== s.suggested_repeats) {
    html += ` \u2014 <span style="color:#f39c12">suggested: ${s.suggested_repeats} (~${sugSteps} steps/epoch, aim for 200\u2013400)</span>`;
    html += ` <button class="btn-select" style="padding:3px 10px;font-size:0.8em;" onclick="applyRepeats(${s.suggested_repeats})">Apply</button>`;
  }
  html += '<br>';
  if (s.full_body_total > 0) {
    const fbPct = Math.round(s.full_body_facing / s.full_body_total * 100);
    html += `<strong>Full body facing camera:</strong> ${s.full_body_facing}/${s.full_body_total} (${fbPct}%)`;
    const target = Math.round(s.full_body_total * 0.7);
    if (s.full_body_facing < target) {
      html += ` \u2014 need ~${target - s.full_body_facing} more front-facing`;
    }
    html += '<br>';
  }
  html += '</div>';

  view.innerHTML = html;
}

async function applyRepeats(value) {
  try {
    const resp = await fetch('/api/config/update', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key: 'NUM_REPEATS', value: String(value)})
    });
    const data = await resp.json();
    if (data.saved) {
      showToast('NUM_REPEATS updated to ' + value);
      loadImages();
    } else showToast(data.error || 'Failed', true);
  } catch (e) { showToast('Failed: ' + e.message, true); }
}

// --- Config tab ---
let configOriginal = '';
let configDirty = false;

async function loadConfig() {
  try {
    const resp = await fetch('/api/config');
    const data = await resp.json();
    document.getElementById('configText').value = data.raw;
    document.getElementById('configPath').textContent = data.path;
    configOriginal = data.raw;
    configDirty = false;
    updateConfigUI();
  } catch (e) { showToast('Failed to load config: ' + e.message, true); }
}

function onConfigInput() {
  configDirty = document.getElementById('configText').value !== configOriginal;
  updateConfigUI();
}

function updateConfigUI() {
  const btn = document.getElementById('configSaveBtn');
  btn.className = configDirty ? 'btn-save dirty' : 'btn-save';
  btn.textContent = configDirty ? 'Save * (Ctrl+S)' : 'Save (Ctrl+S)';
  document.getElementById('configStatus').textContent = configDirty ? 'Unsaved changes' : '';
}

async function saveConfig() {
  const raw = document.getElementById('configText').value;
  try {
    const resp = await fetch('/api/config', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({raw})
    });
    const result = await resp.json();
    if (result.saved) {
      configOriginal = raw;
      configDirty = false;
      updateConfigUI();
      document.getElementById('configStatus').textContent = 'Saved and reloaded';
      showToast('Config saved');
      loadImages();
    } else {
      showToast('Save failed: ' + (result.error || 'unknown'), true);
    }
  } catch (e) { showToast('Save failed: ' + e.message, true); }
}

async function runSetup() {
  try {
    const resp = await fetch('/api/setup', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
    const result = await resp.json();
    if (result.created && result.created.length > 0) {
      showToast('Created: ' + result.created.join(', '));
    } else {
      showToast(result.message);
    }
    loadImages();
  } catch (e) { showToast('Setup failed: ' + e.message, true); }
}

// Attach input listener for config textarea
document.getElementById('configText').addEventListener('input', onConfigInput);

// --- Keyboard shortcuts ---
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 's') {
    e.preventDefault();
    const modal = document.getElementById('modal');
    if (modal.classList.contains('active')) saveCaption();
    else if (activeTab === 'config') saveConfig();
    return;
  }

  const modal = document.getElementById('modal');
  if (!modal.classList.contains('active')) return;

  const inTextarea = document.activeElement === document.getElementById('captionText');

  if (e.key === 'Escape') { closeModal(); e.preventDefault(); }

  if (!inTextarea) {
    if (e.key === 'ArrowLeft') navModal(-1);
    if (e.key === 'ArrowRight') navModal(1);
    if (e.key === 'd' || e.key === 'Delete') { toggleSelect(filteredImages[modalIdx]?.filename); navModal(1); }
  }
});

// --- Toast ---
function showToast(msg, isError) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => toast.className = 'toast', 3000);
}

// --- Dashboard tab ---
async function loadDashboard() {
  try {
    const resp = await fetch('/api/dashboard');
    const d = await resp.json();
    renderDashActive(d.active_training);
    renderDashProjects(d.projects);
    renderDashRuns(d.runs);
    renderDashLoras(d.loras);
  } catch (e) { showToast('Failed to load dashboard', true); }
}

function renderDashActive(active) {
  const el = document.getElementById('dashActiveTraining');
  if (!active) {
    el.innerHTML = '';
    return;
  }
  const t = active.task || {};
  const tr = t.training || {};
  const pct = tr.total_steps > 0 ? Math.round((tr.step / tr.total_steps) * 100) : 0;
  el.innerHTML = `<div class="dash-active">
    <div class="status">Training in progress: ${active.project} (${active.model})</div>
    <div style="color:#aaa;font-size:0.9em;margin-top:5px;">
      Epoch ${tr.epoch || 0}/${tr.total_epochs || '?'} \u2014 Step ${tr.step || 0}/${tr.total_steps || '?'} (${pct}%)
      ${tr.avg_loss != null ? ' \u2014 Loss: ' + tr.avg_loss.toFixed(4) : ''}
      ${tr.eta ? ' \u2014 ETA: ' + tr.eta : ''}
    </div>
    <div style="margin-top:8px;">
      <div class="train-progress-bar" style="height:8px;"><div class="train-progress-fill" style="width:${pct}%"></div></div>
    </div>
  </div>`;
}

function renderDashProjects(projects) {
  const el = document.getElementById('dashProjects');
  if (!projects || projects.length === 0) {
    el.innerHTML = '<div style="color:#888;">No projects found</div>';
    return;
  }
  el.innerHTML = '<div class="dash-projects">' + projects.map(p =>
    `<div class="dash-project" onclick="document.getElementById('projectSwitcher').value='${p.name}';switchProject('${p.name}');switchTab('uncategorized');">
      <div class="name">${p.name}</div>
      <div class="meta">
        Model: ${p.model}<br>
        Trigger: ${p.trigger}<br>
        Images: ${p.images}<br>
        Runs: ${p.runs} | LoRAs: ${p.loras}
      </div>
    </div>`
  ).join('') + '</div>';
}

function renderDashRuns(runs) {
  const el = document.getElementById('dashRuns');
  if (!runs || runs.length === 0) {
    el.innerHTML = '<div style="color:#888;font-size:0.85em;">No training runs yet</div>';
    return;
  }
  el.innerHTML = runs.map(r => {
    const date = r.started_at ? r.started_at.split('T')[0] + ' ' + (r.started_at.split('T')[1] || '').slice(0,5) : '?';
    const loss = r.final_loss != null ? r.final_loss.toFixed(4) : '?';
    return `<div class="run-row" onclick="this.querySelector('.run-detail').classList.toggle('open')">
      <div class="run-header">
        <span style="color:#f39c12;font-weight:600;min-width:160px;">${r.project || '?'}</span>
        <span class="run-date">${date}</span>
        <span class="run-model">${r.model_type || '?'}</span>
        <span>Epochs: ${r.total_epochs || '?'}</span>
        <span class="run-loss">Loss: ${loss}</span>
        <span class="run-status ${r.status || ''}">${r.status || '?'}</span>
      </div>
      <div class="run-detail">
        <div style="color:#888;font-size:0.82em;padding:5px 0;">
          Steps: ${r.total_steps || '?'} |
          Checkpoints: ${(r.checkpoints || []).length} |
          Run ID: ${r.run_id || '?'}
        </div>
      </div>
    </div>`;
  }).join('');
}

function renderDashLoras(loras) {
  const el = document.getElementById('dashLoras');
  if (!loras || loras.length === 0) {
    el.innerHTML = '<div style="color:#888;font-size:0.85em;">No LoRA files yet</div>';
    return;
  }
  el.innerHTML = loras.map(f =>
    `<div class="lora-file">
      <span style="color:#f39c12;font-weight:600;min-width:140px;font-size:0.8em;">${f.project}</span>
      <span class="name">${f.filename}</span>
      <span class="size">${f.size_mb} MB</span>
      <span class="date">${f.modified.split('T')[0]}</span>
      ${f.has_json ? '<span style="color:#2ecc71;font-size:0.75em;">JSON</span>' : ''}
      ${f.has_preview ? '<span style="color:#2ecc71;font-size:0.75em;">Preview</span>' : ''}
    </div>`
  ).join('');
}

// --- Training tab ---
let trainingTaskId = null;
let trainingPollTimer = null;

async function loadTraining() {
  // Check for active training
  try {
    const resp = await fetch('/api/training/active');
    const data = await resp.json();
    if (data.active) {
      trainingTaskId = data.task_id;
      document.getElementById('trainLaunch').style.display = 'none';
      document.getElementById('trainMonitor').style.display = '';
      pollTraining();
    } else {
      trainingTaskId = null;
      document.getElementById('trainLaunch').style.display = '';
      document.getElementById('trainMonitor').style.display = 'none';
      renderTrainSummary();
    }
  } catch (e) {
    renderTrainSummary();
  }
  loadLoraFiles();
  loadRunHistory();
  loadTrainingSamples();
}

function renderTrainSummary() {
  const total = allImages.length;
  const el = document.getElementById('trainSummary');
  el.innerHTML = `<strong>${total}</strong> images in dataset<br>` +
    `Model: <strong>${document.getElementById('projectSwitcher').value}</strong>`;
  if (total === 0) {
    document.getElementById('trainStartBtn').disabled = true;
    el.innerHTML += '<br><span style="color:#e74c3c">Add images before training</span>';
  } else {
    document.getElementById('trainStartBtn').disabled = false;
  }
}

async function startTraining() {
  if (!confirm('Start training? This will use the GPU for ~1-3 hours.')) return;
  try {
    const resp = await fetch('/api/train', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
    const data = await resp.json();
    if (data.error) {
      showToast(data.error, true);
      return;
    }
    trainingTaskId = data.task_id;
    document.getElementById('trainLaunch').style.display = 'none';
    document.getElementById('trainMonitor').style.display = '';
    showToast('Training started');
    pollTraining();
  } catch (e) { showToast('Failed to start: ' + e.message, true); }
}

async function cancelTraining() {
  if (!confirm('Cancel training? Progress will be lost.')) return;
  try {
    await fetch('/api/train/cancel', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
    showToast('Training cancelled');
    trainingTaskId = null;
    if (trainingPollTimer) clearTimeout(trainingPollTimer);
    document.getElementById('trainLaunch').style.display = '';
    document.getElementById('trainMonitor').style.display = 'none';
  } catch (e) { showToast('Cancel failed', true); }
}

function pollTraining() {
  if (!trainingTaskId) return;
  const poll = async () => {
    try {
      const resp = await fetch('/api/tasks/' + trainingTaskId);
      const task = await resp.json();
      const t = task.training || {};

      // Update progress bar
      const pct = t.total_steps > 0 ? Math.round((t.step / t.total_steps) * 100) : 0;
      document.getElementById('trainFill').style.width = pct + '%';
      document.getElementById('trainProgressText').textContent =
        `Epoch ${t.epoch || 0}/${t.total_epochs || '?'} \u2014 Step ${t.step || 0}/${t.total_steps || '?'} (${pct}%)`;

      // Update info
      let info = task.message || '';
      if (t.eta) info += ` \u2014 ETA: ${t.eta}`;
      if (t.elapsed) info += ` (elapsed: ${t.elapsed})`;
      document.getElementById('trainInfo').textContent = info;

      // Update loss
      if (t.avg_loss != null) {
        document.getElementById('trainLoss').textContent = 'Loss: ' + t.avg_loss.toFixed(4);
      }

      // Render loss chart
      if (t.loss_history && t.loss_history.length > 1) {
        renderLossChart(t.loss_history);
      }

      if (task.status === 'running') {
        trainingPollTimer = setTimeout(poll, 2000);
      } else {
        // Training finished
        trainingTaskId = null;
        document.getElementById('trainLaunch').style.display = '';
        document.getElementById('trainMonitor').style.display = 'none';
        if (task.status === 'complete') {
          showToast(task.message || 'Training complete');
          document.getElementById('trainFill').style.background = '#2ecc71';
        } else {
          showToast('Training failed: ' + (task.error || 'unknown'), true);
        }
        loadLoraFiles();
        loadRunHistory();
        loadTrainingSamples();
      }
    } catch (e) {
      trainingPollTimer = setTimeout(poll, 5000);
    }
  };
  poll();
}

function renderLossChart(history) {
  const svg = document.getElementById('lossChart');
  const W = 600, H = 200, PAD = 45, RPAD = 10, TPAD = 10, BPAD = 25;
  const plotW = W - PAD - RPAD, plotH = H - TPAD - BPAD;

  const losses = history.map(p => p.loss);
  const maxStep = history[history.length - 1].step;
  const minLoss = Math.min(...losses) * 0.95;
  const maxLoss = Math.max(...losses) * 1.05;
  const lossRange = maxLoss - minLoss || 0.01;

  const pts = history.map(p => {
    const x = PAD + (p.step / maxStep) * plotW;
    const y = TPAD + (1 - (p.loss - minLoss) / lossRange) * plotH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  let html = '';
  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const y = TPAD + (i / 4) * plotH;
    const val = maxLoss - (i / 4) * lossRange;
    html += `<line x1="${PAD}" y1="${y}" x2="${W-RPAD}" y2="${y}" stroke="#1a1a2e" stroke-width="1"/>`;
    html += `<text x="${PAD-5}" y="${y+4}" fill="#888" font-size="10" text-anchor="end">${val.toFixed(3)}</text>`;
  }
  // X axis labels
  for (let i = 0; i <= 4; i++) {
    const x = PAD + (i / 4) * plotW;
    const step = Math.round((i / 4) * maxStep);
    html += `<text x="${x}" y="${H-5}" fill="#888" font-size="10" text-anchor="middle">${step}</text>`;
  }
  // Loss line
  html += `<polyline points="${pts}" fill="none" stroke="#9b59b6" stroke-width="2"/>`;
  // Latest point
  if (history.length > 0) {
    const last = history[history.length - 1];
    const lx = PAD + (last.step / maxStep) * plotW;
    const ly = TPAD + (1 - (last.loss - minLoss) / lossRange) * plotH;
    html += `<circle cx="${lx}" cy="${ly}" r="4" fill="#f39c12"/>`;
  }
  svg.innerHTML = html;
}

async function loadLoraFiles() {
  try {
    const resp = await fetch('/api/training/loras');
    const data = await resp.json();
    const el = document.getElementById('loraFiles');
    if (!data.files || data.files.length === 0) {
      el.innerHTML = '<div style="color:#888;font-size:0.85em;">No LoRA files yet</div>';
      return;
    }
    el.innerHTML = data.files.map(f => {
      const esc = f.filename.replace(/'/g, "\\'");
      const previewUrl = f.has_preview ? `/api/training/loras/preview/${encodeURIComponent(f.filename)}` : '';
      return `<div class="lora-file" style="flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:12px;width:100%;">
          ${previewUrl ? `<img src="${previewUrl}" style="height:50px;border-radius:4px;">` : ''}
          <span class="name">${f.filename}</span>
          <span class="size">${f.size_mb} MB</span>
          <span class="date">${f.modified.split('T')[0]}</span>
          ${f.has_json ? '<span style="color:#2ecc71;font-size:0.75em;">JSON</span>' : ''}
          ${f.has_preview ? '<span style="color:#2ecc71;font-size:0.75em;">Preview</span>' : ''}
        </div>
        <div style="display:flex;gap:10px;padding:4px 0 0 0;font-size:0.85em;">
          <a href="/api/training/loras/bundle/${encodeURIComponent(f.filename)}" download>Bundle (.zip)</a>
          <a href="/api/training/loras/download/${encodeURIComponent(f.filename)}" download>.safetensors</a>
          <a href="#" onclick="editLoraMetadata('${esc}'); return false;" style="color:#9b59b6;">Metadata</a>
          <a href="#" onclick="pickLoraPreview('${esc}'); return false;" style="color:#f39c12;">Set Preview</a>
          <a href="#" onclick="renameLora('${esc}'); return false;" style="color:#f39c12;">Rename</a>
          <a href="#" onclick="deleteLora('${esc}'); return false;" style="color:#e74c3c;">Delete</a>
        </div>
      </div>`;
    }).join('');
  } catch (e) {}
}

async function deleteLora(filename) {
  if (!confirm('Delete ' + filename + '? Cannot undo.')) return;
  try {
    const resp = await fetch('/api/training/loras/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename})
    });
    const data = await resp.json();
    if (data.deleted) { showToast('Deleted ' + filename); loadLoraFiles(); }
    else showToast(data.error || 'Delete failed', true);
  } catch (e) { showToast('Delete failed', true); }
}

async function renameLora(oldName) {
  const newName = prompt('New filename:', oldName);
  if (!newName || newName === oldName) return;
  try {
    const resp = await fetch('/api/training/loras/rename', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({old_name: oldName, new_name: newName})
    });
    const data = await resp.json();
    if (data.renamed) { showToast('Renamed to ' + data.new_name); loadLoraFiles(); }
    else showToast(data.error || 'Rename failed', true);
  } catch (e) { showToast('Rename failed', true); }
}

let _metaDialogFilename = '';

async function editLoraMetadata(filename) {
  _metaDialogFilename = filename;
  try {
    const resp = await fetch('/api/training/loras/metadata/' + encodeURIComponent(filename));
    const meta = await resp.json();
    const fields = ['description', 'sd version', 'activation text', 'preferred weight', 'notes'];
    const dlg = document.createElement('div');
    dlg.id = 'metaDialog';
    dlg.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);z-index:500;display:flex;align-items:center;justify-content:center;';
    const box = document.createElement('div');
    box.style.cssText = 'background:#16213e;padding:25px;border-radius:10px;width:500px;max-width:90%;';
    box.innerHTML = '<h3 style="color:#f39c12;margin-bottom:15px;">LoRA Metadata: ' + filename + '</h3>';
    fields.forEach(f => {
      const val = meta[f] != null ? meta[f] : '';
      const id = 'meta_' + f.replace(/ /g, '_');
      const lbl = document.createElement('label');
      lbl.style.cssText = 'color:#aaa;font-size:0.8em;display:block;margin:8px 0 3px;';
      lbl.textContent = f;
      box.appendChild(lbl);
      if (f === 'notes') {
        const ta = document.createElement('textarea');
        ta.id = id;
        ta.style.cssText = 'width:100%;height:60px;background:#0f3460;color:#eee;border:1px solid #333;border-radius:4px;padding:6px;font-size:0.85em;';
        ta.value = val;
        box.appendChild(ta);
      } else {
        const inp = document.createElement('input');
        inp.id = id;
        inp.value = String(val);
        inp.style.cssText = 'width:100%;background:#0f3460;color:#eee;border:1px solid #333;border-radius:4px;padding:6px;font-size:0.85em;';
        box.appendChild(inp);
      }
    });
    const btns = document.createElement('div');
    btns.style.cssText = 'margin-top:15px;display:flex;gap:10px;';
    const saveBtn = document.createElement('button');
    saveBtn.className = 'btn-save';
    saveBtn.textContent = 'Save';
    saveBtn.onclick = () => saveMetadataDialog();
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn-refresh';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.onclick = () => dlg.remove();
    btns.appendChild(saveBtn);
    btns.appendChild(cancelBtn);
    box.appendChild(btns);
    dlg.appendChild(box);
    document.body.appendChild(dlg);
  } catch (e) { showToast('Failed to load metadata', true); }
}

async function saveMetadataDialog() {
  const filename = _metaDialogFilename;
  const metadata = {};
  const fields = ['description', 'sd version', 'activation text', 'preferred weight', 'notes'];
  fields.forEach(f => {
    const id = 'meta_' + f.replace(/ /g, '_');
    let val = document.getElementById(id).value;
    if (f === 'preferred weight') val = parseFloat(val) || 0.75;
    metadata[f] = val;
  });
  try {
    const resp = await fetch('/api/training/loras/metadata', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename, metadata})
    });
    const data = await resp.json();
    if (data.saved) {
      showToast('Metadata saved');
      document.getElementById('metaDialog').remove();
      loadLoraFiles();
    } else showToast(data.error || 'Save failed', true);
  } catch (e) { showToast('Save failed', true); }
}

function pickLoraPreview(loraFilename) {
  const overlay = document.createElement('div');
  overlay.id = 'previewPicker';
  overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);z-index:500;display:flex;flex-direction:column;align-items:center;padding:30px;overflow:auto;';
  const title = document.createElement('h3');
  title.style.cssText = 'color:#f39c12;margin-bottom:15px;';
  title.textContent = 'Select preview image for ' + loraFilename;
  overlay.appendChild(title);
  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn-refresh';
  closeBtn.textContent = 'Close';
  closeBtn.style.cssText = 'position:fixed;top:15px;right:20px;z-index:501;';
  closeBtn.onclick = () => overlay.remove();
  overlay.appendChild(closeBtn);
  const grid = document.createElement('div');
  grid.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;justify-content:center;';
  allImages.forEach(img => {
    const el = document.createElement('img');
    el.src = '/img/' + encodeURIComponent(img.filename);
    el.style.cssText = 'height:150px;border-radius:6px;cursor:pointer;border:3px solid transparent;';
    el.loading = 'lazy';
    el.onmouseover = () => el.style.borderColor = '#f39c12';
    el.onmouseout = () => el.style.borderColor = 'transparent';
    el.onclick = () => setLoraPreview(loraFilename, img.filename);
    grid.appendChild(el);
  });
  overlay.appendChild(grid);
  document.body.appendChild(overlay);
}

async function setLoraPreview(loraFilename, imageFilename) {
  try {
    const resp = await fetch('/api/training/loras/preview', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: loraFilename, image: imageFilename})
    });
    const data = await resp.json();
    if (data.saved) {
      showToast('Preview set');
      document.getElementById('previewPicker').remove();
      loadLoraFiles();
    } else showToast(data.error || 'Failed', true);
  } catch (e) { showToast('Failed', true); }
}

async function loadRunHistory() {
  try {
    const resp = await fetch('/api/training/runs');
    const data = await resp.json();
    const el = document.getElementById('runHistory');
    if (!data.runs || data.runs.length === 0) {
      el.innerHTML = '<div style="color:#888;font-size:0.85em;">No training runs yet</div>';
      return;
    }
    el.innerHTML = data.runs.slice().reverse().map((r, i) => {
      const date = r.started_at ? r.started_at.split('T')[0] + ' ' + (r.started_at.split('T')[1] || '').slice(0,5) : '?';
      const loss = r.final_loss != null ? r.final_loss.toFixed(4) : '?';
      return `<div class="run-row" onclick="this.querySelector('.run-detail').classList.toggle('open')">
        <div class="run-header">
          <span class="run-date">${date}</span>
          <span class="run-model">${r.model_type || '?'}</span>
          <span>Epochs: ${r.total_epochs || '?'}</span>
          <span class="run-loss">Loss: ${loss}</span>
          <span class="run-status ${r.status || ''}">${r.status || '?'}</span>
        </div>
        <div class="run-detail">
          <div style="color:#888;font-size:0.82em;padding:5px 0;">
            Steps: ${r.total_steps || '?'} |
            Checkpoints: ${(r.checkpoints || []).length} |
            Run ID: ${r.run_id || '?'}
          </div>
          ${(r.checkpoints || []).map(c =>
            `<div class="lora-file" style="margin:3px 0;">
              <span class="name">${c.filename}</span>
              <span class="size">${c.size_mb} MB</span>
              <a href="/api/training/loras/download/${encodeURIComponent(c.filename)}" download>Download</a>
            </div>`
          ).join('')}
        </div>
      </div>`;
    }).join('');
  } catch (e) {}
}

let selectedSamples = new Set();

async function loadTrainingSamples() {
  selectedSamples.clear();
  try {
    const resp = await fetch('/api/training/samples');
    const data = await resp.json();
    const el = document.getElementById('trainSamples');
    if (!data.samples || data.samples.length === 0) {
      el.innerHTML = '<div style="color:#888;font-size:0.85em;">No samples yet</div>';
      return;
    }
    el.innerHTML = '<div style="margin-bottom:8px;">' +
      '<button class="btn-delete" id="deleteSamplesBtn" disabled onclick="deleteSelectedSamples()" style="padding:5px 12px;font-size:0.8em;">Delete Selected (0)</button> ' +
      '<button class="btn-select" onclick="selectAllSamples()" style="padding:5px 12px;font-size:0.8em;">Select All</button> ' +
      '<button class="btn-select" onclick="selectedSamples.clear();loadTrainingSamples();" style="padding:5px 12px;font-size:0.8em;">Select None</button>' +
      '</div>' +
      '<div class="train-samples">' +
      data.samples.map(s =>
        `<div style="position:relative;display:inline-block;" onclick="toggleSampleSelect('${s.replace(/'/g,"\\'")}', this)">
          <img src="/api/training/sample/${encodeURIComponent(s)}" title="${s}" loading="lazy" style="border:3px solid transparent;">
        </div>`
      ).join('') + '</div>';
  } catch (e) {}
}

function toggleSampleSelect(filename, el) {
  if (selectedSamples.has(filename)) {
    selectedSamples.delete(filename);
    el.querySelector('img').style.borderColor = 'transparent';
  } else {
    selectedSamples.add(filename);
    el.querySelector('img').style.borderColor = '#e74c3c';
  }
  const btn = document.getElementById('deleteSamplesBtn');
  btn.disabled = selectedSamples.size === 0;
  btn.textContent = 'Delete Selected (' + selectedSamples.size + ')';
}

function selectAllSamples() {
  document.querySelectorAll('#trainSamples .train-samples div').forEach(el => {
    const img = el.querySelector('img');
    if (img) {
      const name = decodeURIComponent(img.src.split('/').pop());
      selectedSamples.add(name);
      img.style.borderColor = '#e74c3c';
    }
  });
  const btn = document.getElementById('deleteSamplesBtn');
  btn.disabled = selectedSamples.size === 0;
  btn.textContent = 'Delete Selected (' + selectedSamples.size + ')';
}

async function deleteSelectedSamples() {
  if (selectedSamples.size === 0) return;
  if (!confirm('Delete ' + selectedSamples.size + ' sample(s)? Cannot undo.')) return;
  try {
    const resp = await fetch('/api/training/samples/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({files: Array.from(selectedSamples)})
    });
    const data = await resp.json();
    showToast('Deleted ' + (data.deleted || []).length + ' samples');
    selectedSamples.clear();
    loadTrainingSamples();
  } catch (e) { showToast('Delete failed', true); }
}

// --- Init ---
initProjectSwitcher();
loadImages();
// Check for active training on page load
fetch('/api/training/active').then(r => r.json()).then(data => {
  if (data.active) {
    trainingTaskId = data.task_id;
  }
}).catch(() => {});
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dataset preparation server for LoRA training")
    parser.add_argument("--loras-dir", default=None,
                        help="Parent directory to scan for projects (default: parent of this script)")
    parser.add_argument("--port", type=int, default=8899, help="Port (default: 8899)")
    args = parser.parse_args()

    loras_dir = args.loras_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # If script is at ~/projects/loras/lora-dataset-ui/server.py, loras_dir = ~/projects/loras/

    projects = discover_projects(loras_dir)
    if not projects:
        print(f"No projects found in {loras_dir}")
        print("  Looking for directories containing project.conf")
        sys.exit(1)

    state = ServerState(projects)
    handler = make_handler(state)
    server = HTTPServer(('0.0.0.0', args.port), handler)

    print(f"Dataset Prep Server")
    print(f"  Projects:  {', '.join(p['name'] for p in projects)}")
    print(f"  Active:    {state.current} ({state.model})")
    print(f"  Directory: {os.path.abspath(state.base_dir)}")
    print(f"  URL:       http://0.0.0.0:{args.port}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
