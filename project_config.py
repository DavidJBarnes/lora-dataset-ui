"""
project_config.py — Shared config loader for all project Python scripts.

Reads project.conf and exposes values as a dict.
All Python scripts import this instead of hardcoding paths.

Usage:
    from project_config import conf
    print(conf["TRIGGER"])
    print(conf["DATASET_PATH"])  # Computed: DATASET_DIR/NUM_REPEATS_TRIGGER_CLASS
"""

import os
import sys

CONF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project.conf")


def load_conf(filepath=CONF_FILE):
    """Parse project.conf into a dict, expanding $HOME."""
    if not os.path.isfile(filepath):
        print(f"Error: {filepath} not found. Copy project.conf to project root.")
        sys.exit(1)

    conf = {}
    home = os.path.expanduser("~")

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            value = value.replace("$HOME", home)
            conf[key] = value

    # Computed paths
    trigger = conf.get("TRIGGER", "trigger")
    cls = conf.get("CLASS", "woman")
    repeats = conf.get("NUM_REPEATS", "10")
    dataset_dir = conf.get("DATASET_DIR", "dataset/img")

    conf["SUBSET_NAME"] = f"{repeats}_{trigger}_{cls}"
    conf["DATASET_PATH"] = os.path.join(dataset_dir, conf["SUBSET_NAME"])
    conf["DATASET_PATH_ABS"] = os.path.join(
        conf.get("PROJECT_DIR", "."), dataset_dir, conf["SUBSET_NAME"]
    )

    return conf


# Auto-load on import
conf = load_conf()
