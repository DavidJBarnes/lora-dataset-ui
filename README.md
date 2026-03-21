# LoRA Dataset UI

Web-based dataset preparation tool for SDXL character LoRA training. Single-page app that replaces the manual CLI workflow of generating, tagging, reviewing, and balancing training datasets.

## Features

- **Category tabs** — Images sorted into Uncategorized, Face Closeup, Head/Shoulders, Upper Body, Full Body based on caption tags
- **WD14 auto-tagging** — One-click tagger integration (runs in background, preserves existing captions)
- **Caption editor** — Modal with quick-tag toggle buttons, category quick-set, Ctrl+S save
- **Tag deduplication** — Automatically removes duplicate tags on save
- **Caption cleanup** — Strips character-specific tags, adds model-appropriate prefix (pony/lustify)
- **Dataset stats** — Balance analysis with ideal ratios, deficit counts, and NUM_REPEATS recommendation
- **Config editor** — View/edit `project.conf` from the UI with live reload
- **Run Setup** — Create directory structure from the Config tab

## Quick Start

```bash
# Edit project.conf with your character settings
python server.py --model lustify
python server.py --model pony --port 9000
```

The server auto-creates the dataset directory if it doesn't exist. Open `http://localhost:8899` in a browser.

## Requirements

- Python 3.8+ (no pip dependencies — pure stdlib)
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) for WD14 tagging (optional, path set in `project.conf`)
- A1111 WebUI with FaceSwapLab for image generation (optional, separate workflow)

## Supported Base Models

| Model | Caption prefix | Clip skip |
|-------|---------------|-----------|
| CyberRealistic Pony (`--model pony`) | `score_9, score_8_up, score_7_up, source_realistic, {trigger} {class}` | 2 |
| Lustify-SDXL (`--model lustify`) | `{trigger} {class}` | 1 |

## Keyboard Shortcuts

| Key | Context | Action |
|-----|---------|--------|
| Shift+Click | Gallery | Open modal |
| Double-click | Gallery | Open modal |
| Arrow Left/Right | Modal | Navigate images |
| D / Delete | Modal | Mark for delete + next |
| Ctrl+S | Modal | Save caption |
| Ctrl+S | Config tab | Save project.conf |
| Escape | Modal | Close |

## Project Structure

```
project.conf          # Central config — trigger, paths, character desc
server.py             # Dataset prep SPA (this tool)
project_config.py     # Python config parser (shared by all scripts)
generate_dataset.py   # A1111 API batch generation + FaceSwapLab
tagger.sh             # WD14 tagger CLI wrapper
tagger_cleanup.sh     # Caption cleanup CLI (bash version)
train_character.sh    # Training launcher (generates TOML + sample prompts)
analyze_dataset.py    # Dataset balance CLI analysis
rename_pairs.py       # Sequential file pair renaming
setup.sh              # Directory structure bootstrap (CLI version)
```

## Dataset Workflow

1. Edit `project.conf` (or use Config tab)
2. Run Setup (Config tab or `bash setup.sh`)
3. Generate images (`python generate_dataset.py --model lustify`)
4. Open server — images appear in Uncategorized tab
5. Click **Tag Uncaptioned** — WD14 tags + cleanup, images sort into category tabs
6. Review captions, fix categories, delete bad images
7. Check **Stats** tab for balance — generate more if needed
8. Train (`bash train_character.sh lustify`)

## Target Dataset Balance

| Category | Target % |
|----------|---------|
| Face Closeup | 10% |
| Head/Shoulders | 10% |
| Upper Body | 15% |
| Full Body | 65% |
