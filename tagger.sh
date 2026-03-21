#!/bin/bash
# tagger.sh — Run WD14 auto-tagger on dataset images
# Generates .txt caption files alongside each image
#
# Usage:
#   bash tagger.sh              # Tag all images in dataset dir
#   bash tagger.sh --gpu        # Use GPU (faster, requires onnxruntime-gpu)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/project.conf"

SUBSET="${NUM_REPEATS}_${TRIGGER}_${CLASS}"
FULL_DATASET_PATH="${DATASET_DIR}/${SUBSET}"

# Default to CPU, --gpu flag switches to GPU batch size
BATCH_SIZE=1
if [[ "${1:-}" == "--gpu" ]]; then
  BATCH_SIZE=8
  echo "Mode: GPU (batch_size=$BATCH_SIZE)"
else
  echo "Mode: CPU (batch_size=$BATCH_SIZE)"
fi

# Verify dataset exists
if [[ ! -d "$FULL_DATASET_PATH" ]]; then
  echo "Error: Dataset not found at $FULL_DATASET_PATH"
  echo "Run: bash setup.sh"
  exit 1
fi

# Count images
IMG_COUNT=$(find "$FULL_DATASET_PATH" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" | wc -l)
if [[ "$IMG_COUNT" -eq 0 ]]; then
  echo "Error: No images in $FULL_DATASET_PATH"
  exit 1
fi
echo "Dataset:  $FULL_DATASET_PATH ($IMG_COUNT images)"

# Activate sd-scripts venv
source "$SD_SCRIPTS/venv/bin/activate"

# Run WD14 tagger
echo "Tagging..."
python "$SD_SCRIPTS/finetune/tag_images_by_wd14_tagger.py" \
  --onnx \
  --batch_size $BATCH_SIZE \
  --thresh 0.35 \
  --caption_extension .txt \
  --model_dir "$SD_SCRIPTS/wd14_models" \
  "$FULL_DATASET_PATH"

# Count generated captions
TXT_COUNT=$(find "$FULL_DATASET_PATH" -name "*.txt" | wc -l)

echo ""
echo "Done. $TXT_COUNT caption files generated."
echo "Next: bash tagger_cleanup.sh <pony|lustify>"
