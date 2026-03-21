#!/bin/bash
# tagger_cleanup.sh — Clean WD14 captions with model-appropriate format
#
# Usage:
#   bash tagger_cleanup.sh pony
#   bash tagger_cleanup.sh lustify

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/project.conf"

MODEL="${1:-}"
SUBSET="${NUM_REPEATS}_${TRIGGER}_${CLASS}"
FULL_DATASET_PATH="${DATASET_DIR}/${SUBSET}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: bash tagger_cleanup.sh <pony|lustify>"
  exit 1
fi

case "$MODEL" in
  pony)
    PREFIX="score_9, score_8_up, score_7_up, source_realistic, ${TRIGGER} ${CLASS}"
    echo "Mode: CyberRealistic Pony (booru tags, score prefix)"
    ;;
  lustify)
    PREFIX="${TRIGGER} ${CLASS}"
    echo "Mode: Lustify-SDXL (booru tags, no score prefix)"
    ;;
  *)
    echo "Error: Unknown model '$MODEL'. Use 'pony' or 'lustify'."
    exit 1
    ;;
esac

# Tags to REMOVE — character's fixed features
REMOVE_TAGS=(
  "blonde_hair" "blonde hair"
  "dirty_blonde" "dirty blonde"
  "light_brown_hair" "light brown hair"
  "brown_hair"
  "blue_eyes" "blue eyes"
  "grey_eyes" "grey eyes"
  "green_eyes" "green eyes"
  "freckles"
  "oval_face" "oval face"
  "slim" "thin"
  "pale_skin" "pale skin"
  "light_skin" "light skin"
  "1girl" "solo"
  "breasts" "small_breasts" "medium_breasts" "large_breasts"
)

cd "$FULL_DATASET_PATH" || { echo "Error: $FULL_DATASET_PATH not found"; exit 1; }

count=0
for f in *.txt; do
  content=$(cat "$f")

  # Strip any existing prefix (safe to re-run)
  content=$(echo "$content" | sed "s/^score_9, score_8_up, score_7_up, source_realistic, ${TRIGGER} ${CLASS}, *//")
  content=$(echo "$content" | sed "s/^${TRIGGER} ${CLASS}, *//")

  for tag in "${REMOVE_TAGS[@]}"; do
    content=$(echo "$content" | sed -E "s/, *${tag}//gi; s/^${tag}, *//gi")
  done

  content=$(echo "$content" | sed -E 's/, *,/,/g; s/^, *//; s/, *$//; s/  +/ /g')
  echo "${PREFIX}, ${content}" > "$f"
  echo "Processed: $f"
  count=$((count + 1))
done

echo ""
echo "Done. Processed ${count} files with ${MODEL} format."
