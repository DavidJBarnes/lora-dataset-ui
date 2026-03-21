#!/bin/bash
# setup.sh — Bootstrap project directory structure from project.conf
#
# Run once after cloning or after changing TRIGGER/NUM_REPEATS in project.conf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/project.conf"

SUBSET="${NUM_REPEATS}_${TRIGGER}_${CLASS}"
FULL_DATASET_PATH="${DATASET_DIR}/${SUBSET}"

echo "Project Setup"
echo "============================================================"
echo "  Trigger:     ${TRIGGER}"
echo "  Class:       ${CLASS}"
echo "  Repeats:     ${NUM_REPEATS}"
echo "  Dataset dir: ${FULL_DATASET_PATH}"
echo ""

mkdir -p "$FULL_DATASET_PATH"
mkdir -p raw_images
mkdir -p outputs
mkdir -p logs
mkdir -p samples
mkdir -p docs

echo "Directories created:"
echo "  $FULL_DATASET_PATH"
echo "  raw_images/"
echo "  outputs/"
echo "  logs/"
echo "  samples/"
echo ""
echo "Done. Edit project.conf if you need to change settings."
