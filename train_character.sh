#!/bin/bash
set -euo pipefail

# ============================================================
# Character LoRA Training — Multi-model support
#
# Usage:
#   bash train_character.sh pony
#   bash train_character.sh lustify
#
# Reads all hyperparameters from project.conf.
# Supports weighted face subset (dataset/faces/) for better
# facial detail learning.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/project.conf"

MODEL="${1:-}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: bash train_character.sh <pony|lustify>"
  echo ""
  echo "  pony    — CyberRealistic Pony (clip_skip=2)"
  echo "  lustify — Lustify-SDXL (clip_skip=1)"
  exit 1
fi

# Auto-detect PROJECT_DIR if not set
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"

# Computed paths
DATASET_PATH_ABS="${PROJECT_DIR}/dataset"
FACE_PATH_ABS="${PROJECT_DIR}/dataset/faces"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
LOG_DIR="${PROJECT_DIR}/logs"
CONFIG_FILE="${PROJECT_DIR}/.config_${MODEL}.toml"

# Timestamp for unique output names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Hyperparameters from project.conf (with defaults)
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-AdamW8bit}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
UNET_LR="${LEARNING_RATE}"
TEXT_ENCODER_LR="${TEXT_ENCODER_LR:-2e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine_with_restarts}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
LR_RESTART_CYCLES="${LR_RESTART_CYCLES:-1}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-8}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-2}"
NETWORK_DIM="${NETWORK_DIM:-32}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
NOISE_OFFSET="${NOISE_OFFSET:-0.0357}"
MIN_SNR_GAMMA="${MIN_SNR_GAMMA:-5.0}"
NUM_REPEATS="${NUM_REPEATS:-8}"
FACE_REPEATS="${FACE_REPEATS:-20}"

# Prodigy optimizer settings
if [[ "$OPTIMIZER_TYPE" == "Prodigy" ]]; then
  LEARNING_RATE="1.0"
  UNET_LR="1.0"
  TEXT_ENCODER_LR="1.0"
  LR_SCHEDULER="constant_with_warmup"
  OPTIMIZER_ARGS='--optimizer_args decouple=True weight_decay=0.01 d_coef=2 use_bias_correction=True safeguard_warmup=True'
else
  OPTIMIZER_ARGS=""
fi

# Model-specific settings
case "$MODEL" in
  pony)
    MODEL_PATH="$PONY_MODEL_PATH"
    OUTPUT_NAME="${TRIGGER}_pony_${VERSION}_${TIMESTAMP}"
    KEEP_TOKENS=6
    CLIP_SKIP=2
    echo "Training: CyberRealistic Pony (clip_skip=2)"
    ;;
  lustify)
    MODEL_PATH="$LUSTIFY_MODEL_PATH"
    OUTPUT_NAME="${TRIGGER}_lustify_${VERSION}_${TIMESTAMP}"
    KEEP_TOKENS=1
    CLIP_SKIP=1
    echo "Training: Lustify-SDXL (clip_skip=1)"
    ;;
  realvis)
    MODEL_PATH="$REALVIS_MODEL_PATH"
    OUTPUT_NAME="${TRIGGER}_realvis_${VERSION}_${TIMESTAMP}"
    KEEP_TOKENS=1
    CLIP_SKIP=1
    echo "Training: RealVisXL V5.0 (clip_skip=1)"
    ;;
  *)
    echo "Error: Unknown model '$MODEL'. Use 'pony', 'lustify', or 'realvis'."
    exit 1
    ;;
esac

# Verify model exists
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: Model not found at $MODEL_PATH"
  echo "Find it: find ~ -name '*.safetensors' | grep -i '${MODEL}'"
  exit 1
fi

# Verify dataset exists
if [[ ! -d "$DATASET_PATH_ABS" ]]; then
  echo "Error: Dataset not found at $DATASET_PATH_ABS"
  exit 1
fi

# Count images
IMG_COUNT=$(find "$DATASET_PATH_ABS" -maxdepth 1 -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" | wc -l)
if [[ "$IMG_COUNT" -eq 0 ]]; then
  echo "Error: No images in $DATASET_PATH_ABS"
  exit 1
fi

# Count face images
FACE_COUNT=0
HAS_FACES=false
if [[ -d "$FACE_PATH_ABS" ]]; then
  FACE_COUNT=$(find "$FACE_PATH_ABS" -maxdepth 1 -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" | wc -l)
  if [[ "$FACE_COUNT" -gt 0 ]]; then
    HAS_FACES=true
  fi
fi

MAIN_STEPS=$((IMG_COUNT * NUM_REPEATS))
FACE_STEPS=$((FACE_COUNT * FACE_REPEATS))
TOTAL_STEPS=$((MAIN_STEPS + FACE_STEPS))
echo "Dataset:  ${IMG_COUNT} main images × ${NUM_REPEATS} repeats = ${MAIN_STEPS} steps"
if [[ "$HAS_FACES" == "true" ]]; then
  echo "Faces:    ${FACE_COUNT} face images × ${FACE_REPEATS} repeats = ${FACE_STEPS} steps"
fi
echo "Total:    ${TOTAL_STEPS} steps/epoch"

# Generate TOML config dynamically
cat > "$CONFIG_FILE" << EOF
[general]
enable_bucket = true
resolution = 1024
caption_extension = ".txt"
batch_size = 1
flip_aug = false
color_aug = false
keep_tokens = ${KEEP_TOKENS}
shuffle_caption = false
caption_dropout_rate = 0.0

[[datasets]]
  [[datasets.subsets]]
    image_dir = "${DATASET_PATH_ABS}"
    num_repeats = ${NUM_REPEATS}
EOF

# Add face subset if it exists and has images
if [[ "$HAS_FACES" == "true" ]]; then
  cat >> "$CONFIG_FILE" << EOF

  [[datasets.subsets]]
    image_dir = "${FACE_PATH_ABS}"
    num_repeats = ${FACE_REPEATS}
EOF
fi

# Add regularization images if reg/ directory exists and has images
REG_PATH_ABS="${PROJECT_DIR}/reg"
HAS_REG=false
REG_COUNT=0
if [[ -d "$REG_PATH_ABS" ]]; then
  REG_COUNT=$(find "$REG_PATH_ABS" -maxdepth 1 -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" | wc -l)
  if [[ "$REG_COUNT" -gt 0 ]]; then
    HAS_REG=true
    cat >> "$CONFIG_FILE" << EOF

  [[datasets.subsets]]
    image_dir = "${REG_PATH_ABS}"
    class_tokens = "${CLASS}"
    is_reg = true
    num_repeats = 1
EOF
    echo "Reg:      ${REG_COUNT} regularization images"
  fi
fi

echo "Config:   $CONFIG_FILE (generated)"
echo "Model:    $MODEL_PATH"
echo "Output:   $OUTPUT_DIR/$OUTPUT_NAME"
echo ""
echo "Hyperparameters:"
echo "  learning_rate:    $LEARNING_RATE"
echo "  text_encoder_lr:  $TEXT_ENCODER_LR"
echo "  network_dim:      $NETWORK_DIM"
echo "  network_alpha:    $NETWORK_ALPHA"
echo "  epochs:           $MAX_TRAIN_EPOCHS"
echo "  noise_offset:     $NOISE_OFFSET"
echo "  min_snr_gamma:    $MIN_SNR_GAMMA"

# Generate sample prompts dynamically
SAMPLE_PROMPTS="${PROJECT_DIR}/.sample_prompts_${MODEL}.txt"

case "$MODEL" in
  pony)
    cat > "$SAMPLE_PROMPTS" << EOF
score_9, score_8_up, score_7_up, source_realistic, ${TRIGGER} ${CLASS}, portrait, studio lighting, looking at viewer, neutral expression --n score_6, score_5, score_4, worst quality, low quality, bad anatomy, bad hands, imperfect eyes, skewed eyes, unnatural face --w 1024 --h 1024 --l 5 --s 30
score_9, score_8_up, score_7_up, source_realistic, ${TRIGGER} ${CLASS}, full body, standing, park, natural lighting, casual clothing, smile --n score_6, score_5, score_4, worst quality, low quality, bad anatomy, bad hands, imperfect eyes, skewed eyes, unnatural face --w 832 --h 1216 --l 5 --s 30
score_9, score_8_up, score_7_up, source_realistic, ${TRIGGER} ${CLASS}, close-up face, golden hour, outdoors, looking away --n score_6, score_5, score_4, worst quality, low quality, bad anatomy, bad hands, imperfect eyes, skewed eyes, unnatural face --w 1024 --h 1024 --l 5 --s 30
score_9, score_8_up, score_7_up, source_realistic, ${TRIGGER} ${CLASS}, upper body, office, professional clothing, serious expression --n score_6, score_5, score_4, worst quality, low quality, bad anatomy, bad hands, imperfect eyes, skewed eyes, unnatural face --w 896 --h 1152 --l 5 --s 30
EOF
    ;;
  lustify|realvis)
    cat > "$SAMPLE_PROMPTS" << EOF
${TRIGGER} ${CLASS}, portrait, studio lighting, looking at viewer, neutral expression, detailed skin --n worst quality, low quality, blurry, bad anatomy, bad hands, deformed face --w 1024 --h 1024 --l 5 --s 30
${TRIGGER} ${CLASS}, full body, standing in a park, natural lighting, casual clothing, smile --n worst quality, low quality, blurry, bad anatomy, bad hands, deformed face --w 832 --h 1216 --l 5 --s 30
${TRIGGER} ${CLASS}, close-up face, golden hour, outdoors, looking away, soft lighting --n worst quality, low quality, blurry, bad anatomy, bad hands, deformed face --w 1024 --h 1024 --l 5 --s 30
${TRIGGER} ${CLASS}, upper body, office setting, professional clothing, serious expression --n worst quality, low quality, blurry, bad anatomy, bad hands, deformed face --w 896 --h 1152 --l 5 --s 30
EOF
    ;;
esac

echo "Samples:  $SAMPLE_PROMPTS (generated)"
SAMPLE_ARGS="--sample_prompts=$SAMPLE_PROMPTS --sample_sampler=euler_a --sample_every_n_epochs=2"

# Activate sd-scripts venv
source "$SD_SCRIPTS/venv/bin/activate"

echo ""
echo "Launching training..."
echo ""

cd "$SD_SCRIPTS"
accelerate launch --num_cpu_threads_per_process 4 sdxl_train_network.py \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --dataset_config="$CONFIG_FILE" \
  --output_dir="$OUTPUT_DIR" \
  --output_name="$OUTPUT_NAME" \
  --save_model_as=safetensors \
  --save_every_n_epochs=$SAVE_EVERY_N_EPOCHS \
  --max_train_epochs=$MAX_TRAIN_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --unet_lr=$UNET_LR \
  --text_encoder_lr=$TEXT_ENCODER_LR \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --lr_scheduler_num_cycles=$LR_RESTART_CYCLES \
  --network_module=networks.lora \
  --network_dim=$NETWORK_DIM \
  --network_alpha=$NETWORK_ALPHA \
  --optimizer_type="$OPTIMIZER_TYPE" \
  $OPTIMIZER_ARGS \
  --mixed_precision="bf16" \
  --cache_latents \
  --cache_latents_to_disk \
  --gradient_checkpointing \
  --max_data_loader_n_workers=2 \
  --seed=42 \
  --max_token_length=225 \
  --xformers \
  --bucket_no_upscale \
  --clip_skip=$CLIP_SKIP \
  --min_snr_gamma=$MIN_SNR_GAMMA \
  --noise_offset=$NOISE_OFFSET \
  $SAMPLE_ARGS \
  --logging_dir="$LOG_DIR"

echo ""
echo "========================================="
echo "Training complete! Model: $MODEL"
echo "Checkpoints: $OUTPUT_DIR"
echo "========================================="
