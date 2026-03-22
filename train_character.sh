#!/bin/bash
set -euo pipefail

# ============================================================
# Character LoRA Training — Multi-model support
#
# Usage:
#   bash train_character.sh pony
#   bash train_character.sh lustify
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
OUTPUT_DIR="${PROJECT_DIR}/outputs"
LOG_DIR="${PROJECT_DIR}/logs"
CONFIG_FILE="${PROJECT_DIR}/.config_${MODEL}.toml"

# Model-specific settings
case "$MODEL" in
  pony)
    MODEL_PATH="$PONY_MODEL_PATH"
    OUTPUT_NAME="${TRIGGER}_pony_${VERSION}"
    KEEP_TOKENS=6
    echo "Training: CyberRealistic Pony"
    ;;
  lustify)
    MODEL_PATH="$LUSTIFY_MODEL_PATH"
    OUTPUT_NAME="${TRIGGER}_lustify_${VERSION}"
    KEEP_TOKENS=1
    echo "Training: Lustify-SDXL"
    ;;
  *)
    echo "Error: Unknown model '$MODEL'. Use 'pony' or 'lustify'."
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
  echo "Run: bash setup.sh"
  exit 1
fi

# Count images
IMG_COUNT=$(find "$DATASET_PATH_ABS" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" | wc -l)
if [[ "$IMG_COUNT" -eq 0 ]]; then
  echo "Error: No images in $DATASET_PATH_ABS"
  exit 1
fi
echo "Dataset:  ${IMG_COUNT} images × ${NUM_REPEATS} repeats = $((IMG_COUNT * NUM_REPEATS)) steps/epoch"

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

echo "Config:   $CONFIG_FILE (generated)"
echo "Model:    $MODEL_PATH"
echo "Output:   $OUTPUT_DIR/$OUTPUT_NAME"

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
  lustify)
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

# Hyperparameters
LEARNING_RATE="1e-4"
UNET_LR="1e-4"
TEXT_ENCODER_LR="5e-5"
LR_SCHEDULER="cosine_with_restarts"
LR_WARMUP_STEPS=100
LR_RESTART_CYCLES=1
MAX_TRAIN_EPOCHS=10
SAVE_EVERY_N_EPOCHS=2
NETWORK_DIM=32
NETWORK_ALPHA=16

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
  --optimizer_type="AdamW8bit" \
  --mixed_precision="bf16" \
  --cache_latents \
  --cache_latents_to_disk \
  --gradient_checkpointing \
  --max_data_loader_n_workers=2 \
  --seed=42 \
  --max_token_length=225 \
  --xformers \
  --bucket_no_upscale \
  --min_snr_gamma=5.0 \
  --noise_offset=0.0357 \
  $SAMPLE_ARGS \
  --logging_dir="$LOG_DIR"

echo ""
echo "========================================="
echo "Training complete! Model: $MODEL"
echo "Checkpoints: $OUTPUT_DIR"
echo "========================================="
