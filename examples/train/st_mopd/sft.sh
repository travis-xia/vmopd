#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_SSL_VERIFICATION="${HF_HUB_DISABLE_SSL_VERIFICATION:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="${DATA_DIR:-data/st_mopd}"
DATASET="${DATASET:-$DATA_DIR/mixed_sft.jsonl}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct}"
MODEL="${MODEL:-$LOCAL_MODEL_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-output/st_mopd/sft_mixed}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEEPSPEED="${DEEPSPEED:-zero3}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.01}"

export NPROC_PER_NODE

swift sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --load_from_cache_file true \
  --split_dataset_ratio "$SPLIT_DATASET_RATIO" \
  --check_model false \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --learning_rate "$LR" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --max_length "$MAX_LENGTH" \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --deepspeed "$DEEPSPEED"
