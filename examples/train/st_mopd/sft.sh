#!/usr/bin/env bash
set -euo pipefail

# Edit the values in this block before running the script.
export HF_HUB_DISABLE_SSL_VERIFICATION=1
export CURL_CA_BUNDLE=
export REQUESTS_CA_BUNDLE=
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VIDEO_MIN_PIXELS=12544
export VIDEO_MAX_PIXELS=602112
export VIDEO_TOTAL_PIXELS=2809856
export FPS=2.0
export FPS_MIN_FRAMES=4
export FPS_MAX_FRAMES=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC_PER_NODE=8

swift sft \
  --model "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct" \
  --dataset "data/st_mopd/mixed_sft.jsonl" \
  --max_pixels 12845056 \
  --load_from_cache_file true \
  --split_dataset_ratio "0.01" \
  --check_model false \
  --tuner_type full \
  --freeze_vit false \
  --freeze_aligner false \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate "1e-5" \
  --gradient_accumulation_steps 2 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 5 \
  --logging_steps 5 \
  --max_length 16384 \
  --truncation_strategy delete \
  --output_dir "output/st_mopd/sft_mixed" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --deepspeed "zero3"
