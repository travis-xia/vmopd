#!/usr/bin/env bash
set -euo pipefail

# Edit the values in this block before running the script.
export HF_HUB_DISABLE_SSL_VERIFICATION=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VIDEO_MIN_PIXELS=12544
export VIDEO_MAX_PIXELS=602112
export VIDEO_TOTAL_PIXELS=2809856
export FPS=2.0
export FPS_MIN_FRAMES=4
export FPS_MAX_FRAMES=768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC_PER_NODE=8

swift rlhf \
  --rlhf_type grpo \
  --model "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct" \
  --max_pixels 12845056 \
  --external_plugins st_mopd/reward_plugin.py \
  --reward_funcs st_mopd_format st_mopd_task_iou \
  --reward_weights 1.0 1.0 \
  --dataset "data/st_mopd/mixed_rl.jsonl" \
  --load_from_cache_file true \
  --check_model false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "0.5" \
  --vllm_tensor_parallel_size 1 \
  --torch_dtype bfloat16 \
  --tuner_type full \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate "1e-6" \
  --lr_scheduler_type cosine \
  --max_length 4096 \
  --max_completion_length 512 \
  --num_generations 8 \
  --temperature 1.0 \
  --top_p 0.85 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --output_dir "output/st_mopd/mixed_rl" \
  --deepspeed "zero3" \
  --beta "0.001" \
  --log_completions true \
  --report_to tensorboard
