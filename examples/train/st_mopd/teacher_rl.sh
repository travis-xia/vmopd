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
export FPS_MAX_FRAMES=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DOMAIN="temporal"  # temporal or spatial
export NPROC_PER_NODE=8

case "$DOMAIN" in
  temporal)
    reward_funcs=(st_mopd_format st_mopd_temporal_iou)
    dataset_path="data/st_mopd/temporal_rl.jsonl"
    output_dir="output/st_mopd/temporal_teacher_rl"
    ;;
  spatial)
    reward_funcs=(st_mopd_format st_mopd_spatial_iou)
    dataset_path="data/st_mopd/spatial_rl.jsonl"
    output_dir="output/st_mopd/spatial_teacher_rl"
    ;;
  *)
    echo "DOMAIN must be temporal or spatial, got: $DOMAIN" >&2
    exit 2
    ;;
esac

swift rlhf \
  --rlhf_type grpo \
  --model "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct" \
  --max_pixels 12845056 \
  --external_plugins st_mopd/reward_plugin.py \
  --reward_funcs "${reward_funcs[@]}" \
  --reward_weights 1.0 1.0 \
  --dataset "$dataset_path" \
  --load_from_cache_file true \
  --check_model false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "0.5" \
  --vllm_tensor_parallel_size 1 \
  --torch_dtype bfloat16 \
  --tuner_type full \
  --freeze_vit false \
  --freeze_aligner false \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate "1e-5" \
  --lr_scheduler_type cosine \
  --max_length 16384 \
  --max_completion_length 768 \
  --num_generations 8 \
  --temperature 1.0 \
  --top_p 0.85 \
  --save_steps 200 \
  --save_total_limit 5 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 8 \
  --output_dir "$output_dir" \
  --deepspeed "zero2" \
  --beta "0.001" \
  --log_completions true \
  --report_to tensorboard
