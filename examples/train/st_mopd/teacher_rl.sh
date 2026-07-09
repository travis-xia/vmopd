#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_SSL_VERIFICATION="${HF_HUB_DISABLE_SSL_VERIFICATION:-1}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DOMAIN="${DOMAIN:-temporal}"
DATA_DIR="${DATA_DIR:-data/st_mopd}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TUNER_TYPE="${TUNER_TYPE:-lora}"
DEEPSPEED="${DEEPSPEED:-zero2}"
LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
BETA="${BETA:-0.001}"

export NPROC_PER_NODE

case "$DOMAIN" in
  temporal)
    DATASET="${DATASET:-$DATA_DIR/temporal_rl.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-output/st_mopd/temporal_teacher_rl}"
    reward_funcs=(st_mopd_format st_mopd_temporal_iou)
    ;;
  spatial)
    DATASET="${DATASET:-$DATA_DIR/spatial_rl.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-output/st_mopd/spatial_teacher_rl}"
    reward_funcs=(st_mopd_format st_mopd_spatial_iou)
    ;;
  *)
    echo "DOMAIN must be temporal or spatial, got: $DOMAIN" >&2
    exit 2
    ;;
esac

tuner_args=(--tuner_type "$TUNER_TYPE")
if [[ "$TUNER_TYPE" == "lora" ]]; then
  tuner_args+=(
    --lora_rank 8
    --lora_alpha 32
    --target_modules all-linear
    --freeze_vit true
    --freeze_aligner true
  )
fi

swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --external_plugins st_mopd/reward_plugin.py \
  --reward_funcs "${reward_funcs[@]}" \
  --reward_weights 0.2 1.0 \
  --dataset "$DATASET" \
  --load_from_cache_file true \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --torch_dtype bfloat16 \
  "${tuner_args[@]}" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LR" \
  --lr_scheduler_type cosine \
  --max_length "$MAX_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --num_generations "$NUM_GENERATIONS" \
  --temperature 1.0 \
  --top_p 0.85 \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 3 \
  --logging_steps "$LOGGING_STEPS" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED" \
  --beta "$BETA" \
  --log_completions true \
  --report_to tensorboard
