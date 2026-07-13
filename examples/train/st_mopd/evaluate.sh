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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

ACTION="all"  # build, infer, score, all
EVAL_MODEL_STAGE="base"  # base, sft, rl
EVAL_RUN_NAME="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="output/st_mopd/eval/$EVAL_RUN_NAME"
MODEL_PATH="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct"
RL_SYSTEM_PROMPT="A conversation between user and assistant. The user provides a visual input and asks a question. The assistant MUST first think about the reasoning process in the mind and then provide the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."

mkdir -p "$OUTPUT_DIR"

run_build() {
  python3 st_mopd/evaluate.py build \
    --annotation-root "Videochat-R1/annotations" \
    --output-dir "data/st_mopd_eval" \
    --charades-video-root "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/VideoAuto-R1-Data/CharadesSTA/Charades_v1_480" \
    --activitynet-video-root "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/TVG-R1/video-r1/data/GroundedVLLM/activitynet/videos" \
    --got-root "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/GoT-10k" \
    --charades-split "val" \
    --activitynet-split "val" \
    --got-split "val"
}

run_infer() {
  local system_args=()
  case "$EVAL_MODEL_STAGE" in
    base|original|pretrain|sft)
      ;;
    rl)
      system_args=(--system "$RL_SYSTEM_PROMPT")
      ;;
    *)
      echo "EVAL_MODEL_STAGE must be one of: base, original, pretrain, sft, rl. Got: $EVAL_MODEL_STAGE" >&2
      exit 2
      ;;
  esac

  swift infer \
    --model "$MODEL_PATH" \
    --max_pixels 12845056 \
    --val_dataset "data/st_mopd_eval/mixed_eval.jsonl" \
    --remove_unused_columns false \
    --result_path "$OUTPUT_DIR/predictions.jsonl" \
    --infer_backend "vllm" \
    --vllm_tensor_parallel_size 1 \
    --torch_dtype bfloat16 \
    --max_new_tokens 512 \
    --temperature "0.0" \
    --top_p "1.0" \
    --stream false \
    --write_batch_size 1000 \
    "${system_args[@]}"
}

run_score() {
  python3 st_mopd/evaluate.py score \
    --predictions "$OUTPUT_DIR/predictions.jsonl" \
    --gold-jsonl "data/st_mopd_eval/mixed_eval.jsonl" \
    --output "$OUTPUT_DIR/metrics.json" \
    --per-sample-output "$OUTPUT_DIR/per_sample.jsonl" \
    --pretty
}

case "$ACTION" in
  build)
    run_build
    ;;
  infer)
    run_infer
    ;;
  score)
    run_score
    ;;
  all)
    run_build
    run_infer
    run_score
    ;;
  *)
    echo "ACTION must be one of: build, infer, score, all. Got: $ACTION" >&2
    exit 2
    ;;
esac
