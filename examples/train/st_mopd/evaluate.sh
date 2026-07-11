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

ACTION="all"  # build, infer, score, all
EVAL_RUN_NAME="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="output/st_mopd/eval/$EVAL_RUN_NAME"

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
  swift infer \
    --model "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct" \
    --max_pixels 12845056 \
    --val_dataset "data/st_mopd_eval/mixed_eval.jsonl" \
    --result_path "$OUTPUT_DIR/predictions.jsonl" \
    --infer_backend "vllm" \
    --check_model false \
    --torch_dtype bfloat16 \
    --max_new_tokens 512 \
    --temperature "0.0" \
    --top_p "1.0" \
    --stream false \
    --write_batch_size 1000
}

run_score() {
  python3 st_mopd/evaluate.py score \
    --predictions "$OUTPUT_DIR/predictions.jsonl" \
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
