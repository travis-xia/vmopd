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

PYTHON="${PYTHON:-python3}"
ACTION="${ACTION:-all}"  # build, infer, score, all

ANNOTATION_ROOT="${ANNOTATION_ROOT:-Videochat-R1/annotations}"
DATA_DIR="${DATA_DIR:-data/st_mopd_eval}"
EVAL_DATASET="${EVAL_DATASET:-$DATA_DIR/mixed_eval.jsonl}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-output/st_mopd/eval/$EVAL_RUN_NAME}"
RESULT_PATH="${RESULT_PATH:-$OUTPUT_DIR/predictions.jsonl}"
METRICS_PATH="${METRICS_PATH:-$OUTPUT_DIR/metrics.json}"
PER_SAMPLE_PATH="${PER_SAMPLE_PATH:-$OUTPUT_DIR/per_sample.jsonl}"

CHARADES_VIDEO_ROOT="${CHARADES_VIDEO_ROOT:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/VideoAuto-R1-Data/CharadesSTA/Charades_v1_480}"
ACTIVITYNET_VIDEO_ROOT="${ACTIVITYNET_VIDEO_ROOT:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/TVG-R1/video-r1/data/GroundedVLLM/activitynet/videos}"
GOT_ROOT="${GOT_ROOT:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/GoT-10k}"
CHARADES_SPLIT="${CHARADES_SPLIT:-val}"
ACTIVITYNET_SPLIT="${ACTIVITYNET_SPLIT:-val}"
GOT_SPLIT="${GOT_SPLIT:-val}"
CHARADES_ANNO_FILE="${CHARADES_ANNO_FILE:-}"
ACTIVITYNET_ANNO_FILE="${ACTIVITYNET_ANNO_FILE:-}"
GOT_ANNO_FILE="${GOT_ANNO_FILE:-}"
MAX_TEMPORAL_SAMPLES="${MAX_TEMPORAL_SAMPLES:-}"
MAX_SPATIAL_SAMPLES="${MAX_SPATIAL_SAMPLES:-}"
MAX_SAMPLES_PER_SOURCE="${MAX_SAMPLES_PER_SOURCE:-}"
STRICT_MEDIA="${STRICT_MEDIA:-0}"

LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct}"
MODEL="${MODEL:-$LOCAL_MODEL_DIR}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
WRITE_BATCH_SIZE="${WRITE_BATCH_SIZE:-1000}"
GOLD_JSONL="${GOLD_JSONL:-}"

mkdir -p "$OUTPUT_DIR"

build_extra_args=()
if [[ -n "$CHARADES_ANNO_FILE" ]]; then
  build_extra_args+=(--charades-anno-file "$CHARADES_ANNO_FILE")
fi
if [[ -n "$ACTIVITYNET_ANNO_FILE" ]]; then
  build_extra_args+=(--activitynet-anno-file "$ACTIVITYNET_ANNO_FILE")
fi
if [[ -n "$GOT_ANNO_FILE" ]]; then
  build_extra_args+=(--got-anno-file "$GOT_ANNO_FILE")
fi
if [[ -n "$MAX_TEMPORAL_SAMPLES" ]]; then
  build_extra_args+=(--max-temporal-samples "$MAX_TEMPORAL_SAMPLES")
fi
if [[ -n "$MAX_SPATIAL_SAMPLES" ]]; then
  build_extra_args+=(--max-spatial-samples "$MAX_SPATIAL_SAMPLES")
fi
if [[ -n "$MAX_SAMPLES_PER_SOURCE" ]]; then
  build_extra_args+=(--max-samples-per-source "$MAX_SAMPLES_PER_SOURCE")
fi
if [[ "$STRICT_MEDIA" == "1" ]]; then
  build_extra_args+=(--strict-media)
fi

score_extra_args=()
if [[ -n "$GOLD_JSONL" ]]; then
  score_extra_args+=(--gold-jsonl "$GOLD_JSONL")
fi

run_build() {
  "$PYTHON" st_mopd/evaluate.py build \
    --annotation-root "$ANNOTATION_ROOT" \
    --output-dir "$DATA_DIR" \
    --charades-video-root "$CHARADES_VIDEO_ROOT" \
    --activitynet-video-root "$ACTIVITYNET_VIDEO_ROOT" \
    --got-root "$GOT_ROOT" \
    --charades-split "$CHARADES_SPLIT" \
    --activitynet-split "$ACTIVITYNET_SPLIT" \
    --got-split "$GOT_SPLIT" \
    "${build_extra_args[@]}"
}

run_infer() {
  swift infer \
    --model "$MODEL" \
    --val_dataset "$EVAL_DATASET" \
    --result_path "$RESULT_PATH" \
    --infer_backend "$INFER_BACKEND" \
    --check_model false \
    --torch_dtype bfloat16 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --stream false \
    --write_batch_size "$WRITE_BATCH_SIZE"
}

run_score() {
  "$PYTHON" st_mopd/evaluate.py score \
    --predictions "$RESULT_PATH" \
    --output "$METRICS_PATH" \
    --per-sample-output "$PER_SAMPLE_PATH" \
    --pretty \
    "${score_extra_args[@]}"
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
