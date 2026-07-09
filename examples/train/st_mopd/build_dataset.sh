#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_SSL_VERIFICATION="${HF_HUB_DISABLE_SSL_VERIFICATION:-1}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-}"

OUTPUT_DIR="${OUTPUT_DIR:-data/st_mopd}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-Videochat-R1/annotations}"
CHARADES_VIDEO_ROOT="${CHARADES_VIDEO_ROOT:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/VideoAuto-R1-Data/CharadesSTA/Charades_v1_480}"
ACTIVITYNET_VIDEO_ROOT="${ACTIVITYNET_VIDEO_ROOT:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/TVG-R1/video-r1/data/GroundedVLLM/activitynet/videos}"
GOT_ROOT="${GOT_ROOT:-/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/GoT-10k}"
CHARADES_ANNO_FILE="${CHARADES_ANNO_FILE:-}"
ACTIVITYNET_ANNO_FILE="${ACTIVITYNET_ANNO_FILE:-}"
GOT_ANNO_FILE="${GOT_ANNO_FILE:-}"

extra_args=()
if [[ -n "$CHARADES_ANNO_FILE" ]]; then
  extra_args+=(--charades-anno-file "$CHARADES_ANNO_FILE")
fi
if [[ -n "$ACTIVITYNET_ANNO_FILE" ]]; then
  extra_args+=(--activitynet-anno-file "$ACTIVITYNET_ANNO_FILE")
fi
if [[ -n "$GOT_ANNO_FILE" ]]; then
  extra_args+=(--got-anno-file "$GOT_ANNO_FILE")
fi

python st_mopd/build_dataset.py \
  --annotation-root "$ANNOTATION_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --charades-video-root "$CHARADES_VIDEO_ROOT" \
  --activitynet-video-root "$ACTIVITYNET_VIDEO_ROOT" \
  --got-root "$GOT_ROOT" \
  "${extra_args[@]}" \
  "$@"
