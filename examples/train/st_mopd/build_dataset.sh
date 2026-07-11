#!/usr/bin/env bash
set -euo pipefail

# Edit the values in this block before running the script.
export HF_HUB_DISABLE_SSL_VERIFICATION=1

python st_mopd/build_dataset.py \
  --annotation-root "Videochat-R1/annotations" \
  --output-dir "data/st_mopd" \
  --charades-video-root "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/VideoAuto-R1-Data/CharadesSTA/Charades_v1_480" \
  --activitynet-video-root "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/TVG-R1/video-r1/data/GroundedVLLM/activitynet/videos" \
  --got-root "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/GoT-10k" \
  "$@"
