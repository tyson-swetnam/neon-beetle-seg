#!/usr/bin/env bash
# Run detection + segmentation + measurement over every image pool, in order of priority.
# Resumable: each pool skips images already present in outputs/shards/<pool>/.
# The NEON-joinable pools use SAM 2.1 large; the 44,510 sentinel crops use base-plus, which is
# about twice as fast on the A16 and agrees with large to mask IoU 0.97-0.99 on test images.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for pool in hf2018 hawaii biorepo herp; do
  .venv/bin/python -m neon_beetle_seg.segment "$pool" 2>&1 | grep -v -E 'Warning|warn|sam2_video' | tee -a "logs/segment_${pool}.log"
done
.venv/bin/python -m neon_beetle_seg.segment sentinel --sam facebook/sam2.1-hiera-base-plus 2>&1 \
  | grep -v -E 'Warning|warn|sam2_video' | tee -a logs/segment_sentinel.log
