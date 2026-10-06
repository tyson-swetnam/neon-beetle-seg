#!/usr/bin/env bash
# One-off repair: reprocess images whose stored mask was the backdrop (see segment.fix_backdrop_mask).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m neon_beetle_seg.segment biorepo --recheck-backdrop 2>&1 | grep -v -E 'Warning|warn|sam2_video' | tee -a logs/segment_biorepo_fix.log
.venv/bin/python -m neon_beetle_seg.segment sentinel --recheck-backdrop --sam facebook/sam2.1-hiera-base-plus 2>&1 \
  | grep -v -E 'Warning|warn|sam2_video' | tee -a logs/segment_sentinel_fix.log
