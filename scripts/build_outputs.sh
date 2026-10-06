#!/usr/bin/env bash
# Everything after segmentation: collect results, validate, herp suggestions, lake, docs, report.
# Usage: scripts/build_outputs.sh        (needs the GPU briefly for the herp step)
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $* ($(date -u +%H:%M:%S))"; .venv/bin/python -m "neon_beetle_seg.$1" "${@:2}" 2>&1 | grep -v -i -E 'warn|it/s\]' || true; }
run tables
run validate
run herp_id
run build_ducklake
echo "== schema doc"; .venv/bin/python scripts/make_schema_doc.py
run report
echo "== finished $(date -u +%H:%M:%S)"
