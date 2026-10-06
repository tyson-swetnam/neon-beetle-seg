#!/usr/bin/env bash
# Rebuild the working environment on a fresh VM (the CyVerse VICE container is ephemeral).
# Usage: scripts/bootstrap.sh            # from the repository root
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Python 3.12 + torch (CUDA 12.6 wheels) + everything else, exactly as locked in uv.lock
UV_LINK_MODE=copy uv sync --frozen --python 3.12 --extra yolo --extra dev

# git push through the gh login instead of the image's interactive credential manager
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  git config --local --replace-all credential.https://github.com.helper ""
  git config --local --add credential.https://github.com.helper '!gh auth git-credential'
fi

.venv/bin/python - <<'EOF'
import duckdb, torch
duckdb.sql("INSTALL ducklake; LOAD ducklake;")
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
EOF

if [ ! -f "$HOME/.neon-beetle-secrets.env" ]; then
  cat <<'EOF'

No ~/.neon-beetle-secrets.env yet. Create it (chmod 600) with:
  NEON_TOKEN=...      required for `nbs neon`
  GBIF_USER=...       optional: citable GBIF download instead of the anonymous search API
  GBIF_PWD=...
  GBIF_EMAIL=...
  HF_TOKEN=...        optional: higher rate limits; required for `nbs sam3` (gated model)
EOF
fi
echo "ready: .venv/bin/nbs --help"
