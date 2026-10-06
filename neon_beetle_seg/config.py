"""Paths, secrets and resource limits shared by every pipeline step."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("NBS_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = ROOT / "cache"
METADATA = ROOT / "metadata"

NEON_DIR = DATA / "neon"
BIOREPO_DIR = DATA / "biorepo"
GBIF_DIR = DATA / "gbif"
HF_DIR = DATA / "hf"
IMAGES_DIR = DATA / "images"
TABLES = DATA / "tables"  # parquet tables that feed the lake
LAKE_DIR = DATA / "ducklake"

SECRETS_FILE = Path(os.environ.get("NBS_SECRETS", Path.home() / ".neon-beetle-secrets.env"))
IRODS_ROOT = os.environ.get("NBS_IRODS_ROOT", "/iplant/home/tswetnam/neon-beetle-seg")

NEON_PRODUCT = "DP1.10022.001"
NEON_PUBLISHER_KEY = "e794e60e-e558-4549-99f8-cfb241cdce24"  # GBIF publishing organisation
CARABIDAE_KEY = 3792  # GBIF backbone familyKey

# The container is cgroup-limited to 8 cores although the host reports 128.
N_THREADS = int(os.environ.get("NBS_THREADS", "8"))


def load_secrets() -> dict[str, str]:
    """Read KEY=VALUE lines from the secrets file into os.environ (existing env wins).

    Returns the keys that are set, with values, so callers can check presence. Never log the values.
    """
    found: dict[str, str] = {}
    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            if value and value != "...":
                os.environ.setdefault(key, value)
    for key in ("NEON_TOKEN", "GBIF_USER", "GBIF_PWD", "GBIF_EMAIL", "HF_TOKEN"):
        if os.environ.get(key):
            found[key] = os.environ[key]
    return found


def limit_threads() -> None:
    """Cap BLAS/OpenMP/torch thread pools to the container's real CPU allowance."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(N_THREADS))


def ensure_dirs() -> None:
    for d in (NEON_DIR, BIOREPO_DIR, GBIF_DIR, HF_DIR, IMAGES_DIR, TABLES, LAKE_DIR, OUT, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    # keep model weights and HF datasets on local disk, inside the (gitignored) cache
    os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
