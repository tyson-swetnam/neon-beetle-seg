"""Download the Imageomics beetle datasets from HuggingFace to local disk.

Only the parts the pipeline uses are fetched (see PATTERNS). Each dataset is pinned to the commit
it was downloaded at; the revision is written to data/hf/revisions.json and into run_provenance.
"""
from __future__ import annotations

import json
import sys

from huggingface_hub import HfApi, snapshot_download

from . import config

# dataset -> glob patterns to fetch
PATTERNS: dict[str, list[str]] = {
    # 577 tray photos of ethanol specimens (2018, 30 sites) + Zooniverse elytra annotations.
    # individual_specimens/ and the segmented train/test splits are derived crops and are skipped.
    "imageomics/2018-NEON-beetles": [
        "README.md", "*.csv", "group_images/*", "group_images_resized/*", "individual_specimens/metadata.csv",
    ],
    # 420 pinned-tray photos + 1,614 individuals from PUUM with trait annotations.
    "imageomics/Hawaii-beetles": ["*"],
    # 44,510 individual pinned crops. The images are read from the parquet shards under data/:
    # at revision 11d861b5, 24,856 of the 44,510 entries in flattened_images.zip are git-lfs
    # pointer stubs rather than images, so the zip cannot be used on its own.
    "imageomics/sentinel-beetles": ["README.md", "*.csv", "data/*.parquet", "color_and_scale_images/*"],
}


def local_dir(repo: str):
    return config.HF_DIR / repo.split("/")[1]


def fetch(repos: list[str] | None = None) -> dict[str, str]:
    config.ensure_dirs()
    token = config.load_secrets().get("HF_TOKEN")
    api = HfApi(token=token)
    rev_file = config.HF_DIR / "revisions.json"
    revisions = json.loads(rev_file.read_text()) if rev_file.exists() else {}
    for repo in repos or list(PATTERNS):
        sha = revisions.get(repo) or api.dataset_info(repo).sha
        snapshot_download(
            repo, repo_type="dataset", revision=sha, allow_patterns=PATTERNS[repo],
            local_dir=local_dir(repo), max_workers=config.N_THREADS, token=token,
        )
        revisions[repo] = sha
        rev_file.write_text(json.dumps(revisions, indent=1))
        print(f"{repo} @ {sha[:10]} -> {local_dir(repo)}", flush=True)
    return revisions


def extract_sentinel() -> int:
    """Write each sentinel-beetles image embedded in the parquet shards to images/<public_id>.png."""
    import pyarrow.parquet as pq

    d = local_dir("imageomics/sentinel-beetles")
    out = d / "images"
    out.mkdir(exist_ok=True)
    n = 0
    for f in sorted((d / "data").glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=256, columns=["relative_img_loc", "file_path"]):
            for name, img in zip(batch.column("relative_img_loc").to_pylist(), batch.column("file_path").to_pylist()):
                (out / name).write_bytes(img["bytes"])
                n += 1
    return n


if __name__ == "__main__":
    fetch(sys.argv[1:] or None)
    print(f"sentinel images extracted: {extract_sentinel():,}")
