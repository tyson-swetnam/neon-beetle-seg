"""Download the Biorepository's beetle images listed in biorepo_images.parquet to local disk.

Photographs of bycatch (reptiles, amphibians, mammals) are catalogued in biorepo_images with
their URLs but are not downloaded: they are not beetles and are not segmented.

Resumable: files already on disk are only re-hashed. Writes data/tables/biorepo_image_files.parquet
with one row per image (local path, bytes, sha256, pixel size, status) so every later step can
account for images that failed to download.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import config
from .fetch_biorepo import CARABID_GROUPS

USER_AGENT = "neon-beetle-seg/0.2 (+https://github.com/tyson-swetnam/neon-beetle-seg)"
WORKERS = 6  # stay polite to the portal

Image.MAX_IMAGE_PIXELS = None


def local_path(image_id: str, url: str) -> Path:
    # image_id is "biorepo/<collection>/<stem>"
    return config.IMAGES_DIR / f"{image_id}{Path(url).suffix.lower() or '.jpg'}"


def _fetch_one(client: httpx.Client, image_id: str, url: str) -> dict:
    path = local_path(image_id, url)
    row = {"image_id": image_id, "local_path": str(path.relative_to(config.ROOT)), "status": "ok",
           "bytes": None, "sha256": None, "width": None, "height": None}
    try:
        if not path.exists() or path.stat().st_size == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(4):
                try:
                    r = client.get(url)
                    if r.status_code == 200 and r.content:
                        tmp = path.with_suffix(path.suffix + ".part")
                        tmp.write_bytes(r.content)
                        tmp.rename(path)
                        break
                    if r.status_code in (403, 404, 410):
                        row["status"] = f"http_{r.status_code}"
                        return row
                except httpx.HTTPError:
                    pass
                time.sleep(2 ** attempt)
            else:
                row["status"] = "download_failed"
                return row
        data = path.read_bytes()
        row["bytes"] = len(data)
        row["sha256"] = hashlib.sha256(data).hexdigest()
        with Image.open(path) as im:
            row["width"], row["height"] = im.size
    except Exception as e:  # noqa: BLE001 - record and move on; one bad file must not stop the run
        row["status"] = f"error:{type(e).__name__}"
    return row


def main() -> None:
    config.ensure_dirs()
    images = pd.read_parquet(config.TABLES / "biorepo_images.parquet")
    images = images[images["specimen_group"].isin(CARABID_GROUPS)]
    jobs = list(zip(images["image_id"], images["image_url"]))
    with httpx.Client(follow_redirects=True, timeout=120, headers={"User-Agent": USER_AGENT}) as client:
        with ThreadPoolExecutor(WORKERS) as pool:
            rows = list(tqdm(pool.map(lambda j: _fetch_one(client, *j), jobs), total=len(jobs),
                             desc="biorepo images", mininterval=30))
    out = pd.DataFrame(rows)
    out.to_parquet(config.TABLES / "biorepo_image_files.parquet", index=False)
    print(out["status"].value_counts().to_string())
    print(f"total {out['bytes'].sum() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
