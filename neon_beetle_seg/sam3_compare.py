"""Compare SAM 3 (text-prompted) with the main pipeline on a sample of 2018 tray photos.

SAM 3 segments every instance of a text concept in one pass, with no separate detector. It is a
gated model: the HuggingFace account behind HF_TOKEN must have been granted access to
facebook/sam3. This step is a comparison only; the lake's measurements come from the main
pipeline (Grounding DINO + SAM 2.1).

Writes data/tables/sam3_comparison.parquet: per tray, the number of specimens each method found,
agreement with the human boxes, how many annotated individuals each covers, and seconds per tray.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import config, measure, models
from .segment import MAX_BOX_FRAC_TRAY
from .validate import _match

Image.MAX_IMAGE_PIXELS = None
MAX_SIDE = 2048  # SAM 3 returns a full-size mask per instance; a reduced tray keeps a 266-beetle tray in memory


def _mask_boxes(masks: np.ndarray) -> np.ndarray:
    out = []
    for m in masks:
        ys, xs = np.nonzero(m)
        out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [0, 0, 0, 0])
    return np.array(out, np.float32).reshape(-1, 4)


def run(n_trays: int = 60, prompt: str = "beetle", seed: int = 0) -> pd.DataFrame:
    token = config.load_secrets().get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; SAM 3 is a gated model and cannot be downloaded without it.")
    man = pd.read_parquet(config.TABLES / "image_manifest.parquet")
    trays = man[man["source"] == "hf2018"].sample(n_trays, random_state=seed)
    boxes = pd.read_parquet(config.TABLES / "tray_boxes_2018.parquet")
    # annotation lines in the photo's own pixel frame only (see tables.fix_annotation_frame)
    lines = pd.read_parquet(config.TABLES / "elytra_matches_2018.parquet")
    lines = lines[(lines["structure"] == "ElytraLength") & lines["frame_ok"]]
    inst = pd.read_parquet(config.TABLES / "instances.parquet",
                           columns=["image_id", "box_x1", "box_y1", "box_x2", "box_y2", "win_x1", "win_y1",
                                    "mask_w", "mask_h", "mask_rle"])
    sam3 = models.Sam3(token=token)
    rows = []
    for r in trays.itertuples():
        image = Image.open(config.ROOT / r.local_path).convert("RGB")
        s = MAX_SIDE / max(image.size)
        small = image.resize((round(image.size[0] * s), round(image.size[1] * s)), Image.BILINEAR) if s < 1 else image
        s = small.size[0] / image.size[0]
        torch.cuda.synchronize()
        t = time.time()
        masks, _, scores = sam3.segment_text(small, prompt)
        torch.cuda.synchronize()
        secs = time.time() - t
        area = masks.reshape(len(masks), -1).sum(axis=1) if len(masks) else np.array([])
        keep = area < MAX_BOX_FRAC_TRAY * small.size[0] * small.size[1] if len(masks) else np.array([], bool)
        masks = masks[keep] if len(masks) else masks
        b3 = _mask_boxes(masks) / s
        gt = boxes.loc[boxes["image_id"] == r.image_id, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        mine = inst[inst["image_id"] == r.image_id]
        bm = mine[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(np.float32)
        ln = lines[(lines["image_id"] == r.image_id)]
        users = ln["user_name"].value_counts()
        ln = ln[ln["user_name"] == ("IsaFluck" if "IsaFluck" in users.index else users.index[0])] if len(users) else ln
        if len(ln):
            ln = ln[ln["workflowID"] == ln["workflowID"].value_counts().index[0]]
        px, py = ((ln["x1"] + ln["x2"]) / 2).to_numpy(), ((ln["y1"] + ln["y2"]) / 2).to_numpy()
        union3 = masks.any(axis=0) if len(masks) else np.zeros(small.size[::-1], bool)
        cov3 = sum(bool(union3[min(int(y * s), union3.shape[0] - 1), min(int(x * s), union3.shape[1] - 1)]) for x, y in zip(px, py))
        cov_main = 0
        decoded = [(m.win_x1, m.win_y1, measure.rle_decode(m.mask_rle, m.mask_h, m.mask_w)) for m in mine.itertuples()]
        for x, y in zip(px, py):
            for wx, wy, mk in decoded:
                mx, my = int(round(x - wx)), int(round(y - wy))
                if 0 <= my < mk.shape[0] and 0 <= mx < mk.shape[1] and mk[my, mx]:
                    cov_main += 1
                    break
        rows.append(dict(image_id=r.image_id, n_human_boxes=len(gt), n_annotated=len(ln),
                         sam3_n=len(b3), sam3_tp_iou50=_match(b3, gt), sam3_annotated_covered=cov3, sam3_seconds=secs,
                         main_n=len(bm), main_tp_iou50=_match(bm, gt), main_annotated_covered=cov_main,
                         sam3_prompt=prompt, sam3_input_long_side=max(small.size)))
        del masks
    out = pd.DataFrame(rows)
    out.to_parquet(config.TABLES / "sam3_comparison.parquet", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trays", type=int, default=60)
    ap.add_argument("--prompt", default="beetle")
    args = ap.parse_args()
    d = run(args.trays, args.prompt)
    for name in ("sam3", "main"):
        tp, n, g = d[f"{name}_tp_iou50"].sum(), d[f"{name}_n"].sum(), d["n_human_boxes"].sum()
        cov = d[f"{name}_annotated_covered"].sum() / max(d["n_annotated"].sum(), 1)
        print(f"{name:5s} precision {tp / max(n, 1):.3f}  recall {tp / max(g, 1):.3f}  annotated covered {cov:.3f}")
    print(f"SAM 3: {d['sam3_seconds'].mean():.1f} s per tray; peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
