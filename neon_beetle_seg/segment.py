"""Detect, segment and measure every specimen in every image of the manifest.

Routes (image_manifest.route):
  tray    many specimens per photo: Grounding DINO on the whole photo -> one box per beetle ->
          SAM 2.1 mask per box -> BeetleFlow part labels per specimen crop
  single  one specimen in a larger photo (labels, rulers, grids around it): Grounding DINO ->
          best beetle box -> SAM 2.1 -> BeetleFlow
  crop    image is already a tight crop of one specimen (sentinel-beetles): BeetleFlow on the
          whole crop -> box around its foreground -> SAM 2.1

Output is sharded parquet under outputs/shards/<source>/ so a run can be stopped and resumed:
  instances-*.parquet  one row per specimen (boxes, mask RLE, pixel metrics)
  images-*.parquet     one row per image (status, counts, seconds)
Masks are stored as COCO RLE relative to a window (win_x1, win_y1, mask_w, mask_h) of the image.
"""
from __future__ import annotations

import argparse
import time
import traceback

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import config, measure, models

Image.MAX_IMAGE_PIXELS = None

PROMPT = "a beetle."
BOX_THR, TEXT_THR = 0.25, 0.2
PARTS_PAD = 0.06  # BeetleFlow was trained on detector-box crops, so keep its crop tight
MAX_BOX_FRAC_TRAY = 0.20  # a "beetle" box covering more of a tray than this is the tray itself
TILE = 1856  # tile side for the crowded-tray pass (a third of a 5568 px tray photo)
TILED_GAIN, TILED_MARGIN = 1.25, 3  # tiles win only with > 25% + 3 more specimens than the whole photo


def _open(path) -> Image.Image:
    return Image.open(config.ROOT / path).convert("RGB")


def _blob_box(image: Image.Image) -> np.ndarray | None:
    """Fallback prompt for a single specimen: the largest blob that differs from the backdrop.

    The backdrop colour is taken from the image border; this is the pilot's "dark-blob"
    idea generalised to coloured backdrops.
    """
    small = image.copy()
    small.thumbnail((512, 512))
    a = np.asarray(small).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    dist = np.abs(a - np.median(border, axis=0)).sum(axis=2).astype(np.float32)
    thr = max(40.0, float(np.percentile(dist, 75)))
    m = cv2.morphologyEx((dist > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[k]
    if area < 0.002 * m.size:
        return None
    s = image.size[0] / small.size[0]
    return np.array([x * s, y * s, (x + w) * s, (y + h) * s], np.float32)


BACKDROP_BORDER = 0.5  # a mask covering more than half of its window's border is the backdrop


def border_coverage(mask: np.ndarray) -> float:
    """Fraction of the mask window's outline that the mask covers."""
    return float(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean())


def fix_backdrop_mask(mask: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Turn a backdrop mask into a specimen mask.

    When the prompt box is the whole frame (a tight crop of one specimen on a plain backdrop),
    SAM often returns the backdrop instead of the specimen: about half of the Biorepository's
    2016 individual photos did. The backdrop wraps the window's border, which a specimen never
    does, so such a mask is replaced by the largest blob of its complement.
    """
    if border_coverage(mask) <= BACKDROP_BORDER:
        return mask, None
    inv = measure.largest_component(~mask)
    if inv.sum() < 0.01 * mask.size:
        return mask, "backdrop_unfixed"
    return inv, "inverted_backdrop"


def _parts_window(box, width: int, height: int) -> tuple[int, int, int, int]:
    return models.Sam2.crop_window(box, width, height, pad=PARTS_PAD, min_side=32)


class Pipeline:
    def __init__(self, routes: set[str], parts: bool = True, sam_id: str = models.SAM2_ID):
        self.gd = models.GroundingDino() if routes & {"tray", "single"} else None
        self.sam = models.Sam2(sam_id)
        self.bf = models.BeetleFlow() if parts else None
        self.model_ids = dict(detector=models.GDINO_ID if self.gd else None, segmenter=sam_id,
                              parts_model=self.bf.model_id if self.bf else None)

    # ---- per-route detection -------------------------------------------------------------
    @staticmethod
    def _clean_tray(det: models.Detections, w: int, h: int) -> models.Detections:
        if not len(det):
            return det
        area = (det.boxes[:, 2] - det.boxes[:, 0]) * (det.boxes[:, 3] - det.boxes[:, 1])
        return models.nms(det.select(area < MAX_BOX_FRAC_TRAY * w * h), 0.5, 0.8)

    def boxes_tray(self, image: Image.Image, tiled: bool = True) -> tuple[np.ndarray, np.ndarray, str]:
        """Whole-photo detection, with a tiled pass as a safety net for crowded trays.

        On an ordinary tray the whole photo is the better input (precision 0.985 vs 0.935 for
        tiles, same recall). On a tray of a hundred or more small beetles it collapses: one
        tray of 266 gave a single box. So both are run, and the tiled result is used only
        when it finds clearly more specimens than the whole-photo pass.
        """
        w, h = image.size
        whole = self._clean_tray(self.gd.detect(image, PROMPT, BOX_THR, TEXT_THR), w, h)
        if not tiled:
            return whole.boxes, whole.scores, "grounding_dino"
        tiles = self._clean_tray(self.gd.detect(image, PROMPT, BOX_THR, TEXT_THR, tile=TILE), w, h)
        if len(tiles) >= TILED_GAIN * len(whole) + TILED_MARGIN:
            det = self._clean_tray(models.Detections.concat([whole, tiles]), w, h)
            return det.boxes, det.scores, "grounding_dino_tiled"
        return whole.boxes, whole.scores, "grounding_dino"

    def boxes_single(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, str]:
        det = self.gd.detect(image, PROMPT, BOX_THR, TEXT_THR)
        w, h = image.size
        if len(det):
            area = (det.boxes[:, 2] - det.boxes[:, 0]) * (det.boxes[:, 3] - det.boxes[:, 1])
            det = det.select(area >= 0.005 * w * h)
        if len(det):
            # one specimen per photo: the most confident box, larger boxes winning near-ties
            area = (det.boxes[:, 2] - det.boxes[:, 0]) * (det.boxes[:, 3] - det.boxes[:, 1])
            best = int(np.argmax(det.scores + 0.1 * area / (w * h)))
            return det.boxes[[best]], det.scores[[best]], "grounding_dino"
        blob = _blob_box(image)
        if blob is not None:
            return blob[None], np.array([np.nan], np.float32), "backdrop_blob"
        return np.array([[1, 1, w - 1, h - 1]], np.float32), np.array([np.nan], np.float32), "full_frame"

    # ---- shared tail: SAM mask + parts + metrics ---------------------------------------------
    def finish(self, image: Image.Image, boxes: np.ndarray, scores: np.ndarray, det_source: str,
               part_maps: list[np.ndarray] | None = None, part_windows: list | None = None) -> list[dict]:
        if len(boxes) == 0:
            return []
        w, h = image.size
        sam_out = self.sam.segment_boxes(image, boxes)
        if self.bf is not None and part_maps is None:
            part_windows = [_parts_window(b, w, h) for b in boxes]
            part_maps = []
            for i in range(0, len(boxes), 8):
                part_maps += self.bf.predict([image.crop(win) for win in part_windows[i:i + 8]])
        rows = []
        for k, (box, score, (win, mask, sam_score)) in enumerate(zip(boxes, scores, sam_out)):
            mask, mask_fix = fix_backdrop_mask(mask)
            rgb = np.asarray(image.crop(win))
            row = dict(box_x1=float(box[0]), box_y1=float(box[1]), box_x2=float(box[2]), box_y2=float(box[3]),
                       det_score=None if np.isnan(score) else float(score), det_source=det_source,
                       sam_score=sam_score, win_x1=win[0], win_y1=win[1], mask_w=mask.shape[1],
                       mask_h=mask.shape[0], mask_rle=measure.rle_encode(mask), mask_fix=mask_fix)
            row.update(measure.body_metrics(mask, rgb))
            # a mask running along the frame edge usually means a cut-off specimen or a bad prompt
            ys, xs = np.nonzero(mask)
            row["touches_edge"] = bool(len(xs) and (xs.min() + win[0] <= 1 or ys.min() + win[1] <= 1
                                                   or xs.max() + win[0] >= w - 2 or ys.max() + win[1] >= h - 2))
            if part_maps is not None:
                pwin, labels = part_windows[k], part_maps[k]
                # restrict part labels to this specimen's SAM mask (matters on crowded trays)
                body = np.zeros(labels.shape, bool)
                ox, oy = win[0] - pwin[0], win[1] - pwin[1]
                y1, x1 = max(oy, 0), max(ox, 0)
                y2, x2 = min(oy + mask.shape[0], labels.shape[0]), min(ox + mask.shape[1], labels.shape[1])
                if y2 > y1 and x2 > x1:
                    body[y1:y2, x1:x2] = mask[y1 - oy:y2 - oy, x1 - ox:x2 - ox]
                row.update(measure.part_metrics(labels, body))
                row.update(pwin_x1=pwin[0], pwin_y1=pwin[1], parts_w=labels.shape[1], parts_h=labels.shape[0])
                for name in ("head", "pronotum", "elytra"):
                    row[f"{name}_rle"] = measure.rle_encode((labels == models.PART_CLASSES.index(name)) & body)
            rows.append(row)
        return rows

    def run_crop(self, image: Image.Image) -> list[dict]:
        """Tight single-specimen crop: part labels first, then SAM prompted by their extent."""
        w, h = image.size
        labels = self.bf.predict([image])[0]
        fg = measure.largest_component(labels > 0, fill_holes=False) if (labels > 0).any() else None
        trunk = [(labels == models.PART_CLASSES.index(n)).sum() for n in ("head", "pronotum", "elytra")]
        # only trust the part labels as a prompt when they found a whole beetle; when the head or
        # pronotum is missed, a box around the rest would make SAM segment the elytra alone
        if fg is not None and fg.sum() >= 0.01 * w * h and min(trunk) >= 0.002 * w * h:
            ys, xs = np.nonzero(fg)
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            source = "beetleflow_extent"
        else:
            box, source = np.array([1, 1, w - 1, h - 1], np.float32), "full_frame"
        return self.finish(image, box[None], np.array([np.nan], np.float32), source,
                           part_maps=[labels], part_windows=[(0, 0, w, h)])

    def run_image(self, row) -> list[dict]:
        image = _open(row.local_path)
        if row.route == "crop":
            return self.run_crop(image)
        if row.route == "tray":
            # the tiled pass is for ethanol trays; pinned trays hold a handful of spaced specimens
            boxes, scores, source = self.boxes_tray(image, tiled=row.image_kind == "tray_ethanol")
        else:
            boxes, scores, source = self.boxes_single(image)
        return self.finish(image, boxes, scores, source)


def _done_ids(shard_dir) -> set[str]:
    ids: set[str] = set()
    for f in shard_dir.glob("images-*.parquet"):
        ids |= set(pd.read_parquet(f, columns=["image_id"])["image_id"])
    return ids


def run(source: str, limit: int | None = None, flush_every: int = 200, parts: bool = True,
        sam_id: str = models.SAM2_ID) -> None:
    config.ensure_dirs()
    config.limit_threads()
    torch.set_num_threads(config.N_THREADS)
    man = pd.read_parquet(config.TABLES / "image_manifest.parquet")
    man = man[(man["source"] == source) & man["duplicate_of"].isna() & man["is_carabid"]]
    shard_dir = config.OUT / "shards" / source
    shard_dir.mkdir(parents=True, exist_ok=True)
    todo = man[~man["image_id"].isin(_done_ids(shard_dir))]
    if limit:
        todo = todo.head(limit)
    print(f"{source}: {len(man):,} images, {len(todo):,} to do", flush=True)
    if todo.empty:
        return
    pipe = Pipeline(set(todo["route"]), parts=parts, sam_id=sam_id)
    inst_rows, img_rows, t0, n = [], [], time.time(), 0

    def flush():
        nonlocal inst_rows, img_rows
        if not img_rows:
            return
        tag = f"{int(time.time() * 1000)}"
        pd.DataFrame(img_rows).to_parquet(shard_dir / f"images-{tag}.parquet", index=False)
        if inst_rows:
            pd.DataFrame(inst_rows).to_parquet(shard_dir / f"instances-{tag}.parquet", index=False)
        inst_rows, img_rows = [], []

    rows = list(todo.itertuples())
    for row in rows:
        t = time.time()
        status, found = "ok", []
        try:
            found = pipe.run_image(row)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            status = "error:cuda_oom"
        except Exception as e:  # noqa: BLE001 - one bad image must not stop a multi-hour run
            status = f"error:{type(e).__name__}"
            if n < 5:
                traceback.print_exc()
        for k, r in enumerate(found, 1):
            inst_rows.append(dict(instance_id=f"{row.image_id}#{k:03d}", image_id=row.image_id, instance=k,
                                  **r, **pipe.model_ids))
        img_rows.append(dict(image_id=row.image_id, status=status, n_instances=len(found),
                             seconds=round(time.time() - t, 3)))
        n += 1
        if n % flush_every == 0:
            flush()
            rate = n / (time.time() - t0)
            print(f"  {n:,}/{len(rows):,}  {rate:.2f} img/s  eta {(len(rows) - n) / rate / 60:.0f} min", flush=True)
    flush()
    print(f"{source}: done {n:,} images in {(time.time() - t0) / 60:.1f} min", flush=True)


def drop_from_shards(source: str, image_ids: list[str]) -> None:
    """Remove images from a pool's shards so the next `run` processes them again."""
    ids = set(image_ids)
    for f in sorted((config.OUT / "shards" / source).glob("*.parquet")):
        df = pd.read_parquet(f)
        keep = df[~df["image_id"].isin(ids)]
        if len(keep) != len(df):
            keep.to_parquet(f, index=False) if len(keep) else f.unlink()


def recheck_backdrop(source: str) -> list[str]:
    """Find specimens whose stored mask is the backdrop (see fix_backdrop_mask) and queue their
    images for reprocessing. For results made before that fix existed."""
    redo: set[str] = set()
    for f in sorted((config.OUT / "shards" / source).glob("instances-*.parquet")):
        df = pd.read_parquet(f, columns=["image_id", "mask_rle", "mask_h", "mask_w"])
        for r in df.itertuples():
            if border_coverage(measure.rle_decode(r.mask_rle, r.mask_h, r.mask_w)) > BACKDROP_BORDER:
                redo.add(r.image_id)
    drop_from_shards(source, sorted(redo))
    print(f"{source}: {len(redo)} images to redo (backdrop masks)", flush=True)
    return sorted(redo)


def recheck_trays(source: str = "hf2018") -> list[str]:
    """Re-run detection on trays processed before the tiled pass existed.

    Trays where the tiled pass now wins are removed from the shards, so the next `run` redoes
    them. Returns their image ids.
    """
    config.ensure_dirs()
    shard_dir = config.OUT / "shards" / source
    man = pd.read_parquet(config.TABLES / "image_manifest.parquet").set_index("image_id")
    done = sorted(_done_ids(shard_dir))
    gd = Pipeline.__new__(Pipeline)
    gd.gd = models.GroundingDino()
    redo, t0 = [], time.time()
    for i, image_id in enumerate(done, 1):
        row = man.loc[image_id]
        if row["image_kind"] != "tray_ethanol":
            continue
        _, _, how = gd.boxes_tray(_open(row["local_path"]))
        if how == "grounding_dino_tiled":
            redo.append(image_id)
        if i % 50 == 0:
            print(f"  rechecked {i}/{len(done)}: {len(redo)} to redo, {(time.time() - t0) / i:.1f} s/img", flush=True)
    drop_from_shards(source, redo)
    print(f"{source}: {len(redo)} trays to redo with tiled detection", flush=True)
    return redo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", choices=["hf2018", "hawaii", "biorepo", "sentinel"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-parts", action="store_true")
    ap.add_argument("--sam", default=models.SAM2_ID, help="SAM 2.1 checkpoint (default: %(default)s)")
    ap.add_argument("--recheck-trays", action="store_true",
                    help="first re-run tray detection with the tiled pass and redo trays where it wins")
    ap.add_argument("--recheck-backdrop", action="store_true",
                    help="first queue images whose stored mask is the backdrop for reprocessing")
    args = ap.parse_args()
    if args.recheck_trays:
        recheck_trays(args.source)
    if args.recheck_backdrop:
        recheck_backdrop(args.source)
    run(args.source, args.limit, parts=not args.no_parts, sam_id=args.sam)


if __name__ == "__main__":
    main()
