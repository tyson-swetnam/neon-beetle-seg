"""Pixel-to-millimetre scale for each image.

Hawaii trays use the human scale-bar annotation from the image manifest. This module reads the
scale from the image itself for the rest:

  checkerboard  the 1 cm checkerboard in the 2018 tray photos: the side of its white squares.
  ruler ticks   a millimetre ruler crop (sentinel-beetles ships one per source photo): the tick
                spacing is the dominant period of the dark-stroke profile.
  printed bar   Biorepository macro photos carry a thin printed bar labelled "1 mm" / "5 mm":
                the bar is found as a long thin horizontal line and its label is read with OCR.

Every value comes with a confidence so unreliable reads can be left in pixels.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image


# ---- millimetre ruler ----------------------------------------------------------------------
def _period(profile: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Dominant period (in samples) of a 1-D profile and its normalised autocorrelation strength."""
    x = profile - profile.mean()
    if not x.any():
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    hi = min(hi, len(ac) - 2)
    if hi <= lo:
        return 0.0, 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    if not (ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]):
        return 0.0, 0.0
    # parabolic refinement of the peak position
    d = ac[k - 1] - 2 * ac[k] + ac[k + 1]
    shift = 0.5 * (ac[k - 1] - ac[k + 1]) / d if d != 0 else 0.0
    return float(k + shift), float(ac[k])


def ruler_px_per_mm(image: Image.Image | np.ndarray, min_ticks: int = 6) -> dict:
    """Tick spacing of a millimetre ruler crop. Returns px_per_mm, confidence, orientation, n_ticks.

    The ruler may run horizontally or vertically. For each orientation the crop is cut into
    bands across the ticks; the band whose dark-stroke profile is most strongly periodic wins.
    The spacing is then re-measured over the longest run of evenly spaced tick centres.
    """
    from scipy.signal import find_peaks

    g = np.asarray(image.convert("L") if isinstance(image, Image.Image) else image).astype(np.float32)
    best = dict(px_per_mm=None, confidence=0.0, orientation=None, n_ticks=0)
    for orient, arr in (("horizontal", g), ("vertical", g.T)):
        h, w = arr.shape
        if w < 40 or h < 12:
            continue
        dark = 255.0 - cv2.GaussianBlur(arr, (0, 0), 1.0)
        band = max(8, h // 8)
        for y in range(0, h - band + 1, max(4, band // 2)):
            prof = dark[y:y + band].mean(axis=0)
            prof = prof - cv2.blur(prof[None, :], (max(9, w // 6) | 1, 1))[0]  # remove slow shading
            period, strength = _period(prof, lo=4, hi=w // min_ticks)
            if period <= 0 or strength <= best["confidence"]:
                continue
            peaks, _ = find_peaks(prof, distance=max(2, int(period * 0.6)), prominence=prof.std() * 0.8)
            if len(peaks) < min_ticks:
                continue
            # longest run of evenly spaced ticks; its end-to-end span gives a sub-pixel spacing
            ok = np.abs(np.diff(peaks) - period) < 0.3 * period
            run_best, start = (0, 0), None
            for i, good in enumerate(list(ok) + [False]):
                if good and start is None:
                    start = i
                elif not good and start is not None:
                    if i - start > run_best[1] - run_best[0]:
                        run_best = (start, i)
                    start = None
            n_gaps = run_best[1] - run_best[0]
            if n_gaps < min_ticks - 1:
                continue
            spacing = (peaks[run_best[1]] - peaks[run_best[0]]) / n_gaps
            best = dict(px_per_mm=float(spacing), confidence=float(strength), orientation=orient,
                        n_ticks=int(n_gaps + 1))
    return best


# ---- printed scale bar on Biorepository macro photos --------------------------------------
def find_printed_bar(image: Image.Image) -> dict | None:
    """Locate a long, thin, dark horizontal bar on a light backdrop. Returns its pixel box."""
    g = np.asarray(image.convert("L"))
    h, w = g.shape
    dark = (g < min(110, np.percentile(g, 50) - 60)).astype(np.uint8)
    # a bar is at least 4% of the image width and only a few pixels thick
    k = max(25, int(0.04 * w))
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh > max(8, 0.006 * h) or bw < k or bw > 0.6 * w:
            continue
        # the bar stands alone: the rows just above and below it are backdrop
        pad = max(6, 3 * bh)
        above = dark[max(0, y - pad):max(0, y - 2), x:x + bw]
        below = dark[min(h, y + bh + 2):min(h, y + bh + pad), x:x + bw]
        if above.size and above.mean() > 0.08:
            continue
        if below.size and below[: max(1, pad // 3)].mean() > 0.08:
            continue
        if best is None or bw > best["x2"] - best["x1"]:
            best = dict(x1=int(x), y1=int(y), x2=int(x + bw), y2=int(y + bh))
    return best


_LABEL_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm|µm|um)", re.I)


def parse_label(text: str) -> float | None:
    """Millimetres from an OCR'd scale label such as '5 mm', '1mm', '0.5 mm' or '1 cm'."""
    m = _LABEL_RE.search(text.replace("rnm", "mm").replace("rn", "m"))
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    return value * {"mm": 1.0, "cm": 10.0, "µm": 0.001, "um": 0.001}[unit]


class LabelReader:
    """OCR for the short printed label next to a scale bar (TrOCR, printed text)."""

    def __init__(self, model_id: str = "microsoft/trocr-base-printed"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        from .models import device

        self.model_id = model_id
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device()).eval()
        self.device = device()

    def read(self, crop: Image.Image) -> str:
        import torch

        with torch.inference_mode():
            pixel = self.processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
            ids = self.model.generate(pixel, max_new_tokens=12)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def label_crops(image: Image.Image, bar: dict) -> list[Image.Image]:
    """Candidate regions for the bar's label: under its right end, under its centre, above it."""
    w, h = image.size
    length = bar["x2"] - bar["x1"]
    th = max(24, int(0.035 * h))
    tw = max(80, int(0.5 * length))
    out = []
    for cx, y1, y2 in (
        (bar["x2"] - tw // 3, bar["y2"] + 2, bar["y2"] + 2 + th),
        ((bar["x1"] + bar["x2"]) // 2, bar["y2"] + 2, bar["y2"] + 2 + th),
        ((bar["x1"] + bar["x2"]) // 2, bar["y1"] - 2 - th, bar["y1"] - 2),
        (bar["x1"] + tw // 3, bar["y2"] + 2, bar["y2"] + 2 + th),
    ):
        x1, x2 = max(0, cx - tw // 2), min(w, cx + tw // 2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 > 20 and y2 - y1 > 10:
            out.append(image.crop((x1, y1, x2, y2)))
    return out


def printed_bar_px_per_mm(image: Image.Image, reader: LabelReader) -> dict:
    """Scale from a printed bar and its label. px_per_mm is None when either part is not found."""
    bar = find_printed_bar(image)
    if bar is None:
        return dict(px_per_mm=None, bar_px=None, label_text=None, label_mm=None)
    for crop in label_crops(image, bar):
        text = reader.read(crop)
        mm = parse_label(text)
        if mm:
            length = bar["x2"] - bar["x1"]
            return dict(px_per_mm=length / mm, bar_px=length, label_text=text, label_mm=mm, **bar)
    return dict(px_per_mm=None, bar_px=bar["x2"] - bar["x1"], label_text=None, label_mm=None, **bar)


# ---- checkerboard scale on the 2018 tray photos ----------------------------------------------
def checkerboard_px_per_mm(image: Image.Image, work_side: int = 1856) -> dict:
    """Side length of the white squares of the 1 cm checkerboard scale in the 2018 tray photos.

    The board has a black outline, so its black squares merge with it, but each white square
    is enclosed by black: a bright, square, well-filled blob the same size as its neighbours,
    bordered by dark pixels on all four sides. Returns px_per_mm in original-image pixels, the
    number of squares used and their relative spread.
    """
    w0, h0 = image.size
    s = work_side / max(w0, h0)
    small = image.convert("L").resize((round(w0 * s), round(h0 * s)), Image.BILINEAR) if s < 1 else image.convert("L")
    s = small.size[0] / w0
    g = np.asarray(small)
    dark = g < 90
    bright = cv2.erode((g > 150).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
    sides = []
    H, W = g.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 0.02 * W or bw > 0.2 * W:
            continue
        if abs(bw - bh) > 0.12 * max(bw, bh) or area < 0.85 * bw * bh:
            continue
        # black just outside each of the four sides
        m = max(3, bw // 8)
        cx, cy = x + bw // 2, y + bh // 2
        probes = [dark[cy, max(0, x - m)], dark[cy, min(W - 1, x + bw + m)],
                  dark[max(0, y - m), cx], dark[min(H - 1, y + bh + m), cx]]
        if sum(probes) < 3:  # top- and bottom-row squares face the thin outline on one side
            continue
        sides.append((bw + bh) / 2 + 2)  # +2 undoes the 3x3 erosion
    if len(sides) < 2:
        return dict(px_per_mm=None, n_squares=len(sides), spread=None)
    sides = np.array(sides)
    med = np.median(sides)
    good = sides[np.abs(sides - med) < 0.08 * med]
    if len(good) < 2:
        return dict(px_per_mm=None, n_squares=int(len(good)), spread=None)
    return dict(px_per_mm=float(np.median(good) / s / 10.0), n_squares=int(len(good)),
                spread=float(good.std() / good.mean()))


# ---- per-image scale table -------------------------------------------------------------------
MIN_TICK_CONFIDENCE = 0.4  # autocorrelation strength below this is not a clean ruler
MIN_TICKS = 8
MIN_BAR_FRACTION = 0.15  # printed bars span 22-45% of the frame; shorter "bars" are other lines


MAX_SQUARE_SPREAD = 0.03  # checkerboard squares must agree with each other to 3%
POOLS = ("hf2018", "hawaii", "sentinel", "biorepo")


def build(pools: tuple[str, ...] | None = None):
    """Write data/tables/image_scale.parquet: one row per image with px_per_mm and how it was obtained.

    scale_source values:
      checkerboard      the 1 cm checkerboard in each 2018 tray photo, measured from the image
      hawaii_scalebar   human-drawn 1 cm bar (Hawaii trays)
      ruler_ticks       millimetre ruler crop read by ruler_px_per_mm (sentinel)
      printed_bar       printed bar + OCR label (Biorepository macro photos)
    Images with no trusted scale keep px_per_mm null and their measurements stay in pixels.

    The 2018 trays do not use the volunteers' scale-bar annotation: for two of the five Zooniverse
    workflows (21652, 21827) the annotation coordinates are in a smaller pixel frame than the
    dataset's resized_image_dim column says, which would make the scale 10-22% too small. The
    annotated value is kept in scale_detail for comparison.

    `pools` limits the work to some pools; rows for the others are kept from the existing table.
    """
    import pandas as pd
    from tqdm import tqdm

    from . import config

    Image.MAX_IMAGE_PIXELS = None
    pools = tuple(pools or POOLS)
    man = pd.read_parquet(config.TABLES / "image_manifest.parquet")
    man = man[man["duplicate_of"].isna() & man["is_carabid"]]
    path = config.TABLES / "image_scale.parquet"
    rows = []
    if path.exists():
        old = pd.read_parquet(path)
        rows += old[~old["image_id"].str.split("/").str[0].isin(pools)].to_dict("records")

    if "hawaii" in pools:
        for r in man[man["source"] == "hawaii"].itertuples():
            rows.append(dict(image_id=r.image_id, px_per_mm=r.px_per_mm, scale_source=r.scale_source,
                             scale_confidence=1.0, scale_detail=None))

    if "hf2018" in pools:
        for r in tqdm(list(man[man["source"] == "hf2018"].itertuples()), desc="checkerboards", mininterval=30):
            res = checkerboard_px_per_mm(Image.open(config.ROOT / r.local_path))
            ok = bool(res["px_per_mm"]) and res["n_squares"] >= 2 and res["spread"] <= MAX_SQUARE_SPREAD
            ann = "" if pd.isna(r.px_per_mm) else f";annotated_px_per_mm={r.px_per_mm:.3f}"
            rows.append(dict(image_id=r.image_id, px_per_mm=res["px_per_mm"] if ok else None,
                             scale_source="checkerboard" if ok else None,
                             scale_confidence=1.0 - (res["spread"] or 0) if ok else 0.0,
                             scale_detail=f"squares={res['n_squares']}{ann}"))

    if "sentinel" in pools:
        sent = man[man["source"] == "sentinel"]
        by_bar = {}
        for bar in tqdm(sorted(sent["scalebar_path"].dropna().unique()), desc="sentinel rulers", mininterval=30):
            by_bar[bar] = ruler_px_per_mm(Image.open(config.ROOT / bar))
        for r in sent.itertuples():
            res = by_bar.get(r.scalebar_path, {})
            ok = bool(res.get("px_per_mm")) and res["confidence"] >= MIN_TICK_CONFIDENCE and res["n_ticks"] >= MIN_TICKS
            rows.append(dict(image_id=r.image_id, px_per_mm=res["px_per_mm"] if ok else None,
                             scale_source="ruler_ticks" if ok else None, scale_confidence=res.get("confidence"),
                             scale_detail=f"ticks={res.get('n_ticks')};{res.get('orientation')}"))

    if "biorepo" in pools:
        pinned = man[(man["source"] == "biorepo") & (man["image_kind"] == "pinned")]
        reader = LabelReader()
        for r in tqdm(list(pinned.itertuples()), desc="printed bars", mininterval=30):
            im = Image.open(config.ROOT / r.local_path).convert("RGB")
            res = printed_bar_px_per_mm(im, reader)
            frac = (res["bar_px"] or 0) / im.size[0]
            ok = bool(res["px_per_mm"]) and frac >= MIN_BAR_FRACTION
            rows.append(dict(image_id=r.image_id, px_per_mm=res["px_per_mm"] if ok else None,
                             scale_source="printed_bar" if ok else None, scale_confidence=1.0 if ok else 0.0,
                             scale_detail=f"bar_px={res['bar_px']};label={res['label_text']}"))
    out = pd.DataFrame(rows)
    out.to_parquet(path, index=False)
    return out


def main() -> None:
    import sys

    out = build(tuple(sys.argv[1:]) or None)
    src = out.assign(pool=out["image_id"].str.split("/").str[0], has=out["px_per_mm"].notna())
    print(src.groupby("pool").agg(images=("image_id", "size"), with_scale=("has", "sum"),
                                  median_px_per_mm=("px_per_mm", "median")).to_string())


if __name__ == "__main__":
    main()
