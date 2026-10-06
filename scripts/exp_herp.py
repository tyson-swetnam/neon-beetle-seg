"""Experiment: herp route on sample photos, with overlays and the ruler scale."""
import time, cv2, numpy as np, pandas as pd, torch
from PIL import Image
from neon_beetle_seg import config, measure, scale, segment
man = pd.read_parquet(config.TABLES / "image_manifest.parquet")
h = man[(man.source == "herp") & man.process]
multi = h[pd.to_numeric(h.individualCount, errors="coerce") > 2].sample(4, random_state=1)
rows = pd.concat([h.sample(12, random_state=2), multi])
pipe = segment.Pipeline({"herp"})
tiles, secs = [], []
for r in rows.itertuples():
    t = time.time(); found = pipe.run_image(r); secs.append(time.time() - t)
    im = segment._open(r.local_path); w, hh = im.size
    sc = scale.ruler_px_per_mm(im.crop((0, int(0.62 * hh), int(0.7 * w), hh)), min_ticks=20)
    img = np.asarray(im).copy()
    for f in found:
        m = measure.rle_decode(f["mask_rle"], f["mask_h"], f["mask_w"]); sub = img[f["win_y1"]:f["win_y1"] + m.shape[0], f["win_x1"]:f["win_x1"] + m.shape[1]]
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(sub, cnts, -1, (255, 60, 0), max(2, w // 500))
    ppm = sc["px_per_mm"]
    lens = [round(f["midline_length_px"] / ppm, 1) if ppm and f.get("midline_length_px") else None for f in found]
    print(f"{r.image_id.split('/')[-1]:32s} {str(r.scientificName)[:26]:26s} n_rec {r.individualCount} found {len(found)} px/mm {ppm and round(ppm, 2)} conf {round(sc['confidence'], 2)} ticks {sc['n_ticks']} midline_mm {lens} {secs[-1]:.1f}s", flush=True)
    s = 420 / w; tiles.append(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
H = max(t.shape[0] for t in tiles); sheet = np.full((H * 4 + 12, 420 * 4 + 12, 3), 255, np.uint8)
for k, t in enumerate(tiles): sheet[(k // 4) * (H + 3):(k // 4) * (H + 3) + t.shape[0], (k % 4) * 423:(k % 4) * 423 + t.shape[1]] = t
cv2.imwrite(str(config.OUT / "peek/herp_seg.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80]); print(sheet.shape, "mean s/img", round(np.mean(secs[1:]), 2))
