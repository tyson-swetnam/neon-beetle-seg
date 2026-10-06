"""Experiment: check the ruler-tick reader against Hawaii's human scale-bar annotations."""
import pandas as pd
from PIL import Image

from neon_beetle_seg import config, scale

Image.MAX_IMAGE_PIXELS = None
tr = pd.read_parquet(config.TABLES / "trait_annotations_hawaii.parquet")
one = tr.drop_duplicates("image_id").sample(40, random_state=0)
rows = []
for r in one.itertuples():
    im = Image.open(config.HF_DIR / "Hawaii-beetles/group_images" / (r.image_id.split("/")[1] + ".png")).convert("RGB")
    x1, x2 = sorted([r.scalebar_x1, r.scalebar_x2]); y1, y2 = sorted([r.scalebar_y1, r.scalebar_y2])
    L = r.px_scalebar; pad = 0.7 * L
    crop = im.crop((int(max(0, x1 - pad)), int(max(0, y1 - 0.7 * L)), int(min(im.size[0], x2 + pad)), int(min(im.size[1], y2 + 0.7 * L))))
    res = scale.ruler_px_per_mm(crop)
    rows.append(dict(image_id=r.image_id, truth=L / (r.cm_scalebar * 10), **res))
d = pd.DataFrame(rows); d["ratio"] = d.px_per_mm / d.truth
print(d[["truth", "px_per_mm", "ratio", "confidence", "orientation", "n_ticks"]].describe().round(3).to_string())
print("read:", d.px_per_mm.notna().sum(), "/", len(d), " within 3%:", ((d.ratio - 1).abs() < 0.03).sum(), " ratio~0.5:", ((d.ratio - 0.5).abs() < 0.03).sum(), " ratio~2:", ((d.ratio - 2).abs() < 0.06).sum())
print(d.sort_values("ratio").round(3).head(6).to_string()); crop.save(config.OUT / "peek/hawaii_ruler.png")
# sentinel scale bars
man = pd.read_parquet(config.TABLES / "image_manifest.parquet"); sb = man[man.source == "sentinel"].scalebar_path.drop_duplicates().sample(60, random_state=0)
out = [scale.ruler_px_per_mm(Image.open(config.ROOT / p)) | {"size": Image.open(config.ROOT / p).size} for p in sb]
s = pd.DataFrame(out); print(s.describe().round(3).to_string()); print("sentinel read", s.px_per_mm.notna().sum(), "/", len(s)); print(s.head(12).to_string())
