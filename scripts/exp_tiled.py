"""Experiment: does adding a tiled pass rescue crowded trays without hurting the rest?"""
import time, numpy as np, pandas as pd
from PIL import Image
from neon_beetle_seg import config, models
from neon_beetle_seg.validate import _match
Image.MAX_IMAGE_PIXELS = None
det = pd.read_parquet(config.TABLES / "validation_detection.parquet")
inst = pd.read_parquet(config.TABLES / "instances.parquet", columns=["image_id", "box_x1", "box_y1", "box_x2", "box_y2"])
inst["side"] = np.maximum(inst.box_x2 - inst.box_x1, inst.box_y2 - inst.box_y1)
per = inst[inst.image_id.str.startswith("hf2018")].groupby("image_id").agg(n=("side", "size"), med=("side", "median"))
trig = per[(per.n >= 50) | (per.med < 0.04 * 5568)]
print("trays triggering (n>=50 or median side<223px):", len(trig), "of", len(per))
gt = pd.read_parquet(config.TABLES / "tray_boxes_2018.parquet"); man = pd.read_parquet(config.TABLES / "image_manifest.parquet").set_index("image_id")
worst = det.assign(miss=det.n_human_boxes - det.tp_iou50).sort_values("miss", ascending=False).image_id.head(6).tolist()
rest = det[~det.image_id.isin(worst)].sample(14, random_state=0).image_id.tolist()
gd = models.GroundingDino()
def clean(d, w, h):
    a = (d.boxes[:, 2] - d.boxes[:, 0]) * (d.boxes[:, 3] - d.boxes[:, 1]); return models.nms(d.select(a < 0.2 * w * h), 0.5, 0.8)
for name, ids in (("worst 6", worst), ("random 14", rest)):
    tot = {k: [0, 0, 0] for k in ("whole", "tiled", "union")}; secs = 0
    for iid in ids:
        im = Image.open(config.ROOT / man.loc[iid, "local_path"]).convert("RGB"); w, h = im.size
        g = gt.loc[gt.image_id == iid, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        a = clean(gd.detect(im), w, h); t = time.time(); b = clean(gd.detect(im, tile=1856), w, h); secs += time.time() - t
        u = clean(models.Detections.concat([a, b]), w, h)
        for k, d in (("whole", a), ("tiled", b), ("union", u)):
            tot[k][0] += _match(d.boxes, g); tot[k][1] += len(d); tot[k][2] += len(g)
        if name.startswith("worst"): print("  ", iid, "human", len(g), "whole", len(a), "tiled", len(b), "union", len(u), "triggers" if iid in trig.index else "no trigger")
    print(name, {k: f"P {tp/max(n,1):.3f} R {tp/max(g_,1):.3f} n {n}" for k, (tp, n, g_) in tot.items()}, f"tiled {secs/len(ids):.1f} s/img")
