"""Experiment: compare detectors against the 577-tray human boxes on a sample of trays."""
import sys, time, random
import xml.etree.ElementTree as ET
import numpy as np, torch
from PIL import Image
from neon_beetle_seg import config, models

Image.MAX_IMAGE_PIXELS = None
root = ET.parse(config.DATA / "external/carabidae_beetle_processing/annotations/2018_neon_beetles_bbox.xml").getroot()
gt = {im.get("name").split("/")[-1]: np.array([[float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")] for b in im.findall("box")], np.float32).reshape(-1, 4)
      for im in root.findall("image")}
random.seed(0)
names = random.sample(sorted(gt), int(sys.argv[1]) if len(sys.argv) > 1 else 16)

def score(det_boxes, g, thr=0.5):
    iou = models.box_iou(det_boxes, g)
    tp = 0; used = set()
    for i in np.argsort(-iou.max(1) if len(g) and len(det_boxes) else []):
        j = int(iou[i].argmax())
        if iou[i, j] >= thr and j not in used:
            used.add(j); tp += 1
    return tp, len(det_boxes), len(g)

def drop_huge(d, w, h, frac=0.2):
    a = (d.boxes[:, 2] - d.boxes[:, 0]) * (d.boxes[:, 3] - d.boxes[:, 1])
    return d.select(a < frac * w * h)

gd = models.GroundingDino(); yo = models.YoloBeetle()
runs = {
    "yolo@2048 c.15": lambda im: yo.detect(im, conf=0.15, imgsz=2048),
    "yolo@2048 c.25": lambda im: yo.detect(im, conf=0.25, imgsz=2048),
    "yolo@3008 c.25": lambda im: yo.detect(im, conf=0.25, imgsz=3008),
    "gdino whole .25": lambda im: gd.detect(im, "a beetle.", 0.25, 0.2),
    "gdino tile1856 .25": lambda im: gd.detect(im, "a beetle.", 0.25, 0.2, tile=1856),
    "gdino tile1856 .30": lambda im: gd.detect(im, "a beetle.", 0.30, 0.2, tile=1856),
    "gdino tile1400 .30": lambda im: gd.detect(im, "a beetle.", 0.30, 0.2, tile=1400),
}
tot = {k: [0, 0, 0, 0.0] for k in runs}
for n in names:
    im = Image.open(config.HF_DIR / "2018-NEON-beetles/group_images" / n).convert("RGB")
    for k, f in runs.items():
        t = time.time(); d = f(im); dt = time.time() - t
        if k.startswith("yolo"):
            d = d.select(np.array([l == "beetle" for l in d.labels], bool))
        d = models.nms(drop_huge(d, *im.size), 0.5, 0.8)
        tp, nd, ng = score(d.boxes, gt[n]); a = tot[k]; a[0] += tp; a[1] += nd; a[2] += ng; a[3] += dt
print(f"{len(names)} trays, {tot[next(iter(tot))][2]} human boxes; IoU>=0.5")
for k, (tp, nd, ng, dt) in tot.items():
    print(f"{k:22s} P {tp/max(nd,1):.3f}  R {tp/max(ng,1):.3f}  det {nd:5d}  {dt/len(names):.2f} s/img")
print("VRAM peak MiB", torch.cuda.max_memory_allocated() // 2**20)
