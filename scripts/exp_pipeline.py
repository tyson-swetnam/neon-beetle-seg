"""Experiment: run the pipeline on a few images per route, time it and draw overlays."""
import sys, time
import cv2, numpy as np, pandas as pd, torch
from neon_beetle_seg import config, measure, models, segment

man = pd.read_parquet(config.TABLES / "image_manifest.parquet")
man = man[man.duplicate_of.isna() & man.is_carabid]
picks = {
    "hf2018_tray": man[man.source == "hf2018"].sample(3, random_state=1),
    "hawaii_tray": man[man.source == "hawaii"].sample(3, random_state=1),
    "biorepo_pinned_pv": man[(man.source == "biorepo") & man.image_id.str.contains("CARC-PV")].sample(6, random_state=1),
    "biorepo_pinned_dctc": man[(man.source == "biorepo") & man.image_id.str.contains("DCTC")].sample(6, random_state=1),
    "biorepo_individual": man[(man.source == "biorepo") & (man.image_kind == "individual")].sample(6, random_state=1),
    "sentinel_crop": man[man.source == "sentinel"].sample(60, random_state=1).pipe(lambda d: d[[ (config.ROOT / p).stat().st_size > 2000 for p in d.local_path]]).head(8),
}
pipe = segment.Pipeline({"tray", "single", "crop"})
COL = {"head": (255, 60, 60), "pronotum": (60, 220, 60), "elytra": (60, 120, 255)}
out = config.OUT / "peek"; out.mkdir(parents=True, exist_ok=True)
for name, rows in picks.items():
    tiles, secs, ninst = [], [], []
    for row in rows.itertuples():
        t = time.time(); found = pipe.run_image(row); secs.append(time.time() - t); ninst.append(len(found))
        img = np.asarray(segment._open(row.local_path)).copy()
        for r in found:
            m = measure.rle_decode(r["mask_rle"], r["mask_h"], r["mask_w"])
            y, x = r["win_y1"], r["win_x1"]; sub = img[y:y + m.shape[0], x:x + m.shape[1]]
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            th = max(1, int(round(max(m.shape) / 150)))
            if r.get("elytra_rle"):
                py, px = r["pwin_y1"], r["pwin_x1"]; psub = img[py:py + r["parts_h"], px:px + r["parts_w"]]
                for part, c in COL.items():
                    pm = measure.rle_decode(r[f"{part}_rle"], r["parts_h"], r["parts_w"])
                    psub[pm] = (0.55 * psub[pm] + 0.45 * np.array(c)).astype(np.uint8)
            cv2.drawContours(sub, cnts, -1, (255, 255, 0), th)
        s = 900 / max(img.shape[:2]) if name.endswith("tray") else 440 / max(img.shape[:2])
        tiles.append(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
    H = max(t.shape[0] for t in tiles); W = sum(t.shape[1] for t in tiles) + 8 * len(tiles)
    sheet = np.full((H, W, 3), 255, np.uint8); x = 0
    for t in tiles:
        sheet[:t.shape[0], x:x + t.shape[1]] = t; x += t.shape[1] + 8
    cv2.imwrite(str(out / f"seg_{name}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"{name:22s} {np.mean(secs[1:] or secs):6.2f} s/img  instances {ninst}  sheet {sheet.shape[1]}x{sheet.shape[0]}", flush=True)
print("VRAM peak MiB", torch.cuda.max_memory_allocated() // 2**20)
