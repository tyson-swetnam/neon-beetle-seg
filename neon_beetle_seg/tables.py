"""Collect the segmentation shards into analysis tables.

  instances      one row per segmented specimen, with boxes, mask RLE and pixel metrics
  measurements   the same specimens without the masks, with millimetre values where the image
                 has a trusted scale, the specimen's identifiers and its taxon
  image_results  one row per processed image (status, number of specimens, seconds)

Specimens in tray photos are linked to the human annotations by position: an annotated elytra
line whose midpoint falls inside a specimen's mask belongs to that specimen. That is how Hawaii
specimens get their NEON individualID.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, measure

LENGTH_COLS = ["length_px", "width_px", "perimeter_px", "ellipse_major_px", "ellipse_minor_px", "core_length_px",
               "core_width_px", "body_length_parts_px", "elytra_length_px", "elytra_width_px",
               "pronotum_length_px", "pronotum_width_px", "head_width_px", "elytra_base_width_px",
               "pronotum_base_width_px", "elytra_midline_length_px"]
AREA_COLS = ["area_px", "elytra_area_px", "pronotum_area_px", "head_area_px"]
RLE_COLS = ["mask_rle", "head_rle", "pronotum_rle", "elytra_rle"]


def read_shards(kind: str) -> pd.DataFrame:
    files = sorted((config.OUT / "shards").glob(f"*/{kind}-*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def point_to_instance(inst: pd.DataFrame, points: pd.DataFrame) -> pd.Series:
    """For each (image_id, x, y) point return the instance_id whose mask contains it (or None).

    Falls back to the instance whose detection box contains the point when no mask does, taking
    the smallest such box.
    """
    out = pd.Series(index=points.index, dtype="object")
    groups = {k: g for k, g in inst.groupby("image_id")}
    for image_id, pts in points.groupby("image_id"):
        g = groups.get(image_id)
        if g is None:
            continue
        masks = [(r.instance_id, r.win_x1, r.win_y1, measure.rle_decode(r.mask_rle, r.mask_h, r.mask_w),
                  (r.box_x1, r.box_y1, r.box_x2, r.box_y2)) for r in g.itertuples()]
        for idx, x, y in zip(pts.index, pts["x"], pts["y"]):
            hit, box_hit, box_area = None, None, np.inf
            for iid, wx, wy, m, (bx1, by1, bx2, by2) in masks:
                mx, my = int(round(x - wx)), int(round(y - wy))
                if 0 <= my < m.shape[0] and 0 <= mx < m.shape[1] and m[my, mx]:
                    hit = iid
                    break
                if bx1 <= x <= bx2 and by1 <= y <= by2 and (bx2 - bx1) * (by2 - by1) < box_area:
                    box_hit, box_area = iid, (bx2 - bx1) * (by2 - by1)
            out[idx] = hit or box_hit
    return out


FRAME_TOLERANCE = 0.08


def fix_annotation_frame(el: pd.DataFrame) -> pd.DataFrame:
    """Rescale 2018 annotation lines whose pixel frame does not match the photo.

    Each line comes with the annotator's own 1 cm scale bar, drawn in the same frame. Comparing
    that bar with the checkerboard measured from the photo gives the frame factor. Where it is
    off by more than FRAME_TOLERANCE (Zooniverse workflows 21652 and 21827) the coordinates are
    multiplied by it. `dist_cm` is unaffected: line and bar share a frame, so their ratio holds.
    """
    path = config.TABLES / "image_scale.parquet"
    el = el.copy()
    el["frame_factor"] = 1.0
    if not path.exists():
        return el
    sc = pd.read_parquet(path, columns=["image_id", "px_per_mm", "scale_source"])
    cb = sc[sc["scale_source"] == "checkerboard"].set_index("image_id")["px_per_mm"] * 10
    f = el["image_id"].map(cb) / el["px_per_cm_full"]
    # one factor per photo and workflow: the median is steadier than any single drawn bar
    f = f.groupby([el["image_id"], el["workflowID"]]).transform("median")
    off = f.notna() & ((f - 1).abs() > FRAME_TOLERANCE)
    el.loc[off, "frame_factor"] = f[off]
    for c in ("x1", "y1", "x2", "y2", "dist_px_full"):
        el[c] = el[c] * el["frame_factor"]
    # rescaled lines land near their specimen but not reliably on it, so only lines already in
    # the photo's frame are used to pair a human measurement with a specimen
    el["frame_ok"] = ~off & f.notna()
    return el


def add_base_widths(inst: pd.DataFrame) -> pd.DataFrame:
    """Junction widths from the stored part masks (see measure.base_widths)."""
    rows = []
    for r in inst.itertuples():
        if not isinstance(getattr(r, "elytra_rle", None), str):
            rows.append(dict(elytra_midline_length_px=None, elytra_base_width_px=None, pronotum_base_width_px=None))
            continue
        h, w = int(r.parts_h), int(r.parts_w)
        rows.append(measure.base_widths(*(measure.rle_decode(getattr(r, f"{n}_rle"), h, w)
                                          for n in ("head", "pronotum", "elytra"))))
    return pd.concat([inst.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def link_annotations(inst: pd.DataFrame) -> pd.DataFrame:
    """Per-instance human annotation columns for the two annotated tray datasets."""
    cols = pd.DataFrame({"instance_id": inst["instance_id"]})
    # Hawaii: one annotated individual per specimen -> individualID and three human measurements
    hw = pd.read_parquet(config.TABLES / "trait_annotations_hawaii.parquet")
    hw["x"] = (hw["elytra_max_length_x1"] + hw["elytra_max_length_x2"]) / 2
    hw["y"] = (hw["elytra_max_length_y1"] + hw["elytra_max_length_y2"]) / 2
    hw = hw[hw["x"].notna()].copy()
    hw["instance_id"] = point_to_instance(inst, hw)
    h = hw[hw["instance_id"].notna()].drop_duplicates("instance_id")
    h = h.assign(ann_elytra_length_mm=h["cm_elytra_max_length"] * 10, ann_elytra_width_mm=h["cm_elytra_max_width"] * 10,
                 ann_pronotum_width_mm=h["cm_basal_pronotum_width"] * 10)
    h = h[["instance_id", "individualID", "scientificName", "plotID", "trapID", "collectDate",
           "ann_elytra_length_mm", "ann_elytra_width_mm", "ann_pronotum_width_mm"]].rename(
        columns={"individualID": "ann_individualID", "scientificName": "ann_scientificName",
                 "plotID": "ann_plotID", "trapID": "ann_trapID", "collectDate": "ann_collectDate"})
    hw[["image_id", "individualID", "instance_id"]].to_parquet(config.TABLES / "hawaii_matches.parquet", index=False)

    # 2018 trays: several volunteers drew each line; take the median per specimen and structure
    el = pd.read_parquet(config.TABLES / "elytra_annotations_2018.parquet")
    el = fix_annotation_frame(el)
    el["x"], el["y"] = (el["x1"] + el["x2"]) / 2, (el["y1"] + el["y2"]) / 2
    el["instance_id"] = point_to_instance(inst, el)
    el.drop(columns=["x", "y"]).to_parquet(config.TABLES / "elytra_matches_2018.parquet", index=False)
    e = el[el["instance_id"].notna() & el["frame_ok"]].assign(mm=lambda d: d["dist_cm"] * 10)
    e = e.pivot_table(index="instance_id", columns="structure", values="mm", aggfunc="median").reset_index()
    e = e.rename(columns={"ElytraLength": "ann_elytra_length_mm", "ElytraWidth": "ann_elytra_width_mm"})
    flat = el[el["instance_id"].notna() & el["frame_ok"]].groupby("instance_id")["lying_flat"].agg(lambda s: s.mode().iat[0])
    e["ann_lying_flat"] = e["instance_id"].map(flat)

    ann = pd.concat([h, e], ignore_index=True)
    return cols.merge(ann, on="instance_id", how="left")


def collect() -> dict[str, int]:
    inst = read_shards("instances")
    images = read_shards("images")
    if inst.empty:
        raise SystemExit("no segmentation shards yet (run `nbs segment <pool>` first)")
    man = pd.read_parquet(config.TABLES / "image_manifest.parquet")
    scale_path = config.TABLES / "image_scale.parquet"
    if scale_path.exists():
        sc = pd.read_parquet(scale_path)[["image_id", "px_per_mm", "scale_source", "scale_confidence"]]
    else:  # annotated scales only
        sc = man.loc[man["px_per_mm"].notna(), ["image_id", "px_per_mm", "scale_source"]].assign(scale_confidence=1.0)

    inst = add_base_widths(inst)
    inst.to_parquet(config.TABLES / "instances.parquet", index=False)
    images.to_parquet(config.TABLES / "image_results.parquet", index=False)

    m = inst.drop(columns=[c for c in RLE_COLS if c in inst.columns])
    m = m.merge(sc, on="image_id", how="left")
    for c in LENGTH_COLS:
        if c in m:
            m[c.replace("_px", "_mm")] = m[c] / m["px_per_mm"]
    for c in AREA_COLS:
        if c in m:
            m[c.replace("_px", "_mm2")] = m[c] / m["px_per_mm"] ** 2
    keys = ["image_id", "source", "image_kind", "view", "individualID", "neon_sampleID", "neon_barcode",
            "biorepo_occid", "catalogNumber", "siteID", "plotID", "domainID", "eventDate", "year",
            "scientificName", "sex", "anon_siteID", "anon_domainID", "anon_eventID", "split"]
    m = m.merge(man[keys], on="image_id", how="left")
    m = m.merge(link_annotations(inst), on="instance_id", how="left")
    # Hawaii specimens take their identity from the matched annotation
    for col, ann in (("individualID", "ann_individualID"), ("scientificName", "ann_scientificName")):
        m[col] = m[ann].where(m[ann].notna(), m[col])
    m["plotID"] = m["plotID"].where(m["ann_plotID"].isna(), "PUUM_" + m["ann_plotID"].astype("string").str.zfill(3))
    m = m.drop(columns=["ann_individualID", "ann_scientificName", "ann_plotID"])
    # quality flags: what a user should filter on before analysis
    m["qc_low_solidity"] = m["solidity"] < 0.5
    m["qc_has_scale"] = m["px_per_mm"].notna()
    m.to_parquet(config.TABLES / "measurements.parquet", index=False)
    return {"instances": len(inst), "measurements": len(m), "image_results": len(images)}


def main() -> None:
    for k, v in collect().items():
        print(f"{k:16s} {v:>9,}")


if __name__ == "__main__":
    main()
