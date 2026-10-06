"""Compare the pipeline's output with human annotations and write the results as tables.

  validation_detection   per 2018 tray: specimens found vs human boxes and annotated individuals
  validation_traits      per specimen: predicted vs human elytra / pronotum measurements
  validation_summary     one row per metric (what docs/validation.md and the report quote)

Caveat recorded with the detection metrics: the 2018 tray boxes were exported from CVAT with
source="file", i.e. they were imported machine proposals that people then corrected. Agreement
with a Grounding DINO detector is therefore optimistic. The Zooniverse elytra lines are fully
independent of any detector, so "annotated individuals covered by a mask" is the stricter check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .models import box_iou


def _match(det: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> int:
    """Greedy one-to-one matches between detections and ground-truth boxes at IoU >= thr."""
    if len(det) == 0 or len(gt) == 0:
        return 0
    iou = box_iou(det, gt)
    tp, used = 0, set()
    for i in np.argsort(-iou.max(axis=1)):
        j = int(iou[i].argmax())
        if iou[i, j] >= thr and j not in used:
            used.add(j)
            tp += 1
    return tp


def detection() -> pd.DataFrame:
    inst = pd.read_parquet(config.TABLES / "instances.parquet",
                           columns=["image_id", "instance_id", "box_x1", "box_y1", "box_x2", "box_y2"])
    done = pd.read_parquet(config.TABLES / "image_results.parquet")
    boxes = pd.read_parquet(config.TABLES / "tray_boxes_2018.parquet")
    el = pd.read_parquet(config.TABLES / "elytra_matches_2018.parquet")
    lines = el[(el["structure"] == "ElytraLength") & el["frame_ok"]]
    rows = []
    for image_id in done.loc[done["image_id"].str.startswith("hf2018/") & (done["status"] == "ok"), "image_id"]:
        d = inst.loc[inst["image_id"] == image_id, ["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(np.float32)
        g = boxes.loc[boxes["image_id"] == image_id, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        ln = lines[lines["image_id"] == image_id]
        # one annotation pass over the tray: the dataset's recommended annotator when present,
        # and a single workflow, because some trays were annotated more than once
        users = ln["user_name"].value_counts()
        user = "IsaFluck" if "IsaFluck" in users.index else (users.index[0] if len(users) else None)
        ind = ln[ln["user_name"] == user]
        if len(ind):
            ind = ind[ind["workflowID"] == ind["workflowID"].value_counts().index[0]]
        rows.append(dict(
            image_id=image_id, n_detected=len(d), n_human_boxes=len(g), tp_iou50=_match(d, g),
            n_annotated=len(ind), n_annotated_covered=int(ind["instance_id"].notna().sum()),
            n_instances_annotated=int(ind["instance_id"].nunique()),
        ))
    return pd.DataFrame(rows)


def traits() -> pd.DataFrame:
    m = pd.read_parquet(config.TABLES / "measurements.parquet")
    keep = m[m["ann_elytra_length_mm"].notna() | m["ann_pronotum_width_mm"].notna()]
    return keep[["instance_id", "image_id", "source", "scientificName", "parts_ok", "parts_complete", "touches_edge",
                 "solidity", "px_per_mm", "elytra_length_mm", "elytra_midline_length_mm", "elytra_width_mm", "elytra_base_width_mm",
                 "pronotum_width_mm", "pronotum_base_width_mm",
                 "body_length_parts_mm", "core_length_mm", "ann_elytra_length_mm", "ann_elytra_width_mm",
                 "ann_pronotum_width_mm", "ann_lying_flat"]].copy()


def _agreement(pred: pd.Series, truth: pd.Series) -> dict:
    ok = pred.notna() & truth.notna() & (truth > 0)
    p, t = pred[ok].astype(float), truth[ok].astype(float)
    if len(p) < 3:
        return dict(n=int(len(p)))
    err = p - t
    return dict(n=int(len(p)), bias_mm=float(err.mean()), mae_mm=float(err.abs().mean()),
                median_abs_pct=float((err.abs() / t).median() * 100), within_10pct=float(((err.abs() / t) <= 0.10).mean()),
                pearson_r=float(np.corrcoef(p, t)[0, 1]), slope=float(np.polyfit(t, p, 1)[0]))


def summary(det: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(group, metric, value, n=None, note=None):
        rows.append(dict(group=group, metric=metric, value=None if value is None else float(value), n=n, note=note))

    if len(det):
        tp, nd, ng = det["tp_iou50"].sum(), det["n_detected"].sum(), det["n_human_boxes"].sum()
        note = "human boxes started from machine proposals (CVAT source=file); optimistic for Grounding DINO"
        add("detection_2018_trays", "precision_iou50", tp / max(nd, 1), int(nd), note)
        add("detection_2018_trays", "recall_iou50", tp / max(ng, 1), int(ng), note)
        add("detection_2018_trays", "trays_with_exact_count", (det["n_detected"] == det["n_human_boxes"]).mean(), len(det), note)
        add("detection_2018_trays", "annotated_individuals_covered", det["n_annotated_covered"].sum() / max(det["n_annotated"].sum(), 1),
            int(det["n_annotated"].sum()), "Zooniverse elytra-length lines (frame-consistent workflows only) whose midpoint lies in a predicted mask; detector-independent")
        ann = det[det["n_annotated"] > 0]
        add("detection_2018_trays", "mean_abs_count_error_vs_annotated", (ann["n_detected"] - ann["n_annotated"]).abs().mean(), len(ann),
            "trays with a frame-consistent annotation pass; annotators skipped some specimens, so extra detections are not all errors")
        add("detection_2018_trays", "trays_count_equals_annotated", (ann["n_detected"] == ann["n_annotated"]).mean(), len(ann))
    for source, label in (("hf2018", "traits_2018_trays_ethanol"), ("hawaii", "traits_hawaii_pinned")):
        t = tr[tr["source"] == source]
        if t.empty:
            continue
        add(label, "specimens_with_annotation", len(t), len(t))
        add(label, "parts_ok_fraction", t["parts_ok"].fillna(False).mean(), len(t), "BeetleFlow found elytra on the specimen")
        # compare like with like: the 2018 volunteers drew elytra width at the base of the elytra,
        # Hawaii's annotators measured the maximum elytra width and the basal pronotum width
        ely_w = "elytra_base_width_mm" if source == "hf2018" else "elytra_width_mm"
        pairs = [("elytra_midline_length", "elytra_midline_length_mm", "ann_elytra_length_mm"),
                 ("elytra_full_length", "elytra_length_mm", "ann_elytra_length_mm"),
                 ("elytra_base_width" if source == "hf2018" else "elytra_max_width", ely_w, "ann_elytra_width_mm"),
                 ("pronotum_base_width", "pronotum_base_width_mm", "ann_pronotum_width_mm")]
        subsets = [("all", t)]
        if source == "hf2018":
            subsets.append(("lying_flat", t[t["ann_lying_flat"] == "Yes"]))
        for sub_name, sub in subsets:
            sub = sub[sub["parts_ok"].fillna(False)]
            for name, pcol, tcol in pairs:
                if sub[tcol].notna().sum() < 3:
                    continue
                for k, v in _agreement(sub[pcol], sub[tcol]).items():
                    if k != "n":
                        add(label, f"{name}.{sub_name}.{k}", v, int((sub[pcol].notna() & sub[tcol].notna()).sum()))
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    det, tr = detection(), traits()
    det.to_parquet(config.TABLES / "validation_detection.parquet", index=False)
    tr.to_parquet(config.TABLES / "validation_traits.parquet", index=False)
    s = summary(det, tr)
    s.to_parquet(config.TABLES / "validation_summary.parquet", index=False)
    return s


def main() -> None:
    s = build()
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.max_colwidth", 60):
        print(s.drop(columns=["note"]).to_string(index=False))


if __name__ == "__main__":
    main()
