"""Build the unified image manifest: one row per image across all four image pools.

Pools and how each is processed (`route`):
  biorepo   NEON Biorepository images                    tray or single, by image_kind
  hf2018    imageomics/2018-NEON-beetles group images     tray  (ethanol specimens, 2018)
  hawaii    imageomics/Hawaii-beetles group images        tray  (pinned specimens, PUUM)
  sentinel  imageomics/sentinel-beetles individual crops  crop  (pinned, IDs anonymised)
  herp      NEON Biorepository herptile bycatch photos    herp  (reptiles and amphibians)

`process` marks the images that get segmented: not a duplicate, and either a carabid or a herp.

Writes data/tables/image_manifest.parquet plus the human annotation tables used for validation.
"""
from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config

Image.MAX_IMAGE_PIXELS = None

def _mode(s: pd.Series):
    """Most common non-null value, or None when every value is null."""
    m = s.dropna().mode()
    return m.iat[0] if len(m) else None


COLUMNS = [
    "image_id", "source", "image_kind", "route", "view", "local_path", "width", "height", "license",
    "image_url", "sha256", "duplicate_of", "is_carabid", "specimen_group", "process",
    # join keys back to NEON
    "individualID", "neon_sampleID", "neon_barcode", "biorepo_occid", "catalogNumber",
    "siteID", "plotID", "domainID", "eventDate", "year", "scientificName", "sex", "individualCount",
    # scale known from human annotation (None where it must be read from the image)
    "px_per_mm", "scale_source",
    # sentinel-only: anonymised ids and the linked scale-bar crop
    "anon_siteID", "anon_domainID", "anon_eventID", "split", "scalebar_path",
]


def _rel(p: Path) -> str:
    return str(p.relative_to(config.ROOT))


def _size(p: Path) -> tuple[int, int]:
    with Image.open(p) as im:
        return im.size


def hf2018() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tray manifest, elytra line annotations and human tray boxes, all in full-resolution pixels."""
    d = config.HF_DIR / "2018-NEON-beetles"
    ann = pd.read_csv(d / "BeetleMeasurements.csv")
    ann["pictureID"] = ann["pictureID"].str.replace(".jpg", "", regex=False)
    dims = ann["image_dim"].map(ast.literal_eval)
    rdims = ann["resized_image_dim"].map(ast.literal_eval)
    ann["full_h"] = dims.map(lambda t: t[0]); ann["full_w"] = dims.map(lambda t: t[1])
    ann["up"] = ann["full_w"] / rdims.map(lambda t: t[1])  # resized -> full resolution
    ann["px_per_cm_full"] = ann["cm_pix"] * ann["up"]
    # coords_pix is in resized-image pixels; scale it ourselves (two rows of the dataset's own
    # coords_pix_scaled_up column are the placeholder "(0, 0, 0)")
    line = ann["coords_pix"].map(json.loads)
    for k in ("x1", "y1", "x2", "y2"):
        ann[k] = line.map(lambda c, k=k: float(c[k])) * ann["up"]
    ann["dist_px_full"] = np.hypot(ann["x2"] - ann["x1"], ann["y2"] - ann["y1"])
    ann.insert(0, "image_id", "hf2018/" + ann["pictureID"])
    elytra = ann[["image_id", "pictureID", "individual", "structure", "lying_flat", "x1", "y1", "x2", "y2",
                  "dist_px_full", "dist_cm", "px_per_cm_full", "scientificName", "NEON_sampleID", "siteID",
                  "plotID", "user_name", "workflowID", "measureID"]].copy()

    per = ann.groupby("pictureID").agg(
        px_per_cm=("px_per_cm_full", "median"), full_w=("full_w", "first"), full_h=("full_h", "first"),
        scientificName=("scientificName", _mode), NEON_sampleID=("NEON_sampleID", "first"),
        siteID=("siteID", "first"), plotID=("plotID", "first"), n_annotated=("individual", "nunique"),
    ).reset_index()
    rows = []
    for r in per.itertuples():
        p = d / "group_images" / f"{r.pictureID}.jpg"
        if not p.exists():
            continue
        rows.append(dict(
            image_id=f"hf2018/{r.pictureID}", source="hf2018", image_kind="tray_ethanol", route="tray",
            local_path=_rel(p), width=r.full_w, height=r.full_h,
            license="http://creativecommons.org/licenses/by-sa/4.0/",
            image_url=f"https://huggingface.co/datasets/imageomics/2018-NEON-beetles/resolve/main/group_images/{r.pictureID}.jpg",
            is_carabid=True, neon_sampleID=r.NEON_sampleID, neon_barcode=r.pictureID, siteID=r.siteID,
            plotID=r.plotID, year="2018", scientificName=r.scientificName,
            px_per_mm=r.px_per_cm / 10.0, scale_source="zooniverse_scalebar",
        ))
    # human boxes (CVAT export from Imageomics/carabidae_beetle_processing)
    xml = config.DATA / "external/carabidae_beetle_processing/annotations/2018_neon_beetles_bbox.xml"
    boxes = []
    if xml.exists():
        for im in ET.parse(xml).getroot().findall("image"):
            pid = Path(im.get("name")).stem
            for b in im.findall("box"):
                boxes.append(dict(image_id=f"hf2018/{pid}", x1=float(b.get("xtl")), y1=float(b.get("ytl")),
                                  x2=float(b.get("xbr")), y2=float(b.get("ybr"))))
    return pd.DataFrame(rows), elytra, pd.DataFrame(boxes)


def hawaii() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = config.HF_DIR / "Hawaii-beetles"
    tr = pd.read_csv(d / "trait_annotations.csv")
    meta = pd.read_csv(d / "images_metadata.csv", dtype=str)
    tr["stem"] = tr["groupImageFilePath"].map(lambda s: Path(s).stem)
    tr.insert(0, "image_id", "hawaii/" + tr["stem"])
    for col in ("scalebar", "elytra_max_length", "basal_pronotum_width", "elytra_max_width"):
        pts = tr[f"coords_{col}"].map(lambda s: json.loads(s)[0] if isinstance(s, str) else [np.nan] * 4)
        for i, k in enumerate(("x1", "y1", "x2", "y2")):
            tr[f"{col}_{k}"] = pts.map(lambda p, i=i: float(p[i]))
    tr = tr.merge(meta[["individualID", "taxonID", "scientificName", "plotID", "trapID", "collectDate"]],
                  on="individualID", how="left")
    tr = tr.drop(columns=[c for c in tr.columns if c.startswith("coords_")] + ["stem"])
    rows = []
    for stem, g in tr.groupby(tr["image_id"].str.split("/").str[1]):
        p = d / "group_images" / f"{stem}.png"
        if not p.exists():
            continue
        w, h = _size(p)
        rows.append(dict(
            image_id=f"hawaii/{stem}", source="hawaii", image_kind="tray_pinned", route="tray",
            local_path=_rel(p), width=w, height=h, license="https://creativecommons.org/licenses/by/4.0/",
            image_url=f"https://huggingface.co/datasets/imageomics/Hawaii-beetles/resolve/main/group_images/{stem}.png",
            is_carabid=True, siteID="PUUM", domainID="D20", year=_mode(g["collectDate"].astype("string").str[:4]), scientificName=_mode(g["scientificName"]),
            px_per_mm=float((g["px_scalebar"] / (g["cm_scalebar"] * 10)).median()), scale_source="hawaii_scalebar",
        ))
    return pd.DataFrame(rows), tr


def sentinel() -> pd.DataFrame:
    d = config.HF_DIR / "sentinel-beetles"
    frames = [pd.read_csv(f, dtype=str).assign(split=f.stem) for f in sorted(d.glob("*.csv"))]
    a = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(dict(
        image_id="sentinel/" + a["public_id"], source="sentinel", image_kind="pinned", route="crop",
        local_path=[_rel(d / "images" / f) for f in a["relative_img_loc"]],
        license="https://creativecommons.org/licenses/by/4.0/", is_carabid=True,
        image_url="https://huggingface.co/datasets/imageomics/sentinel-beetles", eventDate=a["collectDate"],
        year=a["collectDate"].str[:4], scientificName=a["scientificName"],
        anon_siteID=a["siteID"], anon_domainID=a["domainID"], anon_eventID=a["eventID"], split=a["split"],
        scalebar_path=[_rel(d / "color_and_scale_images" / f) for f in a["scalebar_path"]],
    ))
    return out


def biorepo(hf_barcodes: set[str]) -> pd.DataFrame:
    im = pd.read_parquet(config.TABLES / "biorepo_images.parquet")
    files = pd.read_parquet(config.TABLES / "biorepo_image_files.parquet").drop_duplicates("image_id")
    m = im.merge(files, on="image_id", how="left")
    m = m[m["status"] == "ok"].copy()
    m["route"] = np.where(m["image_kind"] == "tray_ethanol", "tray", "single")
    herp = m["specimen_group"] == "herptile bycatch"
    m.loc[herp, ["source", "route"]] = ["herp", "herp"]
    m["is_carabid"] = m["family"] == "Carabidae"
    m["biorepo_occid"] = m["occid"]
    # the Biorepository tray photos are downsized copies of the HuggingFace 2018 trays
    stem = m["image_url"].map(lambda u: Path(u).stem.removesuffix("_lg"))
    dup = (m["image_kind"] == "tray_ethanol") & stem.isin(hf_barcodes)
    m["duplicate_of"] = np.where(dup, "hf2018/" + stem, None)
    m["neon_barcode"] = m["neon_barcode"].fillna(stem.where(m["image_kind"] == "tray_ethanol"))
    # a second upload of the same file under another occurrence
    first = m.sort_values("image_id").drop_duplicates("sha256")["image_id"]
    by_sha = dict(zip(m.loc[first.index, "sha256"], first))
    same = m["sha256"].map(by_sha)
    m["duplicate_of"] = m["duplicate_of"].where(same == m["image_id"], m["duplicate_of"].fillna(same))
    return m


def build() -> pd.DataFrame:
    config.ensure_dirs()
    trays, elytra, boxes = hf2018()
    hw, hw_traits = hawaii()
    elytra.to_parquet(config.TABLES / "elytra_annotations_2018.parquet", index=False)
    boxes.to_parquet(config.TABLES / "tray_boxes_2018.parquet", index=False)
    hw_traits.to_parquet(config.TABLES / "trait_annotations_hawaii.parquet", index=False)
    parts = [biorepo(set(trays["neon_barcode"])), trays, hw, sentinel()]
    for part, group in zip(parts[1:], ("bulk carabid", "pinned carabid", "pinned carabid")):
        part["specimen_group"] = group
    man = pd.concat([p.reindex(columns=COLUMNS) for p in parts], ignore_index=True)
    for c in ("year", "individualCount", "biorepo_occid"):
        man[c] = man[c].astype("string")
    man["is_carabid"] = man["is_carabid"].astype(bool)
    man["process"] = man["duplicate_of"].isna() & (man["is_carabid"] | (man["source"] == "herp"))
    man.to_parquet(config.TABLES / "image_manifest.parquet", index=False)
    return man


def main() -> None:
    man = build()
    todo = man[man["process"]]
    print(f"image_manifest: {len(man):,} rows; {len(todo):,} to process")
    print(man.assign(dup=man["duplicate_of"].notna()).groupby(["source", "image_kind", "route"]).agg(
        images=("image_id", "size"), duplicates=("dup", "sum"), non_carabid=("is_carabid", lambda s: (~s).sum()),
        with_scale=("px_per_mm", lambda s: s.notna().sum()), sites=("siteID", "nunique")).to_string())


if __name__ == "__main__":
    main()
