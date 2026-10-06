"""NEON Biorepository Darwin Core Archives -> specimen records and a per-image manifest.

The Biorepository (Symbiota portal at Arizona State University) is the only place NEON pitfall
specimens are photographed. GBIF republishes these archives, and every GBIF media URL points back
here, so the archives are the authoritative image manifest.

Outputs (parquet, in data/tables/):
  biorepo_records   one row per occurrence across the carabid collections
  biorepo_images    one row per image, with image_kind and the NEON identifiers to join on
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from . import config

# archive code -> short collection code used in tables
COLLECTIONS = {
    "NEON-CARC-PV": "CARC-PV",  # pinned vouchers (bet_parataxonomistID individuals)
    "NEON-CARC-TS": "CARC-TS",  # trap sorting, ethanol
    "NEON-CARC-AP": "CARC-AP",  # archive pooling, ethanol vials
    "NEON-DCTC": "DCTC",  # pinned carabids held at NEON domain offices
    "NEON-CARC-DNA": "CARC-DNA",  # DNA extracts
    "ASU-NEON-IV": "NEON-IV",  # invertebrate vouchers at ASU (mixed taxa)
}
DWCA_URL = "https://biorepo.neonscience.org/portal/content/dwca/{code}_DwC-A.zip"

_OTHER_KEYS = {
    "NEON sampleID": "neon_sampleID",
    "NEON sampleCode (barcode)": "neon_barcode",
    "NEON sampleUUID": "neon_sampleUUID",
}
_INDIVIDUAL_RE = re.compile(r"^NEON\.BET\.D\d{2}\.\d+$")
_BARCODE_RE = re.compile(r"^A\d{11}$")


def parse_other_catalog_numbers(value: str | None) -> dict[str, str | None]:
    """Split Symbiota's 'label: value; label: value' string into the NEON identifiers.

    GBIF serialises the same field with '|' instead of ';', so both are accepted.
    """
    out: dict[str, str | None] = {v: None for v in _OTHER_KEYS.values()}
    if not value or not isinstance(value, str):
        return out
    for part in re.split(r"\s*[;|]\s*", value):
        label, sep, val = part.partition(":")
        if sep and label.strip() in _OTHER_KEYS:
            out[_OTHER_KEYS[label.strip()]] = val.strip() or None
    return out


def classify_image(collection: str, url: str, creator: str | None) -> tuple[str, str | None]:
    """Return (image_kind, view) from what the file name and collection tell us.

    image_kind: tray_ethanol  - many loose specimens from one vial, with a scale bar
                individual    - one ethanol specimen photographed alone (2016 series)
                pinned        - one pinned voucher (habitus shot)
    view: dorsal / ventral / lateral / None when the file name does not say.
    """
    stem = Path(url).stem
    low = stem.lower()
    view = next((v for v in ("dorsal", "ventral", "lateral") if v in low), None)
    bare = re.sub(r"_(lg|tn)$", "", stem)
    if _BARCODE_RE.match(bare):
        return "tray_ethanol", view
    if collection in ("CARC-PV", "DCTC", "NEON-IV"):
        return "pinned", view
    return "individual", view


def download_archives(force: bool = False) -> dict[str, Path]:
    dest = config.BIOREPO_DIR / "dwca"
    dest.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    with httpx.Client(follow_redirects=True, timeout=600) as client:
        for code in COLLECTIONS:
            p = dest / f"{code}_DwC-A.zip"
            if force or not p.exists() or p.stat().st_size == 0:
                r = client.get(DWCA_URL.format(code=code))
                r.raise_for_status()
                p.write_bytes(r.content)
            paths[code] = p
    return paths


def _read(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    if name not in zf.namelist():
        return pd.DataFrame()
    with zf.open(name) as fh:
        return pd.read_csv(fh, dtype=str, keep_default_na=False, na_values=[""])


def build(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.ensure_dirs()
    recs, imgs = [], []
    for code, path in download_archives(force).items():
        coll = COLLECTIONS[code]
        with zipfile.ZipFile(path) as zf:
            occ = _read(zf, "occurrences.csv")
            mm = _read(zf, "multimedia.csv")
        occ = occ.rename(columns={"id": "occid"})
        ids = pd.DataFrame([parse_other_catalog_numbers(v) for v in occ["otherCatalogNumbers"]])
        occ = pd.concat([occ.reset_index(drop=True), ids], axis=1)
        occ["collection"] = coll
        occ["plotID"] = occ["locationID"].str.extract(r"^([A-Z]{4}_\d{3})", expand=False)
        occ["siteID"] = occ["locationID"].str.extract(r"^([A-Z]{4})[_.]", expand=False)
        occ["domainID"] = occ["locality"].str.extract(r"\((D\d{2})\)", expand=False)
        # pinned vouchers carry the NEON individualID as their sampleID
        occ["individualID"] = occ["neon_sampleID"].where(
            occ["neon_sampleID"].fillna("").str.match(_INDIVIDUAL_RE)
        )
        keep = [
            "occid", "collection", "catalogNumber", "occurrenceID", "neon_sampleID", "neon_barcode",
            "neon_sampleUUID", "individualID", "siteID", "plotID", "domainID", "eventID", "eventDate",
            "year", "family", "scientificName", "taxonRank", "identifiedBy", "dateIdentified", "sex",
            "lifeStage", "individualCount", "preparations", "stateProvince", "decimalLatitude",
            "decimalLongitude", "minimumElevationInMeters", "habitat", "modified", "references",
        ]
        occ = occ[[c for c in keep if c in occ.columns]].copy()
        n_img = mm.groupby("coreid").size() if len(mm) else pd.Series(dtype=int)
        occ["n_images"] = occ["occid"].map(n_img).fillna(0).astype(int)
        recs.append(occ)

        if len(mm):
            mm = mm[mm["type"].fillna("StillImage") == "StillImage"].copy()
            kinds = [classify_image(coll, u, c) for u, c in zip(mm["accessURI"], mm["creator"])]
            mm["image_kind"] = [k for k, _ in kinds]
            mm["view"] = [v for _, v in kinds]
            mm = mm.rename(columns={
                "coreid": "occid", "accessURI": "image_url", "goodQualityAccessURI": "image_url_medium",
                "thumbnailAccessURI": "image_url_thumb", "rights": "license", "Owner": "owner",
                "providerManagedID": "media_uuid",
            })
            mm = mm[["occid", "image_url", "image_url_medium", "image_url_thumb", "format", "license",
                     "creator", "owner", "media_uuid", "image_kind", "view"]]
            imgs.append(mm.merge(
                occ[["occid", "collection", "catalogNumber", "neon_sampleID", "neon_barcode",
                     "individualID", "siteID", "plotID", "domainID", "eventDate", "year", "family",
                     "scientificName", "sex", "individualCount"]],
                on="occid", how="left"))

    records = pd.concat(recs, ignore_index=True)
    images = pd.concat(imgs, ignore_index=True).drop_duplicates(subset=["image_url"])
    # stable image id: collection + file stem, e.g. CARC-PV/NEON.BET.D14.001435_Dorsal_2x._lg
    images.insert(0, "image_id", "biorepo/" + images["collection"] + "/" +
                  images["image_url"].map(lambda u: Path(u).stem))
    images.insert(1, "source", "biorepo")
    records.to_parquet(config.TABLES / "biorepo_records.parquet", index=False)
    images.to_parquet(config.TABLES / "biorepo_images.parquet", index=False)
    return records, images


def main() -> None:
    records, images = build()
    print(f"biorepo_records: {len(records):,} rows")
    print(records.groupby("collection").agg(records=("occid", "size"), with_images=("n_images", lambda s: (s > 0).sum())))
    print(f"biorepo_images: {len(images):,} rows")
    print(images.groupby(["collection", "image_kind", "view"], dropna=False).size())
    print("sites with images:", images["siteID"].nunique())


if __name__ == "__main__":
    main()
