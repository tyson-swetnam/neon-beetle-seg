"""NEON Biorepository Darwin Core Archives -> specimen records, collection catalog and image list.

The Biorepository (Symbiota portal at Arizona State University) holds everything NEON keeps from
pitfall sampling: pinned carabids, carabids preserved in bulk in ethanol, carabid DNA extracts,
and the bycatch that comes up in the same traps (other invertebrates, reptiles and amphibians,
small mammals). It is also the only place these specimens are photographed. GBIF republishes
some of these archives, and every GBIF media URL points back here.

Outputs (parquet, in data/tables/):
  biorepo_collections       one row per collection: what it is, how it is preserved, which NEON
                            table and field it comes from, counts, GBIF dataset
  biorepo_records           one row per occurrence across all the collections
  biorepo_identifications   determination history of each record
  biorepo_measurements      measurementOrFact rows (habitat, body measurements, ...)
  biorepo_material_samples  preparation / material-sample rows, where a collection has them
  biorepo_images            one row per image, with image_kind and the NEON identifiers to join on
"""
from __future__ import annotations

import html
import re
import time
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from . import config

# archive code -> (short collection code used in tables, specimen group, pooling level)
COLLECTIONS: dict[str, tuple[str, str, str]] = {
    "NEON-CARC-PV": ("CARC-PV", "pinned carabid", "individual"),
    "NEON-DCTC": ("DCTC", "pinned carabid", "individual"),  # held at NEON domain offices
    "NEON-CARC-TS": ("CARC-TS", "bulk carabid", "trap sorting"),  # ethanol, one vial per trap and taxon
    "NEON-CARC-AP": ("CARC-AP", "bulk carabid", "archive pooling"),  # ethanol, pooled per plot and bout
    "NEON-CARC-DNA": ("CARC-DNA", "carabid DNA extract", "individual"),
    "ASU-NEON-IV": ("NEON-IV", "invertebrate voucher", "individual"),  # mixed taxa, mixed protocols
    "NEON-IVBC-TS": ("IVBC-TS", "invertebrate bycatch", "trap sorting"),
    "NEON-IVBC-AP": ("IVBC-AP", "invertebrate bycatch", "archive pooling"),
    "NEON-HEVC-GBTS": ("HEVC-GBTS", "herptile bycatch", "trap sorting"),
    "NEON-HEVC-GBAP": ("HEVC-GBAP", "herptile bycatch", "archive pooling"),
    "NEON-MAMC-VGB": ("MAMC-VGB", "mammal bycatch", "individual"),
}
# groups whose photographs show beetles and go on to segmentation
CARABID_GROUPS = {"pinned carabid", "bulk carabid", "carabid DNA extract", "invertebrate voucher"}
DWCA_URL = "https://biorepo.neonscience.org/portal/content/dwca/{code}_DwC-A.zip"
COLLECTION_API = "https://biorepo.neonscience.org/portal/api/v2/collection"

_OTHER_KEYS = {
    "NEON sampleID": "neon_sampleID",
    # NEON publishes some sample IDs only as a hash (in its own tables too); the hash is the join key then
    "NEON sampleID Hash": "neon_sampleID_hash",
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
                bycatch       - a photograph of a bycatch specimen (not a beetle; not segmented)
    view: dorsal / ventral / lateral / None when the file name does not say.
    """
    stem = Path(url).stem
    low = stem.lower()
    view = next((v for v in ("dorsal", "ventral", "lateral") if v in low), None)
    if collection.startswith(("IVBC", "HEVC", "MAMC")):
        return "bycatch", view
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


def _plain(text: str | None) -> str | None:
    """Collection descriptions are HTML fragments; keep their text."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"<li[^>]*>", " - ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip() or None


def collection_catalog(counts: pd.DataFrame, archive_dates: dict[str, str]) -> pd.DataFrame:
    """Collection-level metadata from the portal API, joined to what was actually downloaded."""
    rows = []
    try:
        api = httpx.get(COLLECTION_API, params={"limit": 500, "offset": 0}, timeout=120).json()["results"]
    except (httpx.HTTPError, KeyError, ValueError):
        api = []
    by_code = {f"{c.get('institutionCode')}-{c.get('collectionCode')}": c for c in api}
    for code, (short, group, pooling) in COLLECTIONS.items():
        c = by_code.get(code, {})
        rows.append(dict(
            collection=short, archive_code=code, collID=c.get("collID"), collection_name=c.get("collectionName"),
            specimen_group=group, pooling=pooling, sample_type=c.get("sampleType"),
            higher_taxon=c.get("higherTaxon"), lower_taxon=c.get("lowerTaxon"),
            neon_product=c.get("productID"),
            # the NEON table and field each record's sample ID comes from, e.g. bet_sorting_in.subsampleID.mam
            neon_source_field=c.get("datasetID"), sample_description=c.get("datasetname"),
            description=_plain(c.get("fullDescription")), rights=c.get("rights"),
            collection_guid=c.get("collectionGuid"), dwca_url=c.get("dwcaUrl") or DWCA_URL.format(code=code),
            portal_url=f"https://biorepo.neonscience.org/portal/collections/misc/collprofiles.php?collid={c.get('collID')}" if c.get("collID") else None,
            archive_downloaded=archive_dates.get(code),
        ))
    cat = pd.DataFrame(rows).merge(counts, on="collection", how="left")
    cat["collID"] = cat["collID"].astype("Int64")
    return cat


def build(force: bool = False) -> dict[str, pd.DataFrame]:
    config.ensure_dirs()
    recs, imgs, idents, facts, materials, dates = [], [], [], [], [], {}
    for code, path in download_archives(force).items():
        coll, group, pooling = COLLECTIONS[code]
        dates[code] = time.strftime("%Y-%m-%d", time.gmtime(path.stat().st_mtime))
        with zipfile.ZipFile(path) as zf:
            occ = _read(zf, "occurrences.csv")
            mm = _read(zf, "multimedia.csv")
            extra = {name: _read(zf, f"{name}.csv") for name in ("identifications", "measurementOrFact", "materialSample")}
        occ = occ.rename(columns={"id": "occid"})
        ids = pd.DataFrame([parse_other_catalog_numbers(v) for v in occ["otherCatalogNumbers"]])
        occ = pd.concat([occ.reset_index(drop=True), ids], axis=1)
        occ["collection"], occ["specimen_group"], occ["pooling"] = coll, group, pooling
        occ["plotID"] = occ["locationID"].str.extract(r"^([A-Z]{4}_\d{3})", expand=False)
        occ["siteID"] = occ["locationID"].str.extract(r"^([A-Z]{4})[_.]", expand=False)
        occ["domainID"] = occ["locality"].str.extract(r"\((D\d{2})\)", expand=False)
        # pinned vouchers carry the NEON individualID as their sampleID
        occ["individualID"] = occ["neon_sampleID"].where(
            occ["neon_sampleID"].fillna("").str.match(_INDIVIDUAL_RE)
        )
        keep = [
            "occid", "collection", "specimen_group", "pooling", "catalogNumber", "occurrenceID", "neon_sampleID",
            "neon_sampleID_hash", "neon_barcode", "neon_sampleUUID", "individualID", "siteID", "plotID", "domainID", "eventID",
            "eventDate", "year", "kingdom", "phylum", "class", "order", "family", "scientificName", "taxonRank",
            "identifiedBy", "dateIdentified", "identificationQualifier", "sex", "lifeStage", "individualCount",
            "preparations", "samplingProtocol", "occurrenceRemarks", "dynamicProperties", "recordedBy",
            "stateProvince", "county", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
            "minimumElevationInMeters", "habitat", "disposition", "modified", "references",
        ]
        occ = occ.reindex(columns=keep).copy()
        n_img = mm.groupby("coreid").size() if len(mm) else pd.Series(dtype=int)
        occ["n_images"] = occ["occid"].map(n_img).fillna(0).astype(int)
        recs.append(occ)

        for name, sink in (("identifications", idents), ("measurementOrFact", facts), ("materialSample", materials)):
            df = extra[name]
            if len(df):
                df = df.rename(columns={"coreid": "occid"})
                df.insert(1, "collection", coll)
                sink.append(df)

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
                occ[["occid", "collection", "specimen_group", "catalogNumber", "neon_sampleID", "neon_barcode",
                     "individualID", "siteID", "plotID", "domainID", "eventDate", "year", "family",
                     "scientificName", "sex", "individualCount"]],
                on="occid", how="left"))

    records = pd.concat(recs, ignore_index=True)
    images = pd.concat(imgs, ignore_index=True).drop_duplicates(subset=["image_url"])
    # stable image id: collection + file stem, e.g. CARC-PV/NEON.BET.D14.001435_Dorsal_2x._lg
    images.insert(0, "image_id", "biorepo/" + images["collection"] + "/" +
                  images["image_url"].map(lambda u: Path(u).stem))
    images.insert(1, "source", "biorepo")
    # a dozen photos were uploaded twice for the same record (same file name, two folders)
    images = images.drop_duplicates("image_id")
    counts = records.groupby("collection").agg(
        records=("occid", "size"), records_with_images=("n_images", lambda s: int((s > 0).sum())),
        sites=("siteID", "nunique"), first_year=("year", "min"), last_year=("year", "max"),
        individuals=("individualCount", lambda s: pd.to_numeric(s, errors="coerce").sum()),
    ).reset_index()
    counts = counts.merge(images.groupby("collection").size().rename("images").reset_index(), on="collection", how="left")
    counts["images"] = counts["images"].fillna(0).astype(int)
    out = {
        "biorepo_collections": collection_catalog(counts, dates),
        "biorepo_records": records,
        "biorepo_images": images,
        "biorepo_identifications": pd.concat(idents, ignore_index=True) if idents else pd.DataFrame(),
        "biorepo_measurements": pd.concat(facts, ignore_index=True) if facts else pd.DataFrame(),
        "biorepo_material_samples": pd.concat(materials, ignore_index=True) if materials else pd.DataFrame(),
    }
    for name, df in out.items():
        if len(df):
            df.to_parquet(config.TABLES / f"{name}.parquet", index=False)
    return out


def main() -> None:
    out = build()
    cat = out["biorepo_collections"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(cat[["collection", "specimen_group", "pooling", "records", "records_with_images", "images", "sites",
                   "first_year", "last_year", "neon_source_field"]].to_string(index=False))
    for name, df in out.items():
        print(f"{name:26s} {len(df):>9,} rows")


if __name__ == "__main__":
    main()
