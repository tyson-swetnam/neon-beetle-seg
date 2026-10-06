"""GBIF records for the NEON Biorepository carabid collections.

GBIF republishes the Biorepository archives. Every GBIF media URL for these records points back
to biorepo.neonscience.org, so GBIF adds no images; what it adds is a stable gbifID per specimen,
the GBIF backbone taxonomy match, and (with an account) a citable download DOI.

Two ways in, same output table:
  download  authenticated GBIF download (GBIF_USER / GBIF_PWD / GBIF_EMAIL) -> citable DOI
  search    anonymous occurrence search, paged per dataset in year-and-state slices (deep
            offsets are too slow to page a 98,000-record dataset in one go)

Output: data/tables/gbif_occurrences.parquet, data/gbif/provenance.json
"""
from __future__ import annotations

import io
import json
import time
import zipfile

import httpx
import pandas as pd

from . import config
from .fetch_biorepo import parse_other_catalog_numbers

API = "https://api.gbif.org/v1"
# GBIF dataset keys of the NEON Biorepository collections that hold carabids
DATASETS = {
    "CARC-PV": "8cb7c449-ba11-4464-865e-8029b8d772e8",
    "CARC-DNA": "44262c91-b3fd-48e4-8e47-1ee03ac2d496",
    "DCTC": "69e5ceb4-30a6-4074-8f9d-d6a0457cb789",
    "CARC-AP": "044d870e-5718-410a-9450-9c2ceac8e1d9",
    "CARC-TS": "2564e9e2-0248-4a3b-8344-24fc0956ed73",
    "NEON-IV": "69f3ca43-ed03-4ed1-92c4-58f9bd0eafc1",
}
FIELDS = ["key", "datasetKey", "catalogNumber", "occurrenceID", "otherCatalogNumbers", "scientificName",
          "acceptedScientificName", "taxonKey", "speciesKey", "species", "genus", "family", "taxonRank",
          "eventDate", "year", "stateProvince", "decimalLatitude", "decimalLongitude", "license"]


def _tidy(df: pd.DataFrame, collection_by_key: dict[str, str]) -> pd.DataFrame:
    df = df.rename(columns={"key": "gbifID"})
    df["gbifID"] = df["gbifID"].astype("string")
    df["collection"] = df["datasetKey"].map(collection_by_key)
    ids = pd.DataFrame([parse_other_catalog_numbers(v) for v in df["otherCatalogNumbers"]])
    df["neon_sampleID"] = ids["neon_sampleID"].values
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    return df


def _get(client: httpx.Client, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = client.get(f"{API}/occurrence/search", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GBIF search failed for {params}")


def via_search() -> tuple[pd.DataFrame, dict]:
    """Page the anonymous search API in small slices.

    GBIF's search gets very slow at deep offsets (minutes per page past ~30,000), so each
    dataset is split by year and state, which keeps every slice a few thousand records at most.
    (Month is no use as a key: collection dates are two-week ranges, and GBIF leaves month empty
    when a range crosses a month boundary.) Records with no year or state cannot be reached this
    way; the shortfall is reported.
    """
    rows, expected = [], {}
    with httpx.Client(timeout=180, headers={"User-Agent": "neon-beetle-seg/0.2"}) as client:
        for coll, key in DATASETS.items():
            base = {"datasetKey": key, "familyKey": config.CARABIDAE_KEY}
            facets = _get(client, {**base, "limit": 0, "facet": ["year", "stateProvince"], "facetLimit": 200})
            expected[coll] = facets["count"]
            by_field = {f["field"]: [c["name"] for c in f["counts"]] for f in facets.get("facets", [])}
            n0 = len(rows)
            for year in sorted(by_field.get("YEAR", [])):
                for state in sorted(by_field.get("STATE_PROVINCE", [])):
                    offset = 0
                    while True:
                        page = _get(client, {**base, "year": year, "stateProvince": state, "limit": 300, "offset": offset})
                        for rec in page["results"]:
                            row = {f: rec.get(f) for f in FIELDS}
                            row["n_media"] = len(rec.get("media", []))
                            rows.append(row)
                        offset += 300
                        if page["endOfRecords"]:
                            break
            print(f"  {coll}: {len(rows) - n0:,} of {expected[coll]:,} records", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("key")
    prov = {"method": "occurrence/search", "accessed": time.strftime("%Y-%m-%d"), "datasets": DATASETS,
            "filter": {"familyKey": config.CARABIDAE_KEY}, "expected": expected, "fetched": int(len(df))}
    return df, prov


def via_download(user: str, pwd: str, email: str) -> tuple[pd.DataFrame, dict]:
    predicate = {"type": "and", "predicates": [
        {"type": "in", "key": "DATASET_KEY", "values": list(DATASETS.values())},
        {"type": "equals", "key": "FAMILY_KEY", "value": str(config.CARABIDAE_KEY)},
    ]}
    body = {"creator": user, "notificationAddresses": [email], "sendNotification": False,
            "format": "SIMPLE_CSV", "predicate": predicate}
    with httpx.Client(timeout=300, auth=(user, pwd)) as client:
        r = client.post(f"{API}/occurrence/download/request", json=body)
        r.raise_for_status()
        key = r.text.strip()
        print(f"  GBIF download {key} requested", flush=True)
        while True:
            meta = client.get(f"{API}/occurrence/download/{key}").json()
            if meta["status"] in ("SUCCEEDED", "FAILED", "KILLED", "CANCELLED"):
                break
            time.sleep(30)
        if meta["status"] != "SUCCEEDED":
            raise RuntimeError(f"GBIF download {key} ended as {meta['status']}")
        z = client.get(meta["downloadLink"], follow_redirects=True)
        z.raise_for_status()
    (config.GBIF_DIR / f"{key}.zip").write_bytes(z.content)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            raw = pd.read_csv(fh, sep="\t", dtype=str, quoting=3)
    # SIMPLE_CSV has no otherCatalogNumbers; the NEON sample ID comes from the Biorepository join
    df = pd.DataFrame({
        "key": raw["gbifID"], "datasetKey": raw["datasetKey"], "catalogNumber": raw["catalogNumber"],
        "occurrenceID": raw["occurrenceID"], "otherCatalogNumbers": None, "scientificName": raw["scientificName"],
        "acceptedScientificName": raw.get("verbatimScientificName"), "taxonKey": raw["taxonKey"],
        "speciesKey": raw["speciesKey"], "species": raw["species"], "genus": raw["genus"], "family": raw["family"],
        "taxonRank": raw["taxonRank"], "eventDate": raw["eventDate"], "year": raw["year"],
        "stateProvince": raw["stateProvince"], "decimalLatitude": raw["decimalLatitude"],
        "decimalLongitude": raw["decimalLongitude"], "license": raw["license"],
        "n_media": raw["mediaType"].fillna("").map(lambda s: len([x for x in s.split(";") if x])),
    })
    prov = {"method": "occurrence/download", "downloadKey": key, "doi": meta.get("doi"),
            "created": meta.get("created"), "totalRecords": meta.get("totalRecords"), "predicate": predicate,
            "citation": f"GBIF.org ({meta.get('created', '')[:10]}) GBIF Occurrence Download https://doi.org/{meta.get('doi')}"}
    return df, prov


def build(prefer_download: bool = True) -> tuple[pd.DataFrame, dict]:
    config.ensure_dirs()
    s = config.load_secrets()
    if prefer_download and all(k in s for k in ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL")):
        df, prov = via_download(s["GBIF_USER"], s["GBIF_PWD"], s["GBIF_EMAIL"])
    else:
        df, prov = via_search()
    df = _tidy(df, {v: k for k, v in DATASETS.items()})
    df.to_parquet(config.TABLES / "gbif_occurrences.parquet", index=False)
    (config.GBIF_DIR / "provenance.json").write_text(json.dumps(prov, indent=1))
    return df, prov


def main() -> None:
    df, prov = build()
    print(f"gbif_occurrences: {len(df):,} rows via {prov['method']}" + (f", DOI {prov['doi']}" if prov.get("doi") else ""))
    print(df.groupby("collection").agg(records=("gbifID", "size"), with_media=("n_media", lambda s: (s > 0).sum())).to_string())


if __name__ == "__main__":
    main()
