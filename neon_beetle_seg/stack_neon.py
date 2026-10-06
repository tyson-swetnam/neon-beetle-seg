"""Download and stack NEON DP1.10022.001 (ground beetles sampled from pitfall traps) for all sites.

Uses the official `neonutilities` package, which stacks the monthly site files into one table per
data table and adds a `release` column: rows from the latest release carry its tag (citable by
DOI), rows newer than the release are marked PROVISIONAL and may still change.

Needs NEON_TOKEN (see docs/environment.md); the NEON data endpoint rejects anonymous requests.

Outputs (parquet, in data/tables/):
  neon_<table>          one per stacked NEON table
  neon_variables, neon_categoricalCodes, neon_issueLog   NEON's own data dictionary and issue log
  specimen_manifest     one row per pinned individual with its best available identification
  preserved_samples     one row per fluid-preserved sample (bulk carabids and bycatch), linked to
                        its Biorepository accession
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import config

TABLES = [
    "bet_fielddata", "bet_sorting", "bet_parataxonomistID", "bet_expertTaxonomistIDProcessed",
    "bet_expertTaxonomistIDRaw", "bet_archivepooling", "bet_identificationHistory", "bet_bycatchIDHistory",
]


class _TokenRequests:
    """Stand-in for the `requests` module inside neonutilities' API helper.

    Before every API call (GET or HEAD) neonutilities 2.0.2 makes an *anonymous* connectivity check. A product
    with ~3,000 site-months makes thousands of calls, the anonymous checks exhaust the
    unauthenticated rate limit, the API answers 429 and the download aborts with "Cannot access
    NEON API". This wrapper sends the token with every request to NEON and answers the repeated
    connectivity check from its first result.
    """

    def __init__(self, token: str):
        import requests

        self._requests, self._token, self._checks = requests, token, {}

    def __getattr__(self, name):
        return getattr(self._requests, name)

    def _call(self, method: str, url, headers=None, **kwargs):
        is_check = url.endswith("products/DP1.00001.001")
        if is_check and method in self._checks:
            return self._checks[method]
        headers = dict(headers or {})
        if "neonscience.org" in url:  # never send the token to the signed storage URLs
            headers.setdefault("X-API-Token", self._token)
        r = getattr(self._requests, method)(url, headers=headers, **kwargs)
        if is_check and r.status_code == 200:
            self._checks[method] = r
        return r

    def get(self, url, headers=None, **kwargs):
        return self._call("get", url, headers, **kwargs)

    def head(self, url, headers=None, **kwargs):
        return self._call("head", url, headers, **kwargs)


def download(release: str = "current", include_provisional: bool = True) -> dict:
    import neonutilities as nu
    from neonutilities.helper_mods import api_helpers

    token = config.load_secrets().get("NEON_TOKEN")
    if not token:
        raise SystemExit(f"NEON_TOKEN is not set (looked in the environment and {config.SECRETS_FILE}); "
                         "the NEON data API returns 403 without one.")
    api_helpers.requests = _TokenRequests(token)
    cwd = os.getcwd()
    os.chdir(config.NEON_DIR)  # neonutilities unpacks into ./filesToStack<product> while it works
    try:
        return nu.load_by_product(
            dpid=config.NEON_PRODUCT, site="all", package="expanded", release=release,
            include_provisional=include_provisional, check_size=False, progress=False, token=token,
        )
    finally:
        os.chdir(cwd)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Make a stacked NEON table parquet-safe: mixed-type object columns become strings."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    return df


def specimen_manifest(para: pd.DataFrame, expert: pd.DataFrame | None, field: pd.DataFrame | None) -> pd.DataFrame:
    """One row per pinned individual. The expert identification wins when there is one."""
    p = para.sort_values("identifiedDate" if "identifiedDate" in para else "individualID")
    p = p.drop_duplicates("individualID", keep="last")
    keep = ["individualID", "domainID", "siteID", "plotID", "trapID", "setDate", "collectDate", "subsampleID",
            "taxonID", "scientificName", "taxonRank", "morphospeciesID", "nativeStatusCode", "sampleCondition",
            "identificationQualifier", "release"]
    man = p[[c for c in keep if c in p.columns]].rename(columns={
        "taxonID": "para_taxonID", "scientificName": "para_scientificName", "taxonRank": "para_taxonRank",
        "identificationQualifier": "para_identificationQualifier"})
    if expert is not None and len(expert):
        e = expert.sort_values("identifiedDate" if "identifiedDate" in expert else "individualID")
        e = e.drop_duplicates("individualID", keep="last")
        ekeep = ["individualID", "taxonID", "scientificName", "taxonRank", "sex", "family", "genus",
                 "identifiedBy", "laboratoryName", "identifiedDate"]
        e = e[[c for c in ekeep if c in e.columns]].rename(columns={
            "taxonID": "expert_taxonID", "scientificName": "expert_scientificName",
            "taxonRank": "expert_taxonRank", "identifiedBy": "expert_identifiedBy",
            "identifiedDate": "expert_identifiedDate"})
        man = man.merge(e, on="individualID", how="left")
    else:
        for c in ("expert_taxonID", "expert_scientificName", "expert_taxonRank", "sex"):
            man[c] = pd.NA
    has_expert = man["expert_scientificName"].notna()
    man["bestTaxonID"] = man["expert_taxonID"].where(has_expert, man["para_taxonID"])
    man["bestScientificName"] = man["expert_scientificName"].where(has_expert, man["para_scientificName"])
    man["bestTaxonRank"] = man["expert_taxonRank"].where(has_expert, man["para_taxonRank"])
    man["idSource"] = has_expert.map({True: "expert", False: "parataxonomist"})
    man["year"] = pd.to_datetime(man["collectDate"], errors="coerce", utc=True).dt.year.astype("Int64")
    if field is not None and "nlcdClass" in field:
        plot = field.drop_duplicates("plotID")[["plotID", "nlcdClass", "decimalLatitude", "decimalLongitude", "elevation"]]
        man = man.merge(plot, on="plotID", how="left")
    return man


def link_biorepository(man: pd.DataFrame) -> pd.DataFrame:
    """Add the Biorepository accession (and GBIF record, when fetched) for each pinned individual."""
    bio_path = config.TABLES / "biorepo_records.parquet"
    if not bio_path.exists():
        return man
    bio = pd.read_parquet(bio_path, columns=["individualID", "collection", "catalogNumber", "occid", "n_images"])
    bio = bio[bio["individualID"].notna()].sort_values("n_images", ascending=False).drop_duplicates("individualID")
    man = man.merge(bio.rename(columns={"collection": "biorepo_collection", "catalogNumber": "biorepo_catalogNumber",
                                        "occid": "biorepo_occid", "n_images": "biorepo_n_images"}),
                    on="individualID", how="left")
    man["in_biorepository"] = man["biorepo_catalogNumber"].notna()
    man["biorepo_n_images"] = man["biorepo_n_images"].fillna(0).astype(int)
    gbif_path = config.TABLES / "gbif_occurrences.parquet"
    if gbif_path.exists():
        g = pd.read_parquet(gbif_path, columns=["gbifID", "catalogNumber"]).drop_duplicates("catalogNumber")
        man = man.merge(g.rename(columns={"catalogNumber": "biorepo_catalogNumber"}), on="biorepo_catalogNumber", how="left")
    return man


SPECIMEN_GROUPS = {  # bet_sorting / bet_archivepooling sampleType -> group used across the lake
    "carabid": "bulk carabid", "other carabid": "bulk carabid", "invert bycatch": "invertebrate bycatch",
    "vert bycatch herp": "herptile bycatch", "vert bycatch mam": "mammal bycatch",
}


def preserved_samples(sorting: pd.DataFrame, pooling: pd.DataFrame | None) -> pd.DataFrame:
    """One row per fluid-preserved sample NEON keeps from pitfall traps, linked to its accession.

    Two levels: the vials made when a trap sample is sorted (bet_sorting: one per trap and taxon,
    or one of bulk invertebrate bycatch per trap), and the archive vials those are pooled into
    per plot and bout (bet_archivepooling). Pinned individuals are in specimen_manifest instead.

    The Biorepository link tries the sample ID, then its hash (NEON publishes some IDs only
    hashed, and the Biorepository records carry the same hash), then the vial barcode.
    """
    common = ["domainID", "siteID", "plotID", "setDate", "collectDate", "sampleType", "taxonID", "scientificName",
              "sampleCondition", "remarks", "release"]
    s = sorting.reindex(columns=common + ["trapID", "sampleID", "subsampleID", "subsampleCode", "individualCount",
                                          "taxonRank", "identifiedBy"]).copy()
    s = s[s["subsampleID"].notna()]
    s = s.rename(columns={"subsampleID": "sample_id", "subsampleCode": "barcode", "sampleID": "trap_sampleID"})
    s.insert(0, "level", "trap sorting")
    parts = [s]
    if pooling is not None and len(pooling):
        a = pooling.reindex(columns=common + ["archiveVialID", "archiveSampleCode", "subsampleIDList",
                                              "pooledFromMultiplePlots"]).copy()
        a = a[a["archiveVialID"].notna()]
        a["n_pooled_subsamples"] = a["subsampleIDList"].fillna("").map(lambda v: len([x for x in v.split("|") if x]))
        a = a.rename(columns={"archiveVialID": "sample_id", "archiveSampleCode": "barcode"}).drop(columns=["subsampleIDList"])
        a.insert(0, "level", "archive pooling")
        parts.append(a)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["level", "sample_id"])
    out["specimen_group"] = out["sampleType"].map(SPECIMEN_GROUPS).fillna(out["sampleType"])
    out["individualCount"] = pd.to_numeric(out.get("individualCount"), errors="coerce")
    out["year"] = pd.to_datetime(out["collectDate"], errors="coerce", utc=True).dt.year.astype("Int64")

    bio_path = config.TABLES / "biorepo_records.parquet"
    if bio_path.exists():
        bio = pd.read_parquet(bio_path, columns=["catalogNumber", "collection", "occid", "n_images", "neon_sampleID",
                                                 "neon_sampleID_hash", "neon_barcode"])
        link = pd.Series(pd.NA, index=out.index, dtype="object")
        how = pd.Series(pd.NA, index=out.index, dtype="object")
        for key, col, label in (("sample_id", "neon_sampleID", "sample ID"), ("sample_id", "neon_sampleID_hash", "sample ID hash"),
                                ("barcode", "neon_barcode", "barcode")):
            lookup = bio[bio[col].notna()].drop_duplicates(col).set_index(col)["catalogNumber"]
            hit = out[key].map(lookup)
            fill = link.isna() & hit.notna()
            link[fill], how[fill] = hit[fill], label
        out["biorepo_catalogNumber"], out["biorepo_link"] = link, how
        acc = bio.drop_duplicates("catalogNumber").set_index("catalogNumber")
        out["biorepo_collection"] = out["biorepo_catalogNumber"].map(acc["collection"])
        out["biorepo_occid"] = out["biorepo_catalogNumber"].map(acc["occid"])
        out["biorepo_n_images"] = out["biorepo_catalogNumber"].map(acc["n_images"]).fillna(0).astype(int)
        out["in_biorepository"] = out["biorepo_catalogNumber"].notna()
        gbif_path = config.TABLES / "gbif_occurrences.parquet"
        if gbif_path.exists():
            g = pd.read_parquet(gbif_path, columns=["gbifID", "catalogNumber"]).drop_duplicates("catalogNumber")
            out["gbifID"] = out["biorepo_catalogNumber"].map(g.set_index("catalogNumber")["gbifID"])
    return out


def derive() -> dict[str, int]:
    """(Re)build the derived tables from the stacked NEON tables already on disk."""
    read = lambda n: pd.read_parquet(config.TABLES / f"neon_{n}.parquet") if (config.TABLES / f"neon_{n}.parquet").exists() else None  # noqa: E731
    man = specimen_manifest(read("parataxonomistID"), read("expertTaxonomistIDProcessed"), read("fielddata"))
    man = link_biorepository(man)
    man.to_parquet(config.TABLES / "specimen_manifest.parquet", index=False)
    pres = preserved_samples(read("sorting"), read("archivepooling"))
    pres.to_parquet(config.TABLES / "preserved_samples.parquet", index=False)
    return {"specimen_manifest": len(man), "preserved_samples": len(pres)}


def build(release: str = "current", include_provisional: bool = True) -> dict[str, int]:
    config.ensure_dirs()
    data = download(release, include_provisional)
    counts: dict[str, int] = {}
    stacked: dict[str, pd.DataFrame] = {}
    for name, obj in data.items():
        if not isinstance(obj, pd.DataFrame):
            continue
        if name in TABLES:
            out = "neon_" + name.removeprefix("bet_")
        elif name.startswith(("variables", "categoricalCodes", "issueLog", "validation")):
            out = "neon_" + name.split("_")[0]
        else:
            continue
        df = _clean(obj)
        df.to_parquet(config.TABLES / f"{out}.parquet", index=False)
        stacked[name], counts[out] = df, len(df)
    # keep NEON's citations and readme next to the tables
    notes = {k: str(v) for k, v in data.items() if not isinstance(v, pd.DataFrame)}
    (config.NEON_DIR / "citations_and_readme.json").write_text(json.dumps(notes, indent=1))

    counts.update(derive())
    return counts


def main() -> None:
    import sys

    # `nbs neon --derive` rebuilds specimen_manifest and preserved_samples without downloading
    counts = derive() if "--derive" in sys.argv else build()
    for k, v in counts.items():
        print(f"{k:40s} {v:>9,}")
    fd = pd.read_parquet(config.TABLES / "neon_fielddata.parquet")
    print(f"sites {fd['siteID'].nunique()}  domains {fd['domainID'].nunique()}  "
          f"collectDate {fd['collectDate'].min()} .. {fd['collectDate'].max()}")
    print(fd["release"].value_counts().to_string())


if __name__ == "__main__":
    main()
