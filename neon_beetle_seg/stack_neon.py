"""Download and stack NEON DP1.10022.001 (ground beetles sampled from pitfall traps) for all sites.

Uses the official `neonutilities` package, which stacks the monthly site files into one table per
data table and adds a `release` column: rows from the latest release carry its tag (citable by
DOI), rows newer than the release are marked PROVISIONAL and may still change.

Needs NEON_TOKEN (see docs/environment.md); the NEON data endpoint rejects anonymous requests.

Outputs (parquet, in data/tables/):
  neon_<table>          one per stacked NEON table
  neon_variables, neon_categoricalCodes, neon_issueLog   NEON's own data dictionary and issue log
  specimen_manifest     one row per pinned individual with its best available identification
"""
from __future__ import annotations

import json

import pandas as pd

from . import config

TABLES = [
    "bet_fielddata", "bet_sorting", "bet_parataxonomistID", "bet_expertTaxonomistIDProcessed",
    "bet_expertTaxonomistIDRaw", "bet_archivepooling", "bet_identificationHistory", "bet_bycatchIDHistory",
]


def download(release: str = "current", include_provisional: bool = True) -> dict:
    import neonutilities as nu

    token = config.load_secrets().get("NEON_TOKEN")
    if not token:
        raise SystemExit(f"NEON_TOKEN is not set (looked in the environment and {config.SECRETS_FILE}); "
                         "the NEON data API returns 403 without one.")
    return nu.load_by_product(
        dpid=config.NEON_PRODUCT, site="all", package="expanded", release=release,
        include_provisional=include_provisional, check_size=False, progress=False, token=token,
    )


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

    man = specimen_manifest(stacked["bet_parataxonomistID"], stacked.get("bet_expertTaxonomistIDProcessed"),
                            stacked.get("bet_fielddata"))
    man = link_biorepository(man)
    man.to_parquet(config.TABLES / "specimen_manifest.parquet", index=False)
    counts["specimen_manifest"] = len(man)
    return counts


def main() -> None:
    counts = build()
    for k, v in counts.items():
        print(f"{k:40s} {v:>9,}")
    fd = pd.read_parquet(config.TABLES / "neon_fielddata.parquet")
    print(f"sites {fd['siteID'].nunique()}  domains {fd['domainID'].nunique()}  "
          f"collectDate {fd['collectDate'].min()} .. {fd['collectDate'].max()}")
    print(fd["release"].value_counts().to_string())


if __name__ == "__main__":
    main()
