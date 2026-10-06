"""Build the DuckLake lakehouse (and a flat .duckdb copy) from the parquet tables in data/tables/.

The catalog is created from inside the lake directory with a *relative* DATA_PATH, so the whole
directory can be moved (to the CyVerse Data Store, to another machine) and attached as-is:

    INSTALL ducklake; LOAD ducklake;
    ATTACH 'ducklake:beetles.ducklake' AS lake (READ_ONLY);   -- run from inside ducklake/

Besides the data tables it writes:
  taxon_terms, landcover_terms, obo_terms, column_tags   ontology links (NCBITaxon, ENVO, PATO, ...)
  run_provenance   model ids and revisions, dataset revisions, package versions, git commit
  table_catalog    one row per table: rows, what it is, where it came from
Column comments carry the OBO CURIEs from metadata/column_tags.csv.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from importlib.metadata import version

import duckdb
import httpx
import pandas as pd

from . import config, models

CATALOG = "beetles.ducklake"

# table -> (description, source)
DESCRIPTIONS = {
    "neon_fielddata": ("NEON bet_fielddata: one row per pitfall trap per bout, all sites", "NEON DP1.10022.001"),
    "neon_sorting": ("NEON bet_sorting: one row per taxon per trap sample", "NEON DP1.10022.001"),
    "neon_parataxonomistID": ("NEON bet_parataxonomistID: one row per pinned individual", "NEON DP1.10022.001"),
    "neon_expertTaxonomistIDProcessed": ("NEON expert identifications, standardised", "NEON DP1.10022.001"),
    "neon_expertTaxonomistIDRaw": ("NEON expert identifications as reported", "NEON DP1.10022.001"),
    "neon_archivepooling": ("NEON bet_archivepooling: pooled ethanol vials", "NEON DP1.10022.001"),
    "neon_identificationHistory": ("NEON past determinations of pinned individuals", "NEON DP1.10022.001"),
    "neon_bycatchIDHistory": ("NEON past determinations of bycatch", "NEON DP1.10022.001"),
    "neon_variables": ("NEON data dictionary for the product's tables", "NEON DP1.10022.001"),
    "neon_categoricalCodes": ("NEON categorical code definitions", "NEON DP1.10022.001"),
    "neon_issueLog": ("NEON issue log for the product", "NEON DP1.10022.001"),
    "neon_validation": ("NEON ingest validation rules", "NEON DP1.10022.001"),
    "specimen_manifest": ("One row per pinned individual with best identification and Biorepository link", "derived"),
    "biorepo_records": ("NEON Biorepository occurrence records for the carabid collections", "biorepo.neonscience.org Darwin Core Archives"),
    "biorepo_images": ("Every image attached to a Biorepository carabid record", "biorepo.neonscience.org Darwin Core Archives"),
    "biorepo_image_files": ("Download result for each Biorepository image (bytes, sha256, pixel size)", "derived"),
    "gbif_occurrences": ("GBIF records for the same collections: gbifID and backbone taxonomy", "GBIF"),
    "image_manifest": ("One row per image across all four image pools, with NEON join keys", "derived"),
    "image_scale": ("Pixel-per-millimetre scale per image and how it was obtained", "derived"),
    "image_results": ("Processing status per image", "pipeline"),
    "instances": ("One row per segmented specimen: boxes, mask RLE, pixel metrics", "pipeline"),
    "measurements": ("Per-specimen morphometrics in pixels and mm, with identifiers and taxon", "pipeline"),
    "elytra_annotations_2018": ("Zooniverse elytra length/width lines on the 2018 tray photos", "imageomics/2018-NEON-beetles"),
    "elytra_matches_2018": ("The same annotations with the segmented specimen each line falls on", "derived"),
    "tray_boxes_2018": ("Human-corrected specimen boxes for the 577 tray photos", "Imageomics/carabidae_beetle_processing"),
    "trait_annotations_hawaii": ("Human scale-bar, elytra and pronotum annotations for PUUM specimens", "imageomics/Hawaii-beetles"),
    "hawaii_matches": ("Hawaii individualID -> segmented specimen", "derived"),
    "validation_detection": ("Per-tray detection counts vs human boxes and annotated individuals", "pipeline"),
    "validation_traits": ("Per-specimen predicted vs human measurements", "pipeline"),
    "validation_summary": ("One row per validation metric", "pipeline"),
    "sam3_comparison": ("SAM 3 text-prompted segmentation vs the main pipeline on validation trays", "pipeline"),
}


def resolve_taxa(names: list[str]) -> pd.DataFrame:
    """NCBITaxon CURIE for each scientific name via EMBL-EBI OLS4, cached in metadata/."""
    cache_path = config.METADATA / "ncbitaxon_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [n for n in names if n and n not in cache]
    if todo:
        with httpx.Client(timeout=30) as client:
            for i, name in enumerate(todo):
                # query the binomial: NEON names can carry subspecies or "sp." suffixes
                parts = name.split()
                query = " ".join(parts[:2]) if len(parts) >= 2 and not parts[1].startswith(("sp", "cf", "nr")) else parts[0]
                entry = dict(query=query, curie=None, label=None, iri=None, exact=False)
                try:
                    r = client.get("https://www.ebi.ac.uk/ols4/api/search",
                                   params={"q": query, "ontology": "ncbitaxon", "rows": 5, "exact": "true",
                                           "fieldList": "obo_id,label,iri"})
                    docs = r.json()["response"]["docs"]
                    hit = next((d for d in docs if d.get("label", "").lower() == query.lower()), None)
                    if hit:
                        entry.update(curie=hit["obo_id"], label=hit["label"], iri=hit["iri"], exact=query == name)
                except (httpx.HTTPError, KeyError, ValueError):
                    pass  # leave unresolved; a later run retries because failures are not cached
                else:
                    cache[name] = entry
                if i % 100 == 99:
                    cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    rows = [dict(scientificName=n, **cache[n]) for n in names if n in cache]
    return pd.DataFrame(rows, columns=["scientificName", "query", "curie", "label", "iri", "exact"])


def provenance() -> pd.DataFrame:
    rows = []

    def add(kind, name, value, detail=None):
        rows.append(dict(kind=kind, name=name, value=None if value is None else str(value), detail=detail))

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=config.ROOT, capture_output=True, text=True).stdout.strip())
        add("code", "git_commit", sha, "working tree had uncommitted changes" if dirty else None)
    except OSError:
        pass
    add("code", "repository", "https://github.com/tyson-swetnam/neon-beetle-seg")
    add("run", "built_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for pkg in ("torch", "torchvision", "transformers", "duckdb", "opencv-python-headless", "neonutilities",
                "huggingface_hub", "pycocotools", "numpy", "pandas"):
        try:
            add("package", pkg, version(pkg))
        except Exception:  # noqa: BLE001
            pass
    for role, mid in (("detector", models.GDINO_ID), ("segmenter", models.SAM2_ID),
                      ("segmenter_sentinel", "facebook/sam2.1-hiera-base-plus"), ("parts", models.BEETLEFLOW_ID + "/5-class"),
                      ("scale_label_ocr", "microsoft/trocr-base-printed")):
        add("model", role, mid, "https://huggingface.co/" + mid.split("/5-class")[0])
    rev = config.HF_DIR / "revisions.json"
    if rev.exists():
        for repo, sha in json.loads(rev.read_text()).items():
            add("dataset", repo, sha, "HuggingFace dataset revision")
    gb = config.GBIF_DIR / "provenance.json"
    if gb.exists():
        p = json.loads(gb.read_text())
        add("dataset", "gbif", p.get("doi") or p.get("method"), p.get("citation") or f"accessed {p.get('accessed')}")
    fd = config.TABLES / "neon_fielddata.parquet"
    if fd.exists():
        rel = pd.read_parquet(fd, columns=["release"])["release"].value_counts()
        for tag, n in rel.items():
            add("dataset", f"NEON {config.NEON_PRODUCT} {tag}", n, "bet_fielddata rows")
    dwca = config.BIOREPO_DIR / "dwca"
    for f in sorted(dwca.glob("*.zip")):
        add("dataset", f"biorepository {f.stem}", time.strftime("%Y-%m-%d", time.gmtime(f.stat().st_mtime)), "archive download date")
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    config.ensure_dirs()
    lake_dir = config.LAKE_DIR
    if lake_dir.exists():
        shutil.rmtree(lake_dir)  # a clean rebuild: no dead files from earlier builds
    (lake_dir / "export").mkdir(parents=True)

    # ontology tables
    meta = {n: pd.read_csv(config.METADATA / f"{n}.csv") for n in ("column_tags", "obo_terms", "landcover_terms")}
    names: set[str] = set()
    for tbl, col in (("specimen_manifest", "bestScientificName"), ("measurements", "scientificName"),
                     ("biorepo_records", "scientificName")):
        p = config.TABLES / f"{tbl}.parquet"
        if p.exists():
            names |= set(pd.read_parquet(p, columns=[col])[col].dropna().unique())
    meta["taxon_terms"] = resolve_taxa(sorted(names))
    meta["run_provenance"] = provenance()
    for name, df in meta.items():
        df.to_parquet(config.TABLES / f"{name}.parquet", index=False)

    cwd = os.getcwd()
    os.chdir(lake_dir)  # relative DATA_PATH is recorded in the catalog -> portable directory
    try:
        con = duckdb.connect()
        con.execute("INSTALL ducklake; LOAD ducklake;")
        con.execute(f"ATTACH 'ducklake:{CATALOG}' AS lake (DATA_PATH 'data/')")
        flat = duckdb.connect("beetles.duckdb")
        catalog_rows = []
        for p in sorted(config.TABLES.glob("*.parquet")):
            name = p.stem
            con.execute(f'CREATE OR REPLACE TABLE lake."{name}" AS SELECT * FROM read_parquet(\'{p}\')')
            flat.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM read_parquet(\'{p}\')')
            shutil.copy2(p, lake_dir / "export" / p.name)
            n = con.execute(f'SELECT count(*) FROM lake."{name}"').fetchone()[0]
            desc, src = DESCRIPTIONS.get(name, (None, "metadata" if name in meta else None))
            if desc:
                con.execute(f'COMMENT ON TABLE lake."{name}" IS ?', [desc])
            catalog_rows.append(dict(table_name=name, rows=n, description=desc, source=src))
        cat = pd.DataFrame(catalog_rows)
        con.execute("CREATE OR REPLACE TABLE lake.table_catalog AS SELECT * FROM cat")
        flat.execute("CREATE OR REPLACE TABLE table_catalog AS SELECT * FROM cat")
        cat.to_parquet(lake_dir / "export" / "table_catalog.parquet", index=False)

        # column comments: "<note> [CURIE; unit CURIE]"
        have = {r[0]: set(c[0] for c in con.execute(f'DESCRIBE lake."{r[0]}"').fetchall()) for r in cat[["table_name"]].itertuples(index=False)}
        for r in meta["column_tags"].itertuples():
            if r.table in have and r.column in have[r.table]:
                tag = r.curie + (f"; unit {r.unit_curie}" if isinstance(r.unit_curie, str) else "")
                con.execute(f'COMMENT ON COLUMN lake."{r.table}"."{r.column}" IS ?', [f"{r.note} [{tag}]"])

        if "measurements" in have and "taxon_terms" in have:
            view = ("SELECT m.*, t.curie AS ncbitaxon FROM {p}measurements m "
                    "LEFT JOIN {p}taxon_terms t ON m.scientificName = t.scientificName")
            con.execute("CREATE OR REPLACE VIEW lake.individual_morphometrics AS " + view.format(p="lake."))
            flat.execute("CREATE OR REPLACE VIEW individual_morphometrics AS " + view.format(p=""))
        snapshots = con.execute("SELECT count(*) FROM lake.snapshots()").fetchone()[0]
        con.execute("DETACH lake")
        flat.close()
    finally:
        os.chdir(cwd)
    print(f"lake: {lake_dir / CATALOG}  ({len(cat)} tables, {snapshots} snapshots)")
    return cat


def main() -> None:
    cat = build()
    print(cat[["table_name", "rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
