# neon-beetle-seg

Segmentation and body measurements of ground beetles (Carabidae) from
[NEON](https://www.neonscience.org/) pitfall traps, for every image of a NEON pitfall specimen
that could be found, linked back to NEON's trap records for all 47 sites.

The pipeline pulls records and images from NEON, the NEON Biorepository, GBIF and HuggingFace,
runs open segmentation models on a single 16 GB GPU, and publishes the result as a
[DuckLake](https://ducklake.select/) lakehouse on the CyVerse Data Store.

**Status: v0.2.0, segmentation run in progress.** Results and validation numbers are added to this page and to [docs/validation.md](docs/validation.md) when the run completes.

## What you get

| | |
|---|---|
| Trap tables | NEON DP1.10022.001 stacked for all sites, release and provisional rows flagged |
| Specimen records | 141,690 NEON Biorepository records with their NEON identifiers, plus GBIF IDs |
| Image manifest | 53,792 images from four pools, each with source URL, sha256 and NEON join keys |
| Masks | one instance mask per specimen, plus head / pronotum / elytra part masks (COCO RLE) |
| Measurements | area, length, width, body length, elytra length and width, pronotum width, colour; in mm where the photo has a readable scale |
| Validation | agreement with human boxes and human elytra / pronotum measurements |

Most NEON pitfall specimens have never been photographed (about 1% of the Biorepository's pinned
vouchers have an image), so the trap tables cover every site while measurements cover only the
imaged specimens. [docs/data-sources.md](docs/data-sources.md) has the details.

## Using the data

The lake lives at `/iplant/home/tswetnam/neon-beetle-seg/ducklake/` on the CyVerse Data Store.
Copy that folder locally (`gocmd get`), then from inside it:

```sql
INSTALL ducklake; LOAD ducklake;
ATTACH 'ducklake:beetles.ducklake' AS lake (READ_ONLY);

-- median body length per species at each site, for specimens with a scale and complete parts
SELECT siteID, scientificName, count(*) AS n, round(median(body_length_parts_mm), 2) AS body_mm
FROM lake.measurements
WHERE qc_has_scale AND parts_complete AND NOT touches_edge AND source <> 'sentinel'
GROUP BY ALL ORDER BY n DESC;
```

`beetles.duckdb` in the same folder is a plain DuckDB copy that needs no extension, and
`export/*.parquet` has one file per table. `table_catalog` lists every table;
[docs/schema.md](docs/schema.md) is the data dictionary.

## Running it

```bash
git clone https://github.com/tyson-swetnam/neon-beetle-seg.git && cd neon-beetle-seg
scripts/bootstrap.sh            # uv environment (Python 3.12, PyTorch CUDA 12.6), GPU check

.venv/bin/nbs biorepo           # Biorepository records and image list
.venv/bin/nbs images            # download the Biorepository images (1.8 GB)
.venv/bin/nbs hf                # Imageomics datasets from HuggingFace (about 36 GB)
.venv/bin/nbs gbif              # GBIF records
.venv/bin/nbs neon              # NEON trap tables, all sites (needs NEON_TOKEN)
.venv/bin/nbs manifest          # one row per image
scripts/run_segmentation.sh     # detect, segment, measure: all four pools (hours on the GPU)
.venv/bin/nbs scale             # millimetre scale per image
.venv/bin/nbs tables            # collect results
.venv/bin/nbs validate          # compare with human annotations
.venv/bin/nbs lake              # build the DuckLake
.venv/bin/nbs report            # HTML report
.venv/bin/nbs upload            # sync to the CyVerse Data Store
```

Credentials go in `~/.neon-beetle-secrets.env`; see [docs/environment.md](docs/environment.md).
Tests: `.venv/bin/python -m pytest`.

### Picking it back up on a fresh VM

The CyVerse VICE container is ephemeral: only this repository and the Data Store collection
survive. To resume, clone, run `scripts/bootstrap.sh`, recreate the secrets file, and re-run the
steps you need. Downloads and segmentation are resumable; everything is re-creatable from the
sources, and `image_manifest` records the URL and sha256 of every image used.

## Documentation

| | |
|---|---|
| [docs/data-sources.md](docs/data-sources.md) | what exists at NEON, the Biorepository, GBIF and HuggingFace; counts, licences, join keys, citations |
| [docs/pipeline.md](docs/pipeline.md) | each step, how an image is processed, what is measured, quality flags |
| [docs/models.md](docs/models.md) | models, licences, and why each was chosen |
| [docs/validation.md](docs/validation.md) | measured accuracy and known biases |
| [docs/schema.md](docs/schema.md) | data dictionary |
| [docs/environment.md](docs/environment.md) | the VM, what persists, setup, secrets, measured GPU speed |
| [docs/history.md](docs/history.md) | what changed from the v0.1.0 pilot |

## Licence

Code: MIT. The optional YOLO detector depends on Ultralytics (AGPL-3.0) and is not installed by
default. Data keep their sources' licences: NEON tables and Biorepository records CC BY 4.0,
Biorepository images and the 2018 tray photos CC BY-SA 4.0, Hawaii and sentinel images CC BY 4.0.
Derived masks and measurements of CC BY-SA images are shared under CC BY-SA 4.0.
