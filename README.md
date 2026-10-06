# neon-beetle-seg

Segmentation and body measurements of ground beetles (Carabidae) from
[NEON](https://www.neonscience.org/) pitfall traps, for every image of a NEON pitfall specimen
that could be found, linked back to NEON's trap records for all 47 sites.

The pipeline pulls records and images from NEON, the NEON Biorepository, GBIF and HuggingFace,
runs open segmentation models on a single 16 GB GPU, and publishes the result as a
[DuckLake](https://ducklake.select/) lakehouse on the CyVerse Data Store.

**Status: v0.2.0** (October 2026). Published to the CyVerse Data Store at
`/iplant/home/tswetnam/neon-beetle-seg/` on 2026-10-06 and verified against the local build (all
93 lake files identical by sha256). The graphical report is `report/report.html` there and
[docs/report.html](docs/report.html) here. The v0.1.0 pilot is kept unchanged under
`archive/v0.1.0-pilot/`.

## Results

| | |
|---|---|
| NEON trap tables | 47 sites, 20 domains, 539 plots, July 2013 to September 2026; 145,906 pinned individuals of 826 taxa |
| Preserved samples | 286,617 ethanol vials (bulk carabids and invertebrate, herptile and mammal bycatch) linked to their Biorepository accessions |
| Biorepository records | 188,522 in 11 collections; 145,263 of them also on GBIF |
| Images processed | 53,257 beetle images from four pools, plus 6,149 herptile bycatch photos |
| Beetles segmented | 65,830 specimens; 57,023 with a millimetre scale; 64,102 with elytra measured |
| Linked to a NEON site | 20,698 beetle specimens at 46 sites (the 44,510 sentinel crops are anonymised and cannot be linked) |
| Herptile bycatch | 9,021 outlines from 6,149 photos of 3,072 records at 41 sites, all with a millimetre scale; species suggestions for 64 of the 161 records not identified to species |

How well it works, against human annotations ([docs/validation.md](docs/validation.md)):

- **Finding beetles in tray photos:** 99.6% of 10,183 annotated individuals have a mask;
  precision 0.971 and recall 0.996 against 11,655 human boxes.
- **Elytra length:** r = 0.98 with human lines on 10,045 ethanol specimens, but it reads about
  0.8 mm (8%) long. Treat absolute lengths as biased high.
- **Elytra width:** median error 6.0% at the base (ethanol trays), 2.0% at the widest point
  (pinned, Hawaii).
- **Scale readers:** within 0.6% of human scale bars where that could be checked.
- **Herptile bycatch:** the number of outlines equals the recorded number of animals on 91.5% of
  photos. Species suggestions are right 99.8% of the time when GBIF leaves one candidate, and
  about three times in four when they rest on the image model.

Things to know before using the numbers:

- Most NEON pitfall specimens have never been photographed (about 1% of the Biorepository's
  pinned vouchers have an image), so the trap tables cover every site while measurements cover
  only the imaged specimens.
- 8,807 beetle specimens have no millimetre scale and stay in pixels: the 5,668 2016 individual
  photos, 1,259 pinned photos without a readable bar, and 1,880 sentinel and tray specimens
  whose scale could not be read.
- Masks were spot-checked, not reviewed one by one. Filter on the quality flags in
  [docs/pipeline.md](docs/pipeline.md#quality-flags).
- Post-2023 NEON rows are provisional and can change; they are flagged in every table.

## What you get

| | |
|---|---|
| Trap tables | NEON DP1.10022.001 stacked for all sites, release and provisional rows flagged |
| Specimen records | 188,522 NEON Biorepository records (pinned and bulk carabids, DNA extracts, bycatch) with their NEON identifiers, plus GBIF IDs |
| Preserved samples | every ethanol vial in NEON's sorting and archive tables, linked to its accession |
| Image manifest | 59,945 images from five pools, each with source URL, sha256 and NEON join keys |
| Masks | one instance mask per specimen, plus head / pronotum / elytra part masks (COCO RLE) |
| Measurements | area, length, width, body length, elytra length and width, pronotum width, colour; in mm where the photo has a readable scale |
| Herptile bycatch | masks and lengths for the photographed reptiles and amphibians, and GBIF-assisted species suggestions for the ones not identified to species |
| Validation | agreement with human boxes and human elytra / pronotum measurements |

## Using the data

The lake lives at `/iplant/home/tswetnam/neon-beetle-seg/ducklake/` on the CyVerse Data Store.
Copy that folder locally (`gocmd get`), then from inside it:

```sql
INSTALL ducklake; LOAD ducklake;
ATTACH 'ducklake:beetles.ducklake' AS lake (READ_ONLY);

-- median body length per species at each site, for specimens with a scale and complete parts
SELECT siteID, scientificName, count(*) AS n, round(median(body_length_parts_mm), 2) AS body_mm
FROM lake.measurements
WHERE qc_has_scale AND parts_complete AND NOT trunk_touches_edge AND source NOT IN ('sentinel', 'herp')
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
.venv/bin/nbs images            # download the Biorepository beetle and herptile images (6.6 GB)
.venv/bin/nbs hf                # Imageomics datasets from HuggingFace (about 36 GB)
.venv/bin/nbs gbif              # GBIF records
.venv/bin/nbs neon              # NEON trap tables, all sites (needs NEON_TOKEN)
.venv/bin/nbs manifest          # one row per image
scripts/run_segmentation.sh     # detect, segment, measure: all five pools (many GPU hours)
.venv/bin/nbs scale             # millimetre scale per image
.venv/bin/nbs tables            # collect results
.venv/bin/nbs validate          # compare with human annotations
.venv/bin/nbs herps             # GBIF-assisted species suggestions for herptile bycatch
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
