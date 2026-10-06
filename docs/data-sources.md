# Data sources

Counts are as downloaded on 2026-10-06. `run_provenance` in the lake records the exact dataset
revisions used for a given build.

## What exists, and what does not

NEON's pitfall programme covers 47 sites in all 20 domains, and the tabular record of what was
caught is complete. **Photographs are not**: most pinned specimens have never been imaged. Of
98,168 pinned vouchers at the Biorepository, 1,105 (about 1%) have a photo. "All of NEON's pitfall
traps" is therefore achievable for the tables, while image-based measurements cover only the
specimens someone photographed:

| Pool | Images | What they are | Joins to NEON by |
|---|---|---|---|
| NEON Biorepository | 8,290 | pinned vouchers, 2016 individual photos, 2018 ethanol trays | `individualID`, sample ID, vial barcode |
| `imageomics/2018-NEON-beetles` | 577 | full-resolution ethanol tray photos, 30 sites, 2018 | vial barcode, `NEON_sampleID` |
| `imageomics/Hawaii-beetles` | 420 | pinned tray photos, PUUM, 2018 to 2024 | `individualID` |
| `imageomics/sentinel-beetles` | 44,510 | individual pinned crops | nothing: identifiers are anonymised |

## NEON: DP1.10022.001, ground beetles sampled from pitfall traps

- 47 sites, 20 domains, 2013-07 to 2026-09.
- Latest release: **RELEASE-2026** (DOI [10.48443/q9ne-6b77](https://doi.org/10.48443/q9ne-6b77)),
  covering 2013-07 to 2023-12. Later months are **provisional**: no DOI, and values can change.
  Every `neon_*` row carries a `release` column so the two can be separated.
- Licence CC BY 4.0. Downloading needs a NEON API token; the data endpoint returns 403 without one.
- Downloaded and stacked with the official `neonutilities` package, expanded package.
- NEON distributes no specimen images with this product.

Tables: `bet_fielddata` (one row per trap per bout), `bet_sorting` (one row per taxon per
sample), `bet_parataxonomistID` (one row per pinned individual), `bet_expertTaxonomistIDProcessed`
and `...Raw`, `bet_archivepooling`, `bet_identificationHistory`, `bet_bycatchIDHistory`.

## NEON Biorepository (Arizona State University)

The Symbiota portal at `biorepo.neonscience.org` publishes each collection as a Darwin Core
Archive at `https://biorepo.neonscience.org/portal/content/dwca/<CODE>_DwC-A.zip`. No account is
needed. Records are CC BY 4.0; images are CC BY-SA 4.0.

| Collection | Archive code | Records | Records with images | Images |
|---|---|---|---|---|
| Carabid Collection (Pinned Vouchers) | `NEON-CARC-PV` | 98,168 | 1,105 | 1,239 |
| Carabid Collection (DNA Extracts) | `NEON-CARC-DNA` | 20,945 | 7 | 9 |
| Pinned Carabids, NEON Domain Offices | `NEON-DCTC` | 10,489 | 728 | 739 |
| Carabid Collection (Archive Pooling) | `NEON-CARC-AP` | 5,964 | 653 | 4,477 |
| Carabid Collection (Trap Sorting) | `NEON-CARC-TS` | 5,497 | 308 | 1,702 |
| Invertebrate Voucher Collection at ASU | `ASU-NEON-IV` | 627 | 64 | 124 |
| **Total** | | **141,690** | | **8,290** |

Of the 8,290 images, 8,288 downloaded (1.83 GB); 2 failed. They are of three kinds:

| `image_kind` | Images | Notes |
|---|---|---|
| `individual` | 5,668 | one ethanol specimen per photo, dorsal and ventral, 2016. These are small crops (median about 110 x 190 px) with no scale in frame. |
| `pinned` | 2,102 | one pinned voucher. CARC-PV has two series: macro shots on white with a printed scale bar, and shots on a grid backdrop with no scale. DCTC photos have labels, rulers and colour cards in frame. |
| `tray_ethanol` | 518 | all 518 are downsized copies of the HuggingFace 2018 tray photos, so they are marked `duplicate_of` and not processed twice. |

19 images belong to non-carabid records in the mixed ASU voucher collection and are skipped.

### Identifiers

`otherCatalogNumbers` holds the NEON identifiers as `label: value` pairs separated by `;`
(GBIF serialises the same field with `|`):

| Label | Example | Meaning |
|---|---|---|
| `NEON sampleID` | `NEON.BET.D16.004428` (pinned) or `GRSM_012.20160906.PTEACU1.01` (vial) | the NEON `individualID` for pinned vouchers; a subsample or pooled-vial ID for ethanol collections |
| `NEON sampleCode (barcode)` | `A00000121908` | vial barcode; equals `pictureID` in the 2018 dataset |
| `NEON sampleUUID` | | NEON sample UUID |

`catalogNumber` is the Biorepository IGSN (e.g. `NEON08HSH`) and `occurrenceID` its DOI.

## GBIF

NEON publishes the same collections to GBIF (publisher key
`e794e60e-e558-4549-99f8-cfb241cdce24`). Every GBIF media URL for them points at
`biorepo.neonscience.org`, so **GBIF adds no images**. It adds a stable `gbifID` per record and the
GBIF backbone taxonomy match, and an authenticated download gives a citable DOI.

| Collection | GBIF dataset key | DOI |
|---|---|---|
| CARC-PV | `8cb7c449-ba11-4464-865e-8029b8d772e8` | 10.15468/zyx3fn |
| CARC-DNA | `44262c91-b3fd-48e4-8e47-1ee03ac2d496` | 10.15468/smm5vp |
| DCTC | `69e5ceb4-30a6-4074-8f9d-d6a0457cb789` | 10.15468/t6ctuu |
| CARC-AP | `044d870e-5718-410a-9450-9c2ceac8e1d9` | 10.15468/xicbza |
| CARC-TS | `2564e9e2-0248-4a3b-8344-24fc0956ed73` | 10.15468/mjtykf |
| NEON-IV | `69f3ca43-ed03-4ed1-92c4-58f9bd0eafc1` | 10.15468/vn96gr |

`nbs gbif` uses an authenticated download when GBIF credentials are set, and otherwise pages the
anonymous occurrence search per dataset (each is under the API's 100,000-record paging limit).
GBIF joins to the Biorepository on `catalogNumber`.

## HuggingFace (Imageomics)

| Dataset | Revision | Licence | What is used |
|---|---|---|---|
| [`imageomics/2018-NEON-beetles`](https://huggingface.co/datasets/imageomics/2018-NEON-beetles) | `61034699` | CC BY-SA 4.0 | 577 tray photos (5568 x 3712), `BeetleMeasurements.csv` |
| [`imageomics/Hawaii-beetles`](https://huggingface.co/datasets/imageomics/Hawaii-beetles) | `a5f5bdac` | CC BY 4.0 | 420 tray photos (6960 x 4640), `trait_annotations.csv`, `images_metadata.csv` |
| [`imageomics/sentinel-beetles`](https://huggingface.co/datasets/imageomics/sentinel-beetles) | `11d861b5` | CC BY 4.0 | 44,510 crops, 1,421 ruler crops |

**2018-NEON-beetles.** `BeetleMeasurements.csv` has 39,064 rows: an elytra length line and an
elytra width line for each annotated specimen, drawn by three Zooniverse volunteers, plus the 1 cm
scale bar per photo. Coordinates are in a resized copy of the photo; they are scaled to full
resolution here (two rows of the dataset's own scaled-up column are placeholders). One tray has
no usable scale. The human specimen boxes for all 577 trays (11,655 boxes) come from
[`Imageomics/carabidae_beetle_processing`](https://github.com/Imageomics/carabidae_beetle_processing).

**Hawaii-beetles.** 417 of the 420 tray photos have trait annotations: scale bar, elytra maximum
length, elytra maximum width and basal pronotum width, as line endpoints in tray-photo pixels,
keyed by NEON `individualID`.

**sentinel-beetles.** `siteID`, `domainID` and `eventID` are anonymised integers and `public_id`
is an opaque number, so these specimens cannot be linked to NEON records; only `scientificName`
and `collectDate` are real. Two things to know when downloading it:

- At this revision, **24,856 of the 44,510 files in `flattened_images.zip` are git-lfs pointer
  stubs, not images**. The images are read from the parquet shards under `data/` instead
  (20 GB), which are complete.
- Each crop is linked to a crop of the ruler from the same tray photo (`scalebar_path`). The
  millimetre scale used here assumes the ruler crop and the specimen crop are at the same
  resolution, which the dataset card implies but does not state.

## Join keys

| From | Key | To |
|---|---|---|
| `neon_parataxonomistID.individualID` | = Biorepository `NEON sampleID` (CARC-PV, DCTC) | `biorepo_records.individualID`, `image_manifest.individualID`, Hawaii `individualID` |
| Vial barcode | Biorepository `NEON sampleCode (barcode)` | 2018 `pictureID`, `image_manifest.neon_barcode` |
| Pooled vial or subsample ID | Biorepository `NEON sampleID` (CARC-AP, CARC-TS) | 2018 `NEON_sampleID`, `image_manifest.neon_sampleID` |
| Biorepository `catalogNumber` | | `gbif_occurrences.catalogNumber` |
| `siteID`, `plotID`, `eventDate` | | coarse joins everywhere |

## Citing

- NEON (National Ecological Observatory Network). Ground beetles sampled from pitfall traps
  (DP1.10022.001), RELEASE-2026. https://doi.org/10.48443/q9ne-6b77 (plus provisional data, if used,
  with the access date).
- NEON Biorepository collections: cite the GBIF dataset DOIs above, or the GBIF download DOI
  recorded in `run_provenance`.
- `imageomics/2018-NEON-beetles`, doi:10.57967/hf/6890; `imageomics/Hawaii-beetles`, doi:10.57967/hf/7272;
  `imageomics/sentinel-beetles`, doi:10.57967/hf/8716. Use the citation given on each dataset card.
- Models: see [models.md](models.md) for the model cards; BeetleFlow is described in arXiv:2511.00255.
