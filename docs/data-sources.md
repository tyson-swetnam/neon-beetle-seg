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
| NEON Biorepository, beetles | 8,290 | pinned vouchers, 2016 individual photos, 2018 ethanol trays | `individualID`, sample ID, vial barcode |
| NEON Biorepository, herptile bycatch | 6,153 | reptiles and amphibians caught in the same traps | sample ID hash, vial barcode |
| `imageomics/2018-NEON-beetles` | 577 | full-resolution ethanol tray photos, 30 sites, 2018 | vial barcode, `NEON_sampleID` |
| `imageomics/Hawaii-beetles` | 420 | pinned tray photos, PUUM, 2018 to 2024 | `individualID` |
| `imageomics/sentinel-beetles` | 44,510 | individual pinned crops | nothing: identifiers are anonymised |

## NEON: DP1.10022.001, ground beetles sampled from pitfall traps

- 47 sites, 20 domains, 539 plots, collection dates 2013-07-01 to 2026-09-16.
- 156,835 trap-bout rows (133,850 with a sample collected), 241,495 sorting rows, 145,906 pinned
  individuals (95,751 with an expert identification), 826 taxa among the pinned individuals.
- Latest release: **RELEASE-2026** (DOI [10.48443/q9ne-6b77](https://doi.org/10.48443/q9ne-6b77)),
  covering 2013-07 to 2023-12. Later months are **provisional**: no DOI, and values can change.
  Every `neon_*` row carries a `release` column so the two can be separated.
- Licence CC BY 4.0. Downloading needs a NEON API token; the data endpoint returns 403 without one.
- Downloaded and stacked with the official `neonutilities` package, expanded package: 2,988
  monthly site files, 963 MB. Version 2.0.2 makes an anonymous connectivity check before every
  API call, which exhausts the unauthenticated rate limit on a product this size and aborts with
  "Cannot access NEON API"; `stack_neon.py` works around it by sending the token with every
  request.
- NEON distributes no specimen images with this product.

Tables: `bet_fielddata` (one row per trap per bout), `bet_sorting` (one row per taxon per
sample; its `sampleType` separates carabids from invertebrate, herptile and mammal bycatch), `bet_parataxonomistID` (one row per pinned individual), `bet_expertTaxonomistIDProcessed`
and `...Raw`, `bet_archivepooling`, `bet_identificationHistory`, `bet_bycatchIDHistory`.

## NEON Biorepository (Arizona State University)

The Symbiota portal at `biorepo.neonscience.org` publishes each collection as a Darwin Core
Archive at `https://biorepo.neonscience.org/portal/content/dwca/<CODE>_DwC-A.zip`. No account is
needed. Records are CC BY 4.0; images are CC BY-SA 4.0.

A pitfall sample is sorted into carabids, other invertebrates, reptiles and amphibians, and small
mammals. Some carabids are pinned; everything else is kept in ethanol, first as one vial per trap
(trap sorting), then pooled per plot and bout (archive pooling). Eleven collections hold the
result:

| Collection | Archive code | Group | Level | Records | With images | Images |
|---|---|---|---|---|---|---|
| Carabid Collection (Pinned Vouchers) | `NEON-CARC-PV` | pinned carabid | individual | 98,168 | 1,105 | 1,239 |
| Pinned Carabids, NEON Domain Offices | `NEON-DCTC` | pinned carabid | individual | 10,489 | 728 | 739 |
| Carabid Collection (Trap Sorting) | `NEON-CARC-TS` | bulk carabid | trap sorting | 5,497 | 308 | 1,702 |
| Carabid Collection (Archive Pooling) | `NEON-CARC-AP` | bulk carabid | archive pooling | 5,964 | 653 | 4,477 |
| Carabid Collection (DNA Extracts) | `NEON-CARC-DNA` | carabid DNA extract | individual | 20,945 | 7 | 9 |
| Invertebrate Voucher Collection at ASU | `ASU-NEON-IV` | invertebrate voucher | individual | 627 | 64 | 124 |
| Invertebrate Bycatch Collection (Trap Sorting) | `NEON-IVBC-TS` | invertebrate bycatch | trap sorting | 5,787 | 0 | 0 |
| Invertebrate Bycatch Collection (Archive Pooling) | `NEON-IVBC-AP` | invertebrate bycatch | archive pooling | 37,472 | 3 | 3 |
| Herptile Voucher Collection (Ground Beetle Sampling Bycatch Trap Sorting) | `NEON-HEVC-GBTS` | herptile bycatch | trap sorting | 3,078 | 2,745 | 5,511 |
| Herptile Voucher Collection (Ground Beetle Sampling Bycatch Archive Pooling) | `NEON-HEVC-GBAP` | herptile bycatch | archive pooling | 352 | 328 | 654 |
| Mammal Collection (Vouchers [Ground Beetle Sampling Bycatch]) | `NEON-MAMC-VGB` | mammal bycatch | individual | 143 | 21 | 53 |
| **Total** | | | | **188,522** | | **14,511** |

`biorepo_collections` has one row per collection with its description, preservation, and the
NEON table and field its sample IDs come from (for example `bet_sorting_in.subsampleID.herp`).
Three other pinned-carabid collections are listed on the portal but have no usable archive (the
Essig Museum's returns 404; Carnegie and Bishop Museum publish none).

What each bycatch group contains:

- **Invertebrate bycatch** is unsorted: every record is "Bulk Terrestrial Invertebrates" in
  ethanol, with no counts or lower taxa. Its value is as an index of what exists and where.
- **Herptile bycatch** is identified, mostly to species (3,269 of 3,430 records; 2,858 amphibians,
  572 reptiles), counted, and photographed: 3,073 records have photos, usually two per vial.
- **Mammal bycatch** is 143 small mammals (115 shrews and moles, 28 rodents). 118 of them carry
  body measurements as JSON in `dynamicProperties` (total length, tail, hind foot, ear, mass).

### Images

Twelve of the 14,511 image rows are second uploads of the same file for the same record and are
dropped, leaving 14,499. 14,453 of the 14,455 beetle and herptile images downloaded (6.56 GB).
The 56 mammal and invertebrate bycatch photos are catalogued in `biorepo_images` with their URLs
but not downloaded.

| `image_kind` | Images | Notes |
|---|---|---|
| `individual` | 5,668 | one ethanol carabid per photo, dorsal and ventral, 2016. These are small crops (median about 110 x 190 px) with no scale in frame. |
| `pinned` | 2,102 | one pinned carabid. CARC-PV has two series: macro shots on white with a printed scale bar, and shots on a grid backdrop with no scale. DCTC photos have labels, rulers and colour cards in frame. 16 are non-carabid records in the mixed ASU voucher collection and are skipped. |
| `tray_ethanol` | 518 | all 518 are downsized copies of the HuggingFace 2018 tray photos, so they are marked `duplicate_of` and not processed twice. |
| `bycatch` (herptile) | 6,153 | reptiles and amphibians on a white backdrop with a metric ruler and colour card; 3,200 px wide or larger. Segmented through their own route. |
| `bycatch` (other) | 56 | mammal and invertebrate bycatch; catalogued only. |

### Identifiers

`otherCatalogNumbers` holds the NEON identifiers as `label: value` pairs separated by `;`
(GBIF serialises the same field with `|`):

| Label | Example | Meaning |
|---|---|---|
| `NEON sampleID` | `NEON.BET.D16.004428` (pinned) or `GRSM_012.20160906.PTEACU1.01` (vial) | the NEON `individualID` for pinned vouchers; a subsample or pooled-vial ID for ethanol collections |
| `NEON sampleCode (barcode)` | `A00000121908` | vial barcode; equals `pictureID` in the 2018 dataset |
| `NEON sampleUUID` | | NEON sample UUID |

| `NEON sampleID Hash` | `JDiKFzN43rr...` | a hash of the sample ID. **This is the key that links to NEON's tables**: `bet_sorting.subsampleID` and `bet_archivepooling.archiveVialID` are published as these hashes, not as readable IDs |

`catalogNumber` is the Biorepository IGSN (e.g. `NEON08HSH`) and `occurrenceID` its DOI.
Domain-office (DCTC) records have no catalog number.

`preserved_samples` uses the hash to link every vial in NEON's sorting and archive tables to
its accession. From the Biorepository side the match is nearly complete: all 11,461 bulk-carabid
records, all 3,430 herptile records, 43,243 of 43,259 invertebrate-bycatch records and 140 of 143
mammal records correspond to a NEON sample. From NEON's side most trap-sorting vials have no
accession of their own, because they were pooled into archive vials.

## GBIF

NEON publishes nine of the eleven collections to GBIF (publisher key
`e794e60e-e558-4549-99f8-cfb241cdce24`); the two invertebrate-bycatch collections are not there.
Every GBIF media URL for them points at `biorepo.neonscience.org`, so **GBIF adds no images**. It
adds a stable `gbifID` per record, the GBIF backbone taxonomy match, and a citable download.

| Collection | GBIF dataset key | DOI | Records |
|---|---|---|---|
| CARC-PV | `8cb7c449-ba11-4464-865e-8029b8d772e8` | 10.15468/zyx3fn | 98,168 |
| CARC-DNA | `44262c91-b3fd-48e4-8e47-1ee03ac2d496` | 10.15468/smm5vp | 20,945 |
| DCTC | `69e5ceb4-30a6-4074-8f9d-d6a0457cb789` | 10.15468/t6ctuu | 10,489 |
| CARC-AP | `044d870e-5718-410a-9450-9c2ceac8e1d9` | 10.15468/xicbza | 5,964 |
| CARC-TS | `2564e9e2-0248-4a3b-8344-24fc0956ed73` | 10.15468/mjtykf | 5,497 |
| HEVC-GBTS | `c113d947-5dcf-4a52-aaa2-735dba1089b7` | 10.15468/zhkuay | 3,078 |
| NEON-IV | `69f3ca43-ed03-4ed1-92c4-58f9bd0eafc1` | 10.15468/vn96gr | 627 |
| HEVC-GBAP | `c8e9fe8e-391b-44b9-ad94-a5d15f4b5788` | 10.15468/5ta21z | 352 |
| MAMC-VGB | `d7b80380-1156-4d50-a261-c5b1ac1ea59a` | 10.15468/naaenf | 143 |

`gbif_occurrences` holds all 145,263 records, from an authenticated download:
**GBIF.org (2026-10-06) GBIF Occurrence Download https://doi.org/10.15468/dl.k4ajhj**.
GBIF joins to the Biorepository on `catalogNumber`.

How `nbs gbif` gets them:

- With `GBIF_USER`, `GBIF_PWD` and `GBIF_EMAIL` set, it posts a download request to
  `/occurrence/download/request` (HTTP Basic auth; there is no API key), waits for GBIF to
  prepare the file, and records the download DOI.
- Without credentials it pages the anonymous search API. GBIF's search becomes very slow at deep
  offsets (more than two minutes a page past about 30,000), so each dataset is sliced by year
  and state. That reached 140,987 of 141,078 carabid records in testing; the rest have no year or
  state.

GBIF is also used for the herptile identification suggestions: backbone name matching and, for
each site, which species people have observed nearby (see [pipeline.md](pipeline.md)).

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
