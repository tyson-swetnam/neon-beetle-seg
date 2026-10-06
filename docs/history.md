# History

## v0.2.0 (October 2026): all sites, all available images

### What the pilot was

v0.1.0 (2026-09-15) covered eight sites in Colorado, Arizona, New Mexico and Utah for 2025. It
stacked 44 provisional site-months of DP1.10022.001, measured 270 Biorepository voucher photos
(one mask each, in pixels) and 63 specimens on 8 demonstration trays, and published a lake and a
report to the Data Store.

Its pipeline code, images and masks were never uploaded: only the tables and the report
survived. v0.2.0 rebuilds the pipeline from the report's description, in this repository. The
pilot's files are kept unchanged on the Data Store under `archive/v0.1.0-pilot/`.

### What changed

| | v0.1.0 pilot | v0.2.0 |
|---|---|---|
| NEON tables | 8 sites, 2025, provisional | all sites, 2013 onward, release and provisional flagged |
| Biorepository | 270 vouchers found through GBIF by state and year | every image in six collections, from the Darwin Core Archives |
| Other images | 8 demonstration trays | 577 trays (2018), 417 Hawaii trays, 44,510 sentinel crops |
| Bulk and bycatch | NEON tables only, 8 sites | all 11 Biorepository collections; every ethanol vial linked to its accession; 6,149 herptile photos segmented, with GBIF-assisted species suggestions |
| Image manifest | one row per specimen; extra views dropped | one row per image |
| Detector | YOLOv8m beetle detector (Ultralytics) | Grounding DINO |
| Segmenter | MobileSAM | SAM 2.1 (large; base-plus for sentinel) |
| Body parts | none | BeetleFlow head / pronotum / elytra |
| Scale | Zooniverse bar on trays; vouchers in pixels | checkerboard, ruler ticks, printed bar + OCR, Hawaii annotations |
| Masks | RGBA crop files, not uploaded | COCO RLE in the lake |
| Validation | 8 trays | all annotated trays, against boxes and elytra / pronotum lines |
| Code | not preserved | this repository, with a locked environment and tests |

### Things found along the way

- **The pilot's lake had a stale data path.** Its catalog recorded `data/ducklake/data/`, the path
  on the machine that built it, and carried 54 dead parquet files from three superseded builds.
  The new lake is built from inside its own directory with a relative path, from scratch.
- **The pilot tagged `elytra_annotations_2018.dist_cm` as millimetres.** It is centimetres; the
  column tag now says so.
- **GBIF is not a second source of images.** Every media link on NEON's GBIF records points back
  to the Biorepository.
- **The Biorepository's 518 tray photos are downsized copies** of the 2018 HuggingFace trays.
  They are marked as duplicates and processed once, at full resolution.
- **Two Zooniverse workflows in the 2018 annotations use a different pixel frame.** For
  workflows 21652 and 21827 the line coordinates are in a frame roughly 10 to 20% smaller than the
  dataset's `resized_image_dim` column says, by an amount that varies from photo to photo. Scaling
  coordinates and scale bars by that column therefore puts lines beside their specimens and makes
  the scale too small. The scale now comes from the checkerboard in each photo; lines from those
  two workflows are not used to pair human measurements with specimens.
- **The 2018 volunteers measured elytra width at the base of the elytra**, not at the widest
  point, so predicted maximum width reads about 30% larger. The comparison uses a width taken at
  the base.
- **`flattened_images.zip` in sentinel-beetles is incomplete.** At revision `11d861b5`, 24,856
  of its 44,510 entries are git-lfs pointer stubs. The parquet shards are complete.
- **SAM returned the backdrop for half of the 2016 individual photos.** On a tight crop the prompt
  box is the whole frame, and SAM segmented the plain backdrop instead of the beetle on 2,965 of
  5,668 photos. A random contact sheet showed it; summary statistics had not, because a
  backdrop mask is a perfectly well-formed mask. Such masks are now recognised by how much of
  the frame's border they cover, and inverted.
- **Whole-photo detection collapses on crowded trays.** One tray of 266 small beetles yielded a
  single box. A tiled pass now takes over when it finds clearly more specimens.
- **NEON publishes its sample IDs hashed.** `bet_sorting.subsampleID` and
  `bet_archivepooling.archiveVialID` are hashes, not the readable IDs in older documentation. The
  Biorepository publishes the same hash for each record, and that is what links the two.
- **neonutilities 2.0.2 cannot download this product as shipped.** It makes an anonymous
  connectivity check before every API call; a product with 2,988 site-months exhausts the
  unauthenticated rate limit and the download aborts.
- **GBIF's search API is too slow to page a large dataset.** Past an offset of about 30,000 a
  page takes more than two minutes. The anonymous fallback slices by year and state instead.
- **`git push` hangs on this VM image** because the global credential helper is interactive.
  `scripts/bootstrap.sh` points the repository at the `gh` login.
