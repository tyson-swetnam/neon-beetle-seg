# Pipeline

Each step is a module in `neon_beetle_seg/` and a sub-command of `nbs`. Steps write parquet tables
to `data/tables/`; `nbs lake` turns those into the DuckLake. Run them in this order:

| # | Command | Module | Needs | Writes |
|---|---|---|---|---|
| 1 | `nbs biorepo` | `fetch_biorepo.py` | network | `biorepo_records`, `biorepo_images` |
| 2 | `nbs images` | `fetch_images.py` | network | image files, `biorepo_image_files` |
| 3 | `nbs hf` | `fetch_hf.py` | network, ~36 GB disk | the three Imageomics datasets |
| 4 | `nbs gbif` | `fetch_gbif.py` | network (GBIF account optional) | `gbif_occurrences` |
| 5 | `nbs neon` | `stack_neon.py` | `NEON_TOKEN` | `neon_*`, `specimen_manifest` |
| 6 | `nbs manifest` | `manifest.py` | steps 1 to 3 | `image_manifest`, annotation tables |
| 7 | `nbs segment <pool>` | `segment.py` | GPU | `outputs/shards/<pool>/` |
| 8 | `nbs scale` | `scale.py` | GPU (small) | `image_scale` |
| 9 | `nbs tables` | `tables.py` | steps 7, 8 | `instances`, `measurements`, `image_results` |
| 10 | `nbs validate` | `validate.py` | step 9 | `validation_*` |
| 11 | `nbs sam3` | `sam3_compare.py` | GPU, `HF_TOKEN` with SAM 3 access | `sam3_comparison` (optional) |
| 12 | `nbs lake` | `build_ducklake.py` | everything above | `data/ducklake/` |
| 13 | `nbs report` | `report.py` | step 12 | `outputs/report/report.html` |
| 14 | `nbs upload` | `upload.py` | `gocmd` login | the Data Store collection |

`scripts/run_segmentation.sh` runs step 7 for all four pools in priority order. It is resumable:
each pool skips images already present in its shards.

## How an image is processed

`image_manifest.route` decides the path through step 7:

**`tray`** (2018 ethanol trays, Hawaii pinned trays): many specimens per photo.

1. Grounding DINO on the whole photo, prompt "a beetle.", box threshold 0.25.
2. Boxes covering more than 20% of the photo are dropped (the tray itself), then non-maximum
   suppression at IoU 0.5, which also drops a box mostly contained in a higher-scoring one.
   On ethanol trays a second, tiled pass (1856 px tiles, 25% overlap, boxes cut by an inner tile
   edge dropped) runs as well. Its result replaces the whole-photo result only when it finds
   more than 25% + 3 more specimens (`det_source = grounding_dino_tiled`): whole-photo detection
   is more precise on ordinary trays but collapses on trays of a hundred or more small beetles.
3. SAM 2.1 gives one mask per box, each on its own padded crop.
4. BeetleFlow labels head, pronotum, elytra, legs and antennae on a tight crop of each specimen;
   labels outside the specimen's own mask are discarded so neighbours do not leak in.

**`single`** (Biorepository pinned vouchers and individual photos): one specimen, with labels,
rulers, grids or colour cards around it.

1. Grounding DINO as above; the most confident box wins, larger boxes winning near-ties.
2. If nothing is detected, the largest blob that differs from the backdrop colour becomes the
   box (`det_source = backdrop_blob`); failing that, the whole frame (`full_frame`).
3. SAM 2.1 and BeetleFlow as above. When the box is the whole frame (a tight crop on a plain
   backdrop) SAM often returns the backdrop instead of the specimen. A mask covering more than
   half of its window's border is therefore replaced by the largest blob of its complement
   (`mask_fix = inverted_backdrop`).

**`crop`** (sentinel-beetles): the image is already a tight crop of one pinned specimen.

1. BeetleFlow on the whole crop.
2. If it found head, pronotum and elytra, the box around them prompts SAM 2.1
   (`det_source = beetleflow_extent`); otherwise the whole frame does. No detector is run.

## What is measured

All from the mask, in pixels (`measure.py`); millimetres are derived in step 9 when the image has
a trusted scale.

| Column | Meaning |
|---|---|
| `area_px`, `perimeter_px` | mask area and outline length (largest blob, holes filled) |
| `length_px`, `width_px` | sides of the minimum-area rectangle around the whole mask, legs and antennae included |
| `core_length_px`, `core_width_px` | the same after a morphological opening removes thin appendages |
| `ellipse_major_px`, `ellipse_minor_px`, `orientation_deg` | fitted ellipse and rectangle angle |
| `solidity` | mask area / convex hull area |
| `mean_r`, `mean_g`, `mean_b` | mean colour inside the mask |
| `body_length_parts_px` | extent of head + pronotum + elytra along the body axis; the preferred body length |
| `elytra_length_px`, `elytra_width_px` | extent of the elytra along and across the body axis |
| `pronotum_length_px`, `pronotum_width_px`, `head_width_px` | likewise for pronotum and head |
| `elytra_midline_length_px` | elytra length along the midline, base to apex; how both annotated datasets define elytra length |
| `elytra_base_width_px`, `pronotum_base_width_px` | widths at the pronotum-elytra junction (the 2018 volunteers' elytra width; Hawaii's basal pronotum width) |

The body axis is the principal axis of the head, pronotum and elytra pixels together. Shape
metrics on masks larger than 768 px are computed on a downscaled copy and scaled back; areas are
exact.

## Scale

| `scale_source` | Where | How |
|---|---|---|
| `checkerboard` | 2018 trays | the side of the white squares of the 1 cm checkerboard in each photo, measured from the image. The volunteers' scale-bar annotation is not used: two of the five Zooniverse workflows record it in a different pixel frame (see [history.md](history.md)) |
| `hawaii_scalebar` | Hawaii trays | annotated 1 cm bar |
| `ruler_ticks` | sentinel | each crop is linked to a crop of the millimetre ruler from the same tray photo; the tick spacing is the dominant period of the dark-stroke profile |
| `printed_bar` | Biorepository macro photos | the thin printed bar is located as a long isolated horizontal line and its label ("5 mm", "1 mm") is read by OCR |
| none | everything else | measurements stay in pixels |

Photos with no trusted scale are the Biorepository grid-backdrop series, most domain-office
(DCTC) photos, and the 2016 individual photos, which are small crops with no scale in frame.

## Masks

`instances.mask_rle` is a COCO compressed RLE of the mask inside a window of the image:
`(win_x1, win_y1)` is the window's top-left corner and `(mask_w, mask_h)` its size. Part masks
(`head_rle`, `pronotum_rle`, `elytra_rle`) use the window `(pwin_x1, pwin_y1, parts_w, parts_h)`.

```python
from neon_beetle_seg.measure import rle_decode
mask = rle_decode(row.mask_rle, row.mask_h, row.mask_w)   # bool array
# its top-left pixel is image pixel (row.win_x1, row.win_y1)
```

## Quality flags

Filter on these before analysis:

| Column | Meaning |
|---|---|
| `qc_has_scale` | millimetre values are available |
| `parts_ok` | BeetleFlow found elytra, so elytra measurements exist |
| `parts_complete` | head, pronotum and elytra were all found, so `body_length_parts_*` is meaningful |
| `qc_low_solidity` | mask area under half of its convex hull: splayed legs (common in ethanol specimens) or a poor mask |
| `trunk_touches_edge` | head, pronotum or elytra reach the photo's edge: the body is cut off |
| `touches_edge` | any part of the mask reaches the edge; true for most tight crops, where legs and antennae run out of frame |
| `mask_fix` | `inverted_backdrop` when SAM returned the backdrop and the mask is its complement |
| `det_source` | how the box was obtained; `full_frame` and `backdrop_blob` are fallbacks |
| `view` (image manifest) | `ventral` and `lateral` photos give masks but not dorsal measurements |
