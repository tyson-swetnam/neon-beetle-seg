# Models

All models run locally on the 16 GB GPU. Everything except the optional YOLO detector loads
through `transformers`, so nothing has to be compiled.

| Role | Model | Licence | Gated | Used for |
|---|---|---|---|---|
| Detection | [`IDEA-Research/grounding-dino-base`](https://huggingface.co/IDEA-Research/grounding-dino-base), prompt `"a beetle."` (herps: `"a lizard. a frog. a salamander. a snake. a toad."`) | Apache-2.0 | no | one box per specimen in tray, single-specimen and herp photos |
| Instance masks | [`facebook/sam2.1-hiera-large`](https://huggingface.co/facebook/sam2.1-hiera-large) | Apache-2.0 | no | one mask per box (Biorepository, 2018 trays, Hawaii) |
| Instance masks | [`facebook/sam2.1-hiera-base-plus`](https://huggingface.co/facebook/sam2.1-hiera-base-plus) | Apache-2.0 | no | the 44,510 sentinel crops (about twice as fast) |
| Part labels | [`imageomics/BeetleFlow`](https://huggingface.co/imageomics/BeetleFlow) `5-class` (Mask2Former, Swin-L) | MIT | no | head, pronotum, elytra, legs, antennae per specimen |
| Herp species scoring | [`imageomics/bioclip-2`](https://huggingface.co/imageomics/bioclip-2) via `open_clip` | MIT | no | zero-shot scoring of herptile bycatch against GBIF candidate species |
| Scale label OCR | [`microsoft/trocr-base-printed`](https://huggingface.co/microsoft/trocr-base-printed) | MIT | no | reading "5 mm" / "1 mm" beside printed scale bars |
| Detection (optional) | [`imageomics/yolo_beetle_detection`](https://huggingface.co/imageomics/yolo_beetle_detection) (YOLOv8m) | weights MIT, Ultralytics AGPL-3.0 | no | comparison only; the pilot's detector |
| Comparison (optional) | [`facebook/sam3`](https://huggingface.co/facebook/sam3) | SAM License | yes, manual approval | text-prompted segmentation, `nbs sam3` |

The repository is MIT-licensed. `ultralytics` is AGPL-3.0, so it is an optional extra
(`uv sync --extra yolo`) and the default pipeline does not import it.

## Why these choices

**Grounding DINO instead of the YOLO beetle detector.** On 16 tray photos (389 human boxes,
IoU >= 0.5) the YOLOv8m detector the pilot used reached precision 0.69 to 0.74 and recall 0.81 to
0.84; it was trained on 29 images. Grounding DINO on the whole photo reached precision 0.985 and
recall 0.992. Two caveats: the human boxes started life as machine proposals that were then
corrected by hand, which favours a Grounding DINO detector, and tiling the photo made Grounding
DINO worse (precision 0.76 to 0.89), so the whole photo is used. The full-run comparison against
detector-independent annotations is in [validation.md](validation.md).

**One padded crop per box for SAM.** SAM resizes its input to 1024 px. In a 5568 px tray photo a
2 mm beetle would be a couple of dozen pixels wide after that. Each box is therefore segmented on
its own crop (box plus 35% padding), so small specimens are seen at close to full resolution. The
cost is one image-encoder pass per specimen instead of one per photo.

**SAM 2.1 large, base-plus for sentinel.** On test photos base-plus masks agreed with large at
IoU 0.97 to 0.99 and small at 0.97 to 0.985. Large is used where specimens join back to NEON
records. The sentinel crops are small (median 250 x 451 px in a sample of 600) and are by far
the largest pool, so they use base-plus to keep that run to a few hours.

**bf16 autocast.** Mixed precision roughly halves Grounding DINO and BeetleFlow time on the A16.
Boxes agreed with fp32 at IoU > 0.99 and part labels on > 99.8% of pixels.

**BeetleFlow at 512 x 512.** The model was trained on specimen crops squashed to 512 x 512
(`train.py --imgsz`), so inference does the same and maps the labels back to the crop's size.
Its checkpoint loads into `transformers` with two unused backbone parameters reported missing
(`swin.layernorm`); Mask2Former does not use that layer. BeetleFlow was trained on pinned
specimens; on ethanol specimens in tray photos it is out of its training domain, and
[validation.md](validation.md) reports how much that costs.

## SAM 3

`facebook/sam3` segments every instance of a text concept ("beetle") in one pass. It needs an
approved HuggingFace account and `HF_TOKEN`. `nbs sam3` runs it on a sample of 2018 tray photos
and records, per tray, how its instances compare with the human boxes and with the main
pipeline. It is a comparison only: no measurement in the lake comes from SAM 3. Its licence (the
SAM License) is a custom Meta licence; read it before redistributing derived weights.
