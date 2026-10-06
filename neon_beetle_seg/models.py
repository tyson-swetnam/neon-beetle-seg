"""Thin wrappers around the detection and segmentation models, all loaded through `transformers`
(no compiled CUDA ops are needed) except the optional Ultralytics YOLO beetle detector.

Every wrapper takes and returns plain numpy in *original image pixel coordinates* so the
pipeline steps do not need to know how each model resizes its input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

GDINO_ID = "IDEA-Research/grounding-dino-base"
SAM2_ID = "facebook/sam2.1-hiera-large"
SAM3_ID = "facebook/sam3"
BEETLEFLOW_ID = "imageomics/BeetleFlow"
YOLO_BEETLE_ID = "imageomics/yolo_beetle_detection"

# BeetleFlow 5-class head (pipeline/3_segmentation/config.py in Imageomics/BeetleFlow)
PART_CLASSES = ["background", "head", "pronotum", "elytra", "legs", "antennas"]


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) float32 xyxy, original pixels
    scores: np.ndarray  # (N,)
    labels: list[str]

    def __len__(self) -> int:
        return len(self.boxes)

    def select(self, keep) -> "Detections":
        keep = np.asarray(keep)
        idx = np.flatnonzero(keep) if keep.dtype == bool else keep
        return Detections(self.boxes[idx], self.scores[idx], [self.labels[i] for i in idx])

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), [])

    @staticmethod
    def concat(parts: list["Detections"]) -> "Detections":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Detections.empty()
        return Detections(np.concatenate([p.boxes for p in parts]), np.concatenate([p.scores for p in parts]),
                          [lab for p in parts for lab in p.labels])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(det: Detections, iou_thr: float = 0.5, containment_thr: float = 0.8) -> Detections:
    """Greedy NMS that also drops a box mostly contained in a higher-scoring one.

    Containment catches the usual tiling artefact: a partial detection of a specimen cut by a
    tile edge sitting inside the full detection from the neighbouring tile.
    """
    if len(det) == 0:
        return det
    order = np.argsort(-det.scores)
    boxes = det.boxes[order]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    for i in range(len(boxes)):
        if keep:
            k = boxes[keep]
            x1 = np.maximum(boxes[i, 0], k[:, 0]); y1 = np.maximum(boxes[i, 1], k[:, 1])
            x2 = np.minimum(boxes[i, 2], k[:, 2]); y2 = np.minimum(boxes[i, 3], k[:, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            iou = inter / (area[i] + area[keep] - inter)
            if (iou > iou_thr).any() or (inter / max(area[i], 1e-9) > containment_thr).any():
                continue
        keep.append(i)
    return det.select(order[keep])


def tile_grid(width: int, height: int, tile: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles (x1, y1, x2, y2) covering the image; a single tile if the image is small."""
    if max(width, height) <= tile:
        return [(0, 0, width, height)]

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        step = int(tile * (1 - overlap))
        s = list(range(0, size - tile, step)) + [size - tile]
        return sorted(set(s))

    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]


class GroundingDino:
    """Open-vocabulary detector. Prompts are lower-case phrases ending in a full stop."""

    def __init__(self, model_id: str = GDINO_ID):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device()).eval()

    @torch.inference_mode()
    def _detect_one(self, image: Image.Image, text: str, box_thr: float, text_thr: float) -> Detections:
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(device())
        with autocast():  # bf16 halves the time on the A16; boxes agree with fp32 to IoU > 0.99
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = res["text_labels"] if "text_labels" in res else res["labels"]
        return Detections(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(),
                          [str(x) for x in labels])

    def detect(self, image: Image.Image, text: str = "a beetle.", box_thr: float = 0.25,
               text_thr: float = 0.2, tile: int | None = None, overlap: float = 0.25) -> Detections:
        """Detect on the whole image, or on overlapping tiles when `tile` is set.

        Tiling matters for tray photos: at the model's ~1333 px input a 2 mm beetle in a
        5568 px tray is only a few dozen pixels long.
        """
        if tile is None:
            return self._detect_one(image, text, box_thr, text_thr)
        parts = []
        w, h = image.size
        for x1, y1, x2, y2 in tile_grid(w, h, tile, overlap):
            d = self._detect_one(image.crop((x1, y1, x2, y2)), text, box_thr, text_thr)
            if not len(d):
                continue
            # a box touching an inner tile edge is a specimen cut by the tile; the overlapping
            # neighbour sees it whole, so the cut copy is dropped rather than left for NMS
            b, m = d.boxes, 3
            cut = (((b[:, 0] <= m) & (x1 > 0)) | ((b[:, 1] <= m) & (y1 > 0))
                   | ((b[:, 2] >= (x2 - x1) - m) & (x2 < w)) | ((b[:, 3] >= (y2 - y1) - m) & (y2 < h)))
            d = d.select(~cut)
            if len(d):
                d.boxes[:, [0, 2]] += x1
                d.boxes[:, [1, 3]] += y1
                parts.append(d)
        return Detections.concat(parts)


class YoloBeetle:
    """Imageomics YOLOv8m fine-tuned on 2018 tray photos (classes: beetle, scale_bar). AGPL-3.0."""

    def __init__(self, model_id: str = YOLO_BEETLE_ID, weights: str = "yolo_beetles_best.pt"):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        self.model_id = model_id
        self.model = YOLO(hf_hub_download(model_id, weights))

    def detect(self, image: Image.Image, conf: float = 0.15, imgsz: int = 2048) -> Detections:
        r = self.model.predict(image, conf=conf, imgsz=imgsz, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
        names = r.names
        return Detections(r.boxes.xyxy.float().cpu().numpy(), r.boxes.conf.float().cpu().numpy(),
                          [names[int(c)] for c in r.boxes.cls.cpu().numpy()])


class Sam2:
    """SAM 2.1 prompted with one box per specimen.

    Each box is segmented on its own padded crop, so a small specimen in a large tray photo is
    seen at close to full resolution instead of the few pixels it would occupy after the whole
    tray is resized to the model's 1024 px input.
    """

    def __init__(self, model_id: str = SAM2_ID):
        from transformers import Sam2Model, Sam2Processor

        self.model_id = model_id
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device()).eval()

    @staticmethod
    def crop_window(box, width: int, height: int, pad: float = 0.35, min_side: int = 96):
        x1, y1, x2, y2 = box
        side_x = max((x2 - x1) * (1 + 2 * pad), min_side)
        side_y = max((y2 - y1) * (1 + 2 * pad), min_side)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        wx1 = int(max(0, np.floor(cx - side_x / 2))); wy1 = int(max(0, np.floor(cy - side_y / 2)))
        wx2 = int(min(width, np.ceil(cx + side_x / 2))); wy2 = int(min(height, np.ceil(cy + side_y / 2)))
        return wx1, wy1, wx2, wy2

    @torch.inference_mode()
    def segment_boxes(self, image: Image.Image, boxes: np.ndarray, batch: int = 8, pad: float = 0.35):
        """Return a list of (window, mask, score): `mask` is a bool array covering `window`
        (x1, y1, x2, y2) of the original image."""
        out = []
        w, h = image.size
        for i in range(0, len(boxes), batch):
            chunk = boxes[i:i + batch]
            windows = [self.crop_window(b, w, h, pad) for b in chunk]
            crops = [image.crop(win) for win in windows]
            local = [[[float(b[0] - win[0]), float(b[1] - win[1]), float(b[2] - win[0]), float(b[3] - win[1])]]
                     for b, win in zip(chunk, windows)]
            inputs = self.processor(images=crops, input_boxes=local, return_tensors="pt").to(device())
            with autocast():
                outputs = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu())
            scores = outputs.iou_scores.float().cpu().numpy().reshape(len(chunk), -1)[:, 0]
            for win, m, s in zip(windows, masks, scores):
                out.append((win, m.reshape(m.shape[-2:]).numpy().astype(bool), float(s)))
        return out


class Sam3:
    """SAM 3 text-prompted concept segmentation (gated model: needs an approved HF account)."""

    def __init__(self, model_id: str = SAM3_ID, token: str | None = None):
        from transformers import Sam3Model, Sam3Processor

        self.model_id = model_id
        self.processor = Sam3Processor.from_pretrained(model_id, token=token)
        self.model = Sam3Model.from_pretrained(model_id, token=token).to(device()).eval()

    @torch.inference_mode()
    def segment_text(self, image: Image.Image, text: str = "beetle", threshold: float = 0.5,
                     mask_threshold: float = 0.5):
        """Return (masks (N,H,W) bool, boxes (N,4) xyxy, scores (N,)) for every instance of `text`."""
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(device())
        with autocast():
            outputs = self.model(**inputs)
        # post-processing upsamples every instance mask to the full image as int64; on a tray
        # of a hundred beetles that is several GiB, so it is done in main memory, not on the GPU
        for key, value in list(outputs.items()):
            if isinstance(value, torch.Tensor):
                outputs[key] = value.float().cpu()
        torch.cuda.empty_cache()
        res = self.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=mask_threshold, target_sizes=[image.size[::-1]],
        )[0]
        return (res["masks"].cpu().numpy().astype(bool), res["boxes"].float().cpu().numpy(),
                res["scores"].float().cpu().numpy())


class BeetleFlow:
    """Imageomics BeetleFlow Mask2Former: semantic part labels for one cropped pinned beetle."""

    def __init__(self, model_id: str = BEETLEFLOW_ID, subfolder: str = "5-class"):
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        self.model_id = f"{model_id}/{subfolder}"
        self.processor = Mask2FormerImageProcessor(do_reduce_labels=False, do_resize=False)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, subfolder=subfolder).to(device()).eval()

    # BeetleFlow was trained on crops squashed to 512 x 512 (train.py --imgsz default), so
    # inference does the same and maps the labels back to the crop's own size.
    INPUT_SIZE = (512, 512)

    @torch.inference_mode()
    def predict(self, crops: list[Image.Image]) -> list[np.ndarray]:
        """Return one uint8 label map per crop, indices into PART_CLASSES."""
        sizes = [c.size[::-1] for c in crops]
        arrays = [np.asarray(c.convert("RGB").resize(self.INPUT_SIZE, Image.BILINEAR)) for c in crops]
        inputs = self.processor(images=arrays, return_tensors="pt").to(device())
        with autocast():  # labels agree with fp32 on > 99.8% of pixels
            outputs = self.model(**inputs)
        maps = self.processor.post_process_semantic_segmentation(outputs, target_sizes=sizes)
        return [m.cpu().numpy().astype(np.uint8) for m in maps]
