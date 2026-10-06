"""Morphometrics from instance masks and part label maps. Everything here is in pixels;
millimetre columns are derived later from px_per_mm so the scale can be revised without re-running
the GPU steps.
"""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from .models import PART_CLASSES


def rle_encode(mask: np.ndarray) -> str:
    """COCO compressed RLE `counts` string for a 2-D bool mask (size is stored separately)."""
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def rle_decode(counts: str, height: int, width: int) -> np.ndarray:
    return coco_mask.decode({"size": [int(height), int(width)], "counts": counts.encode("ascii")}).astype(bool)


def largest_component(mask: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    """Keep the largest connected blob; optionally fill holes (pin heads, glare)."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (labels == keep).astype(np.uint8)
    if fill_holes:
        contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


WORK_SIDE = 768  # shape metrics are computed on masks no larger than this, then scaled back


def _downscale(arr: np.ndarray, nearest: bool = True) -> tuple[np.ndarray, float]:
    """Shrink a mask / label map so its longer side is <= WORK_SIDE. Returns (array, scale)."""
    side = max(arr.shape[:2])
    if side <= WORK_SIDE:
        return arr, 1.0
    s = WORK_SIDE / side
    small = cv2.resize(arr.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    return small, s


def _rect_dims(mask: np.ndarray) -> tuple[float, float, float]:
    """(long side, short side, angle of the long side in degrees) of the minimum-area rectangle."""
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None or len(pts) < 3:
        return 0.0, 0.0, 0.0
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    if w < h:
        w, h, ang = h, w, ang + 90.0
    return float(w), float(h), float(ang % 180.0)


def body_metrics(mask: np.ndarray, rgb: np.ndarray | None = None) -> dict:
    """Whole-specimen metrics for one bool mask (legs and antennae included unless stated)."""
    out = dict.fromkeys(
        ["area_px", "perimeter_px", "length_px", "width_px", "ellipse_major_px", "ellipse_minor_px",
         "orientation_deg", "core_length_px", "core_width_px", "solidity", "mean_r", "mean_g", "mean_b",
         "largest_blob_frac"])
    raw_area = int(mask.sum())
    if raw_area < 9:
        return out
    m = largest_component(mask, fill_holes=False)
    # a clean mask is one blob; a speckled, scattered mask is a failed segmentation
    out["largest_blob_frac"] = float(m.sum()) / raw_area
    m = largest_component(mask)
    area = float(m.sum())  # exact, at full resolution
    small, sc = _downscale(m)
    m8 = small.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    length, width, ang = _rect_dims(m8)
    out.update(area_px=area, perimeter_px=float(cv2.arcLength(cnt, True)) / sc, length_px=length / sc,
               width_px=width / sc, orientation_deg=ang)
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    out["solidity"] = float(m8.sum()) / hull_area if hull_area > 0 else None
    if len(cnt) >= 5:
        (_, _), (a, b), _ = cv2.fitEllipse(cnt)
        out["ellipse_major_px"], out["ellipse_minor_px"] = float(max(a, b)) / sc, float(min(a, b)) / sc
    # "core" body: an opening removes legs and antennae, which are thin relative to the body
    k = max(3, int(round(0.25 * width)) | 1)
    core = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if core.any():
        cl, cw, _ = _rect_dims(largest_component(core.astype(bool), fill_holes=False))
        out["core_length_px"], out["core_width_px"] = cl / sc, cw / sc
    if rgb is not None:
        px = rgb[m]
        out["mean_r"], out["mean_g"], out["mean_b"] = (float(v) for v in px.mean(axis=0))
    return out


def _axis_extent(mask: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Extent of the mask's pixels along `axis` and perpendicular to it."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    along = xs * axis[0] + ys * axis[1]
    perp = -xs * axis[1] + ys * axis[0]
    return float(along.max() - along.min() + 1), float(perp.max() - perp.min() + 1)


def body_axis(mask: np.ndarray) -> np.ndarray | None:
    """Unit vector of the principal axis of the mask's pixels."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1).astype(np.float64)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    return vecs[:, -1]


def part_metrics(labels: np.ndarray, body: np.ndarray | None = None) -> dict:
    """Elytra / pronotum / head dimensions from a BeetleFlow label map.

    The body axis is the principal axis of head + pronotum + elytra. Lengths are extents along
    that axis, widths are extents across it, each on the largest blob of that part. `body`
    (bool, same shape) restricts the labels to one specimen's instance mask when given.
    """
    out = dict.fromkeys(
        ["parts_ok", "parts_complete", "body_length_parts_px", "elytra_length_px", "elytra_width_px", "pronotum_length_px",
         "pronotum_width_px", "head_width_px", "elytra_area_px", "pronotum_area_px", "head_area_px",
         "legs_area_px", "antennae_area_px"])
    lab = labels if body is None else np.where(body, labels, 0)
    idx = {name: i for i, name in enumerate(PART_CLASSES)}
    for n, col in (("elytra", "elytra_area_px"), ("pronotum", "pronotum_area_px"), ("head", "head_area_px"),
                   ("legs", "legs_area_px"), ("antennas", "antennae_area_px")):
        out[col] = float((lab == idx[n]).sum())  # exact, at full resolution
    lab, sc = _downscale(lab)
    part = {n: lab == idx[n] for n in ("head", "pronotum", "elytra", "legs", "antennas")}
    trunk = part["head"] | part["pronotum"] | part["elytra"]
    axis = body_axis(largest_component(trunk, fill_holes=False)) if trunk.any() else None
    out["parts_ok"] = bool(axis is not None and part["elytra"].sum() >= 25)
    out["parts_complete"] = bool(out["parts_ok"] and part["pronotum"].sum() >= 9 and part["head"].sum() >= 9)
    if not out["parts_ok"]:
        return out
    out["body_length_parts_px"] = _axis_extent(largest_component(trunk, fill_holes=False), axis)[0] / sc
    ely = largest_component(part["elytra"], fill_holes=False)
    out["elytra_length_px"], out["elytra_width_px"] = (v / sc for v in _axis_extent(ely, axis))
    if part["pronotum"].sum() >= 9:
        pro = largest_component(part["pronotum"], fill_holes=False)
        out["pronotum_length_px"], out["pronotum_width_px"] = (v / sc for v in _axis_extent(pro, axis))
    if part["head"].sum() >= 9:
        out["head_width_px"] = _axis_extent(largest_component(part["head"], fill_holes=False), axis)[1] / sc
    return out


def base_widths(head: np.ndarray, pronotum: np.ndarray, elytra: np.ndarray, slab: float = 0.15) -> dict:
    """Caliper-style measurements at the pronotum-elytra junction and along the midline.

    elytra_midline_length_px along the body axis within the central fifth of the elytra's width:
                             from the middle of the elytra's base to the middle of the apex,
                             which is how both annotated datasets define elytra length. The
                             full extent (elytra_length_px) also counts the shoulders, which
                             project forward of the base's midpoint.
    elytra_base_width_px     width across both elytra within the anterior `slab` of their length
                             (the humeral width; the 2018 Zooniverse "ElytraWidth" lines sit here)
    pronotum_base_width_px   width of the pronotum within its posterior `slab`
                             (the Hawaii "basal pronotum width")
    "Anterior" is the end of the elytra nearer the pronotum (or head, when no pronotum was found).
    """
    out = dict(elytra_midline_length_px=None, elytra_base_width_px=None, pronotum_base_width_px=None)
    trunk = head | pronotum | elytra
    if elytra.sum() < 25 or not trunk.any():
        return out
    (elytra, pronotum, head, trunk), sc = zip(*[_downscale(m) for m in (elytra, pronotum, head, trunk)])
    sc = sc[0]
    elytra, pronotum, head, trunk = (m.astype(bool) for m in (elytra, pronotum, head, trunk))
    axis = body_axis(largest_component(trunk, fill_holes=False))
    front = pronotum if pronotum.sum() >= 9 else head
    if axis is None or front.sum() < 9:
        return out

    def coords(mask):
        ys, xs = np.nonzero(mask)
        return xs * axis[0] + ys * axis[1], -xs * axis[1] + ys * axis[0]

    ea, ep = coords(largest_component(elytra, fill_holes=False))
    fa, fp = coords(largest_component(front, fill_holes=False))
    mid = np.abs(ep - (ep.min() + ep.max()) / 2) <= 0.1 * (ep.max() - ep.min() + 1)
    if mid.sum() >= 3:
        out["elytra_midline_length_px"] = float(ea[mid].max() - ea[mid].min() + 1) / sc
    lo, hi = ea.min(), ea.max()
    toward_front = fa.mean() < ea.mean()  # is the pronotum at the low end of the axis?
    span = (hi - lo) * slab
    sel = (ea <= lo + span) & (ea >= lo + 0.2 * span) if toward_front else (ea >= hi - span) & (ea <= hi - 0.2 * span)
    if sel.sum() >= 3:
        out["elytra_base_width_px"] = float(ep[sel].max() - ep[sel].min() + 1) / sc
    if pronotum.sum() >= 9:
        plo, phi = fa.min(), fa.max()
        pspan = (phi - plo) * slab
        # the pronotum's posterior end is the one facing the elytra
        psel = (fa >= phi - pspan) if toward_front else (fa <= plo + pspan)
        if psel.sum() >= 3:
            out["pronotum_base_width_px"] = float(fp[psel].max() - fp[psel].min() + 1) / sc
    return out


def midline_length(mask: np.ndarray) -> float | None:
    """Length of the longest path through the mask's skeleton, in pixels.

    For a long, bent animal (a salamander, a lizard with its tail) the bounding rectangle
    understates length; the skeleton follows the body. The longest skeleton path runs from one
    extremity to the other, so on a specimen with a tail it approximates total length, and on a
    frog with a leg stretched out it runs to the toes. It is not snout-vent length. The
    skeleton stops about half a body-width short of each tip, which is added back from the
    distance transform.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from skimage.morphology import skeletonize

    m = largest_component(mask)
    small, sc = _downscale(m)
    small = small.astype(bool)
    if small.sum() < 20:
        return None
    skel = skeletonize(small)
    ys, xs = np.nonzero(skel)
    n = len(xs)
    if n < 2:
        return None
    index = -np.ones(skel.shape, np.int64)
    index[ys, xs] = np.arange(n)
    rows, cols, weights = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < skel.shape[0]) & (x2 >= 0) & (x2 < skel.shape[1])
        j = np.full(n, -1)
        j[ok] = index[y2[ok], x2[ok]]
        hit = j >= 0
        rows += [np.flatnonzero(hit)]
        cols += [j[hit]]
        weights += [np.full(hit.sum(), float(np.hypot(dy, dx)))]
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    # two sweeps find the two ends of the longest path in a tree-like skeleton
    d0 = dijkstra(graph, directed=False, indices=0)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))
    d1 = dijkstra(graph, directed=False, indices=a)
    d1[~np.isfinite(d1)] = -1
    b = int(np.argmax(d1))
    dist = cv2.distanceTransform(small.astype(np.uint8), cv2.DIST_L2, 5)
    return float(d1[b] + dist[ys[a], xs[a]] + dist[ys[b], xs[b]]) / sc
