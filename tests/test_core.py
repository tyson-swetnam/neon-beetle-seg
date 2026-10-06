"""Unit tests for the parts of the pipeline that do not need a GPU or downloaded data."""
import cv2
import numpy as np
import pytest
from PIL import Image

from neon_beetle_seg import measure, scale
from neon_beetle_seg.fetch_biorepo import classify_image, parse_other_catalog_numbers
from neon_beetle_seg.models import PART_CLASSES, Detections, Sam2, box_iou, nms, tile_grid
from neon_beetle_seg.segment import border_coverage, drop_duplicate_masks, fix_backdrop_mask


# ---- identifiers ---------------------------------------------------------------------------------
def test_parse_other_catalog_numbers_biorepo_and_gbif_separators():
    bio = "NEON sampleID: GRSM_012.20160906.PTEACU1.01; NEON sampleID Hash: abc=; NEON sampleUUID: 634a-uuid"
    assert parse_other_catalog_numbers(bio) == {
        "neon_sampleID": "GRSM_012.20160906.PTEACU1.01", "neon_sampleID_hash": "abc=", "neon_barcode": None,
        "neon_sampleUUID": "634a-uuid"}
    gbif = "NEON sampleID: STER_032.20190904.CRADUB.01|NEON sampleCode (barcode): A00000121908|NEON sampleUUID: ee-1"
    out = parse_other_catalog_numbers(gbif)
    assert out["neon_barcode"] == "A00000121908" and out["neon_sampleID"] == "STER_032.20190904.CRADUB.01"
    assert parse_other_catalog_numbers(None)["neon_sampleID"] is None


@pytest.mark.parametrize("collection,url,expected", [
    ("CARC-AP", "https://x/media/NEON_CARC-AP/A00000033/A00000033605_lg.jpg", ("tray_ethanol", None)),
    ("CARC-TS", "https://x/WOOD_024.W.20160531.PTECOR3.01.13_dorsal.jpg", ("individual", "dorsal")),
    ("CARC-AP", "https://x/CPER_005.20160714.DISPAR1.01.8_ventral.jpg", ("individual", "ventral")),
    ("CARC-PV", "https://x/NEON.BET.D14.001435_Dorsal_2x._lg.jpg", ("pinned", "dorsal")),
    ("DCTC", "https://x/NEONcarabid8375.jpg", ("pinned", None)),
    ("HEVC-GBTS", "https://x/A00000091986_dorsal.jpg", ("bycatch", "dorsal")),
])
def test_classify_image(collection, url, expected):
    assert classify_image(collection, url, None) == expected


# ---- boxes ---------------------------------------------------------------------------------------
def test_box_iou_and_nms_containment():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], np.float32)
    iou = box_iou(a, b)[0]
    assert iou[0] == pytest.approx(1.0) and iou[1] == pytest.approx(25 / 175) and iou[2] == 0
    # a small box inside a higher-scoring big one is dropped although its IoU is low
    det = Detections(np.array([[0, 0, 100, 100], [10, 10, 30, 30], [200, 200, 240, 240]], np.float32),
                     np.array([0.9, 0.8, 0.7], np.float32), ["b", "b", "b"])
    kept = nms(det, iou_thr=0.5, containment_thr=0.8)
    assert len(kept) == 2 and kept.boxes[:, 0].tolist() == [0, 200]


def test_tile_grid_covers_image():
    assert tile_grid(800, 600, 1000) == [(0, 0, 800, 600)]
    tiles = tile_grid(5568, 3712, 1856)
    cover = np.zeros((3712, 5568), bool)
    for x1, y1, x2, y2 in tiles:
        assert x2 - x1 <= 1856 and y2 - y1 <= 1856
        cover[y1:y2, x1:x2] = True
    assert cover.all()


def test_crop_window_stays_inside_image():
    win = Sam2.crop_window((5, 5, 50, 80), 100, 90, pad=0.5)
    assert win[0] >= 0 and win[1] >= 0 and win[2] <= 100 and win[3] <= 90
    assert win[0] <= 5 and win[3] >= 80


# ---- masks and measurements ----------------------------------------------------------------------
def _ellipse(h, w, center, axes, angle=0):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, center, axes, angle, 0, 360, 1, -1)
    return m.astype(bool)


def test_rle_roundtrip():
    m = _ellipse(120, 200, (100, 60), (70, 30), 20)
    assert (measure.rle_decode(measure.rle_encode(m), 120, 200) == m).all()


def test_body_metrics_on_ellipse():
    m = _ellipse(300, 400, (200, 150), (120, 50))  # full axes 240 x 100
    r = measure.body_metrics(m)
    assert r["length_px"] == pytest.approx(240, abs=3) and r["width_px"] == pytest.approx(100, abs=3)
    assert r["area_px"] == pytest.approx(np.pi * 120 * 50, rel=0.02)
    assert r["solidity"] > 0.97
    assert r["core_length_px"] == pytest.approx(240, abs=8)


def test_core_length_ignores_thin_legs():
    m = _ellipse(400, 600, (300, 200), (120, 50))
    with_leg = m.astype(np.uint8)
    cv2.line(with_leg, (300, 200), (560, 380), 1, 5)
    r = measure.body_metrics(with_leg.astype(bool))
    assert r["length_px"] > 300  # the leg stretches the whole-mask rectangle
    assert r["core_length_px"] == pytest.approx(240, abs=10)  # but not the core body


def test_body_metrics_downscaled_matches_full_resolution():
    big = _ellipse(2400, 3200, (1600, 1200), (960, 400), 30)
    r = measure.body_metrics(big)
    assert r["length_px"] == pytest.approx(1920, rel=0.01) and r["width_px"] == pytest.approx(800, rel=0.01)


def test_part_metrics_axis_extents():
    lab = np.zeros((200, 400), np.uint8)
    idx = {n: i for i, n in enumerate(PART_CLASSES)}
    lab[70:130, 20:60] = idx["head"]        # 40 long, 60 wide
    lab[60:140, 60:130] = idx["pronotum"]   # 70 long, 80 wide
    lab[50:150, 130:330] = idx["elytra"]    # 200 long, 100 wide
    r = measure.part_metrics(lab)
    assert r["parts_ok"] and r["parts_complete"]
    assert r["elytra_length_px"] == pytest.approx(200, abs=3) and r["elytra_width_px"] == pytest.approx(100, abs=3)
    assert r["pronotum_width_px"] == pytest.approx(80, abs=3)
    assert r["body_length_parts_px"] == pytest.approx(310, abs=4)


def test_part_metrics_without_elytra_is_not_ok():
    assert measure.part_metrics(np.zeros((50, 50), np.uint8))["parts_ok"] is False


# ---- scale ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("spacing", [12.0, 23.5, 31.25])
def test_ruler_tick_spacing(spacing):
    img = np.full((120, 480), 235, np.uint8)
    for i in range(11):
        x = int(round(40 + i * spacing))
        img[20:80 if i % 5 == 0 else 60, x:x + 2] = 20
    res = scale.ruler_px_per_mm(Image.fromarray(img))
    assert res["px_per_mm"] == pytest.approx(spacing, rel=0.02) and res["n_ticks"] >= 8
    # a vertical ruler gives the same answer
    assert scale.ruler_px_per_mm(Image.fromarray(img.T))["px_per_mm"] == pytest.approx(spacing, rel=0.02)


def test_ruler_reader_rejects_blank_image():
    assert scale.ruler_px_per_mm(Image.fromarray(np.full((100, 300), 200, np.uint8)))["px_per_mm"] is None


@pytest.mark.parametrize("text,mm", [("5MM", 5.0), ("1 mm", 1.0), ("SR: 5MM", 5.0), ("0.5 mm", 0.5), ("1 cm", 10.0), ("hello", None)])
def test_parse_label(text, mm):
    assert scale.parse_label(text) == mm


def test_find_printed_bar():
    img = np.full((1000, 1500, 3), 240, np.uint8)
    img[900:904, 900:1300] = 30
    bar = scale.find_printed_bar(Image.fromarray(img))
    assert bar is not None and bar["x2"] - bar["x1"] == pytest.approx(400, abs=2)


# ---- backdrop masks ------------------------------------------------------------------------------
def test_backdrop_mask_is_inverted_and_specimen_mask_is_kept():
    beetle = _ellipse(200, 120, (60, 100), (35, 80))
    kept, how = fix_backdrop_mask(beetle)
    assert how is None and (kept == beetle).all() and border_coverage(beetle) == 0
    fixed, how = fix_backdrop_mask(~beetle)  # SAM returned the backdrop
    assert how == "inverted_backdrop" and (fixed == beetle).all()


# ---- midline length ------------------------------------------------------------------------------
def test_midline_length_follows_a_bent_body():
    m = np.zeros((400, 400), np.uint8)
    pts = np.array([[40, 40], [200, 60], [340, 200], [360, 360]], np.int32)  # an L-shaped "salamander"
    cv2.polylines(m, [pts], False, 1, thickness=18)
    true = sum(float(np.hypot(*(pts[i + 1] - pts[i]))) for i in range(len(pts) - 1)) + 18
    got = measure.midline_length(m.astype(bool))
    assert got == pytest.approx(true, rel=0.06)
    rect_long = measure.body_metrics(m.astype(bool))["length_px"]
    assert got > rect_long  # the rectangle cuts the corner


def test_drop_duplicate_masks_keeps_distinct_specimens():
    def row(mask, x, y):
        return dict(mask_rle=measure.rle_encode(mask), mask_h=mask.shape[0], mask_w=mask.shape[1], win_x1=x, win_y1=y,
                    area_px=float(mask.sum()))
    body = _ellipse(100, 200, (100, 50), (80, 30))
    limb = np.zeros((100, 200), bool)
    limb[40:60, 20:60] = body[40:60, 20:60]  # part of the same animal
    other = _ellipse(100, 200, (100, 50), (80, 30))
    kept = drop_duplicate_masks([row(limb, 0, 0), row(body, 0, 0), row(other, 500, 0)])
    assert len(kept) == 2 and {r["win_x1"] for r in kept} == {0, 500}
    assert max(r["area_px"] for r in kept if r["win_x1"] == 0) == float(body.sum())


def test_largest_blob_frac_flags_speckle():
    clean = _ellipse(200, 200, (100, 100), (60, 40))
    assert measure.body_metrics(clean)["largest_blob_frac"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    speckle = clean | (rng.random((200, 200)) > 0.8)
    assert measure.body_metrics(speckle)["largest_blob_frac"] < 0.8
