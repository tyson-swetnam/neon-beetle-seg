"""Write outputs/report/report.html: a self-contained summary of what was built and how well it works.

Everything is read from data/tables/, so the report always describes the current build. Charts
are inline SVG (no JavaScript, no network), each with its numbers in a table underneath.
"""
from __future__ import annotations

import base64
import html
import io
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import __version__, config, measure

Image.MAX_IMAGE_PIXELS = None

CSS = """
:root{--surface:#fcfcfb;--panel:#f3f2ef;--text:#0b0b0b;--text2:#52514e;--muted:#8a8983;--grid:#e4e3df;
--s1:#2a78d6;--s2:#eb6834;--wash:rgba(42,120,214,.10)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#242422;--text:#fff;
--text2:#c3c2b7;--muted:#8f8e86;--grid:#383835;--s1:#3987e5;--s2:#d95926;--wash:rgba(57,135,229,.14);color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#242422;--text:#fff;--text2:#c3c2b7;--muted:#8f8e86;--grid:#383835;
--s1:#3987e5;--s2:#d95926;--wash:rgba(57,135,229,.14);color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding-block:32px 64px;padding-inline:16px}a{color:var(--s1)}
h1,h2,h3{text-wrap:balance}th,td{font-variant-numeric:tabular-nums}
h1{font-size:30px;line-height:1.2;margin:0 0 6px}h2{font-size:21px;margin:44px 0 10px}h3{font-size:16px;margin:22px 0 6px}
p{margin:8px 0;max-width:76ch}.sub{color:var(--text2)}.note{color:var(--text2);font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}
.tile{background:var(--panel);border-radius:10px;padding:14px 16px}.tile .v{font-size:28px;font-weight:600;line-height:1.15}
.tile .l{color:var(--text2);font-size:14px}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:14px;margin:8px 0 4px;min-width:60%}
th,td{padding:5px 12px 5px 0;text-align:left;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--text2);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:14px 0}figcaption{color:var(--text2);font-size:14px;margin-top:4px;max-width:76ch}
svg{max-width:100%;height:auto;display:block}svg text{fill:var(--text2);font:12px system-ui,sans-serif}
svg .ax{stroke:var(--grid);stroke-width:1}svg .bar{fill:var(--s1)}svg .bar2{fill:var(--s2)}
svg .pt{fill:var(--s1);fill-opacity:.45}svg .ref{stroke:var(--muted);stroke-width:1}svg .lab{fill:var(--text)}
details{margin:6px 0 0}summary{cursor:pointer;color:var(--text2);font-size:14px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}.grid2 svg{max-width:380px}
.shots{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;align-items:end}.shots figure{margin:0;min-width:0}
.shots img{max-width:100%;max-height:300px;width:auto;height:auto;border-radius:8px;display:block}.shots code{word-break:break-all}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:14px;color:var(--text2);margin:4px 0}.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px}
code{font:13px ui-monospace,Menlo,Consolas,monospace;background:var(--panel);padding:1px 5px;border-radius:4px}
ul{padding-left:20px;max-width:76ch}li{margin:5px 0}
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt(x, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (int, np.integer)) or digits == 0:
        return f"{int(round(float(x))):,}"
    return f"{float(x):,.{digits}f}"


def table(df: pd.DataFrame, numeric: tuple[str, ...] = (), digits: dict | None = None) -> str:
    digits = digits or {}
    head = "".join(f'<th class="{"n" if c in numeric else ""}">{esc(c)}</th>' for c in df.columns)
    body = []
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, row):
            if c in numeric:
                cells.append(f'<td class="n">{fmt(v, digits.get(c, 0))}</td>')
            else:
                cells.append(f"<td>{esc(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def flag(series: pd.Series, missing: bool = False) -> pd.Series:
    """A nullable flag column as plain booleans. Object-typed True/False/None columns must not be
    negated with ~ directly: Python then computes ~True == -2."""
    return series.fillna(missing).astype(bool)


def read(name: str) -> pd.DataFrame | None:
    p = config.TABLES / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ---- charts ----------------------------------------------------------------------------------
def hbar(labels: list[str], values: list[float], unit: str, width: int = 720, label_w: int = 70) -> str:
    """Horizontal bars, one series: value label at the bar tip, native tooltip per bar."""
    row, top = 18, 6
    h = top + row * len(labels) + 22
    vmax = max(values) if values else 1
    plot = width - label_w - 70
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="bar chart">']
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = top + i * row
        w = max(1.0, plot * v / vmax)
        out.append(f'<text x="{label_w - 8}" y="{y + 12}" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<path class="bar" d="M{label_w},{y + 2} h{max(w - 4, 0):.1f} a4,4 0 0 1 4,4 v4 a4,4 0 0 1 -4,4 h-{max(w - 4, 0):.1f} z">'
                   f'<title>{esc(lab)}: {fmt(v)} {esc(unit)}</title></path>')
        out.append(f'<text class="lab" x="{label_w + w + 6:.1f}" y="{y + 12}">{fmt(v)}</text>')
    out.append(f'<line class="ax" x1="{label_w}" y1="{top}" x2="{label_w}" y2="{top + row * len(labels)}"/>')
    out.append(f'<text x="{label_w}" y="{h - 4}">{esc(unit)}</text></svg>')
    return "".join(out)


def scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, size: int = 330, max_points: int = 1500) -> str:
    """Predicted vs human, one series, with the 1:1 line."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return ""
    if len(x) > max_points:
        idx = np.random.default_rng(0).choice(len(x), max_points, replace=False)
        x, y = x[idx], y[idx]
    top = float(np.percentile(np.concatenate([x, y]), 99.5)) * 1.05
    step = next(st for st in (1, 2, 5, 10, 20, 50) if top / st <= 6)  # round tick spacing
    hi = float(np.ceil(top / step) * step)
    m, pad = 44, 10
    s = (size - m - pad) / hi

    def px(v):
        return m + v * s

    def py(v):
        return size - m - v * s

    ticks = list(np.arange(step, hi + step / 2, step))
    out = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="scatter plot">']
    for t in ticks:
        out.append(f'<line class="ax" x1="{m}" y1="{py(t):.1f}" x2="{size - pad}" y2="{py(t):.1f}"/>')
        out.append(f'<text x="{m - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{t:.0f}</text>')
        out.append(f'<text x="{px(t):.1f}" y="{size - m + 16}" text-anchor="middle">{t:.0f}</text>')
    out.append(f'<line class="ax" x1="{m}" y1="{size - m}" x2="{size - pad}" y2="{size - m}"/>')
    out.append(f'<line class="ax" x1="{m}" y1="{pad}" x2="{m}" y2="{size - m}"/>')
    out.append(f'<line class="ref" x1="{px(0)}" y1="{py(0)}" x2="{px(hi):.1f}" y2="{py(hi):.1f}"><title>1:1 line</title></line>')
    for a, b in zip(x, y):
        if a <= hi and b <= hi:
            out.append(f'<circle class="pt" cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.2"/>')
    out.append(f'<text x="{(m + size - pad) / 2}" y="{size - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text transform="translate(12,{(size - m + pad) / 2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text></svg>')
    return "".join(out)


def histogram(values: np.ndarray, xlabel: str, bins: np.ndarray, width: int = 330, height: int = 190) -> str:
    counts, edges = np.histogram(values[np.isfinite(values)], bins=bins)
    m_l, m_b, pad = 44, 34, 8
    pw, ph = width - m_l - pad, height - m_b - pad
    peak = max(int(counts.max()), 1)
    mag = 10 ** int(np.floor(np.log10(peak)))
    cmax = int(next(m * mag for m in (1, 2, 4, 5, 10) if m * mag >= peak))  # round axis maximum
    bw = pw / len(counts)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="histogram">']
    for frac in (0.5, 1.0):
        y = pad + ph * (1 - frac)
        out.append(f'<line class="ax" x1="{m_l}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}"/>')
        out.append(f'<text x="{m_l - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt(cmax * frac)}</text>')
    for i, c in enumerate(counts):
        hgt = ph * c / cmax
        if c:
            out.append(f'<rect class="bar" x="{m_l + i * bw + 1:.1f}" y="{pad + ph - hgt:.1f}" width="{max(bw - 2, 1):.1f}" height="{hgt:.1f}">'
                       f'<title>{edges[i]:.0f}–{edges[i + 1]:.0f} mm: {c:,} specimens</title></rect>')
    out.append(f'<line class="ax" x1="{m_l}" y1="{pad + ph}" x2="{width - pad}" y2="{pad + ph}"/>')
    for t in edges[::max(1, len(edges) // 6)]:
        x = m_l + (t - edges[0]) / (edges[-1] - edges[0]) * pw
        out.append(f'<text x="{x:.1f}" y="{pad + ph + 15}" text-anchor="middle">{t:.0f}</text>')
    out.append(f'<text x="{m_l + pw / 2}" y="{height - 3}" text-anchor="middle">{esc(xlabel)}</text></svg>')
    return "".join(out)



def stacked_columns(labels: list, series: dict[str, list[float]], unit: str, width: int = 720, height: int = 250) -> str:
    """Columns stacked by series (two series at most), with a 2px surface gap between segments."""
    classes = ["bar", "bar2"]
    m_l, m_b, pad = 56, 30, 10
    pw, ph = width - m_l - pad, height - m_b - pad
    totals = [sum(v[i] for v in series.values()) for i in range(len(labels))]
    peak = max(totals) if totals else 1
    mag = 10 ** int(np.floor(np.log10(max(peak, 1))))
    top = next(m * mag for m in (1, 2, 4, 5, 10) if m * mag >= peak)
    slot = pw / max(len(labels), 1)
    bw = min(24, slot - 6)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="stacked column chart">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = pad + ph * (1 - frac)
        out.append(f'<line class="ax" x1="{m_l}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}"/>')
        out.append(f'<text x="{m_l - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt(top * frac)}</text>')
    for i, lab in enumerate(labels):
        x = m_l + i * slot + (slot - bw) / 2
        y = pad + ph
        for k, (name, vals) in enumerate(series.items()):
            hgt = ph * vals[i] / top
            if hgt > 0.5:
                gap = 2 if k else 0
                out.append(f'<rect class="{classes[k]}" x="{x:.1f}" y="{y - hgt:.1f}" width="{bw:.1f}" height="{max(hgt - gap, 0.5):.1f}">'
                           f'<title>{esc(lab)} · {esc(name)}: {fmt(vals[i])} {esc(unit)}</title></rect>')
            y -= hgt
        out.append(f'<text x="{x + bw / 2:.1f}" y="{pad + ph + 16}" text-anchor="middle">{esc(lab)}</text>')
    out.append(f'<line class="ax" x1="{m_l}" y1="{pad + ph}" x2="{width - pad}" y2="{pad + ph}"/></svg>')
    return "".join(out)


def legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><span class="sw" style="background:var(--{var})"></span>{esc(name)}</span>' for name, var in items) + "</div>"


def funnel(steps: list[tuple[str, float]], width: int = 720) -> str:
    """Horizontal bars on one scale: how many specimens survive each step toward a measurement."""
    row, top, label_w = 34, 4, 250
    plot = width - label_w - 90
    vmax = max(v for _, v in steps) or 1
    out = [f'<svg viewBox="0 0 {width} {top + row * len(steps)}" role="img" aria-label="coverage funnel">']
    for i, (lab, v) in enumerate(steps):
        y = top + i * row
        w = max(1.5, plot * v / vmax)
        out.append(f'<text class="lab" x="{label_w - 10}" y="{y + 17}" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<path class="bar" d="M{label_w},{y + 3} h{max(w - 4, 0):.1f} a4,4 0 0 1 4,4 v10 a4,4 0 0 1 -4,4 h-{max(w - 4, 0):.1f} z">'
                   f'<title>{esc(lab)}: {fmt(v)}</title></path>')
        share = f" ({v / vmax * 100:.1f}%)" if i else ""
        out.append(f'<text class="lab" x="{label_w + w + 8:.1f}" y="{y + 17}">{fmt(v)}{share}</text>')
    out.append("</svg>")
    return "".join(out)


def site_map(sites: pd.DataFrame, width: int = 720) -> str:
    """NEON sites as dots: area = pinned individuals, filled = has measured specimens.

    A plain projection with no coastline. The contiguous states share one frame; Alaska, Hawaii
    and Puerto Rico sit in labelled insets, each on its own scale.
    """
    frames = {  # name -> (selector, x, y, w, h)
        "main": (lambda d: (d.lat.between(24, 50)) & (d.lon.between(-125, -66)), 150, 8, 562, 268),
        "Alaska": (lambda d: d.lat > 50, 8, 8, 132, 96),
        "Hawaii": (lambda d: (d.lat < 25) & (d.lon < -150), 8, 114, 132, 76),
        "Puerto Rico": (lambda d: (d.lat < 20) & (d.lon > -70), 8, 200, 132, 76),
    }
    vmax = max(sites["individuals"].max(), 1)
    out = [f'<svg viewBox="0 0 {width} 284" role="img" aria-label="map of NEON sites">']
    for name, (sel, fx, fy, fw, fh) in frames.items():
        d = sites[sel(sites)]
        out.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" rx="6" fill="var(--panel)"/>')
        if name != "main":
            out.append(f'<text x="{fx + 6}" y="{fy + 14}">{name}</text>')
        if d.empty:
            continue
        k = np.cos(np.radians(d.lat.mean()))
        xs, ys = d.lon * k, -d.lat
        pad = 30 if name == "main" else 26
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if x1 - x0 < 3:  # a single site, or a tight cluster: give it some room
            x0, x1, y0, y1 = x0 - 1.5, x1 + 1.5, y0 - 1.5, y1 + 1.5
        sc = min((fw - 2 * pad) / (x1 - x0), (fh - 2 * pad) / (y1 - y0))
        ox = fx + (fw - (x1 - x0) * sc) / 2
        oy = fy + (fh - (y1 - y0) * sc) / 2
        dots = []
        for r, x, y in sorted(zip(d.itertuples(), xs, ys), key=lambda t: -t[0].individuals):
            px, py = ox + (x - x0) * sc, oy + (y - y0) * sc
            dots.append((r, px, py, 3 + 8 * np.sqrt(r.individuals / vmax)))
        for r, px, py, rad in dots:
            fill = 'fill="var(--s1)" fill-opacity=".75"' if r.measured > 0 else 'fill="none"'
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rad:.1f}" {fill} stroke="var(--surface)" stroke-width="1.5">'
                       f'<title>{esc(r.siteID)} ({esc(r.domainID)}): {fmt(r.individuals)} pinned individuals, {fmt(r.taxa)} taxa, '
                       f'{fmt(r.measured)} specimens measured from images</title></circle>')
            if r.measured == 0:
                out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rad:.1f}" fill="none" stroke="var(--s1)" stroke-width="1.5"/>')
        # labels: for each dot, the first of eight positions that hits no other label or dot
        placed = []
        for r, px, py, rad in dots:
            g = rad + 3
            options = [(g, 3.5, "start"), (-g, 3.5, "end"), (0, -g - 1, "middle"), (0, g + 9, "middle"),
                       (g * .8, -g * .6, "start"), (-g * .8, -g * .6, "end"), (g * .8, g * .6 + 8, "start"), (-g * .8, g * .6 + 8, "end")]
            best = options[0]
            for dx, dy, anchor in options:
                lx, ly = px + dx, py + dy
                cx = lx + {"start": 13, "end": -13, "middle": 0}[anchor]  # label centre (about 26 x 10 px)
                clear_labels = all(abs(cx - qx) > 27 or abs(ly - qy) > 10 for qx, qy in placed)
                clear_dots = all((abs(cx - qx) > 13 + qr or abs(ly - 3.5 - qy) > 5 + qr) for q, qx, qy, qr in dots if q is not r)
                inside = fx + 2 < cx - 13 and cx + 13 < fx + fw - 2 and fy + 10 < ly < fy + fh - 2
                if clear_labels and clear_dots and inside:
                    best = (dx, dy, anchor)
                    break
            lx, ly = px + best[0], py + best[1]
            placed.append((lx + {"start": 13, "end": -13, "middle": 0}[best[2]], ly))
            out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{best[2]}" style="font-size:9.5px">{esc(r.siteID)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---- example overlays -------------------------------------------------------------------------
PART_COLOURS = {"head": (255, 70, 70), "pronotum": (70, 215, 70), "elytra": (70, 130, 255)}


def overlay(image_id: str, inst: pd.DataFrame, man: pd.DataFrame, max_side: int = 620) -> str | None:
    """The photo, resized for the page, with each specimen's outline and part masks drawn on it."""
    row = man[man["image_id"] == image_id]
    if row.empty:
        return None
    full = np.asarray(Image.open(config.ROOT / row.iloc[0]["local_path"]).convert("RGB"))
    s = min(max_side / max(full.shape[:2]), 3.0)  # small crops are enlarged, but not into a blur
    img = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)

    def place(mask: np.ndarray, x: float, y: float):
        """Mask resized to page scale, and the slice of the page image it covers."""
        h, w = max(1, round(mask.shape[0] * s)), max(1, round(mask.shape[1] * s))
        m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        y0, x0 = round(y * s), round(x * s)
        sub = img[y0:y0 + h, x0:x0 + w]
        return m[:sub.shape[0], :sub.shape[1]], sub

    for r in inst[inst["image_id"] == image_id].itertuples():
        if isinstance(getattr(r, "elytra_rle", None), str):
            for part, col in PART_COLOURS.items():
                pm, sub = place(measure.rle_decode(getattr(r, f"{part}_rle"), int(r.parts_h), int(r.parts_w)), r.pwin_x1, r.pwin_y1)
                sub[pm] = (0.55 * sub[pm] + 0.45 * np.array(col)).astype(np.uint8)
        m, sub = place(measure.rle_decode(r.mask_rle, int(r.mask_h), int(r.mask_w)), r.win_x1, r.win_y1)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(sub, cnts, -1, (255, 235, 0), 2)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- report ----------------------------------------------------------------------------------
POOL_NAMES = {"biorepo": "NEON Biorepository beetles", "hf2018": "2018 ethanol trays (HuggingFace)",
              "hawaii": "Hawaii pinned trays (HuggingFace)", "sentinel": "sentinel-beetles crops (HuggingFace)",
              "herp": "NEON Biorepository herptile bycatch"}


def build() -> str:
    man, allmeas, res = read("image_manifest"), read("measurements"), read("image_results")
    inst = read("instances")
    val, vtraits = read("validation_summary"), read("validation_traits")
    neon_field, spec = read("neon_fielddata"), read("specimen_manifest")
    gbif, sam3 = read("gbif_occurrences"), read("sam3_comparison")
    colls, pres = read("biorepo_collections"), read("preserved_samples")
    scale_tbl, prov = read("image_scale"), read("run_provenance")
    herp_sug, herp_val = read("herp_id_suggestions"), read("herp_id_validation")
    meas = allmeas[allmeas["source"] != "herp"]  # beetles
    herps = allmeas[allmeas["source"] == "herp"]
    todo = man[man["process"]]
    done_ids = set(res.loc[res["status"] == "ok", "image_id"]) if res is not None else set()
    P: list[str] = []

    P.append(f"<h1>NEON pitfall traps: ground beetles, bycatch, and what the photographs measure</h1>"
             f'<p class="sub">neon-beetle-seg v{__version__} · built {time.strftime("%Y-%m-%d")} · all NEON terrestrial sites · '
             f'<a href="https://github.com/tyson-swetnam/neon-beetle-seg">code and methods</a></p>')
    joinable = meas[meas["source"] != "sentinel"]
    tiles = []
    if neon_field is not None:
        tiles.append((fmt(neon_field["siteID"].nunique()), "NEON sites"))
    if spec is not None:
        tiles.append((fmt(len(spec)), "pinned beetles in NEON's tables"))
    if pres is not None:
        tiles.append((fmt(len(pres)), "ethanol vials: bulk beetles and bycatch"))
    tiles += [(fmt(len(meas)), "beetles segmented from images"),
              (fmt(int(meas["qc_has_scale"].sum())), "of those measured in millimetres")]
    if len(herps):
        tiles.append((fmt(len(herps)), "reptiles and amphibians segmented"))
    P.append('<div class="tiles">' + "".join(f'<div class="tile"><div class="v">{v}</div><div class="l">{esc(lab)}</div></div>' for v, lab in tiles) + "</div>")
    pending = len(todo) - len(done_ids)
    if pending > 0:
        P.append(f'<p class="note">This build is partial: {fmt(pending)} of {fmt(len(todo))} images had not been processed when it was written.</p>')

    # ---- where and when -------------------------------------------------------------------------
    if neon_field is not None and spec is not None:
        P.append("<h2>Where and when</h2>")
        rel = neon_field["release"].astype(str).value_counts()
        P.append(f"<p>NEON's ground-beetle product (DP1.10022.001) was stacked for {neon_field['siteID'].nunique()} sites in "
                 f"{neon_field['domainID'].nunique()} domains and {fmt(neon_field['plotID'].nunique())} plots, collection dates "
                 f"{str(neon_field['collectDate'].min())[:10]} to {str(neon_field['collectDate'].max())[:10]}: "
                 f"{fmt(len(neon_field))} trap-bout records and {fmt(len(spec))} pinned individuals of {fmt(spec['bestScientificName'].nunique())} taxa.</p>")
        site = spec.groupby("siteID").agg(individuals=("individualID", "size"), taxa=("bestScientificName", "nunique"),
                                          domainID=("domainID", "first")).reset_index()
        loc = neon_field.groupby("siteID").agg(lat=("decimalLatitude", "mean"), lon=("decimalLongitude", "mean")).reset_index()
        site = site.merge(loc, on="siteID").merge(joinable.groupby("siteID").size().rename("measured").reset_index(), on="siteID", how="left")
        site["measured"] = site["measured"].fillna(0).astype(int)
        P.append("<figure>" + site_map(site) +
                 "<figcaption>NEON terrestrial sites. Circle area is the number of pinned beetles in NEON's tables; a filled circle means "
                 "at least one specimen from that site was segmented from an image, a hollow one means none was. No coastline is drawn; "
                 "Alaska, Hawaii and Puerto Rico are insets on their own scales. Hover for counts.</figcaption></figure>")
        yr = spec.assign(rel=np.where(spec["release"].astype(str) == "PROVISIONAL", "provisional", "released")).groupby(["year", "rel"]).size().unstack(fill_value=0)
        yr = yr.reindex(columns=["released", "provisional"], fill_value=0)
        P.append("<h3>Pinned beetles per collection year</h3><figure>" +
                 stacked_columns([str(int(y)) for y in yr.index], {"released": yr["released"].tolist(), "provisional": yr["provisional"].tolist()}, "individuals") +
                 legend([(f"in {next((r for r in rel.index if r != 'PROVISIONAL'), 'the latest release')}", "s1"), ("provisional: no DOI, may change", "s2")]) +
                 "<figcaption>Individuals pinned and identified, by year collected.</figcaption></figure>")
        per = spec.groupby("siteID").agg(individuals=("individualID", "size"), taxa=("bestScientificName", "nunique"),
                                         expert_ids=("idSource", lambda x: (x == "expert").sum()), first=("year", "min"), last=("year", "max"),
                                         accessioned=("in_biorepository", "sum"), photographed=("biorepo_n_images", lambda x: (x > 0).sum())).reset_index()
        per.columns = ["Site", "Pinned individuals", "Taxa", "Expert-identified", "First year", "Last year", "In Biorepository", "Photographed"]
        P.append("<details><summary>Table: pinned individuals per site</summary>" +
                 table(per, numeric=("Pinned individuals", "Taxa", "Expert-identified", "In Biorepository", "Photographed")) + "</details>")
    else:
        P.append("<h2>NEON trap tables</h2><p><strong>Not in this build.</strong> The NEON data endpoint requires an API token. "
                 "Run <code>nbs neon</code> with <code>NEON_TOKEN</code> set, then <code>nbs lake</code>.</p>")

    # ---- from trap to measurement ---------------------------------------------------------------
    P.append("<h2>From trap to measurement</h2>")
    P.append("<p>The trap records are complete for every site. Photographs are not: most specimens were never imaged, and a "
             "measurement can only exist where an image does.</p>")
    if spec is not None:
        measured_ind = meas.loc[meas["source"] == "biorepo", "individualID"].dropna().nunique()
        P.append("<figure>" + funnel([("Pinned beetles in NEON's tables", len(spec)),
                                     ("with a Biorepository record", int(spec["in_biorepository"].sum())),
                                     ("photographed at the Biorepository", int((spec["biorepo_n_images"] > 0).sum())),
                                     ("segmented and measured here", int(measured_ind))]) +
                 "<figcaption>Pinned individuals at each step, as a share of all pinned individuals. The other image pools below "
                 "add specimens that are not pinned vouchers (ethanol trays) or cannot be traced to an individual (sentinel).</figcaption></figure>")
    P.append("<h3>Image pools</h3>")
    rows = []
    for src, g in todo[todo["source"] != "herp"].groupby("source"):
        mm = meas[meas["source"] == src]
        rows.append({"Image pool": POOL_NAMES.get(src, src), "Images": len(g),
                     "Processed": int(g["image_id"].isin(done_ids).sum()), "Beetles": len(mm),
                     "With mm scale": int(mm["qc_has_scale"].sum()), "Elytra measured": int(mm["parts_ok"].fillna(False).sum()),
                     "NEON sites": g["siteID"].nunique()})
    P.append(table(pd.DataFrame(rows), numeric=("Images", "Processed", "Beetles", "With mm scale", "Elytra measured", "NEON sites")))
    P.append(f'<p class="note">The first three pools link back to NEON records. The sentinel crops do not: their site and specimen '
             f'identifiers are anonymised. {fmt(int(man["duplicate_of"].notna().sum()))} images in the manifest are duplicates and were '
             "processed once (the Biorepository's tray photos are downsized copies of the 2018 trays).</p>")
    by_site = joinable.groupby("siteID").size().sort_values(ascending=False)
    if len(by_site):
        P.append("<h3>Beetles measured per NEON site</h3>")
        P.append("<figure>" + hbar(list(by_site.index), [int(v) for v in by_site.values], "specimens") +
                 "<figcaption>Specimens segmented from images that link to a NEON site (Biorepository, 2018 trays, Hawaii). "
                 "Coverage follows where imaging happened, not where beetles were caught.</figcaption></figure>")

    # ---- everything preserved from the traps ---------------------------------------------------
    P.append("<h2>What NEON keeps from each trap</h2>")
    P.append("<p>A pitfall sample is sorted into carabids, other invertebrates, reptiles and amphibians, and small mammals. "
             "A subset of carabids is pinned; the rest, and all the bycatch, is preserved in ethanol, first as one vial per trap "
             "(trap sorting) and then pooled per plot and bout for long-term storage (archive pooling). The Biorepository holds "
             "these as separate collections.</p>")
    if pres is not None and len(pres):
        ts = pres[pres["level"] == "trap sorting"].groupby("specimen_group").size().sort_values(ascending=False)
        P.append("<h3>Vials made at trap sorting, by content</h3><figure>" +
                 hbar([str(i) for i in ts.index], [int(v) for v in ts.values], "vials", label_w=160) +
                 "<figcaption>One vial per trap and taxon for carabids and vertebrates; one of unsorted material per trap for "
                 "invertebrate bycatch. All sites, all years.</figcaption></figure>")
        g = pres.groupby(["specimen_group", "level"]).agg(
            samples=("sample_id", "size"), individuals=("individualCount", "sum"), sites=("siteID", "nunique"),
            accessioned=("in_biorepository", "sum")).reset_index()
        g["individuals"] = g["individuals"].where(g["individuals"] > 0)
        g["pct"] = g["accessioned"] / g["samples"] * 100
        g.columns = ["Group", "Level", "NEON vials", "Individuals counted", "Sites", "Linked to an accession", "Linked (%)"]
        P.append("<p>Every vial in <code>bet_sorting</code> and <code>bet_archivepooling</code> is matched to its Biorepository record "
                 "through the hashed sample ID both sides publish (<code>preserved_samples</code>).</p>")
        P.append(table(g, numeric=("NEON vials", "Individuals counted", "Sites", "Linked to an accession", "Linked (%)"), digits={"Linked (%)": 1}))
        P.append('<p class="note">Individuals are counted only at trap sorting, and not for invertebrate bycatch. Most trap-sorting vials '
                 "have no accession of their own because their contents were pooled into archive vials.</p>")
    if colls is not None:
        c = colls.copy()
        if gbif is not None:
            c = c.merge(gbif.groupby("collection").size().rename("gbif").reset_index(), on="collection", how="left")
        else:
            c["gbif"] = np.nan
        c["years"] = c["first_year"].astype(str) + "–" + c["last_year"].astype(str)
        c = c[["collection", "collection_name", "specimen_group", "pooling", "records", "records_with_images", "images", "sites", "years", "gbif"]]
        c.columns = ["Code", "Biorepository collection", "Group", "Level", "Records", "With images", "Images", "Sites", "Years", "On GBIF"]
        P.append("<h3>Biorepository collections</h3>" + table(c, numeric=("Records", "With images", "Images", "Sites", "On GBIF")))
        P.append('<p class="note">GBIF republishes nine of these collections; the two invertebrate-bycatch collections are not on GBIF. '
                 "GBIF's media links point back to the Biorepository, so it contributes identifiers and taxonomy, not images.</p>")

    # ---- herptile bycatch -----------------------------------------------------------------------
    if len(herps):
        P.append("<h2>Reptiles and amphibians in the traps</h2>")
        hman = man[man["source"] == "herp"]
        hdone = int(hman["image_id"].isin(done_ids).sum())
        cnt = hman[hman["process"]].merge(res[["image_id", "n_instances"]], on="image_id", how="inner")
        cnt["n_rec"] = pd.to_numeric(cnt["individualCount"], errors="coerce")
        cnt = cnt[cnt["n_rec"].notna()]
        P.append(f"<p>{fmt(hdone)} of {fmt(int(hman['process'].sum()))} herptile bycatch photographs were segmented, giving "
                 f"{fmt(len(herps))} outlines from {fmt(herps['biorepo_occid'].nunique())} records at {fmt(herps['siteID'].nunique())} sites. "
                 "Most vials were photographed twice, once from each side, so an animal usually appears as two outlines. "
                 "Each photo has a metric ruler in frame, so every outline has a millimetre scale. Length here is the longest path "
                 "along the body's midline: total length including the tail for salamanders and lizards, and out to the toes for a frog "
                 "with a leg extended. It is not snout-vent length.</p>")
        if len(cnt):
            P.append(f"<p>There are no human outlines to compare with, but each record states how many animals the vial holds. "
                     f"The number of outlines in a photo equals that count on {(cnt['n_instances'] == cnt['n_rec']).mean() * 100:.1f}% of "
                     f"{fmt(len(cnt))} photos and is within one on {((cnt['n_instances'] - cnt['n_rec']).abs() <= 1).mean() * 100:.1f}%; "
                     f"it is higher on {(cnt['n_instances'] > cnt['n_rec']).mean() * 100:.1f}% (detached limbs and tails outlined separately) "
                     f"and lower on {(cnt['n_instances'] < cnt['n_rec']).mean() * 100:.1f}%. "
                     f"{fmt(int((herps['largest_blob_frac'] < 0.8).sum()))} outlines are speckled, failed masks and are flagged.</p>")
        good = herps[herps["qc_has_scale"] & (herps["largest_blob_frac"].fillna(1) > 0.8) & ~flag(herps["touches_edge"])]
        top = good.groupby("scientificName").size().sort_values(ascending=False).head(15)
        P.append("<h3>Most photographed taxa</h3><figure>" + hbar(list(top.index), [int(v) for v in top.values], "outlines", label_w=190) +
                 "<figcaption>The 15 most frequent names among the outlines, as recorded by NEON and the collection's curators.</figcaption></figure>")
        panels = []
        bins = np.arange(0, 210, 10)
        rec_class = good["biorepo_occid"].astype(str).map(herp_sug.set_index("occid")["class"]) if herp_sug is not None else None
        if rec_class is not None:
            for cls, name in (("Amphibia", "Amphibians"), ("Reptilia", "Reptiles")):
                vals = good.loc[(rec_class == cls).to_numpy(), "midline_length_mm"].to_numpy(float)
                if len(vals) >= 10:
                    panels.append(f"<figure>{histogram(vals, 'midline length (mm)', bins)}<figcaption>{name}: {fmt(len(vals))} outlines, "
                                  f"median {np.nanmedian(vals):.0f} mm.</figcaption></figure>")
        if panels:
            P.append("<h3>Length</h3>" + '<div class="grid2">' + "".join(panels) + "</div>")
        if herp_sug is not None:
            need = herp_sug[~flag(herp_sug["to_species"])]
            n_sug = int(need["suggested_species"].notna().sum())
            P.append("<h3>Species suggestions from GBIF</h3>")
            known1 = herp_sug[herp_sug["confident"] & (herp_sug["n_candidates"] == 1) & herp_sug["range_agrees"].notna()]
            n_single = int((need["suggestion_basis"].fillna("").str.startswith("only")).sum())
            P.append(f"<p>{fmt(int(herp_sug['to_species'].sum()))} of {fmt(len(herp_sug))} herptile records are already identified to species. "
                     f"For the other {fmt(len(need))}, GBIF was asked which species of the recorded genus or family people have observed "
                     f"within 100 km of the site, and BioCLIP 2 scored each segmented animal against those candidates. "
                     f"{fmt(n_sug)} records get a suggestion: {fmt(n_single)} because only one candidate species occurs nearby, and "
                     f"{fmt(n_sug - n_single)} because the image model is at least 60% sure among several. "
                     "These are suggestions for a curator, not determinations.</p>")
            if len(known1):
                P.append(f"<p>How far to trust them, from running the same procedure where the species is known: when only one candidate "
                         f"occurs nearby it is the recorded species on {known1['range_agrees'].mean() * 100:.1f}% of {fmt(len(known1))} records. "
                         "When there are several candidates the image model alone is weak on preserved specimens, and worth using only "
                         "when it is confident (table below).</p>")
            if herp_val is not None and len(herp_val):
                hv = herp_val.copy()
                for c2 in ("recorded_species_among_candidates", "range_pick_accuracy", "bioclip_pick_accuracy"):
                    hv[c2] = hv[c2] * 100
                hv.columns = ["Checked on records with a known species", "Records", "Median candidates", "Known species among candidates (%)",
                              "Most-observed candidate correct (%)", "BioCLIP pick correct (%)"]
                P.append(table(hv, numeric=tuple(hv.columns[1:]), digits={"Median candidates": 0, "Known species among candidates (%)": 1,
                                                                         "Most-observed candidate correct (%)": 1, "BioCLIP pick correct (%)": 1}))
                P.append('<p class="note">The same procedure was run where the answer is known. "Most-observed candidate" is the range-only '
                         "guess: the candidate with the most GBIF observations near the site.</p>")
            sug = need[need["suggested_species"].notna()][["catalogNumber", "siteID", "recorded_name", "suggested_species", "n_candidates", "bioclip_prob", "suggestion_basis"]]
            if len(sug):
                sug.columns = ["Catalog number", "Site", "Recorded as", "Suggested species", "Candidates", "BioCLIP probability", "Basis"]
                P.append("<details><summary>Table: suggestions for records not identified to species</summary>" +
                         table(sug, numeric=("Candidates", "BioCLIP probability"), digits={"BioCLIP probability": 2}) + "</details>")

    # ---- validation -----------------------------------------------------------------------------
    P.append("<h2>How well it works</h2>")
    if val is not None and len(val):
        v = val.set_index(["group", "metric"])

        def get(group, metric):
            try:
                r = v.loc[(group, metric)]
                return float(r["value"]), int(r["n"]) if pd.notna(r["n"]) else None
            except KeyError:
                return None, None

        prec, n_det = get("detection_2018_trays", "precision_iou50")
        rec, n_gt = get("detection_2018_trays", "recall_iou50")
        cov, n_ann = get("detection_2018_trays", "annotated_individuals_covered")
        exact, n_tr = get("detection_2018_trays", "trays_with_exact_count")
        if prec is not None:
            P.append("<h3>Finding specimens in tray photos</h3>")
            P.append(f"<p>On {fmt(n_tr)} tray photos from 2018, the pipeline found {fmt(n_det)} specimens against {fmt(n_gt)} human boxes: "
                     f"precision {prec:.3f}, recall {rec:.3f} at IoU 0.5, and the count matched exactly on {exact * 100:.0f}% of trays. "
                     f"Those boxes began as machine proposals that people corrected, which flatters a detector of the same family. "
                     f"The independent check is the volunteers' elytra lines: {cov * 100:.1f}% of {fmt(n_ann)} annotated individuals "
                     f"have a predicted mask under the line.</p>")
        rows = []
        labels = {"elytra_midline_length": "Elytra length (midline)", "elytra_full_length": "Elytra length (full extent)",
                  "elytra_base_width": "Elytra width at base", "elytra_max_width": "Elytra maximum width",
                  "pronotum_base_width": "Pronotum basal width"}
        for group, gname in (("traits_2018_trays_ethanol", "2018 ethanol trays"), ("traits_hawaii_pinned", "Hawaii pinned")):
            for key, lab in labels.items():
                r, n = get(group, f"{key}.all.pearson_r")
                if r is None:
                    continue
                rows.append({"Dataset": gname, "Measurement": lab, "Specimens": n,
                             "Bias (mm)": get(group, f"{key}.all.bias_mm")[0], "Mean abs. error (mm)": get(group, f"{key}.all.mae_mm")[0],
                             "Median abs. error (%)": get(group, f"{key}.all.median_abs_pct")[0],
                             "Within 10% (%)": get(group, f"{key}.all.within_10pct")[0] * 100, "Pearson r": r,
                             "Slope": get(group, f"{key}.all.slope")[0]})
        if rows:
            P.append("<h3>Measurements against human annotations</h3>")
            P.append("<p>Each predicted measurement is compared with the line a person drew on the same specimen. "
                     "Bias is predicted minus human.</p>")
            P.append(table(pd.DataFrame(rows), numeric=("Specimens", "Bias (mm)", "Mean abs. error (mm)", "Median abs. error (%)", "Within 10% (%)", "Pearson r", "Slope"),
                           digits={"Bias (mm)": 2, "Mean abs. error (mm)": 2, "Median abs. error (%)": 1, "Within 10% (%)": 0, "Pearson r": 3, "Slope": 3}))
            panels = []
            for src, name, pcol in (("hf2018", "2018 ethanol trays", "elytra_midline_length_mm"), ("hawaii", "Hawaii pinned", "elytra_midline_length_mm")):
                t = vtraits[(vtraits["source"] == src) & vtraits["parts_ok"].fillna(False)] if vtraits is not None else pd.DataFrame()
                if len(t) >= 3:
                    panels.append(f"<figure>{scatter(t['ann_elytra_length_mm'].to_numpy(float), t[pcol].to_numpy(float), 'human elytra length (mm)', 'predicted (mm)')}"
                                  f"<figcaption>{esc(name)}: predicted midline elytra length against the human line, with the 1:1 line. "
                                  f"{fmt(len(t))} specimens (a random 1,500 drawn if more).</figcaption></figure>")
            if panels:
                P.append('<div class="grid2">' + "".join(panels) + "</div>")
    else:
        P.append("<p>Validation has not been run for this build.</p>")
    if sam3 is not None and len(sam3):
        g = sam3["n_human_boxes"].sum()
        srow = []
        for name, lab in (("main", "Grounding DINO + SAM 2.1 (this pipeline)"), ("sam3", "SAM 3, text prompt \"" + str(sam3["sam3_prompt"].iat[0]) + "\"")):
            tp, n = sam3[f"{name}_tp_iou50"].sum(), sam3[f"{name}_n"].sum()
            srow.append({"Method": lab, "Specimens found": int(n), "Precision": tp / max(n, 1), "Recall": tp / max(g, 1),
                         "Annotated individuals covered (%)": sam3[f"{name}_annotated_covered"].sum() / max(sam3["n_annotated"].sum(), 1) * 100})
        P.append(f"<h3>SAM 3 comparison</h3><p>On {len(sam3)} tray photos ({fmt(g)} human boxes). SAM 3 took "
                 f"{sam3['sam3_seconds'].mean():.1f} s per tray at {int(sam3['sam3_input_long_side'].iat[0])} px.</p>")
        P.append(table(pd.DataFrame(srow), numeric=("Specimens found", "Precision", "Recall", "Annotated individuals covered (%)"),
                       digits={"Precision": 3, "Recall": 3, "Annotated individuals covered (%)": 1}))

    # ---- scale ----------------------------------------------------------------------------------
    if scale_tbl is not None:
        P.append("<h2>Scale</h2>")
        s = scale_tbl.assign(pool=scale_tbl["image_id"].str.split("/").str[0])
        s = s.groupby("pool").agg(images=("image_id", "size"), with_scale=("px_per_mm", lambda x: x.notna().sum()),
                                  source=("scale_source", lambda x: ", ".join(sorted(set(x.dropna()))) or "none")).reset_index()
        s["pool"] = s["pool"].map(POOL_NAMES).fillna(s["pool"])
        s.columns = ["Image pool", "Images checked", "With scale", "How"]
        P.append(table(s, numeric=("Images checked", "With scale")))
        P.append('<p class="note">Biorepository rows count pinned-specimen photos only; the 2016 individual photos have no scale in frame. '
                 "Specimens without a trusted scale keep pixel measurements and null millimetre columns.</p>")

    # ---- sizes ----------------------------------------------------------------------------------
    sized = meas[flag(meas["qc_has_scale"]) & flag(meas["parts_complete"]) & ~flag(meas["trunk_touches_edge"], missing=True)]
    panels = []
    bins = np.arange(0, 42, 2)
    for src in ("hf2018", "hawaii", "biorepo", "sentinel"):
        vals = sized.loc[sized["source"] == src, "body_length_parts_mm"].to_numpy(float)
        if len(vals) >= 20:
            panels.append(f"<figure>{histogram(vals, 'body length (mm)', bins)}<figcaption>{esc(POOL_NAMES[src])}: "
                          f"{fmt(len(vals))} specimens, median {np.nanmedian(vals):.1f} mm.</figcaption></figure>")
    if panels:
        P.append("<h2>Body length</h2><p>Head + pronotum + elytra along the body axis, for specimens with a scale, all three parts "
                 "found and the body clear of the photo's edge. Each panel has its own count axis.</p>")
        P.append('<div class="grid2">' + "".join(panels) + "</div>")

    # ---- examples -------------------------------------------------------------------------------
    if inst is not None:
        picks = []
        for src, kind in (("hf2018", None), ("hawaii", None), ("biorepo", "pinned"), ("biorepo", "individual"), ("sentinel", None), ("herp", None)):
            ids = allmeas[(allmeas["source"] == src) & ((allmeas["image_kind"] == kind) if kind else True)]["image_id"].drop_duplicates()
            if len(ids):
                picks.append((src, kind, ids.sample(1, random_state=7).iat[0]))
        shots = []
        for src, kind, image_id in picks:
            uri = overlay(image_id, inst, man)
            if uri:
                cap = POOL_NAMES[src] + (f", {kind}" if kind else "")
                shots.append(f'<figure><img src="{uri}" alt="segmentation overlay for {esc(image_id)}"><figcaption>{esc(cap)}<br><code>{esc(image_id)}</code></figcaption></figure>')
        if shots:
            P.append("<h2>Examples</h2><p>One randomly chosen image per pool, not selected for quality. Yellow outline: specimen mask. "
                     "Red, green, blue: head, pronotum, elytra (beetles only).</p>")
            P.append('<div class="shots">' + "".join(shots) + "</div>")

    # ---- limitations ----------------------------------------------------------------------------
    n_noscale = int((~flag(meas["qc_has_scale"])).sum())
    fallback = meas["det_source"].isin(["backdrop_blob", "full_frame"])
    P.append("<h2>Limitations</h2><ul>"
             "<li><strong>Images, not specimens, set the coverage.</strong> About 1% of pinned vouchers at the Biorepository have a photo.</li>"
             f"<li><strong>{fmt(n_noscale)} specimens have no millimetre scale</strong> (Biorepository grid-backdrop photos, most domain-office photos, "
             "the 2016 individual photos, and rulers the tick reader rejected). Their measurements are in pixels.</li>"
             "<li><strong>Elytra length reads long.</strong> Against people's lines the predicted elytra length is about 8 to 10% too long "
             "(0.8 mm on the 2018 ethanol specimens), consistently, with a high correlation. Treat absolute elytra lengths as biased high; "
             "comparisons within a dataset are much better than that.</li>"
             "<li><strong>The part model was trained on pinned specimens.</strong> It is applied to ethanol specimens as well; the validation table shows what that costs.</li>"
             "<li><strong>Herptile lengths are midline lengths of outlines</strong>, not snout-vent lengths, each animal is usually outlined twice "
             "(two photos per vial), and the species suggestions are a curator's starting point, not identifications.</li>"
             "<li><strong>Sentinel scale rests on an assumption</strong>: that each ruler crop has the same resolution as the specimen crops from the same tray photo. "
             "Sentinel specimens also cannot be linked to NEON sites.</li>"
             f"<li><strong>{fmt(int(fallback.sum()))} specimens used a fallback prompt</strong> (no detection or incomplete part labels); check <code>det_source</code>.</li>"
             f"<li><strong>Quality flags are not a review.</strong> {fmt(int(meas['qc_low_solidity'].fillna(False).sum()))} masks have solidity below 0.5 and "
             f"{fmt(int(meas['trunk_touches_edge'].fillna(False).sum()))} specimens have a body cut by the photo's edge. Nobody inspected the masks one by one.</li>"
             "<li><strong>The 2016 individual photos are the weakest pool.</strong> They are crops about 110 by 190 pixels with no scale; "
             f"{fmt(int((~flag(meas.loc[(meas['source'] == 'biorepo') & (meas['image_kind'] == 'individual'), 'parts_ok'])).sum()))} of "
             f"{fmt(int(((meas['source'] == 'biorepo') & (meas['image_kind'] == 'individual')).sum()))} have no elytra found, and some outlines are plainly wrong.</li>"
             "<li><strong>Ventral and lateral photos</strong> are segmented like dorsal ones; filter on <code>view</code> before using their measurements.</li></ul>")

    # ---- provenance -----------------------------------------------------------------------------
    if prov is not None:
        P.append("<h2>Provenance</h2>" + table(prov.rename(columns={"kind": "Kind", "name": "Name", "value": "Value", "detail": "Detail"})))
    P.append('<p class="note">Lake: <code>ducklake/beetles.ducklake</code> (attach from inside <code>ducklake/</code>) and '
             "<code>ducklake/beetles.duckdb</code>. Table descriptions are in <code>table_catalog</code>; see the repository's "
             "<code>docs/</code> for the data dictionary and methods.</p>")

    body = f"<main>{''.join(P)}</main>"
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>NEON Pitfall Beetles</title><style>{CSS}</style></head><body>{body}</body></html>")
    out_dir = config.OUT / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(page)
    # the same page without the document shell, for viewers that supply their own (a published web artifact)
    (config.OUT / "report_fragment.html").write_text(f"<title>NEON Pitfall Beetles</title>\n<style>{CSS}</style>\n{body}")
    # small companion tables people tend to want as CSV
    slim = allmeas.drop(columns=[c for c in allmeas.columns if c.endswith("_rle")], errors="ignore")
    slim[slim["source"] != "sentinel"].dropna(axis=1, how="all").to_csv(out_dir / "measurements_neon_linked.csv", index=False)
    if val is not None:
        val.to_csv(out_dir / "validation_summary.csv", index=False)
    return str(out_dir / "report.html")


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
