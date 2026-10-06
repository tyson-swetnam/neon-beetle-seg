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
--text2:#c3c2b7;--muted:#8f8e86;--grid:#383835;--s1:#3987e5;--s2:#d95926;--wash:rgba(57,135,229,.14)}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#242422;--text:#fff;--text2:#c3c2b7;--muted:#8f8e86;--grid:#383835;
--s1:#3987e5;--s2:#d95926;--wash:rgba(57,135,229,.14)}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
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
.shots{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.shots img{width:100%;border-radius:8px;display:block}
.legend{display:flex;gap:16px;font-size:14px;color:var(--text2);margin:4px 0}.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px}
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


# ---- example overlays -------------------------------------------------------------------------
PART_COLOURS = {"head": (255, 70, 70), "pronotum": (70, 215, 70), "elytra": (70, 130, 255)}


def overlay(image_id: str, inst: pd.DataFrame, man: pd.DataFrame, max_side: int = 620) -> str | None:
    row = man[man["image_id"] == image_id]
    if row.empty:
        return None
    img = np.asarray(Image.open(config.ROOT / row.iloc[0]["local_path"]).convert("RGB")).copy()
    for r in inst[inst["image_id"] == image_id].itertuples():
        if isinstance(getattr(r, "elytra_rle", None), str):
            sub = img[int(r.pwin_y1):int(r.pwin_y1) + int(r.parts_h), int(r.pwin_x1):int(r.pwin_x1) + int(r.parts_w)]
            for part, col in PART_COLOURS.items():
                pm = measure.rle_decode(getattr(r, f"{part}_rle"), int(r.parts_h), int(r.parts_w))[:sub.shape[0], :sub.shape[1]]
                sub[pm] = (0.55 * sub[pm] + 0.45 * np.array(col)).astype(np.uint8)
        m = measure.rle_decode(r.mask_rle, int(r.mask_h), int(r.mask_w))
        sub = img[int(r.win_y1):int(r.win_y1) + m.shape[0], int(r.win_x1):int(r.win_x1) + m.shape[1]]
        cnts, _ = cv2.findContours(m[:sub.shape[0], :sub.shape[1]].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(sub, cnts, -1, (255, 235, 0), max(1, round(max(img.shape[:2]) / 700)))
    s = max_side / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- report ----------------------------------------------------------------------------------
POOL_NAMES = {"biorepo": "NEON Biorepository", "hf2018": "2018 ethanol trays (HuggingFace)",
              "hawaii": "Hawaii pinned trays (HuggingFace)", "sentinel": "sentinel-beetles crops (HuggingFace)"}


def build() -> str:
    man, meas, res = read("image_manifest"), read("measurements"), read("image_results")
    inst = read("instances")
    val, vtraits = read("validation_summary"), read("validation_traits")
    neon_field, spec = read("neon_fielddata"), read("specimen_manifest")
    bio, gbif, sam3 = read("biorepo_records"), read("gbif_occurrences"), read("sam3_comparison")
    scale_tbl, prov = read("image_scale"), read("run_provenance")
    todo = man[man["duplicate_of"].isna() & man["is_carabid"]]
    done_ids = set(res.loc[res["status"] == "ok", "image_id"]) if res is not None else set()
    P: list[str] = []

    P.append(f"<h1>NEON ground beetles: segmentation and measurements, all sites</h1>"
             f'<p class="sub">neon-beetle-seg v{__version__} · built {time.strftime("%Y-%m-%d")} · '
             f'<a href="https://github.com/tyson-swetnam/neon-beetle-seg">code</a></p>')
    joinable = meas[meas["source"] != "sentinel"]
    n_sites = joinable["siteID"].nunique()
    tiles = [
        (fmt(len(done_ids)), "images processed"),
        (fmt(len(meas)), "specimens segmented"),
        (fmt(int(meas["qc_has_scale"].sum())), "with millimetre scale"),
        (fmt(n_sites), "NEON sites with measured specimens"),
        (fmt(meas["scientificName"].nunique()), "taxa measured"),
    ]
    if neon_field is not None:
        tiles.insert(0, (fmt(neon_field["siteID"].nunique()), "NEON sites in the trap tables"))
    P.append('<div class="tiles">' + "".join(f'<div class="tile"><div class="v">{v}</div><div class="l">{esc(lab)}</div></div>' for v, lab in tiles) + "</div>")
    pending = len(todo) - len(done_ids)
    if pending > 0:
        P.append(f'<p class="note">This build is partial: {fmt(pending)} of {fmt(len(todo))} images had not been processed when it was written.</p>')

    # ---- coverage -------------------------------------------------------------------------------
    P.append("<h2>What is covered</h2>")
    P.append("<p>NEON's pitfall programme has trap records for every site, but most specimens were never "
             "photographed. Measurements exist only where an image exists. Four image pools were processed; "
             "the first three link back to NEON records, the sentinel crops do not because their site and "
             "specimen identifiers are anonymised.</p>")
    rows = []
    for src, g in todo.groupby("source"):
        mm = meas[meas["source"] == src]
        rows.append({"Image pool": POOL_NAMES.get(src, src), "Images": len(g),
                     "Processed": int(g["image_id"].isin(done_ids).sum()), "Specimens": len(mm),
                     "With mm scale": int(mm["qc_has_scale"].sum()), "Elytra found": int(mm["parts_ok"].fillna(False).sum()),
                     "NEON sites": g["siteID"].nunique()})
    P.append(table(pd.DataFrame(rows), numeric=("Images", "Processed", "Specimens", "With mm scale", "Elytra found", "NEON sites")))
    skipped = man[man["duplicate_of"].notna() | ~man["is_carabid"]]
    P.append(f'<p class="note">{fmt(int(man["duplicate_of"].notna().sum()))} images are duplicates (the Biorepository\'s tray photos are '
             f'downsized copies of the 2018 HuggingFace trays) and {fmt(int((~man["is_carabid"]).sum()))} belong to non-carabid records; '
             f"these {fmt(len(skipped))} are in the manifest but were not processed.</p>")

    by_site = joinable.groupby("siteID").size().sort_values(ascending=False)
    if len(by_site):
        P.append("<h3>Specimens measured per NEON site</h3>")
        P.append("<figure>" + hbar(list(by_site.index), [int(v) for v in by_site.values], "specimens") +
                 "<figcaption>Specimens segmented from images that link to a NEON site (Biorepository, 2018 trays, Hawaii). "
                 "Coverage follows where imaging happened, not where beetles were caught.</figcaption></figure>")

    # ---- NEON tables ----------------------------------------------------------------------------
    P.append("<h2>NEON trap tables</h2>")
    if neon_field is not None and spec is not None:
        rel = neon_field["release"].astype(str).value_counts()
        P.append(f"<p>DP1.10022.001 stacked for {neon_field['siteID'].nunique()} sites in {neon_field['domainID'].nunique()} domains, "
                 f"collection dates {str(neon_field['collectDate'].min())[:10]} to {str(neon_field['collectDate'].max())[:10]}. "
                 f"{fmt(len(spec))} pinned individuals; {fmt(int(spec.get('in_biorepository', pd.Series(dtype=bool)).sum()))} are accessioned at the Biorepository and "
                 f"{fmt(int((spec.get('biorepo_n_images', pd.Series(dtype=int)) > 0).sum()))} have a photo.</p>")
        P.append(table(pd.DataFrame({"Release": rel.index, "bet_fielddata rows": rel.values}), numeric=("bet_fielddata rows",)))
        P.append('<p class="note">Rows marked PROVISIONAL are newer than the latest release, have no DOI and can still change.</p>')
        per = spec.groupby("siteID").agg(individuals=("individualID", "size"), taxa=("bestScientificName", "nunique"),
                                         expert_ids=("idSource", lambda s: (s == "expert").sum()),
                                         first=("year", "min"), last=("year", "max")).reset_index()
        per = per.rename(columns={"siteID": "Site", "individuals": "Pinned individuals", "taxa": "Taxa",
                                  "expert_ids": "Expert-identified", "first": "First year", "last": "Last year"})
        P.append("<details><summary>Pinned individuals per site</summary>" +
                 table(per, numeric=("Pinned individuals", "Taxa", "Expert-identified")) + "</details>")
    else:
        P.append("<p><strong>Not in this build.</strong> The NEON data endpoint requires an API token, and none was available when "
                 "this was built. Run <code>nbs neon</code> with <code>NEON_TOKEN</code> set, then <code>nbs lake</code>.</p>")
    if bio is not None:
        b = bio.groupby("collection").agg(records=("occid", "size"), with_images=("n_images", lambda s: (s > 0).sum()),
                                          sites=("siteID", "nunique")).reset_index()
        b.columns = ["Biorepository collection", "Records", "With images", "Sites"]
        if gbif is not None:
            g = gbif.groupby("collection").size().rename("GBIF records").reset_index().rename(columns={"collection": "Biorepository collection"})
            b = b.merge(g, on="Biorepository collection", how="left")
        P.append("<h3>Biorepository and GBIF records</h3>" + table(b, numeric=tuple(c for c in b.columns if c != "Biorepository collection")))
        P.append('<p class="note">GBIF republishes the Biorepository records and its media links point back to the Biorepository, '
                 "so it contributes identifiers and taxonomy, not images.</p>")

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
    sized = meas[meas["qc_has_scale"] & meas["parts_complete"].fillna(False) & ~meas["touches_edge"].fillna(False)]
    panels = []
    bins = np.arange(0, 42, 2)
    for src in ("hf2018", "hawaii", "biorepo", "sentinel"):
        vals = sized.loc[sized["source"] == src, "body_length_parts_mm"].to_numpy(float)
        if len(vals) >= 20:
            panels.append(f"<figure>{histogram(vals, 'body length (mm)', bins)}<figcaption>{esc(POOL_NAMES[src])}: "
                          f"{fmt(len(vals))} specimens, median {np.nanmedian(vals):.1f} mm.</figcaption></figure>")
    if panels:
        P.append("<h2>Body length</h2><p>Head + pronotum + elytra along the body axis, for specimens with a scale, all three parts "
                 "found and a mask clear of the photo's edge. Each panel has its own count axis.</p>")
        P.append('<div class="grid2">' + "".join(panels) + "</div>")

    # ---- examples -------------------------------------------------------------------------------
    if inst is not None:
        picks = []
        for src, kind in (("hf2018", None), ("hawaii", None), ("biorepo", "pinned"), ("biorepo", "individual"), ("sentinel", None)):
            ids = meas[(meas["source"] == src) & ((meas["image_kind"] == kind) if kind else True)]["image_id"].drop_duplicates()
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
                     "Red, green, blue: head, pronotum, elytra.</p>")
            P.append('<div class="shots">' + "".join(shots) + "</div>")

    # ---- limitations ----------------------------------------------------------------------------
    n_noscale = int((~meas["qc_has_scale"]).sum())
    fallback = meas["det_source"].isin(["backdrop_blob", "full_frame"])
    P.append("<h2>Limitations</h2><ul>"
             "<li><strong>Images, not specimens, set the coverage.</strong> About 1% of pinned vouchers at the Biorepository have a photo.</li>"
             f"<li><strong>{fmt(n_noscale)} specimens have no millimetre scale</strong> (Biorepository grid-backdrop photos, most domain-office photos, "
             "the 2016 individual photos, and rulers the tick reader rejected). Their measurements are in pixels.</li>"
             "<li><strong>Elytra length reads long.</strong> Against the volunteers' lines the predicted elytra length carries a positive bias of roughly a millimetre "
             "on ethanol specimens (see the table above); the slope is close to 1, so it is an offset rather than a scaling error. Treat absolute elytra lengths accordingly.</li>"
             "<li><strong>The part model was trained on pinned specimens.</strong> It is applied to ethanol specimens as well; the validation table shows what that costs.</li>"
             "<li><strong>Sentinel scale rests on an assumption</strong>: that each ruler crop has the same resolution as the specimen crops from the same tray photo. "
             "Sentinel specimens also cannot be linked to NEON sites.</li>"
             f"<li><strong>{fmt(int(fallback.sum()))} specimens used a fallback prompt</strong> (no detection or incomplete part labels); check <code>det_source</code>.</li>"
             f"<li><strong>Quality flags are not a review.</strong> {fmt(int(meas['qc_low_solidity'].fillna(False).sum()))} masks have solidity below 0.5 and "
             f"{fmt(int(meas['touches_edge'].fillna(False).sum()))} touch the photo's edge. Nobody inspected the masks one by one.</li>"
             "<li><strong>Ventral and lateral photos</strong> are segmented like dorsal ones; filter on <code>view</code> before using their measurements.</li></ul>")

    # ---- provenance -----------------------------------------------------------------------------
    if prov is not None:
        P.append("<h2>Provenance</h2>" + table(prov.rename(columns={"kind": "Kind", "name": "Name", "value": "Value", "detail": "Detail"})))
    P.append('<p class="note">Lake: <code>ducklake/beetles.ducklake</code> (attach from inside <code>ducklake/</code>) and '
             "<code>ducklake/beetles.duckdb</code>. Table descriptions are in <code>table_catalog</code>; see the repository's "
             "<code>docs/</code> for the data dictionary and methods.</p>")

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>NEON beetle segmentation report</title><style>{CSS}</style></head><body><main>{''.join(P)}</main></body></html>")
    out_dir = config.OUT / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(page)
    # small companion tables people tend to want as CSV
    slim = meas.drop(columns=[c for c in meas.columns if c.endswith("_rle")], errors="ignore")
    slim[slim["source"] != "sentinel"].to_csv(out_dir / "measurements_neon_linked.csv", index=False)
    if val is not None:
        val.to_csv(out_dir / "validation_summary.csv", index=False)
    return str(out_dir / "report.html")


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
