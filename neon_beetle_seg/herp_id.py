"""GBIF-assisted identification of the herptile bycatch (reptiles and amphibians).

Most herp bycatch records are already identified to species by NEON and the museum curators.
The rest stop at genus, family or order. For those, two independent sources are combined into a
*suggestion* (never a determination):

  1. Range. GBIF is asked which species of the recorded genus (or family, or order) people have
     observed within RADIUS_KM of the NEON site. Only human observations are counted, so NEON's
     own specimen records cannot vote for themselves. Often this leaves one or two candidates.
  2. Appearance. BioCLIP 2 (a vision-language model trained on the tree of life) scores the
     segmented specimen against the candidate species' names, zero-shot.

The same procedure is run on the records that *are* identified to species, where the answer is
known. That measures how far the suggestions can be trusted, and the result is stored with them.

Outputs (parquet, in data/tables/):
  herp_taxa             every recorded name matched to the GBIF backbone
  herp_candidates       candidate species per site and recorded higher taxon, with observation counts
  herp_id_suggestions   one row per herp record with photos: candidates, range pick, BioCLIP pick
  herp_id_validation    accuracy of each method on species-level records
"""
from __future__ import annotations

import json
import time

import httpx
import numpy as np
import pandas as pd
from PIL import Image

from . import config, measure

API = "https://api.gbif.org/v1"
RADIUS_KM = 100
BIOCLIP_ID = "hf-hub:imageomics/bioclip-2"
RANKS = ("species", "subspecies", "variety")
MIN_PROB = 0.6  # below this the BioCLIP pick is reported but not offered as the suggestion

Image.MAX_IMAGE_PIXELS = None


def _get(client: httpx.Client, url: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GBIF request failed: {url} {params}")


def herp_records() -> pd.DataFrame:
    r = pd.read_parquet(config.TABLES / "biorepo_records.parquet")
    r = r[r["specimen_group"] == "herptile bycatch"].copy()
    r["rank"] = r["taxonRank"].fillna("").str.lower()
    r["to_species"] = r["rank"].isin(RANKS)
    # the name to look candidates up under: the genus for species-level names, else the name itself
    r["higher_taxon"] = np.where(r["to_species"], r["scientificName"].fillna("").str.split().str[0], r["scientificName"])
    r["confident"] = r["to_species"] & r["identificationQualifier"].isna()
    for c in ("decimalLatitude", "decimalLongitude"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    return r


def match_taxa(names: list[tuple[str, str | None]], client: httpx.Client, cache: dict) -> pd.DataFrame:
    """GBIF backbone match for (name, class) pairs."""
    rows = []
    for name, cls in names:
        key = f"match|{name}|{cls}"
        if key not in cache:
            m = _get(client, f"{API}/species/match", {"name": name, "class": cls, "strict": "false"})
            cache[key] = {k: m.get(k) for k in ("usageKey", "acceptedUsageKey", "scientificName", "canonicalName", "rank",
                                                "status", "matchType", "confidence", "kingdom", "phylum", "class", "order",
                                                "family", "genus", "species", "speciesKey", "genusKey", "familyKey", "orderKey")}
        m = cache[key]
        rows.append(dict(recorded_name=name, recorded_class=cls, gbif_usageKey=m["usageKey"],
                         gbif_acceptedKey=m["acceptedUsageKey"] or m["usageKey"], gbif_name=m["scientificName"],
                         gbif_canonical=m["canonicalName"], gbif_rank=m["rank"], gbif_status=m["status"],
                         gbif_matchType=m["matchType"], gbif_confidence=m["confidence"], gbif_class=m["class"],
                         gbif_order=m["order"], gbif_family=m["family"], gbif_genus=m["genus"], gbif_species=m["species"],
                         gbif_speciesKey=m["speciesKey"], gbif_genusKey=m["genusKey"], gbif_familyKey=m["familyKey"],
                         gbif_orderKey=m["orderKey"]))
    return pd.DataFrame(rows)


def candidates(pairs: pd.DataFrame, client: httpx.Client, cache: dict) -> pd.DataFrame:
    """Species of each higher taxon observed near each site (GBIF human observations)."""
    rows = []
    for p in pairs.itertuples():
        key = f"cand|{p.taxonKey}|{p.siteID}|{RADIUS_KM}"
        if key not in cache:
            res = _get(client, f"{API}/occurrence/search", {
                "taxonKey": p.taxonKey, "geoDistance": f"{p.lat:.4f},{p.lon:.4f},{RADIUS_KM}km",
                "basisOfRecord": "HUMAN_OBSERVATION", "occurrenceStatus": "PRESENT", "limit": 0,
                "facet": "speciesKey", "facetLimit": 200})
            counts = res["facets"][0]["counts"] if res.get("facets") else []
            cache[key] = [(int(c["name"]), int(c["count"])) for c in counts]
        for species_key, n in cache[key]:
            skey = f"sp|{species_key}"
            if skey not in cache:
                s = _get(client, f"{API}/species/{species_key}", {})
                cache[skey] = {k: s.get(k) for k in ("canonicalName", "kingdom", "phylum", "class", "order", "family", "genus", "vernacularName")}
            s = cache[skey]
            rows.append(dict(siteID=p.siteID, higher_taxon=p.higher_taxon, taxonKey=p.taxonKey, speciesKey=species_key,
                             species=s["canonicalName"], n_observations=n, radius_km=RADIUS_KM,
                             # the taxonomic string BioCLIP was trained to read
                             prompt="a photo of " + " ".join(str(s[k]) for k in ("kingdom", "phylum", "class", "order", "family") if s.get(k))
                                    + f" {s['canonicalName']}"))
    return pd.DataFrame(rows)


def build_candidates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """GBIF part: backbone matches and per-site candidate lists. Network only, no GPU."""
    recs = herp_records()
    cache_path = config.GBIF_DIR / "herp_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with httpx.Client(timeout=120, headers={"User-Agent": "neon-beetle-seg/0.2"}) as client:
        names = sorted({(n, c) for n, c in zip(recs["scientificName"], recs["class"]) if isinstance(n, str)}
                       | {(n, c) for n, c in zip(recs["higher_taxon"], recs["class"]) if isinstance(n, str) and n})
        taxa = match_taxa(names, client, cache)
        cache_path.write_text(json.dumps(cache))
        tkey = taxa.drop_duplicates("recorded_name").set_index("recorded_name")
        recs["speciesKey"] = recs["scientificName"].map(tkey["gbif_speciesKey"]).where(recs["to_species"])
        recs["taxonKey"] = recs["higher_taxon"].map(tkey["gbif_acceptedKey"])
        site = recs.groupby("siteID").agg(lat=("decimalLatitude", "mean"), lon=("decimalLongitude", "mean"))
        pairs = recs.dropna(subset=["taxonKey", "siteID"]).drop_duplicates(["siteID", "higher_taxon"])[["siteID", "higher_taxon", "taxonKey"]]
        pairs = pairs.join(site, on="siteID").dropna(subset=["lat", "lon"])
        pairs["taxonKey"] = pairs["taxonKey"].astype(int)
        cand = candidates(pairs, client, cache)
        cache_path.write_text(json.dumps(cache))
    taxa.to_parquet(config.TABLES / "herp_taxa.parquet", index=False)
    cand.to_parquet(config.TABLES / "herp_candidates.parquet", index=False)
    return recs, taxa, cand


class BioClip:
    def __init__(self):
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(BIOCLIP_ID)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(BIOCLIP_ID)

    def text(self, prompts: list[str]) -> np.ndarray:
        with self.torch.inference_mode():
            out = []
            for i in range(0, len(prompts), 256):
                t = self.model.encode_text(self.tokenizer(prompts[i:i + 256]).to(self.device))
                out.append((t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        return np.concatenate(out)

    def image(self, crops: list[Image.Image]) -> np.ndarray:
        with self.torch.inference_mode():
            out = []
            for i in range(0, len(crops), 64):
                x = self.torch.stack([self.preprocess(c) for c in crops[i:i + 64]]).to(self.device)
                f = self.model.encode_image(x)
                out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        return np.concatenate(out)

    @property
    def scale(self) -> float:
        return float(self.model.logit_scale.exp())


def specimen_crop(image: Image.Image, r, pad: float = 0.08) -> Image.Image:
    """The specimen on a white square, everything outside its mask blanked out."""
    m = measure.rle_decode(r.mask_rle, int(r.mask_h), int(r.mask_w))
    m = measure.largest_component(m)
    ys, xs = np.nonzero(m)
    win = np.asarray(image.crop((int(r.win_x1), int(r.win_y1), int(r.win_x1) + m.shape[1], int(r.win_y1) + m.shape[0])))
    sub = win[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
    sub[~m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]] = 255
    side = int(max(sub.shape[:2]) * (1 + 2 * pad))
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(Image.fromarray(sub), ((side - sub.shape[1]) // 2, (side - sub.shape[0]) // 2))
    return canvas


def classify(recs: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """BioCLIP scores of every segmented herp against its record's candidate species."""
    inst = pd.read_parquet(config.TABLES / "instances.parquet",
                           columns=["instance_id", "image_id", "mask_rle", "mask_h", "mask_w", "win_x1", "win_y1",
                                    "area_px", "largest_blob_frac"])
    man = pd.read_parquet(config.TABLES / "image_manifest.parquet", columns=["image_id", "source", "local_path", "biorepo_occid"])
    man = man[man["source"] == "herp"]
    inst = inst.merge(man, on="image_id")
    inst = inst[inst["largest_blob_frac"].fillna(1) > 0.8]  # skip speckled masks
    clip = BioClip()
    prompts = cand.drop_duplicates("speciesKey").set_index("speciesKey")["prompt"]
    text = pd.DataFrame(clip.text(prompts.tolist()), index=prompts.index)
    feats, ids = [], []
    for image_id, g in inst.groupby("image_id"):
        image = Image.open(config.ROOT / g["local_path"].iat[0]).convert("RGB")
        feats.append(clip.image([specimen_crop(image, r) for r in g.itertuples()]))
        ids += g["instance_id"].tolist()
    f = pd.DataFrame(np.concatenate(feats), index=ids)
    by_pair = {k: g for k, g in cand.groupby(["siteID", "higher_taxon"])}
    occ_of = dict(zip(inst["instance_id"], inst["biorepo_occid"].astype(str)))
    rec_by_occ = recs.assign(occid=recs["occid"].astype(str)).set_index("occid")
    rows = []
    for occid, insts in pd.Series(ids).groupby(pd.Series(ids).map(occ_of)):
        if occid not in rec_by_occ.index:
            continue
        rec = rec_by_occ.loc[occid]
        c = by_pair.get((rec["siteID"], rec["higher_taxon"]))
        row = dict(occid=occid, n_specimens_scored=len(insts), n_candidates=0 if c is None else len(c))
        if c is not None and len(c):
            c = c.sort_values("n_observations", ascending=False)
            row.update(range_pick=c["species"].iat[0], range_pick_key=int(c["speciesKey"].iat[0]),
                       range_pick_share=float(c["n_observations"].iat[0] / c["n_observations"].sum()),
                       candidates="|".join(c["species"].astype(str)))
            # softmax over the candidates for each specimen, then the mean over the record's specimens
            logits = clip.scale * f.loc[insts].to_numpy() @ text.loc[c["speciesKey"]].to_numpy().T
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = (p / p.sum(axis=1, keepdims=True)).mean(axis=0)
            k = int(p.argmax())
            row.update(bioclip_pick=c["species"].iat[k], bioclip_pick_key=int(c["speciesKey"].iat[k]), bioclip_prob=float(p[k]))
        rows.append(row)
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    recs, taxa, cand = build_candidates()
    scored = classify(recs, cand)
    recs = recs.assign(occid=recs["occid"].astype(str))
    out = recs[["occid", "collection", "catalogNumber", "siteID", "eventDate", "class", "order", "family", "scientificName",
                "taxonRank", "identificationQualifier", "identifiedBy", "individualCount", "n_images", "to_species",
                "confident", "higher_taxon", "speciesKey"]].merge(scored, on="occid", how="left")
    out = out.rename(columns={"scientificName": "recorded_name", "speciesKey": "recorded_speciesKey"})
    for method in ("range", "bioclip"):
        agree = out[f"{method}_pick_key"] == out["recorded_speciesKey"]
        out[f"{method}_agrees"] = agree.where(out["to_species"] & out[f"{method}_pick_key"].notna())
    # what is offered for records not identified to species
    need = ~out["to_species"].astype(bool) & out["n_candidates"].fillna(0).gt(0)
    single = need & (out["n_candidates"] == 1)
    strong = need & (out["n_candidates"] > 1) & (out["bioclip_prob"] >= MIN_PROB)
    out["suggested_species"] = np.where(single, out["range_pick"], np.where(strong, out["bioclip_pick"], None))
    out["suggestion_basis"] = np.where(single, "only species of this taxon observed nearby (GBIF)",
                                       np.where(strong, "BioCLIP 2 among species observed nearby (GBIF)", None))
    out.to_parquet(config.TABLES / "herp_id_suggestions.parquet", index=False)

    # how good are the picks where the species is known and there was a real choice to make
    known = out[out["confident"] & out["n_candidates"].fillna(0).gt(0) & out["bioclip_pick"].notna()]
    rows = []
    for label, sub in (("all species-level records", known), ("two or more candidates", known[known["n_candidates"] > 1]),
                       (f"two or more candidates, BioCLIP probability >= {MIN_PROB}",
                        known[(known["n_candidates"] > 1) & (known["bioclip_prob"] >= MIN_PROB)])):
        if len(sub):
            in_list = sub.apply(lambda r: str(r["recorded_name"]).split()[:2] == str(r["recorded_name"]).split()[:2]
                                and str(int(r["recorded_speciesKey"])) in {str(k) for k in cand[(cand["siteID"] == r["siteID"]) & (cand["higher_taxon"] == r["higher_taxon"])]["speciesKey"]}
                                if pd.notna(r["recorded_speciesKey"]) else False, axis=1)
            rows.append(dict(subset=label, records=len(sub), median_candidates=float(sub["n_candidates"].median()),
                             recorded_species_among_candidates=float(in_list.mean()),
                             range_pick_accuracy=float(sub["range_agrees"].mean()),
                             bioclip_pick_accuracy=float(sub["bioclip_agrees"].mean())))
    val = pd.DataFrame(rows)
    val.to_parquet(config.TABLES / "herp_id_validation.parquet", index=False)
    return out


def main() -> None:
    out = build()
    val = pd.read_parquet(config.TABLES / "herp_id_validation.parquet")
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(val.round(3).to_string(index=False))
    need = out[~out["to_species"].astype(bool)]
    print(f"records not identified to species: {len(need)}; with a suggestion: {int(need['suggested_species'].notna().sum())}")
    print(need["suggestion_basis"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
