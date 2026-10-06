"""`nbs` command: one sub-command per pipeline step (see docs/pipeline.md for the order)."""
from __future__ import annotations

import importlib
import sys

STEPS = {
    "neon": ("stack_neon", "download and stack NEON DP1.10022.001 for all sites (needs NEON_TOKEN)"),
    "biorepo": ("fetch_biorepo", "Biorepository records and image manifest from Darwin Core Archives"),
    "gbif": ("fetch_gbif", "GBIF records for the same collections"),
    "hf": ("fetch_hf", "download the Imageomics datasets from HuggingFace"),
    "images": ("fetch_images", "download the Biorepository images"),
    "manifest": ("manifest", "build the unified image manifest and annotation tables"),
    "segment": ("segment", "detect, segment and measure one image pool on the GPU"),
    "scale": ("scale", "read pixel-per-mm scales from rulers and printed bars"),
    "tables": ("tables", "collect segmentation shards into instances / measurements"),
    "validate": ("validate", "compare with human annotations"),
    "herps": ("herp_id", "GBIF candidate species and BioCLIP suggestions for herptile bycatch"),
    "sam3": ("sam3_compare", "SAM 3 text-prompted comparison on validation trays (gated model)"),
    "lake": ("build_ducklake", "build the DuckLake lakehouse and flat DuckDB file"),
    "report": ("report", "write the HTML report"),
    "upload": ("upload", "archive the pilot and sync results to the CyVerse Data Store"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in STEPS:
        print("usage: nbs <step> [options]\n")
        for name, (_, desc) in STEPS.items():
            print(f"  {name:10s} {desc}")
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    module, _ = STEPS[sys.argv[1]]
    sys.argv = [f"nbs {sys.argv[1]}"] + sys.argv[2:]
    importlib.import_module(f"neon_beetle_seg.{module}").main()


if __name__ == "__main__":
    main()
