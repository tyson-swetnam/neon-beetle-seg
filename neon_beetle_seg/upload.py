"""Publish the build to the CyVerse Data Store with gocmd.

1. Archive the v0.1.0 pilot: its ducklake/, report/, stacked/ and metadata/ collections are
   moved (server side, nothing is deleted) into archive/v0.1.0-pilot/. Skipped once that
   archive exists.
2. Sync the v0.2 build next to it:
     ducklake/   beetles.ducklake + data/ + export/*.parquet + beetles.duckdb
     report/     report.html and the summary CSVs
     metadata/   ontology tables used for the column tags
     docs/       README and docs from the repository at the commit that built the lake

Source images, model weights and caches are not uploaded: every image is recorded in
image_manifest with its URL and sha256 and can be fetched again.

gocmd reads its login from ~/.irods (already configured on CyVerse VICE apps). File transfer goes
through gocmd rather than the /data-store FUSE mount, which costs 1-2 s per file open.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess

from . import config

PILOT_COLLECTIONS = ["ducklake", "report", "stacked", "metadata"]
ARCHIVE = "archive/v0.1.0-pilot"


def gocmd(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gocmd", *args], capture_output=True, text=True, check=check)


def exists(path: str) -> bool:
    return gocmd("ls", path, check=False).returncode == 0


def archive_pilot(root: str, dry_run: bool) -> None:
    dest = f"{root}/{ARCHIVE}"
    if exists(dest):
        print(f"pilot already archived at {dest}")
        return
    present = [c for c in PILOT_COLLECTIONS if exists(f"{root}/{c}")]
    # the pilot is recognised by its stacked/ collection, which v0.2 does not produce
    if "stacked" not in present:
        print("no pilot layout found at the top level; nothing to archive")
        return
    print(f"archiving pilot collections {present} -> {dest}")
    if dry_run:
        return
    gocmd("mkdir", "-p", dest)
    for c in present:
        gocmd("mv", f"{root}/{c}", f"{dest}/{c}")


def stage() -> dict:
    """Assemble what gets uploaded under outputs/publish/ and return {collection: local path}."""
    pub = config.OUT / "publish"
    if pub.exists():
        shutil.rmtree(pub)
    out = {}
    out["ducklake"] = config.LAKE_DIR
    rep = pub / "report"
    rep.mkdir(parents=True)
    for f in (config.OUT / "report").glob("*"):
        if f.is_file():
            shutil.copy2(f, rep / f.name)
    out["report"] = rep
    out["metadata"] = config.METADATA
    docs = pub / "docs"
    docs.mkdir()
    shutil.copy2(config.ROOT / "README.md", docs / "README.md")
    for f in (config.ROOT / "docs").glob("*.md"):
        shutil.copy2(f, docs / f.name)
    out["docs"] = docs
    return out


def sync(root: str, dry_run: bool) -> None:
    for name, local in stage().items():
        n = sum(1 for p in local.rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
        print(f"sync {name}/  {n} files, {size / 1e6:.1f} MB -> {root}/{name}")
        if dry_run:
            continue
        gocmd("mkdir", "-p", f"{root}/{name}")
        r = gocmd("sync", "--no_root", "--delete", str(local), f"i:{root}/{name}", check=False)
        if r.returncode != 0:
            raise SystemExit(f"gocmd sync failed for {name}:\n{r.stderr[-2000:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=config.IRODS_ROOT, help="iRODS collection (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen")
    args = ap.parse_args()
    archive_pilot(args.root, args.dry_run)
    sync(args.root, args.dry_run)
    print("done" if not args.dry_run else "dry run only; nothing changed")


if __name__ == "__main__":
    main()
