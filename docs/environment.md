# Environment

This project was built on a CyVerse VICE GPU workspace. The notes below are what you need to know
to run it there again, or to move it elsewhere.

## The VM

| | |
|---|---|
| OS | Ubuntu 24.04 container (Kubernetes pod), passwordless `sudo` |
| GPU | one NVIDIA A16 slice, 15,356 MiB (about 14.9 GiB usable by PyTorch), compute capability 8.6 |
| Driver | 610.57.04; no CUDA toolkit (`nvcc`) and no system cuDNN |
| CPU / RAM | cgroup-limited to **8 cores and 32 GiB**, although `nproc` and `free` report the host's 128 cores and 503 GiB |
| `/dev/shm` | 2 GB |
| Disk | several TB free on the container overlay |

Two consequences:

- **Nothing compiled against CUDA.** Every model is loaded through `transformers` (or Ultralytics
  for the optional YOLO detector), which need only the PyTorch wheels.
- **Set thread counts yourself.** Libraries see 128 CPUs; `config.limit_threads()` caps them at 8.

## What persists

The container's disk, including the home directory, is **lost when the analysis ends**. Only two
places persist:

| Place | What goes there |
|---|---|
| GitHub (`tyson-swetnam/neon-beetle-seg`) | code, `uv.lock`, docs, ontology metadata |
| CyVerse Data Store (`/iplant/home/tswetnam/neon-beetle-seg`) | the lake, tables, report |

Downloaded images, HuggingFace datasets, model weights and the virtual environment are all
re-creatable and are not stored anywhere permanent. `image_manifest` records the URL and sha256 of
every image.

The Data Store is mounted at `/data-store` through FUSE, but each file open costs 1 to 2 seconds
there. Work on local disk and transfer with `gocmd` (`nbs upload` does this). Do not run `git`
on the mount.

## Setup on a fresh VM

```bash
git clone https://github.com/tyson-swetnam/neon-beetle-seg.git
cd neon-beetle-seg
scripts/bootstrap.sh
```

`bootstrap.sh` installs `uv` if needed, creates `.venv` from `uv.lock` (Python 3.12, PyTorch from
the CUDA 12.6 wheel index), installs the DuckDB `ducklake` extension and prints a GPU check.

On this VM image `git push` hangs, because the global credential helper is an interactive
credential manager. `bootstrap.sh` points this repository at the `gh` login instead, so run
`gh auth login` first.

## Secrets

Credentials are read from `~/.neon-beetle-secrets.env` (mode 600), never from the repository:

```bash
NEON_TOKEN=...    # required for `nbs neon`; the NEON data endpoint returns 403 without a token
GBIF_USER=...     # optional; enables a citable GBIF download (otherwise the anonymous search API is used)
GBIF_PWD=...
GBIF_EMAIL=...
HF_TOKEN=...      # optional; required only for `nbs sam3` (facebook/sam3 is a gated model)
```

Environment variables of the same names take precedence over the file.

## Packages

Versions are pinned in `uv.lock`. The ones that matter:

| Package | Version | Note |
|---|---|---|
| torch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 | cu126 is the build verified on this driver |
| transformers | 5.18.0 | Grounding DINO, SAM 2.1, SAM 3, Mask2Former, TrOCR |
| duckdb | 1.5.5 | same version as the CyVerse MESA tooling; `ducklake` extension |
| ultralytics | 8.4.173 | optional extra `yolo`; AGPL-3.0 |
| neonutilities | 2.0.2 | official NEON download and stacking package |

## GPU memory and speed measured here

Peak GPU memory for the full pipeline (Grounding DINO base + SAM 2.1 large + BeetleFlow loaded
together) was about 5.5 GiB. Per-call times on the A16 with bf16 autocast:

| Step | Time |
|---|---|
| Grounding DINO base, one photo | 0.59 s (1.04 s in fp32) |
| SAM 2.1 large, one box | 0.34 s |
| SAM 2.1 base-plus, one box | 0.16 s |
| SAM 2.1 small, one box | 0.10 s |
| BeetleFlow part labels, one crop | 0.17 s (0.35 s in fp32) |

Batching did not help: the GPU is the bottleneck, not the Python loop.
