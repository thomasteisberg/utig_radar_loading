# utig_radar_loading
(Name to be updated -- not just for UTIG data anymore!)

This repository contains an alternative set of Python tools to do initial ingest of
radar soudner data into [Open Polar Radar](https://openpolarradar.org)
(OPR). It handles indexing raw data, defining segments, building parameter spreadsheets,
and generating GPS support and temporary header files — plus assorted utilities for
inspecting OPR products.

Two families of radar systems are currently supported:

- Texas radars: **HiCARS 1/2 + MARFA** — pre-2020 ICECAP data from HiCARS 1/2 and MARFA (file types: `RADjh1` / `RADnh3` / `RADnh5`)
- **Open Radar Code Architecture (ORCA)** — Open-source software-defined radio-based radar sounders (see the [project website](https://orca.hisnr.com/) and [GitHub repository](https://github.com/HI-SNR-Lab/uhd_radar))

## Documentation

Start with the workflow docs in [`docs/`](docs/):

- [`docs/UTIG_Ingest_Workflow.md`](docs/UTIG_Ingest_Workflow.md) — end-to-end UTIG ingest
- `docs/ORCA_Ingest_Workflow.md` — end-to-end ORCA ingest

## Layout

```
src/opr_ingest/     Installable package
  core/             Shared OPR I/O: params, layers, GPS .mat, v7.3 HDF5, geo/maps
  utig/             UTIG raw stream parsing, GPS pipeline, headers, segmenting
  orca/             ORCA equivalents
utig/, orca/        Per-system CLI scripts and per-season YAML configs
scripts/            Standalone analysis/QC tools (check_surface, channel equalization, ...)
notebooks/          Exploratory notebooks
```

## Setup

Uses [uv](https://docs.astral.sh/uv/).

This package is intended to be run on the CReSIS servers.

You should first update your `user_config.yaml` in the repository root to match the
appropriate data paths. Replace `[USERNAME_HERE]` with your CReSIS username.

Once you have your config setup, you can run scripts from the repo root:

```bash
uv run utig/scripts/define_segments.py utig/seasons_config/2015_Antarctica_BaslerJKB.yaml
```

(But you should really read the relevant `docs/` pages first!)