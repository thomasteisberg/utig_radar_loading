# ORCA Radar Ingest Workflow

This document describes the workflow for ingesting [ORCA](https://orca.radioglaciology.com/) radar data into OPR using a set of Python CLI scripts. This document and the process it describes are based on the ingest process for UTIG radar described in `UTIG_Ingest_Workflow.md`. 

All commands below are run from the repo root.

## Prerequisites

- **Python**: 3.12+ with `uv` for dependency management
- **MATLAB**: Required for radar processing stages (after Python preprocessing)
- **Data access**: Raw ORCA recordings at the path set as `orca_raw_data_base_path` in `user_config.yaml` (e.g. `/cresis/data2/MIT/`). ORCA has no separate post-processed GPS source — GPS comes from the `*_gpspipe_stdout.log` sibling of each recording.
- **Configuration**: A `user_config.yaml` in the repo root with your local paths (see below)
- **Season config**: A YAML file in `orca/seasons_config/` for the season being processed

## Configuration

### `user_config.yaml`

This file exists in the repo root.
It contains user/environment-specific paths that are not season-specific and is shared across radar-system pipelines. Edit it to change paths as needed:

```yaml
# Base path for raw ORCA recordings (flat folders of YYYYMMDD_HHMMSS-prefixed
# files: *_rx_samps.bin / *_config.yaml / *_uhd_stdout.log / *_gpspipe_stdout.log)
orca_raw_data_base_path: "/cresis/data2/MIT/"

# File index cache (speeds up repeated runs)
orca_file_index_cache: "outputs/orca_file_index.csv"

# Base dir for ORCA GPS .mat files. MUST be absolute and writable by you:
# define_segments writes records.csv gps.fn as <orca_gps_base_dir>/<season>/
# gps_<seg>.mat, and OPR loads absolute paths directly.
orca_gps_base_dir: "/kucresis/scratch/<you>_sta/opr_user_tmp/gps"

# Base directories for outputs (season name is appended automatically)
header_base_dir: "/cresis/dataproducts/opr_data/opr_tmp/headers/rds"
params_output_base_dir: "outputs/params"
maps_output_base_dir: "outputs/maps"

# Dask parallelization settings
dask_workers: 10
```

This file is shared across the UTIG and ORCA pipelines, so it will also contain UTIG-specific keys (`raw_data_base_path`, `file_index_cache`, `gps_support_base_dir`); only the ORCA-relevant ones are shown above.

### Season config

Each ORCA season has a YAML file in `orca/seasons_config/`. Required fields at the top level:

```yaml
season_name: "2025_Antarctica_ORCA"
datasets: ["ORCA1"]

params:
  # ... default parameter values for the spreadsheet tabs
```

## Stage 1: Define Segments

Indexes raw ORCA recordings, assigns segments, checks each recording's GPS
coverage against OPR's downstream requirements, and outputs CSV files for the
parameter spreadsheet.

```bash
uv run orca/scripts/define_segments.py orca/seasons_config/<season>.yaml
```

**What it does:**
1. Indexes raw ORCA recordings under `orca_raw_data_base_path` (matching
   `*_rx_samps.bin` files and their `_config.yaml` / `_uhd_stdout.log` /
   `_gpspipe_stdout.log` siblings)
2. Assigns segment numbers based on time gaps between recordings (one recording =
   one segment)
3. Checks each recording's GPS coverage against the checks that OPR's
   `records_create_sync_gps` and SAR's `sar_coord_task` perform downstream
   (enough GPS fixes, non-stationary path, GPS window bracketing the radar
   window), flagging failures with a "do not process" reason up front instead
   of letting the MATLAB run halt later
4. Reads per-recording radar parameters (`fs`, `prf`, `f0`, `f1`, `Tpd`,
   `presums`, `DDC_freq`) from each recording's `_config.yaml`, since these
   can vary between recordings for ORCA
5. Generates one CSV per spreadsheet tab in `params_output_base_dir/season_name/`

**Files generated:**

CSVs under `<params_output_base_dir>/<season_name>/`:
- `cmd.csv`
- `records.csv`
- `radar.csv` (per-segment, populated from each recording's `_config.yaml`)
- `qlook.csv`, `sar.csv`, `array.csv`, `post.csv`, `analysis_noise.csv` (only
  if defaults are provided in the season config)

Map under `<maps_output_base_dir>/`:
- `<season_name>.html` — interactive map of segments, built from the gpspipe
  GPS logs

Also updated/created:
- `<orca_file_index_cache>` (e.g. `outputs/orca_file_index.csv`) — cached
  raw-data file index, reused on subsequent runs

**After running:**
1. Review the segment report, especially any "do not process" reasons
   (insufficient GPS fixes, stationary recording, GPS not bracketing the
   radar window, unreadable header)
2. Review the CSV outputs, editing them directly if needed (adjust default
   parameter values, mark additional segments "do not process" in
   `cmd.csv`'s `notes` column, etc.)

## Stage 2: Create GPS Support Files

Reads the CSV parameter directory (or an xlsx spreadsheet) and generates GPS
support `.mat` files from the gpspipe logs.

```bash
uv run orca/scripts/create_gps_support.py outputs/params/<season_name> --overwrite
```

**What it does:**
1. Reads `<params_output_base_dir>/<season_name>/` (accepts either the CSV
   directory Stage 1 writes, or an xlsx spreadsheet path)
2. Skips segments marked "do not process" in `cmd.csv`'s `notes` column
3. For each remaining segment, parses the segment's `gps.field_fn` gpspipe
   log(s) and generates a GPS support file

**Files generated** (under `<orca_gps_base_dir>/<season_name>/`, per the
`gps.fn` column of `records.csv`):
- One `gps_<YYYYMMDD>_<NN>.mat` per processable segment

**After running:** Review the output for any errors or skipped segments (no `gps.fn`, no
   `gps.field_fn` gpspipe paths)

## Stage 3: Create Temporary Headers

Reads the CSV parameter directory (or an xlsx spreadsheet) and generates
temporary header `.mat` files for MATLAB `records_create`.

```bash
uv run orca/scripts/create_headers.py outputs/params/<season_name> --overwrite
```

**What it does:**
1. Reads `<params_output_base_dir>/<season_name>/`, same input as Stage 2
2. Skips segments marked "do not process"
3. Locates each segment's `_rx_samps.bin` from `file.base_dir` + `file.prefix`

**Files generated** (under `<header_base_dir>/<season_name>/`):
- One `<rx_samps_stem>.mat` per raw recording (ORCA is flat — no per-board
  subdirectory nesting the way UTIG's `bxds.mat` layout has)

**After running:** Review the output for errors or missing radar files

## Stage 4: Assemble Parameter Spreadsheet

Assembles the CSV directory into an OPR-format xlsx spreadsheet — this is
the one step in the ORCA workflow that produces the artifact MATLAB actually
reads (OPR's `read_param_xls_generic.m` expects xlsx, not raw CSVs).

```bash
uv run scripts/csvs_to_xlsx.py outputs/params/<season_name> \
    --output outputs/params/<season_name>/rds_param_<season_name>.xlsx
```

TODO add a note about setting the output path of the xlsx file

**What it does:**
1. Reads every `*.csv` in the given directory
2. Writes one sheet per CSV, adding the header/type-code rows
   `read_param_xls_generic.m` requires (a 5-row header block for `cmd`, a
   2-row block for every other sheet)

**Files generated:**
- `<csv_dir>/<season_name>.xlsx` by default, or wherever `--output` points

**After running:**
1. Copy the xlsx to wherever your MATLAB run scripts read parameter
   spreadsheets from, e.g.:
   ```bash
   cp outputs/params/<season_name>/rds_param_<season_name>.xlsx \
      /cresis/users/<you>_sta/scripts/opr_params/
   ```

## Full command sequence

Putting Stages 1–4 together, a typical run looks like:

```bash
rm outputs/orca_file_index.csv   # force reindex if raw data changed

uv run orca/scripts/define_segments.py orca/seasons_config/<season>.yaml
uv run orca/scripts/create_gps_support.py outputs/params/<season_name> --overwrite
uv run orca/scripts/create_headers.py outputs/params/<season_name> --overwrite
uv run scripts/csvs_to_xlsx.py outputs/params/<season_name> \
    --output outputs/params/<season_name>/rds_param_<season_name>.xlsx

cp outputs/params/<season_name>/rds_param_<season_name>.xlsx \
   /cresis/users/<you>_sta/scripts/opr_params/
```

## MATLAB Radar Processing

From here, the the MATLAB-based radar processing steps are the same as for the UTIG radar. Refer to [docs/UTIG_Ingest_Workflow.md#matlab-radar-processing](https://github.com/englacial/utig_radar_loading/blob/main/docs/UTIG_Ingest_Workflow.md#matlab-radar-processing). The rough outline of steps is:

- Update the lever_arm.m file (`matlab/processing/lever_arm.m` in [openpolarradar/opr](https://gitlab.com/openpolarradar/opr/)) with information about your radar's GPS, IMU, and antenna phase center locations (you could use all zeros for testing, or if the positions are unknown but small).

- Come up with `Tsys` and/or `Tadc_adjust` values for your radar. These parameter are delays that account for constant signal delays in your radar. On ORCA radars, these would include the amount of coax cable between your SDR and your antennas, plus any fixed delay in your SDR's analog-to-digital converter. The [field processing notebook](https://github.com/HI-SNR-Lab/uhd_radar/blob/3026b9c715de30b96bcaf5a38d1587877ccf1226/postprocessing/notebooks/Field%20Processing.ipynb) (as of August 2026) includes some of these delays (in units of samples):
  ```
  #zero_sample_idx = 36 # X310, fs = 20 MHz
  #zero_sample_idx = 63 # X310, fs = 50 MHz
  zero_sample_idx = 159 # B205mini, fs = 56 MHz
  #zero_sample_idx = 166 # B205mini, fs = 20 MHz
  ```

- Add a section for you survey to the UTIG/ORCA-specific run files. Those files are at `/kucresis/scratch/[USERNAME_HERE]/scripts/run_opr/rds/UTIG`. Each run script mentioned below has a `year = XXXX` and a series of `if-else` statements for each year. You'll need to add a new elseif to each file. 

- Then, starting running through the matlab scripts. The general order is `run_records_create_UTIG.m` -> `run_analysis_UTIG.m` -> `run_collate_coh_noise_UTIG.m` -> `run_qlook_UTIG.m` -> `run_all_create_track_files_UTIG.m` -> `run_layer_tracker_UTIG.m`, then take a look at the results so far with `imb.picker`, then continue on with `run_sar_UTIG.m` -> `run_array_UTIG.m` -> `run_post_UTIG.m` to generate output files.

Refer to [`docs/UTIG_Ingest_Workflow.md`](https://github.com/englacial/utig_radar_loading/blob/main/docs/UTIG_Ingest_Workflow.md) for more details!
