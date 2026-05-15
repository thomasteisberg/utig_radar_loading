# UTIG Radar Ingest Workflow

This document describes the workflow for ingesting pre-2020 UTIG ICECAP radar data into OPR using a set of Python CLI scripts.

All commands below are run from the repo root.

## Prerequisites

- **Python**: 3.12+ with `uv` for dependency management
- **MATLAB**: Required for radar processing stages (after Python preprocessing)
- **Data access**: Raw UTIG field data at `/kucresis/scratch/data/UTIG/`, post-processed GPS data at `/resfs/GROUPS/CRESIS/dataproducts/metadata/`
- **Configuration**: A `user_config.yaml` in the repo root with your local paths (see below)
- **Season config**: A YAML file in `utig/seasons_config/` for the season being processed

## Configuration

### `user_config.yaml`

Create this file in the repo root. It contains user/environment-specific paths that are not season-specific and is shared across radar-system pipelines:

```yaml
raw_data_base_path: "/kucresis/scratch/data/UTIG"
file_index_cache: "outputs/file_index.csv"
gps_support_base_dir: "/resfs/GROUPS/CRESIS/dataproducts/opr_support/gps"
header_base_dir: "/kucresis/scratch/tteisberg_sta/scripts/opr_user_tmp/headers/rds"
params_output_base_dir: "outputs/params"
maps_output_base_dir: "outputs/maps"
dask_workers: 10
```

### Season config

Each season has a YAML file in `utig/seasons_config/`. Required fields at the top level:

```yaml
season_name: "2015_Antarctica_BaslerJKB"
postprocessed_gps_dir: "/resfs/GROUPS/CRESIS/dataproducts/metadata/2015_Antarctica_BaslerJKB/gps"
datasets: ["UTIG1"]

params:
  # ... default parameter values for the spreadsheet tabs
```

## Stage 1: Define Segments

Indexes raw data, matches it with post-processed GPS files, assigns segments, and outputs CSV files for the parameter spreadsheet.

```bash
uv run utig/scripts/define_segments.py utig/seasons_config/2015_Antarctica_BaslerJKB.yaml
```

**What it does:**
1. Enumerates post-processed GPS files and finds matching radar transects
2. Assigns segments based on time gaps between transects
3. Generates one CSV per spreadsheet tab in `params_output_base_dir/season_name/`

**Files generated:**

CSVs under `<params_output_base_dir>/<season_name>/`:
- `cmd.csv`
- `records.csv`
- `qlook.csv`
- `sar.csv`
- `array.csv`
- `radar.csv`
- `post.csv`
- `analysis_noise.csv`

Map under `<maps_output_base_dir>/`:
- `<season_name>.html` — interactive map of matched segments and missing-radar transects

Also updated/created:
- `<file_index_cache>` (e.g. `outputs/file_index.csv`) — cached raw-data file index, reused on subsequent runs

**After running:**
1. Review the match report for any unmatched GPS files or missing radar data
2. Review the CSV outputs
3. Copy the CSVs into an xlsx parameter spreadsheet template (use an existing season's spreadsheet as a starting point)
4. Edit the spreadsheet as needed:
   - Mark segments as "do not process" in the `cmd.notes` column
   - Adjust any default parameter values
   - Verify `gps.fn` paths point to the correct GPS support file locations

## Stage 2: Create GPS Support Files

Reads the xlsx parameter spreadsheet and generates GPS support `.mat` files.

```bash
uv run utig/scripts/create_gps_support.py path/to/season_params.xlsx [--overwrite]
```

**What it does:**
1. Reads the xlsx spreadsheet to get segment definitions
2. Skips segments marked "do not process"
3. For each segment, generates a GPS support file from field and/or post-processed GPS data

**Files generated** (under `<gps_support_base_dir>/<season_name>/`):
- One `gps_<YYYYMMDD>_<NN>.mat` per processable segment (path taken from the `gps.fn` column of the `records` sheet)

**After running:**
1. Review the output for any errors
2. Copy GPS support files to permanent storage:
   ```
   cp -r <gps_support_base_dir>/<season_name> /resfs/GROUPS/CRESIS/dataproducts/opr_support/gps/
   ```

## Stage 3: Create Temporary Headers

Reads the xlsx parameter spreadsheet and generates temporary header `.mat` files for MATLAB `records_create`.

```bash
uv run utig/scripts/create_headers.py path/to/season_params.xlsx [--overwrite]
```

**What it does:**
1. Reads the xlsx spreadsheet to get radar file paths
2. Skips segments marked "do not process"
3. Uses Dask for parallel header generation
4. Expands RADjh1 entries to process both bxds1/bxds2 channels

**Files generated** (under `<header_base_dir>/<season_name>/`):
- One `<bxds_filename>.mat` per raw radar file, mirroring the last four path components of the source file (e.g. `UTIG1/orig/xlob/<prj>/<set>/<trn>/<RADxxx>/bxds.mat`)
- For RADjh1 segments, both `bxds1.mat` and `bxds2.mat` are produced per transect

**After running:**
1. Review the output for errors
2. Copy header files to the default OPR location:
   ```
   cp -r <header_base_dir>/<season_name> /cresis/dataproducts/opr_data/opr_tmp/headers/rds/
   ```

## MATLAB Radar Processing

After the Python preprocessing stages, continue with MATLAB-based radar processing.

## Update lever_arm.m

Update `lever_arm.m` with the GPS, IMU, and radar locations.

## Run the processing

There are UTIG-specific run scripts in `/kucresis/scratch/tteisberg_sta/scripts/run_opr/rds/UTIG`

Each run script mentioned below has a `year = XXXX` and a series of if-else statements for each year. You'll need to add a new `elseif` to each file, such as the following:

```
elseif year == 2016
  params = read_param_xls(opr_filename_param('rds_param_2016_Antarctica_BaslerJKB.xlsx'));
  params = opr_set_params(params,'cmd.records',1);
  params = opr_set_params(params,'cmd.records',0,'cmd.notes','do not process');
```

In most cases, the recommendation is to set the script to run everything everything not marked with a "do not process" note, but you can set individual segments as well.

The recommended workflow is as follows:

1. `run_records_create_UTIG.m`
2. `run_analysis_UTIG.m`
3. `run_collate_coh_noise_UTIG.m`
4. `run_qlook_UTIG.m`
5. `run_all_create_track_files.m`
6. `run_layer_tracker_UTIG.m`

For the layer tracker, run it using snake processing on the surface. Optionally (but highly recommended), run it a second time to copy the surface DEM for reference.

At this stage, you should manually inspect some data using `imb.picker` and make sure it's looking reasonable.

6. `run_check_surface.m`

Run check surface on a handful of segments. If needed, update Tadc_adjust following the instructions in `check_surface.m`. Get this part right before proceeding.

Manually review all of the surface picks and fix any you find that are incorrect. After fixing surface picks, you'll need to re-combine the images:

6. `run_img_combine_update_UTIG.m`

You're now ready for SAR processing and generating the CSARP_standard product:

7. `run_sar_UTIG.m`
8. `run_array_UTIG.m`

Manually review data again to check that everything so far has worked.

9. `run_post_UTIG.m`

Download the `CSARP_post` directory and review the PDFs.

Then copy `CSARP_qlook` and `CSARP_standard` into `CSARP_post`. Move `CSARP_post` to the public data directory and symlink it from the working directory.

That's it!
