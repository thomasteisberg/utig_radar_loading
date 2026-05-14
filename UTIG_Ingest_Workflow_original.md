This page documents the process for ingesting pre-2020 UTIG data from MARFA (and hopefully eventually HiCARS 1 + 2) into OPR.

This is a somewhat modified version of the workflow described in [Processing Steps](Processing-Steps) that relies on a Python library to generate temporary header files, GPS support files, and the parameter spreadsheet.

The first section below briefly explains some notable differences about how the UTIG data used to be collected. The rest of this page is setup to explain the workflow of adding a new season into OPR.

# Notes on UTIG data

This page documents the process for ingesting the UTIG "ICECAP" surveys that span from 2008 through 2018 (season year, so through 2019).

UTIG data prior to the COLDEX seasons was collected according to Project/Set/Transect triples. UTIG refers to this as "science-based" indexing. Originally, each P/S/T corresponded to a (generally straight) transect segment that was planned out as a flight path for data collection. Data was recorded only on these flight segments and the radar was turned off for any transit segments needed to get to those planned P/S/T segments, including turns during a gridded survey.

At some point, data started being recorded on the turns as well, but files were still separated at the start of each P/S/T. Data recorded during transits or on turns was generally put into a special project category of "ICPX" for ICECAP season X. For example, data from the 2018 season is divided into ICP10 projects on turns and transit with the non-ICP10 data being the straight-line science data collections:

![image](uploads/54e3588b508b1ee63d596fa4587dbc55/image.png){width=686 height=333}

In early surveys, the process of starting new data collections was manual, so there were long breaks between the end of one collection and the start of another. These gaps decreased over the seasons, however OPR generally only tolerates a few seconds of missing data before creating a new segment. As a result, all of the seasons prior to 2018 are divided into one OPR segment (i.e. `YYYYMMDD_XX`) per P/S/T from the original data. This, unfortunately, leads to a lot of very short segments with only one frame. There are, however, also much longer segments where continuous data collection occurred over hundreds of kilometers.

# Python-based processing steps

For now, the Python-based part of the process lives in a separate Git repository located here: https://github.com/englacial/utig_radar_loading

(The goal is to merge this into the main opr repository eventually.)

## Gathering post-processed GPS data

Duncan provided all of the UTIG field data that was available for release to `/kucresis/scratch/data/UTIG/UTIG1/` and `/kucresis/scratch/data/UTIG/UTIG2`. The distinction originally referred only to which hard drive was used to transfer the data. `UTIG2` contains the 2017 and 2018 seasons. `UTIG1` contains all prior seasons.

The missing piece of data are the post-processed positioning information. There is technically enough information to fully process everything without these, however it is strongly recommended to get the post-processed positioning information before processing the radar data.

Duncan has been working on getting this available on various DOIs and we are collected it, along with a README.txt in each folder identifying the source DOI, in directories as follows

`/resfs/GROUPS/CRESIS/dataproducts/metadata/SEASON_NAME/gps/`

## Running `define_season.ipynb`

The `define_season.ipynb` ([link](https://github.com/englacial/utig_radar_loading/blob/main/define_season.ipynb)) notebook is the primary pathway for processing data.

### Step 0: Indexing all files

This stage indexes all of the UTIG data files that can be found and creates a DataFrame of "artifacts" corresponding to each individual data file.

This artifacts DataFrame is then processed to group it by transect (P/S/T triples in UTIG terminology). Finally, some data is read from the context files to assign seasons to each artifact. Once seasons are defined, a list of possible season years is printed and the user can select a year to process for the following stages.

At the end of this stage, you should get an output like this:

```
The following seasons were found in the dataset:
[2008. 2009. 2010. 2011. 2012. 2014. 2015. 2016. 2017. 2018.   nan]
```

You can select any of these years as a season to process.

### Step 1: Select a single season to extract

From here on, the rest of this workflow operates on a single season at a time. The season is identified initially by a year, corresponding to the year at the start of the season.

If available, `season_gps_postprocessed_dir` should be set to the path of the post-processed GPS and IMU data. If it is not available, it can be set to None:

```
season_gps_postprocessed_dir = "/resfs/GROUPS/CRESIS/dataproducts/metadata/2015_Antarctica_BaslerJKB/gps" # Preferred if available
season_gps_postprocessed_dir = None # Data can be processed from field GPS and IMU data only if that's all that's available
```

The second cell will generate an output like this that should be manually reviewed:

```
[WARNING] Missing radar data for 48 transects out of 214
[WARNING] Missing post-processed GPS data for 46 transects
GPS stream types: ['GPSap3']
Radar stream types: ['RADnh3']
IMU stream types: [nan]
Sets: ['JKB2o' 'JKB2n']
Projects: ['ALG2' 'ASB1' 'ASUMA' 'BNZ' 'ICP7' 'MUI' 'NWZ' 'NWZ1' 'OIA' 'PEL' 'SCT'
 'TOT']
Aircraft identifier: JKB
Season name: 2015_Antarctica_BaslerJKB
```

Confirm that:
1. The automatically generated season name makes sense
2. The radar stream types match the expected stream format type(s) for this season
3. The projects list looks reasonable. The projects list for a season should include identifiers for any major surveys that were part of the season. It should also include ONLY ONE `ICPx` project.

Then review the warnings about missing radar data and missing post-processed GPS data. It may be helpful to run the rest of the cells in this section to generate a map that will show where these missing bits of data are. There may be good reasons for missing data (for example, a gravity survey done over sea ice might have not had the radar turned on at all). There might also be missing post-processed GPS information for transects that were not considered of scientific importance.

It's fine to go ahead with missing data, but there should be known reasons for why it's missing.

Run the rest of the cells in this step to generate a map of the data. For example:

![image](uploads/0e860f0a6ed0d39fa2dbd0b8cc1d7e45/image.png){width=480 height=600}

Segments are each shown as a thin line in a different color. Segments with missing radar data are shown as thick red lines. Segments with missing post-processed GPS data are shown as thick purple lines.

Iterate through the steps above until you're satisfied that all of the available data has been found.

### Step 2: Create GPS support files for each segment

The next cell will read the available GPS files (using post-processed GPS data if it is available) and write out OPR-formatted [GPS support files](https://gitlab.com/openpolarradar/opr/-/wikis/GPS-File-Guide).

This may take quite a while to run -- possibly an hour or more for a new season.

The output should be examined carefully for errors. It is not uncommon for a handful of segments to have issues that may need to be manually addressed or marked as "do not process" if the issue cannot be fixed.

GPS support files are saved to `output_base_dir`, but should be copied to permanent store on the OPR servers:

```
cp -r outputs/gps/SEASON_NAME /resfs/GROUPS/CRESIS/dataproducts/opr_support/gps/
```

### Step 3: Generate temporary header files

"Temporary" header files serve as an index into the raw radar files. These files store byte offsets into the raw radar files per sample that allow the OPR pipeline to read data out of the binary files without knowing their structure.

This cell relies on the UTIG unfoc library (https://github.com/UTIG/unfoc/), which is expected to be become public and open source shortly.

Dask is used to parallelize this task as it can take quite a long time (potentially hours, even in parallelized, for a new season).

The outputs are stored in the path set by `header_base_dir`:

```
header_base_dir = f"/kucresis/scratch/tteisberg_sta/scripts/opr_user_tmp/headers/rds/{season_name}/"
```

These files should be copied to the default `opr_tmp/headers/rds` directory after generation:

```
cp -r /kucresis/scratch/tteisberg_sta/scripts/opr_user_tmp/headers/rds/SEASON_NAME /cresis/dataproducts/opr_data/opr_tmp/headers/rds/
```

### Step 4: Create parameter spreadsheet starting templates

These cells will generate one CSV file per tab of a standard OPR parameter spreadsheet. These CSV files are intended to be copy and pasted into a template parameter spreadsheet. It's likely easiest to pick an existing parameter spreadsheet from a similar season as a starting point.

Before running this set of cells, create a `SEASON_NAME.yaml` file in `seasons_config/` with the defaults you want to populate. It is recommended to find a similar season and start from that as a template.

Note that these YAML files are not intended to be an autoritative reference and should not be re-used elsewhere. These are starting points for the parameter spreadsheet, but the actual parameter spreadsheet stored in the `opr_params` repository is always the authoritative copy.

After running these cells, copy the outputs from `outputs/params/SEASON_NAME` into a template paramter spreadsheet.

Check the `gps.fn` paths in the `records` tab and correct the paths (by find and replace) to the permanent location of the GPS support files.

# MATLAB-based radar processing

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
