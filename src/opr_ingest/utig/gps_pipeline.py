"""End-to-end GPS .mat generation pipeline for UTIG segments.

Orchestrates field-GPS, IMU, and (optional) post-processed GPS streams into the
OPR GPS .mat file format.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from opr_ingest.core.opr_gps_matlab import (
    create_gps_matlab_structure,
    merge_df,
    merge_position_files,
    save_gps_matlab_file,
)
from opr_ingest.utig import stream_util
from opr_ingest.utig.postprocessed_gps import load_and_parse_postprocessed_gps_file


def load_and_parse_gps_file(gps_path: Union[str, Path]) -> pd.DataFrame:
    """Load one UTIG field GPS file (e.g. GPSnc1/xds.gz) and parse to OPR columns."""
    gps_path = Path(gps_path)

    try:
        df = stream_util.load_xds_stream_file(gps_path, parse=True)
    except Exception as e:
        warnings.warn(f"Failed to load {gps_path}: {e}")
        return pd.DataFrame()

    required_cols = ['GPS_TIME', 'RADAR_TIME']
    if not all(col in df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {gps_path}. Required columns: {required_cols}, found columns: {df.columns.tolist()}")
        return pd.DataFrame()

    return df


def load_and_parse_imu_file(imu_path: Union[str, Path]) -> pd.DataFrame:
    """Load one UTIG IMU file (e.g. AVNnp1) and parse to OPR columns."""
    imu_path = Path(imu_path)

    stream_type = imu_path.parent.name

    if stream_type == 'AVNnp1':
        imu_df = stream_util.parse_binary_AVNnp1(imu_path)
    else:
        warnings.warn(f"Unknown IMU stream type '{stream_type}' in {imu_path}")
        return pd.DataFrame()

    required_cols = ['ROLL', 'PITCH', 'HEADING']
    if not all(col in imu_df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {imu_path}")
        return pd.DataFrame()

    if 'TIMESTAMP' not in imu_df.columns and 'GPS_TIME' not in imu_df.columns:
        warnings.warn(f"Missing TIMESTAMP or GPS_TIME in {imu_path}")
        return pd.DataFrame()

    imu_df['source_file'] = str(imu_path)
    return imu_df


def generate_gps_file(gps_paths: List[Union[str, Path]],
                     output_path: Union[str, Path],
                     output_path_temporary_df: Optional[Union[str, Path]] = None,
                     imu_paths: Optional[List[Union[str, Path]]] = None,
                     postprocessed_gps_paths: Optional[List[Union[str, Path]]] = None,
                     gps_pad_time_s: float = 2) -> None:
    """Build and save one OPR GPS .mat file from a set of UTIG field/IMU/post-processed inputs."""

    postproc_df = None
    field_imu_df = None

    if postprocessed_gps_paths and len(postprocessed_gps_paths) > 0:
        postproc_df = merge_position_files(
            file_paths=postprocessed_gps_paths,
            load_function=load_and_parse_postprocessed_gps_file,
            time_sort_key='GPS_TIME',
        )
        print(f"Merged {len(postproc_df)} post-processed GPS records from {postprocessed_gps_paths}")

    field_gps_df = merge_position_files(
        file_paths=gps_paths,
        load_function=load_and_parse_gps_file,
        time_sort_key='GPS_TIME',
        remove_duplicates=True,
    )
    print(f"Merged {len(field_gps_df)} GPS records")
    print(f"field_gps_df columns: {field_gps_df.columns.tolist()}")

    if imu_paths and len(imu_paths) > 0:
        field_imu_df = merge_position_files(
            file_paths=imu_paths,
            load_function=load_and_parse_imu_file,
            time_sort_key='GPS_TIME',
            load_kwargs={},
            remove_duplicates=False,
        )
        print(f"Merged {len(field_imu_df)} IMU records")

    if postproc_df is not None and not postproc_df.empty:
        merged_df = postproc_df
        source = "UTIG_EPUTG1B-postproc"
        merged_df = merge_df(merged_df, field_gps_df, interp_x_key='GPS_TIME',
                             interp_y_keys=['RADAR_TIME', 'COMP_TIME'],
                             other_keys_suffix="_field_gps")
    else:
        merged_df = field_gps_df
        source = "UTIG_GPSnc1-field"

        if field_imu_df is not None and not field_imu_df.empty:
            merged_df = merge_df(merged_df, field_imu_df, interp_x_key='GPS_TIME',
                                 interp_y_keys=['HEADING', 'PITCH', 'ROLL'],
                                 other_keys_suffix="_field_imu")

    merged_df = merged_df.dropna(subset=['GPS_TIME', 'COMP_TIME', 'RADAR_TIME', 'LAT', 'LON'])
    print(f"Merged dataframe columns: {merged_df.columns.tolist()} (len: {len(merged_df)}) (source: {source})")

    # Pad with extrapolated boundary entries to cover small GPS/radar time gaps.
    if gps_pad_time_s and gps_pad_time_s > 0:
        entry_before = merged_df.iloc[0]
        entry_after = merged_df.iloc[-1]
        for t_key in ['GPS_TIME', 'COMP_TIME', 'RADAR_TIME']:
            t = merged_df[t_key]
            dt_per_gps_time = (t.iloc[-1] - t.iloc[0]) / (merged_df['GPS_TIME'].iloc[-1] - merged_df['GPS_TIME'].iloc[0])
            entry_before[t_key] -= dt_per_gps_time * gps_pad_time_s
            entry_after[t_key] += dt_per_gps_time * gps_pad_time_s

        merged_df = pd.concat([pd.DataFrame([entry_before]), merged_df, pd.DataFrame([entry_after])], ignore_index=True)

    if output_path_temporary_df is not None:
        output_path_temporary_df = Path(output_path_temporary_df)
        output_path_temporary_df.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path_temporary_df, index=False)
        print(f"Saved merged dataframe to: {output_path_temporary_df}")

    gps_struct = create_gps_matlab_structure(merged_df, source=source)
    save_gps_matlab_file(gps_struct, output_path)
    print(f"Saved GPS file to: {output_path}")

    print(f"GPS time range: {gps_struct['gps_time'][0,0]:.2f} to {gps_struct['gps_time'][0,-1]:.2f}")
    print(f"Lat range: {gps_struct['lat'].min():.6f} to {gps_struct['lat'].max():.6f}")
    print(f"Lon range: {gps_struct['lon'].min():.6f} to {gps_struct['lon'].max():.6f}")
    print(f"Elev range: {gps_struct['elev'].min():.2f} to {gps_struct['elev'].max():.2f} meters")


def make_segment_gps_file(x, output_base_dir, overwrite=False):
    """Apply to a segment-grouped DataFrame to produce one GPS .mat per segment.

    Example: `df_season.groupby(['segment_date_str','segment_number'])[[...]].apply(make_segment_gps_file, ...)`
    """
    x = x.reset_index()
    print(f"{x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}")
    gps_paths = list(x['gps_path'].unique())

    if 'imu_path' in x:
        if x['imu_path'].isnull().any():
            warnings.warn(f"IMU paths contain null values for segment {x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}")
            imu_paths = None
        else:
            imu_paths = list(x['imu_path'].unique())
    else:
        imu_paths = None

    if 'postprocessed_gps_path' in x:
        if x['postprocessed_gps_path'].isnull().any():
            warnings.warn(f"Post-processed GPS paths contain null values for segment {x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}")
            postprocessed_gps_paths = None
        else:
            postprocessed_gps_paths = list(x['postprocessed_gps_path'].unique())
    else:
        print("[WARNING] No post-processed GPS paths provided.")
        postprocessed_gps_paths = None

    output_path = output_base_dir / Path(f"gps_{x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}.mat")

    if (not output_path.exists()) or overwrite:
        try:
            generate_gps_file(gps_paths, output_path, imu_paths=imu_paths,
                              postprocessed_gps_paths=postprocessed_gps_paths,
                              gps_pad_time_s=3)
        except ValueError as e:
            print(f"Failed to generate GPS file for segment {x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}: {e}")
            return None
    else:
        print(f"File {output_path} already exists. Skipping generation. If you want to regenerate, delete the file or set overwrite=True.")

    return output_path.resolve()
