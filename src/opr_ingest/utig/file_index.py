"""Discover UTIG data files on disk and arrange them by transect."""

import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_file_index_df(base_path: str, cache_file: str, read_cache: bool = True) -> pd.DataFrame:
    """Build (or load from cache) a DataFrame of every xds.gz / bxds* file under `base_path`."""
    cache_path = Path(cache_file)
    if read_cache and cache_path.exists():
        print(f"Reading from cache file {cache_path}")
        df_files = pd.read_csv(cache_path, index_col=0, low_memory=False)
        df_files.columns = df_files.columns.astype(int)
    else:
        print(f"Generating file index")
        print(f"Looking for xds.gz")
        xds_files = glob.glob(f"{base_path}/**/xds.gz", recursive=True)
        print(f"Looking for bxds")
        bxds_files = glob.glob(f"{base_path}/**/bxds*", recursive=True)
        df_files = pd.DataFrame([Path(f).parts for f in (xds_files + bxds_files)])

        if cache_file is not None:
            print(f"Saving new cache to {cache_file}")
            df_files.to_csv(cache_path)

    return df_files


def create_artifacts_df(file_index_df: pd.DataFrame, datasets=['UTIG1', 'UTIG2']) -> pd.DataFrame:
    column_mapping = {
        5: "dataset",
        6: "processing_level",
        7: "processing_type",
        8: "prj",
        9: "set",
        10: "trn",
        11: "stream",
        12: "file_name",
        "full_path": "full_path",
    }

    df_tmp = file_index_df.copy()
    df_tmp = df_tmp.dropna(axis='columns')
    df_tmp["full_path"] = df_tmp.apply(lambda row: Path(*row).as_posix(), axis=1)
    df_tmp = df_tmp[list(column_mapping.keys())]
    df_artifacts = df_tmp.rename(columns=column_mapping)

    if datasets is not None:
        df_artifacts = df_artifacts[df_artifacts['dataset'].isin(datasets)]

    df_artifacts['artifact'] = df_artifacts.apply(
        lambda row: tuple(row[['processing_level', 'processing_type', 'stream']]), axis='columns'
    )
    return df_artifacts


def arrange_by_transect(df_artifacts, streams, ignore_set=False):
    """Group `df_artifacts` by transect (prj, set, trn) and pull out the requested data streams.

    `streams` maps data-category name to {`stream_types`: [...], `file_names`: [...]}.
    The result has `<category>_stream_type` and `<category>_path` columns; preference
    goes to earlier entries in `stream_types`. If `ignore_set` is True, group by (prj, trn).
    """

    def agg_fn(group):
        df = pd.DataFrame(index=[0])

        for data_category in streams.keys():
            df[f"{data_category}_stream_type"] = np.nan
            df[f"{data_category}_path"] = np.nan

            matching_entry = group[
                (group['stream'].isin(streams[data_category]['stream_types']))
                & (group['file_name'].isin(streams[data_category]['file_names']))
            ].copy()
            if not matching_entry.empty:
                preferred_order = {stype: i for i, stype in enumerate(streams[data_category]['stream_types'])}
                matching_entry['preferred_order'] = matching_entry['stream'].map(preferred_order)
                matching_entry = matching_entry.sort_values('preferred_order')

                df[f"{data_category}_stream_type"] = matching_entry['stream'].values[0]
                df[f"{data_category}_path"] = matching_entry['full_path'].values[0]

        for k in group:
            if k in ['full_path', 'stream', 'processing_level', 'processing_type']:
                continue
            if len(group[k].unique()) == 1:
                df[k] = str(group[k].values[0])

        return df

    groupby_keys = ['prj', 'trn'] if ignore_set else ['prj', 'set', 'trn']
    df = df_artifacts.groupby(groupby_keys).apply(agg_fn, include_groups=False)
    df.index = df.index.droplevel(-1)
    return df
