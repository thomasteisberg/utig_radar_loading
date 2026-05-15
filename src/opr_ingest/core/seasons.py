"""Field-season assignment for Antarctic data.

A field season spans an austral summer and is named for the year it starts in.
Data collected in Jan-May belongs to the season that started the previous calendar year.
"""

import pandas as pd


def season_from_datetime(d: pd.Timestamp) -> int:
    if d.month >= 6:
        return d.year
    return d.year - 1


def assign_seasons(df_transects: pd.DataFrame, get_start_timestamp) -> pd.DataFrame:
    """Add `start_timestamp` and `season` columns to `df_transects`.

    `get_start_timestamp` is a callable that takes a row and returns a pd.Timestamp.
    """
    df = df_transects.copy()
    df['start_timestamp'] = df.apply(get_start_timestamp, axis=1)
    df['season'] = df['start_timestamp'].apply(season_from_datetime)
    df['season'] = df['season'].astype('Int32')
    df = df.sort_values('prj')
    return df
