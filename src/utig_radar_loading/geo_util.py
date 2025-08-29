import pyproj
from shapely import LineString
import numpy as np
import pandas as pd
import holoviews as hv

def project_split_and_simplify(lon, lat, projection='EPSG:3031', simplify_tolerance=1000,
                                split_dist=2000, calc_length=False):
    # Project
    transformer = pyproj.Transformer.from_crs('EPSG:4326', projection, always_xy=True)
    x_proj, y_proj = transformer.transform(lon, lat)

    # Break into separate segments
    dist_deltas = np.sqrt(np.diff(x_proj)**2 + np.diff(y_proj)**2)
    segment_indices = np.array(np.where(dist_deltas > split_dist)) + 1
    segment_indices = np.insert(segment_indices, 0, 0)
    segment_indices = np.append(segment_indices, len(x_proj))

    x_simplified = []
    y_simplified = []

    length = 0

    for start_idx, end_idx in zip(segment_indices[:-1], segment_indices[1:]):
        if end_idx - start_idx < 5:
            continue

        x_segment = x_proj[start_idx:end_idx]
        y_segment = y_proj[start_idx:end_idx]

        if np.isnan(x_segment).any() or np.isnan(y_segment).any():
            print(f"Warning: NaN values found in segment {start_idx}:{end_idx}")
            continue

        # Use shapely to simplify paths to 1km tolerance
        line = LineString(zip(x_segment, y_segment))
        if calc_length:
            length += line.length
        
        if simplify_tolerance:
            line = line.simplify(tolerance=simplify_tolerance)
        coords = list(line.coords)

        x_simplified.extend([c[0] for c in coords])
        y_simplified.extend([c[1] for c in coords])
        x_simplified.append(np.nan)
        y_simplified.append(np.nan)

    if calc_length:
        return x_simplified, y_simplified, length
    else:
        return x_simplified, y_simplified

def create_path(segment_dfs, path_opts_kwargs={}):
    dfs = []

    for idx, df_sub in enumerate(segment_dfs):
        df_tmp = df_sub.copy()
        df_tmp = df_tmp[df_tmp['LAT'] <= -50]

        if len(df_tmp) < 3:
            continue

        try:
            x_proj, y_proj = project_split_and_simplify(df_tmp['LON'].values, df_tmp['LAT'].values)
        except Exception as e:
            print(f"Error processing segment {idx}: {e}")
            #print(df_tmp)
            continue

        # Finish with a nan to divide from next
        x_proj.append(np.nan)
        y_proj.append(np.nan)

        # Add projected coordinates to dataframe
        df_simplified = pd.DataFrame({
            'x': x_proj,
            'y': y_proj
        })

        required_fields = ['prj', 'set', 'trn', 'clk_y']
        display_fields = ['prj', 'set', 'trn', 'clk_y']
        if 'radar_stream_type' in df_tmp:
            required_fields.append('radar_stream_type')
            display_fields.append('radar')

        for k in required_fields:
            df_simplified[k] = df_tmp[k].iloc[0]
            if len(df_tmp[k].unique()) > 1:
                print(f"segment_dfs[{idx}]['{k}'].unique(): {df_tmp[k].unique()}")

        if 'radar_stream_type' in df_tmp:
            df_simplified['radar'] = df_simplified['radar_stream_type']

        dfs.append(df_simplified)

    df_combined = pd.concat(dfs, ignore_index=True)
    # Create hv.Path with already projected coordinates
    path = hv.Path(df_combined,
                ['x', 'y'],
                display_fields,
                ).opts(
                    tools=['hover'],
                    line_width=2,
                    show_legend=True,
                    **path_opts_kwargs
                )
    return dfs, path