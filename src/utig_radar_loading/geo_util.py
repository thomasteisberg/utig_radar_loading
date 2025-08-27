import pyproj
from shapely import LineString
import numpy as np

def project_split_and_simplify(lon, lat, projection='EPSG:3031', simplify_tolerance=1000,
                                split_dist=2000):
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
        simplified_line = line.simplify(tolerance=simplify_tolerance)
        coords = list(simplified_line.coords)

        x_simplified.extend([c[0] for c in coords])
        y_simplified.extend([c[1] for c in coords])
        x_simplified.append(np.nan)
        y_simplified.append(np.nan)
    
    return x_simplified, y_simplified