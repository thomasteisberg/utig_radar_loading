"""Read OPR layer files, resolving layers by name (never by column order).

A CSARP_layer/<segment> directory holds one layer-organizer file
``layer_<seg>.mat`` (maps layer name -> lyr_id) and one per-frame data file
``Data_<seg>_<frm>.mat`` (twtt etc., rows keyed by lyr_id). See the OPR Layer
File Guide: https://gitlab.com/openpolarradar/opr/-/wikis/Layer-File-Guide
"""

import re
from pathlib import Path

import h5py
import numpy as np

from .matlab_io import deref_str_cell, mat_str, orient

FRAME_RE = re.compile(r"Data_(\d{8}_\d+)_(\d+)\.mat$")


def read_layer_organizer(layer_dir) -> dict[str, int]:
    """Return {layer_name: lyr_id} from the layer_<seg>.mat organizer file."""
    layer_dir = Path(layer_dir)
    org = list(layer_dir.glob("layer_*.mat"))
    if len(org) != 1:
        raise FileNotFoundError(
            f"Expected exactly one layer organizer in {layer_dir}, found {len(org)}"
        )
    with h5py.File(org[0], "r") as f:
        ftype = mat_str(f["file_type"]) if "file_type" in f else ""
        if "layer_organizer" not in ftype:
            raise ValueError(f"{org[0].name}: file_type={ftype!r}, not a layer_organizer")
        names = deref_str_cell(f, f["lyr_name"])
        ids = np.asarray(f["lyr_id"][()]).ravel().astype(int)
    return {name: int(i) for name, i in zip(names, ids)}


def _read_frame(path: Path, want_ids: list[int]) -> tuple[dict, dict]:
    """Read one per-frame file: (record fields, {lyr_id: twtt}) for want_ids."""
    with h5py.File(path, "r") as f:
        gps_time = np.asarray(f["gps_time"][()]).ravel()
        nx = gps_time.size
        rec = {
            "gps_time": gps_time,
            "lat": np.asarray(f["lat"][()]).ravel(),
            "lon": np.asarray(f["lon"][()]).ravel(),
            "elev": np.asarray(f["elev"][()]).ravel(),
        }
        ids = np.asarray(f["id"][()]).ravel().astype(int)
        twtt = orient(np.asarray(f["twtt"][()]), nx)  # (Nx, Nlayer)
    by_id = {}
    for lid in want_ids:
        col = np.where(ids == lid)[0]
        if col.size:
            by_id[lid] = twtt[:, col[0]]
    return rec, by_id


def load_segment(layer_dir, layer_names: list[str]) -> list[tuple[str, dict, dict]]:
    """Load every frame in a CSARP_layer/<segment> directory.

    Returns a list of (frame_id, record_fields, {layer_name: twtt}). Layer names
    are resolved through the organizer file; a missing name raises KeyError.
    """
    layer_dir = Path(layer_dir)
    name_to_id = read_layer_organizer(layer_dir)
    wanted = list(dict.fromkeys(layer_names))  # de-dup, keep order
    missing = [n for n in wanted if n not in name_to_id]
    if missing:
        raise KeyError(
            f"{layer_dir.name}: layer(s) {missing} not in organizer "
            f"(available: {sorted(name_to_id)})"
        )
    id_for = {n: name_to_id[n] for n in wanted}
    frames = []
    for p in sorted(layer_dir.glob("Data_*.mat")):
        m = FRAME_RE.search(p.name)
        if not m:
            continue
        rec, by_id = _read_frame(p, list(set(id_for.values())))
        layers = {n: by_id[id_for[n]] for n in wanted if id_for[n] in by_id}
        frames.append((f"{m[1]}_{m[2]}", rec, layers))
    if not frames:
        raise FileNotFoundError(f"No Data_*.mat layer frames in {layer_dir}")
    return frames
