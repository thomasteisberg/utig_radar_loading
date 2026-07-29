"""Low-level helpers for reading MATLAB v7.3 (HDF5) .mat files with h5py.

MATLAB stores arrays column-major, so h5py reads an (M, N) MATLAB array as
(N, M). Strings are uint16 char arrays; cell arrays of strings are stored as
HDF5 object references.
"""

import numpy as np


def mat_str(value) -> str:
    """Decode a MATLAB char array (uint16) to a Python str."""
    arr = np.asarray(value).ravel()
    return "".join(chr(int(c)) for c in arr if c != 0)


def deref_str_cell(f, dataset) -> list[str]:
    """Decode a MATLAB cell array of strings stored as HDF5 object references."""
    return [mat_str(f[ref]) for ref in np.asarray(dataset[()]).ravel()]


def orient(arr: np.ndarray, nx: int) -> np.ndarray:
    """Return `arr` with the length-`nx` (record) axis first.

    Layer `twtt` is Nlayer-by-Nx in MATLAB, so h5py yields (Nx, Nlayer); this
    normalizes regardless of how a particular file happened to be written.
    """
    arr = np.atleast_2d(arr)
    if arr.shape[0] != nx and arr.shape[1] == nx:
        arr = arr.T
    return arr
