"""OPR parameter spreadsheet I/O and season-defaults YAML loader."""

import re
import warnings
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import yaml


def _parse_opr_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Parse an OPR parameter spreadsheet sheet.

    OPR sheets have this structure:
      - cmd sheet: rows 0-2 are metadata (Version/Radar/Season), row 3 is column
        names (Date, Segment, frms, ...), row 4 is type codes, data starts at row 5.
      - Other sheets: row 0 is column names, row 1 is type codes, data starts at row 2.

    We detect which format by checking whether the first cell contains 'Version'/'Radar'.
    """
    raw = df.copy()

    first_col_vals = raw.iloc[:, 0].astype(str).tolist()
    metadata_rows = 0
    for val in first_col_vals:
        if val.lower() in ("version", "radar", "season"):
            metadata_rows += 1
        else:
            break

    header_row = metadata_rows
    data_start = metadata_rows + 2

    if data_start > len(raw):
        return pd.DataFrame()

    col_names = raw.iloc[header_row].tolist()
    col_names = [str(c) if pd.notna(c) else f"_col{i}" for i, c in enumerate(col_names)]
    if len(col_names) >= 2:
        col_names[0] = "day_seg_date"
        col_names[1] = "day_seg_num"

    result = raw.iloc[data_start:].copy()
    result.columns = col_names[: len(result.columns)]
    result = result.reset_index(drop=True)
    return result


def read_xlsx(path: Path) -> Dict[str, pd.DataFrame]:
    """Read an OPR xlsx parameter spreadsheet. Returns {sheet_name: DataFrame}."""
    path = Path(path)
    raw_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl", header=None)
    return {name: _parse_opr_sheet(df) for name, df in raw_sheets.items()}


def read_csvs(directory: Path) -> Dict[str, pd.DataFrame]:
    """Read a directory of per-sheet CSVs (as written by the ingest pipeline).

    These CSVs use the simple format from define_segments.py (no OPR metadata rows).
    """
    directory = Path(directory)
    sheets = {}
    for csv_file in sorted(directory.glob("*.csv")):
        tab_name = csv_file.stem
        df = pd.read_csv(csv_file)
        cols = list(df.columns)
        if len(cols) >= 2:
            rename = {cols[0]: "day_seg_date", cols[1]: "day_seg_num"}
            df = df.rename(columns=rename)
        sheets[tab_name] = df
    return sheets


def parse_matlab_cell_string(s: str) -> list[str]:
    """Parse a MATLAB cell-string like "{'a', 'b'}" into ['a', 'b']."""
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if not s:
        return []
    parts = re.split(r",\s*", s)
    result = []
    for part in parts:
        part = part.strip().strip("'\"")
        if part:
            result.append(part)
    return result


def segments_to_process(sheets: Dict[str, pd.DataFrame]) -> pd.Index:
    """Return the index of segments in the cmd tab not marked 'do not process'."""
    if "cmd" not in sheets:
        raise ValueError("No 'cmd' sheet found in spreadsheet")
    cmd = sheets["cmd"]
    if "notes" not in cmd.columns:
        return cmd.index
    mask = ~cmd["notes"].astype(str).str.contains("do not process", case=False, na=False)
    return cmd.index[mask]


def load_defaults(yaml_file: Union[str, Path]) -> Dict:
    """Load season-default parameters from a YAML file."""
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        warnings.warn(f"Defaults file not found: {yaml_path}")
        return {}
    with open(yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    return defaults if defaults else {}
