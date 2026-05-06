"""Utilities for reading and writing OPR parameter spreadsheets."""

import re
from pathlib import Path
import pandas as pd


def _parse_opr_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Parse an OPR parameter spreadsheet sheet.

    OPR sheets have this structure:
      - cmd sheet: rows 0-2 are metadata (Version/Radar/Season), row 3 is column
        names (Date, Segment, frms, ...), row 4 is type codes, data starts at row 5.
      - Other sheets: row 0 is column names, row 1 is type codes, data starts at row 2.

    We detect which format by checking if the first cell contains 'Version' or 'Radar'.
    """
    raw = df.copy()

    # Detect cmd-style metadata header (Version/Radar/Season rows before column names)
    first_col_vals = raw.iloc[:, 0].astype(str).tolist()
    metadata_rows = 0
    for val in first_col_vals:
        if val.lower() in ("version", "radar", "season"):
            metadata_rows += 1
        else:
            break

    # Row after metadata is column names, then type codes, then data
    header_row = metadata_rows
    type_row = metadata_rows + 1
    data_start = metadata_rows + 2

    if data_start > len(raw):
        return pd.DataFrame()

    # Extract column names from the header row
    col_names = raw.iloc[header_row].tolist()
    # Clean up: replace NaN with generated names
    col_names = [
        str(c) if pd.notna(c) else f"_col{i}" for i, c in enumerate(col_names)
    ]
    # Rename the first two columns to standard names
    if len(col_names) >= 2:
        col_names[0] = "day_seg_date"
        col_names[1] = "day_seg_num"

    # Extract data rows
    result = raw.iloc[data_start:].copy()
    result.columns = col_names[: len(result.columns)]
    result = result.reset_index(drop=True)

    return result


def read_xlsx(path: Path) -> dict[str, pd.DataFrame]:
    """Read OPR xlsx parameter spreadsheet, return {sheet_name: DataFrame}.

    Handles the OPR header format (metadata rows, column names, type codes).
    """
    path = Path(path)
    raw_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl", header=None)
    return {name: _parse_opr_sheet(df) for name, df in raw_sheets.items()}


def read_csvs(directory: Path) -> dict[str, pd.DataFrame]:
    """Read directory of CSVs (one per tab), return {tab_name: DataFrame}.

    Tab name is derived from filename (e.g., 'cmd.csv' -> 'cmd').
    These CSVs use the simple format from define_segments.py (no OPR metadata rows).
    """
    directory = Path(directory)
    sheets = {}
    for csv_file in sorted(directory.glob("*.csv")):
        tab_name = csv_file.stem
        df = pd.read_csv(csv_file)
        # Rename first two columns to match xlsx convention
        cols = list(df.columns)
        if len(cols) >= 2:
            rename = {cols[0]: "day_seg_date", cols[1]: "day_seg_num"}
            df = df.rename(columns=rename)
        sheets[tab_name] = df
    return sheets


def parse_matlab_cell_string(s: str) -> list[str]:
    """Parse MATLAB cell string like "{'a', 'b'}" -> ['a', 'b'].

    Handles single-element, empty, and bare string cases.
    """
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    # Remove outer braces if present
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if not s:
        return []
    # Split on commas, strip quotes and whitespace
    parts = re.split(r",\s*", s)
    result = []
    for part in parts:
        part = part.strip().strip("'\"")
        if part:
            result.append(part)
    return result


def segments_to_process(sheets: dict[str, pd.DataFrame]) -> pd.Index:
    """Return index of segments not marked 'do not process' in cmd tab."""
    if "cmd" not in sheets:
        raise ValueError("No 'cmd' sheet found in spreadsheet")
    cmd = sheets["cmd"]
    if "notes" not in cmd.columns:
        return cmd.index
    mask = ~cmd["notes"].astype(str).str.contains("do not process", case=False, na=False)
    return cmd.index[mask]
