"""Assemble a directory of per-sheet CSVs into an OPR parameter xlsx.

Emits OPR's expected layout: the `cmd` sheet gets a 5-row header block
(Version / Radar / Season metadata, then column names, then per-column
type codes), every other sheet gets a 2-row header block (column names,
then type codes). OPR's `read_param_xls_generic.m` literally searches
for "Date" in column 1 to locate the header row and errors if any
column has a name but no type code.

Type codes per OPR convention:
  r       general read (numeric, MATLAB expressions, cell strings)
  b       boolean (0/1)
  t       text (paths, names, profiles)

Usage:
    uv run scripts/csvs_to_xlsx.py <csv_dir> [-o <path.xlsx>] [--radar-name rds]

Default output: `<csv_dir>/<csv_dir.name>.xlsx`. Default radar_name `rds`.
Season name is taken from the directory name.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


CANONICAL_ORDER = [
    "cmd",
    "records",
    "qlook",
    "sar",
    "array",
    "radar",
    "post",
    "analysis_noise",
]

PARAM_FILE_VERSION = "4.0"

# Type-code overrides by (sheet, column). Anything not listed defaults to 'r'.
_CMD_TYPECODES = {
    "frms": "r",
    "records": "b",
    "qlook": "b",
    "sar": "b",
    "array": "b",
    "generic": "r",
    "mission_names": "t",
    "notes": "t",
}

# Radar-sheet columns that are segment-level (flat) rather than per-waveform.
# Everything else on the radar sheet is routed into the param.radar.wfs(1)
# struct array via the 'ar:wfs(1)' type code (see _typecode). This mirrors
# OPR's radar schema: read_param_xls builds param.radar.wfs(N) from columns
# whose type code is 'ar:wfs(N)', and a flat param.radar.<field> from 'r'.
# Without this, qlook's img_combine_input_check errors with
# "Unrecognized field name 'wfs'".
_RADAR_FLAT_COLUMNS = {
    "fs",
    "prf",
    "adc_bits",
    "Vpp_scale",
    "lever_arm_fh",
}

# Columns that must be read as literal text (paths, etc.) rather than
# evaluated as MATLAB expressions. Applies to non-cmd sheets.
_TEXT_COLUMNS = {
    # records
    "file.base_dir",
    "file.prefix",
    "frames.geotiff_fn",
    "gps.fn",
    # qlook
    "out_path",
    "surf.profile",
    # array
    "in_path",
}


def order_csvs(csv_paths):
    by_stem = {p.stem: p for p in csv_paths}
    ordered = [by_stem.pop(name) for name in CANONICAL_ORDER if name in by_stem]
    ordered.extend(by_stem[stem] for stem in sorted(by_stem))
    return ordered


def _typecode(sheet_name: str, col_name: str) -> str:
    if sheet_name == "cmd":
        return _CMD_TYPECODES.get(col_name, "r")
    if sheet_name == "radar":
        if col_name == "size":
            return "ar:wfs"
        if col_name not in _RADAR_FLAT_COLUMNS:
            return "ar:wfs(1)"
        # flat radar fields fall through to the text/r logic below
    return "t" if col_name in _TEXT_COLUMNS else "r"


def _build_opr_rows(df: pd.DataFrame, sheet_name: str, season_name: str, radar_name: str):
    """Return a list of rows (each row a list of cell values) in OPR layout."""
    # Columns 1 and 2 are always Date/Segment; columns 3+ are the actual fields.
    field_cols = list(df.columns[2:])
    type_codes = [_typecode(sheet_name, c) for c in field_cols]

    if sheet_name == "cmd":
        header_rows = [
            ["Version", PARAM_FILE_VERSION] + [None] * len(field_cols),
            ["Radar", radar_name] + [None] * len(field_cols),
            ["Season", season_name] + [None] * len(field_cols),
            ["Date", None] + field_cols,
            ["YYYYMMDD", "Segment"] + type_codes,
        ]
    else:
        header_rows = [
            ["Date", "Segment"] + field_cols,
            ["YYYYMMDD", "Segment"] + type_codes,
        ]

    data_rows = df.values.tolist()
    return header_rows + data_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_dir", type=Path, help="Directory containing per-sheet CSVs")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output xlsx path (default: <csv_dir>/<csv_dir.name>.xlsx)",
    )
    parser.add_argument(
        "--radar-name", default="rds",
        help="OPR radar_name written to cell B2 of the cmd sheet (default: rds)",
    )
    args = parser.parse_args()

    csv_dir = args.csv_dir
    if not csv_dir.is_dir():
        sys.exit(f"Not a directory: {csv_dir}")

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        sys.exit(f"No CSVs found in {csv_dir}")

    season_name = csv_dir.name
    output = args.output or csv_dir / f"{csv_dir.name}.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)

    ordered_paths = order_csvs(csv_paths)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for path in ordered_paths:
            df = pd.read_csv(path)
            rows = _build_opr_rows(df, path.stem, season_name, args.radar_name)
            out_df = pd.DataFrame(rows)
            out_df.to_excel(
                writer, sheet_name=path.stem, index=False, header=False
            )
            print(f"  {path.stem:16s}  rows={len(df):>4}  cols={df.shape[1]:>2}  ← {path.name}")

    print(f"\nWrote {len(ordered_paths)} sheets to {output}")
    print(f"  season_name = {season_name}")
    print(f"  radar_name  = {args.radar_name}")


if __name__ == "__main__":
    main()
