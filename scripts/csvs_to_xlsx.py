"""Assemble a directory of per-sheet CSVs into a single OPR parameter xlsx.

Each `<name>.csv` in the input directory becomes a `<name>` sheet in the output
workbook (one CSV → one tab, preserving header + data). Sheets are ordered by
the canonical OPR sequence (cmd, records, qlook, sar, array, radar, post,
analysis_noise) when present, then any extra CSVs are appended alphabetically.

Usage:
    uv run scripts/csvs_to_xlsx.py <csv_dir> [--output <path.xlsx>]

The default output is `<csv_dir>/<csv_dir.name>.xlsx`.
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


def order_csvs(csv_paths):
    by_stem = {p.stem: p for p in csv_paths}
    ordered = [by_stem.pop(name) for name in CANONICAL_ORDER if name in by_stem]
    ordered.extend(by_stem[stem] for stem in sorted(by_stem))
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_dir", type=Path, help="Directory containing per-sheet CSVs")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output xlsx path (default: <csv_dir>/<csv_dir.name>.xlsx)",
    )
    args = parser.parse_args()

    csv_dir = args.csv_dir
    if not csv_dir.is_dir():
        sys.exit(f"Not a directory: {csv_dir}")

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        sys.exit(f"No CSVs found in {csv_dir}")

    output = args.output or csv_dir / f"{csv_dir.name}.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)

    ordered_paths = order_csvs(csv_paths)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for path in ordered_paths:
            df = pd.read_csv(path)
            df.to_excel(writer, sheet_name=path.stem, index=False)
            print(f"  {path.stem:16s}  rows={len(df):>4}  cols={df.shape[1]:>2}  ← {path.name}")

    print(f"\nWrote {len(ordered_paths)} sheets to {output}")


if __name__ == "__main__":
    main()
