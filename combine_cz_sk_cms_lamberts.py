"""
combine_cz_sk_cms_lamberts.py
──────────────────────────────
Combines the Czech/Slovak young central-midfielder screen
(export_cz_sk_young_cms.py) with the Jamestown / Marc Lamberts recruitment
methodology (build_lamberts_total.py) to rank the shortlist by SQS Rank,
Market Value Rank, Lamberts Index and vs-Hradec fit.

Reads
─────
  data/CZ_SK_Young_CMs.csv                    — CZ/SK, age <= 23, CMF/LCMF/RCMF
  data/Lamberts_Index_Model_All_Leagues.xlsx  — "All Targets" sheet (age <= 30,
                                                 400+ minutes, all leagues), the
                                                 pool the Lamberts Index percentiles
                                                 are computed against

Players below the model's 400-minute threshold have no Lamberts score; they
are kept in the output and flagged "NOT SCORED (<400 min)" rather than dropped.

Usage
─────
  python combine_cz_sk_cms_lamberts.py [--output data/CZ_SK_Young_CMs_Lamberts.xlsx]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from build_lamberts_total import C, _fill, write_data_sheet

ROOT = Path(__file__).parent

SHORTLIST_CSV = ROOT / "data" / "CZ_SK_Young_CMs.csv"
LAMBERTS_XLSX = ROOT / "data" / "Lamberts_Index_Model_All_Leagues.xlsx"

OUTPUT_COLS = [
    "Player", "Team", "League", "Age", "Passport country", "Foot",
    "Minutes played", "Tier", "SQS Rank", "Lamberts Index", "Status",
    "Mkt Val (€)", "Model Val (€)", "Val Ratio", "vs Hradec",
    "Goals/90", "Assists/90", "xA/90", "Prog Pass/90", "Key Pass/90",
]


def build_combined() -> pd.DataFrame:
    shortlist = pd.read_csv(SHORTLIST_CSV)

    lamberts = pd.read_excel(LAMBERTS_XLSX, sheet_name="All Targets", skiprows=2)
    lamberts = lamberts[lamberts["Pos"] == "CM"].copy()

    merged = shortlist.merge(
        lamberts,
        left_on=["Player", "Team", "_League"],
        right_on=["Player", "Team", "League"],
        how="left",
        suffixes=("", "_lamberts"),
    )

    merged["Tier"] = merged["Tier"].fillna("NOT SCORED (<400 min)")
    merged["Status"] = merged["Status"].fillna("NOT SCORED")
    merged["League"] = merged["League"].fillna(merged["_League"])
    merged["Age"] = merged["Age_lamberts"].fillna(merged["Age"])

    for col in ["SQS Rank", "Lamberts Index", "Mkt Val (€)", "Model Val (€)", "vs Hradec"]:
        if col not in merged.columns:
            merged[col] = None

    merged = merged.sort_values(
        "Lamberts Index", ascending=False, na_position="last"
    ).reset_index(drop=True)

    return merged


def write_report(df: pd.DataFrame, output: Path) -> None:
    out_df = df[[c for c in OUTPUT_COLS if c in df.columns]].copy()

    wb = Workbook()
    ws = wb.active
    ws.sheet_view.showGridLines = False
    scored = int(df["Lamberts Index"].notna().sum())
    write_data_sheet(
        ws,
        f"CZ / SK CENTRAL MIDFIELDERS, AGE ≤ 23 — LAMBERTS INDEX RANKED",
        f"Jamestown / Marc Lamberts methodology  ·  {len(out_df)} candidates  ·  "
        f"{scored} scored (400+ minutes)  ·  Ranked by Lamberts Index",
        out_df,
    )
    ws.title = "CZ-SK U23 CM"

    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "CZ_SK_Young_CMs_Lamberts.xlsx"
    )
    args = parser.parse_args()

    df = build_combined()
    csv_path = args.output.with_suffix(".csv")
    df[[c for c in OUTPUT_COLS if c in df.columns]].to_csv(csv_path, index=False)
    write_report(df, args.output)

    print(f"{len(df)} players  ·  {int(df['Lamberts Index'].notna().sum())} scored")
    print(f"CSV   → {csv_path}")
    print(f"Excel → {args.output}")


if __name__ == "__main__":
    main()
