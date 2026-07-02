"""
combine_cz_sk_cms_lamberts.py
──────────────────────────────
Combines the Czech/Slovak young central-midfielder screen
(export_cz_sk_young_cms.py) with three named recruitment-scoring lenses,
each on its own workbook tab:

  Lamberts Model   — Jamestown / Marc Lamberts methodology (build_lamberts_total.py):
                      SQS Rank vs Market Value Rank. Flags statistically
                      undervalued output (this repo's existing "value" model).
  Benham Model     — modeled on Matthew Benham / Brentford's data-led approach:
                      weights underlying chance creation (xA, key passes, xG,
                      progressive passing) rather than results already banked.
  Red Bull Model    — modeled on the Red Bull Group pipeline archetype:
                      weights youth, pressing/duel intensity, and progressive
                      ball-carrying — the "athletic upside, sell-on" profile.

The Benham and Red Bull blueprints are NOT pulled from
"data/FCHK Model V3 - Recruitment Scores.xlsx" (which does carry
BenhamScore/RedBullScore columns) because that workbook's player pool does
not overlap with the Wyscout-sourced shortlist here (different name format,
different — likely demo/sample — squads). Instead they're built directly
from the same Wyscout per-90 data as the Lamberts model, using weighted
blueprints inspired by each club's publicly known recruitment philosophy —
they are proxies, not the clubs' actual proprietary formulas.

Reads
─────
  data/CZ_SK_Young_CMs.csv                    — CZ/SK, age <= 23, CMF/LCMF/RCMF
  data/Lamberts_Index_Model_All_Leagues.xlsx  — "All Targets" sheet (age <= 30,
                                                 400+ minutes, all leagues) — both
                                                 the Lamberts Index source and the
                                                 percentile pool for all 3 models

Players below the model's 400-minute threshold have no score on any tab; they
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

from build_lamberts_total import write_data_sheet

ROOT = Path(__file__).parent

SHORTLIST_CSV = ROOT / "data" / "CZ_SK_Young_CMs.csv"
LAMBERTS_XLSX = ROOT / "data" / "Lamberts_Index_Model_All_Leagues.xlsx"

LAMBERTS_COLS = [
    "Player", "Team", "League", "Age", "Passport country", "Foot",
    "Minutes played", "Tier", "SQS Rank", "Lamberts Index", "Status",
    "Mkt Val (€)", "Model Val (€)", "Val Ratio", "vs Hradec",
    "Goals/90", "Assists/90", "xA/90", "Prog Pass/90", "Key Pass/90",
]

# (metric column in the CM pool, weight). Higher weight = more influence.
BENHAM_BLUEPRINT: list[tuple[str, float]] = [
    ("xA/90",        3.0),
    ("Key Pass/90",  2.5),
    ("xG/90",        2.0),
    ("Prog Pass/90", 2.0),
    ("Assists/90",   1.5),
]

REDBULL_BLUEPRINT: list[tuple[str, float]] = [
    ("_age_score",   3.0),   # younger = higher, see compute_model_scores
    ("Prog Run/90",  2.5),
    ("Def Duel %",   2.0),
    ("Interceptions",1.5),
    ("Dribbles/90",  1.5),
    ("Drib Succ %",  1.0),
    ("Aerial %",     1.0),
]


def grade(score: float) -> str:
    if pd.isna(score):
        return "NOT SCORED"
    if score >= 80:
        return "TOP TARGET"
    if score >= 65:
        return "STRONG"
    if score >= 50:
        return "SOLID"
    if score >= 35:
        return "FRINGE"
    return "LOW PRIORITY"


def compute_model_scores(pool: pd.DataFrame) -> pd.DataFrame:
    """Add BenhamScore / RedBullScore (0-100 percentile) to the CM pool."""
    pool = pool.copy()
    pool["_age_score"] = 100 - pd.to_numeric(pool["Age"], errors="coerce").rank(pct=True) * 100

    for name, blueprint in (("Benham", BENHAM_BLUEPRINT), ("RedBull", REDBULL_BLUEPRINT)):
        raw = pd.Series(0.0, index=pool.index)
        total_w = 0.0
        for metric, w in blueprint:
            if metric not in pool.columns:
                continue
            vals = pd.to_numeric(pool[metric], errors="coerce").fillna(0)
            raw += w * (vals.rank(pct=True) * 100)
            total_w += w
        if total_w > 0:
            raw /= total_w
        pool[f"{name}Score"] = (raw.rank(pct=True) * 100).round(2)

    return pool


def load_cm_pool() -> pd.DataFrame:
    pool = pd.read_excel(LAMBERTS_XLSX, sheet_name="All Targets", skiprows=2)
    pool = pool[pool["Pos"] == "CM"].copy()
    return compute_model_scores(pool)


def merge_shortlist(shortlist: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    merged = shortlist.merge(
        pool,
        left_on=["Player", "Team", "_League"],
        right_on=["Player", "Team", "League"],
        how="left",
        suffixes=("", "_pool"),
    )
    merged["League"] = merged["League"].fillna(merged["_League"])
    merged["Age"] = merged["Age_pool"].fillna(merged["Age"])
    merged["Tier"] = merged["Tier"].fillna("NOT SCORED (<400 min)")
    merged["Status"] = merged["Status"].fillna("NOT SCORED")
    for col in ["SQS Rank", "Lamberts Index", "Mkt Val (€)", "Model Val (€)",
                "vs Hradec", "BenhamScore", "RedBullScore"]:
        if col not in merged.columns:
            merged[col] = None
    return merged


def write_report(df: pd.DataFrame, output: Path) -> None:
    wb = Workbook()
    wb.remove(wb.active)

    # ── Lamberts Model ──────────────────────────────────────────────────
    lamberts_df = df[[c for c in LAMBERTS_COLS if c in df.columns]].copy()
    lamberts_df = lamberts_df.sort_values(
        "Lamberts Index", ascending=False, na_position="last"
    )
    scored = int(df["Lamberts Index"].notna().sum())
    ws1 = wb.create_sheet("Lamberts Model")
    write_data_sheet(
        ws1,
        "CZ / SK CENTRAL MIDFIELDERS, AGE ≤ 23 — LAMBERTS MODEL",
        f"Jamestown / Marc Lamberts methodology  ·  {len(lamberts_df)} candidates  ·  "
        f"{scored} scored (400+ minutes)  ·  Ranked by Lamberts Index (SQS Rank − Market Value Rank)",
        lamberts_df,
    )

    # ── Benham Model ─────────────────────────────────────────────────────
    benham_cols = ["Player", "Team", "League", "Age", "Passport country",
                   "Minutes played", "BenhamScore", "xA/90", "Key Pass/90",
                   "xG/90", "Prog Pass/90", "Assists/90"]
    benham_df = df[[c for c in benham_cols if c in df.columns]].copy()
    benham_df.insert(benham_df.columns.get_loc("BenhamScore"), "Grade",
                      benham_df["BenhamScore"].apply(grade))
    benham_df = benham_df.sort_values("BenhamScore", ascending=False, na_position="last")
    ws2 = wb.create_sheet("Benham Model")
    write_data_sheet(
        ws2,
        "CZ / SK CENTRAL MIDFIELDERS, AGE ≤ 23 — BENHAM MODEL",
        "Modeled on Matthew Benham / Brentford's data-led approach  ·  "
        "weights underlying chance creation (xA, key passes, xG, progressive passing)  ·  "
        "proxy score, not the club's proprietary formula",
        benham_df,
    )

    # ── Red Bull Model ───────────────────────────────────────────────────
    redbull_cols = ["Player", "Team", "League", "Age", "Passport country",
                     "Minutes played", "RedBullScore", "Prog Run/90",
                     "Def Duel %", "Interceptions", "Dribbles/90",
                     "Drib Succ %", "Aerial %"]
    redbull_df = df[[c for c in redbull_cols if c in df.columns]].copy()
    redbull_df.insert(redbull_df.columns.get_loc("RedBullScore"), "Grade",
                       redbull_df["RedBullScore"].apply(grade))
    redbull_df = redbull_df.sort_values("RedBullScore", ascending=False, na_position="last")
    ws3 = wb.create_sheet("Red Bull Model")
    write_data_sheet(
        ws3,
        "CZ / SK CENTRAL MIDFIELDERS, AGE ≤ 23 — RED BULL MODEL",
        "Modeled on the Red Bull Group pipeline archetype  ·  "
        "weights youth, pressing/duel intensity and progressive ball-carrying  ·  "
        "proxy score, not the club's proprietary formula",
        redbull_df,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "CZ_SK_Young_CMs_Lamberts.xlsx"
    )
    args = parser.parse_args()

    shortlist = pd.read_csv(SHORTLIST_CSV)
    pool = load_cm_pool()
    df = merge_shortlist(shortlist, pool)

    csv_path = args.output.with_suffix(".csv")
    export_cols = LAMBERTS_COLS + ["BenhamScore", "RedBullScore"]
    df[[c for c in export_cols if c in df.columns]].drop_duplicates(
        subset=["Player", "Team", "League"]
    ).to_csv(csv_path, index=False)

    write_report(df, args.output)

    print(f"{len(df)} players")
    print(f"  Lamberts scored: {int(df['Lamberts Index'].notna().sum())}")
    print(f"  Benham scored:   {int(df['BenhamScore'].notna().sum())}")
    print(f"  Red Bull scored: {int(df['RedBullScore'].notna().sum())}")
    print(f"CSV   → {csv_path}")
    print(f"Excel → {args.output}  (Lamberts Model / Benham Model / Red Bull Model tabs)")


if __name__ == "__main__":
    main()
