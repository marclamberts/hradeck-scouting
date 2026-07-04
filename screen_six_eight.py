"""
screen_six_eight.py
────────────────────
Screens the full Wyscout database for two central-midfield archetypes:

  Number 6  — defensive midfielder (DMF/LDMF/RDMF): progressive passing
              combined with defensive-actions volume.
  Number 8  — box-to-box central midfielder (CMF/LCMF/RCMF): progressive
              passing, key passes, through passes, and shot output.

Each archetype gets a 0-100 "Six Score" / "Eight Score", computed as a
weighted blend of percentile ranks within a broad same-position pool
(all leagues, 400+ minutes — the same floor used elsewhere in this repo
for the Lamberts/Benham/RedBull models), so the score reflects standing
against a realistic recruitment pool rather than just the shortlist itself.

Filters applied to the final shortlists
────────────────────────────────────────
  Age            <= 23 (max-age, "U23")
  Market value   < max-value (0/blank market value passes — Wyscout leaves
                 it blank for most lower-league players; excluding those
                 would gut the pool)
  Region         cz_sk is filtered by passport nationality (Czech Republic
                 or Slovakia, wherever the player plays). other is filtered
                 by which domestic league Excel file the player is IN
                 (Estonia.xlsx, Sweden.xlsx, ... regardless of passport),
                 i.e. it's a "who plays in these leagues" screen, not a
                 "who holds these passports" screen.

Regions
───────
  cz_sk   — nationality: Czech Republic, Slovakia
  other   — league files: Baltics (Estonia, Latvia, Lithuania) + Scandinavia
            (Sweden I-III, Norway I-III, Denmark I-IV) + Finland I-II,
            Iceland, Poland I-III, Slovenia I-II

Squad level
───────────
  Every row is tagged "Squad Type":
    Youth Academy      — team name or league itself is a U15-U23 youth side
                          (e.g. "Slavia Praha U19", the "Czech U17" league)
    Reserve / B Team    — team name ends in a reserve suffix (II, III, B)
                          e.g. "Baník Ostrava II", "Internazionale II"
    Senior First Team   — everything else
  "Academy" collapses the first two into one Yes/No flag (neither plays in
  the club's senior first team), and each Number 6 / Number 8 sheet is
  split into a "- Senior" and "- Academy" tab on that flag.

Output
──────
  One workbook per region, each with 4 tabs (Number 6/8 x Senior/Academy):
    data/CZ_SK_U23_Six_Eight.xlsx
    data/Baltics_Scandi_Poland_Slovenia_U23_Six_Eight.xlsx

Usage
─────
  python screen_six_eight.py [--min-minutes 400] [--max-age 23] [--max-value 800000]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

from build_lamberts_total import write_data_sheet

ROOT = Path(__file__).parent
WYSCOUT_DIR = ROOT / "data" / "Wyscout DB"

YOUTH_RE = re.compile(r"\bU1[5-9]\b|\bU2[0-3]\b")
RESERVE_RE = re.compile(r"\b(II|III|B)$")


def classify_squad(team: object, league: object) -> str:
    league_s = str(league or "")
    team_s = str(team or "")
    if YOUTH_RE.search(league_s) or YOUTH_RE.search(team_s):
        return "Youth Academy"
    if RESERVE_RE.search(team_s):
        return "Reserve / B Team"
    return "Senior First Team"

SIX_POSITIONS = {"DMF", "LDMF", "RDMF"}
EIGHT_POSITIONS = {"CMF", "LCMF", "RCMF"}

SIX_BLUEPRINT: list[tuple[str, float]] = [
    ("Progressive passes per 90",           3.0),
    ("Accurate passes, %",                  1.5),
    ("Passes per 90",                       1.0),
    ("Interceptions per 90",                2.0),
    ("PAdj Interceptions",                  1.5),
    ("Successful defensive actions per 90", 2.5),
    ("Defensive duels won, %",              2.0),
    ("Aerial duels won, %",                 1.0),
]

EIGHT_BLUEPRINT: list[tuple[str, float]] = [
    ("Progressive passes per 90", 2.5),
    ("Key passes per 90",         2.5),
    ("Through passes per 90",     2.0),
    ("Shots per 90",              1.5),
    ("Goals per 90",              1.5),
    ("xG per 90",                 1.5),
    ("Progressive runs per 90",   1.5),
    ("Accurate passes, %",        1.0),
]

# mode "nationality" filters on Passport country; mode "league" filters on
# which Wyscout league file (_League, i.e. the file stem) the row came from.
REGIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cz_sk": ("nationality", ("Czech Republic", "Slovakia")),
    "other": ("league", (
        "Estonia", "Latvia", "Lithuania",                                  # Baltics
        "Sweden", "Sweden II", "Sweden III",                               # Scandinavia
        "Norway", "Norway II", "Norway III",
        "Denmark", "Denmark II", "Denmark III", "Denmark IV",
        "Finland", "Finland II", "Iceland",
        "Poland", "Poland II", "Poland III",
        "Slovenia", "Slovenia II",
    )),
}

OUTPUT_FILES = {
    "cz_sk": ROOT / "data" / "CZ_SK_U23_Six_Eight.xlsx",
    "other": ROOT / "data" / "Baltics_Scandi_Poland_Slovenia_U23_Six_Eight.xlsx",
}

DISPLAY_COLS_SIX = [
    "Player", "Team", "_League", "Squad Type", "Age", "Passport country", "Foot",
    "Minutes played", "Market value", "Six Score",
    "Progressive passes per 90", "Accurate passes, %",
    "Interceptions per 90", "PAdj Interceptions",
    "Successful defensive actions per 90", "Defensive duels won, %",
    "Aerial duels won, %",
]

DISPLAY_COLS_EIGHT = [
    "Player", "Team", "_League", "Squad Type", "Age", "Passport country", "Foot",
    "Minutes played", "Market value", "Eight Score",
    "Progressive passes per 90", "Key passes per 90",
    "Through passes per 90", "Shots per 90", "Goals per 90", "xG per 90",
    "Progressive runs per 90",
]


def load_pool(min_minutes: int) -> pd.DataFrame:
    frames = []
    for path in sorted(WYSCOUT_DIR.glob("*.xlsx")):
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "Position" not in df.columns:
            continue
        df = df.copy()
        df["_League"] = path.stem

        first_pos = df["Position"].astype(str).str.split(",").str[0].str.strip()
        df["_arch"] = np.where(
            first_pos.isin(SIX_POSITIONS), "SIX",
            np.where(first_pos.isin(EIGHT_POSITIONS), "EIGHT", None),
        )
        df = df[df["_arch"].notna()]
        if df.empty:
            continue

        mins = pd.to_numeric(df.get("Minutes played"), errors="coerce").fillna(0)
        df = df[mins >= min_minutes]
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool["Squad Type"] = [
        classify_squad(t, lg) for t, lg in zip(pool["Team"], pool["_League"])
    ]
    pool["Academy"] = pool["Squad Type"] != "Senior First Team"
    return pool


def score_archetype(pool: pd.DataFrame, arch: str, blueprint: list[tuple[str, float]],
                     score_name: str) -> pd.DataFrame:
    mask = pool["_arch"] == arch
    grp = pool.loc[mask].copy()
    raw = pd.Series(0.0, index=grp.index)
    total_w = 0.0
    for metric, w in blueprint:
        if metric not in grp.columns:
            continue
        vals = pd.to_numeric(grp[metric], errors="coerce").fillna(0)
        raw += w * (vals.rank(pct=True) * 100)
        total_w += w
    if total_w > 0:
        raw /= total_w
    pool.loc[mask, score_name] = (raw.rank(pct=True) * 100).round(2)
    return pool


def filter_region(df: pd.DataFrame, mode: str, values: tuple[str, ...],
                   max_age: int, max_value: float, score_col: str) -> pd.DataFrame:
    if mode == "nationality":
        passport = df["Passport country"].fillna("").astype(str)
        keep = passport.apply(lambda s: any(v in s for v in values))
    elif mode == "league":
        keep = df["_League"].isin(values)
    else:
        raise ValueError(f"Unknown region mode: {mode}")

    age = pd.to_numeric(df["Age"], errors="coerce")
    mv = pd.to_numeric(df.get("Market value"), errors="coerce").fillna(0)
    out = df.loc[keep & (age <= max_age) & (mv < max_value)].copy()
    # Same player/team can appear in split league files (e.g. "Italy III -
    # Part III/IV") or first-team + reserve exports; keep the top-scoring row.
    out = out.sort_values(score_col, ascending=False).drop_duplicates(
        subset=["Player", "Team"], keep="first"
    )
    return out


def write_region_workbook(six_df: pd.DataFrame, eight_df: pd.DataFrame,
                           region_label: str, max_age: int, max_value: float,
                           output: Path, top_n: int) -> None:
    wb = Workbook()
    wb.remove(wb.active)

    specs = [
        ("Number 6", six_df, DISPLAY_COLS_SIX, "Six Score", "NUMBER 6 — DEFENSIVE MIDFIELDER",
         "Progressive passing + defensive-actions volume"),
        ("Number 8", eight_df, DISPLAY_COLS_EIGHT, "Eight Score", "NUMBER 8 — BOX-TO-BOX MIDFIELDER",
         "Progressive passing, key passes, through passes, shot output"),
    ]

    for base_name, df, display_cols, score_col, title, blurb in specs:
        for tab_suffix, academy_flag in (("Senior", False), ("Academy", True)):
            sub = df[df["Academy"] == academy_flag] if "Academy" in df.columns else df
            out = sub[[c for c in display_cols if c in sub.columns]].rename(
                columns={"_League": "League"}
            ).sort_values(score_col, ascending=False).head(top_n)
            ws = wb.create_sheet(f"{base_name} - {tab_suffix}")
            scope = ("senior first-team players only" if tab_suffix == "Senior"
                     else "youth academy + reserve/B-team players only")
            write_data_sheet(
                ws,
                f"{title}, AGE ≤ {max_age}, MKT VAL < €{max_value:,.0f} — {tab_suffix.upper()}",
                f"{region_label}  ·  Top {len(out)} ({scope})  ·  "
                f"{blurb}  ·  Ranked by {score_col}",
                out,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


REGION_LABELS = {
    "cz_sk": "Czech Republic + Slovakia (by nationality)",
    "other": "Baltics + Scandinavia + Finland + Iceland + Poland + Slovenia leagues "
             "(by league played in, any nationality)",
}


def write_csv(df: pd.DataFrame, display_cols: list[str], score_col: str,
              top_n: int, path: Path) -> None:
    df[[c for c in display_cols if c in df.columns]].rename(
        columns={"_League": "League"}
    ).sort_values(score_col, ascending=False).head(top_n).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-minutes", type=int, default=400)
    parser.add_argument("--max-age", type=int, default=23)
    parser.add_argument("--max-value", type=float, default=800_000)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    print("Loading Wyscout database (DM + CM positions, all leagues)…")
    pool = load_pool(args.min_minutes)
    print(f"  → {len(pool)} players in scoring pool "
          f"({(pool['_arch']=='SIX').sum()} Six, {(pool['_arch']=='EIGHT').sum()} Eight)")

    pool = score_archetype(pool, "SIX", SIX_BLUEPRINT, "Six Score")
    pool = score_archetype(pool, "EIGHT", EIGHT_BLUEPRINT, "Eight Score")

    six_pool = pool[pool["_arch"] == "SIX"]
    eight_pool = pool[pool["_arch"] == "EIGHT"]

    for region, (mode, values) in REGIONS.items():
        six_out = filter_region(six_pool, mode, values, args.max_age, args.max_value, "Six Score")
        eight_out = filter_region(eight_pool, mode, values, args.max_age, args.max_value, "Eight Score")

        output = OUTPUT_FILES[region]
        write_region_workbook(
            six_out, eight_out, REGION_LABELS[region], args.max_age, args.max_value,
            output, args.top_n,
        )

        for base_name, df, display_cols, score_col in (
            ("Number6", six_out, DISPLAY_COLS_SIX, "Six Score"),
            ("Number8", eight_out, DISPLAY_COLS_EIGHT, "Eight Score"),
        ):
            for tab_suffix, academy_flag in (("Senior", False), ("Academy", True)):
                sub = df[df["Academy"] == academy_flag] if "Academy" in df.columns else df
                csv_path = output.with_name(f"{output.stem}_{base_name}_{tab_suffix}.csv")
                write_csv(sub, display_cols, score_col, args.top_n, csv_path)

        print(f"\n{region}: {len(six_out)} No.6 / {len(eight_out)} No.8 candidates "
              f"(top {args.top_n} per tab written)")
        print(f"  Excel → {output}")


if __name__ == "__main__":
    main()
