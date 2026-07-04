"""
export_cz_sk_young_cms.py
──────────────────────────
Screens the full Wyscout database for young central midfielders playing
in the Czech or Slovak leagues.

Filters
───────
  Position        — central midfielder (Wyscout CMF/LCMF/RCMF, mapped to
                     the "CM" group used across this repo, see
                     WYSCOUT_POSITION_MAP in wyscout_model.py)
  League           — the player's team appears in one of the Czech/Slovak
                     Wyscout league files (--leagues), regardless of the
                     player's own passport — a "who plays in these leagues"
                     screen, not a "who holds these passports" screen.
  Age              — 2003-born or younger. Wyscout exports only carry a
                      point-in-time Age, not a birth date, so this is
                      approximated as Age <= 23 as of the export date.

Usage
─────
  python export_cz_sk_young_cms.py [--min-minutes 0] [--leagues Czech "Czech II" ...]
                                    [--output data/CZ_SK_Young_CMs.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
WYSCOUT_DIR = ROOT / "data" / "Wyscout DB"

DEFAULT_LEAGUES = ("Czech", "Czech II", "Czech U17", "Czech U19", "Slovakia", "Slovakia II")
CM_POSITIONS = {"CMF", "LCMF", "RCMF"}

COLUMNS = [
    "Player", "Team", "_League", "Age", "Position", "Passport country",
    "Foot", "Height", "Minutes played", "Matches played",
    "Goals", "Assists", "xG", "xA",
    "Progressive passes per 90", "Accurate progressive passes, %",
    "Key passes per 90", "Passes per 90", "Accurate passes, %",
    "Duels won, %", "Interceptions per 90",
]


def load_players(max_age: int, leagues: tuple[str, ...] = DEFAULT_LEAGUES) -> pd.DataFrame:
    frames = []
    files = sorted(WYSCOUT_DIR.glob("*.xlsx"))
    files = [f for f in files if f.stem in leagues]
    for path in files:
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "Position" not in df.columns:
            continue

        df = df.copy()
        df["_League"] = path.stem
        # First-listed position, matching the convention in wyscout_model.py
        first_pos = df["Position"].astype(str).str.split(",").str[0].str.strip()

        is_cm = first_pos.isin(CM_POSITIONS)
        age = pd.to_numeric(df.get("Age"), errors="coerce")
        is_young = age <= max_age

        frames.append(df.loc[is_cm & is_young])

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Player", "_League", "Team"])
    cols = [c for c in COLUMNS if c in out.columns]
    out = out[cols].sort_values(["Age", "Player"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-minutes", type=int, default=0)
    parser.add_argument("--max-age", type=int, default=23)
    parser.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "CZ_SK_Young_CMs.csv"
    )
    args = parser.parse_args()

    df = load_players(max_age=args.max_age, leagues=args.leagues)

    if args.min_minutes and "Minutes played" in df.columns:
        df = df.loc[
            pd.to_numeric(df["Minutes played"], errors="coerce").fillna(0)
            >= args.min_minutes
        ]

    df.to_csv(args.output, index=False)
    print(f"Found {len(df)} players. Written to {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
