"""
build_czech_set_pieces.py
─────────────────────────
Set-piece score workbook for the Czech top flight (Wyscout Files/Czech.xlsx).

Sheets produced
───────────────
  All Players          — full player table with SetPieceScore + role scores
  SP Output            — delivery volume: corners, free kicks, crosses per 90
  SP Quality           — accuracy and efficiency from set-piece delivery
  SP Anomalies         — statistical outliers across set-piece metrics
  SP Corner Taker      — top corner-kick deliverers
  SP Dead Ball Spec.   — top direct free-kick takers
  SP Crossing Threat   — top crossers
  SP Aerial Threat     — top aerial duel winners / headers
  SP Box Presence      — top box-threat receivers
  SP Set Piece Blocker — top defensive headers / blockers

Usage
─────
  python build_czech_set_pieces.py [--min-minutes 400]
                                   [--output "data/Czech Set Piece Scores.xlsx"]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scouting_model import SetPieceAnalyzer, SET_PIECE_ROLES
from wyscout_model import WYSCOUT_POSITION_MAP, SCORE_BLUEPRINTS

# ── Position mapping (mirrors build_lamberts_total.py) ────────────────────────

POS_MAP: dict[str, str] = WYSCOUT_POSITION_MAP

# ── Set-piece metric groups ────────────────────────────────────────────────────

SP_SCORE_BLUEPRINT = SCORE_BLUEPRINTS["SetPieceScore"]

SP_OUTPUT_COLS = [
    "Corners per 90",
    "Free kicks per 90",
    "Direct free kicks per 90",
    "Crosses per 90",
    "Deep completed crosses per 90",
]

SP_QUALITY_COLS = [
    "Direct free kicks on target, %",
    "Accurate crosses, %",
    "Aerial duels won, %",
    "Head goals per 90",
    "Aerial duels per 90",
]

SP_ALL_METRICS = list(dict.fromkeys(
    SP_OUTPUT_COLS + SP_QUALITY_COLS + [
        "Key passes per 90",
        "xG per 90",
        "Shots per 90",
        "Shots on target, %",
        "Goal conversion, %",
        "Touches in box per 90",
        "Shots blocked per 90",
    ]
))

BIO_COLS = [
    "Player", "Team", "Position", "PositionGroup", "Age",
    "Minutes played", "Matches played", "Height", "Foot",
    "Birth country", "Passport country", "Contract expires", "Market value",
]

# ── Scoring helpers ────────────────────────────────────────────────────────────

def _z_to_pct(z: np.ndarray) -> np.ndarray:
    return norm.cdf(z) * 100


def compute_set_piece_score(df: pd.DataFrame) -> pd.Series:
    """Weighted z-score SetPieceScore (0–100 percentile) across all players."""
    available = [(m, w) for m, w in SP_SCORE_BLUEPRINT if m in df.columns]
    if not available:
        return pd.Series(50.0, index=df.index)

    total_w = sum(w for _, w in available)
    z_composite = pd.Series(0.0, index=df.index)
    for metric, weight in available:
        col = pd.to_numeric(df[metric], errors="coerce").fillna(0)
        mu = col.mean()
        sig = col.std() or 1e-9
        z_composite += (weight / total_w) * (col - mu) / sig

    return pd.Series(_z_to_pct(z_composite.values), index=df.index).round(1)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_czech(min_minutes: int) -> pd.DataFrame:
    path = ROOT / "Wyscout Files" / "Czech.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Take first listed position only
    for col in ["Position", "Pos"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(",").str[0].str.strip()

    # Map position → group
    pos_col = next((c for c in ["Position", "Pos"] if c in df.columns), None)
    if pos_col:
        df["PositionGroup"] = df[pos_col].map(POS_MAP).fillna("Other")

    # Coerce numerics
    skip = {
        "Player", "Team", "Position", "PositionGroup",
        "Birth country", "Passport country", "Foot", "On loan",
        "Team within selected timeframe", "Contract expires",
    }
    for col in df.columns:
        if col not in skip:
            c = pd.to_numeric(df[col], errors="coerce")
            if c.notna().any():
                df[col] = c

    # Minutes filter
    mins_col = next((c for c in ["Minutes played", "MinutesPlayed"] if c in df.columns), None)
    if mins_col and min_minutes > 0:
        df = df.loc[pd.to_numeric(df[mins_col], errors="coerce").fillna(0) >= min_minutes]

    df["_League"] = "Czech"
    return df.reset_index(drop=True)


# ── Sheet builders ─────────────────────────────────────────────────────────────

def build_all_players(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # SetPieceScore from blueprint
    out["SetPieceScore"] = compute_set_piece_score(out)

    # Role scores from SetPieceAnalyzer (internal _sp_role_* columns)
    keep_bio = [c for c in BIO_COLS if c in out.columns]
    role_cols = [f"_sp_role_{r}" for r in SET_PIECE_ROLES if f"_sp_role_{r}" in out.columns]
    sp_cols = ["SetPieceScore", "_sp_primary_role", "_sp_composite"] + role_cols

    stat_display = [
        "Corners per 90", "Free kicks per 90", "Direct free kicks per 90",
        "Direct free kicks on target, %", "Crosses per 90", "Accurate crosses, %",
        "Deep completed crosses per 90", "Aerial duels per 90", "Aerial duels won, %",
        "Head goals per 90", "Shots blocked per 90",
    ]

    keep = keep_bio + [c for c in sp_cols + stat_display if c in out.columns and c not in keep_bio]
    return (
        out[keep]
        .rename(columns={
            "_sp_primary_role": "SP Primary Role",
            "_sp_composite":    "SP Composite",
            **{f"_sp_role_{r}": r for r in SET_PIECE_ROLES},
        })
        .sort_values("SetPieceScore", ascending=False)
        .round(2)
        .reset_index(drop=True)
    )


def build_sp_output(df: pd.DataFrame) -> pd.DataFrame:
    keep_bio = [c for c in ["Player", "Team", "Position", "PositionGroup", "Age",
                             "Minutes played"] if c in df.columns]
    out = df[keep_bio].copy()

    total = pd.Series(0.0, index=df.index)
    for c in SP_OUTPUT_COLS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0)
            out[c] = v.round(2)
            total += v
    out["Total SP Delivery per 90"] = total.round(2)

    if "SetPieceScore" in df.columns:
        out["SetPieceScore"] = pd.to_numeric(df["SetPieceScore"], errors="coerce").round(1)

    return out.sort_values("Total SP Delivery per 90", ascending=False).reset_index(drop=True)


def build_sp_quality(df: pd.DataFrame) -> pd.DataFrame:
    keep_bio = [c for c in ["Player", "Team", "Position", "PositionGroup", "Age",
                             "Minutes played"] if c in df.columns]
    out = df[keep_bio].copy()

    for c in SP_QUALITY_COLS:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    # Composite quality index
    quality_weights = {
        "Direct free kicks on target, %": 2.5,
        "Accurate crosses, %":            2.0,
        "Aerial duels won, %":            1.5,
        "Head goals per 90":              2.0,
    }
    score = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for c, w in quality_weights.items():
        if c in df.columns:
            score += w * pd.to_numeric(df[c], errors="coerce").fillna(0)
            total_w += w
    out["SP Quality Index"] = (score / (total_w or 1)).round(2)

    if "xG per 90" in df.columns and "Goals per 90" in df.columns:
        xg  = pd.to_numeric(df["xG per 90"], errors="coerce").fillna(0)
        gls = pd.to_numeric(df["Goals per 90"], errors="coerce").fillna(0)
        out["Goals vs xG Overperformance"] = (gls - xg).round(3)

    return out.sort_values("SP Quality Index", ascending=False).reset_index(drop=True)


def build_sp_anomalies(enriched: pd.DataFrame) -> pd.DataFrame:
    if "_sp_peak_z" not in enriched.columns:
        return pd.DataFrame()

    name_col = next((c for c in ["Player", "PlayerName"] if c in enriched.columns), None)
    keep = [c for c in [
        name_col, "Team", "Position", "Age", "Minutes played",
        "_sp_primary_role", "_sp_peak_z", "_sp_breadth", "_sp_composite",
        "SetPieceScore",
        "Corners per 90", "Direct free kicks per 90", "Direct free kicks on target, %",
        "Crosses per 90", "Accurate crosses, %", "Deep completed crosses per 90",
        "Aerial duels per 90", "Aerial duels won, %", "Head goals per 90",
    ] if c and c in enriched.columns]

    return (
        enriched.loc[enriched["_sp_is_anomaly"]]
        .sort_values("_sp_peak_z", ascending=False)
        [keep]
        .rename(columns={
            name_col: "Player" if name_col else "Player",
            "_sp_primary_role": "Primary Role",
            "_sp_peak_z":       "Peak Z",
            "_sp_breadth":      "Metric Breadth",
            "_sp_composite":    "SP Composite",
        })
        .round(3)
        .reset_index(drop=True)
    )


def build_role_sheet(enriched: pd.DataFrame, role: str, top_n: int = 30) -> pd.DataFrame:
    col = f"_sp_role_{role}"
    if col not in enriched.columns:
        return pd.DataFrame()

    name_col = next((c for c in ["Player", "PlayerName"] if c in enriched.columns), None)
    keep = [c for c in [
        name_col, "Team", "Position", "Age", "Minutes played",
        col, "_sp_composite", "SetPieceScore",
        "Corners per 90", "Free kicks per 90", "Direct free kicks per 90",
        "Direct free kicks on target, %", "Crosses per 90", "Accurate crosses, %",
        "Deep completed crosses per 90", "Aerial duels per 90", "Aerial duels won, %",
        "Head goals per 90", "Shots blocked per 90",
    ] if c and c in enriched.columns]

    return (
        enriched[keep]
        .sort_values(col, ascending=False)
        .head(top_n)
        .rename(columns={
            col:             "Role Score",
            "_sp_composite": "SP Composite",
        })
        .round(3)
        .reset_index(drop=True)
    )


# ── Excel helpers ──────────────────────────────────────────────────────────────

def _autofit(ws) -> None:
    for col_cells in ws.columns:
        try:
            max_len = max(
                len(str(col_cells[0].value or "")),
                *(len(str(c.value or "")) for c in col_cells[1:6]),
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 44)
        except Exception:
            pass


def _write_sheet(writer, name: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print(f"  [skip] {name!r} — no data")
        return
    sname = name[:31]
    df.to_excel(writer, sheet_name=sname, index=False)
    _autofit(writer.sheets[sname])
    print(f"  ✓ {name!r} — {len(df):,} rows")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(min_minutes: int, output: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  Czech Set Piece Score Builder")
    print(f"  Min minutes: {min_minutes}  |  Source: Wyscout Files/Czech.xlsx")
    print(f"{'='*60}")

    # 1. Load
    print("Loading Czech.xlsx…")
    df = load_czech(min_minutes)
    print(f"  → {len(df):,} players after min-minutes filter")

    # 2. Compute SetPieceScore
    print("Computing SetPieceScore…")
    df["SetPieceScore"] = compute_set_piece_score(df)

    # 3. Run SetPieceAnalyzer (role scores + anomaly flags)
    print("Running SetPieceAnalyzer…")
    analyzer = SetPieceAnalyzer(threshold=1.5)
    enriched = analyzer.fit_transform(df)

    # 4. Build sheets
    print("Building sheets…")
    all_players  = build_all_players(enriched)
    sp_output    = build_sp_output(enriched)
    sp_quality   = build_sp_quality(enriched)
    sp_anomalies = build_sp_anomalies(enriched)

    # 5. Write workbook
    print(f"\nWriting workbook → {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _write_sheet(writer, "All Players",         all_players)
        _write_sheet(writer, "SP Output",           sp_output)
        _write_sheet(writer, "SP Quality",          sp_quality)
        _write_sheet(writer, "SP Anomalies",        sp_anomalies)
        for role in SET_PIECE_ROLES:
            sheet = build_role_sheet(enriched, role, top_n=30)
            _write_sheet(writer, f"SP {role}"[:31], sheet)

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {size_mb:.1f} MB → {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Czech set-piece score workbook")
    parser.add_argument("--min-minutes", type=int,  default=400)
    parser.add_argument("--output",      type=Path,
                        default=ROOT / "data" / "Czech Set Piece Scores.xlsx")
    args = parser.parse_args()
    run(min_minutes=args.min_minutes, output=args.output)
