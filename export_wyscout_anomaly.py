"""
export_wyscout_anomaly.py
─────────────────────────
Wyscout-only anomaly scouting report.

Loads all Wyscout DB files, runs z-score anomaly detection using
Wyscout per-90 metrics (grouped by mapped position), and writes
a multi-sheet Excel report.

Sheets
──────
  All Players           — full Wyscout table with z-scores + SP tags
  Anomaly Overview      — all anomalies ranked by score
  Hidden Gems           — low appearances, high signal
  Specialist Elite      — elite in 1–2 metrics
  Multi-dimensional     — 4+ metrics above threshold
  Age-adjusted Gems     — ≤23 anomalies
  Consistent Overperf.  — broad positive signal
  Similarity — Cosine   — top-10 similar for each anomaly
  Similarity — Pearson
  Similarity — Euclidean
  SP Anomalies          — set-piece outliers
  SP Corner Taker       — top players per SP role
  SP Dead Ball Spec.
  SP Crossing Threat
  SP Aerial Threat
  SP Box Presence
  SP Set Piece Blocker

Usage
─────
  python export_wyscout_anomaly.py [--threshold 1.8] [--min-minutes 400]
                                   [--output data/Wyscout Anomaly Report.xlsx]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
WYSCOUT_DIR = DATA_DIR / "Wyscout DB"

sys.path.insert(0, str(ROOT))
from scouting_model import (
    AnomalyEngine,
    SimilarityEngine,
    SetPieceAnalyzer,
    SET_PIECE_ROLES,
)

# ── Position mapping (raw Wyscout → 8 groups) ─────────────────────────────────
POSITION_MAP: dict[str, str] = {
    "CF": "ST", "SS": "ST",
    "LW": "W",  "RW": "W",  "LWF": "W",  "RWF": "W",  "WF": "W",
    "AMF": "AM", "LAMF": "AM", "RAMF": "AM",
    "CMF": "CM", "LCM": "CM",  "RCM": "CM",  "LCMF": "CM", "RCMF": "CM",
    "DMF": "DM", "LDM": "DM",  "RDM": "DM",  "LDMF": "DM", "RDMF": "DM",
    "LB": "FB",  "RB": "FB",   "LWB": "FB",  "RWB": "FB",
    "CB": "CB",  "LCB": "CB",  "RCB": "CB",
    "GK": "GK",
}

# ── Per-position anomaly metrics ───────────────────────────────────────────────
POSITION_METRICS: dict[str, list[str]] = {
    "ST": [
        "Goals per 90", "Non-penalty goals per 90", "xG per 90",
        "Shots per 90", "Shots on target, %", "Goal conversion, %",
        "Touches in box per 90", "Aerial duels won, %",
        "Dribbles per 90", "Successful dribbles, %",
        "Progressive runs per 90", "xA per 90",
    ],
    "W": [
        "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
        "Key passes per 90", "Dribbles per 90", "Successful dribbles, %",
        "Crosses per 90", "Accurate crosses, %",
        "Progressive runs per 90", "Touches in box per 90",
        "Successful defensive actions per 90",
    ],
    "AM": [
        "Key passes per 90", "xA per 90", "Assists per 90",
        "Smart passes per 90", "Accurate smart passes, %",
        "Goals per 90", "xG per 90", "Touches in box per 90",
        "Dribbles per 90", "Progressive passes per 90",
        "Through passes per 90", "Successful dribbles, %",
    ],
    "CM": [
        "Passes per 90", "Accurate passes, %",
        "Forward passes per 90", "Accurate forward passes, %",
        "Progressive passes per 90", "Key passes per 90",
        "xA per 90", "Progressive runs per 90",
        "Successful defensive actions per 90", "Defensive duels won, %",
        "Interceptions per 90", "Duels won, %",
    ],
    "DM": [
        "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Interceptions per 90",
        "PAdj Interceptions", "Aerial duels won, %", "Duels won, %",
        "Passes per 90", "Accurate passes, %",
        "Progressive passes per 90", "Key passes per 90", "Fouls per 90",
    ],
    "FB": [
        "Crosses per 90", "Accurate crosses, %",
        "xA per 90", "Assists per 90", "Key passes per 90",
        "Progressive runs per 90", "Dribbles per 90",
        "Successful defensive actions per 90", "Defensive duels won, %",
        "Aerial duels won, %", "Accurate passes, %",
        "Progressive passes per 90",
    ],
    "CB": [
        "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
        "Interceptions per 90", "PAdj Interceptions", "Shots blocked per 90",
        "Fouls per 90", "Accurate passes, %",
        "Accurate forward passes, %", "Progressive passes per 90",
    ],
    "GK": [
        "Save rate, %", "Prevented goals per 90", "Conceded goals per 90",
        "Shots against per 90", "Clean sheets",
        "Exits per 90", "Aerial duels per 90.1",
        "Accurate passes, %", "Accurate long passes, %",
        "Back passes received as GK per 90",
    ],
}

# Shared similarity features (available across most positions)
SIMILARITY_FEATURES = [
    "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
    "Key passes per 90", "Passes per 90", "Accurate passes, %",
    "Dribbles per 90", "Successful dribbles, %",
    "Successful defensive actions per 90", "Defensive duels won, %",
    "Aerial duels won, %", "Progressive passes per 90",
    "Progressive runs per 90", "Touches in box per 90",
]

# Columns included in display sheets
DISPLAY_BASE = [
    "Player", "Team", "Position", "PositionGroup", "Age", "_League",
    "Minutes played", "Matches played",
    "_anomaly_type", "_anomaly_score", "_peak_z", "_mean_z", "_anomaly_breadth",
]

# Key stat columns to include alongside anomaly scores
KEY_STATS = [
    "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
    "Key passes per 90", "Passes per 90", "Accurate passes, %",
    "Dribbles per 90", "Successful defensive actions per 90",
    "Aerial duels won, %", "Progressive passes per 90",
]


# ── Data loaders ───────────────────────────────────────────────────────────────

def _first_position(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Position", "Pos"]:
        if col in df.columns:
            df = df.copy()
            df[col] = df[col].astype(str).str.split(",").str[0].str.strip()
    return df


def load_wyscout(min_minutes: int) -> pd.DataFrame:
    if not WYSCOUT_DIR.exists():
        raise FileNotFoundError(f"Wyscout DB directory not found: {WYSCOUT_DIR}")
    frames: list[pd.DataFrame] = []
    files = sorted(WYSCOUT_DIR.glob("*.xlsx"))
    print(f"  Reading {len(files)} Wyscout files…")
    for path in files:
        try:
            raw = pd.read_excel(path)
            raw.columns = [str(c).strip() for c in raw.columns]
            raw = _first_position(raw)
            raw = pd.concat(
                [raw, pd.Series(path.stem, index=raw.index, name="_League")], axis=1
            )
            frames.append(raw)
        except Exception as e:
            print(f"    [warn] {path.name}: {e}")

    if not frames:
        raise RuntimeError("No Wyscout files could be loaded.")

    df = pd.concat(frames, ignore_index=True)

    # Alias Age → AgeYears so AnomalyEngine classification works
    if "Age" in df.columns and "AgeYears" not in df.columns:
        df["AgeYears"] = pd.to_numeric(df["Age"], errors="coerce")

    # For Hidden Gem classification: proxy CompositeRecruitmentScore with
    # normalised minutes played (fewer minutes = more "hidden")
    mins_col = next((c for c in ["Minutes played", "MinutesPlayed"] if c in df.columns), None)
    if mins_col and "CompositeRecruitmentScore" not in df.columns:
        mins = pd.to_numeric(df[mins_col], errors="coerce").fillna(0)
        # Scale 0-100: fewer minutes = lower composite → triggers Hidden Gem flag
        df["CompositeRecruitmentScore"] = (
            (mins - mins.min()) / (mins.max() - mins.min() + 1e-9) * 100
        ).clip(0, 100)

    # Map raw position → position group
    pos_col = next((c for c in ["Position", "Pos"] if c in df.columns), None)
    if pos_col:
        df["PositionGroup"] = df[pos_col].map(POSITION_MAP).fillna("Other")

    # Filter by minutes
    mins_col = next((c for c in ["Minutes played", "MinutesPlayed"] if c in df.columns), None)
    if mins_col:
        df = df.loc[pd.to_numeric(df[mins_col], errors="coerce").fillna(0) >= min_minutes]

    # Numeric coercion for stat columns
    skip_text = {"Player", "Team", "Position", "PositionGroup", "_League",
                 "Birth country", "Passport country", "Foot", "On loan",
                 "Team within selected timeframe"}
    for col in df.columns:
        if col not in skip_text:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted

    return df.reset_index(drop=True)


# ── Anomaly engine — per position group ───────────────────────────────────────

def run_anomaly(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Run AnomalyEngine on each position group using position-specific metrics.
    Returns the full DataFrame enriched with _z_*, _anomaly_* columns.
    """
    frames: list[pd.DataFrame] = []
    for pos_group, grp in df.groupby("PositionGroup"):
        metrics = [m for m in POSITION_METRICS.get(str(pos_group), []) if m in grp.columns]
        if not metrics or len(grp) < 5:
            frames.append(grp)
            continue
        engine = AnomalyEngine(threshold=threshold, method="z-score", groupby=None)
        try:
            enriched = engine.fit_transform(grp, metrics)
            frames.append(enriched)
        except Exception as e:
            print(f"    [warn] Anomaly engine failed for {pos_group}: {e}")
            frames.append(grp)

    return pd.concat(frames, ignore_index=True)


# ── Similarity summary ─────────────────────────────────────────────────────────

def build_similarity_sheet(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    method: str,
    n: int = 10,
) -> pd.DataFrame:
    feats  = [f for f in SIMILARITY_FEATURES if f in df.columns]
    engine = SimilarityEngine(feats)
    rows: list[dict] = []
    for _, target in anomalies.iterrows():
        results = engine.find_similar(df, target, method=method, n=n, same_position=True)
        for rank, (_, sim_row) in enumerate(results.iterrows(), start=1):
            rows.append({
                "Target Player":   target.get("Player", ""),
                "Target Team":     target.get("Team", ""),
                "Target Position": target.get("PositionGroup", ""),
                "Target League":   target.get("_League", ""),
                "Rank":            rank,
                "Similar Player":  sim_row.get("Player", ""),
                "Similar Team":    sim_row.get("Team", ""),
                "Similar League":  sim_row.get("_League", ""),
                "Similar Age":     sim_row.get("Age", ""),
                "Similarity":      round(float(sim_row.get("_similarity", 0)), 4),
            })
    return pd.DataFrame(rows)


# ── Display helpers ────────────────────────────────────────────────────────────

def _display(df: pd.DataFrame, z_cols: list[str] | None = None) -> pd.DataFrame:
    keep = [c for c in DISPLAY_BASE + KEY_STATS + (z_cols or []) if c in df.columns]
    out  = df[keep].copy()
    out  = out.rename(columns={
        "_League": "League",
        "_anomaly_type": "Anomaly Type",
        "_anomaly_score": "Anomaly Score",
        "_peak_z": "Peak Z",
        "_mean_z": "Mean Z",
        "_anomaly_breadth": "Metric Breadth",
    })
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(3)
    return out.reset_index(drop=True)


def _autofit(ws) -> None:
    for col_cells in ws.columns:
        max_len = max(
            len(str(col_cells[0].value or "")),
            *(len(str(c.value or "")) for c in col_cells[1:6]),  # sample first 6 rows for speed
        )
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 42)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(threshold: float, min_minutes: int, output: Path) -> None:
    print(f"\nLoading Wyscout data (min {min_minutes} min)…")
    df = load_wyscout(min_minutes)
    print(f"  → {len(df):,} players | {df['_League'].nunique()} leagues | {df['PositionGroup'].nunique()} position groups")

    # ── Anomaly detection ──────────────────────────────────────────────────────
    print("Running position-specific anomaly detection…")
    zdf = run_anomaly(df, threshold)

    # Pull out each type
    if "_peak_z" not in zdf.columns:
        print("  [warn] No z-score columns produced — check metric availability.")
        anomalies = pd.DataFrame()
    else:
        engine    = AnomalyEngine(threshold=threshold)
        anomalies = zdf.loc[zdf["_peak_z"] >= threshold].sort_values("_anomaly_score", ascending=False)

    z_cols = [c for c in zdf.columns if c.startswith("_z_")]

    def _type(t: str) -> pd.DataFrame:
        if anomalies.empty or "_anomaly_type" not in anomalies.columns:
            return pd.DataFrame()
        return anomalies.loc[anomalies["_anomaly_type"] == t].head(300)

    # ── Similarity sheets ──────────────────────────────────────────────────────
    print("Computing similarity matrices…")
    top_anom = anomalies.head(150)
    sim_cosine    = build_similarity_sheet(zdf, top_anom, "cosine")
    sim_pearson   = build_similarity_sheet(zdf, top_anom, "pearson")
    sim_euclidean = build_similarity_sheet(zdf, top_anom, "euclidean")

    # ── Set-piece analysis ─────────────────────────────────────────────────────
    print("Running set-piece analyzer…")
    sp_analyzer = SetPieceAnalyzer(threshold=threshold * 0.85)
    ws_enriched = sp_analyzer.fit_transform(zdf)
    sp_anomaly  = sp_analyzer.anomaly_table(ws_enriched, top_n=300)
    role_leaders = sp_analyzer.top_players_by_role(ws_enriched, top_n=30)

    # ── All Players sheet ──────────────────────────────────────────────────────
    ALL_COLS = [
        "Player", "Team", "Position", "PositionGroup", "Age", "_League",
        "Minutes played", "Matches played",
        "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
        "Shots per 90", "Shots on target, %", "Key passes per 90",
        "Passes per 90", "Accurate passes, %", "Forward passes per 90",
        "Progressive passes per 90", "Progressive runs per 90",
        "Dribbles per 90", "Successful dribbles, %",
        "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
        "Interceptions per 90", "PAdj Interceptions", "Shots blocked per 90",
        "Crosses per 90", "Accurate crosses, %",
        "Deep completed crosses per 90", "Corners per 90",
        "Free kicks per 90", "Direct free kicks per 90",
        "Direct free kicks on target, %", "Head goals per 90",
        "Touches in box per 90", "Duels won, %", "Fouls per 90",
        # SP columns
        "_sp_primary_role", "_sp_composite", "_sp_peak_z",
        # Anomaly columns
        "_anomaly_type", "_anomaly_score", "_peak_z",
    ]
    keep_all = [c for c in ALL_COLS if c in ws_enriched.columns]
    all_players = (
        ws_enriched[keep_all]
        .rename(columns={
            "_League": "League", "_sp_primary_role": "SP Role",
            "_sp_composite": "SP Score", "_sp_peak_z": "SP Peak Z",
            "_anomaly_type": "Anomaly Type", "_anomaly_score": "Anomaly Score",
            "_peak_z": "Peak Z",
        })
        .sort_values("Player" if "Player" in keep_all else keep_all[0])
        .round(3)
        .reset_index(drop=True)
    )

    # ── Assemble workbook ──────────────────────────────────────────────────────
    sheets: dict[str, pd.DataFrame] = {
        "All Players":            all_players,
        "Anomaly Overview":       _display(anomalies.head(500), z_cols),
        "Hidden Gems":            _display(_type("Hidden Gem")),
        "Specialist Elite":       _display(_type("Specialist Elite")),
        "Multi-dimensional":      _display(_type("Multi-dimensional")),
        "Age-adjusted Gems":      _display(_type("Age-adjusted Gem")),
        "Consistent Overperf":    _display(_type("Consistent Overperformer")),
        "Similarity — Cosine":    sim_cosine,
        "Similarity — Pearson":   sim_pearson,
        "Similarity — Euclidean": sim_euclidean,
        "SP Anomalies":           sp_anomaly,
    }
    for role, df_role in role_leaders.items():
        sheets[f"SP {role}"[:31]] = df_role.round(3).reset_index(drop=True)

    print(f"\nWriting {len(sheets)} sheets → {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            if df_sheet is None or df_sheet.empty:
                print(f"  [skip] {sheet_name!r} — no data")
                continue
            df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            _autofit(writer.sheets[sheet_name[:31]])
            print(f"  ✓ {sheet_name!r} — {len(df_sheet):,} rows")

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {size_mb:.1f} MB → {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wyscout-only anomaly scouting report")
    parser.add_argument("--threshold",   type=float, default=1.8,
                        help="Z-score anomaly threshold (default 1.8)")
    parser.add_argument("--min-minutes", type=int,   default=400,
                        help="Minimum minutes played (default 400)")
    parser.add_argument("--output",      type=Path,
                        default=DATA_DIR / "Wyscout Anomaly Report.xlsx",
                        help="Output .xlsx path")
    args = parser.parse_args()
    run(threshold=args.threshold, min_minutes=args.min_minutes, output=args.output)
