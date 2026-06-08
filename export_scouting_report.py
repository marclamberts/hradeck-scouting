"""
export_scouting_report.py
─────────────────────────
Generates  data/FCHK Scouting Report.xlsx  with one sheet per analysis area.

Sheets produced
───────────────
  Anomaly Overview      — all IMPECT players with z-scores + type (per position)
  Hidden Gems           — low composite, high signal outliers
  Specialist Elite      — extreme in 1–2 metrics
  Multi-dimensional     — 4+ metrics above threshold
  Age-adjusted Gems     — ≤23 with elite signal
  Similarity — Cosine   — top-10 most similar players for each anomaly (cosine)
  Similarity — Pearson  — same, Pearson
  Similarity — Euclidean— same, Euclidean
  SP Anomalies          — Wyscout set-piece outliers
  SP Corner Taker       — top set-piece role leaders (one sheet per role)
  SP Dead Ball Spec.    — …
  SP Crossing Threat    — …
  SP Aerial Threat      — …
  SP Box Presence       — …
  SP Set Piece Blocker  — …

Usage
─────
  python export_scouting_report.py [--threshold 1.8] [--min-minutes 500]
                                   [--output data/custom_name.xlsx]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
WYSCOUT_DIR = DATA_DIR / "Wyscout DB"

RECRUITMENT_FILE = DATA_DIR / "FCHK Model V3 - Recruitment Scores.xlsx"
PLAYER_SCORES    = DATA_DIR / "FCHK Model V3 - Player Scores.xlsx"
PLAYER_STYLES    = DATA_DIR / "FCHK Model V3 - Player Styles.xlsx"

sys.path.insert(0, str(ROOT))
from scouting_model import (
    AnomalyEngine,
    SimilarityEngine,
    SetPieceAnalyzer,
    SET_PIECE_ALL,
)

# ── Column aliases (mirrors app.py normalize_model_columns) ───────────────────
COLUMN_ALIASES: dict[str, list[str]] = {
    "ScoringThreatScore":        ["ScoringThreatScore", "AttackingScore", "UnderlyingThreatScore"],
    "CreativeProgressionScore":  ["CreativeProgressionScore", "CreationScore"],
    "DefensiveDisruptionScore":  ["DefensiveDisruptionScore", "DefendingScore"],
    "PressingScore":             ["PressingScore", "PressTransitionScore"],
    "ExpectedThreatScore":       ["ExpectedThreatScore", "xThreatScore"],
    "ASA_GoalsAddedScore":       ["ASA_GoalsAddedScore", "GoalsAddedScore"],
    "PerformanceReliabilityScore": ["PerformanceReliabilityScore", "ReliabilityScore", "SampleConfidence"],
    "BallSecurityScore":         ["BallSecurityScore"],
    "DecisionScore":             ["DecisionScore"],
    "ValueRecruitmentScore":     ["ValueRecruitmentScore"],
    "CompositeRecruitmentScore": ["CompositeRecruitmentScore"],
    "BundleLabel":               ["BundleLabel", "LeagueLabel"],
}

SCORE_METRICS = [
    "DecisionScore", "ValueRecruitmentScore", "CompositeRecruitmentScore",
    "ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore",
    "PressingScore", "BallSecurityScore", "ExpectedThreatScore",
    "ASA_GoalsAddedScore", "AgeResaleScore", "PerformanceReliabilityScore",
]

SIMILARITY_FEATURES = [
    "ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore",
    "PressingScore", "BallSecurityScore", "ExpectedThreatScore",
    "ASA_GoalsAddedScore", "DecisionScore",
]


# ── Data loaders ───────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    text_cols = {"PlayerName", "TeamName", "PositionGroup", "LeagueLabel",
                 "CountryLabel", "TierLabel", "PrimaryPlayerStyle",
                 "SecondaryPlayerStyle", "ClosestArchetype"}
    for col in df.columns:
        if col not in text_cols:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
    return df


def _alias(df: pd.DataFrame) -> pd.DataFrame:
    extra: dict[str, pd.Series] = {}
    for target, candidates in COLUMN_ALIASES.items():
        if target not in df.columns:
            for c in candidates:
                if c in df.columns:
                    extra[target] = df[c]
                    break
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
    return df


def _first_position(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Position", "Pos"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(",").str[0].str.strip()
    return df


def load_impect(min_minutes: int = 500) -> pd.DataFrame:
    if not RECRUITMENT_FILE.exists():
        raise FileNotFoundError(f"Missing: {RECRUITMENT_FILE}")
    df = _alias(_clean(pd.read_excel(RECRUITMENT_FILE, sheet_name="Recruitment Scores")))
    for extra_file, sheet in [
        (PLAYER_SCORES, "Player Scores"),
        (PLAYER_STYLES, "Player Styles"),
    ]:
        if extra_file.exists():
            extra = _alias(_clean(pd.read_excel(extra_file, sheet_name=sheet)))
            keys  = [c for c in ["PlayerName", "TeamName", "PositionGroup"] if c in df.columns and c in extra.columns]
            add   = [c for c in extra.columns if c not in df.columns and c not in keys]
            if keys and add:
                df = df.merge(extra[keys + add].drop_duplicates(keys), on=keys, how="left")

    if "MinutesPlayed" in df.columns:
        df = df.loc[pd.to_numeric(df["MinutesPlayed"], errors="coerce").fillna(0) >= min_minutes]

    return df.reset_index(drop=True)


def load_wyscout(min_minutes: int = 500) -> pd.DataFrame:
    if not WYSCOUT_DIR.exists():
        print("  [warn] Wyscout DB directory not found — skipping set-piece sheets.")
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(WYSCOUT_DIR.glob("*.xlsx")):
        try:
            raw = pd.read_excel(path)
            raw = raw.copy()
            raw.columns = [str(c).strip() for c in raw.columns]
            raw = _first_position(raw)
            raw = pd.concat([raw, pd.Series(path.stem, index=raw.index, name="_League")], axis=1)
            frames.append(raw)
        except Exception as e:
            print(f"  [warn] Could not read {path.name}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    mins_col = next((c for c in ["Minutes played", "MinutesPlayed"] if c in df.columns), None)
    if mins_col:
        df = df.loc[pd.to_numeric(df[mins_col], errors="coerce").fillna(0) >= min_minutes]
    return df.reset_index(drop=True)


# ── Similarity summary ────────────────────────────────────────────────────────

def build_similarity_sheet(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    method: str,
    n: int = 10,
) -> pd.DataFrame:
    """For each anomaly player, find top-n similar players and return a flat table."""
    features = [f for f in SIMILARITY_FEATURES if f in df.columns]
    if not features:
        return pd.DataFrame()
    engine  = SimilarityEngine(features)
    rows: list[dict] = []
    for _, target in anomalies.iterrows():
        results = engine.find_similar(df, target, method=method, n=n, same_position=True)
        for rank, (_, sim_row) in enumerate(results.iterrows(), start=1):
            rows.append({
                "Target Player":    target.get("PlayerName", ""),
                "Target Team":      target.get("TeamName", ""),
                "Target Position":  target.get("PositionGroup", ""),
                "Rank":             rank,
                "Similar Player":   sim_row.get("PlayerName", ""),
                "Similar Team":     sim_row.get("TeamName", ""),
                "Similar League":   sim_row.get("BundleLabel", ""),
                "Similar Age":      sim_row.get("AgeYears", ""),
                "Similarity Score": round(float(sim_row.get("_similarity", 0)), 4),
            })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(threshold: float, min_minutes: int, output: Path) -> None:
    print(f"Loading IMPECT data (min {min_minutes} min)…")
    impect = load_impect(min_minutes)
    print(f"  → {len(impect):,} players")

    # ── IMPECT anomaly analysis ───────────────────────────────────────────────
    print("Running anomaly engine…")
    metrics = [m for m in SCORE_METRICS if m in impect.columns]
    engine  = AnomalyEngine(threshold=threshold, method="z-score", groupby="PositionGroup")
    zdf     = engine.fit_transform(impect, metrics)

    anomalies     = engine.filter_anomalies(zdf, top_n=500)
    hidden_gems   = engine.filter_anomalies(zdf, types=["Hidden Gem"],            top_n=200)
    specialists   = engine.filter_anomalies(zdf, types=["Specialist Elite"],       top_n=200)
    multidim      = engine.filter_anomalies(zdf, types=["Multi-dimensional"],      top_n=200)
    age_gems      = engine.filter_anomalies(zdf, types=["Age-adjusted Gem"],       top_n=200)
    consistent    = engine.filter_anomalies(zdf, types=["Consistent Overperformer"], top_n=200)

    # Clean display columns (drop internal _z_ columns from overview; keep in full sheet)
    DISPLAY_BASE = ["PlayerName", "TeamName", "PositionGroup", "BundleLabel", "AgeYears",
                    "MinutesPlayed", "_anomaly_type", "_anomaly_score", "_peak_z",
                    "_mean_z", "_anomaly_breadth"]
    z_cols       = [f"_z_{m}" for m in metrics]

    def _display(df: pd.DataFrame, include_z: bool = False) -> pd.DataFrame:
        keep = [c for c in DISPLAY_BASE + (z_cols if include_z else []) if c in df.columns]
        return (
            df[keep]
            .rename(columns={
                "PlayerName": "Player", "TeamName": "Team", "PositionGroup": "Position",
                "BundleLabel": "League", "AgeYears": "Age",
                "_anomaly_type": "Anomaly Type", "_anomaly_score": "Anomaly Score",
                "_peak_z": "Peak Z", "_mean_z": "Mean Z", "_anomaly_breadth": "Metric Breadth",
            })
            .round(3)
            .reset_index(drop=True)
        )

    # ── Similarity sheets ─────────────────────────────────────────────────────
    print("Computing similarity matrices…")
    sim_cosine    = build_similarity_sheet(zdf, anomalies.head(100), "cosine")
    sim_pearson   = build_similarity_sheet(zdf, anomalies.head(100), "pearson")
    sim_euclidean = build_similarity_sheet(zdf, anomalies.head(100), "euclidean")

    # ── Wyscout set-piece analysis ────────────────────────────────────────────
    print("Loading Wyscout data for set-piece analysis…")
    wyscout = load_wyscout(min_minutes)
    sp_sheets: dict[str, pd.DataFrame] = {}

    if not wyscout.empty:
        print(f"  → {len(wyscout):,} players across {wyscout['_League'].nunique()} leagues")
        print("Running set-piece analyzer…")
        sp_analyzer = SetPieceAnalyzer(threshold=threshold * 0.85)
        ws_enriched = sp_analyzer.fit_transform(wyscout)
        sp_anomaly  = sp_analyzer.anomaly_table(ws_enriched, top_n=300)
        role_leaders = sp_analyzer.top_players_by_role(ws_enriched, top_n=30)

        sp_sheets["SP Anomalies"] = sp_anomaly
        for role, df_role in role_leaders.items():
            sheet_name = f"SP {role}"[:31]
            sp_sheets[sheet_name] = df_role.round(3).reset_index(drop=True)
    else:
        print("  [skip] No Wyscout data found.")

    # ── Assemble workbook ─────────────────────────────────────────────────────
    sheets: dict[str, pd.DataFrame] = {
        "Anomaly Overview":       _display(zdf.sort_values("_anomaly_score", ascending=False), include_z=True),
        "Hidden Gems":            _display(hidden_gems),
        "Specialist Elite":       _display(specialists),
        "Multi-dimensional":      _display(multidim),
        "Age-adjusted Gems":      _display(age_gems),
        "Consistent Overperform": _display(consistent),
        "Similarity — Cosine":    sim_cosine,
        "Similarity — Pearson":   sim_pearson,
        "Similarity — Euclidean": sim_euclidean,
        **sp_sheets,
    }

    print(f"\nWriting {len(sheets)} sheets → {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            if df_sheet.empty:
                print(f"  [skip] {sheet_name!r} — no data")
                continue
            df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            # Auto-fit column widths
            ws = writer.sheets[sheet_name[:31]]
            for col_cells in ws.columns:
                max_len = max(
                    len(str(col_cells[0].value or "")),
                    *(len(str(c.value or "")) for c in col_cells[1:]),
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

            print(f"  ✓ {sheet_name!r} — {len(df_sheet):,} rows")

    print(f"\nDone. Report saved to:\n  {output.resolve()}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FCHK scouting Excel report")
    parser.add_argument("--threshold",   type=float, default=1.8,   help="Z-score anomaly threshold (default 1.8)")
    parser.add_argument("--min-minutes", type=int,   default=500,   help="Minimum minutes played (default 500)")
    parser.add_argument("--output",      type=Path,
                        default=DATA_DIR / "FCHK Scouting Report.xlsx",
                        help="Output .xlsx path")
    args = parser.parse_args()
    run(threshold=args.threshold, min_minutes=args.min_minutes, output=args.output)
