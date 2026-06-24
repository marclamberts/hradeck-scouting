"""
export_wyscout_dm.py
────────────────────
Defensive Midfielder positional scouting report (Wyscout data).

Produces a focused Excel workbook covering only the DM position group
(DMF, LDMF, RDMF, LDM, RDM mapped to "DM").

Sheets produced
───────────────
  DM Scouting Board      — ranked board with all DM-relevant metrics + scores
  Anomaly Overview        — DM anomalies ranked by score
  Hidden Gems             — high signal, low exposure
  Specialist Elite        — elite in 1–2 DM metrics
  Multi-dimensional       — 4+ metrics above threshold
  Age-adjusted Gems       — ≤23 anomalies
  Consistent Overperf     — broad positive signal
  Profiles                — full composite score card per DM
  Tiers                   — tiered recruitment board (T1–T4)
  Playing Styles          — DM-relevant style classification
  Similarity — Cosine     — top-10 comparables per anomaly
  Similarity — Pearson
  Similarity — Euclidean

Usage
─────
  python export_wyscout_dm.py [--threshold 1.8] [--min-minutes 400]
                              [--output "data/Wyscout files/Wyscout DM Scouting.xlsx"]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR  = DATA_DIR / "Wyscout files"

sys.path.insert(0, str(ROOT))
from scouting_model import AnomalyEngine, SimilarityEngine
from wyscout_model import compute_wyscout_scores, load_wyscout_raw, WYSCOUT_POSITION_MAP

# ── DM-specific configuration ──────────────────────────────────────────────────

DM_ANOMALY_METRICS = [
    "Successful defensive actions per 90",
    "Defensive duels per 90",
    "Defensive duels won, %",
    "Interceptions per 90",
    "PAdj Interceptions",
    "Aerial duels won, %",
    "Duels won, %",
    "Passes per 90",
    "Accurate passes, %",
    "Progressive passes per 90",
    "Key passes per 90",
    "Fouls per 90",
]

DM_DISPLAY_COLS = [
    "Player", "Team", "Position", "Age", "_League",
    "Minutes played", "Matches played", "Height", "Foot",
    "Contract expires", "Market value",
    # Defensive
    "Successful defensive actions per 90",
    "Defensive duels per 90",
    "Defensive duels won, %",
    "Interceptions per 90",
    "PAdj Interceptions",
    "Shots blocked per 90",
    "Duels per 90",
    "Duels won, %",
    "Aerial duels per 90",
    "Aerial duels won, %",
    "Fouls per 90",
    # Ball playing
    "Passes per 90",
    "Accurate passes, %",
    "Forward passes per 90",
    "Accurate forward passes, %",
    "Progressive passes per 90",
    "Key passes per 90",
    "xA per 90",
    "Smart passes per 90",
    # Pressing
    "Sliding tackles per 90",
    "Progressive runs per 90",
    # Scores
    "DefensiveDisruptionScore",
    "PressingScore",
    "BallSecurityScore",
    "CreativeProgressionScore",
    "AerialScore",
    "CompositeRecruitmentScore",
    # Anomaly tags
    "_anomaly_type", "_anomaly_score", "_peak_z",
]

DM_SIMILARITY_FEATURES = [
    "Successful defensive actions per 90",
    "Defensive duels per 90",
    "Defensive duels won, %",
    "Interceptions per 90",
    "Aerial duels won, %",
    "Duels won, %",
    "Passes per 90",
    "Accurate passes, %",
    "Progressive passes per 90",
    "Key passes per 90",
    "xA per 90",
]

DM_STYLE_WEIGHTS: dict[str, dict[str, float]] = {
    "Defensive Enforcer":    {"DefensiveDisruptionScore": 3.0, "PressingScore": 2.0},
    "Press Monster":         {"PressingScore": 3.0, "DefensiveDisruptionScore": 2.0},
    "Ball Playing Anchor":   {"BallSecurityScore": 3.0, "CreativeProgressionScore": 2.0, "PressingScore": 1.0},
    "Aerial Dominator":      {"AerialScore": 3.0, "DefensiveDisruptionScore": 1.5},
    "Deep Lying Playmaker":  {"CreativeProgressionScore": 3.0, "BallSecurityScore": 2.5, "DefensiveDisruptionScore": 1.0},
    "Complete Midfielder":   {
        "DefensiveDisruptionScore": 1.5, "PressingScore": 1.5,
        "BallSecurityScore": 1.5, "CreativeProgressionScore": 1.5,
    },
}

TIER_LABELS = {
    1: "T1 — Elite (90+)",
    2: "T2 — Strong (75–89)",
    3: "T3 — Promising (60–74)",
    4: "T4 — Below Average (<60)",
}

SCORE_COLS = [
    "ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore",
    "PressingScore", "BallSecurityScore", "ExpectedThreatScore",
    "ASA_GoalsAddedScore", "AerialScore", "SetPieceScore",
    "CompositeRecruitmentScore",
]

BIO_COLS = [
    "Player", "Team", "Position", "Age", "_League",
    "Minutes played", "Matches played", "Height", "Foot",
    "Birth country", "Passport country", "Contract expires", "Market value",
]

DISPLAY_BASE = [
    "Player", "Team", "Position", "Age", "_League",
    "Minutes played", "Matches played",
    "_anomaly_type", "_anomaly_score", "_peak_z", "_mean_z", "_anomaly_breadth",
]

KEY_STATS = [
    "Successful defensive actions per 90", "Defensive duels per 90",
    "Defensive duels won, %", "Interceptions per 90", "PAdj Interceptions",
    "Aerial duels won, %", "Passes per 90", "Accurate passes, %",
    "Progressive passes per 90", "Key passes per 90",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _assign_tier(score: float) -> int:
    if score >= 90: return 1
    if score >= 75: return 2
    if score >= 60: return 3
    return 4


def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]


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


def _display(df: pd.DataFrame, z_cols: list[str] | None = None) -> pd.DataFrame:
    keep = [c for c in DISPLAY_BASE + KEY_STATS + (z_cols or []) if c in df.columns]
    out = df[keep].copy().rename(columns={
        "_League": "League", "_anomaly_type": "Anomaly Type",
        "_anomaly_score": "Anomaly Score", "_peak_z": "Peak Z",
        "_mean_z": "Mean Z", "_anomaly_breadth": "Metric Breadth",
    })
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(3)
    return out.reset_index(drop=True)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(threshold: float, min_minutes: int, output: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  Wyscout DM Scouting Report")
    print(f"  Threshold: {threshold}  |  Min minutes: {min_minutes}")
    print(f"{'='*60}")

    # 1. Load + score all positions (needed for percentile context), then filter DM
    print("Loading Wyscout data…")
    raw = load_wyscout_raw(min_minutes=min_minutes)
    if raw.empty:
        raise RuntimeError("No Wyscout data found. Check data/Wyscout DB/")
    print(f"  → {len(raw):,} total players across {raw['_League'].nunique()} leagues")

    scored = compute_wyscout_scores(raw)
    scored = _dedup_cols(scored)

    if "PositionGroup" not in scored.columns:
        pos_col = next((c for c in ["Position", "Pos"] if c in scored.columns), None)
        if pos_col:
            scored["PositionGroup"] = scored[pos_col].map(WYSCOUT_POSITION_MAP).fillna("Other")

    dm = scored.loc[scored["PositionGroup"] == "DM"].copy().reset_index(drop=True)
    print(f"  → {len(dm):,} defensive midfielders")

    if dm.empty:
        raise RuntimeError("No DM players found after filtering.")

    # 2. Anomaly detection within DM group
    print("Running DM anomaly detection…")
    metrics = [m for m in DM_ANOMALY_METRICS if m in dm.columns]
    if metrics and len(dm) >= 5:
        engine = AnomalyEngine(threshold=threshold, method="z-score", groupby=None)
        try:
            dm = engine.fit_transform(dm, metrics)
        except Exception as e:
            print(f"  [warn] Anomaly engine failed: {e}")

    z_cols = [c for c in dm.columns if c.startswith("_z_")]

    anomalies = pd.DataFrame()
    if "_peak_z" in dm.columns:
        anomalies = dm.loc[dm["_peak_z"] >= threshold].sort_values("_anomaly_score", ascending=False)
    print(f"  → {len(anomalies):,} anomalies detected")

    def _type(t: str) -> pd.DataFrame:
        if anomalies.empty or "_anomaly_type" not in anomalies.columns:
            return pd.DataFrame()
        return anomalies.loc[anomalies["_anomaly_type"] == t].head(300)

    # 3. Similarity
    print("Computing similarity matrices…")
    feats = [f for f in DM_SIMILARITY_FEATURES if f in dm.columns]
    sim_engine = SimilarityEngine(feats)
    top_anom = anomalies.head(100)

    def _build_sim(method: str) -> pd.DataFrame:
        rows: list[dict] = []
        for _, target in top_anom.iterrows():
            results = sim_engine.find_similar(dm, target, method=method, n=10, same_position=False)
            for rank, (_, sim_row) in enumerate(results.iterrows(), start=1):
                rows.append({
                    "Target Player":   target.get("Player", ""),
                    "Target Team":     target.get("Team", ""),
                    "Target League":   target.get("_League", ""),
                    "Anomaly Type":    target.get("_anomaly_type", ""),
                    "Rank":            rank,
                    "Similar Player":  sim_row.get("Player", ""),
                    "Similar Team":    sim_row.get("Team", ""),
                    "Similar League":  sim_row.get("_League", ""),
                    "Similar Age":     sim_row.get("Age", ""),
                    "Similar Minutes": sim_row.get("Minutes played", ""),
                    "Similarity":      round(float(sim_row.get("_similarity", 0)), 4),
                })
        return pd.DataFrame(rows)

    sim_cosine    = _build_sim("cosine")
    sim_pearson   = _build_sim("pearson")
    sim_euclidean = _build_sim("euclidean")

    # 4. Profiles
    print("Building profiles…")
    bio    = [c for c in BIO_COLS if c in dm.columns]
    scores = [c for c in SCORE_COLS if c in dm.columns]
    prof_keep = bio + [c for c in scores if c not in bio]
    profiles = (
        dm[prof_keep].copy()
        .rename(columns={"_League": "League"})
        .sort_values("CompositeRecruitmentScore", ascending=False)
        .round(1)
        .reset_index(drop=True)
    )

    # 5. Tiers
    print("Building tiers…")
    tier_want = ["Player", "Team", "Position", "Age", "_League", "Minutes played"] + SCORE_COLS
    tier_keep = list(dict.fromkeys(c for c in tier_want if c in dm.columns))
    tiers = dm[tier_keep].copy().rename(columns={"_League": "League"})
    tiers["Tier"] = pd.to_numeric(tiers["CompositeRecruitmentScore"], errors="coerce").fillna(0).apply(_assign_tier)
    tiers["Tier Label"] = tiers["Tier"].map(TIER_LABELS)
    tiers = (
        tiers.sort_values(["Tier", "CompositeRecruitmentScore"], ascending=[True, False])
        .round(1)
        .reset_index(drop=True)
    )

    # 6. Playing styles
    print("Building playing styles…")
    available_scores = [c for c in SCORE_COLS if c in dm.columns]
    if available_scores:
        style_scores: dict[str, pd.Series] = {}
        for style, weights in DM_STYLE_WEIGHTS.items():
            score = pd.Series(0.0, index=dm.index)
            total_w = 0.0
            for metric, w in weights.items():
                if metric in dm.columns:
                    score += w * pd.to_numeric(dm[metric], errors="coerce").fillna(50)
                    total_w += w
            style_scores[style] = (score / (total_w or 1)).round(1)

        style_df = pd.DataFrame(style_scores, index=dm.index)
        keep_bio2 = [c for c in ["Player", "Team", "Position", "Age", "_League",
                                   "Minutes played", "CompositeRecruitmentScore"] if c in dm.columns]
        playing_styles = dm[keep_bio2].copy().rename(columns={"_League": "League"})
        playing_styles["Primary Style"] = style_df.idxmax(axis=1)
        playing_styles["Style Score"]   = style_df.max(axis=1).round(1)
        for style in DM_STYLE_WEIGHTS:
            playing_styles[style] = style_scores[style].values
        playing_styles = (
            playing_styles.sort_values("CompositeRecruitmentScore", ascending=False)
            .round(1)
            .reset_index(drop=True)
        )
    else:
        playing_styles = pd.DataFrame()

    # 7. Main scouting board
    keep_board = [c for c in DM_DISPLAY_COLS if c in dm.columns]
    scouting_board = (
        dm[keep_board].copy()
        .rename(columns={
            "_League": "League",
            "_anomaly_type": "Anomaly Type",
            "_anomaly_score": "Anomaly Score",
            "_peak_z": "Peak Z",
        })
        .sort_values("CompositeRecruitmentScore", ascending=False)
        .round(2)
        .reset_index(drop=True)
    )

    # ── Write workbook ─────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting workbook → {output}")

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _write_sheet(writer, "DM Scouting Board",      scouting_board)
        _write_sheet(writer, "Anomaly Overview",        _display(anomalies.head(500), z_cols))
        _write_sheet(writer, "Hidden Gems",             _display(_type("Hidden Gem")))
        _write_sheet(writer, "Specialist Elite",        _display(_type("Specialist Elite")))
        _write_sheet(writer, "Multi-dimensional",       _display(_type("Multi-dimensional")))
        _write_sheet(writer, "Age-adjusted Gems",       _display(_type("Age-adjusted Gem")))
        _write_sheet(writer, "Consistent Overperf",     _display(_type("Consistent Overperformer")))
        _write_sheet(writer, "Profiles",                profiles)
        _write_sheet(writer, "Tiers",                   tiers)
        _write_sheet(writer, "Playing Styles",          playing_styles)
        _write_sheet(writer, "Similarity — Cosine",     sim_cosine)
        _write_sheet(writer, "Similarity — Pearson",    sim_pearson)
        _write_sheet(writer, "Similarity — Euclidean",  sim_euclidean)

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {size_mb:.1f} MB → {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wyscout DM positional scouting report")
    parser.add_argument("--threshold",   type=float, default=1.8,
                        help="Z-score anomaly threshold (default 1.8)")
    parser.add_argument("--min-minutes", type=int,   default=400,
                        help="Minimum minutes played (default 400)")
    parser.add_argument("--output",      type=Path,
                        default=OUT_DIR / "Wyscout DM Scouting.xlsx",
                        help="Output .xlsx path")
    args = parser.parse_args()
    run(threshold=args.threshold, min_minutes=args.min_minutes, output=args.output)
