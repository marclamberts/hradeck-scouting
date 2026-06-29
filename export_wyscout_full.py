"""
export_wyscout_full.py
──────────────────────
Comprehensive Wyscout scouting workbook.

Sheets produced
───────────────
  All Players            — full player table with composite scores + anomaly tags
  Anomaly Overview       — all anomalies ranked by score
  Hidden Gems            — high signal, low exposure
  Specialist Elite       — elite in 1–2 metrics
  Multi-dimensional      — 4+ metrics above threshold
  Age-adjusted Gems      — ≤23 anomalies
  Consistent Overperf    — broad positive signal
  Similarity — Cosine    — top-10 comparables per anomaly
  Similarity — Pearson
  Similarity — Euclidean
  Profiles               — full composite score card per player
  Tiers                  — tiered recruitment board per position (T1–T4)
  Playing Styles         — rule-based style classification + radar summary
  Scouting Uncertainty   — sample-size uncertainty ranking with confidence bands
  SP Reliance            — how dependent each player is on set pieces
  SP Output              — volume: corners, crosses, free kicks delivered
  SP Quality             — accuracy + xG efficiency from set pieces
  SP Anomalies           — statistical outliers in set-piece metrics
  SP Corner Taker        — top players per set-piece role (×6)
  SP Dead Ball Spec.
  SP Crossing Threat
  SP Aerial Threat
  SP Box Presence
  SP Set Piece Blocker
  GK Scouting            — goalkeeper-only deep dive (shot-stopping + sweeping + distribution)
  GK Tiers               — goalkeepers tiered by profile
  Position — ST          — positional ranked board per group (×7)
  Position — W
  Position — AM
  Position — CM
  Position — DM
  Position — FB
  Position — CB

Usage
─────
  python export_wyscout_full.py [--threshold 1.8] [--min-minutes 400]
                                [--output "data/Wyscout Full Scouting Report.xlsx"]
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
from scouting_model import AnomalyEngine, SimilarityEngine, SetPieceAnalyzer, SET_PIECE_ROLES
from wyscout_model import (
    compute_wyscout_scores,
    load_wyscout_raw,
    SCORE_BLUEPRINTS,
    WYSCOUT_POSITION_MAP,
)

# ── Constants ──────────────────────────────────────────────────────────────────

POSITION_MAP = WYSCOUT_POSITION_MAP

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
        "Exits per 90", "Aerial duels per 90",
        "Accurate passes, %", "Accurate long passes, %",
        "Back passes received as GK per 90",
    ],
}

SIMILARITY_FEATURES = [
    "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
    "Key passes per 90", "Passes per 90", "Accurate passes, %",
    "Dribbles per 90", "Successful dribbles, %",
    "Successful defensive actions per 90", "Defensive duels won, %",
    "Aerial duels won, %", "Progressive passes per 90",
    "Progressive runs per 90", "Touches in box per 90",
]

SCORE_COLS = [
    "ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore",
    "PressingScore", "BallSecurityScore", "ExpectedThreatScore",
    "ASA_GoalsAddedScore", "AerialScore", "SetPieceScore",
    "CompositeRecruitmentScore", "Rating", "ScoutingUncertainty",
]

DISPLAY_BASE = [
    "Player", "Team", "Position", "PositionGroup", "Age", "_League",
    "Minutes played", "Matches played",
    "_anomaly_type", "_anomaly_score", "_peak_z", "_mean_z", "_anomaly_breadth",
]

KEY_STATS = [
    "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
    "Key passes per 90", "Passes per 90", "Accurate passes, %",
    "Dribbles per 90", "Successful defensive actions per 90",
    "Aerial duels won, %", "Progressive passes per 90",
]

BIO_COLS = [
    "Player", "Team", "Position", "PositionGroup", "Age", "_League",
    "Minutes played", "Matches played", "Height", "Foot",
    "Birth country", "Passport country", "Contract expires",
    "Market value",
]

# ── Playing style definitions (score-dimension-based) ──────────────────────────

PLAYING_STYLE_WEIGHTS: dict[str, dict[str, float]] = {
    "Goal Machine":        {"ScoringThreatScore": 3.0, "ExpectedThreatScore": 2.0, "ASA_GoalsAddedScore": 2.0},
    "Creative Playmaker":  {"CreativeProgressionScore": 3.0, "BallSecurityScore": 1.5, "ExpectedThreatScore": 1.5},
    "Press Monster":       {"PressingScore": 3.0, "DefensiveDisruptionScore": 2.0},
    "Defensive Enforcer":  {"DefensiveDisruptionScore": 3.0, "PressingScore": 2.0, "BallSecurityScore": 1.0},
    "Ball Playing Anchor": {"BallSecurityScore": 3.0, "CreativeProgressionScore": 2.0, "PressingScore": 1.0},
    "Aerial Dominator":    {"AerialScore": 3.0, "DefensiveDisruptionScore": 1.5},
    "Set Piece Weapon":    {"SetPieceScore": 3.0, "AerialScore": 1.5, "ScoringThreatScore": 1.0},
    "Complete Player":     {c: 1.0 for c in [
        "ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore",
        "PressingScore", "BallSecurityScore", "ExpectedThreatScore",
    ]},
    "Box-to-Box Engine":   {
        "PressingScore": 1.5, "CreativeProgressionScore": 1.5,
        "ScoringThreatScore": 1.0, "DefensiveDisruptionScore": 1.0,
    },
    "Wide Carrier":        {
        "ScoringThreatScore": 1.5, "CreativeProgressionScore": 1.5,
        "BallSecurityScore": 1.0, "PressingScore": 1.0,
    },
}

GK_STYLE_WEIGHTS: dict[str, list[str]] = {
    "Shot Stopper":     ["Save rate, %", "Prevented goals per 90"],
    "Sweeper GK":       ["Exits per 90", "Aerial duels per 90"],
    "Ball Playing GK":  ["Accurate passes, %", "Accurate long passes, %"],
    "Command of Area":  ["Aerial duels per 90", "Shots blocked per 90"],
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_score(min_minutes: int) -> pd.DataFrame:
    print(f"  Loading raw Wyscout data (min {min_minutes} min)…")
    raw = load_wyscout_raw(min_minutes=min_minutes)
    if raw.empty:
        raise RuntimeError("No Wyscout data found. Check data/Wyscout DB/")
    print(f"  → {len(raw):,} players across {raw['_League'].nunique()} leagues")

    print("  Computing composite scores…")
    scored = compute_wyscout_scores(raw)

    # Ensure PositionGroup exists
    if "PositionGroup" not in scored.columns:
        pos_col = next((c for c in ["Position", "Pos"] if c in scored.columns), None)
        if pos_col:
            scored["PositionGroup"] = scored[pos_col].map(POSITION_MAP).fillna("Other")

    # Hidden Gem proxy: fewer minutes = lower composite exposure
    if "CompositeRecruitmentScore" not in scored.columns:
        mins = pd.to_numeric(scored.get("Minutes played", scored.get("MinutesPlayed", 0)), errors="coerce").fillna(0)
        scored["CompositeRecruitmentScore"] = (
            (mins - mins.min()) / (mins.max() - mins.min() + 1e-9) * 100
        ).clip(0, 100)

    return scored.reset_index(drop=True)


# ── Anomaly detection ──────────────────────────────────────────────────────────

def run_anomaly(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
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
    return pd.concat(frames, ignore_index=True) if frames else df


# ── Similarity ─────────────────────────────────────────────────────────────────

def build_similarity_sheet(df: pd.DataFrame, anomalies: pd.DataFrame, method: str, n: int = 10) -> pd.DataFrame:
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


# ── Profiles sheet ─────────────────────────────────────────────────────────────

def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate columns, keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()]


def build_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = _dedup_cols(df)
    bio    = [c for c in BIO_COLS if c in df.columns]
    scores = [c for c in SCORE_COLS if c in df.columns]
    keep   = bio + [c for c in scores if c not in bio]
    out    = df[keep].copy().rename(columns={"_League": "League"})
    out    = out.sort_values("CompositeRecruitmentScore", ascending=False) if "CompositeRecruitmentScore" in out.columns else out
    return out.round(1).reset_index(drop=True)


# ── Tiers sheet ────────────────────────────────────────────────────────────────

TIER_LABELS = {1: "T1 — Elite (90+)", 2: "T2 — Strong (75–89)", 3: "T3 — Promising (60–74)", 4: "T4 — Below Average (<60)"}

def _assign_tier(score: float) -> int:
    if score >= 90: return 1
    if score >= 75: return 2
    if score >= 60: return 3
    return 4


def build_tiers(df: pd.DataFrame) -> pd.DataFrame:
    df = _dedup_cols(df)
    if "CompositeRecruitmentScore" not in df.columns:
        return pd.DataFrame()
    seen: set[str] = set()
    keep = []
    for c in ["Player", "Team", "Position", "PositionGroup", "Age", "_League",
              "Minutes played"] + SCORE_COLS:
        if c in df.columns and c not in seen:
            keep.append(c)
            seen.add(c)
    out = df[keep].copy().rename(columns={"_League": "League"})
    out["Tier"] = pd.to_numeric(out["CompositeRecruitmentScore"], errors="coerce").fillna(0).apply(_assign_tier)
    out["Tier Label"] = out["Tier"].map(TIER_LABELS)
    out = out.sort_values(["PositionGroup", "Tier", "CompositeRecruitmentScore"], ascending=[True, True, False])
    return out.round(1).reset_index(drop=True)


# ── Playing Styles ─────────────────────────────────────────────────────────────

def build_playing_styles(df: pd.DataFrame) -> pd.DataFrame:
    df = _dedup_cols(df)
    available_scores = [c for c in SCORE_COLS if c in df.columns]
    if not available_scores:
        return pd.DataFrame()

    # Score each style for each player
    style_scores: dict[str, pd.Series] = {}
    for style, weights in PLAYING_STYLE_WEIGHTS.items():
        score = pd.Series(0.0, index=df.index)
        total_w = 0.0
        for metric, w in weights.items():
            if metric in df.columns:
                score += w * pd.to_numeric(df[metric], errors="coerce").fillna(50)
                total_w += w
        style_scores[style] = (score / (total_w or 1)).round(1)

    style_df = pd.DataFrame(style_scores, index=df.index)
    primary_style = style_df.idxmax(axis=1)
    style_score   = style_df.max(axis=1).round(1)

    keep_bio = [c for c in ["Player", "Team", "Position", "PositionGroup", "Age", "_League",
                              "Minutes played", "CompositeRecruitmentScore"] if c in df.columns]
    out = df[keep_bio].copy().rename(columns={"_League": "League"})
    out["Primary Style"] = primary_style
    out["Style Score"]   = style_score

    # Append each style score column
    for style in PLAYING_STYLE_WEIGHTS:
        out[style] = style_scores[style].values

    return out.sort_values("CompositeRecruitmentScore", ascending=False).round(1).reset_index(drop=True)


# ── Set-Piece: Reliance / Output / Quality ─────────────────────────────────────

SP_OUTPUT_COLS = [
    "Corners per 90", "Free kicks per 90", "Direct free kicks per 90",
    "Crosses per 90", "Deep completed crosses per 90",
]

SP_QUALITY_COLS = [
    "Direct free kicks on target, %", "Accurate crosses, %",
    "Head goals per 90", "Aerial duels won, %", "Aerial duels per 90",
]

SP_RELIANCE_NUMERATORS = ["Corners per 90", "Free kicks per 90", "Direct free kicks per 90", "Head goals per 90"]
SP_RELIANCE_DENOMINATORS = ["Goals per 90", "xG per 90", "Key passes per 90", "Passes per 90"]


def build_sp_reliance(df: pd.DataFrame) -> pd.DataFrame:
    keep_bio = [c for c in ["Player", "Team", "Position", "PositionGroup", "Age", "_League",
                              "Minutes played"] if c in df.columns]
    out = df[keep_bio].copy().rename(columns={"_League": "League"})

    # Attacking output that came from set pieces vs total
    sp_attack = sum(
        pd.to_numeric(df[c], errors="coerce").fillna(0)
        for c in SP_RELIANCE_NUMERATORS if c in df.columns
    )
    total_attack = sum(
        pd.to_numeric(df[c], errors="coerce").fillna(0)
        for c in ["Goals per 90", "xG per 90", "Key passes per 90"] if c in df.columns
    )
    out["SP Attack Contribution"] = (sp_attack / (total_attack + 1e-9) * 100).clip(0, 100).round(1)

    # SP delivery volume
    delivery = sum(
        pd.to_numeric(df[c], errors="coerce").fillna(0)
        for c in ["Corners per 90", "Free kicks per 90", "Direct free kicks per 90", "Crosses per 90"] if c in df.columns
    )
    out["SP Delivery Volume per 90"] = delivery.round(2)

    # Aerial reliance
    if "Aerial duels per 90" in df.columns and "Duels per 90" in df.columns:
        aerial = pd.to_numeric(df["Aerial duels per 90"], errors="coerce").fillna(0)
        total_duels = pd.to_numeric(df["Duels per 90"], errors="coerce").fillna(0)
        out["Aerial Duel Reliance %"] = (aerial / (total_duels + 1e-9) * 100).clip(0, 100).round(1)

    for c in SP_RELIANCE_NUMERATORS:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    return out.sort_values("SP Attack Contribution", ascending=False).reset_index(drop=True)


def build_sp_output(df: pd.DataFrame) -> pd.DataFrame:
    keep_bio = [c for c in ["Player", "Team", "Position", "PositionGroup", "Age", "_League",
                              "Minutes played"] if c in df.columns]
    out = df[keep_bio].copy().rename(columns={"_League": "League"})

    total_delivery = pd.Series(0.0, index=df.index)
    for c in SP_OUTPUT_COLS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0)
            out[c] = v.round(2)
            total_delivery += v
    out["Total SP Delivery per 90"] = total_delivery.round(2)

    if "SetPieceScore" in df.columns:
        out["SetPieceScore"] = pd.to_numeric(df["SetPieceScore"], errors="coerce").round(1)

    return out.sort_values("Total SP Delivery per 90", ascending=False).reset_index(drop=True)


def build_sp_quality(df: pd.DataFrame) -> pd.DataFrame:
    keep_bio = [c for c in ["Player", "Team", "Position", "PositionGroup", "Age", "_League",
                              "Minutes played"] if c in df.columns]
    out = df[keep_bio].copy().rename(columns={"_League": "League"})

    for c in SP_QUALITY_COLS:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    # SP quality composite (accuracy-weighted)
    quality_score = pd.Series(0.0, index=df.index)
    quality_w = 0.0
    quality_weights = {
        "Direct free kicks on target, %": 2.5,
        "Accurate crosses, %":            2.0,
        "Aerial duels won, %":            1.5,
        "Head goals per 90":              2.0,
    }
    for c, w in quality_weights.items():
        if c in df.columns:
            quality_score += w * pd.to_numeric(df[c], errors="coerce").fillna(0)
            quality_w     += w
    out["SP Quality Index"] = (quality_score / (quality_w or 1)).round(2)

    if "xG per 90" in df.columns and "Goals per 90" in df.columns:
        xg  = pd.to_numeric(df["xG per 90"], errors="coerce").fillna(0)
        gls = pd.to_numeric(df["Goals per 90"], errors="coerce").fillna(0)
        out["Goals vs xG Overperformance"] = (gls - xg).round(3)

    return out.sort_values("SP Quality Index", ascending=False).reset_index(drop=True)


# ── Goalkeeper scouting ────────────────────────────────────────────────────────

GK_DISPLAY_COLS = [
    "Player", "Team", "Age", "_League", "Minutes played", "Matches played",
    "Height", "Foot",
    # Shot stopping
    "Save rate, %", "Prevented goals per 90", "Conceded goals per 90",
    "Shots against per 90", "Clean sheets",
    # Sweeping
    "Exits per 90", "Aerial duels per 90",
    # Distribution
    "Accurate passes, %", "Accurate long passes, %",
    "Back passes received as GK per 90",
    # Computed
    "BallSecurityScore", "DefensiveDisruptionScore", "CompositeRecruitmentScore",
]

GK_ANOMALY_METRICS = POSITION_METRICS["GK"]


def build_gk_scouting(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    gks = df.loc[df["PositionGroup"] == "GK"].copy()
    if gks.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Run GK-specific anomaly detection
    metrics = [m for m in GK_ANOMALY_METRICS if m in gks.columns]
    if metrics and len(gks) >= 5:
        engine = AnomalyEngine(threshold=threshold * 0.9, method="z-score", groupby=None)
        try:
            gks = engine.fit_transform(gks, metrics)
        except Exception:
            pass

    # Classify GK playing style
    gk_styles: list[str] = []
    for _, row in gks.iterrows():
        best_style = "All-Rounder"
        best_score = -np.inf
        for style, cols in GK_STYLE_WEIGHTS.items():
            available = [c for c in cols if c in gks.columns]
            if not available:
                continue
            score = sum(float(pd.to_numeric(row.get(c, 0), errors="coerce") or 0) for c in available) / len(available)
            if score > best_score:
                best_score = score
                best_style = style
        gk_styles.append(best_style)
    gks["GK Style"] = gk_styles

    keep = [c for c in GK_DISPLAY_COLS + ["GK Style", "_anomaly_score", "_peak_z", "_anomaly_type"] if c in gks.columns]
    scouting = (
        gks[keep]
        .rename(columns={"_League": "League", "_anomaly_score": "Anomaly Score",
                         "_peak_z": "Peak Z", "_anomaly_type": "Anomaly Type"})
        .sort_values("CompositeRecruitmentScore" if "CompositeRecruitmentScore" in keep else keep[0],
                     ascending=False)
        .round(2)
        .reset_index(drop=True)
    )

    # GK Tiers
    if "CompositeRecruitmentScore" in gks.columns:
        _tier_want = ["Player", "Team", "Age", "_League", "Minutes played",
                      "Save rate, %", "Prevented goals per 90", "Exits per 90",
                      "Accurate passes, %", "GK Style", "CompositeRecruitmentScore"]
        tier_keep = list(dict.fromkeys(c for c in _tier_want if c in gks.columns))
        tiers_gk = gks[tier_keep].copy().rename(columns={"_League": "League"})
        tiers_gk["Tier"] = pd.to_numeric(tiers_gk["CompositeRecruitmentScore"], errors="coerce").fillna(0).apply(_assign_tier)
        tiers_gk["Tier Label"] = tiers_gk["Tier"].map(TIER_LABELS)
        tiers_gk = tiers_gk.sort_values(["Tier", "CompositeRecruitmentScore"], ascending=[True, False]).round(2).reset_index(drop=True)
    else:
        tiers_gk = pd.DataFrame()

    return scouting, tiers_gk


# ── Positional scouting boards ─────────────────────────────────────────────────

POSITION_DISPLAY: dict[str, list[str]] = {
    "ST": ["Goals per 90", "Non-penalty goals per 90", "xG per 90", "Shots per 90",
           "Shots on target, %", "Goal conversion, %", "Touches in box per 90",
           "Aerial duels won, %", "Dribbles per 90", "xA per 90",
           "ScoringThreatScore", "ExpectedThreatScore", "CompositeRecruitmentScore",
           "Rating", "ScoutingUncertainty"],
    "W":  ["Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
           "Key passes per 90", "Dribbles per 90", "Successful dribbles, %",
           "Crosses per 90", "Accurate crosses, %", "Progressive runs per 90",
           "ScoringThreatScore", "CreativeProgressionScore", "CompositeRecruitmentScore",
           "Rating", "ScoutingUncertainty"],
    "AM": ["Key passes per 90", "xA per 90", "Assists per 90", "Smart passes per 90",
           "Goals per 90", "xG per 90", "Touches in box per 90", "Through passes per 90",
           "Dribbles per 90", "Progressive passes per 90",
           "CreativeProgressionScore", "ExpectedThreatScore", "CompositeRecruitmentScore",
           "Rating", "ScoutingUncertainty"],
    "CM": ["Passes per 90", "Accurate passes, %", "Progressive passes per 90",
           "Key passes per 90", "xA per 90", "Progressive runs per 90",
           "Successful defensive actions per 90", "Interceptions per 90", "Duels won, %",
           "CreativeProgressionScore", "BallSecurityScore", "CompositeRecruitmentScore",
           "Rating", "ScoutingUncertainty"],
    "DM": ["Successful defensive actions per 90", "Defensive duels per 90",
           "Defensive duels won, %", "Interceptions per 90", "PAdj Interceptions",
           "Aerial duels won, %", "Passes per 90", "Accurate passes, %",
           "Progressive passes per 90",
           "DefensiveDisruptionScore", "PressingScore", "CompositeRecruitmentScore",
           "Rating", "ScoutingUncertainty"],
    "FB": ["Crosses per 90", "Accurate crosses, %", "xA per 90", "Assists per 90",
           "Progressive runs per 90", "Dribbles per 90",
           "Successful defensive actions per 90", "Defensive duels won, %",
           "Aerial duels won, %", "Progressive passes per 90",
           "CreativeProgressionScore", "DefensiveDisruptionScore", "CompositeRecruitmentScore",
           "Rating", "ScoutingUncertainty"],
    "CB": ["Successful defensive actions per 90", "Defensive duels per 90",
           "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
           "Interceptions per 90", "PAdj Interceptions", "Shots blocked per 90",
           "Accurate passes, %", "Progressive passes per 90",
           "DefensiveDisruptionScore", "BallSecurityScore", "CompositeRecruitmentScore",
           "Rating", "ScoutingUncertainty"],
}


def build_positional_boards(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    boards: dict[str, pd.DataFrame] = {}
    for pos, stat_cols in POSITION_DISPLAY.items():
        grp = df.loc[df["PositionGroup"] == pos].copy()
        if grp.empty:
            continue
        bio = [c for c in ["Player", "Team", "Age", "_League", "Minutes played",
                             "Foot", "Height", "Contract expires", "Market value",
                             "_anomaly_type", "_anomaly_score"] if c in grp.columns]
        keep = bio + [c for c in stat_cols if c in grp.columns and c not in bio]
        board = (
            grp[keep]
            .rename(columns={"_League": "League", "_anomaly_type": "Anomaly Type",
                              "_anomaly_score": "Anomaly Score"})
            .sort_values("CompositeRecruitmentScore" if "CompositeRecruitmentScore" in keep else keep[0],
                         ascending=False)
            .round(2)
            .reset_index(drop=True)
        )
        boards[pos] = board
    return boards


# ── Scouting Uncertainty ──────────────────────────────────────────────────────

def build_scouting_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sheet showing Rating and Scouting Uncertainty for every player,
    sorted from most to least certain.
    """
    keep_bio = [c for c in [
        "Player", "Team", "Position", "PositionGroup", "Age", "_League",
        "Minutes played", "Matches played",
    ] if c in df.columns]
    out = df[keep_bio].copy().rename(columns={"_League": "League"})

    if "Rating" in df.columns:
        out["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").round(1)
    if "ScoutingUncertainty" in df.columns:
        out["Scouting Uncertainty"] = pd.to_numeric(df["ScoutingUncertainty"], errors="coerce").round(1)

    sort_col = "Scouting Uncertainty" if "Scouting Uncertainty" in out.columns else out.columns[0]
    return out.sort_values(sort_col).reset_index(drop=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _display(df: pd.DataFrame, z_cols: list[str] | None = None) -> pd.DataFrame:
    keep = [c for c in DISPLAY_BASE + KEY_STATS + (z_cols or []) if c in df.columns]
    out  = df[keep].copy().rename(columns={
        "_League": "League", "_anomaly_type": "Anomaly Type",
        "_anomaly_score": "Anomaly Score", "_peak_z": "Peak Z",
        "_mean_z": "Mean Z", "_anomaly_breadth": "Metric Breadth",
    })
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(3)
    return out.reset_index(drop=True)


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

def run(threshold: float, min_minutes: int, output: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  Wyscout Full Scouting Report")
    print(f"  Threshold: {threshold}  |  Min minutes: {min_minutes}")
    print(f"{'='*60}")

    # 1. Load + score
    df = load_and_score(min_minutes)

    # Drop any duplicate columns produced by the scoring pipeline
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Anomaly detection
    print("Running position-specific anomaly detection…")
    zdf = run_anomaly(df, threshold)
    z_cols = [c for c in zdf.columns if c.startswith("_z_")]

    anomalies = pd.DataFrame()
    if "_peak_z" in zdf.columns:
        anomalies = zdf.loc[zdf["_peak_z"] >= threshold].sort_values("_anomaly_score", ascending=False)
    print(f"  → {len(anomalies):,} anomalies detected")

    def _type(t: str) -> pd.DataFrame:
        if anomalies.empty or "_anomaly_type" not in anomalies.columns:
            return pd.DataFrame()
        return anomalies.loc[anomalies["_anomaly_type"] == t].head(300)

    # 3. Similarity
    print("Computing similarity matrices (top 150 anomalies)…")
    top_anom = anomalies.head(150)
    sim_cosine    = build_similarity_sheet(zdf, top_anom, "cosine")
    sim_pearson   = build_similarity_sheet(zdf, top_anom, "pearson")
    sim_euclidean = build_similarity_sheet(zdf, top_anom, "euclidean")

    # 4. Set-piece analysis
    print("Running set-piece analyzer…")
    sp_analyzer  = SetPieceAnalyzer(threshold=threshold * 0.85)
    ws_enriched  = sp_analyzer.fit_transform(zdf)
    sp_anomaly   = sp_analyzer.anomaly_table(ws_enriched, top_n=300)
    role_leaders = sp_analyzer.top_players_by_role(ws_enriched, top_n=30)

    # 5. Profiles + Tiers + Styles + Uncertainty
    print("Building profiles, tiers, playing styles, and scouting uncertainty…")
    profiles           = build_profiles(ws_enriched)
    tiers              = build_tiers(ws_enriched)
    playing_styles     = build_playing_styles(ws_enriched)
    scouting_uncert    = build_scouting_uncertainty(ws_enriched)

    # 6. Set-piece reliance / output / quality
    print("Analysing set-piece reliance, output, and quality…")
    sp_reliance = build_sp_reliance(ws_enriched)
    sp_output   = build_sp_output(ws_enriched)
    sp_quality  = build_sp_quality(ws_enriched)

    # 7. GK scouting
    print("Building goalkeeper scouting board…")
    gk_scouting, gk_tiers = build_gk_scouting(ws_enriched, threshold)

    # 8. Positional boards
    print("Building positional scouting boards…")
    pos_boards = build_positional_boards(ws_enriched)

    # 9. All Players sheet
    ALL_DISPLAY_COLS = [
        "Player", "Team", "Position", "PositionGroup", "Age", "_League",
        "Minutes played", "Matches played", "Height", "Foot",
        "Contract expires", "Market value",
        "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
        "Shots per 90", "Shots on target, %", "Key passes per 90",
        "Passes per 90", "Accurate passes, %", "Progressive passes per 90",
        "Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %",
        "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
        "Interceptions per 90", "Crosses per 90", "Accurate crosses, %",
        "Head goals per 90", "Touches in box per 90", "Duels won, %",
        # Scores
        "ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore",
        "PressingScore", "BallSecurityScore", "ExpectedThreatScore",
        "ASA_GoalsAddedScore", "AerialScore", "SetPieceScore",
        "CompositeRecruitmentScore",
        # Scout numbers
        "Rating", "ScoutingUncertainty",
        # SP + Anomaly tags
        "_sp_primary_role", "_sp_composite",
        "_anomaly_type", "_anomaly_score", "_peak_z",
    ]
    keep_all = [c for c in ALL_DISPLAY_COLS if c in ws_enriched.columns]
    all_players = (
        ws_enriched[keep_all]
        .rename(columns={
            "_League": "League", "_sp_primary_role": "SP Role",
            "_sp_composite": "SP Score",
            "_anomaly_type": "Anomaly Type", "_anomaly_score": "Anomaly Score",
            "_peak_z": "Peak Z",
        })
        .sort_values("CompositeRecruitmentScore", ascending=False)
        .round(3)
        .reset_index(drop=True)
    )

    # ── Assemble workbook ──────────────────────────────────────────────────────
    print(f"\nWriting workbook → {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Core
        _write_sheet(writer, "All Players",           all_players)
        _write_sheet(writer, "Anomaly Overview",      _display(anomalies.head(500), z_cols))
        _write_sheet(writer, "Hidden Gems",           _display(_type("Hidden Gem")))
        _write_sheet(writer, "Specialist Elite",      _display(_type("Specialist Elite")))
        _write_sheet(writer, "Multi-dimensional",     _display(_type("Multi-dimensional")))
        _write_sheet(writer, "Age-adjusted Gems",     _display(_type("Age-adjusted Gem")))
        _write_sheet(writer, "Consistent Overperf",   _display(_type("Consistent Overperformer")))
        # Similarity
        _write_sheet(writer, "Similarity — Cosine",    sim_cosine)
        _write_sheet(writer, "Similarity — Pearson",   sim_pearson)
        _write_sheet(writer, "Similarity — Euclidean", sim_euclidean)
        # Profiles / Tiers / Styles / Uncertainty
        _write_sheet(writer, "Profiles",            profiles)
        _write_sheet(writer, "Tiers",               tiers)
        _write_sheet(writer, "Playing Styles",      playing_styles)
        _write_sheet(writer, "Scouting Uncertainty", scouting_uncert)
        # Set piece
        _write_sheet(writer, "SP Reliance",     sp_reliance)
        _write_sheet(writer, "SP Output",       sp_output)
        _write_sheet(writer, "SP Quality",      sp_quality)
        _write_sheet(writer, "SP Anomalies",    sp_anomaly)
        for role, df_role in role_leaders.items():
            _write_sheet(writer, f"SP {role}"[:31], df_role.round(3).reset_index(drop=True))
        # GK
        _write_sheet(writer, "GK Scouting",    gk_scouting)
        _write_sheet(writer, "GK Tiers",        gk_tiers)
        # Positional boards
        for pos, board in pos_boards.items():
            _write_sheet(writer, f"Position — {pos}", board)

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {size_mb:.1f} MB → {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Wyscout scouting workbook")
    parser.add_argument("--threshold",   type=float, default=1.8)
    parser.add_argument("--min-minutes", type=int,   default=400)
    parser.add_argument("--output",      type=Path,
                        default=DATA_DIR / "Wyscout Full Scouting Report.xlsx")
    args = parser.parse_args()
    run(threshold=args.threshold, min_minutes=args.min_minutes, output=args.output)
