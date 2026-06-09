"""
wyscout_model.py
────────────────
Full analytical model built on Wyscout per-90 data.

Mirrors the IMPECT score system by computing position-normalised
composite scores (0–100 percentile scale) from raw Wyscout metrics,
so all downstream workspaces (anomaly, similarity, projections) can
operate on either data source interchangeably.

Score columns produced (all 0–100)
───────────────────────────────────
  ScoringThreatScore        — goals, xG, shots, box presence
  CreativeProgressionScore  — key passes, xA, progressive passing
  DefensiveDisruptionScore  — defensive actions, interceptions, aerial duels
  PressingScore             — defensive duels, pressing activity
  BallSecurityScore         — passing accuracy, duel success rate
  ExpectedThreatScore       — xG + xA combined
  ASA_GoalsAddedScore       — goals + assists value proxy
  AerialScore               — aerial duel dominance
  SetPieceScore             — corner/free-kick delivery
  CompositeRecruitmentScore — weighted composite of the above
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore

# ── Position mapping ───────────────────────────────────────────────────────────
WYSCOUT_POSITION_MAP: dict[str, str] = {
    "CF": "ST", "SS": "ST",
    "LW": "W",  "RW": "W",  "LWF": "W",  "RWF": "W",  "WF": "W",
    "AMF": "AM", "LAMF": "AM", "RAMF": "AM",
    "CMF": "CM", "LCM": "CM",  "RCM": "CM",  "LCMF": "CM", "RCMF": "CM",
    "DMF": "DM", "LDM": "DM",  "RDM": "DM",  "LDMF": "DM", "RDMF": "DM",
    "LB": "FB",  "RB": "FB",   "LWB": "FB",  "RWB": "FB",
    "CB": "CB",  "LCB": "CB",  "RCB": "CB",
    "GK": "GK",
}

# ── Metric blueprints for each composite score ─────────────────────────────────
# (metric_name, weight)
SCORE_BLUEPRINTS: dict[str, list[tuple[str, float]]] = {
    "ScoringThreatScore": [
        ("Goals per 90",               3.0),
        ("Non-penalty goals per 90",   2.5),
        ("xG per 90",                  2.5),
        ("Shots per 90",               1.0),
        ("Shots on target, %",         1.5),
        ("Goal conversion, %",         1.5),
        ("Touches in box per 90",      1.0),
    ],
    "CreativeProgressionScore": [
        ("Key passes per 90",          3.0),
        ("xA per 90",                  2.5),
        ("Assists per 90",             2.0),
        ("Smart passes per 90",        1.5),
        ("Accurate smart passes, %",   1.0),
        ("Progressive passes per 90",  2.0),
        ("Deep completions per 90",    1.0),
        ("Through passes per 90",      1.0),
        ("Passes to final third per 90", 1.0),
    ],
    "DefensiveDisruptionScore": [
        ("Successful defensive actions per 90", 3.0),
        ("Interceptions per 90",       2.5),
        ("PAdj Interceptions",         2.0),
        ("Shots blocked per 90",       1.5),
        ("Aerial duels won, %",        1.5),
        ("Aerial duels per 90",        1.0),
        ("Sliding tackles per 90",     1.0),
    ],
    "PressingScore": [
        ("Defensive duels per 90",     2.5),
        ("Defensive duels won, %",     2.0),
        ("Successful defensive actions per 90", 2.0),
        ("Fouls per 90",               0.5),  # positive: shows pressing attempt
        ("Duels per 90",               1.0),
    ],
    "BallSecurityScore": [
        ("Accurate passes, %",         3.0),
        ("Duels won, %",               2.0),
        ("Accurate short / medium passes, %", 1.5),
        ("Accurate long passes, %",    1.0),
    ],
    "ExpectedThreatScore": [
        ("xG per 90",                  3.0),
        ("xA per 90",                  3.0),
        ("Shot assists per 90",        1.5),
        ("Second assists per 90",      1.0),
    ],
    "ASA_GoalsAddedScore": [
        ("Goals per 90",               3.0),
        ("Assists per 90",             2.5),
        ("xG per 90",                  1.5),
        ("xA per 90",                  1.5),
        ("Key passes per 90",          1.0),
    ],
    "AerialScore": [
        ("Aerial duels won, %",        3.0),
        ("Aerial duels per 90",        2.0),
        ("Head goals per 90",          2.5),
    ],
    "SetPieceScore": [
        ("Corners per 90",             3.0),
        ("Direct free kicks per 90",   2.5),
        ("Direct free kicks on target, %", 2.0),
        ("Free kicks per 90",          1.5),
        ("Deep completed crosses per 90", 1.5),
    ],
}

# Composite = weighted sum of individual score z-values
COMPOSITE_WEIGHTS: dict[str, float] = {
    "ScoringThreatScore":       2.0,
    "CreativeProgressionScore": 2.0,
    "DefensiveDisruptionScore": 1.8,
    "PressingScore":            1.4,
    "BallSecurityScore":        1.4,
    "ExpectedThreatScore":      1.8,
    "ASA_GoalsAddedScore":      1.6,
}

# Per-position score relevance weights (multiplied into composite)
POSITION_RELEVANCE: dict[str, dict[str, float]] = {
    "ST": {"ScoringThreatScore": 1.4, "ExpectedThreatScore": 1.3, "ASA_GoalsAddedScore": 1.2, "PressingScore": 0.7, "DefensiveDisruptionScore": 0.5},
    "W":  {"ScoringThreatScore": 1.2, "CreativeProgressionScore": 1.2, "ExpectedThreatScore": 1.2, "DefensiveDisruptionScore": 0.8},
    "AM": {"CreativeProgressionScore": 1.4, "ExpectedThreatScore": 1.3, "ASA_GoalsAddedScore": 1.1},
    "CM": {"CreativeProgressionScore": 1.2, "PressingScore": 1.1, "BallSecurityScore": 1.2, "DefensiveDisruptionScore": 0.9},
    "DM": {"DefensiveDisruptionScore": 1.4, "PressingScore": 1.3, "BallSecurityScore": 1.2, "CreativeProgressionScore": 0.8},
    "FB": {"CreativeProgressionScore": 1.1, "DefensiveDisruptionScore": 1.2, "PressingScore": 1.1},
    "CB": {"DefensiveDisruptionScore": 1.5, "PressingScore": 1.2, "BallSecurityScore": 1.1, "ScoringThreatScore": 0.4, "ExpectedThreatScore": 0.4},
    "GK": {"BallSecurityScore": 1.2, "DefensiveDisruptionScore": 1.2},
}

WYSCOUT_DB_DIR = Path(__file__).parent / "data" / "Wyscout DB"

ALL_SCORE_COLS = list(SCORE_BLUEPRINTS.keys()) + ["AerialScore", "SetPieceScore", "CompositeRecruitmentScore"]
PROJECTION_METRICS = [
    "ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore",
    "PressingScore", "BallSecurityScore", "ExpectedThreatScore", "ASA_GoalsAddedScore",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _z_to_percentile(z: np.ndarray) -> np.ndarray:
    """Convert z-scores to 0–100 percentile using normal CDF."""
    return norm.cdf(z) * 100


def _weighted_zscore_score(
    df_group: pd.DataFrame,
    blueprint: list[tuple[str, float]],
) -> pd.Series:
    """
    Compute a position-group composite score for one score column.
    Returns a Series of 0–100 values (percentile within the group).
    """
    available = [(m, w) for m, w in blueprint if m in df_group.columns]
    if not available:
        return pd.Series(50.0, index=df_group.index)

    total_w = sum(w for _, w in available)
    z_composite = pd.Series(0.0, index=df_group.index)

    for metric, weight in available:
        col = pd.to_numeric(df_group[metric], errors="coerce").fillna(0)
        mu  = col.mean()
        sig = col.std() or 1e-9
        z_composite += (weight / total_w) * (col - mu) / sig

    return pd.Series(_z_to_percentile(z_composite.values), index=df_group.index)


# ── Main loader ────────────────────────────────────────────────────────────────

def _load_one_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
        # Take first position only
        for col in ["Position", "Pos"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.split(",").str[0].str.strip()
        df = pd.concat(
            [df, pd.Series(path.stem, index=df.index, name="_League")], axis=1
        )
        return df
    except Exception:
        return pd.DataFrame()


def load_wyscout_raw(
    min_minutes: int = 400,
    leagues: list[str] | None = None,
) -> pd.DataFrame:
    """Load raw Wyscout data from all (or selected) league files."""
    if not WYSCOUT_DB_DIR.exists():
        return pd.DataFrame()
    files = sorted(WYSCOUT_DB_DIR.glob("*.xlsx"))
    if leagues:
        files = [f for f in files if f.stem in leagues]

    frames = [_load_one_file(p) for p in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Map position → group
    pos_col = next((c for c in ["Position", "Pos"] if c in df.columns), None)
    if pos_col:
        df["PositionGroup"] = df[pos_col].map(WYSCOUT_POSITION_MAP).fillna("Other")

    # Numeric coercion
    skip = {"Player", "Team", "Position", "PositionGroup", "_League",
            "Birth country", "Passport country", "Foot", "On loan",
            "Team within selected timeframe", "Contract expires"}
    for col in df.columns:
        if col not in skip:
            c = pd.to_numeric(df[col], errors="coerce")
            if c.notna().any():
                df[col] = c

    # Minutes filter
    mins_col = next((c for c in ["Minutes played", "MinutesPlayed"] if c in df.columns), None)
    if mins_col:
        df = df.loc[pd.to_numeric(df[mins_col], errors="coerce").fillna(0) >= min_minutes]

    return df.reset_index(drop=True)


# ── Score computation ──────────────────────────────────────────────────────────

def compute_wyscout_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite score columns (0–100) to a Wyscout DataFrame.
    Scores are computed per-position group.
    Returns enriched DataFrame with IMPECT-compatible score column names.
    """
    df = df.copy()
    score_frames: list[pd.DataFrame] = []

    for pos_group, grp in df.groupby("PositionGroup"):
        grp = grp.copy()
        pos_str = str(pos_group)

        # Compute each score blueprint
        for score_col, blueprint in SCORE_BLUEPRINTS.items():
            grp[score_col] = _weighted_zscore_score(grp, blueprint)

        # CompositeRecruitmentScore — position-adjusted weighted average
        composite = pd.Series(0.0, index=grp.index)
        total_w   = 0.0
        rel       = POSITION_RELEVANCE.get(pos_str, {})
        for sc, base_w in COMPOSITE_WEIGHTS.items():
            if sc in grp.columns:
                w = base_w * rel.get(sc, 1.0)
                composite += w * grp[sc]
                total_w   += w
        grp["CompositeRecruitmentScore"] = (composite / (total_w or 1)).clip(0, 100)

        score_frames.append(grp)

    result = pd.concat(score_frames, ignore_index=True) if score_frames else df

    # Canonical column aliases so downstream code (workspaces, projections) works
    if "Player" in result.columns and "PlayerName" not in result.columns:
        result["PlayerName"] = result["Player"]
    if "Team" in result.columns and "TeamName" not in result.columns:
        result["TeamName"] = result["Team"]
    if "Age" in result.columns and "AgeYears" not in result.columns:
        result["AgeYears"] = pd.to_numeric(result["Age"], errors="coerce")
    mins_col = next((c for c in ["Minutes played", "MinutesPlayed"] if c in result.columns), None)
    if mins_col and "MinutesPlayed" not in result.columns:
        result["MinutesPlayed"] = pd.to_numeric(result[mins_col], errors="coerce")

    # BundleLabel = league name
    if "_League" in result.columns and "BundleLabel" not in result.columns:
        result["BundleLabel"] = result["_League"]

    # DecisionScore proxy (accuracy + composites)
    if "DecisionScore" not in result.columns:
        result["DecisionScore"] = (
            result.get("BallSecurityScore", pd.Series(50, index=result.index)).fillna(50) * 0.5
            + result.get("CreativeProgressionScore", pd.Series(50, index=result.index)).fillna(50) * 0.5
        )

    # ValueRecruitmentScore proxy (composite adjusted for age)
    if "ValueRecruitmentScore" not in result.columns and "AgeYears" in result.columns:
        age_bonus = (25 - result["AgeYears"].clip(16, 35)).clip(0, 9) * 2
        result["ValueRecruitmentScore"] = (result["CompositeRecruitmentScore"].fillna(50) + age_bonus).clip(0, 100)

    # AgeResaleScore
    if "AgeResaleScore" not in result.columns and "AgeYears" in result.columns:
        age_s = result["AgeYears"].fillna(25)
        result["AgeResaleScore"] = ((30 - age_s.clip(16, 35)) / 14 * 100).clip(0, 100)

    # PerformanceReliabilityScore
    if "PerformanceReliabilityScore" not in result.columns:
        mins_s = result.get("MinutesPlayed", pd.Series(900, index=result.index)).fillna(0)
        result["PerformanceReliabilityScore"] = (mins_s / 2700 * 100).clip(0, 100)

    # SuccessProbability proxy
    if "SuccessProbability" not in result.columns:
        result["SuccessProbability"] = (
            result["CompositeRecruitmentScore"].fillna(50) * 0.6
            + result.get("PerformanceReliabilityScore", pd.Series(50, index=result.index)).fillna(50) * 0.4
        ).clip(0, 100)

    return result


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_wyscout_dashboard_data(
    min_minutes: int = 400,
    leagues: list[str] | None = None,
) -> pd.DataFrame:
    """Load, score, and return a Wyscout DataFrame ready for all app workspaces."""
    raw = load_wyscout_raw(min_minutes, leagues)
    if raw.empty:
        return raw
    return compute_wyscout_scores(raw)


def available_leagues() -> list[str]:
    """Return all available Wyscout league file stems."""
    if not WYSCOUT_DB_DIR.exists():
        return []
    return sorted(p.stem for p in WYSCOUT_DB_DIR.glob("*.xlsx"))
