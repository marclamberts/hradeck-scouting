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

ALL_SCORE_COLS = list(SCORE_BLUEPRINTS.keys()) + [
    "AerialScore", "SetPieceScore", "CompositeRecruitmentScore",
    "ScoutingGrade", "UncertaintyBand", "DataReliability",
    "RatingWithBand", "GradeWithBand", "ConfidenceLabel",
]

# ── League quality index ───────────────────────────────────────────────────────
# Factor in [0.0, 1.0]: lower = stronger league (less uncertainty from context).
# Used by the confidence framework; unmapped leagues default to 0.30.
LEAGUE_QUALITY_FACTOR: dict[str, float] = {
    # ── Tier 1: Big-5 European (factor = 0.00) ──────────────────────────────
    "England": 0.00, "Germany": 0.00, "Spain": 0.00, "France": 0.00, "Italy": 0.00,
    # ── Tier 2: Strong European + elite non-European ─────────────────────────
    "Netherlands": 0.08, "Portugal": 0.08, "Russia": 0.08, "Turkiye": 0.08,
    "Belgium": 0.08, "Brazil": 0.10, "Argentina": 0.10, "Mexico": 0.10,
    "Ukraine": 0.10, "Scotland": 0.10, "Switzerland": 0.10, "Austria": 0.10,
    "Greece": 0.10, "Denmark": 0.10, "Sweden": 0.10, "Norway": 0.10,
    "Japan": 0.12, "Korea": 0.12, "USA": 0.12, "Saudi": 0.12,
    "Czech": 0.12, "Poland": 0.12, "Serbia": 0.12, "Croatia": 0.12,
    "China": 0.14,
    # ── Tier 3: Second divisions of Big-5 + solid mid-tier ───────────────────
    "England II": 0.08, "Germany II": 0.08, "Spain II": 0.10,
    "France II": 0.10, "Italy II": 0.10,
    "Netherlands II": 0.12, "Portugal II": 0.12, "Belgium II": 0.12,
    "Russia II": 0.15, "Turkiye II": 0.15, "Ukraine II": 0.15,
    "Scotland II": 0.15, "Switzerland II": 0.15, "Austria II": 0.15,
    "Greece II": 0.15, "Denmark II": 0.15, "Sweden II": 0.15,
    "Norway II": 0.15, "Czech II": 0.16, "Poland II": 0.16, "Serbia II": 0.16,
    "Brazil II": 0.15, "Argentina II": 0.15, "Mexico II": 0.15,
    "Japan II III": 0.18, "USA II": 0.18, "Korea II": 0.18, "Saudi II": 0.18,
    "China II": 0.20, "Slovakia": 0.15, "Hungary": 0.15, "Romania": 0.15,
    "Bulgaria": 0.15, "Colombia": 0.22, "Chile": 0.22, "Australia": 0.22,
    "Morocco": 0.25, "Tunisia": 0.25, "Egypt": 0.25,
    # ── Tier 4: Third divisions + lower non-European ─────────────────────────
    "England III": 0.15, "England IV": 0.20, "England V": 0.25,
    "Germany III": 0.15,
    "Germany 4 - Part I": 0.20, "Germany 4 - Part II": 0.20,
    "Germany 4 - Part III": 0.20, "Germany 4 - Part IV": 0.20,
    "Spain III": 0.18, "France III": 0.18,
    "Italy III - Part I": 0.18, "Italy III - Part II": 0.18,
    "Italy III - Part III": 0.18, "Italy III - Part IV": 0.18,
    "Netherlands III": 0.20, "Portugal III": 0.20, "Poland III": 0.22,
    "Norway III": 0.25, "Sweden III": 0.25, "Denmark III": 0.25,
    "Denmark IV": 0.28, "Scotland III": 0.25, "Scotland IV": 0.28,
    "Slovakia II": 0.22, "Slovenia": 0.22, "Slovenia II": 0.25,
    "Bosnia": 0.22, "Montenegro": 0.25, "Albania": 0.25,
    "Latvia": 0.25, "Lithuania": 0.25, "Estonia": 0.25,
    "Finland": 0.20, "Finland II": 0.25, "Iceland": 0.25,
    "Ireland": 0.22, "Ireland II": 0.28, "Northern Ireland": 0.28,
    "Wales": 0.28, "Hungary II": 0.28, "Kosovo": 0.28,
    "Georgia": 0.28, "Armenia": 0.28, "Azerbaijan": 0.28,
    "Moldovia": 0.30, "Faroe Islands": 0.35, "Malta": 0.35, "Andorra": 0.40,
    "Chile II": 0.28, "Uruguay": 0.22, "Uruguay II": 0.28,
    "Paraguay": 0.25, "Peru": 0.28, "Ecuador": 0.25, "Ecuador II": 0.30,
    "Bolivia": 0.30, "Venezuela": 0.30, "Costa Rica": 0.30,
    "Guatemala": 0.35, "Honduras": 0.35, "Panama": 0.35,
    "El Salvador": 0.35, "Nicaragua": 0.38,
    "Australia II - Part I": 0.28, "Australia II - Part II": 0.28,
    "Australia II - Part III": 0.28, "Australia II - Part IV": 0.28,
    "Australia II - Part V": 0.28, "Australia II - Part VI": 0.28,
    "Australia II - Part VII": 0.28,
    "Canada": 0.25, "USA III": 0.25, "Korea III": 0.28,
    "Nigeria": 0.30, "South Africa": 0.30,
    "Qatar": 0.28, "UAE": 0.28, "Jordan": 0.30, "Bahrain": 0.32,
    "India": 0.32, "Thailand": 0.35, "Malaysia": 0.38,
    "Indonesia": 0.38, "Vietnam": 0.38, "Philippines": 0.40,
    "Singapore": 0.40, "Cambodia": 0.45, "Hong Kong": 0.38,
    "Kazakhstan": 0.32, "Uzbekistan": 0.32, "Kyrgystan": 0.40,
    "Bangladesh": 0.45, "Cyprus": 0.25, "Cyprus II": 0.30,
    "Czech U17": 0.30, "Czech U19": 0.25,
}
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

    # ── Five-factor scouting confidence framework ─────────────────────────────
    # Produces:
    #   ScoutingGrade    (1–10)  — the scout's number for the player
    #   UncertaintyBand  (pts)   — ±X on a 0–100 scale; display as "82 ± 11"
    #   DataReliability  (0–100) — how much to trust the grade (100 = full trust)
    #   RatingWithBand           — formatted "82 ± 11" string
    #   GradeWithBand            — formatted "8.2 ± 1.1" string
    #   CF_*                     — individual factor components (for drill-down)
    if "UncertaintyBand" not in result.columns:
        _MIN_QUAL  = 400.0
        _KEY_STATS = [
            "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
            "Key passes per 90", "Passes per 90", "Accurate passes, %",
            "Dribbles per 90", "Successful defensive actions per 90",
            "Aerial duels won, %", "Progressive passes per 90",
        ]

        # Factor 1 — Sample size (35 %)
        # SE of per-90 stats ∝ 1/√(minutes/90); normalised so 400 min = 1.0
        _mins = result.get(
            "MinutesPlayed", pd.Series(_MIN_QUAL, index=result.index)
        ).fillna(_MIN_QUAL).clip(lower=_MIN_QUAL)
        _cf_sample = np.sqrt(_MIN_QUAL / _mins).clip(0, 1)

        # Factor 2 — League quality (25 %)
        _league = result.get("_League", pd.Series("", index=result.index)).fillna("")
        _cf_league = _league.map(LEAGUE_QUALITY_FACTOR).fillna(0.30)

        # Factor 3 — Age (20 %)
        # Peak certainty at 24–27; higher uncertainty for teenagers and 30+ players
        _age = result.get("AgeYears", pd.Series(25.0, index=result.index)).fillna(25)
        _cf_age = pd.Series(0.0, index=result.index, dtype=float)
        _cf_age = _cf_age.where(_age >= 17, 0.40)          # extreme youth: unknown ceiling
        _cf_age = _cf_age.where(~(_age.between(17, 19.99)), _cf_age)
        _cf_age[_age.between(17, 19.99)] = 0.35
        _cf_age[_age.between(20, 22.99)] = 0.20
        _cf_age[_age.between(23, 27.99)] = 0.00            # peak — most reliable
        _cf_age[_age.between(28, 30.99)] = 0.12
        _cf_age[_age >= 31] = 0.25

        # Factor 4 — Tactical stability / availability (10 %)
        # Proxied by average minutes-per-match; rotation players are less representative
        _matches = pd.to_numeric(
            result.get("Matches played", pd.Series(10, index=result.index)),
            errors="coerce",
        ).fillna(10).clip(lower=1)
        _avg_min = (_mins / _matches).clip(0, 90)
        _cf_avail = ((90 - _avg_min) / 90).clip(0, 1) * 0.40  # max 0.40 if 0 min/game

        # Factor 5 — Data completeness (10 %)
        # Missing key stats = less evidence to anchor the composite score
        _avail_stats = [c for c in _KEY_STATS if c in result.columns]
        if _avail_stats:
            _present = result[_avail_stats].notna().sum(axis=1)
            _cf_complete = (1 - _present / len(_avail_stats)).clip(0, 1)
        else:
            _cf_complete = pd.Series(0.20, index=result.index)

        # Combined factor (weighted sum, clamped to [0, 1])
        _combined = (
            0.35 * _cf_sample
            + 0.25 * _cf_league
            + 0.20 * _cf_age
            + 0.10 * _cf_avail
            + 0.10 * _cf_complete
        ).clip(0, 1)

        # Uncertainty band in composite-score points (0–100 scale)
        # Calibrated so: best case (top league, regular starter, peak age) ≈ ±5 pts
        #                worst case (min minutes, weak league, teen)       ≈ ±28 pts
        _comp = result.get(
            "CompositeRecruitmentScore", pd.Series(50.0, index=result.index)
        ).fillna(50).clip(0, 100)
        _band = (_comp * _combined * 0.30).clip(3, 30).round(1)

        result["ScoutingGrade"]   = (_comp / 10).clip(0, 10).round(1)
        result["UncertaintyBand"] = _band
        result["DataReliability"] = (100 - _combined * 100).clip(0, 100).round(1)

        # Store individual factors for drill-down on the uncertainty sheet
        result["CF_SampleSize"]        = (_cf_sample * 100).round(1)
        result["CF_LeagueQuality"]     = (_cf_league * 100).round(1)
        result["CF_Age"]               = (_cf_age * 100).round(1)
        result["CF_TacticalStability"] = (_cf_avail * 100).round(1)
        result["CF_DataCompleteness"]  = (_cf_complete * 100).round(1)

        # Formatted strings — "82 ± 11" and "8.2 ± 1.1"
        _rating = _comp.round(0).astype(int)
        _band_i = _band.round(0).astype(int)
        result["RatingWithBand"] = _rating.astype(str) + " ± " + _band_i.astype(str)
        result["GradeWithBand"]  = (
            result["ScoutingGrade"].astype(str)
            + " ± "
            + (_band / 10).round(1).astype(str)
        )

        def _conf_label(r: float) -> str:
            if r >= 85: return "High Confidence"
            if r >= 70: return "Good Confidence"
            if r >= 50: return "Moderate Confidence"
            if r >= 30: return "Low Confidence"
            return "Very Low Confidence"

        result["ConfidenceLabel"] = result["DataReliability"].apply(_conf_label)

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
