from __future__ import annotations

import os
from html import escape
from io import BytesIO
from pathlib import Path
from textwrap import shorten

APP_DIR = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hradeck_scouting_matplotlib")

import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st
from mplsoccer import PyPizza
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


DEFAULT_MODEL_OUTPUT_DIR = APP_DIR / "data"
MODEL_OUTPUT_FILES = {
    "recruitment": "FCHK Model V3 - Recruitment Scores.xlsx",
    "player_scores": "FCHK Model V3 - Player Scores.xlsx",
    "player_styles": "FCHK Model V3 - Player Styles.xlsx",
    "smart_club": "FCHK Model V3 - Smart Club Closeness.xlsx",
    "loaded_leagues": "FCHK Model V3 - Loaded Leagues.xlsx",
    "model_input": "FCHK Model V3 - Model Input.xlsx",
    "summary": "FCHK Model V3 - Summary.xlsx",
    "exports": "FCHK Model V3 Scores.xlsx",
}
MODEL_SHEETS = {
    "recruitment": "Recruitment Scores",
    "player_scores": "Player Scores",
    "player_styles": "Player Styles",
    "smart_club": "Smart Club Closeness",
    "loaded_leagues": "Loaded Leagues",
    "model_input": "Model Input",
    "summary": "Summary",
    "exports": "Export Files",
}
SCORE_COLUMNS = [
    "DecisionScore",
    "ValueRecruitmentScore",
    "CompositeRecruitmentScore",
    "ScoringThreatScore",
    "CreativeProgressionScore",
    "DefensiveDisruptionScore",
    "PressingScore",
    "BallSecurityScore",
    "ExpectedThreatScore",
    "ASA_GoalsAddedScore",
    "AgeResaleScore",
    "PerformanceReliabilityScore",
]
CORE_COLUMNS = [
    "PlayerName",
    "TeamName",
    "PositionGroup",
    "BundleLabel",
    "AgeYears",
    "MinutesPlayed",
    "MatchesPlayed",
    "ScoutFitScore",
    "Archetype",
    "SuccessProbability",
    "DecisionScore",
    "CompositeRecruitmentScore",
    "ValueRecruitmentScore",
    "PerformanceReliabilityScore",
    "SecurityRisk_per90",
]
POSITION_COLORS = {
    "GK": "#667085",
    "CB": "#2f5f98",
    "FB": "#00a6a6",
    "DM": "#6b8e23",
    "CM": "#2f855a",
    "AM": "#d97706",
    "W": "#e76f51",
    "ST": "#c2410c",
}
TIER_COLORS = {
    "Must scout": "#e76f51",
    "Priority": "#f4a261",
    "Shortlist": "#2a9d8f",
    "Depth": "#457b9d",
    "Watch": "#8d99ae",
}
RISK_COLORS = {
    "Low": "#2a9d8f",
    "Moderate": "#e9c46a",
    "Elevated": "#f4a261",
    "High": "#e76f51",
}
ROLE_SCORE_WEIGHTS = {
    "GK": {"CompositeRecruitmentScore": 2.0, "DecisionScore": 2.5, "PerformanceReliabilityScore": 2.5, "BallSecurityScore": 1.5, "SecurityRisk_per90": -1.5},
    "CB": {"CompositeRecruitmentScore": 2.2, "DecisionScore": 2.0, "DefensiveDisruptionScore": 2.5, "BallSecurityScore": 1.4, "PerformanceReliabilityScore": 1.8, "SecurityRisk_per90": -1.3},
    "FB": {"CompositeRecruitmentScore": 2.0, "CreativeProgressionScore": 2.0, "DefensiveDisruptionScore": 1.8, "PressingScore": 1.4, "ExpectedThreatScore": 1.3, "SecurityRisk_per90": -1.0},
    "DM": {"CompositeRecruitmentScore": 2.0, "DefensiveDisruptionScore": 2.2, "BallSecurityScore": 2.0, "CreativeProgressionScore": 1.6, "PerformanceReliabilityScore": 1.6, "SecurityRisk_per90": -1.2},
    "CM": {"CompositeRecruitmentScore": 2.2, "CreativeProgressionScore": 2.4, "BallSecurityScore": 1.8, "ExpectedThreatScore": 1.4, "DecisionScore": 1.6, "SecurityRisk_per90": -1.0},
    "AM": {"CompositeRecruitmentScore": 2.0, "CreativeProgressionScore": 2.8, "ExpectedThreatScore": 2.0, "ScoringThreatScore": 1.4, "ValueRecruitmentScore": 1.4, "SecurityRisk_per90": -0.8},
    "W": {"CompositeRecruitmentScore": 1.8, "ScoringThreatScore": 2.4, "ExpectedThreatScore": 2.2, "CreativeProgressionScore": 1.8, "PressingScore": 1.2, "SecurityRisk_per90": -0.8},
    "ST": {"CompositeRecruitmentScore": 2.0, "ScoringThreatScore": 3.0, "ExpectedThreatScore": 2.2, "DecisionScore": 1.4, "SuccessProbability": 1.6, "SecurityRisk_per90": -0.7},
}
PIZZA_METRICS = {
    "Decision": "DecisionScore",
    "Value": "ValueRecruitmentScore",
    "Composite": "CompositeRecruitmentScore",
    "Scoring": "ScoringThreatScore",
    "Creation": "CreativeProgressionScore",
    "Defense": "DefensiveDisruptionScore",
    "Pressing": "PressingScore",
    "Security": "BallSecurityScore",
    "xThreat": "ExpectedThreatScore",
    "G+": "ASA_GoalsAddedScore",
    "Resale": "AgeResaleScore",
    "Reliability": "PerformanceReliabilityScore",
}
PIZZA_CATEGORY_COLORS = {
    "Decision": "#457b9d",
    "Value": "#2a9d8f",
    "Composite": "#2a9d8f",
    "Scoring": "#e76f51",
    "Creation": "#f4a261",
    "Defense": "#2f5f98",
    "Pressing": "#6b8e23",
    "Security": "#667085",
    "xThreat": "#e76f51",
    "G+": "#457b9d",
    "Resale": "#f4a261",
    "Reliability": "#102a43",
}
EUROPE_COUNTRY_COORDS = {
    "Albania": (41.1533, 20.1683),
    "Andorra": (42.5063, 1.5218),
    "Armenia": (40.0691, 45.0382),
    "Austria": (47.5162, 14.5501),
    "Azerbaijan": (40.1431, 47.5769),
    "Belarus": (53.7098, 27.9534),
    "Belgium": (50.5039, 4.4699),
    "Bosnia and Herzegovina": (43.9159, 17.6791),
    "Bulgaria": (42.7339, 25.4858),
    "Croatia": (45.1000, 15.2000),
    "Cyprus": (35.1264, 33.4299),
    "Czech Republic": (49.8175, 15.4730),
    "Czechia": (49.8175, 15.4730),
    "Denmark": (56.2639, 9.5018),
    "England": (52.3555, -1.1743),
    "Estonia": (58.5953, 25.0136),
    "Faroe Islands": (61.8926, -6.9118),
    "Finland": (61.9241, 25.7482),
    "France": (46.2276, 2.2137),
    "Georgia": (42.3154, 43.3569),
    "Germany": (51.1657, 10.4515),
    "Gibraltar": (36.1408, -5.3536),
    "Greece": (39.0742, 21.8243),
    "Hungary": (47.1625, 19.5033),
    "Iceland": (64.9631, -19.0208),
    "Ireland": (53.1424, -7.6921),
    "Israel": (31.0461, 34.8516),
    "Italy": (41.8719, 12.5674),
    "Kazakhstan": (48.0196, 66.9237),
    "Kosovo": (42.6026, 20.9030),
    "Latvia": (56.8796, 24.6032),
    "Liechtenstein": (47.1660, 9.5554),
    "Lithuania": (55.1694, 23.8813),
    "Luxembourg": (49.8153, 6.1296),
    "Malta": (35.9375, 14.3754),
    "Moldova": (47.4116, 28.3699),
    "Montenegro": (42.7087, 19.3744),
    "Netherlands": (52.1326, 5.2913),
    "North Macedonia": (41.6086, 21.7453),
    "Northern Ireland": (54.7877, -6.4923),
    "Norway": (60.4720, 8.4689),
    "Poland": (51.9194, 19.1451),
    "Portugal": (39.3999, -8.2245),
    "Romania": (45.9432, 24.9668),
    "Russia": (55.7558, 37.6173),
    "San Marino": (43.9424, 12.4578),
    "Scotland": (56.4907, -4.2026),
    "Serbia": (44.0165, 21.0059),
    "Slovakia": (48.6690, 19.6990),
    "Slovenia": (46.1512, 14.9955),
    "Spain": (40.4637, -3.7492),
    "Sweden": (60.1282, 18.6435),
    "Switzerland": (46.8182, 8.2275),
    "Turkey": (38.9637, 35.2433),
    "Türkiye": (38.9637, 35.2433),
    "Ukraine": (48.3794, 31.1656),
    "Wales": (52.1307, -3.7837),
}


st.set_page_config(
    page_title="FCHK Scouting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    text_columns = {
        "PlayerName",
        "TeamName",
        "PositionGroup",
        "BundleLabel",
        "LeagueLabel",
        "CountryLabel",
        "TierLabel",
        "PrimaryPlayerStyle",
        "SecondaryPlayerStyle",
        "PlayerStyleSummary",
        "PlayerStyleDrivers",
        "ClosestArchetype",
        "WhyThisClubStyle",
        "SmartClubTop3",
        "SmartClubClosenessTier",
    }
    for col in df.columns:
        if col not in text_columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
    return df


def _model_file(name: str) -> Path:
    return DEFAULT_MODEL_OUTPUT_DIR / MODEL_OUTPUT_FILES[name]


def _read_model_output(name: str) -> pd.DataFrame:
    return _clean_columns(pd.read_excel(_model_file(name), sheet_name=MODEL_SHEETS[name]))


def _merge_missing_columns(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    keys = [col for col in ["PlayerName", "TeamName", "PositionGroup"] if col in base.columns and col in other.columns]
    if not keys:
        return base
    add_cols = [col for col in other.columns if col not in base.columns and col not in keys]
    if not add_cols:
        return base
    return base.merge(other[keys + add_cols].drop_duplicates(keys), on=keys, how="left")


def normalize_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "BundleLabel": ["BundleLabel", "LeagueLabel", "LeagueSlug"],
        "MatchesPlayed": ["MatchesPlayed", "matches"],
        "ScoringThreatScore": ["ScoringThreatScore", "AttackingScore", "UnderlyingThreatScore", "BoxThreatScore"],
        "CreativeProgressionScore": ["CreativeProgressionScore", "CreationScore", "CreativeAMScore"],
        "DefensiveDisruptionScore": ["DefensiveDisruptionScore", "DefendingScore", "BallWinningDMScore"],
        "PressingScore": ["PressingScore", "PressTransitionScore", "PressingForwardScore"],
        "ExpectedThreatScore": ["ExpectedThreatScore", "xThreatScore"],
        "ASA_GoalsAddedScore": ["ASA_GoalsAddedScore", "GoalsAddedScore"],
        "PerformanceReliabilityScore": ["PerformanceReliabilityScore", "ReliabilityScore", "SampleConfidence"],
        "SecurityRisk_per90": ["SecurityRisk_per90", "CriticalBallLosses_per90", "Losses_evt_per90", "BallLosses_per90"],
        "OwnHalfLosses_per90": ["OwnHalfLosses_per90", "Losses_evt_per90", "BallLosses_per90"],
        "DangerOwnHalfLosses_per90": ["DangerOwnHalfLosses_per90", "CriticalBallLosses_per90"],
        "xG_per90": ["xG_per90", "xG_model_per90", "xG_evt_per90"],
        "xA_per90": ["xA_per90"],
        "KeyPasses_per90": ["KeyPasses_per90"],
        "ProgressivePasses_per90": ["ProgressivePasses_per90"],
        "PressingDuelsWon_per90": ["PressingDuelsWon_per90"],
        "CounterpressRecovs_per90": ["CounterpressRecovs_per90"],
        "Imp_BypassedOpp_per90": ["Imp_BypassedOpp_per90", "BypassedOpp_per90"],
        "Imp_BypassedDef_per90": ["Imp_BypassedDef_per90", "BypassedDef_per90"],
        "Imp_BallWin_per90": ["Imp_BallWin_per90", "BallWins_per90"],
        "Imp_BallLoss_per90": ["Imp_BallLoss_per90", "BallLosses_per90", "Losses_evt_per90"],
    }
    for target, candidates in aliases.items():
        if target not in df:
            for candidate in candidates:
                if candidate in df:
                    df[target] = df[candidate]
                    break
    if "BundleLabel" in df:
        bundle = df["BundleLabel"].fillna("").astype(str)
        if "CountryLabel" in df:
            country = df["CountryLabel"].fillna("").astype(str)
            bundle = np.where(country.ne(""), bundle + " · " + country, bundle)
        if "TierLabel" in df:
            tier = df["TierLabel"].fillna("").astype(str)
            bundle = np.where(tier.ne(""), pd.Series(bundle, index=df.index) + " · " + tier, bundle)
        df["BundleLabel"] = pd.Series(bundle, index=df.index).replace("", "Unknown league")
    if "IsU23Target" not in df and "AgeYears" in df:
        df["IsU23Target"] = safe_col(df, "AgeYears").le(23)
    return df


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    required_file = _model_file("recruitment")
    if not required_file.exists():
        st.error(f"Missing required model output: {required_file}")
        st.stop()

    df = _read_model_output("recruitment")
    for name in ["player_scores", "player_styles", "smart_club"]:
        if _model_file(name).exists():
            df = _merge_missing_columns(df, _read_model_output(name))
    return normalize_model_columns(df)


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict[str, pd.DataFrame]:
    metadata: dict[str, pd.DataFrame] = {}
    for name in MODEL_OUTPUT_FILES:
        if _model_file(name).exists():
            metadata[name] = _read_model_output(name)
    return metadata


def file_inventory_frame(metadata: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, file_name in MODEL_OUTPUT_FILES.items():
        path = _model_file(name)
        frame = metadata.get(name)
        rows.append(
            {
                "File": file_name,
                "Purpose": name.replace("_", " ").title(),
                "Status": "Loaded" if frame is not None else "Missing",
                "Rows": 0 if frame is None else len(frame),
                "Columns": 0 if frame is None else len(frame.columns),
                "Sheet": MODEL_SHEETS[name],
                "Path": str(path),
            }
        )
    return pd.DataFrame(rows)


def coverage_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "Player identity": ["PlayerName", "TeamName", "PositionGroup"],
        "League context": ["LeagueLabel", "CountryLabel", "TierLabel", "SeasonLabel"],
        "Core recruitment scores": ["DecisionScore", "ValueRecruitmentScore", "CompositeRecruitmentScore", "SuccessProbability"],
        "Style profile": ["PrimaryPlayerStyle", "SecondaryPlayerStyle", "PlayerStyleSummary", "ClosestArchetype"],
        "Smart club fit": ["SmartClubScore", "SmartClubTop3", "SmartClubClosenessTier"],
        "Performance value": ["GoalsAddedScore", "RAPMScore", "PVScore", "xThreatScore", "ActionValueScore"],
        "Per-90 event detail": ["xG_model_per90", "xA_per90", "Shots_per90", "ProgressivePasses_per90", "CriticalBallLosses_per90"],
        "Risk and confidence": ["SampleConfidence", "ConfidenceBand", "DataCoverageScore", "MinutesRiskFlag", "DataCoverageFlag"],
        "Scouting workflow": ["VideoReviewed", "LiveReviewed", "ScoutGrade", "ScoutFitGrade", "ModelScoutAgreement"],
    }
    rows = []
    for area, cols in checks.items():
        present = [c for c in cols if c in df.columns]
        populated = 0.0
        if present:
            populated = float(df[present].notna().mean().mean() * 100)
        rows.append(
            {
                "Area": area,
                "Columns present": f"{len(present)}/{len(cols)}",
                "Population": populated,
                "Available columns": ", ".join(present) if present else "None",
            }
        )
    return pd.DataFrame(rows)


def top_missing_columns(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    missing = df.isna().mean().sort_values(ascending=False).head(limit).reset_index()
    missing.columns = ["Column", "Missing share"]
    missing["Missing share"] = missing["Missing share"] * 100
    return missing


def country_name_series(df: pd.DataFrame) -> pd.Series:
    if "CountryLabel" in df.columns:
        country = df["CountryLabel"].fillna("").astype(str).str.strip()
    else:
        country = pd.Series("", index=df.index)
    if country.eq("").all() and "BundleLabel" in df.columns:
        country = df["BundleLabel"].fillna("").astype(str).str.split(" · ").str[-1].str.strip()
    aliases = {
        "UK": "England",
        "United Kingdom": "England",
        "Czech Rep.": "Czechia",
        "Czech Republic": "Czechia",
        "Turkiye": "Türkiye",
        "Macedonia": "North Macedonia",
        "Bosnia": "Bosnia and Herzegovina",
    }
    return country.replace(aliases)


def european_market_map_frame(df: pd.DataFrame, metric: str = "ScoutFitScore") -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working["Country"] = country_name_series(working)
    working = working.loc[working["Country"].isin(EUROPE_COUNTRY_COORDS)].copy()
    if working.empty:
        return pd.DataFrame()
    market = (
        working.groupby("Country")
        .agg(
            Players=("PlayerName", "count"),
            MedianScore=(metric, "median"),
            TopScore=(metric, "max"),
            HighQuality=("QualityTier", lambda x: x.isin(["High quality", "Elite"]).sum()),
            MedianAge=("AgeYears", "median"),
        )
        .round(1)
        .reset_index()
    )
    market["HighQualityShare"] = np.where(market["Players"].gt(0), market["HighQuality"] / market["Players"] * 100, 0).round(1)
    market["lat"] = market["Country"].map(lambda c: EUROPE_COUNTRY_COORDS[c][0])
    market["lon"] = market["Country"].map(lambda c: EUROPE_COUNTRY_COORDS[c][1])
    market["GoScore"] = (market["MedianScore"] * 0.65 + market["HighQualityShare"] * 0.25 + np.minimum(market["Players"], 80) * 0.10).round(1)
    market["Recommendation"] = pd.cut(
        market["GoScore"],
        bins=[-np.inf, 42, 52, 62, np.inf],
        labels=["Monitor", "Watch trip", "Scout next", "Go now"],
    ).astype(str)
    return market.sort_values("GoScore", ascending=False)


def safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def assign_archetypes(df: pd.DataFrame) -> pd.Series:
    attack = safe_col(df, "ScoringThreatScore") + safe_col(df, "ExpectedThreatScore")
    creation = safe_col(df, "CreativeProgressionScore") + safe_col(df, "KeyPasses_per90", 50)
    progression = safe_col(df, "ProgressivePasses_per90") + safe_col(df, "PassesToFinalThird_per90")
    defense = safe_col(df, "DefensiveDisruptionScore") + safe_col(df, "Imp_BallWin_per90", 50)
    press = safe_col(df, "PressingScore") + safe_col(df, "CounterpressRecovs_per90", 50)
    security = safe_col(df, "BallSecurityScore") - safe_col(df, "SecurityRisk_per90")
    value = safe_col(df, "ValueRecruitmentScore") + safe_col(df, "AgeResaleScore")
    position = df.get("PositionGroup", pd.Series("", index=df.index)).fillna("").astype(str)

    labels = np.select(
        [
            position.eq("GK"),
            attack.ge(135),
            creation.ge(120),
            progression.ge(65) & position.isin(["CM", "DM", "FB"]),
            defense.ge(120) & position.isin(["CB", "DM", "FB"]),
            press.ge(105),
            security.ge(62),
            value.ge(125),
        ],
        [
            "Shot-stopping GK",
            "Final-third finisher",
            "Chance creator",
            "Progressive carrier/passer",
            "Defensive disruptor",
            "Pressing catalyst",
            "Secure connector",
            "Value upside play",
        ],
        default="Balanced profile",
    )
    return pd.Series(labels, index=df.index)


def weighted_role_score(row: pd.Series) -> float:
    weights = ROLE_SCORE_WEIGHTS.get(str(row.get("PositionGroup", "")), ROLE_SCORE_WEIGHTS["CM"])
    positive_total = sum(v for v in weights.values() if v > 0)
    raw = 0.0
    for col, weight in weights.items():
        value = pd.to_numeric(row.get(col, 0), errors="coerce")
        value = 0 if pd.isna(value) else float(value)
        if weight < 0:
            raw += max(0, 100 - (value * 6)) * abs(weight)
        else:
            raw += value * weight
    return float(np.clip(raw / max(positive_total + sum(abs(v) for v in weights.values() if v < 0), 1), 0, 100))


def fit_drivers(row: pd.Series, limit: int = 3) -> str:
    candidates = {
        "Composite": row.get("CompositeRecruitmentScore", 0),
        "Decision": row.get("DecisionScore", 0),
        "Value": row.get("ValueRecruitmentScore", 0),
        "Scoring": row.get("ScoringThreatScore", 0),
        "Creation": row.get("CreativeProgressionScore", 0),
        "Defense": row.get("DefensiveDisruptionScore", 0),
        "Pressing": row.get("PressingScore", 0),
        "Security": row.get("BallSecurityScore", 0),
        "xThreat": row.get("ExpectedThreatScore", 0),
        "Reliability": row.get("PerformanceReliabilityScore", 0),
    }
    top = sorted(candidates.items(), key=lambda item: pd.to_numeric(item[1], errors="coerce"), reverse=True)[:limit]
    return ", ".join(f"{label} {float(score):.0f}" for label, score in top)


def quality_drivers(row: pd.Series, limit: int = 3) -> str:
    candidates = {
        "Role": row.get("RoleFitScore", 0),
        "Impact": row.get("ProfileScore", 0),
        "Decision": row.get("DecisionScore", 0),
        "Reliability": row.get("PerformanceReliabilityScore", 0),
        "Threat": row.get("ExpectedThreatScore", 0),
        "Security": row.get("BallSecurityScore", 0),
        "Creation": row.get("CreativeProgressionScore", 0),
        "Defense": row.get("DefensiveDisruptionScore", 0),
        "Pressing": row.get("PressingScore", 0),
    }
    top = sorted(candidates.items(), key=lambda item: pd.to_numeric(item[1], errors="coerce"), reverse=True)[:limit]
    return ", ".join(f"{label} {float(score):.0f}" for label, score in top)


def risk_flags(row: pd.Series) -> list[str]:
    flags = []
    if row.get("MinutesPlayed", 0) < 900:
        flags.append("low minutes")
    if row.get("PerformanceReliabilityScore", 100) < 55:
        flags.append("low reliability")
    if row.get("SecurityRisk_per90", 0) > 13:
        flags.append("high security risk")
    if row.get("OwnHalfLosses_per90", 0) > 3:
        flags.append("own-half losses")
    if row.get("DangerOwnHalfLosses_per90", 0) > 0.5:
        flags.append("danger losses")
    if row.get("TeamPossProxy", 1) > 1.25:
        flags.append("high-possession context")
    return flags or ["clean profile"]


def tier_reason(row: pd.Series) -> str:
    return (
        f"{row.get('MarketTier', 'Watch')} because Scout Fit is {row.get('ScoutFitScore', 0):.1f}, "
        f"driven by {fit_drivers(row)}. Main risk check: {', '.join(risk_flags(row)[:2])}."
    )


@st.cache_data(show_spinner=False)
def add_scouting_fields(df: pd.DataFrame, weights: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["Archetype"] = assign_archetypes(out)
    numerator = (
        safe_col(out, "CompositeRecruitmentScore") * weights["Composite"]
        + safe_col(out, "DecisionScore") * weights["Decision"]
        + safe_col(out, "ValueRecruitmentScore") * weights["Value"]
        + safe_col(out, "SuccessProbability") * weights["Success"]
        + safe_col(out, "PerformanceReliabilityScore") * weights["Reliability"]
        - safe_col(out, "SecurityRisk_per90") * weights["Risk penalty"]
    )
    denominator = max(sum(v for k, v in weights.items() if k != "Risk penalty"), 1)
    out["ModelFitScore"] = (numerator / denominator).clip(0, 100)
    out["RoleFitScore"] = out.apply(weighted_role_score, axis=1)
    out["ScoutFitScore"] = ((out["ModelFitScore"] * 0.58) + (out["RoleFitScore"] * 0.42)).clip(0, 100)
    out["AgeYears"] = safe_col(out, "AgeYears").round(1)
    out["RiskBand"] = pd.cut(
        safe_col(out, "SecurityRisk_per90"),
        bins=[-np.inf, 5, 9, 13, np.inf],
        labels=["Low", "Moderate", "Elevated", "High"],
    ).astype(str)
    out["Readiness"] = pd.cut(
        safe_col(out, "PerformanceReliabilityScore"),
        bins=[-np.inf, 55, 70, 85, np.inf],
        labels=["Monitor", "Develop", "Ready", "High confidence"],
    ).astype(str)
    out["MarketTier"] = pd.cut(
        out["ScoutFitScore"],
        bins=[-np.inf, 42, 50, 58, 66, np.inf],
        labels=["Watch", "Depth", "Shortlist", "Priority", "Must scout"],
    ).astype(str)
    out["ProfileScore"] = (
        safe_col(out, "ScoringThreatScore")
        + safe_col(out, "CreativeProgressionScore")
        + safe_col(out, "DefensiveDisruptionScore")
        + safe_col(out, "BallSecurityScore")
    ) / 4
    out["QualityScore"] = (
        safe_col(out, "RoleFitScore") * 0.34
        + safe_col(out, "ProfileScore") * 0.24
        + safe_col(out, "DecisionScore") * 0.16
        + safe_col(out, "PerformanceReliabilityScore") * 0.14
        + safe_col(out, "ExpectedThreatScore") * 0.07
        + safe_col(out, "BallSecurityScore") * 0.05
    ).clip(0, 100)
    out["QualityTier"] = pd.cut(
        out["QualityScore"],
        bins=[-np.inf, 42, 52, 62, 72, np.inf],
        labels=["Monitor", "Useful", "Good", "High quality", "Elite"],
    ).astype(str)
    out["FitDrivers"] = out.apply(fit_drivers, axis=1)
    out["QualityDrivers"] = out.apply(quality_drivers, axis=1)
    out["RiskFlags"] = out.apply(lambda row: ", ".join(risk_flags(row)), axis=1)
    out["TierReason"] = out.apply(tier_reason, axis=1)
    return out


def percentile_rank(series: pd.Series, value: float) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float((series.le(value).mean() * 100).round(1))


def player_strengths(row: pd.Series) -> str:
    focus = {
        "Scoring": row.get("ScoringThreatScore", 0),
        "Creation": row.get("CreativeProgressionScore", 0),
        "Defense": row.get("DefensiveDisruptionScore", 0),
        "Pressing": row.get("PressingScore", 0),
        "Security": row.get("BallSecurityScore", 0),
        "xThreat": row.get("ExpectedThreatScore", 0),
    }
    top = sorted(focus.items(), key=lambda item: item[1], reverse=True)[:3]
    return " / ".join(f"{name} {score:.0f}" for name, score in top)


def profile_note(row: pd.Series) -> str:
    return (
        f"{row['Archetype']} with {player_strengths(row)}. "
        f"Reliability: {row['Readiness']}; risk: {row['RiskBand']}; market tier: {row['MarketTier']}."
    )


def percentile_values(df: pd.DataFrame, row: pd.Series, metrics: dict[str, str]) -> list[float]:
    values = []
    for col in metrics.values():
        values.append(percentile_rank(df[col], float(row[col])) if col in df else 0.0)
    return values


def similarity_features(df: pd.DataFrame) -> list[str]:
    base = [
        "ScoringThreatScore",
        "CreativeProgressionScore",
        "DefensiveDisruptionScore",
        "PressingScore",
        "BallSecurityScore",
        "ExpectedThreatScore",
        "ASA_GoalsAddedScore",
        "xG_per90",
        "xA_per90",
        "Shots_per90",
        "KeyPasses_per90",
        "ProgressivePasses_per90",
        "PassesToFinalThird_per90",
        "PressingDuelsWon_per90",
        "Imp_BypassedOpp_per90",
        "Imp_BypassedDef_per90",
        "Imp_BallWin_per90",
        "Imp_BallLoss_per90",
    ]
    return [col for col in base if col in df.columns]


def similar_players(df: pd.DataFrame, row: pd.Series, same_position: bool = True, n: int = 10) -> pd.DataFrame:
    pool = df.copy()
    if same_position:
        pool = pool.loc[pool["PositionGroup"].eq(row["PositionGroup"])].copy()
    features = similarity_features(pool)
    matrix = pool[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    center = matrix.mean()
    spread = matrix.std().replace(0, 1)
    z = (matrix - center) / spread
    target = (pd.to_numeric(row[features], errors="coerce").fillna(0) - center) / spread
    pool["SimilarityScore"] = (100 - np.sqrt(((z - target) ** 2).sum(axis=1)) * 8).clip(0, 100)
    pool = pool.loc[pool["PlayerName"].ne(row["PlayerName"])]
    return pool.sort_values("SimilarityScore", ascending=False).head(n)


def render_player_pizza(reference_df: pd.DataFrame, row: pd.Series):
    params = list(PIZZA_METRICS.keys())
    values = percentile_values(reference_df, row, PIZZA_METRICS)
    slice_colors = [PIZZA_CATEGORY_COLORS[param] for param in params]
    value_box_colors = ["#ffffff"] * len(params)
    value_text_colors = ["#111827"] * len(params)
    baker = PyPizza(
        params=params,
        background_color="#f4f7f9",
        straight_line_color="#d9e2e7",
        straight_line_lw=1,
        last_circle_color="#102a43",
        last_circle_lw=1.4,
        other_circle_color="#d9e2e7",
        other_circle_lw=0.7,
        inner_circle_size=18,
    )
    fig, _ = baker.make_pizza(
        values,
        figsize=(8, 8),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=value_text_colors,
        value_bck_colors=value_box_colors,
        blank_alpha=0.22,
        kwargs_slices={"edgecolor": "#ffffff", "linewidth": 1.1},
        kwargs_params={"color": "#10212b", "fontsize": 9, "fontweight": "bold"},
        kwargs_values={
            "color": "#111827",
            "fontsize": 9,
            "fontweight": "bold",
            "bbox": {
                "boxstyle": "round,pad=0.28",
                "facecolor": "#ffffff",
                "edgecolor": "#111827",
                "linewidth": 0.8,
            },
        },
    )
    fig.text(0.5, 0.975, row["PlayerName"], ha="center", va="center", fontsize=18, fontweight="bold", color="#10212b")
    fig.text(
        0.5,
        0.945,
        f"{row['TeamName']} | {row['PositionGroup']} | percentiles vs selected reference pool",
        ha="center",
        va="center",
        fontsize=10,
        color="#667085",
    )
    return fig


def render_score_distribution(df: pd.DataFrame, metric: str, highlight: float | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor("#f4f7f9")
    ax.set_facecolor("#ffffff")
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    ax.hist(values, bins=28, color="#457b9d", alpha=0.82, edgecolor="#ffffff", linewidth=0.8)
    ax.axvline(values.median(), color="#102a43", linewidth=2, label=f"Median {values.median():.1f}")
    if highlight is not None:
        ax.axvline(highlight, color="#e76f51", linewidth=2.5, label=f"Selected {highlight:.1f}")
    ax.set_title(f"{metric} distribution", loc="left", fontsize=14, fontweight="bold", color="#10212b")
    ax.set_xlabel("Score")
    ax.set_ylabel("Players")
    ax.grid(axis="y", color="#d9e2e7", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def render_league_heatmap(df: pd.DataFrame, metric: str):
    pivot = (
        df.pivot_table(index="BundleLabel", columns="PositionGroup", values=metric, aggfunc="median")
        .reindex(columns=["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"])
        .dropna(how="all")
        .sort_index()
    )
    if len(pivot) > 18:
        top_leagues = df.groupby("BundleLabel")["PlayerName"].count().sort_values(ascending=False).head(18).index
        pivot = pivot.loc[pivot.index.intersection(top_leagues)]
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(pivot) * 0.34)), dpi=150)
    fig.patch.set_facecolor("#f4f7f9")
    im = ax.imshow(pivot.fillna(np.nan), cmap="YlGnBu", aspect="auto", vmin=20, vmax=75)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index, fontsize=8)
    ax.set_title(f"League depth heatmap | median {metric}", loc="left", fontsize=14, fontweight="bold", color="#10212b")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="#10212b")
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def render_position_boxplot(df: pd.DataFrame, metric: str):
    order = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
    groups = [pd.to_numeric(df.loc[df["PositionGroup"].eq(pos), metric], errors="coerce").dropna() for pos in order]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    fig.patch.set_facecolor("#f4f7f9")
    ax.set_facecolor("#ffffff")
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=order, showfliers=False)
    for patch, pos in zip(bp["boxes"], order):
        patch.set_facecolor(POSITION_COLORS.get(pos, "#457b9d"))
        patch.set_alpha(0.72)
        patch.set_edgecolor("#10212b")
    for item in bp["medians"]:
        item.set_color("#ffffff")
        item.set_linewidth(2)
    ax.set_title(f"{metric} by position", loc="left", fontsize=14, fontweight="bold", color="#10212b")
    ax.set_ylabel("Score")
    ax.grid(axis="y", color="#d9e2e7", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def metric_card(label: str, value: str, caption: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption or ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pdf_table(rows: list[list[str]], header_color: str = "#102a43") -> Table:
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f9")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d8e2e7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _format_pdf_frame(frame: pd.DataFrame, max_text: int = 30) -> list[list[str]]:
    out = frame.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: f"{x:.1f}")
    out = out.map(lambda x: shorten(str(x), width=max_text, placeholder="..."))
    return [list(out.columns)] + out.values.tolist()


def build_pdf(df: pd.DataFrame, title: str, scope_note: str = "Filtered view", top_n: int = 50) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=28,
        rightMargin=28,
        topMargin=28,
        bottomMargin=24,
    )
    styles = getSampleStyleSheet()
    sort_col = "QualityScore" if "QualityScore" in df.columns else "ScoutFitScore"
    sorted_df = df.sort_values(sort_col, ascending=False).copy()
    high_quality_count = sorted_df["QualityTier"].isin(["High quality", "Elite"]).sum() if "QualityTier" in sorted_df else 0
    median_quality = sorted_df[sort_col].median() if not sorted_df.empty else 0
    median_age = sorted_df["AgeYears"].median() if not sorted_df.empty else 0

    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(scope_note, styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Executive Summary", styles["Heading2"]),
        _pdf_table(
            [
                ["Players", "Median quality", "Median age", "High quality"],
                [f"{len(sorted_df):,}", f"{median_quality:.1f}", f"{median_age:.1f}", f"{high_quality_count:,}"],
            ],
            header_color="#2a9d8f",
        ),
        Spacer(1, 10),
    ]

    if not sorted_df.empty:
        pos_summary = (
            sorted_df.groupby("PositionGroup")
            .agg(
                Players=("PlayerName", "count"),
                MedianQuality=(sort_col, "median"),
                MedianAge=("AgeYears", "median"),
                HighQuality=("QualityTier", lambda x: x.isin(["High quality", "Elite"]).sum()),
            )
            .round(1)
            .reset_index()
            .sort_values("MedianQuality", ascending=False)
        )
        story.extend(
            [
                Paragraph("Position Summary", styles["Heading2"]),
                _pdf_table(_format_pdf_frame(pos_summary, max_text=24), header_color="#457b9d"),
                Spacer(1, 10),
            ]
        )

        archetype_summary = (
            sorted_df["Archetype"]
            .value_counts()
            .head(10)
            .rename_axis("Archetype")
            .reset_index(name="Players")
        )
        story.extend(
            [
                Paragraph("Top Archetypes", styles["Heading2"]),
                _pdf_table(_format_pdf_frame(archetype_summary, max_text=34), header_color="#102a43"),
                Spacer(1, 10),
            ]
        )

        top_u23 = sorted_df.loc[sorted_df.get("IsU23Target", False).astype(bool)].head(15)
        if not top_u23.empty:
            top_u23 = top_u23[
                [
                    "PlayerName",
                    "TeamName",
                    "PositionGroup",
                    "AgeYears",
                    sort_col,
                    "RoleFitScore",
                    "QualityTier",
                    "QualityDrivers",
                ]
            ].rename(
                columns={
                    "PlayerName": "Player",
                    "TeamName": "Team",
                    "PositionGroup": "Pos",
                    "AgeYears": "Age",
                    sort_col: "Quality",
                    "RoleFitScore": "Role Fit",
                    "QualityDrivers": "Drivers",
                }
            )
            story.extend(
                [
                    Paragraph("Top U23 Targets", styles["Heading2"]),
                    _pdf_table(_format_pdf_frame(top_u23, max_text=32), header_color="#2a9d8f"),
                    Spacer(1, 10),
                ]
            )

        xi_order = ["GK", "CB", "CB", "FB", "FB", "DM", "CM", "AM", "W", "W", "ST"]
        used_names = set()
        xi_rows = []
        for slot in xi_order:
            candidates = sorted_df.loc[sorted_df["PositionGroup"].eq(slot) & ~sorted_df["PlayerName"].isin(used_names)]
            if candidates.empty:
                continue
            pick = candidates.iloc[0]
            used_names.add(pick["PlayerName"])
            xi_rows.append(
                {
                    "Role": slot,
                    "Player": pick["PlayerName"],
                    "Team": pick["TeamName"],
                    "Age": pick["AgeYears"],
                    "Quality": pick[sort_col],
                    "Role Fit": pick.get("RoleFitScore", 0),
                    "Drivers": pick.get("QualityDrivers", ""),
                }
            )
        if xi_rows:
            story.extend(
                [
                    Paragraph("Best XI From Scope", styles["Heading2"]),
                    _pdf_table(_format_pdf_frame(pd.DataFrame(xi_rows), max_text=32), header_color="#e76f51"),
                    Spacer(1, 10),
                ]
            )

    table_cols = [
        "PlayerName",
        "TeamName",
        "PositionGroup",
        "AgeYears",
        sort_col,
        "QualityTier",
        "Archetype",
        "CompositeRecruitmentScore",
        "DecisionScore",
        "Readiness",
        "RiskBand",
        "RoleFitScore",
        "ProfileScore",
        "QualityDrivers",
        "RiskFlags",
    ]
    export = sorted_df.head(top_n)
    export = export[[c for c in table_cols if c in export.columns]].copy()
    export = export.rename(
        columns={
            "PlayerName": "Player",
            "TeamName": "Team",
            "PositionGroup": "Pos",
            "AgeYears": "Age",
            sort_col: "Quality",
            "CompositeRecruitmentScore": "Composite",
            "DecisionScore": "Decision",
            "RiskBand": "Risk",
            "RoleFitScore": "Role Fit",
            "ProfileScore": "Impact",
            "QualityDrivers": "Drivers",
            "RiskFlags": "Risk Flags",
        }
    )
    story.extend(
        [
            Paragraph(f"Top {min(top_n, len(export))} Targets", styles["Heading2"]),
            _pdf_table(_format_pdf_frame(export, max_text=28), header_color="#e76f51"),
        ]
    )
    doc.build(story)
    return buffer.getvalue()


def download_name(prefix: str, suffix: str) -> str:
    return f"fchk_{prefix.lower().replace(' ', '_')}.{suffix}"


def reset_filters() -> None:
    for key in [
        "positions_filter",
        "bundles_filter",
        "archetypes_filter",
        "u23_filter",
        "age_filter",
        "minutes_filter",
        "fit_floor",
        "quality_floor",
        "composite_floor",
        "reliability_floor",
        "max_risk",
        "search_filter",
    ]:
        st.session_state.pop(key, None)
    st.session_state["quick_mode"] = "Full board"


def add_to_shortlist(player_name: str) -> None:
    shortlist = set(st.session_state.get("shortlist_players", []))
    shortlist.add(player_name)
    st.session_state["shortlist_players"] = sorted(shortlist)


def clear_shortlist() -> None:
    st.session_state["shortlist_players"] = []


def set_quick_mode(mode: str) -> None:
    st.session_state["quick_mode"] = mode
    if mode == "U23 quality":
        st.session_state["u23_filter"] = True
        st.session_state["fit_floor"] = 35
        st.session_state["quality_floor"] = 52
        st.session_state["max_risk"] = 18.0
    elif mode == "Elite quality":
        st.session_state["fit_floor"] = 35
        st.session_state["quality_floor"] = 68
        st.session_state["composite_floor"] = 45
        st.session_state["reliability_floor"] = 55
    elif mode == "Reliable quality":
        st.session_state["max_risk"] = 9.0
        st.session_state["reliability_floor"] = 70
        st.session_state["quality_floor"] = 55
    else:
        reset_filters()


WORKSPACES = ["Scouting", "Recruitment", "Goalkeepers", "Team", "Model"]


def set_workspace(section: str) -> None:
    st.session_state["active_workspace"] = section
    st.session_state["show_scouting_workspace"] = True
    st.session_state.pop("landing_notice", None)


def enter_scouting_workspace() -> None:
    set_workspace("Scouting")


def render_workspace_nav(location: str = "top") -> None:
    st.markdown("<div class='workspace-nav-spacer'></div>", unsafe_allow_html=True)
    nav_cols = st.columns(len(WORKSPACES))
    active = st.session_state.get("active_workspace", "Scouting")
    for idx, section in enumerate(WORKSPACES):
        with nav_cols[idx]:
            st.button(
                section,
                key=f"workspace_{location}_{section}",
                type="primary" if active == section else "secondary",
                width="stretch",
                on_click=set_workspace,
                args=(section,),
            )


BALANCED_WEIGHTS = {"Composite": 3, "Decision": 2, "Value": 2, "Success": 1, "Reliability": 1, "Risk penalty": 1}


def czech_market(df: pd.DataFrame) -> pd.DataFrame:
    country = df.get("CountryLabel", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    league = df.get("LeagueLabel", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    return df.loc[country.eq("czechia") | country.eq("czech republic") | league.str.contains("fortuna|chance narodni|czech", regex=True)].copy()


def hradec_squad(df: pd.DataFrame) -> pd.DataFrame:
    team = df.get("TeamName", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    return df.loc[team.str.contains("hradec|kralove|králové", regex=True)].copy()


def render_model_workspace(data: pd.DataFrame, metadata: dict[str, pd.DataFrame]) -> None:
    st.markdown("<div class='workspace-label'>Model workspace</div>", unsafe_allow_html=True)
    st.subheader("Smart club model")

    smart_df = metadata.get("smart_club", pd.DataFrame()).copy()
    if smart_df.empty:
        st.info("Smart Club Closeness workbook not loaded.")
        return

    score_col = "SmartClubClosenessScore"
    model_col = "SmartClubModel"
    required_cols = {model_col, score_col}
    if not required_cols.issubset(smart_df.columns):
        st.info("Smart Club Closeness workbook is loaded, but the expected model and closeness columns were not found.")
        st.dataframe(smart_df.head(50), width="stretch", hide_index=True)
        return

    smart_df[score_col] = pd.to_numeric(smart_df[score_col], errors="coerce")
    smart_df = smart_df.dropna(subset=[model_col, score_col])
    if smart_df.empty:
        st.info("Smart Club Closeness workbook has no usable model rows after cleaning.")
        return

    summary = (
        smart_df.groupby(model_col)
        .agg(
            Players=("PlayerName", "count"),
            MedianCloseness=(score_col, "median"),
            BestCloseness=(score_col, "max"),
            EliteLinks=(score_col, lambda values: values.ge(90).sum()),
        )
        .round(1)
        .reset_index()
        .sort_values(["MedianCloseness", "BestCloseness"], ascending=False)
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Club models", f"{summary[model_col].nunique():,}", "smart-club profiles")
    with metric_cols[1]:
        metric_card("Player links", f"{len(smart_df):,}", "model comparisons")
    with metric_cols[2]:
        metric_card("Median closeness", f"{smart_df[score_col].median():.1f}", "all links")
    with metric_cols[3]:
        metric_card("90+ links", f"{smart_df[score_col].ge(90).sum():,}", "elite model matches")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Club model ranking")
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
            column_config={
                "MedianCloseness": st.column_config.ProgressColumn("Median", min_value=0, max_value=100, format="%.1f"),
                "BestCloseness": st.column_config.ProgressColumn("Best", min_value=0, max_value=100, format="%.1f"),
            },
        )

    with right:
        selected_model = st.selectbox("Club model", summary[model_col].tolist())
        selected = smart_df.loc[smart_df[model_col].eq(selected_model)].copy()
        st.subheader("Position spread")
        if "PositionGroup" in selected.columns:
            position_summary = (
                selected.groupby("PositionGroup")
                .agg(Players=("PlayerName", "count"), MedianCloseness=(score_col, "median"))
                .round(1)
                .reset_index()
                .sort_values("Players", ascending=False)
            )
            st.dataframe(
                position_summary,
                width="stretch",
                hide_index=True,
                column_config={
                    "MedianCloseness": st.column_config.ProgressColumn("Median", min_value=0, max_value=100, format="%.1f"),
                },
            )
        else:
            st.info("No position detail found for this club model.")

    st.subheader(f"Top matches for {selected_model}")
    top_cols = [
        col
        for col in [
            "SmartClubRank",
            "PlayerName",
            "TeamName",
            "PositionGroup",
            "LeagueLabel",
            "AgeYears",
            "MinutesPlayed",
            "SmartClubClosenessScore",
        ]
        if col in selected.columns
    ]
    sort_cols = [score_col] + (["PlayerName"] if "PlayerName" in selected.columns else [])
    top_matches = selected.sort_values(sort_cols, ascending=[False] + ([True] if len(sort_cols) > 1 else [])).head(100)
    st.dataframe(
        top_matches[top_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "SmartClubClosenessScore": st.column_config.ProgressColumn("Closeness", min_value=0, max_value=100, format="%.1f"),
            "AgeYears": st.column_config.NumberColumn("Age", format="%.1f"),
            "MinutesPlayed": st.column_config.NumberColumn("Minutes", format="%d"),
        },
    )

    style_df = metadata.get("player_styles", pd.DataFrame()).copy()
    if not style_df.empty and "SmartClubTop3" in style_df.columns:
        related_styles = style_df.loc[
            style_df["SmartClubTop3"].fillna("").astype(str).str.contains(str(selected_model), case=False, regex=False)
        ].copy()
        if not related_styles.empty:
            st.subheader("Style notes linked to this model")
            style_cols = [
                col
                for col in [
                    "PlayerName",
                    "TeamName",
                    "PositionGroup",
                    "PrimaryPlayerStyle",
                    "SecondaryPlayerStyle",
                    "ClosestArchetype",
                    "WhyThisClubStyle",
                    "SmartClubTop3",
                    "SmartClubClosenessTier",
                ]
                if col in related_styles.columns
            ]
            st.dataframe(related_styles[style_cols].head(50), width="stretch", hide_index=True)


def render_goalkeepers_workspace(data: pd.DataFrame) -> None:
    keeper_df = add_scouting_fields(data, BALANCED_WEIGHTS)
    keeper_df = keeper_df.loc[keeper_df["PositionGroup"].astype(str).eq("GK")].sort_values(
        ["ScoutFitScore", "PerformanceReliabilityScore"],
        ascending=False,
    )

    st.markdown("<div class='workspace-label'>Goalkeepers workspace</div>", unsafe_allow_html=True)
    st.subheader("Goalkeeper board")

    if keeper_df.empty:
        st.info("No goalkeeper rows were found in the loaded model outputs.")
        return

    top_keeper = keeper_df.head(1)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Goalkeepers", f"{len(keeper_df):,}", "dedicated GK pool")
    with metric_cols[1]:
        metric_card("Best fit", f"{top_keeper.iloc[0]['ScoutFitScore']:.1f}", str(top_keeper.iloc[0]["PlayerName"]))
    with metric_cols[2]:
        metric_card("Median reliability", f"{keeper_df['PerformanceReliabilityScore'].median():.1f}", "sample confidence")
    with metric_cols[3]:
        metric_card("U23 keepers", f"{keeper_df['IsU23Target'].fillna(False).sum():,}", "age upside")

    st.markdown(
        """
        <div class="note-box">
            Goalkeepers are kept out of the Scouting workspace and collected here so the board only compares GK profiles with other GK profiles.
        </div>
        """,
        unsafe_allow_html=True,
    )

    league_options = sorted(keeper_df["BundleLabel"].dropna().astype(str).unique())
    selected_leagues = st.multiselect("Leagues / bundles", league_options, default=league_options, key="gk_bundles_filter")
    age_min = float(np.floor(keeper_df["AgeYears"].min()))
    age_max = float(np.ceil(keeper_df["AgeYears"].max()))
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max), step=0.5, key="gk_age_filter")

    gk_filtered = keeper_df.loc[
        keeper_df["BundleLabel"].astype(str).isin(selected_leagues)
        & keeper_df["AgeYears"].between(age_range[0], age_range[1])
    ].copy()

    board_cols = [
        "PlayerName",
        "TeamName",
        "BundleLabel",
        "AgeYears",
        "MinutesPlayed",
        "ScoutFitScore",
        "CompositeRecruitmentScore",
        "DecisionScore",
        "PerformanceReliabilityScore",
        "BallSecurityScore",
        "SecurityRisk_per90",
        "Readiness",
        "RiskBand",
        "FitDrivers",
    ]
    board = gk_filtered[[c for c in board_cols if c in gk_filtered.columns]].rename(
        columns={
            "PlayerName": "Player",
            "TeamName": "Team",
            "BundleLabel": "League",
            "AgeYears": "Age",
            "MinutesPlayed": "Minutes",
            "ScoutFitScore": "Fit",
            "CompositeRecruitmentScore": "Model",
            "DecisionScore": "Decision",
            "PerformanceReliabilityScore": "Reliability",
            "BallSecurityScore": "Security",
            "SecurityRisk_per90": "Risk/90",
            "FitDrivers": "Drivers",
        }
    )
    st.dataframe(
        board.round(2),
        width="stretch",
        hide_index=True,
        column_config={
            "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.1f"),
            "Model": st.column_config.ProgressColumn("Model", min_value=0, max_value=100, format="%.1f"),
            "Decision": st.column_config.ProgressColumn("Decision", min_value=0, max_value=100, format="%.1f"),
            "Reliability": st.column_config.ProgressColumn("Reliability", min_value=0, max_value=100, format="%.1f"),
        },
    )


def render_recruitment_workspace(data: pd.DataFrame) -> None:
    recruitment_df = add_scouting_fields(data, BALANCED_WEIGHTS)
    recruitment_df = recruitment_df.loc[~recruitment_df["PositionGroup"].astype(str).eq("GK")].copy()

    st.markdown("<div class='workspace-label'>Recruitment workspace</div>", unsafe_allow_html=True)
    st.subheader("Recruitment case board")

    if recruitment_df.empty:
        st.info("No outfield recruitment rows were found in the loaded model outputs.")
        return

    value_score = safe_col(recruitment_df, "ValueRecruitmentScore")
    resale_score = safe_col(recruitment_df, "AgeResaleScore")
    style_score = safe_col(recruitment_df, "FCHKStyleScore", safe_col(recruitment_df, "SmartClubScore").median())
    smart_club_score = safe_col(recruitment_df, "SmartClubScore", style_score.median())
    success_score = safe_col(recruitment_df, "SuccessProbability")
    readiness_score = safe_col(recruitment_df, "PerformanceReliabilityScore")
    wage_risk = safe_col(recruitment_df, "WageRisk")
    fee_risk = safe_col(recruitment_df, "FeeRisk")

    recruitment_df["RecruitmentCaseScore"] = (
        value_score * 0.24
        + resale_score * 0.20
        + style_score * 0.18
        + smart_club_score * 0.14
        + success_score * 0.14
        + readiness_score * 0.10
        - wage_risk * 0.04
        - fee_risk * 0.04
    ).clip(0, 100)
    recruitment_df["StyleFit"] = np.where(style_score.ge(70), "Strong", np.where(style_score.ge(55), "Useful", "Question"))
    recruitment_df["ResaleProfile"] = np.where(resale_score.ge(70), "Upside", np.where(resale_score.ge(55), "Neutral", "Limited"))
    recruitment_df["CostRisk"] = np.select(
        [fee_risk.ge(70) | wage_risk.ge(70), fee_risk.ge(50) | wage_risk.ge(50)],
        ["High", "Medium"],
        default="Low",
    )
    recruitment_df["RecruitmentNote"] = (
        "Value "
        + value_score.round(0).astype(int).astype(str)
        + " · resale "
        + resale_score.round(0).astype(int).astype(str)
        + " · style "
        + style_score.round(0).astype(int).astype(str)
        + " · cost risk "
        + recruitment_df["CostRisk"].astype(str)
    )

    recruitment_df = recruitment_df.sort_values(["RecruitmentCaseScore", "ValueRecruitmentScore"], ascending=False)

    metric_cols = st.columns(5)
    with metric_cols[0]:
        metric_card("Recruitment pool", f"{len(recruitment_df):,}", "outfield players")
    with metric_cols[1]:
        metric_card("Median value", f"{value_score.median():.1f}", "transfer value")
    with metric_cols[2]:
        metric_card("Median resale", f"{resale_score.median():.1f}", "age upside")
    with metric_cols[3]:
        metric_card("Strong style fit", f"{recruitment_df['StyleFit'].eq('Strong').sum():,}", "FCHK / smart-club")
    with metric_cols[4]:
        metric_card("Low cost risk", f"{recruitment_df['CostRisk'].eq('Low').sum():,}", "fee + wage")

    st.markdown(
        """
        <div class="note-box">
            This view ranks recruitment cases: value, resale upside, FCHK style fit, smart-club fit, success probability, readiness, and cost risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_cols = st.columns([1, 1, 1])
    with filter_cols[0]:
        role_options = sorted(recruitment_df["PositionGroup"].dropna().astype(str).unique())
        selected_roles = st.multiselect("Roles", role_options, default=role_options, key="recruitment_roles_filter")
    with filter_cols[1]:
        min_resale = st.slider("Minimum resale score", 0, 100, 45, key="recruitment_resale_floor")
    with filter_cols[2]:
        style_options = ["Strong", "Useful", "Question"]
        selected_style_fits = st.multiselect("Style fit", style_options, default=style_options, key="recruitment_style_filter")

    filtered_recruitment = recruitment_df.loc[
        recruitment_df["PositionGroup"].astype(str).isin(selected_roles)
        & recruitment_df["AgeResaleScore"].ge(min_resale)
        & recruitment_df["StyleFit"].isin(selected_style_fits)
    ].copy()

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Best recruitment cases")
        board_cols = [
            "PlayerName",
            "TeamName",
            "PositionGroup",
            "BundleLabel",
            "AgeYears",
            "RecruitmentCaseScore",
            "ValueRecruitmentScore",
            "AgeResaleScore",
            "FCHKStyleScore",
            "SmartClubScore",
            "SuccessProbability",
            "ResaleProfile",
            "StyleFit",
            "CostRisk",
            "PrimaryPlayerStyle",
            "SmartClubTop3",
            "RecruitmentNote",
        ]
        board = filtered_recruitment[[c for c in board_cols if c in filtered_recruitment.columns]].head(100).rename(
            columns={
                "PlayerName": "Player",
                "TeamName": "Team",
                "PositionGroup": "Role",
                "BundleLabel": "League",
                "AgeYears": "Age",
                "RecruitmentCaseScore": "Case",
                "ValueRecruitmentScore": "Value",
                "AgeResaleScore": "Resale",
                "FCHKStyleScore": "FCHK Style",
                "SmartClubScore": "Smart Club",
                "SuccessProbability": "Success",
                "PrimaryPlayerStyle": "Playing Style",
                "SmartClubTop3": "Style Clubs",
                "RecruitmentNote": "Why",
            }
        )
        st.dataframe(
            board.round(2),
            width="stretch",
            hide_index=True,
            column_config={
                "Case": st.column_config.ProgressColumn("Case", min_value=0, max_value=100, format="%.1f"),
                "Value": st.column_config.ProgressColumn("Value", min_value=0, max_value=100, format="%.1f"),
                "Resale": st.column_config.ProgressColumn("Resale", min_value=0, max_value=100, format="%.1f"),
                "FCHK Style": st.column_config.ProgressColumn("FCHK Style", min_value=0, max_value=100, format="%.1f"),
                "Smart Club": st.column_config.ProgressColumn("Smart Club", min_value=0, max_value=100, format="%.1f"),
                "Success": st.column_config.ProgressColumn("Success", min_value=0, max_value=100, format="%.1f"),
            },
        )

    with right:
        st.subheader("Recruitment shape")
        shape = (
            filtered_recruitment.groupby(["PositionGroup", "StyleFit"])
            .agg(Players=("PlayerName", "count"), MedianCase=("RecruitmentCaseScore", "median"), MedianResale=("AgeResaleScore", "median"))
            .round(1)
            .reset_index()
            .sort_values("MedianCase", ascending=False)
        )
        st.dataframe(
            shape,
            width="stretch",
            hide_index=True,
            column_config={
                "MedianCase": st.column_config.ProgressColumn("Case", min_value=0, max_value=100, format="%.1f"),
                "MedianResale": st.column_config.ProgressColumn("Resale", min_value=0, max_value=100, format="%.1f"),
            },
        )

        chart_df = filtered_recruitment[
            ["PlayerName", "TeamName", "PositionGroup", "RecruitmentCaseScore", "AgeResaleScore", "FCHKStyleScore", "CostRisk"]
        ].dropna()
        if not chart_df.empty:
            case_chart = (
                alt.Chart(chart_df.head(300))
                .mark_circle(size=72, opacity=0.78)
                .encode(
                    x=alt.X("AgeResaleScore:Q", title="Resale upside"),
                    y=alt.Y("FCHKStyleScore:Q", title="Playing-style fit"),
                    color=alt.Color("CostRisk:N", title="Cost risk", scale=alt.Scale(domain=["Low", "Medium", "High"], range=["#2a9d8f", "#f4a261", "#e76f51"])),
                    shape=alt.Shape("PositionGroup:N", title="Role"),
                    tooltip=[
                        "PlayerName",
                        "TeamName",
                        "PositionGroup",
                        alt.Tooltip("RecruitmentCaseScore:Q", format=".1f"),
                        alt.Tooltip("AgeResaleScore:Q", format=".1f"),
                        alt.Tooltip("FCHKStyleScore:Q", format=".1f"),
                        "CostRisk",
                    ],
                )
                .properties(height=340)
                .interactive()
            )
            st.altair_chart(case_chart, width="stretch")


def render_team_workspace(data: pd.DataFrame) -> None:
    team_df = add_scouting_fields(data, BALANCED_WEIGHTS)
    czech_df = czech_market(team_df)
    hradec_df = hradec_squad(team_df)
    external_czech = czech_df.loc[~czech_df["TeamName"].isin(hradec_df["TeamName"].unique())].copy()

    st.markdown("<div class='workspace-label'>Team workspace</div>", unsafe_allow_html=True)
    st.subheader("Hradec Kralove focus")

    h_top = hradec_df.sort_values("ScoutFitScore", ascending=False).head(1)
    cz_top = external_czech.sort_values("ScoutFitScore", ascending=False).head(1)
    team_cols = st.columns(5)
    with team_cols[0]:
        metric_card("Czech players", f"{len(czech_df):,}", "Fortuna + Chance Narodni")
    with team_cols[1]:
        metric_card("Czech leagues", f"{czech_df['LeagueLabel'].nunique():,}", "loaded in model")
    with team_cols[2]:
        metric_card("Hradec squad", f"{len(hradec_df):,}", "players in model")
    with team_cols[3]:
        metric_card("Hradec median fit", "n/a" if hradec_df.empty else f"{hradec_df['ScoutFitScore'].median():.1f}", "balanced model")
    with team_cols[4]:
        metric_card("Best Hradec", "n/a" if h_top.empty else str(h_top.iloc[0]["PlayerName"]), "current top signal")

    if hradec_df.empty:
        st.info("No FC Hradec Kralove rows were found in the loaded model outputs.")
        return

    st.markdown(
        """
        <div class="note-box">
            Team view is intentionally compact: first understand Hradec's own squad, then compare it with the Czech market, then use the gaps to decide where to scout.
        </div>
        """,
        unsafe_allow_html=True,
    )

    squad_cols = [
        "PlayerName",
        "PositionGroup",
        "AgeYears",
        "MinutesPlayed",
        "ScoutFitScore",
        "CompositeRecruitmentScore",
        "ValueRecruitmentScore",
        "DecisionScore",
        "Readiness",
        "RiskBand",
        "FitDrivers",
    ]
    squad_view = hradec_df[[c for c in squad_cols if c in hradec_df.columns]].sort_values("ScoutFitScore", ascending=False).rename(
        columns={
            "PlayerName": "Player",
            "PositionGroup": "Role",
            "AgeYears": "Age",
            "MinutesPlayed": "Minutes",
            "ScoutFitScore": "Fit",
            "CompositeRecruitmentScore": "Model",
            "ValueRecruitmentScore": "Value",
            "DecisionScore": "Decision",
            "RiskBand": "Risk",
            "FitDrivers": "Drivers",
        }
    )
    st.subheader("Squad read")
    st.dataframe(
        squad_view.round(2),
        width="stretch",
        hide_index=True,
        column_config={
            "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.1f"),
            "Model": st.column_config.ProgressColumn("Model", min_value=0, max_value=100, format="%.1f"),
            "Value": st.column_config.ProgressColumn("Value", min_value=0, max_value=100, format="%.1f"),
        },
    )

    role_summary = (
        czech_df.groupby("PositionGroup")
        .agg(CzechPlayers=("PlayerName", "count"), CzechMedianFit=("ScoutFitScore", "median"), CzechMedianValue=("ValueRecruitmentScore", "median"))
        .join(
            hradec_df.groupby("PositionGroup")
            .agg(HradecPlayers=("PlayerName", "count"), HradecMedianFit=("ScoutFitScore", "median"), HradecMedianValue=("ValueRecruitmentScore", "median")),
            how="left",
        )
        .fillna({"HradecPlayers": 0})
        .reset_index()
    )
    role_summary["FitGap"] = (role_summary["HradecMedianFit"] - role_summary["CzechMedianFit"]).round(1)
    role_summary["ValueGap"] = (role_summary["HradecMedianValue"] - role_summary["CzechMedianValue"]).round(1)
    role_summary["NeedSignal"] = np.select(
        [
            role_summary["HradecPlayers"].eq(0),
            role_summary["FitGap"].lt(-6),
            role_summary["ValueGap"].lt(-6),
        ],
        ["No depth", "Fit gap", "Value gap"],
        default="Stable",
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Role gaps")
        st.dataframe(
            role_summary.rename(columns={"PositionGroup": "Role"}).round(1),
            width="stretch",
            hide_index=True,
            column_config={
                "CzechMedianFit": st.column_config.ProgressColumn("Czech fit", min_value=0, max_value=100, format="%.1f"),
                "HradecMedianFit": st.column_config.ProgressColumn("Hradec fit", min_value=0, max_value=100, format="%.1f"),
            },
        )
    with right:
        st.subheader("Immediate team signals")
        facts = [
            f"Best internal model signal: {h_top.iloc[0]['PlayerName']} ({h_top.iloc[0]['PositionGroup']}, fit {h_top.iloc[0]['ScoutFitScore']:.1f}).",
            f"Best Czech external signal: {cz_top.iloc[0]['PlayerName']} ({cz_top.iloc[0]['TeamName']}, fit {cz_top.iloc[0]['ScoutFitScore']:.1f})." if not cz_top.empty else "No external Czech target found.",
            f"Hradec median age: {hradec_df['AgeYears'].median():.1f}; Czech market median age: {czech_df['AgeYears'].median():.1f}.",
            f"Hradec U23 players: {int(hradec_df['IsU23Target'].fillna(False).sum())}; Czech U23 market: {int(czech_df['IsU23Target'].fillna(False).sum())}.",
        ]
        st.markdown("<div class='section-card'>" + "".join(f"<div class='metric-caption'>{escape(fact)}</div>" for fact in facts) + "</div>", unsafe_allow_html=True)

    external_cols = [
        "PlayerName",
        "TeamName",
        "PositionGroup",
        "AgeYears",
        "MinutesPlayed",
        "ScoutFitScore",
        "ValueRecruitmentScore",
        "DecisionScore",
        "Readiness",
        "RiskBand",
        "FitDrivers",
    ]
    with st.expander("Czech league watchlist", expanded=True):
        watch = external_czech.sort_values(["ScoutFitScore", "ValueRecruitmentScore"], ascending=False).head(30)
        watch_view = watch[[c for c in external_cols if c in watch.columns]].rename(
            columns={
                "PlayerName": "Player",
                "TeamName": "Team",
                "PositionGroup": "Role",
                "AgeYears": "Age",
                "MinutesPlayed": "Minutes",
                "ScoutFitScore": "Fit",
                "ValueRecruitmentScore": "Value",
                "DecisionScore": "Decision",
                "RiskBand": "Risk",
                "FitDrivers": "Drivers",
            }
        )
        st.dataframe(
            watch_view.round(2),
            width="stretch",
            hide_index=True,
            column_config={"Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.1f")},
        )

    with st.expander("Czech league detail", expanded=False):
        league_summary = (
            czech_df.groupby(["LeagueLabel", "PositionGroup"])
            .agg(
                Players=("PlayerName", "count"),
                MedianFit=("ScoutFitScore", "median"),
                MedianValue=("ValueRecruitmentScore", "median"),
                MedianAge=("AgeYears", "median"),
                U23=("IsU23Target", lambda x: x.fillna(False).astype(bool).sum()),
            )
            .round(1)
            .reset_index()
            .sort_values(["LeagueLabel", "MedianFit"], ascending=[True, False])
        )
        st.dataframe(league_summary, width="stretch", hide_index=True)


st.markdown(
    """
    <style>
    :root {
        --ink: #071118;
        --muted: #657481;
        --panel: #ffffff;
        --line: #c8d5dc;
        --line-strong: #8da0aa;
        --navy: #071118;
        --navy-2: #10202b;
        --teal: #00b894;
        --green: #8bd450;
        --amber: #ffb020;
        --red: #e24a4a;
        --blue: #2e6fba;
        --wash: #edf2f5;
        --pitch: #10271f;
    }

    .stApp {
        background:
            linear-gradient(90deg, rgba(7,17,24,.035) 1px, transparent 1px),
            linear-gradient(0deg, rgba(7,17,24,.035) 1px, transparent 1px),
            var(--wash);
        background-size: 24px 24px;
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.05rem;
        padding-bottom: 1.6rem;
        max-width: 96%;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
        font-weight: 950;
    }

    h2 {
        font-size: .92rem !important;
        text-transform: uppercase;
        letter-spacing: .08em;
        border-bottom: 1px solid var(--line);
        padding-bottom: 6px;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #071118 0%, #10202b 52%, #edf2f5 52%, #edf2f5 100%);
        border-right: 1px solid #071118;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: .8rem;
    }

    .workspace-nav-spacer {
        height: 24px;
    }

    .hero {
        max-width: 920px;
        margin: 12vh auto 28px auto;
        color: white;
        text-align: center;
        border: 1px solid #071118;
        background:
            linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,.055) 1px, transparent 1px),
            linear-gradient(135deg, rgba(0,184,148,.24), rgba(139,212,80,.08) 44%, rgba(7,17,24,0) 44%),
            #071118;
        background-size: 36px 36px, 36px 36px, auto, auto;
        padding: 42px 34px;
        box-shadow: 0 18px 42px rgba(7, 17, 24, .16);
    }

    .hero-content,
    .hero-panel {
        position: relative;
        z-index: 1;
    }

    .hero-kicker {
        color: var(--green);
        font-size: .62rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .16em;
    }

    .hero h1 {
        color: white;
        margin: 12px auto 10px auto;
        max-width: 760px;
        font-size: clamp(2.4rem, 5vw, 4.8rem);
        line-height: .96;
        letter-spacing: 0;
    }

    .hero p {
        margin: 0;
        max-width: 560px;
        margin: 0 auto;
        color: #c8d5dc;
        font-size: .95rem;
        line-height: 1.5;
    }

    .home-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 12px;
    }

    .home-nav span {
        border: 1px solid rgba(124, 234, 213, .42);
        background: rgba(255, 255, 255, .055);
        color: #effffc;
        border-radius: 2px;
        padding: 5px 8px;
        font-size: .62rem;
        font-weight: 950;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .hero-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 20px;
    }

    .hero-action-primary,
    .hero-action-secondary {
        border-radius: 2px;
        padding: 9px 12px;
        font-size: .68rem;
        font-weight: 950;
        letter-spacing: .11em;
        text-transform: uppercase;
    }

    .hero-action-primary {
        background: #7cead5;
        border: 1px solid #7cead5;
        color: #07111a;
    }

    .hero-action-secondary {
        border: 1px solid rgba(255, 255, 255, .24);
        color: #effffc;
    }

    .hero-panel {
        border: 1px solid rgba(255, 255, 255, .16);
        background: rgba(255, 255, 255, .06);
        padding: 14px;
    }

    .hero-panel-title {
        color: white;
        font-size: 1.18rem;
        font-weight: 950;
        letter-spacing: 0;
        line-height: 1.05;
    }

    .hero-panel-copy {
        color: #b7c5cd;
        font-size: .76rem;
        line-height: 1.38;
        margin-top: 7px;
    }

    .landing-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 8px;
        max-width: 920px;
        margin: 0 auto;
    }

    .landing-card {
        border: 1px solid rgba(255, 255, 255, .14);
        background: rgba(7, 17, 26, .72);
        padding: 11px;
        min-height: 118px;
    }

    .landing-card-label {
        color: #7cead5;
        font-size: .56rem;
        font-weight: 950;
        letter-spacing: .13em;
        text-transform: uppercase;
    }

    .landing-card-title {
        color: white;
        font-size: 1rem;
        font-weight: 950;
        letter-spacing: 0;
        margin-top: 4px;
    }

    .landing-card-copy {
        color: #aebfc8;
        font-size: .68rem;
        line-height: 1.32;
        margin-top: 5px;
    }

    .workspace-label {
        border-left: 4px solid var(--teal);
        color: var(--navy);
        font-size: .62rem;
        font-weight: 950;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin: 8px 0 10px 0;
        padding: 4px 0 4px 8px;
        background: rgba(255, 255, 255, .62);
    }

    .intel-strip {
        border: 1px solid #071118;
        background:
            linear-gradient(90deg, rgba(0,184,148,.20), transparent 36%),
            #071118;
        color: #dfe8ed;
        padding: 9px 11px;
        margin: 8px 0 12px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }

    .intel-strip-title {
        color: white;
        font-size: .84rem;
        font-weight: 950;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .intel-strip-meta {
        color: #9fb0ba;
        font-size: .62rem;
        font-weight: 850;
        letter-spacing: .09em;
        text-transform: uppercase;
    }

    .analysis-panel {
        border: 1px solid #071118;
        border-radius: 0;
        background:
            linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,.05) 1px, transparent 1px),
            linear-gradient(180deg, #071118 0%, #10202b 100%);
        background-size: 28px 28px;
        color: #dce5ea;
        padding: 12px;
    }

    .analysis-panel .panel-kicker {
        color: var(--green);
        font-size: .58rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .15em;
        margin-bottom: 4px;
    }

    .analysis-panel .panel-title {
        color: white;
        font-size: 1rem;
        font-weight: 950;
        line-height: 1;
        letter-spacing: 0;
    }

    .analysis-panel .panel-copy {
        color: #b7c5cd;
        font-size: .72rem;
        line-height: 1.35;
        margin-top: 5px;
    }

    .metric-card {
        border: 1px solid var(--line);
        border-top: 3px solid var(--teal);
        border-radius: 0;
        padding: 11px 12px;
        background: rgba(255, 255, 255, .96);
        min-height: 84px;
        box-shadow: 0 8px 18px rgba(7, 17, 24, .055);
    }

    .metric-label {
        color: var(--muted);
        font-size: .58rem;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-weight: 950;
    }

    .metric-value {
        margin-top: 2px;
        color: var(--navy);
        font-size: 1.34rem;
        font-weight: 950;
        line-height: 1;
        letter-spacing: 0;
    }

    .metric-caption {
        color: var(--muted);
        font-size: .66rem;
        margin-top: 4px;
        line-height: 1.25;
    }

    .profile-card {
        border: 1px solid var(--line);
        border-left: 4px solid var(--blue);
        border-radius: 0;
        padding: 11px;
        background: rgba(255, 255, 255, .98);
        min-height: 104px;
    }

    .profile-name {
        color: var(--navy);
        font-weight: 950;
        font-size: .95rem;
        line-height: 1;
        letter-spacing: 0;
    }

    .profile-meta {
        color: var(--muted);
        font-size: .68rem;
        margin-top: 2px;
        line-height: 1.3;
    }

    .pill-row {
        margin-top: 7px;
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }

    .pill {
        border: 1px solid #aab9c2;
        border-radius: 0;
        color: var(--navy);
        background: #eef4f7;
        padding: 3px 6px;
        font-size: .58rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .note-box {
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        background: #ffffff;
        padding: 9px 10px;
        border-radius: 0;
        color: #10212b;
        font-size: .74rem;
        line-height: 1.35;
    }

    .scout-label {
        display: inline-flex;
        align-items: center;
        border: 1px solid #ccd8df;
        border-radius: 0;
        background: #fff;
        padding: 3px 6px;
        margin: 1px 3px 1px 0;
        font-size: .56rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .1em;
        color: #09131c;
    }

    .section-card {
        border: 1px solid var(--line);
        border-radius: 0;
        background: rgba(255, 255, 255, .97);
        padding: 10px;
    }

    .homepage {
        display: grid;
        grid-template-columns: 1.35fr .95fr;
        gap: 12px;
        align-items: stretch;
        margin-bottom: 12px;
    }

    .home-feature {
        border: 1px solid #071118;
        border-radius: 0;
        background:
            linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,.055) 1px, transparent 1px),
            linear-gradient(135deg, rgba(0, 184, 148, .24) 0%, rgba(0, 184, 148, 0) 40%),
            linear-gradient(180deg, #071118 0%, #10202b 100%);
        background-size: 34px 34px, 34px 34px, auto, auto;
        color: white;
        min-height: 356px;
        padding: 18px;
        position: relative;
        overflow: hidden;
    }

    .home-feature:after {
        content: "";
        position: absolute;
        inset: auto -10% -18% 38%;
        height: 220px;
        border: 1px solid rgba(139, 212, 80, .22);
        background:
            linear-gradient(90deg, transparent 0 18%, rgba(255,255,255,.06) 18% 19%, transparent 19% 39%, rgba(255,255,255,.06) 39% 40%, transparent 40% 60%, rgba(255,255,255,.06) 60% 61%, transparent 61% 82%, rgba(255,255,255,.06) 82% 83%, transparent 83%),
            linear-gradient(0deg, transparent 0 22%, rgba(255,255,255,.05) 22% 23%, transparent 23% 48%, rgba(255,255,255,.05) 48% 49%, transparent 49% 74%, rgba(255,255,255,.05) 74% 75%, transparent 75%);
        transform: rotate(-8deg);
        opacity: .9;
    }

    .home-kicker {
        color: var(--green);
        font-size: .62rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .16em;
        position: relative;
        z-index: 1;
    }

    .home-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 950;
        letter-spacing: 0;
        line-height: .95;
        max-width: 760px;
        margin-top: 8px;
        position: relative;
        z-index: 1;
    }

    .home-copy {
        color: #c4d0d6;
        font-size: .88rem;
        line-height: 1.45;
        max-width: 680px;
        margin-top: 12px;
        position: relative;
        z-index: 1;
    }

    .home-stat-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        margin-top: 22px;
        max-width: 640px;
        position: relative;
        z-index: 1;
    }

    .home-stat {
        border: 1px solid rgba(255, 255, 255, .16);
        background: rgba(255, 255, 255, .075);
        border-radius: 0;
        padding: 9px;
    }

    .home-stat-value {
        font-size: 1.32rem;
        font-weight: 950;
        letter-spacing: 0;
        line-height: 1;
    }

    .home-stat-label {
        color: #aebfc8;
        font-size: .58rem;
        font-weight: 950;
        letter-spacing: .11em;
        text-transform: uppercase;
        margin-top: 5px;
    }

    .home-pillar-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }

    .home-pillar {
        border: 1px solid var(--line);
        border-radius: 0;
        background: rgba(255, 255, 255, .98);
        padding: 12px;
        min-height: 172px;
        box-shadow: 0 8px 18px rgba(7, 17, 24, .045);
    }

    .home-pillar-label {
        color: var(--teal);
        font-size: .58rem;
        font-weight: 950;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    .home-pillar-title {
        color: var(--navy);
        font-size: 1.18rem;
        font-weight: 950;
        letter-spacing: 0;
        line-height: 1.05;
        margin-top: 5px;
    }

    .home-pillar-copy {
        color: #5f6b76;
        font-size: .72rem;
        line-height: 1.35;
        margin-top: 8px;
    }

    .home-focus-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin-top: 12px;
    }

    .filter-summary {
        border: 1px solid var(--line);
        border-radius: 0;
        background: rgba(255, 255, 255, .96);
        padding: 10px 12px;
        margin: 10px 0 12px 0;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        align-items: center;
    }

    .filter-summary-label {
        color: #66737d;
        font-size: .58rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .12em;
        margin-right: 4px;
    }

    .filter-token {
        border: 1px solid #aab9c2;
        border-radius: 0;
        background: #eef4f7;
        color: var(--navy);
        padding: 3px 8px;
        font-size: .68rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .scouting-command {
        border: 1px solid #071118;
        background:
            linear-gradient(90deg, rgba(255,255,255,.07) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,.06) 1px, transparent 1px),
            radial-gradient(circle at 86% 18%, rgba(139, 212, 80, .18), transparent 18%),
            linear-gradient(135deg, rgba(0, 184, 148, .28), transparent 42%),
            #071118;
        background-size: 30px 30px, 30px 30px, auto, auto, auto;
        color: white;
        padding: 18px;
        margin: 6px 0 10px 0;
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(260px, .8fr);
        gap: 16px;
        align-items: end;
        box-shadow: 0 18px 38px rgba(7, 17, 24, .14);
    }

    .scouting-kicker {
        color: var(--green);
        font-size: .62rem;
        font-weight: 950;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .scouting-title {
        color: white;
        font-size: clamp(2rem, 4vw, 4rem);
        font-weight: 950;
        line-height: .96;
        margin-top: 7px;
    }

    .scouting-copy {
        color: #c8d5dc;
        font-size: .84rem;
        line-height: 1.42;
        max-width: 760px;
        margin-top: 10px;
    }

    .scouting-mode-panel {
        border-left: 1px solid rgba(255,255,255,.18);
        padding-left: 14px;
    }

    .scouting-mode-label {
        color: #9fb0ba;
        font-size: .56rem;
        font-weight: 950;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    .scouting-mode-value {
        color: white;
        font-size: 1.36rem;
        font-weight: 950;
        line-height: 1;
        margin-top: 5px;
    }

    .scouting-cockpit {
        border: 1px solid #071118;
        background:
            linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,.05) 1px, transparent 1px),
            #10202b;
        background-size: 26px 26px;
        padding: 10px;
        margin: 10px 0 12px 0;
    }

    .cockpit-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 8px;
    }

    .cockpit-tile {
        border: 1px solid rgba(255,255,255,.14);
        background: rgba(255,255,255,.06);
        padding: 10px;
        min-height: 82px;
    }

    .cockpit-label {
        color: #9fb0ba;
        font-size: .55rem;
        font-weight: 950;
        letter-spacing: .13em;
        text-transform: uppercase;
    }

    .cockpit-value {
        color: white;
        font-size: 1.42rem;
        font-weight: 950;
        line-height: 1;
        margin-top: 4px;
    }

    .cockpit-note {
        color: #c8d5dc;
        font-size: .64rem;
        line-height: 1.25;
        margin-top: 5px;
    }

    .role-rail {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 6px;
        margin: 8px 0 12px 0;
    }

    .role-cell {
        border: 1px solid var(--line);
        border-top: 3px solid var(--teal);
        background: rgba(255,255,255,.96);
        padding: 8px;
        min-height: 62px;
    }

    .role-cell-role {
        color: var(--muted);
        font-size: .55rem;
        font-weight: 950;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .role-cell-score {
        color: var(--navy);
        font-size: 1.08rem;
        font-weight: 950;
        line-height: 1;
        margin-top: 4px;
    }

    .role-cell-count {
        color: var(--muted);
        font-size: .62rem;
        margin-top: 3px;
    }

    .quality-hero {
        border: 1px solid #071118;
        background:
            linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,.05) 1px, transparent 1px),
            linear-gradient(135deg, rgba(0,184,148,.22), transparent 44%),
            #071118;
        background-size: 28px 28px, 28px 28px, auto, auto;
        color: white;
        padding: 14px;
        margin-bottom: 10px;
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(240px, .75fr);
        gap: 12px;
        align-items: end;
    }

    .quality-title {
        color: white;
        font-size: 1.6rem;
        font-weight: 950;
        line-height: 1;
    }

    .quality-copy {
        color: #c8d5dc;
        font-size: .74rem;
        line-height: 1.35;
        margin-top: 6px;
    }

    .quality-signal {
        border-left: 1px solid rgba(255,255,255,.18);
        padding-left: 12px;
    }

    .quality-signal-label {
        color: #9fb0ba;
        font-size: .55rem;
        font-weight: 950;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    .quality-signal-value {
        color: var(--green);
        font-size: 1.3rem;
        font-weight: 950;
        margin-top: 4px;
    }

    .quality-card-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        margin: 10px 0;
    }

    .quality-player-card {
        border: 1px solid var(--line);
        border-top: 3px solid var(--teal);
        background: rgba(255,255,255,.98);
        padding: 10px;
        min-height: 96px;
    }

    .quality-rank {
        color: var(--teal);
        font-size: .55rem;
        font-weight: 950;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    .quality-player-name {
        color: var(--navy);
        font-size: 1rem;
        font-weight: 950;
        line-height: 1.05;
        margin-top: 4px;
    }

    .quality-player-meta {
        color: var(--muted);
        font-size: .66rem;
        line-height: 1.25;
        margin-top: 4px;
    }

    @media (max-width: 900px) {
        .hero,
        .homepage,
        .home-pillar-grid,
        .home-focus-strip,
        .home-stat-row,
        .landing-grid,
        .scouting-command,
        .cockpit-grid,
        .role-rail,
        .quality-hero,
        .quality-card-grid {
            grid-template-columns: 1fr;
        }

        .hero {
            min-height: auto;
            padding: 22px;
        }

        .workspace-nav-spacer {
            height: 18px;
        }

        .home-title {
            font-size: 2.05rem;
        }
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 0;
        border: 1px solid #9eb0bb;
        background: #ffffff;
        color: var(--ink);
        font-weight: 900;
        min-height: 36px;
        font-size: .72rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        box-shadow: none;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background:
            linear-gradient(90deg, rgba(0, 184, 148, .22), transparent 46%),
            var(--navy);
        border-color: var(--navy);
        color: white;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--teal);
        color: var(--navy);
    }

    div[data-testid="stTabs"] button {
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-size: .64rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #9eb0bb;
        border-radius: 0;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(7, 17, 24, .055);
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 0;
        background: rgba(255, 255, 255, .93);
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-top: 3px solid var(--green);
        padding: 8px 10px;
        background: #fff;
    }

    input,
    textarea,
    div[data-baseweb="select"] > div,
    div[data-baseweb="tag"] {
        border-radius: 0 !important;
    }
</style>
    """,
    unsafe_allow_html=True,
)

if "show_scouting_workspace" not in st.session_state:
    st.session_state["show_scouting_workspace"] = False
if "active_workspace" not in st.session_state:
    st.session_state["active_workspace"] = "Scouting"

if not st.session_state["show_scouting_workspace"]:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="hero-kicker">Hradeck Intelligence</div>
                <h1>Player quality command room</h1>
                <p>Scouting, recruitment, goalkeeper, team, and model views built for fast football decisions.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='landing-grid'>", unsafe_allow_html=True)
    landing_cols = st.columns(len(WORKSPACES))
    for idx, section in enumerate(WORKSPACES):
        with landing_cols[idx]:
            st.button(
                section,
                type="primary" if section == "Scouting" else "secondary",
                width="stretch",
                on_click=set_workspace,
                args=(section,),
            )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("landing_notice"):
        st.info(st.session_state["landing_notice"])
    st.stop()

render_workspace_nav("main")

active_workspace = st.session_state.get("active_workspace", "Scouting")
data = load_default_data()
model_metadata = load_model_metadata()
st.markdown(
    f"""
    <div class="intel-strip">
        <div class="intel-strip-title">{escape(active_workspace)} intelligence</div>
        <div class="intel-strip-meta">{len(data):,} players · {data['BundleLabel'].nunique()} leagues · model v3</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if active_workspace != "Scouting":
    if active_workspace == "Model":
        render_model_workspace(data, model_metadata)
    elif active_workspace == "Recruitment":
        render_recruitment_workspace(data)
    elif active_workspace == "Goalkeepers":
        render_goalkeepers_workspace(data)
    elif active_workspace == "Team":
        render_team_workspace(data)
    else:
        st.markdown(f"<div class='workspace-label'>{active_workspace} workspace</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="section-card">
                <div class="metric-label">Coming next</div>
                <div class="metric-value" style="font-size:1.45rem;">{active_workspace}</div>
                <div class="metric-caption">This workspace is ready in the navigation, but the detailed tools are still being shaped. Use the buttons above to switch back to Scouting or move between workspaces.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

if "quick_mode" not in st.session_state:
    st.session_state["quick_mode"] = "Full board"
if "shortlist_players" not in st.session_state:
    st.session_state["shortlist_players"] = []

outfield_data = data.loc[~data["PositionGroup"].astype(str).eq("GK")].copy()
quick_modes = ["Full board", "U23 quality", "Elite quality", "Reliable quality"]
if st.session_state["quick_mode"] not in quick_modes:
    st.session_state["quick_mode"] = "Full board"

preset_weights = {
    "Balanced quality": {"Composite": 3, "Decision": 2, "Value": 0, "Success": 1, "Reliability": 2, "Risk penalty": 1},
    "Technique": {"Composite": 2, "Decision": 3, "Value": 0, "Success": 1, "Reliability": 1, "Risk penalty": 1},
    "Ready quality": {"Composite": 3, "Decision": 3, "Value": 0, "Success": 1, "Reliability": 3, "Risk penalty": 2},
    "Upside quality": {"Composite": 2, "Decision": 2, "Value": 0, "Success": 2, "Reliability": 1, "Risk penalty": 0},
}

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Scouting IQ</div>
            <div class="sidebar-brand-meta">{len(outfield_data):,} outfield players · GK excluded</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='menu-caption'>Pure player quality. Recruitment value, resale, wage, and fee risk are intentionally outside this room.</div>", unsafe_allow_html=True)
    model_preset = st.segmented_control("Quality lens", list(preset_weights), default="Balanced quality", width="stretch")
    defaults = preset_weights[model_preset]
    with st.expander("Lens weights", expanded=False):
        weights = {
            "Composite": st.slider("Model quality", 0, 5, defaults["Composite"]),
            "Decision": st.slider("Decision quality", 0, 5, defaults["Decision"]),
            "Value": 0,
            "Success": st.slider("Success probability", 0, 5, defaults["Success"]),
            "Reliability": st.slider("Reliability", 0, 5, defaults["Reliability"]),
            "Risk penalty": st.slider("Risk penalty", 0, 5, defaults["Risk penalty"]),
        }

df = add_scouting_fields(outfield_data, weights)
position_groups = sorted(df["PositionGroup"].dropna().astype(str).unique())
bundle_groups = sorted(df["BundleLabel"].dropna().astype(str).unique())
archetype_groups = sorted(df["Archetype"].dropna().astype(str).unique())

with st.sidebar:
    st.subheader("Quality Controls")
    search = st.text_input("Search", key="search_filter", placeholder="Player or club")
    saved_positions = st.session_state.get("positions_filter")
    if saved_positions and any(position not in position_groups for position in saved_positions):
        st.session_state["positions_filter"] = position_groups
    positions = st.multiselect("Roles", position_groups, default=position_groups, key="positions_filter")
    bundles = st.multiselect("Leagues", bundle_groups, default=bundle_groups, key="bundles_filter")
    archetypes = st.multiselect("Archetypes", archetype_groups, default=archetype_groups, key="archetypes_filter")
    u23_only = st.toggle("U23 only", value=False, key="u23_filter")
    age_range = st.slider(
        "Age",
        float(np.floor(df["AgeYears"].min())),
        float(np.ceil(df["AgeYears"].max())),
        (float(np.floor(df["AgeYears"].min())), float(np.ceil(df["AgeYears"].max()))),
        step=0.5,
        key="age_filter",
    )
    minutes_range = st.slider(
        "Minutes",
        int(df["MinutesPlayed"].min()),
        int(df["MinutesPlayed"].max()),
        (900, int(df["MinutesPlayed"].max())),
        step=100,
        key="minutes_filter",
    )
    quality_floor = st.slider("Quality floor", 0, 100, 35, key="quality_floor")
    role_floor = st.slider("Role-fit floor", 0, 100, 35, key="fit_floor")
    reliability_floor = st.slider("Reliability floor", 0, 100, 45, key="reliability_floor")
    max_risk = st.slider("Max security risk/90", 0.0, 25.0, 18.0, step=0.5, key="max_risk")

mask = (
    df["PositionGroup"].astype(str).isin(positions)
    & df["BundleLabel"].astype(str).isin(bundles)
    & df["Archetype"].astype(str).isin(archetypes)
    & df["AgeYears"].between(age_range[0], age_range[1])
    & df["MinutesPlayed"].between(minutes_range[0], minutes_range[1])
    & df["QualityScore"].ge(quality_floor)
    & df["RoleFitScore"].ge(role_floor)
    & df["PerformanceReliabilityScore"].ge(reliability_floor)
    & df["SecurityRisk_per90"].le(max_risk)
)
if u23_only and "IsU23Target" in df:
    mask &= df["IsU23Target"].fillna(False).astype(bool)
if search:
    haystack = (df["PlayerName"].fillna("").astype(str) + " " + df["TeamName"].fillna("").astype(str)).str.lower()
    mask &= haystack.str.contains(search.lower(), regex=False)

filtered = df.loc[mask].sort_values(["QualityScore", "RoleFitScore", "ProfileScore"], ascending=False)
leader = filtered.head(1)
leader_name = "No player" if leader.empty else str(leader.iloc[0]["PlayerName"])
leader_score = "n/a" if leader.empty else f"{leader.iloc[0]['QualityScore']:.1f}"
best_role = "n/a" if filtered.empty else filtered.groupby("PositionGroup")["QualityScore"].median().sort_values(ascending=False).index[0]
high_quality_count = 0 if filtered.empty else int(filtered["QualityTier"].isin(["High quality", "Elite"]).sum())
median_quality = "n/a" if filtered.empty else f"{filtered['QualityScore'].median():.1f}"
median_impact = "n/a" if filtered.empty else f"{filtered['ProfileScore'].median():.1f}"

if filtered.empty:
    st.info("No players match the current quality controls. Loosen the filters in the sidebar to open Player Lab.")
    st.stop()

quality_tab, player_tab, compare_tab, export_tab = st.tabs(["Quality", "Player Lab", "Compare", "Export"])

with quality_tab:
    top_quality_player = filtered.iloc[0]
    st.markdown(
        f"""
        <div class="quality-hero">
            <div>
                <div class="scouting-kicker">Quality command</div>
                <div class="quality-title">Rank the football, not the deal.</div>
                <div class="quality-copy">Quality combines role fit, impact, decision-making, reliability, and repeatable strengths. Use this tab as the first pass before opening Player Lab.</div>
            </div>
            <div class="quality-signal">
                <div class="quality-signal-label">Board leader</div>
                <div class="quality-signal-value">{escape(str(top_quality_player['PlayerName']))}</div>
                <div class="cockpit-note">Quality {top_quality_player['QualityScore']:.1f} · {escape(str(top_quality_player['PositionGroup']))} · {escape(str(top_quality_player['TeamName']))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_cols = st.columns(4)
    with summary_cols[0]:
        metric_card("Players", f"{len(filtered):,}", "in current quality scope")
    with summary_cols[1]:
        metric_card("Median quality", median_quality, "current board")
    with summary_cols[2]:
        metric_card("High quality", f"{high_quality_count:,}", "elite + high quality")
    with summary_cols[3]:
        metric_card("Best role", str(best_role), "by median quality")

    top_cards = "".join(
        f"""
        <div class="quality-player-card">
            <div class="quality-rank">Rank {idx}</div>
            <div class="quality-player-name">{escape(str(row.PlayerName))}</div>
            <div class="quality-player-meta">{escape(str(row.TeamName))} · {escape(str(row.PositionGroup))} · quality {row.QualityScore:.1f}</div>
            <div class="quality-player-meta">{escape(str(row.QualityDrivers))}</div>
        </div>
        """
        for idx, row in enumerate(filtered.head(3).itertuples(index=False), start=1)
    )
    st.markdown(f"<div class='quality-card-grid'>{top_cards}</div>", unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        quality_cols = [
            "PlayerName",
            "TeamName",
            "PositionGroup",
            "BundleLabel",
            "AgeYears",
            "MinutesPlayed",
            "QualityScore",
            "QualityTier",
            "RoleFitScore",
            "ProfileScore",
            "DecisionScore",
            "QualityDrivers",
        ]
        quality_board = filtered[[c for c in quality_cols if c in filtered.columns]].rename(
            columns={
                "PlayerName": "Player",
                "TeamName": "Team",
                "PositionGroup": "Role",
                "BundleLabel": "League",
                "AgeYears": "Age",
                "MinutesPlayed": "Minutes",
                "QualityScore": "Quality",
                "QualityTier": "Tier",
                "RoleFitScore": "Role Fit",
                "ProfileScore": "Impact",
                "DecisionScore": "Decision",
                "QualityDrivers": "Drivers",
            }
        )
        st.dataframe(
            quality_board.round(2),
            width="stretch",
            hide_index=True,
            column_config={
                "Quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.1f"),
                "Role Fit": st.column_config.ProgressColumn("Role Fit", min_value=0, max_value=100, format="%.1f"),
                "Impact": st.column_config.ProgressColumn("Impact", min_value=0, max_value=100, format="%.1f"),
                "Decision": st.column_config.ProgressColumn("Decision", min_value=0, max_value=100, format="%.1f"),
            },
        )
    with right:
        st.subheader("Role Quality")
        role_quality = (
            filtered.groupby("PositionGroup")
            .agg(
                Players=("PlayerName", "count"),
                MedianQuality=("QualityScore", "median"),
                MedianImpact=("ProfileScore", "median"),
                TopQuality=("QualityScore", "max"),
            )
            .round(1)
            .reset_index()
            .sort_values("MedianQuality", ascending=False)
        )
        st.dataframe(
            role_quality,
            width="stretch",
            hide_index=True,
            column_config={
                "MedianQuality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.1f"),
                "MedianImpact": st.column_config.ProgressColumn("Impact", min_value=0, max_value=100, format="%.1f"),
                "TopQuality": st.column_config.ProgressColumn("Top", min_value=0, max_value=100, format="%.1f"),
            },
        )
        chart = (
            alt.Chart(role_quality)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("PositionGroup:N", title="Role", sort="-y"),
                y=alt.Y("MedianQuality:Q", title="Median quality", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("MedianQuality:Q", legend=None, scale=alt.Scale(domain=[35, 75], range=["#2e6fba", "#00b894", "#8bd450"])),
                tooltip=["PositionGroup", "Players", alt.Tooltip("MedianQuality:Q", format=".1f"), alt.Tooltip("TopQuality:Q", format=".1f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")

        st.subheader("Quality Tier Mix")
        tier_order = ["Elite", "High quality", "Good", "Useful", "Monitor"]
        tier_mix = filtered["QualityTier"].value_counts().reindex(tier_order).fillna(0).astype(int).reset_index()
        tier_mix.columns = ["Tier", "Players"]
        tier_chart = (
            alt.Chart(tier_mix)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                y=alt.Y("Tier:N", title=None, sort=tier_order),
                x=alt.X("Players:Q", title="Players"),
                color=alt.Color("Tier:N", legend=None, scale=alt.Scale(domain=tier_order, range=["#8bd450", "#00b894", "#2e6fba", "#ffb020", "#657481"])),
                tooltip=["Tier", "Players"],
            )
            .properties(height=210)
        )
        st.altair_chart(tier_chart, width="stretch")

with player_tab:
    st.subheader("Player Lab")
    player_options = filtered.assign(_label=filtered["PlayerName"] + " | " + filtered["TeamName"] + " | " + filtered["PositionGroup"]).sort_values("QualityScore", ascending=False)
    selected_label = st.selectbox("Player", player_options["_label"].tolist())
    player = player_options.loc[player_options["_label"].eq(selected_label)].iloc[0]
    st.markdown(f"""<div class="profile-card"><div class="profile-name">{escape(str(player['PlayerName']))}</div><div class="profile-meta">{escape(str(player['TeamName']))} · {escape(str(player['BundleLabel']))} · {escape(str(player['PositionGroup']))} · {player['AgeYears']:.1f} years · {int(player['MinutesPlayed']):,} minutes</div><div class="pill-row"><span class="pill">Quality {player['QualityScore']:.1f}</span><span class="pill">{escape(str(player['QualityTier']))}</span><span class="pill">{escape(str(player['Archetype']))}</span><span class="pill">{escape(str(player['RiskBand']))} risk</span></div></div>""", unsafe_allow_html=True)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Quality", f"{player['QualityScore']:.1f}")
    metric_cols[1].metric("Role Fit", f"{player['RoleFitScore']:.1f}")
    metric_cols[2].metric("Impact", f"{player['ProfileScore']:.1f}")
    metric_cols[3].metric("Decision", f"{player['DecisionScore']:.1f}")
    metric_cols[4].metric("Position Pctl", f"{percentile_rank(df.loc[df['PositionGroup'].eq(player['PositionGroup']), 'QualityScore'], player['QualityScore']):.0f}")
    st.markdown(f"<div class='note-box'>Quality drivers: {escape(str(player['QualityDrivers']))}. Reliability: {escape(str(player['Readiness']))}; risk: {escape(str(player['RiskBand']))}.</div>", unsafe_allow_html=True)
    if st.button("Add player to shortlist", type="primary"):
        add_to_shortlist(player["PlayerName"])
    lab_left, lab_right = st.columns([1, 1])
    with lab_left:
        st.pyplot(render_player_pizza(df.loc[df["PositionGroup"].eq(player["PositionGroup"])], player), clear_figure=True)
    with lab_right:
        per90_cols = [c for c in filtered.columns if c.endswith("_per90") or c.startswith("Imp_")]
        per90 = pd.DataFrame({"Metric": per90_cols, "Value": [float(player[c]) for c in per90_cols]}).sort_values("Value", ascending=False).head(30)
        st.dataframe(per90.round(3), width="stretch", hide_index=True)
    comparable = similar_players(df, player, same_position=True, n=12)
    st.subheader("Closest Quality Profiles")
    similar_cols = ["PlayerName", "TeamName", "PositionGroup", "AgeYears", "MinutesPlayed", "SimilarityScore", "QualityScore", "RoleFitScore", "ProfileScore", "Archetype"]
    st.dataframe(comparable[[c for c in similar_cols if c in comparable.columns]].round(2), width="stretch", hide_index=True)

with compare_tab:
    st.subheader("Compare Players")
    compare_options = filtered.assign(_label=filtered["PlayerName"] + " | " + filtered["TeamName"] + " | " + filtered["PositionGroup"]).sort_values("QualityScore", ascending=False)["_label"].tolist()
    selected_compare = st.multiselect("Select 2-4 players", compare_options, default=compare_options[: min(3, len(compare_options))], max_selections=4)
    compare_names = [label.split(" | ")[0] for label in selected_compare]
    compare_df = filtered.loc[filtered["PlayerName"].isin(compare_names)].sort_values("QualityScore", ascending=False)
    if len(compare_df) < 2:
        st.info("Pick at least two players to compare.")
    else:
        st.dataframe(compare_df[["PlayerName", "TeamName", "PositionGroup", "QualityScore", "RoleFitScore", "ProfileScore", "DecisionScore", "QualityDrivers"]].round(2), width="stretch", hide_index=True)
        compare_scores = compare_df[["PlayerName", "QualityScore", "RoleFitScore", "ProfileScore", "DecisionScore", "PerformanceReliabilityScore"]].melt(id_vars="PlayerName", var_name="Metric", value_name="Score")
        compare_chart = alt.Chart(compare_scores).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=-30)), y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])), color=alt.Color("PlayerName:N", title="Player"), xOffset="PlayerName:N", tooltip=["PlayerName", "Metric", alt.Tooltip("Score:Q", format=".1f")]).properties(height=360)
        st.altair_chart(compare_chart, width="stretch")

with export_tab:
    st.subheader("Export Quality Work")
    export_cols = [c for c in ["PlayerName", "TeamName", "PositionGroup", "BundleLabel", "AgeYears", "MinutesPlayed", "QualityScore", "QualityTier", "RoleFitScore", "ProfileScore", "DecisionScore", "PerformanceReliabilityScore", "Readiness", "RiskBand", "Archetype", "QualityDrivers", "RiskFlags"] if c in filtered.columns]
    export_df = filtered[export_cols].round(3)
    shortlist_df = df.loc[df["PlayerName"].isin(st.session_state.get("shortlist_players", []))].sort_values("QualityScore", ascending=False)
    export_left, export_right = st.columns([1, 1])
    with export_left:
        st.download_button("Download board CSV", data=export_df.to_csv(index=False).encode("utf-8"), file_name="fchk_quality_board.csv", mime="text/csv", width="stretch")
    with export_right:
        st.download_button("Download board PDF", data=build_pdf(filtered, "FCHK Quality Scouting Report", scope_note=f"{len(filtered):,} outfield players · {model_preset} lens", top_n=75), file_name="fchk_quality_scouting_report.pdf", mime="application/pdf", type="primary", width="stretch")
    if not shortlist_df.empty:
        st.subheader("Current Shortlist")
        st.dataframe(shortlist_df[[c for c in export_cols if c in shortlist_df.columns]].round(2), width="stretch", hide_index=True)
