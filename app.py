from __future__ import annotations

import os
from html import escape
from io import BytesIO
from pathlib import Path
from textwrap import shorten

import json
from datetime import date as _date

APP_DIR = Path(__file__).parent
SHORTLIST_FILE = APP_DIR / "data" / "shortlist.json"
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
    value_box_colors = ["#0f1623"] * len(params)
    value_text_colors = ["#e8edf3"] * len(params)
    baker = PyPizza(
        params=params,
        background_color="#080c14",
        straight_line_color="#1e2d3d",
        straight_line_lw=1,
        last_circle_color="#00d4a8",
        last_circle_lw=1.4,
        other_circle_color="#1e2d3d",
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
        blank_alpha=0.18,
        kwargs_slices={"edgecolor": "#080c14", "linewidth": 1.1},
        kwargs_params={"color": "#8fa3b1", "fontsize": 9, "fontweight": "bold"},
        kwargs_values={
            "color": "#e8edf3",
            "fontsize": 9,
            "fontweight": "bold",
            "bbox": {
                "boxstyle": "round,pad=0.28",
                "facecolor": "#0f1623",
                "edgecolor": "#1e2d3d",
                "linewidth": 0.8,
            },
        },
    )
    fig.text(0.5, 0.975, row["PlayerName"], ha="center", va="center", fontsize=18, fontweight="bold", color="#e8edf3")
    fig.text(
        0.5,
        0.945,
        f"{row['TeamName']} | {row['PositionGroup']} | percentiles vs position pool",
        ha="center",
        va="center",
        fontsize=10,
        color="#8fa3b1",
    )
    return fig


def render_score_distribution(df: pd.DataFrame, metric: str, highlight: float | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0f1623")
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    ax.hist(values, bins=28, color="#00d4a8", alpha=0.65, edgecolor="#080c14", linewidth=0.8)
    ax.axvline(values.median(), color="#8fa3b1", linewidth=1.8, linestyle="--", label=f"Median {values.median():.1f}")
    if highlight is not None:
        ax.axvline(highlight, color="#f59e0b", linewidth=2.5, label=f"Selected {highlight:.1f}")
    ax.set_title(f"{metric} distribution", loc="left", fontsize=13, fontweight="bold", color="#e8edf3")
    ax.set_xlabel("Score", color="#8fa3b1", fontsize=9)
    ax.set_ylabel("Players", color="#8fa3b1", fontsize=9)
    ax.tick_params(colors="#8fa3b1")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d3d")
    ax.grid(axis="y", color="#1e2d3d", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    legend = ax.legend(frameon=True, facecolor="#141d2b", edgecolor="#1e2d3d", labelcolor="#e8edf3", fontsize=9)
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
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0f1623")
    im = ax.imshow(pivot.fillna(np.nan), cmap="YlGnBu", aspect="auto", vmin=20, vmax=75)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, fontsize=9, fontweight="bold", color="#e8edf3")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index, fontsize=8, color="#8fa3b1")
    ax.tick_params(colors="#8fa3b1")
    ax.set_title(f"League depth | median {metric}", loc="left", fontsize=13, fontweight="bold", color="#e8edf3")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="#080c14", fontweight="bold")
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="#8fa3b1")
    fig.tight_layout()
    return fig


def render_position_boxplot(df: pd.DataFrame, metric: str):
    order = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
    groups = [pd.to_numeric(df.loc[df["PositionGroup"].eq(pos), metric], errors="coerce").dropna() for pos in order]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0f1623")
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=order, showfliers=False,
                    medianprops={"color": "#00d4a8", "linewidth": 2},
                    whiskerprops={"color": "#1e2d3d"}, capprops={"color": "#1e2d3d"})
    for patch, pos in zip(bp["boxes"], order):
        patch.set_facecolor(POSITION_COLORS.get(pos, "#457b9d"))
        patch.set_alpha(0.8)
        patch.set_edgecolor("#1e2d3d")
    ax.set_title(f"{metric} by position", loc="left", fontsize=13, fontweight="bold", color="#e8edf3")
    ax.set_ylabel("Score", color="#8fa3b1", fontsize=9)
    ax.tick_params(colors="#8fa3b1")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d3d")
    ax.grid(axis="y", color="#1e2d3d", linewidth=0.7)
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


def build_player_card_pdf(player: pd.Series, ref_df: pd.DataFrame, scout_notes: str = "") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Player Card: {player.get('PlayerName', 'Unknown')}", styles["Title"]))
    meta = " · ".join(filter(None, [
        str(player.get("TeamName", "")), str(player.get("PositionGroup", "")),
        str(player.get("BundleLabel", "")), f"Age {player.get('AgeYears', ''):.1f}" if pd.notna(player.get("AgeYears")) else "",
        f"{int(player.get('MinutesPlayed', 0)):,} min" if pd.notna(player.get("MinutesPlayed")) else "",
    ]))
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 12))

    scores_data = [["Quality", "Role Fit", "Impact", "Decision", "Reliability", "Risk Band"]]
    scores_data.append([
        f"{player.get('QualityScore', 0):.1f}", f"{player.get('RoleFitScore', 0):.1f}",
        f"{player.get('ProfileScore', 0):.1f}", f"{player.get('DecisionScore', 0):.1f}",
        f"{player.get('PerformanceReliabilityScore', 0):.1f}", str(player.get("RiskBand", "?")),
    ])
    story.append(Paragraph("Scores", styles["Heading2"]))
    story.append(_pdf_table(scores_data, header_color="#2a9d8f"))
    story.append(Spacer(1, 10))

    profile_rows = [["Field", "Value"]]
    for field, key in [("Archetype", "Archetype"), ("Quality tier", "QualityTier"), ("Readiness", "Readiness"),
                       ("Primary style", "PrimaryPlayerStyle"), ("Secondary style", "SecondaryPlayerStyle"),
                       ("Style summary", "PlayerStyleSummary"), ("Hradec fit", "WhyThisClubStyle"),
                       ("Style clubs", "SmartClubTop3"), ("Closeness tier", "SmartClubClosenessTier")]:
        val = str(player.get(key, "") or "")
        if val and val not in ("nan", "None", ""):
            profile_rows.append([field, shorten(val, width=80, placeholder="…")])
    if len(profile_rows) > 1:
        story.append(Paragraph("Profile", styles["Heading2"]))
        story.append(_pdf_table(profile_rows, header_color="#457b9d"))
        story.append(Spacer(1, 10))

    for field, key in [("Quality drivers", "QualityDrivers"), ("Fit drivers", "FitDrivers"), ("Risk flags", "RiskFlags")]:
        val = str(player.get(key, "") or "")
        if val and val not in ("nan", "None", ""):
            story.append(Paragraph(f"<b>{field}:</b> {shorten(val, width=120, placeholder='…')}", styles["Normal"]))
    story.append(Spacer(1, 8))

    if scout_notes:
        story.append(Paragraph("Scout Notes", styles["Heading2"]))
        story.append(Paragraph(scout_notes, styles["Normal"]))
        story.append(Spacer(1, 8))

    pos_group = str(player.get("PositionGroup", ""))
    comp = ref_df.loc[ref_df["PositionGroup"].eq(pos_group) & ref_df["PlayerName"].ne(str(player.get("PlayerName", "")))]
    comp = comp.sort_values("QualityScore", ascending=False).head(8)
    if not comp.empty:
        comp_cols = [c for c in ["PlayerName", "TeamName", "AgeYears", "QualityScore", "RoleFitScore", "Archetype"] if c in comp.columns]
        comp_view = comp[comp_cols].rename(columns={"PlayerName":"Player","TeamName":"Team","AgeYears":"Age","QualityScore":"Quality","RoleFitScore":"Fit"})
        story.append(Paragraph(f"Top {pos_group} comparators (position pool)", styles["Heading2"]))
        story.append(_pdf_table(_format_pdf_frame(comp_view, max_text=28), header_color="#102a43"))

    doc.build(story)
    return buffer.getvalue()


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


def hradec_recruitment_targets(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Score every external player on how well they'd address Hradec Kralove's squad needs."""
    hradec = hradec_squad(df)
    external = df.loc[
        ~df["TeamName"].fillna("").astype(str).str.lower().str.contains("hradec|kralove|králové", regex=True)
    ].copy()

    if hradec.empty or external.empty:
        external["HradecTargetScore"] = safe_col(external, "QualityScore")
        external["PositionNeed"] = "Unknown"
        return external.sort_values("HradecTargetScore", ascending=False).head(top_n)

    hradec_pos_quality = hradec.groupby("PositionGroup")["QualityScore"].median()
    hradec_pos_count = hradec.groupby("PositionGroup")["PlayerName"].count()

    def target_score(row: pd.Series) -> float:
        pos = str(row.get("PositionGroup", ""))
        quality = float(pd.to_numeric(row.get("QualityScore", 0), errors="coerce") or 0)
        value = float(pd.to_numeric(row.get("ValueRecruitmentScore", 50), errors="coerce") or 50)
        resale = float(pd.to_numeric(row.get("AgeResaleScore", 50), errors="coerce") or 50)
        smart = float(pd.to_numeric(row.get("SmartClubScore", 50), errors="coerce") or 50)
        reliability = float(pd.to_numeric(row.get("PerformanceReliabilityScore", 50), errors="coerce") or 50)
        hradec_q = float(hradec_pos_quality.get(pos, 50))
        count = int(hradec_pos_count.get(pos, 3))
        # Boost for positions where Hradec is weak or thin
        gap_bonus = max(0.0, (65.0 - hradec_q) / 65.0) * 20
        depth_bonus = max(0.0, (4.0 - count) / 4.0) * 10
        base = quality * 0.38 + value * 0.18 + resale * 0.12 + smart * 0.16 + reliability * 0.16
        return float(np.clip(base + gap_bonus + depth_bonus, 0, 130))

    external["HradecTargetScore"] = external.apply(target_score, axis=1)

    def need_label(pos: str) -> str:
        q = float(hradec_pos_quality.get(pos, 55))
        c = int(hradec_pos_count.get(pos, 3))
        if c == 0:
            return "🔴 No depth"
        if q < 48 or c <= 1:
            return "🔴 High need"
        if q < 55 or c <= 2:
            return "🟡 Medium need"
        return "🟢 Covered"

    external["PositionNeed"] = external["PositionGroup"].apply(need_label)
    return external.sort_values("HradecTargetScore", ascending=False).head(top_n)


def render_european_map(market_df: pd.DataFrame) -> alt.Chart:
    """Altair bubble map of Europe coloured by scout priority score."""
    background = alt.Chart({"values": [{}]}).mark_geoshape(
        fill="#0a1220", stroke="#1e2d3d", strokeWidth=0.4
    ).properties(width=700, height=440).project("mercator")

    bubbles = (
        alt.Chart(market_df)
        .mark_circle(opacity=0.88, stroke="#080c14", strokeWidth=1)
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            size=alt.Size(
                "Players:Q",
                scale=alt.Scale(range=[40, 700]),
                legend=alt.Legend(title="Players", orient="bottom-right", labelColor="#8fa3b1", titleColor="#8fa3b1"),
            ),
            color=alt.Color(
                "GoScore:Q",
                scale=alt.Scale(domain=[30, 75], range=["#1e2d3d", "#00d4a8"]),
                legend=alt.Legend(title="Scout priority", orient="bottom-left", labelColor="#8fa3b1", titleColor="#8fa3b1"),
            ),
            tooltip=[
                alt.Tooltip("Country:N", title="Country"),
                alt.Tooltip("Players:Q", title="Players"),
                alt.Tooltip("MedianScore:Q", title="Median quality", format=".1f"),
                alt.Tooltip("HighQualityShare:Q", title="High quality %", format=".1f"),
                alt.Tooltip("GoScore:Q", title="Scout priority", format=".1f"),
                alt.Tooltip("Recommendation:N", title="Action"),
                alt.Tooltip("MedianAge:Q", title="Median age", format=".1f"),
            ],
        )
    )

    labels = (
        alt.Chart(market_df.loc[market_df["GoScore"] >= 55])
        .mark_text(fontSize=9, fontWeight="bold", dy=-14, color="#e8edf3")
        .encode(longitude="lon:Q", latitude="lat:Q", text="Country:N")
    )

    return (
        (bubbles + labels)
        .properties(width=700, height=440)
        .project(type="mercator", center=[15, 52], scale=480)
        .configure_view(fill="#080c14", stroke=None)
        .configure(background="#080c14")
    )


def generate_scout_report(player: pd.Series, ref_df: pd.DataFrame) -> str:
    """Call Claude API to generate a 150-word scout report for a player."""
    try:
        import anthropic as _anthropic
        _client = _anthropic.Anthropic()
        _pos_df = ref_df.loc[ref_df["PositionGroup"].eq(player["PositionGroup"])]
        _pctile = percentile_rank(_pos_df["QualityScore"].dropna(), float(player["QualityScore"]))
        _prompt = (
            f"You are a professional football scout. Write a concise 150-word scouting report.\n\n"
            f"Player: {player['PlayerName']} | Team: {player['TeamName']} | League: {player.get('BundleLabel','?')}\n"
            f"Position: {player['PositionGroup']} | Age: {player['AgeYears']:.1f} | Minutes: {int(player['MinutesPlayed']):,}\n"
            f"Quality: {player['QualityScore']:.1f} ({_pctile:.0f}th pctile for {player['PositionGroup']}) | "
            f"Role Fit: {player.get('RoleFitScore',0):.1f} | Impact: {player.get('ProfileScore',0):.1f} | "
            f"Decision: {player.get('DecisionScore',0):.1f} | Reliability: {player.get('PerformanceReliabilityScore',0):.1f}\n"
            f"Risk: {player.get('RiskBand','?')} | Archetype: {player.get('Archetype','?')} | Tier: {player.get('QualityTier','?')}\n"
            f"Key drivers: {player.get('QualityDrivers','?')}\n\n"
            f"Write as a scout would: strengths, playing style, reliability/risk, and a signing verdict for "
            f"FC Hradec Králové (Czech top-flight club). Plain prose, no bullet points, ~150 words."
        )
        _msg = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=450,
            messages=[{"role": "user", "content": _prompt}],
        )
        return _msg.content[0].text
    except Exception as _e:
        return f"Report generation failed: {_e}"


def download_name(prefix: str, suffix: str) -> str:
    return f"fchk_{prefix.lower().replace(' ', '_')}.{suffix}"


def reset_filters() -> None:
    for key in [
        "positions_filter",
        "bundles_filter",
        "archetypes_filter",
        "countries_filter",
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


def _load_shortlist_file() -> dict:
    try:
        if SHORTLIST_FILE.exists():
            return json.loads(SHORTLIST_FILE.read_text())
    except Exception:
        pass
    return {}

def _save_shortlist_file(data: dict) -> None:
    try:
        SHORTLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        SHORTLIST_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

def add_to_shortlist(player_name: str, priority: str = "Watch", notes: str = "") -> None:
    sl = st.session_state.get("shortlist_data", {})
    if player_name not in sl:
        sl[player_name] = {"priority": priority, "notes": notes, "added": str(_date.today())}
    else:
        # update priority/notes if provided
        if priority != "Watch" or notes:
            sl[player_name]["priority"] = priority
            sl[player_name]["notes"] = notes
    st.session_state["shortlist_data"] = sl
    st.session_state["shortlist_players"] = list(sl.keys())
    _save_shortlist_file(sl)

def clear_shortlist() -> None:
    st.session_state["shortlist_data"] = {}
    st.session_state["shortlist_players"] = []
    _save_shortlist_file({})


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


WORKSPACES = ["Recruitment", "Scouting", "Goalkeepers", "Team", "Model"]
WYSCOUT_DB_DIR = APP_DIR / "data" / "Wyscout DB"


def set_workspace(section: str) -> None:
    st.session_state["active_workspace"] = section
    st.session_state["show_scouting_workspace"] = True
    st.session_state.pop("landing_notice", None)


def enter_scouting_workspace() -> None:
    set_workspace("Recruitment")


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


def _load_wyscout_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            return _clean_columns(pd.read_excel(path))
        except Exception:
            xl = pd.ExcelFile(path)
            frames = []
            for sheet in xl.sheet_names:
                try:
                    frames.append(_clean_columns(pd.read_excel(xl, sheet_name=sheet)))
                except Exception:
                    pass
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elif suffix == ".csv":
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return _clean_columns(pd.read_csv(path, encoding=enc))
            except Exception:
                continue
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_leagues_overview() -> pd.DataFrame:
    path = APP_DIR / "data" / "Leagues Overview.xlsx"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


_COUNTRY_ALIASES: dict[str, str] = {
    "Czech": "Czech Republic",
    "Moldovia": "Moldova",
    "Moldovia": "Moldova",
    "Turkiye": "Türkiye",
    "Saudi": "Saudi Arabia",
    "Korea": "South Korea",
    "Bosnia": "Bosnia",
    "USA": "USA",
    "Japan II III": "Japan",
}

_TIER_COLORS_WS: dict[str, str] = {
    "Elite":          "#e76f51",
    "Top":            "#f4a261",
    "Strong":         "#2a9d8f",
    "Developing":     "#457b9d",
    "Lower":          "#637d96",
    "Youth/Grassroots": "#4a5568",
}

_TIER_ICONS: dict[str, str] = {
    "Elite": "🏆",
    "Top": "⭐",
    "Strong": "💪",
    "Developing": "📈",
    "Lower": "🔍",
    "Youth/Grassroots": "🌱",
}


def _parse_wyscout_filename(filename: str) -> tuple[str, int | str]:
    """Return (country, division) from a Wyscout DB filename."""
    name = filename
    for ext in (".xlsx", ".xls", ".csv"):
        name = name.replace(ext, "")
    # Japan II III is a combined file
    if name == "Japan II III":
        return "Japan", 2
    # Numbered parts: "Germany 4 - Part I" → Germany, div 4
    import re as _re
    m = _re.match(r"^(.+?)\s+(\d)\s+-\s+Part\s+", name)
    if m:
        return m.group(1).strip(), int(m.group(2))
    # Roman-suffix parts: "Australia II - Part I" → Australia, div 2
    m2 = _re.match(r"^(.+?)\s+(IV|III|II)\s+-\s+Part\s+", name)
    if m2:
        return m2.group(1).strip(), {"II": 2, "III": 3, "IV": 4}[m2.group(2)]
    # Youth suffixes
    for tag, div in (("U19", "U19"), ("U17", "U17")):
        if name.endswith(tag):
            return name[: -len(tag)].strip(), div
        if f" {tag}" in name:
            return name.split(f" {tag}")[0].strip(), div
    # Standard roman suffixes
    for suffix, num in ((" IV", 4), (" III", 3), (" II", 2)):
        if name.endswith(suffix):
            return name[: -len(suffix)].strip(), num
    return name.strip(), 1


def _match_league(filename: str, leagues_df: pd.DataFrame) -> pd.Series | None:
    if leagues_df.empty:
        return None
    country_raw, division = _parse_wyscout_filename(filename)
    country = _COUNTRY_ALIASES.get(country_raw, country_raw)
    div_str = str(division)
    mask = (
        leagues_df["Country"].fillna("").str.lower().str.contains(country.lower(), regex=False)
        & leagues_df["Division"].astype(str).str.strip().eq(div_str)
    )
    match = leagues_df.loc[mask]
    if not match.empty:
        return match.iloc[0]
    # Fuzzy: just match country, take lowest division number for that country
    mask2 = leagues_df["Country"].fillna("").str.lower().str.contains(country.lower(), regex=False)
    match2 = leagues_df.loc[mask2]
    if not match2.empty:
        num_divs = pd.to_numeric(match2["Division"], errors="coerce")
        best = match2.loc[num_divs.eq(num_divs.min())]
        return best.iloc[0]
    return None


def render_scouting_workspace() -> None:
    st.markdown("<div class='workspace-label'>🔍 Scouting — Wyscout database browser</div>", unsafe_allow_html=True)

    leagues_df = load_leagues_overview()
    wyscout_files = sorted(
        [p for p in WYSCOUT_DB_DIR.glob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("~")],
        key=lambda p: p.name.lower(),
    )

    if not wyscout_files:
        st.markdown(
            '<div class="note-box" style="border-left-color:var(--amber);">'
            '<strong style="color:var(--amber);">No Wyscout files found.</strong><br>'
            f'Upload your exported Wyscout Excel or CSV files to '
            f'<code style="color:var(--teal);background:rgba(16,212,170,.08);padding:2px 6px;border-radius:4px;">data/Wyscout DB/</code>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Build file→league index ──────────────────────────────────────────────
    file_meta: dict[str, pd.Series | None] = {p.name: _match_league(p.name, leagues_df) for p in wyscout_files}

    def _tier_of(fname: str) -> str:
        m = file_meta.get(fname)
        return str(m["Tier Label"]) if m is not None and pd.notna(m.get("Tier Label")) else "Unknown"

    # ── Split layout: left nav | right table ─────────────────────────────────
    nav_col, main_col = st.columns([1, 3], gap="medium")

    with nav_col:
        # Tier filter
        all_tiers = ["All", "Elite", "Top", "Strong", "Developing", "Lower", "Youth/Grassroots"]
        sel_tier = st.selectbox("League tier", all_tiers, key="ws_tier_filter")
        if sel_tier and sel_tier != "All":
            visible_files = [p for p in wyscout_files if _tier_of(p.name) == sel_tier]
        else:
            visible_files = wyscout_files

        if not visible_files:
            st.info(f"No files for tier: {sel_tier}")
            return

        # File selector — show league name when available
        def _file_label(p: Path) -> str:
            m = file_meta.get(p.name)
            if m is not None and pd.notna(m.get("League Name")):
                icon = _TIER_ICONS.get(str(m["Tier Label"]), "")
                return f"{icon} {m['League Name']}"
            return p.name

        file_labels = [_file_label(p) for p in visible_files]
        _ws_key = "wyscout_selected_file"
        if st.session_state.get(_ws_key) not in file_labels:
            st.session_state[_ws_key] = file_labels[0]

        selected_label_ws = st.selectbox("League", file_labels, key=_ws_key)
        selected_path = visible_files[file_labels.index(selected_label_ws)]
        selected_name = selected_path.name

        # League info mini-card
        league_info = file_meta.get(selected_name)
        if league_info is not None:
            _t_col = _TIER_COLORS_WS.get(str(league_info.get("Tier Label", "")), "#637d96")
            _notes_str = f'<br><span style="color:var(--faint);font-size:.68rem;font-style:italic;">{escape(str(league_info["Notes"]))}</span>' if pd.notna(league_info.get("Notes")) else ""
            st.markdown(
                f'<div style="background:var(--surface);border:1px solid var(--border);border-left:3px solid {_t_col};'
                f'border-radius:6px;padding:9px 12px;margin:4px 0 10px;">'
                f'<div style="color:{_t_col};font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;">'
                f'{_TIER_ICONS.get(str(league_info["Tier Label"]),"")} {escape(str(league_info["Tier Label"]))} · Tier {int(league_info["Tier"])}</div>'
                f'<div style="color:var(--muted);font-size:.72rem;margin-top:3px;">{escape(str(league_info["Country"]))} · Div {escape(str(league_info["Division"]))}</div>'
                f'<div style="color:var(--faint);font-size:.68rem;">{int(league_info["Players in DB"]):,} players in DB</div>'
                f'{_notes_str}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            league_info = None

        # Load data here so filters below can use it
        @st.cache_data(show_spinner=False, ttl=300)
        def _cached_wyscout(path_str: str) -> pd.DataFrame:
            return _load_wyscout_file(Path(path_str))

        ws_df = _cached_wyscout(str(selected_path))
        if ws_df.empty:
            st.warning(f"Could not read **{selected_name}**.")
            return

        _ws_numeric_cols = ws_df.select_dtypes(include=[np.number]).columns.tolist()
        _ws_text_cols    = ws_df.select_dtypes(exclude=[np.number]).columns.tolist()
        _player_col = next((c for c in ["Player","PlayerName","Name","player","name"] if c in ws_df.columns), None)
        _team_col   = next((c for c in ["Team","TeamName","Club","team","club"] if c in ws_df.columns), None)
        _league_col = next((c for c in ["League","Competition","LeagueLabel","league","competition"] if c in ws_df.columns), None)
        _pos_col    = next((c for c in ["Position","PositionGroup","Pos","position","pos"] if c in ws_df.columns), None)
        _age_col    = next((c for c in ["Age","AgeYears","age"] if c in ws_df.columns), None)

        st.markdown("<div style='border-top:1px solid var(--border);margin:6px 0 10px;'></div>", unsafe_allow_html=True)

        # Filters — all stacked vertically
        ws_filtered = ws_df.copy()

        ws_search = st.text_input("Search player / team", key="ws_search", placeholder="Name…")
        if ws_search:
            _hcols = [c for c in [_player_col, _team_col] if c]
            if _hcols:
                _hay = ws_filtered[_hcols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                ws_filtered = ws_filtered.loc[_hay.str.contains(ws_search.lower(), regex=False)]

        if _pos_col:
            pos_opts = sorted(ws_df[_pos_col].dropna().astype(str).unique())
            sel_pos = st.multiselect("Position", pos_opts, default=pos_opts, key="ws_pos_filter")
            ws_filtered = ws_filtered.loc[ws_filtered[_pos_col].astype(str).isin(sel_pos)]

        if _team_col:
            team_opts = sorted(ws_df[_team_col].dropna().astype(str).unique())
            sel_teams = st.multiselect("Team", team_opts, default=team_opts, key="ws_team_filter")
            ws_filtered = ws_filtered.loc[ws_filtered[_team_col].astype(str).isin(sel_teams)]

        if _league_col:
            league_opts = sorted(ws_df[_league_col].dropna().astype(str).unique())
            sel_leagues = st.multiselect("Competition", league_opts, default=league_opts, key="ws_league_filter")
            ws_filtered = ws_filtered.loc[ws_filtered[_league_col].astype(str).isin(sel_leagues)]

        if _age_col:
            _age_s = pd.to_numeric(ws_df[_age_col], errors="coerce").dropna()
            if not _age_s.empty:
                _amin, _amax = float(np.floor(_age_s.min())), float(np.ceil(_age_s.max()))
                if _amin < _amax:
                    sel_age = st.slider("Age", _amin, _amax, (_amin, _amax), step=1.0, key="ws_age_filter")
                    ws_filtered = ws_filtered.loc[pd.to_numeric(ws_filtered[_age_col], errors="coerce").between(sel_age[0], sel_age[1])]

        if _ws_numeric_cols:
            sort_col_sel = st.selectbox("Sort by", _ws_numeric_cols, key="ws_sort_col")
        else:
            sort_col_sel = None

        st.markdown("<div style='border-top:1px solid var(--border);margin:6px 0 10px;'></div>", unsafe_allow_html=True)

        st.markdown(
            f'<div style="color:var(--faint);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;">'
            f'<span style="color:var(--ink);">{len(ws_filtered):,}</span> / {len(ws_df):,} rows</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download CSV",
            data=ws_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"wyscout_{selected_name.replace(' ','_').lower()}_filtered.csv",
            mime="text/csv",
            width="stretch",
        )

        # League index expander
        if not leagues_df.empty:
            with st.expander("🌍 League index", expanded=False):
                present_names: set[str] = set()
                for p in wyscout_files:
                    m = file_meta.get(p.name)
                    if m is not None:
                        present_names.add(str(m.get("League Name", "")))
                lo_view = leagues_df.copy()
                lo_view["✓"] = lo_view["League Name"].isin(present_names).map({True: "✓", False: ""})
                li_tier = st.selectbox("Tier", ["All"] + sorted(lo_view["Tier Label"].dropna().unique().tolist()), key="ws_li_tier")
                if li_tier != "All":
                    lo_view = lo_view.loc[lo_view["Tier Label"].eq(li_tier)]
                st.dataframe(
                    lo_view[["League Name","Country","Tier Label","Division","✓"]],
                    use_container_width=True, hide_index=True, height=420,
                )

    # ── Right column: KPI tiles + table ──────────────────────────────────────
    with main_col:
        from datetime import datetime as _dt
        _file_date = _dt.fromtimestamp(selected_path.stat().st_mtime).strftime("%-d %b %Y")
        _tier_label_ws = str(league_info["Tier Label"]) if league_info is not None else "—"
        _tier_num_ws   = f"Tier {int(league_info['Tier'])}" if league_info is not None else "—"

        st.markdown(
            '<div class="scouting-cockpit"><div class="cockpit-grid">'
            f'<div class="cockpit-tile"><div class="cockpit-label">Rows loaded</div>'
            f'<div class="cockpit-value">{len(ws_df):,}</div>'
            f'<div class="cockpit-note">{len(ws_df.columns)} cols · {_file_date}</div></div>'
            f'<div class="cockpit-tile"><div class="cockpit-label">Showing</div>'
            f'<div class="cockpit-value">{len(ws_filtered):,}</div>'
            f'<div class="cockpit-note">after filters</div></div>'
            f'<div class="cockpit-tile"><div class="cockpit-label">Players</div>'
            f'<div class="cockpit-value">{ws_df[_player_col].nunique() if _player_col else "?"}</div>'
            f'<div class="cockpit-note">unique</div></div>'
            f'<div class="cockpit-tile"><div class="cockpit-label">League tier</div>'
            f'<div class="cockpit-value" style="font-size:.9rem;">{_tier_label_ws}</div>'
            f'<div class="cockpit-note">{_tier_num_ws}</div></div>'
            f'<div class="cockpit-tile"><div class="cockpit-label">Metrics</div>'
            f'<div class="cockpit-value">{len(_ws_numeric_cols)}</div>'
            f'<div class="cockpit-note">{len(_ws_text_cols)} text cols</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # Main table
        if sort_col_sel and sort_col_sel in ws_filtered.columns:
            ws_display = ws_filtered.sort_values(sort_col_sel, ascending=False).reset_index(drop=True)
        else:
            ws_display = ws_filtered.reset_index(drop=True)

        col_config = {}
        for c in _ws_numeric_cols:
            if c in ws_display.columns:
                _cmin = float(ws_display[c].min()) if not ws_display[c].isna().all() else 0.0
                _cmax = float(ws_display[c].max()) if not ws_display[c].isna().all() else 100.0
                if _cmax > _cmin and 0 <= _cmin and _cmax <= 100:
                    col_config[c] = st.column_config.ProgressColumn(c, min_value=_cmin, max_value=_cmax, format="%.2f")

        st.dataframe(ws_display, use_container_width=True, hide_index=True, height=680, column_config=col_config)

        # Column explorer — below table as expander
        if _ws_numeric_cols:
            with st.expander("📊 Column explorer", expanded=False):
                _explore_col = st.selectbox("Metric", _ws_numeric_cols, key="ws_explore_col")
                if _explore_col in ws_filtered.columns:
                    _series = pd.to_numeric(ws_filtered[_explore_col], errors="coerce").dropna()
                    if not _series.empty:
                        exp_left, exp_right = st.columns([2, 1])
                        with exp_left:
                            _hist_chart = (
                                alt.Chart(pd.DataFrame({"value": _series}))
                                .mark_bar(color="#10d4aa", opacity=0.7, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                                .encode(
                                    x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=_explore_col,
                                            axis=alt.Axis(labelColor="#8fa3b1", titleColor="#8fa3b1", gridColor="#1e2d3d")),
                                    y=alt.Y("count():Q", title="Players", axis=alt.Axis(labelColor="#8fa3b1", gridColor="#1e2d3d")),
                                )
                                .properties(height=220)
                                .configure_view(fill="#0f1623", stroke=None).configure(background="#080c14")
                            )
                            st.altair_chart(_hist_chart, use_container_width=True)
                        with exp_right:
                            _stats = _series.describe()
                            _stats.index = ["Count","Mean","Std","Min","P25","Median","P75","Max"]
                            st.dataframe(_stats.reset_index().rename(columns={"index":"Stat",0:"Value"}).round(2),
                                         use_container_width=True, hide_index=True)


def render_model_workspace(data: pd.DataFrame, metadata: dict[str, pd.DataFrame]) -> None:
    st.markdown("<div class='workspace-label'>🔍 Scouting — Wyscout database browser</div>", unsafe_allow_html=True)

    wyscout_files = sorted([
        p for p in WYSCOUT_DB_DIR.glob("*")
        if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("~")
    ], key=lambda p: p.name.lower())

    if not wyscout_files:
        st.markdown(
            '<div class="note-box" style="border-left-color:var(--amber);">'
            '<strong style="color:var(--amber);">No Wyscout files found.</strong><br>'
            'Upload your exported Wyscout Excel or CSV files to:<br>'
            '<code style="color:var(--teal);background:rgba(16,212,170,.08);padding:2px 6px;border-radius:4px;">data/Wyscout DB/</code><br><br>'
            'Supported formats: <strong>.xlsx</strong>, <strong>.xls</strong>, <strong>.csv</strong>. '
            'Any Wyscout export works — player lists, match data, league exports.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # File selector
    file_names = [p.name for p in wyscout_files]
    _ws_key = "wyscout_selected_file"
    selected_name = st.selectbox("Data file", file_names, key=_ws_key)
    selected_path = WYSCOUT_DB_DIR / selected_name

    @st.cache_data(show_spinner=False, ttl=300)
    def _cached_wyscout(path_str: str) -> pd.DataFrame:
        return _load_wyscout_file(Path(path_str))

    with st.spinner("Loading…"):
        ws_df = _cached_wyscout(str(selected_path))

    if ws_df.empty:
        st.warning(f"Could not read data from **{selected_name}**. Check that the file is a valid Wyscout export.")
        return

    _mod_ts = selected_path.stat().st_mtime
    from datetime import datetime as _dt
    _file_date = _dt.fromtimestamp(_mod_ts).strftime("%-d %b %Y")

    # KPI tiles
    _ws_numeric_cols = ws_df.select_dtypes(include=[np.number]).columns.tolist()
    _ws_text_cols = ws_df.select_dtypes(exclude=[np.number]).columns.tolist()
    _player_col = next((c for c in ["Player", "PlayerName", "Name", "player", "name"] if c in ws_df.columns), None)
    _team_col = next((c for c in ["Team", "TeamName", "Club", "team", "club"] if c in ws_df.columns), None)
    _league_col = next((c for c in ["League", "Competition", "LeagueLabel", "league", "competition"] if c in ws_df.columns), None)
    _pos_col = next((c for c in ["Position", "PositionGroup", "Pos", "position", "pos"] if c in ws_df.columns), None)
    _age_col = next((c for c in ["Age", "AgeYears", "age"] if c in ws_df.columns), None)

    st.markdown(
        '<div class="scouting-cockpit"><div class="cockpit-grid">'
        f'<div class="cockpit-tile"><div class="cockpit-label">Rows</div>'
        f'<div class="cockpit-value">{len(ws_df):,}</div>'
        f'<div class="cockpit-note">{len(ws_df.columns)} columns · {_file_date}</div></div>'
        f'<div class="cockpit-tile"><div class="cockpit-label">Players</div>'
        f'<div class="cockpit-value">{ws_df[_player_col].nunique() if _player_col else "?"}</div>'
        f'<div class="cockpit-note">unique names</div></div>'
        f'<div class="cockpit-tile"><div class="cockpit-label">Teams</div>'
        f'<div class="cockpit-value">{ws_df[_team_col].nunique() if _team_col else "?"}</div>'
        f'<div class="cockpit-note">in this file</div></div>'
        f'<div class="cockpit-tile"><div class="cockpit-label">Leagues</div>'
        f'<div class="cockpit-value">{ws_df[_league_col].nunique() if _league_col else "?"}</div>'
        f'<div class="cockpit-note">competitions</div></div>'
        f'<div class="cockpit-tile"><div class="cockpit-label">Numeric metrics</div>'
        f'<div class="cockpit-value">{len(_ws_numeric_cols)}</div>'
        f'<div class="cockpit-note">{len(_ws_text_cols)} text columns</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Filters
    filt_cols = st.columns([1, 1, 1, 1])
    ws_filtered = ws_df.copy()

    with filt_cols[0]:
        if _pos_col:
            pos_opts = sorted(ws_df[_pos_col].dropna().astype(str).unique())
            sel_pos = st.multiselect("Position", pos_opts, default=pos_opts, key="ws_pos_filter")
            ws_filtered = ws_filtered.loc[ws_filtered[_pos_col].astype(str).isin(sel_pos)]
    with filt_cols[1]:
        if _team_col:
            team_opts = sorted(ws_df[_team_col].dropna().astype(str).unique())
            sel_teams = st.multiselect("Team", team_opts, default=team_opts, key="ws_team_filter")
            ws_filtered = ws_filtered.loc[ws_filtered[_team_col].astype(str).isin(sel_teams)]
    with filt_cols[2]:
        if _league_col:
            league_opts = sorted(ws_df[_league_col].dropna().astype(str).unique())
            sel_leagues = st.multiselect("League", league_opts, default=league_opts, key="ws_league_filter")
            ws_filtered = ws_filtered.loc[ws_filtered[_league_col].astype(str).isin(sel_leagues)]
    with filt_cols[3]:
        if _age_col and ws_df[_age_col].dtype in [np.float64, np.int64, float, int]:
            _age_min = float(np.floor(ws_df[_age_col].min()))
            _age_max = float(np.ceil(ws_df[_age_col].max()))
            if _age_min < _age_max:
                sel_age = st.slider("Age", _age_min, _age_max, (_age_min, _age_max), step=1.0, key="ws_age_filter")
                ws_filtered = ws_filtered.loc[ws_filtered[_age_col].between(sel_age[0], sel_age[1])]

    ws_search = st.text_input("Search player or team", key="ws_search", placeholder="Type name…")
    if ws_search:
        _hay_cols = [c for c in [_player_col, _team_col] if c]
        if _hay_cols:
            _haystack = ws_filtered[_hay_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
            ws_filtered = ws_filtered.loc[_haystack.str.contains(ws_search.lower(), regex=False)]

    st.markdown(
        f'<div style="color:var(--faint);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin:4px 0 8px;">'
        f'Showing <span style="color:var(--ink);">{len(ws_filtered):,}</span> of {len(ws_df):,} rows'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Sort control
    sort_opts = [c for c in _ws_numeric_cols[:20] if c in ws_filtered.columns]
    sort_tabs, data_tab = st.tabs(["📋 Data table", "📊 Column explorer"])

    with sort_tabs:
        if sort_opts:
            sort_col_sel = st.selectbox("Sort by", sort_opts, key="ws_sort_col")
            ws_display = ws_filtered.sort_values(sort_col_sel, ascending=False).reset_index(drop=True)
        else:
            ws_display = ws_filtered.reset_index(drop=True)
        col_config = {}
        for c in _ws_numeric_cols:
            if c in ws_display.columns:
                _cmin = float(ws_display[c].min()) if not ws_display[c].isna().all() else 0.0
                _cmax = float(ws_display[c].max()) if not ws_display[c].isna().all() else 100.0
                if _cmax > _cmin and _cmax <= 100 and _cmin >= 0:
                    col_config[c] = st.column_config.ProgressColumn(c, min_value=_cmin, max_value=_cmax, format="%.2f")
        st.dataframe(ws_display, use_container_width=True, hide_index=True, height=560, column_config=col_config)
        st.download_button(
            "⬇ Download filtered CSV",
            data=ws_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"wyscout_{selected_name.replace(' ', '_').lower()}_filtered.csv",
            mime="text/csv",
            width="stretch",
        )

    with data_tab:
        if _ws_numeric_cols:
            _explore_col = st.selectbox("Metric to explore", _ws_numeric_cols, key="ws_explore_col")
            if _explore_col in ws_filtered.columns:
                _series = pd.to_numeric(ws_filtered[_explore_col], errors="coerce").dropna()
                _hist_df = pd.DataFrame({"value": _series})
                _hist_chart = (
                    alt.Chart(_hist_df)
                    .mark_bar(color="#10d4aa", opacity=0.7, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                    .encode(
                        x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=_explore_col,
                                axis=alt.Axis(labelColor="#8fa3b1", titleColor="#8fa3b1", gridColor="#1e2d3d")),
                        y=alt.Y("count():Q", title="Players",
                                axis=alt.Axis(labelColor="#8fa3b1", gridColor="#1e2d3d")),
                    )
                    .properties(height=280, title=alt.TitleParams(f"Distribution of {_explore_col}", color="#8fa3b1", fontSize=12))
                    .configure_view(fill="#0f1623", stroke=None)
                    .configure(background="#080c14")
                )
                st.altair_chart(_hist_chart, use_container_width=True)
                _stats = _series.describe().rename({
                    "count": "Count", "mean": "Mean", "std": "Std dev",
                    "min": "Min", "25%": "25th pctile", "50%": "Median",
                    "75%": "75th pctile", "max": "Max",
                })
                st.dataframe(_stats.reset_index().rename(columns={"index": "Stat", 0: "Value"}).round(3),
                             use_container_width=True, hide_index=True)
                if _player_col and not ws_filtered.empty:
                    top10 = ws_filtered[[_player_col] + ([_team_col] if _team_col else []) + [_explore_col]].dropna(subset=[_explore_col]).sort_values(_explore_col, ascending=False).head(10)
                    st.markdown(f"<div class='workspace-label' style='font-size:.58rem;margin:10px 0 6px;'>Top 10 by {_explore_col}</div>", unsafe_allow_html=True)
                    st.dataframe(top10.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("No numeric columns detected in this file.")

    # File index when multiple files loaded
    if len(wyscout_files) > 1:
        with st.expander(f"All files in Wyscout DB ({len(wyscout_files)})", expanded=False):
            _file_rows = []
            for fp in wyscout_files:
                _fts = _dt.fromtimestamp(fp.stat().st_mtime).strftime("%-d %b %Y")
                _fsize = f"{fp.stat().st_size / 1024:.0f} KB"
                _file_rows.append({"File": fp.name, "Size": _fsize, "Modified": _fts})
            st.dataframe(pd.DataFrame(_file_rows), use_container_width=True, hide_index=True)


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
        st.dataframe(smart_df.head(50), use_container_width=True, hide_index=True)
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
            use_container_width=True,
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
                use_container_width=True,
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
        use_container_width=True,
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
            st.dataframe(related_styles[style_cols].head(50), use_container_width=True, hide_index=True)


def render_goalkeepers_workspace(data: pd.DataFrame) -> None:
    keeper_df = add_scouting_fields(data, BALANCED_WEIGHTS)
    keeper_df = keeper_df.loc[keeper_df["PositionGroup"].astype(str).eq("GK")].sort_values(
        ["ScoutFitScore", "PerformanceReliabilityScore"],
        ascending=False,
    )

    st.markdown("<div class='workspace-label'>Goalkeepers workspace</div>", unsafe_allow_html=True)

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
            Goalkeepers are kept separate so the board only compares GK profiles with other GK profiles.
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
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.1f"),
            "Model": st.column_config.ProgressColumn("Model", min_value=0, max_value=100, format="%.1f"),
            "Decision": st.column_config.ProgressColumn("Decision", min_value=0, max_value=100, format="%.1f"),
            "Reliability": st.column_config.ProgressColumn("Reliability", min_value=0, max_value=100, format="%.1f"),
        },
    )


def render_case_analysis_tab(filtered_df: pd.DataFrame) -> None:
    """Transfer value, resale upside, style-fit and cost-risk analysis tab."""
    rd = filtered_df.copy()
    value_score = safe_col(rd, "ValueRecruitmentScore")
    resale_score = safe_col(rd, "AgeResaleScore")
    style_score = safe_col(rd, "FCHKStyleScore", safe_col(rd, "SmartClubScore").median())
    smart_club_score = safe_col(rd, "SmartClubScore", style_score.median())
    success_score = safe_col(rd, "SuccessProbability")
    readiness_score = safe_col(rd, "PerformanceReliabilityScore")
    wage_risk = safe_col(rd, "WageRisk")
    fee_risk = safe_col(rd, "FeeRisk")

    rd["CaseScore"] = (
        value_score * 0.24 + resale_score * 0.20 + style_score * 0.18
        + smart_club_score * 0.14 + success_score * 0.14 + readiness_score * 0.10
        - wage_risk * 0.04 - fee_risk * 0.04
    ).clip(0, 100)
    rd["StyleFit"] = np.where(style_score.ge(70), "Strong", np.where(style_score.ge(55), "Useful", "Question"))
    rd["CostRisk"] = np.select(
        [fee_risk.ge(70) | wage_risk.ge(70), fee_risk.ge(50) | wage_risk.ge(50)],
        ["High", "Medium"], default="Low",
    )
    rd = rd.sort_values(["CaseScore", "ValueRecruitmentScore"], ascending=False)

    _kpi_cols = st.columns(4)
    with _kpi_cols[0]: metric_card("Median transfer value", f"{value_score.median():.1f}", "0–100 scale")
    with _kpi_cols[1]: metric_card("Median resale upside", f"{resale_score.median():.1f}", "age × potential")
    with _kpi_cols[2]: metric_card("Strong style fit", f"{rd['StyleFit'].eq('Strong').sum():,}", "FCHK + smart-club")
    with _kpi_cols[3]: metric_card("Low cost risk", f"{rd['CostRisk'].eq('Low').sum():,}", "fee + wage combined")

    st.markdown(
        '<div class="note-box">Case score = value 24% + resale 20% + style fit 18% + smart-club 14% + success 14% + readiness 10% − cost risk 8%.</div>',
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        role_opts = sorted(rd["PositionGroup"].dropna().astype(str).unique())
        sel_roles = st.multiselect("Roles", role_opts, default=role_opts, key="ca_roles")
    with fc2:
        min_resale = st.slider("Min resale score", 0, 100, 40, key="ca_resale")
    with fc3:
        sel_style = st.multiselect("Style fit", ["Strong", "Useful", "Question"], default=["Strong", "Useful", "Question"], key="ca_style")

    rd = rd.loc[rd["PositionGroup"].astype(str).isin(sel_roles) & rd["AgeResaleScore"].ge(min_resale) & rd["StyleFit"].isin(sel_style)].copy()

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("<div class='workspace-label' style='font-size:.58rem;margin-bottom:6px;'>Best recruitment cases</div>", unsafe_allow_html=True)
        _board_cols = [c for c in ["PlayerName","TeamName","PositionGroup","AgeYears","CaseScore","ValueRecruitmentScore","AgeResaleScore","SuccessProbability","StyleFit","CostRisk","PrimaryPlayerStyle","SmartClubTop3"] if c in rd.columns]
        st.dataframe(
            rd[_board_cols].head(100).rename(columns={"PlayerName":"Player","TeamName":"Team","PositionGroup":"Role","AgeYears":"Age","CaseScore":"Case","ValueRecruitmentScore":"Value","AgeResaleScore":"Resale","SuccessProbability":"Success","PrimaryPlayerStyle":"Style","SmartClubTop3":"Style Clubs"}).round(2),
            use_container_width=True, hide_index=True,
            column_config={
                "Case": st.column_config.ProgressColumn("Case", min_value=0, max_value=100, format="%.1f"),
                "Value": st.column_config.ProgressColumn("Value", min_value=0, max_value=100, format="%.1f"),
                "Resale": st.column_config.ProgressColumn("Resale", min_value=0, max_value=100, format="%.1f"),
                "Success": st.column_config.ProgressColumn("Success", min_value=0, max_value=100, format="%.1f"),
                "Age": st.column_config.NumberColumn("Age", format="%.1f"),
            },
        )
    with right:
        chart_df = rd[["PlayerName","PositionGroup","CaseScore","AgeResaleScore","FCHKStyleScore","CostRisk"]].dropna() if "FCHKStyleScore" in rd.columns else pd.DataFrame()
        if not chart_df.empty:
            case_chart = (
                alt.Chart(chart_df.head(300))
                .mark_circle(size=72, opacity=0.78)
                .encode(
                    x=alt.X("AgeResaleScore:Q", title="Resale upside", axis=alt.Axis(labelColor="#8fa3b1", titleColor="#8fa3b1", gridColor="#1e2d3d")),
                    y=alt.Y("FCHKStyleScore:Q", title="Playing-style fit", axis=alt.Axis(labelColor="#8fa3b1", titleColor="#8fa3b1", gridColor="#1e2d3d")),
                    color=alt.Color("CostRisk:N", title="Cost risk", scale=alt.Scale(domain=["Low","Medium","High"], range=["#2a9d8f","#f4a261","#e76f51"])),
                    shape=alt.Shape("PositionGroup:N", title="Role"),
                    tooltip=["PlayerName","PositionGroup",alt.Tooltip("CaseScore:Q",format=".1f"),alt.Tooltip("AgeResaleScore:Q",format=".1f"),"CostRisk"],
                )
                .properties(height=380, title=alt.TitleParams("Resale upside vs style fit", color="#8fa3b1", fontSize=12))
                .configure_view(fill="#0f1623", stroke=None).configure(background="#080c14")
                .interactive()
            )
            st.altair_chart(case_chart, use_container_width=True)
        else:
            _shape = rd.groupby(["PositionGroup","StyleFit"]).agg(Players=("PlayerName","count"),MedianCase=("CaseScore","median")).round(1).reset_index().sort_values("MedianCase",ascending=False)
            st.dataframe(_shape, use_container_width=True, hide_index=True)


def render_team_workspace(data: pd.DataFrame) -> None:
    team_df = add_scouting_fields(data, BALANCED_WEIGHTS)
    czech_df = czech_market(team_df)
    hradec_df = hradec_squad(team_df)
    external_czech = czech_df.loc[~czech_df["TeamName"].isin(hradec_df["TeamName"].unique())].copy()

    st.markdown("<div class='workspace-label'>Team workspace</div>", unsafe_allow_html=True)

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
        use_container_width=True,
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
            use_container_width=True,
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
            use_container_width=True,
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
        st.dataframe(league_summary, use_container_width=True, hide_index=True)


st.markdown(
    """
    <style>
    /* ═══════════════════════════════════════════════════════════════
       DESIGN TOKENS  — softer dark: slate-blue palette, not black
    ═══════════════════════════════════════════════════════════════ */
    :root {
        --bg:       #111827;   /* page background – dark slate, not black */
        --surface:  #1c2537;   /* card / panel background                 */
        --raised:   #243044;   /* elevated elements, inputs               */
        --border:   #2e3f55;   /* subtle dividers                         */
        --border2:  #3a5068;   /* stronger borders, outlines              */
        --teal:     #10d4aa;   /* primary accent                          */
        --teal-dim: #0aaa88;   /* darker teal for hover                   */
        --amber:    #f5a623;   /* warning / resale                        */
        --red:      #f05252;   /* danger / high risk                      */
        --green:    #34d399;   /* success / confirmed                     */
        --blue:     #60a5fa;   /* info / links                            */
        --purple:   #a78bfa;   /* special labels                          */
        --ink:      #f1f5f9;   /* primary text – soft white               */
        --muted:    #94afc4;   /* secondary text                          */
        --faint:    #637d96;   /* placeholder, disabled                   */
        --shadow:   0 4px 20px rgba(0,0,0,.35);
        --shadow-sm:0 2px 8px  rgba(0,0,0,.25);
    }

    /* ── GLOBAL ──────────────────────────────────────────────── */
    .stApp { background: var(--bg) !important; color: var(--ink); }

    .block-container {
        padding-top: .5rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* ── TIGHTEN STREAMLIT'S BUILT-IN VERTICAL RHYTHM ────────── */
    /* Each rendered widget/markdown sits in an .element-container;
       Streamlit's default gap is ~1rem — pull it in to 0.45rem    */
    div[data-testid="stVerticalBlock"] > div.element-container,
    div[data-testid="stVerticalBlockBorderWrapper"]
        > div[data-testid="stVerticalBlock"]
        > div.element-container {
        margin-bottom: 0.45rem !important;
    }
    /* The stVerticalBlock flex gap itself */
    div[data-testid="stVerticalBlock"] { gap: 0 !important; }
    /* Tabs content — a little more breathing room inside each tab */
    div[data-testid="stTabsContent"] > div[data-testid="stVerticalBlock"] > div.element-container {
        margin-bottom: 0.6rem !important;
    }
    /* Reduce the large bottom margin st.subheader / h2 adds */
    h2 { margin-bottom: 6px !important; padding-bottom: 4px; }
    h3 { margin-bottom: 4px !important; }
    /* Columns: tighten default gap between col children */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"]
        > div.element-container {
        margin-bottom: 0.4rem !important;
    }
    /* Reduce st.metric bottom margin */
    div[data-testid="stMetric"] { margin-bottom: 0 !important; }

    /* make dataframes truly full-width */
    [data-testid="stDataFrame"] > div { width: 100% !important; }

    h1, h2, h3 { color: var(--ink) !important; font-weight: 800; }

    h2 {
        font-size: .82rem !important;
        text-transform: uppercase;
        letter-spacing: .12em;
        border-bottom: 1px solid var(--border) !important;
        padding-bottom: 7px;
        margin-bottom: 12px !important;
        color: var(--muted) !important;
    }

    /* ── SIDEBAR ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0d1422 !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] .block-container { padding-top: .8rem; }
    section[data-testid="stSidebar"] label  { color: var(--muted) !important; font-size: .76rem !important; font-weight: 600 !important; }
    section[data-testid="stSidebar"] p      { color: var(--muted) !important; }
    section[data-testid="stSidebar"] h2     { border-bottom-color: var(--border) !important; }

    /* ── SIDEBAR BRAND ───────────────────────────────────────── */
    .sidebar-brand {
        background: linear-gradient(135deg, rgba(16,212,170,.12) 0%, transparent 60%);
        border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        padding: 12px 14px;
        margin-bottom: 16px;
        border-radius: 6px;
    }
    .sidebar-brand-title { color: var(--teal) !important; font-size: 1rem; font-weight: 900; letter-spacing: .04em; }
    .sidebar-brand-meta  { color: var(--faint) !important; font-size: .68rem; margin-top: 3px; }

    .menu-caption {
        color: var(--faint) !important;
        font-size: .72rem;
        line-height: 1.5;
        background: rgba(255,255,255,.03);
        border-left: 2px solid var(--border2);
        padding: 7px 10px;
        margin-bottom: 14px;
        border-radius: 0 4px 4px 0;
    }

    /* ── QUICK CHIPS ─────────────────────────────────────────── */
    .quick-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 0 0 6px 0;
        align-items: center;
    }
    .quick-chip-label {
        color: var(--faint);
        font-size: .62rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .1em;
    }

    /* ── LANDING HERO ────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #162035 0%, #1c2537 55%, #1a2d3e 100%);
        border: 1px solid var(--border2);
        border-radius: 10px;
        padding: 48px 44px 40px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -80px; right: -80px;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(16,212,170,.14) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero::after {
        content: "";
        position: absolute;
        bottom: -40px; left: 30%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(96,165,250,.07) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero-kicker {
        color: var(--teal);
        font-size: .68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .2em;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .hero h1 {
        color: #fff !important;
        font-size: clamp(2rem, 3.5vw, 3.4rem);
        font-weight: 900;
        line-height: 1.05;
        margin: 0 0 14px 0;
    }
    .hero h1 span { color: var(--teal); }
    .hero p { color: var(--muted); font-size: .95rem; line-height: 1.65; max-width: 540px; margin: 0; }
    .hero-stats { display: flex; gap: 36px; margin-top: 32px; flex-wrap: wrap; }
    .hero-stat-item { display: flex; flex-direction: column; gap: 3px; }
    .hero-stat-value { color: #fff; font-size: 1.7rem; font-weight: 900; line-height: 1; }
    .hero-stat-label { color: var(--faint); font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; }

    /* ── LANDING CARDS ───────────────────────────────────────── */
    .landing-grid { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 10px; margin-bottom: 10px; }
    .landing-card {
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 8px;
        padding: 16px 14px 14px;
        min-height: 106px;
        box-shadow: var(--shadow-sm);
        transition: border-color .2s, box-shadow .2s;
    }
    .landing-card:hover { border-color: var(--teal); box-shadow: 0 0 0 1px var(--teal), var(--shadow); }
    .landing-card-label { color: var(--teal); font-size: .6rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
    .landing-card-title { color: var(--ink); font-size: 1rem; font-weight: 800; margin-top: 6px; line-height: 1.1; }
    .landing-card-copy  { color: var(--faint); font-size: .72rem; line-height: 1.4; margin-top: 6px; }

    /* ── WORKSPACE LABEL ─────────────────────────────────────── */
    .workspace-label {
        border-left: 2px solid var(--teal);
        color: var(--teal);
        font-size: .6rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin: 2px 0 8px 0;
        padding: 4px 0 4px 10px;
        background: rgba(16,212,170,.06);
        border-radius: 0 4px 4px 0;
    }

    /* ── INTEL STRIP ─────────────────────────────────────────── */
    .intel-strip {
        border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        background: var(--surface);
        border-radius: 6px;
        padding: 7px 13px;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        box-shadow: var(--shadow-sm);
    }
    .intel-strip-title { color: var(--ink); font-size: .82rem; font-weight: 800; letter-spacing: .03em; }
    .intel-strip-meta  { color: var(--faint); font-size: .63rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }

    /* ── METRIC / KPI CARDS ──────────────────────────────────── */
    .metric-card {
        border: 1px solid var(--border);
        border-top: 2px solid var(--teal);
        background: var(--surface);
        padding: 10px 13px;
        border-radius: 6px;
        box-shadow: var(--shadow-sm);
        transition: border-top-color .18s;
    }
    .metric-card:hover { border-top-color: var(--teal-dim); }
    .metric-label  { color: var(--faint); font-size: .57rem; text-transform: uppercase; letter-spacing: .13em; font-weight: 700; }
    .metric-value  { margin-top: 3px; color: var(--ink); font-size: 1.35rem; font-weight: 900; line-height: 1; }
    .metric-caption{ color: var(--muted); font-size: .64rem; margin-top: 3px; line-height: 1.3; }

    /* ── COCKPIT KPIs ────────────────────────────────────────── */
    .scouting-cockpit { border: 1px solid var(--border); background: var(--surface); border-radius: 8px; padding: 10px 11px; margin: 0 0 10px 0; box-shadow: var(--shadow-sm); }
    .cockpit-grid { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 8px; }
    .cockpit-tile {
        border: 1px solid var(--border);
        background: var(--raised);
        border-radius: 6px;
        padding: 10px 12px 9px;
        min-height: 66px;
        position: relative;
        overflow: hidden;
        transition: border-color .18s;
    }
    .cockpit-tile:hover { border-color: var(--border2); }
    .cockpit-tile::after {
        content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--teal), transparent);
    }
    .cockpit-label { color: var(--faint); font-size: .56rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .cockpit-value { color: var(--ink); font-size: 1.3rem; font-weight: 900; line-height: 1; margin-top: 4px; }
    .cockpit-note  { color: var(--muted); font-size: .61rem; line-height: 1.3; margin-top: 3px; }

    /* ── ROLE RAIL ───────────────────────────────────────────── */
    .role-rail { display: grid; grid-template-columns: repeat(7, minmax(0,1fr)); gap: 6px; margin: 0 0 10px 0; }
    .role-cell {
        border: 1px solid var(--border);
        border-top: 2px solid var(--teal);
        background: var(--raised);
        border-radius: 6px;
        padding: 7px 9px 6px;
        min-height: 52px;
        transition: border-top-color .18s;
    }
    .role-cell-role  { color: var(--faint); font-size: .54rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .role-cell-score { color: var(--ink); font-size: 1rem; font-weight: 900; line-height: 1; margin-top: 3px; }
    .role-cell-count { color: var(--faint); font-size: .6rem; margin-top: 2px; }

    /* ── PROFILE CARD ────────────────────────────────────────── */
    .profile-card {
        border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        background: var(--surface);
        padding: 11px 14px;
        border-radius: 6px;
        box-shadow: var(--shadow-sm);
    }
    .profile-name { color: var(--ink); font-weight: 900; font-size: 1rem; line-height: 1.1; }
    .profile-meta { color: var(--muted); font-size: .72rem; margin-top: 3px; line-height: 1.4; }
    .pill-row { margin-top: 7px; display: flex; flex-wrap: wrap; gap: 5px; }
    .pill {
        border: 1px solid var(--border2);
        color: var(--muted);
        background: var(--raised);
        padding: 3px 8px;
        font-size: .59rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
        border-radius: 20px;
    }
    .pill.teal  { border-color: rgba(16,212,170,.5);  color: var(--teal);  background: rgba(16,212,170,.1); }
    .pill.amber { border-color: rgba(245,166,35,.5);  color: var(--amber); background: rgba(245,166,35,.1); }
    .pill.red   { border-color: rgba(240,82,82,.5);   color: var(--red);   background: rgba(240,82,82,.1);  }

    /* ── NOTE BOX ────────────────────────────────────────────── */
    .note-box {
        border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        background: rgba(16,212,170,.05);
        padding: 9px 12px;
        color: var(--muted);
        font-size: .75rem;
        line-height: 1.5;
        border-radius: 0 6px 6px 0;
    }

    .section-card { border: 1px solid var(--border); background: var(--surface); padding: 10px 12px; border-radius: 6px; }

    /* ── SCOUTING COMMAND ────────────────────────────────────── */
    .scouting-command {
        background: linear-gradient(135deg, #162035 0%, #1c2537 100%);
        border: 1px solid var(--border2);
        border-radius: 10px;
        padding: 22px 24px;
        margin: 0 0 16px 0;
        display: grid;
        grid-template-columns: minmax(0,1.5fr) minmax(220px,.8fr);
        gap: 20px;
        align-items: center;
        box-shadow: var(--shadow);
    }
    .scouting-kicker { color: var(--teal); font-size: .62rem; font-weight: 800; letter-spacing: .2em; text-transform: uppercase; }
    .scouting-title  { color: var(--ink); font-size: clamp(1.6rem, 3vw, 2.6rem); font-weight: 900; line-height: 1.05; margin-top: 6px; }
    .scouting-copy   { color: var(--muted); font-size: .82rem; line-height: 1.55; margin-top: 8px; }
    .scouting-mode-panel { border-left: 1px solid var(--border2); padding-left: 22px; }
    .scouting-mode-label { color: var(--faint); font-size: .58rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .scouting-mode-value { color: var(--ink); font-size: 1.2rem; font-weight: 900; line-height: 1; margin-top: 6px; }

    /* ── QUALITY HERO ────────────────────────────────────────── */
    .quality-hero {
        background: linear-gradient(135deg, #162035 0%, #1c2537 100%);
        border: 1px solid var(--border2);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: grid;
        grid-template-columns: minmax(0,1.3fr) minmax(200px,.7fr);
        gap: 14px;
        align-items: center;
        box-shadow: var(--shadow-sm);
    }
    .quality-title { color: var(--ink); font-size: 1.5rem; font-weight: 900; line-height: 1; }
    .quality-copy  { color: var(--muted); font-size: .76rem; line-height: 1.5; margin-top: 6px; }
    .quality-signal { border-left: 1px solid var(--border2); padding-left: 18px; }
    .quality-signal-label { color: var(--faint); font-size: .58rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .quality-signal-value { color: var(--teal); font-size: 1.35rem; font-weight: 900; margin-top: 5px; }

    .quality-card-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; margin: 14px 0; }
    .quality-player-card {
        border: 1px solid var(--border);
        border-top: 2px solid var(--teal);
        background: var(--raised);
        border-radius: 6px;
        padding: 12px 14px;
        min-height: 90px;
        box-shadow: var(--shadow-sm);
        transition: border-top-color .2s, box-shadow .2s;
    }
    .quality-player-card:hover { border-top-color: var(--teal-dim); box-shadow: var(--shadow); }
    .quality-rank       { color: var(--teal); font-size: .58rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .quality-player-name{ color: var(--ink); font-size: 1rem; font-weight: 800; line-height: 1.1; margin-top: 5px; }
    .quality-player-meta{ color: var(--muted); font-size: .68rem; line-height: 1.35; margin-top: 4px; }

    /* ── RESPONSIVE ──────────────────────────────────────────── */
    @media (max-width: 900px) {
        .hero, .landing-grid, .scouting-command, .cockpit-grid,
        .role-rail, .quality-hero, .quality-card-grid { grid-template-columns: 1fr; }
        .hero { padding: 26px 22px; }
    }

    /* ═══════════════════════════════════════════════════════════
       STREAMLIT WIDGET OVERRIDES
    ═══════════════════════════════════════════════════════════ */

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 6px !important;
        border: 1px solid var(--border2) !important;
        background: var(--raised) !important;
        color: var(--muted) !important;
        font-weight: 700 !important;
        font-size: .72rem;
        letter-spacing: .05em;
        text-transform: uppercase;
        box-shadow: none !important;
        transition: all .18s !important;
        padding: 0 12px !important;
        min-height: 32px;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: rgba(16,212,170,.15) !important;
        border-color: var(--teal) !important;
        color: var(--teal) !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--teal) !important;
        color: var(--teal) !important;
        background: rgba(16,212,170,.1) !important;
    }

    /* Tabs */
    div[data-testid="stTabs"] { border-bottom: 1px solid var(--border); margin-bottom: 0 !important; }
    div[data-testid="stTabs"] button {
        font-weight: 700 !important;
        font-size: .7rem !important;
        letter-spacing: .06em;
        color: var(--faint) !important;
        padding: 8px 14px !important;
        border-radius: 0 !important;
        transition: color .18s !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal) !important;
        border-bottom: 2px solid var(--teal) !important;
        background: rgba(16,212,170,.06) !important;
    }
    div[data-testid="stTabs"] button:hover { color: var(--ink) !important; }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        background: var(--surface) !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stExpander"] summary { color: var(--muted) !important; font-weight: 600 !important; }

    /* Native st.metric */
    div[data-testid="stMetric"] {
        border: 1px solid var(--border) !important;
        border-top: 3px solid var(--teal) !important;
        padding: 12px 14px;
        background: var(--surface) !important;
        border-radius: 6px !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stMetric"] label { color: var(--faint) !important; font-size: .62rem !important; text-transform: uppercase; letter-spacing: .1em; font-weight: 700 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--ink) !important; font-size: 1.35rem !important; font-weight: 900 !important; }

    /* Selects, inputs */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: var(--raised) !important;
        border-color: var(--border) !important;
        border-radius: 6px !important;
        color: var(--ink) !important;
    }
    input, textarea {
        background: var(--raised) !important;
        border-color: var(--border) !important;
        color: var(--ink) !important;
        border-radius: 6px !important;
    }
    input::placeholder, textarea::placeholder { color: var(--faint) !important; }
    [data-baseweb="tag"] { background: rgba(16,212,170,.14) !important; border-radius: 4px !important; color: var(--teal) !important; }

    /* Sliders */
    .stSlider [data-baseweb="slider"] [role="slider"] { background: var(--teal) !important; }
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] { color: var(--faint) !important; }

    /* Segmented control */
    div[data-testid="stSegmentedControl"] button {
        background: var(--raised) !important;
        color: var(--muted) !important;
        border-color: var(--border) !important;
        border-radius: 6px !important;
        font-size: .72rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: rgba(16,212,170,.18) !important;
        border-color: var(--teal) !important;
        color: var(--teal) !important;
    }

    /* Toggle */
    div[data-testid="stToggle"] label { color: var(--muted) !important; }

    /* Alerts / info boxes */
    .stAlert {
        background: var(--surface) !important;
        border-color: var(--border2) !important;
        border-radius: 6px !important;
        color: var(--muted) !important;
    }

    /* Multiselect option hover */
    li[role="option"]:hover { background: rgba(16,212,170,.1) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--surface); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--faint); }

    /* ═══════════════════════════════════════════════════════════
       DASHBOARD ICON CARDS  (landing page)
    ═══════════════════════════════════════════════════════════ */
    .dash-card {
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 12px;
        padding: 28px 18px 22px;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: border-color .2s, box-shadow .2s, transform .15s;
        cursor: default;
        margin-bottom: 0px;
        min-height: 210px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .dash-card:hover {
        border-color: var(--teal);
        box-shadow: 0 0 0 1px var(--teal), var(--shadow);
        transform: translateY(-2px);
    }
    .dash-card--primary {
        border-color: var(--teal);
        background: linear-gradient(160deg, rgba(16,212,170,.12) 0%, var(--surface) 55%);
    }
    .dash-card--primary:hover {
        box-shadow: 0 0 0 2px var(--teal), var(--shadow);
    }
    .dash-card-icon {
        font-size: 3.6rem;
        line-height: 1;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,.4));
    }
    .dash-card-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -.01em;
    }
    .dash-card-desc {
        color: var(--faint);
        font-size: .72rem;
        line-height: 1.45;
        max-width: 160px;
    }

    /* tighten the gap between the card and its button */
    .dash-card + div[data-testid="stButton"],
    .dash-card + div > div[data-testid="stButton"] {
        margin-top: 0 !important;
    }

    /* ═══════════════════════════════════════════════════════════
       FULL-BLEED LANDING PAGE  (.lp-*)
    ═══════════════════════════════════════════════════════════ */

    .lp-wrapper { width: 100%; background: var(--bg); }

    /* ── HERO ─────────────────────────────────────────────── */
    .lp-hero {
        display: grid;
        grid-template-columns: minmax(0,1.1fr) minmax(0,.9fr);
        gap: 48px;
        min-height: 62vh;
        padding: 64px 64px 56px;
        background: linear-gradient(140deg, #080e1c 0%, #0d1625 40%, #111827 70%, #0f1e30 100%);
        border-bottom: 1px solid var(--border);
        position: relative;
        overflow: hidden;
        align-items: center;
    }
    /* teal glow — top right */
    .lp-hero::before {
        content:"";
        position:absolute; top:-100px; right:-80px;
        width:560px; height:560px;
        background:radial-gradient(circle, rgba(16,212,170,.16) 0%, transparent 62%);
        pointer-events:none;
    }
    /* blue glow — bottom left */
    .lp-hero::after {
        content:"";
        position:absolute; bottom:-120px; left:5%;
        width:480px; height:480px;
        background:radial-gradient(circle, rgba(96,165,250,.08) 0%, transparent 62%);
        pointer-events:none;
    }

    .lp-hero-left {
        display:flex; flex-direction:column; justify-content:center;
        position:relative; z-index:1;
    }

    /* pulsing dot before badge */
    .lp-badge {
        display:inline-flex; align-items:center; gap:8px;
        color:var(--teal);
        font-size:.63rem; font-weight:800; letter-spacing:.2em; text-transform:uppercase;
        background:rgba(16,212,170,.08);
        border:1px solid rgba(16,212,170,.28);
        border-radius:100px;
        padding:6px 16px 6px 12px;
        margin-bottom:24px;
        width:fit-content;
        box-shadow:0 0 12px rgba(16,212,170,.12);
    }
    .lp-badge::before {
        content:"";
        display:inline-block; width:6px; height:6px;
        background:var(--teal);
        border-radius:50%;
        box-shadow:0 0 6px var(--teal);
        animation:lp-pulse 2s ease-in-out infinite;
    }
    @keyframes lp-pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:.4; transform:scale(.75); }
    }

    .lp-title {
        color:#fff !important;
        font-size:clamp(2.6rem,4.8vw,4.4rem);
        font-weight:900;
        line-height:1.0;
        letter-spacing:-.03em;
        margin:0 0 20px 0;
    }
    .lp-title-accent {
        color:var(--teal);
        text-shadow:0 0 40px rgba(16,212,170,.35);
    }

    /* thin teal underline accent */
    .lp-title-accent::after {
        content:"";
        display:block;
        height:3px;
        width:100%;
        background:linear-gradient(90deg, var(--teal) 0%, transparent 80%);
        border-radius:2px;
        margin-top:4px;
        opacity:.7;
    }

    .lp-sub {
        color:var(--muted);
        font-size:.98rem;
        line-height:1.75;
        max-width:480px;
        margin:0 0 36px 0;
    }

    /* stats row */
    .lp-stats {
        display:flex; align-items:center; gap:0;
        background:rgba(255,255,255,.03);
        border:1px solid var(--border);
        border-radius:12px;
        padding:16px 24px;
        width:fit-content;
        backdrop-filter:blur(4px);
    }
    .lp-stat { display:flex; flex-direction:column; gap:4px; padding:0 28px 0 0; }
    .lp-stat-n {
        color:#fff;
        font-size:2.2rem; font-weight:900; line-height:1;
        letter-spacing:-.02em;
    }
    .lp-stat-l {
        color:var(--faint);
        font-size:.58rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase;
    }
    .lp-stat-div {
        width:1px; height:40px;
        background:linear-gradient(180deg, transparent, var(--border2), transparent);
        margin:0 28px 0 0; flex-shrink:0;
    }

    /* right side */
    .lp-hero-right {
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:22px; position:relative; z-index:1;
    }

    .lp-pitch {
        width:100%; max-width:400px;
        background:linear-gradient(160deg, rgba(16,212,170,.06) 0%, rgba(16,212,170,.02) 100%);
        border:1px solid rgba(16,212,170,.22);
        border-radius:16px;
        padding:20px 20px 14px;
        display:flex; flex-direction:column; align-items:center; gap:12px;
        box-shadow:0 0 40px rgba(16,212,170,.08), inset 0 1px 0 rgba(255,255,255,.04);
    }
    .lp-pitch-label {
        color:var(--teal);
        font-size:.58rem; font-weight:800; letter-spacing:.22em; text-transform:uppercase;
        opacity:.8;
    }

    .lp-feature-pills {
        display:flex; flex-wrap:wrap; justify-content:center; gap:8px; max-width:400px;
    }
    .lp-pill {
        background:rgba(255,255,255,.04);
        border:1px solid var(--border);
        color:var(--muted);
        font-size:.67rem; font-weight:600;
        padding:5px 13px;
        border-radius:100px;
        letter-spacing:.03em;
        transition:border-color .2s, color .2s;
    }
    .lp-pill:hover { border-color:var(--border2); color:var(--ink); }
    .lp-pill.teal {
        background:rgba(16,212,170,.1);
        border-color:rgba(16,212,170,.35);
        color:var(--teal);
        box-shadow:0 0 10px rgba(16,212,170,.1);
    }

    /* ── SECTION DIVIDER ──────────────────────────────────── */
    .lp-section-header {
        display:flex; align-items:center; gap:20px;
        padding:32px 64px 24px;
        background:var(--bg);
    }
    .lp-section-line {
        flex:1; height:1px;
        background:linear-gradient(90deg, transparent, var(--border), transparent);
    }
    .lp-section-text {
        color:var(--faint);
        font-size:.6rem; font-weight:800; letter-spacing:.24em; text-transform:uppercase;
        white-space:nowrap;
        padding:0 4px;
    }

    /* ── WORKSPACE CARDS AREA ─────────────────────────────── */
    /* Override Streamlit column container with padding */
    div[data-testid="stHorizontalBlock"] {
        padding:0 52px 8px !important;
        gap:16px !important;
    }
    /* Bigger, polished cards */
    .dash-card {
        border:1px solid var(--border) !important;
        background:linear-gradient(160deg, var(--surface) 0%, #1a2236 100%) !important;
        border-radius:14px !important;
        padding:32px 20px 26px !important;
        text-align:center !important;
        box-shadow:0 2px 16px rgba(0,0,0,.28) !important;
        transition:border-color .22s, box-shadow .22s, transform .18s !important;
        min-height:230px !important;
        display:flex !important;
        flex-direction:column !important;
        align-items:center !important;
        justify-content:center !important;
        gap:10px !important;
        position:relative !important;
        overflow:hidden !important;
    }
    .dash-card::before {
        content:"";
        position:absolute; top:0; left:0; right:0; height:1px;
        background:linear-gradient(90deg, transparent, rgba(255,255,255,.07), transparent);
    }
    .dash-card:hover {
        border-color:var(--teal) !important;
        box-shadow:0 0 0 1px var(--teal), 0 8px 32px rgba(16,212,170,.12) !important;
        transform:translateY(-3px) !important;
    }
    .dash-card--primary {
        border-color:rgba(16,212,170,.4) !important;
        background:linear-gradient(160deg, rgba(16,212,170,.1) 0%, rgba(16,212,170,.04) 40%, #1a2236 100%) !important;
        box-shadow:0 0 0 1px rgba(16,212,170,.2), 0 4px 24px rgba(16,212,170,.1) !important;
    }
    .dash-card--primary:hover {
        box-shadow:0 0 0 2px var(--teal), 0 8px 36px rgba(16,212,170,.2) !important;
    }
    .dash-card-icon {
        font-size:3rem !important;
        line-height:1 !important;
        filter:drop-shadow(0 2px 10px rgba(0,0,0,.5)) !important;
        margin-bottom:2px !important;
    }
    .dash-card-title {
        color:var(--ink) !important;
        font-size:.96rem !important;
        font-weight:800 !important;
        line-height:1.2 !important;
        letter-spacing:-.01em !important;
    }
    .dash-card-desc {
        color:var(--faint) !important;
        font-size:.7rem !important;
        line-height:1.5 !important;
        max-width:150px !important;
    }

    /* ── BOTTOM FEATURE STRIP ─────────────────────────────── */
    .lp-strip {
        display:grid;
        grid-template-columns:repeat(4,minmax(0,1fr));
        border-top:1px solid var(--border);
        margin-top:40px;
        background:linear-gradient(180deg, var(--surface) 0%, #182032 100%);
    }
    .lp-strip-item {
        display:flex; align-items:flex-start; gap:16px;
        padding:28px 32px;
        border-right:1px solid var(--border);
        transition:background .2s;
        position:relative;
    }
    .lp-strip-item::before {
        content:"";
        position:absolute; top:0; left:0; right:0; height:2px;
        background:linear-gradient(90deg, var(--teal), transparent);
        opacity:0;
        transition:opacity .25s;
    }
    .lp-strip-item:hover { background:rgba(16,212,170,.04); }
    .lp-strip-item:hover::before { opacity:1; }
    .lp-strip-item:last-child { border-right:none; }
    .lp-strip-icon {
        font-size:1.7rem; line-height:1; flex-shrink:0; margin-top:1px;
        filter:drop-shadow(0 2px 6px rgba(0,0,0,.4));
    }
    .lp-strip-body { display:flex; flex-direction:column; gap:6px; }
    .lp-strip-title {
        color:var(--ink);
        font-size:.86rem; font-weight:800; line-height:1.15;
    }
    .lp-strip-desc {
        color:var(--faint);
        font-size:.71rem; line-height:1.55;
    }

    /* ── RESPONSIVE ───────────────────────────────────────── */
    @media (max-width:960px) {
        .lp-hero { grid-template-columns:1fr; padding:36px 28px; min-height:auto; gap:28px; }
        .lp-hero-right { display:none; }
        .lp-section-header { padding:24px 28px 18px; }
        div[data-testid="stHorizontalBlock"] { padding:0 28px 8px !important; }
        .lp-strip { grid-template-columns:1fr 1fr; }
        .lp-strip-item { border-bottom:1px solid var(--border); }
    }
    @media (max-width:600px) {
        .lp-title { font-size:2.2rem; }
        .lp-strip { grid-template-columns:1fr; }
        div[data-testid="stHorizontalBlock"] { padding:0 16px 8px !important; }
    }
</style>
    """,
    unsafe_allow_html=True,
)


# Load persistent shortlist from disk on first run
if "shortlist_data" not in st.session_state:
    st.session_state["shortlist_data"] = _load_shortlist_file()
    st.session_state["shortlist_players"] = list(st.session_state["shortlist_data"].keys())

if "show_scouting_workspace" not in st.session_state:
    st.session_state["show_scouting_workspace"] = False
if "active_workspace" not in st.session_state:
    st.session_state["active_workspace"] = "Recruitment"

if not st.session_state["show_scouting_workspace"]:
    _ldata = load_default_data()
    _n_players = len(_ldata)
    _n_leagues = _ldata["BundleLabel"].nunique() if "BundleLabel" in _ldata.columns else 0
    _n_positions = _ldata["PositionGroup"].nunique() if "PositionGroup" in _ldata.columns else 0
    _n_high = int(_ldata.get("QualityTier", pd.Series()).isin(["High quality", "Elite"]).sum()) if "QualityTier" in _ldata.columns else 0

    # ── Landing-page full-bleed overrides ────────────────────────────────────
    # Hide sidebar, remove container padding so the page fills the viewport.
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]  { display: none !important; }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    div[data-testid="stToolbar"]   { display: none !important; }
    /* Workspace card columns — add horizontal breathing room */
    div[data-testid="stHorizontalBlock"] {
        padding: 0 48px !important;
        gap: 14px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero (no blank lines — blank lines break CommonMark HTML blocks) ─────
    _pitch_svg = '<svg viewBox="0 0 400 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;opacity:.55;"><rect x="2" y="2" width="396" height="256" rx="4" fill="none" stroke="#10d4aa" stroke-width="1.5"/><line x1="200" y1="2" x2="200" y2="258" stroke="#10d4aa" stroke-width="1"/><circle cx="200" cy="130" r="36" fill="none" stroke="#10d4aa" stroke-width="1"/><circle cx="200" cy="130" r="3" fill="#10d4aa"/><rect x="2" y="75" width="66" height="110" fill="none" stroke="#10d4aa" stroke-width="1"/><rect x="2" y="98" width="22" height="64" fill="none" stroke="#10d4aa" stroke-width="1"/><circle cx="56" cy="130" r="2" fill="#10d4aa"/><rect x="332" y="75" width="66" height="110" fill="none" stroke="#10d4aa" stroke-width="1"/><rect x="376" y="98" width="22" height="64" fill="none" stroke="#10d4aa" stroke-width="1"/><circle cx="344" cy="130" r="2" fill="#10d4aa"/><path d="M2 14 A12 12 0 0 1 14 2" fill="none" stroke="#10d4aa" stroke-width="1"/><path d="M386 2 A12 12 0 0 1 398 14" fill="none" stroke="#10d4aa" stroke-width="1"/><path d="M398 246 A12 12 0 0 1 386 258" fill="none" stroke="#10d4aa" stroke-width="1"/><path d="M14 258 A12 12 0 0 1 2 246" fill="none" stroke="#10d4aa" stroke-width="1"/></svg>'
    st.markdown(
        f'<div class="lp-wrapper">'
        f'<div class="lp-hero">'
        f'<div class="lp-hero-left">'
        f'<div class="lp-badge">&#x26BD; FCHK Model V3 &middot; Hradeck Scouting</div>'
        f'<h1 class="lp-title">Player Quality<br><span class="lp-title-accent">Command Room</span></h1>'
        f'<p class="lp-sub">Professional-grade football scouting and recruitment intelligence. Every decision backed by data &mdash; from Czech leagues to Europe.</p>'
        f'<div class="lp-stats">'
        f'<div class="lp-stat"><span class="lp-stat-n">{_n_players:,}</span><span class="lp-stat-l">Players</span></div>'
        f'<div class="lp-stat-div"></div>'
        f'<div class="lp-stat"><span class="lp-stat-n">{_n_leagues}</span><span class="lp-stat-l">Leagues</span></div>'
        f'<div class="lp-stat-div"></div>'
        f'<div class="lp-stat"><span class="lp-stat-n">{_n_positions}</span><span class="lp-stat-l">Positions</span></div>'
        f'<div class="lp-stat-div"></div>'
        f'<div class="lp-stat"><span class="lp-stat-n">12</span><span class="lp-stat-l">Score Dims</span></div>'
        f'</div>'
        f'</div>'
        f'<div class="lp-hero-right">'
        f'<div class="lp-pitch">{_pitch_svg}<div class="lp-pitch-label">FCHK Intelligence Engine</div></div>'
        f'<div class="lp-feature-pills">'
        f'<span class="lp-pill">&#x1F4CA; Quality scoring</span>'
        f'<span class="lp-pill">&#x1F3AF; Position fit</span>'
        f'<span class="lp-pill">&#x1F30D; European scouting</span>'
        f'<span class="lp-pill">&#x1F9E4; GK analysis</span>'
        f'<span class="lp-pill teal">&#x26A1; Real-time filters</span>'
        f'<span class="lp-pill">&#x1F4E5; PDF &amp; CSV export</span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div class="lp-section-header">'
        f'<span class="lp-section-line"></span>'
        f'<span class="lp-section-text">Choose your workspace</span>'
        f'<span class="lp-section-line"></span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Dashboard icon cards ──────────────────────────────────────────────────
    _workspace_meta = {
        "Recruitment": ("📊", "Recruitment Board",  "Quality rankings, player profiles, radar charts, transfer value & signing cases"),
        "Scouting":    ("🔍", "Scouting",           "Raw Wyscout data browser — search & filter imported scouting files"),
        "Goalkeepers": ("🧤", "Goalkeepers",        "GK-exclusive boards, shot-stopping metrics & reliability"),
        "Team":        ("🏟️", "Team Intelligence",  "Squad gaps, Czech market benchmarks & watchlist"),
        "Model":       ("🔬", "Model & Data",       "Smart club closeness, data coverage & confidence bands"),
    }
    landing_cols = st.columns(len(WORKSPACES), gap="medium")
    for idx, section in enumerate(WORKSPACES):
        icon, title, desc = _workspace_meta.get(section, ("⚽", section, ""))
        is_primary = section == "Recruitment"
        with landing_cols[idx]:
            _card_cls = "dash-card dash-card--primary" if is_primary else "dash-card"
            st.markdown(
                f'<div class="{_card_cls}"><div class="dash-card-icon">{icon}</div><div class="dash-card-title">{title}</div><div class="dash-card-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )
            st.button(
                f"{'→  ' if is_primary else ''}Open {title}",
                key=f"landing_{section}",
                type="primary" if is_primary else "secondary",
                width="stretch",
                on_click=set_workspace,
                args=(section,),
            )

    # ── Bottom feature strip ──────────────────────────────────────────────────
    st.markdown(
        '<div class="lp-strip">'
        '<div class="lp-strip-item"><div class="lp-strip-icon">&#x26A1;</div><div class="lp-strip-body"><div class="lp-strip-title">Instant quality lenses</div><div class="lp-strip-desc">Switch between Balanced, Technique, Ready and Upside presets &mdash; or set your own weights.</div></div></div>'
        '<div class="lp-strip-item"><div class="lp-strip-icon">&#x1F3AF;</div><div class="lp-strip-body"><div class="lp-strip-title">Hradec-fit algorithm</div><div class="lp-strip-desc">Every external player scored on exactly how well they&rsquo;d address your squad&rsquo;s positional gaps.</div></div></div>'
        '<div class="lp-strip-item"><div class="lp-strip-icon">&#x1F30D;</div><div class="lp-strip-body"><div class="lp-strip-title">European market map</div><div class="lp-strip-desc">Interactive bubble map of scout-priority by country &mdash; see where to send your scouts next.</div></div></div>'
        '<div class="lp-strip-item"><div class="lp-strip-icon">&#x1F4E5;</div><div class="lp-strip-body"><div class="lp-strip-title">Board-ready exports</div><div class="lp-strip-desc">One-click PDF scouting reports and CSV exports for every filtered view and shortlist.</div></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("landing_notice"):
        st.info(st.session_state["landing_notice"])
    st.stop()

render_workspace_nav("main")

active_workspace = st.session_state.get("active_workspace", "Recruitment")
data = load_default_data()
model_metadata = load_model_metadata()
_rec_file = _model_file("recruitment")
_data_updated = "unknown"
if _rec_file.exists():
    from datetime import datetime as _dt
    _data_updated = _dt.fromtimestamp(_rec_file.stat().st_mtime).strftime("%-d %b %Y")
st.markdown(
    f'<div class="intel-strip">'
    f'<div class="intel-strip-title">{escape(active_workspace)} intelligence</div>'
    f'<div class="intel-strip-meta">{len(data):,} players · {data["BundleLabel"].nunique()} leagues · model v3'
    f' · <span style="color:var(--teal);">data {_data_updated}</span></div>'
    f'</div>',
    unsafe_allow_html=True,
)
if active_workspace != "Recruitment":
    if active_workspace == "Scouting":
        render_scouting_workspace()
    elif active_workspace == "Model":
        render_model_workspace(data, model_metadata)
    elif active_workspace == "Goalkeepers":
        render_goalkeepers_workspace(data)
    elif active_workspace == "Team":
        render_team_workspace(data)
    else:
        st.markdown(f"<div class='workspace-label'>{active_workspace} workspace</div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-card">'
            f'<div class="metric-label">Coming next</div>'
            f'<div class="metric-value" style="font-size:1.45rem;">{active_workspace}</div>'
            f'<div class="metric-caption">This workspace is being built. Use the navigation above to switch between workspaces.</div>'
            f'</div>',
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
            <div class="sidebar-brand-title">⚽ FCHK Recruitment IQ</div>
            <div class="sidebar-brand-meta">{len(outfield_data):,} outfield · GK separate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='menu-caption'>Quality &amp; fit model — adjust lens weights to rank players for different recruitment profiles.</div>", unsafe_allow_html=True)
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
    if "CountryLabel" in df.columns:
        _country_opts = sorted(df["CountryLabel"].dropna().astype(str).replace("", "Unknown").unique())
        countries = st.multiselect("Country", _country_opts, default=_country_opts, key="countries_filter")
    else:
        countries = []
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
if countries and "CountryLabel" in df.columns:
    mask &= df["CountryLabel"].astype(str).replace("", "Unknown").isin(countries)
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

# ── Quick-mode filter chips ─────────────────────────────────────────────────
# Use on_click= so set_quick_mode runs BEFORE widgets are instantiated (avoids
# "cannot modify key after widget is rendered" StreamlitAPIException).
st.markdown("<div class='quick-chips'><span class='quick-chip-label'>Quick filter:</span></div>", unsafe_allow_html=True)
qm_cols = st.columns([1, 1, 1, 1, 2])
_qm = st.session_state.get("quick_mode", "Full board")
with qm_cols[0]:
    st.button("⚡ Full board",  type="primary" if _qm == "Full board"      else "secondary", width="stretch", on_click=set_quick_mode, args=("Full board",))
with qm_cols[1]:
    st.button("🌱 U23 quality", type="primary" if _qm == "U23 quality"     else "secondary", width="stretch", on_click=set_quick_mode, args=("U23 quality",))
with qm_cols[2]:
    st.button("🏆 Elite only",  type="primary" if _qm == "Elite quality"   else "secondary", width="stretch", on_click=set_quick_mode, args=("Elite quality",))
with qm_cols[3]:
    st.button("🛡 Reliable",    type="primary" if _qm == "Reliable quality" else "secondary", width="stretch", on_click=set_quick_mode, args=("Reliable quality",))
with qm_cols[4]:
    _shortlist_count = len(st.session_state.get("shortlist_players", []))
    st.markdown(
        f"<div style='padding:6px 0;color:var(--faint);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;'>"
        f"Shortlist: <span style='color:var(--teal)'>{_shortlist_count} player{'s' if _shortlist_count != 1 else ''}</span> · "
        f"Showing <span style='color:var(--ink)'>{len(filtered):,}</span> of {len(df):,}"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── KPI Cockpit ────────────────────────────────────────────────────────────────
_role_stats = (
    filtered.groupby("PositionGroup")
    .agg(Count=("PlayerName", "count"), MedQ=("QualityScore", "median"))
    .round(1).reset_index().sort_values("MedQ", ascending=False)
)
_cockpit_pct = f"{len(filtered)*100//max(len(df),1)}%"
_hq_pct = f"{high_quality_count*100//max(len(filtered),1)}%"
st.markdown(
    '<div class="scouting-cockpit"><div class="cockpit-grid">'
    f'<div class="cockpit-tile"><div class="cockpit-label">Filtered players</div>'
    f'<div class="cockpit-value">{len(filtered):,}</div>'
    f'<div class="cockpit-note">{_cockpit_pct} of {len(df):,} in model</div></div>'
    f'<div class="cockpit-tile"><div class="cockpit-label">Elite / High quality</div>'
    f'<div class="cockpit-value">{high_quality_count}</div>'
    f'<div class="cockpit-note">{_hq_pct} of filtered set</div></div>'
    f'<div class="cockpit-tile"><div class="cockpit-label">Median quality</div>'
    f'<div class="cockpit-value">{median_quality}</div>'
    f'<div class="cockpit-note">Median impact: {median_impact}</div></div>'
    f'<div class="cockpit-tile"><div class="cockpit-label">No. 1 ranked</div>'
    f'<div class="cockpit-value" style="font-size:.95rem;line-height:1.25;">{escape(leader_name)}</div>'
    f'<div class="cockpit-note">Quality {leader_score}</div></div>'
    f'<div class="cockpit-tile"><div class="cockpit-label">Strongest role</div>'
    f'<div class="cockpit-value">{escape(str(best_role))}</div>'
    f'<div class="cockpit-note">Highest median quality</div></div>'
    '</div></div>',
    unsafe_allow_html=True,
)
# Role rail — one tile per position showing median quality + player count
_rr_html = '<div class="role-rail">'
for _, _rr in _role_stats.iterrows():
    _rr_html += (
        f'<div class="role-cell">'
        f'<div class="role-cell-role">{escape(str(_rr["PositionGroup"]))}</div>'
        f'<div class="role-cell-score">{_rr["MedQ"]:.0f}</div>'
        f'<div class="role-cell-count">{int(_rr["Count"])} players</div>'
        f'</div>'
    )
_rr_html += '</div>'
st.markdown(_rr_html, unsafe_allow_html=True)

# Clickable position focus — filters entire board to one role
_pos_options = ["All positions"] + sorted(filtered["PositionGroup"].dropna().astype(str).unique().tolist())
_pos_focus = st.segmented_control("Position focus", _pos_options, default="All positions", key="pos_focus_rail")
if _pos_focus and _pos_focus != "All positions":
    filtered = filtered.loc[filtered["PositionGroup"].eq(_pos_focus)].copy()

quality_tab, player_tab, compare_tab, hradec_tab, intel_tab, case_tab, export_tab = st.tabs(
    ["📊 Quality Board", "🔬 Player Lab", "⚖️ Compare", "🎯 Hradec Targets", "🌍 League Intel", "💼 Case Analysis", "📥 Export"]
)

with quality_tab:
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
        "PerformanceReliabilityScore",
        "Readiness",
        "RiskBand",
        "Archetype",
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
            "PerformanceReliabilityScore": "Reliability",
            "RiskBand": "Risk",
            "QualityDrivers": "Drivers",
        }
    )
    # Tier distribution pills
    _tier_order = ["Elite", "High quality", "Standard", "Developing"]
    _tier_cls   = {"Elite": "teal", "High quality": "teal", "Standard": "", "Developing": ""}
    _tcounts    = filtered["QualityTier"].value_counts().to_dict() if "QualityTier" in filtered.columns else {}
    _empty_str  = ""
    _tier_pills = "".join(
        f'<span class="pill {_tier_cls.get(t, _empty_str)}">{t}: {_tcounts.get(t,0)}</span>'
        for t in _tier_order if _tcounts.get(t, 0) > 0
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
        f'<span style="color:var(--faint);font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;">Tiers:</span>'
        f'<div class="pill-row" style="margin:0;">{_tier_pills}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        quality_board.round(2),
        use_container_width=True,
        hide_index=True,
        height=600,
        column_config={
            "Quality":     st.column_config.ProgressColumn("Quality",     min_value=0, max_value=100, format="%.1f"),
            "Role Fit":    st.column_config.ProgressColumn("Role Fit",    min_value=0, max_value=100, format="%.1f"),
            "Impact":      st.column_config.ProgressColumn("Impact",      min_value=0, max_value=100, format="%.1f"),
            "Decision":    st.column_config.ProgressColumn("Decision",    min_value=0, max_value=100, format="%.1f"),
            "Reliability": st.column_config.ProgressColumn("Reliability", min_value=0, max_value=100, format="%.1f"),
            "Age":         st.column_config.NumberColumn("Age", format="%.1f"),
            "Minutes":     st.column_config.NumberColumn("Minutes", format="%d"),
        },
    )

with player_tab:
    st.markdown("<div class='workspace-label'>🔬 Player Lab — deep dive profile & radar</div>", unsafe_allow_html=True)
    player_options = filtered.assign(_label=filtered["PlayerName"] + " | " + filtered["TeamName"] + " | " + filtered["PositionGroup"]).sort_values("QualityScore", ascending=False)
    selected_label = st.selectbox("Select player", player_options["_label"].tolist())
    player = player_options.loc[player_options["_label"].eq(selected_label)].iloc[0]
    _risk_cls = {"Low": "teal", "Moderate": "amber", "Elevated": "amber", "High": "red"}.get(str(player.get("RiskBand", "")), "")
    st.markdown(
        f"""<div class="profile-card">
            <div class="profile-name">{escape(str(player['PlayerName']))}</div>
            <div class="profile-meta">{escape(str(player['TeamName']))} · {escape(str(player['BundleLabel']))} · {escape(str(player['PositionGroup']))} · {player['AgeYears']:.1f} yrs · {int(player['MinutesPlayed']):,} min</div>
            <div class="pill-row">
                <span class="pill teal">Quality {player['QualityScore']:.1f}</span>
                <span class="pill">{escape(str(player['QualityTier']))}</span>
                <span class="pill">{escape(str(player['Archetype']))}</span>
                <span class="pill {_risk_cls}">{escape(str(player['RiskBand']))} risk</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(5)
    metric_cols[0].metric("Quality", f"{player['QualityScore']:.1f}")
    metric_cols[1].metric("Role Fit", f"{player['RoleFitScore']:.1f}")
    metric_cols[2].metric("Impact", f"{player['ProfileScore']:.1f}")
    metric_cols[3].metric("Decision", f"{player['DecisionScore']:.1f}")
    metric_cols[4].metric("Position Pctl", f"{percentile_rank(df.loc[df['PositionGroup'].eq(player['PositionGroup']), 'QualityScore'], player['QualityScore']):.0f}")
    _pnote = profile_note(player)
    _pstrengths = player_strengths(player)
    st.markdown(
        f'<div class="note-box">'
        f'<strong style="color:var(--teal);">Profile:</strong> {escape(_pnote)}<br>'
        f'<strong style="color:var(--teal);">Strengths:</strong> {escape(_pstrengths)}<br>'
        f'<strong style="color:var(--muted);">Drivers:</strong> {escape(str(player["QualityDrivers"]))} &nbsp;·&nbsp; '
        f'Reliability: {escape(str(player["Readiness"]))} &nbsp;·&nbsp; Risk: {escape(str(player["RiskBand"]))}'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Style profile — surface model style columns when available
    _primary_style = str(player.get("PrimaryPlayerStyle", "") or "").strip()
    _secondary_style = str(player.get("SecondaryPlayerStyle", "") or "").strip()
    _style_summary = str(player.get("PlayerStyleSummary", "") or "").strip()
    _why_club = str(player.get("WhyThisClubStyle", "") or "").strip()
    _smart_top3 = str(player.get("SmartClubTop3", "") or "").strip()
    _closeness_tier = str(player.get("SmartClubClosenessTier", "") or "").strip()
    _has_style_data = any(v and v not in ("nan", "None") for v in [_primary_style, _style_summary, _why_club, _smart_top3])
    if _has_style_data:
        _style_parts = []
        if _primary_style and _primary_style not in ("nan", "None"):
            _sty = escape(_primary_style)
            if _secondary_style and _secondary_style not in ("nan", "None"):
                _sty += f' <span style="color:var(--muted);">· {escape(_secondary_style)}</span>'
            _style_parts.append(f'<strong style="color:var(--teal);">Playing style:</strong> {_sty}')
        if _style_summary and _style_summary not in ("nan", "None"):
            _style_parts.append(f'<span style="color:var(--muted);font-size:.8rem;">{escape(_style_summary)}</span>')
        if _why_club and _why_club not in ("nan", "None"):
            _style_parts.append(f'<strong style="color:var(--teal);">Hradec fit:</strong> {escape(_why_club)}')
        if _smart_top3 and _smart_top3 not in ("nan", "None"):
            _tier_badge = f' <span class="pill">{escape(_closeness_tier)}</span>' if _closeness_tier and _closeness_tier not in ("nan","None") else ""
            _style_parts.append(f'<strong style="color:var(--muted);">Style clubs:</strong> {escape(_smart_top3)}{_tier_badge}')
        st.markdown(
            '<div class="note-box" style="margin-top:6px;border-left-color:rgba(16,212,170,.5);">'
            + '<br>'.join(_style_parts)
            + '</div>',
            unsafe_allow_html=True,
        )
    _sl_add_cols = st.columns([1, 1, 2])
    with _sl_add_cols[0]:
        _sl_priority = st.selectbox("Priority", ["Watch", "Hot", "Observed"], key=f"sl_pri_{selected_label[:20]}")
    with _sl_add_cols[1]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("➕ Add to shortlist", type="primary", width="stretch"):
            _sl_notes_val = st.session_state.get(f"sl_notes_{selected_label[:20]}", "")
            add_to_shortlist(player["PlayerName"], priority=_sl_priority, notes=_sl_notes_val)
            st.success(f"Added {player['PlayerName']} to shortlist as {_sl_priority}")
    with _sl_add_cols[2]:
        st.text_input("Scout note (optional)", key=f"sl_notes_{selected_label[:20]}", placeholder="E.g. Watched vs Sparta, strong in press…")
    _card_notes = st.session_state.get(f"sl_notes_{selected_label[:20]}", "")
    st.download_button(
        "📄 Download player card PDF",
        data=build_player_card_pdf(player, df.loc[df["PositionGroup"].eq(player["PositionGroup"])], scout_notes=_card_notes),
        file_name=f"fchk_player_{player['PlayerName'].replace(' ', '_').lower()}.pdf",
        mime="application/pdf",
        width="stretch",
    )
    lab_left, lab_right = st.columns([1, 1])
    with lab_left:
        st.pyplot(render_player_pizza(df.loc[df["PositionGroup"].eq(player["PositionGroup"])], player), clear_figure=True)
    with lab_right:
        st.markdown("<div class='workspace-label' style='font-size:.58rem;margin-bottom:8px;'>Top per-90 metrics</div>", unsafe_allow_html=True)
        per90_cols = [c for c in filtered.columns if c.endswith("_per90") or c.startswith("Imp_")]
        _per90_labels = {c: c.replace("_per90","").replace("Imp_","").replace("_"," ").title() for c in per90_cols}
        per90 = (
            pd.DataFrame({"Metric": [_per90_labels[c] for c in per90_cols], "Value": [float(player[c]) for c in per90_cols], "Pctile": [percentile_rank(filtered[c].dropna(), float(player[c])) for c in per90_cols]})
            .sort_values("Value", ascending=False).head(20).round(2)
        )
        st.dataframe(
            per90,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn("Value", format="%.2f"),
                "Pctile": st.column_config.ProgressColumn("Pctile %", min_value=0, max_value=100, format="%.0f"),
            },
        )
    # AI Scout Report
    _api_key_set = bool(os.getenv("ANTHROPIC_API_KEY", ""))
    _report_key = f"scout_report_{player['PlayerName']}"
    with st.expander("🤖 AI Scout Report" + (" (set ANTHROPIC_API_KEY to enable)" if not _api_key_set else ""), expanded=False):
        if not _api_key_set:
            st.info("Set the ANTHROPIC_API_KEY environment variable to generate AI scouting reports.")
        else:
            if st.button("Generate scout report", key=f"gen_report_{selected_label[:20]}", type="primary"):
                with st.spinner("Writing report…"):
                    _report = generate_scout_report(player, df)
                    st.session_state[_report_key] = _report
            if _report_key in st.session_state:
                st.markdown(
                    f'<div class="note-box" style="font-size:.82rem;line-height:1.7;">{escape(st.session_state[_report_key])}</div>',
                    unsafe_allow_html=True,
                )

    # Age-quality development curve
    _age_pos_df = df.loc[df["PositionGroup"].eq(player["PositionGroup"]) & df["QualityScore"].notna() & df["AgeYears"].notna()][["PlayerName","AgeYears","QualityScore"]].copy()
    if len(_age_pos_df) > 10:
        _age_base = (
            alt.Chart(_age_pos_df)
            .mark_circle(opacity=0.25, size=28, color="#2e4a6e")
            .encode(
                x=alt.X("AgeYears:Q", title="Age", scale=alt.Scale(domain=[16, 38]), axis=alt.Axis(labelColor="#8fa3b1", gridColor="#1e2d3d", titleColor="#8fa3b1")),
                y=alt.Y("QualityScore:Q", title="Quality", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelColor="#8fa3b1", gridColor="#1e2d3d", titleColor="#8fa3b1")),
                tooltip=["PlayerName:N", alt.Tooltip("AgeYears:Q", format=".1f", title="Age"), alt.Tooltip("QualityScore:Q", format=".1f", title="Quality")],
            )
        )
        _age_trend = (
            alt.Chart(_age_pos_df)
            .transform_regression("AgeYears", "QualityScore", method="poly", order=3)
            .mark_line(color="#f5a623", opacity=0.55, strokeWidth=2, strokeDash=[5, 3])
            .encode(x="AgeYears:Q", y="QualityScore:Q")
        )
        _age_highlight = (
            alt.Chart(pd.DataFrame([{"AgeYears": float(player["AgeYears"]), "QualityScore": float(player["QualityScore"]), "Name": str(player["PlayerName"])}]))
            .mark_circle(opacity=1, size=130, color="#10d4aa")
            .encode(
                x="AgeYears:Q", y="QualityScore:Q",
                tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("AgeYears:Q", format=".1f", title="Age"), alt.Tooltip("QualityScore:Q", format=".1f", title="Quality")],
            )
        )
        _age_label = (
            alt.Chart(pd.DataFrame([{"AgeYears": float(player["AgeYears"]), "QualityScore": float(player["QualityScore"]), "Name": str(player["PlayerName"])}]))
            .mark_text(color="#10d4aa", fontSize=11, dx=10, dy=-8, align="left")
            .encode(x="AgeYears:Q", y="QualityScore:Q", text="Name:N")
        )
        _age_chart = (
            (_age_base + _age_trend + _age_highlight + _age_label)
            .properties(height=260, title=alt.TitleParams(f"{player['PositionGroup']} age–quality curve (all leagues)", color="#8fa3b1", fontSize=11))
            .configure_view(fill="#0f1623", stroke=None)
            .configure(background="#080c14")
        )
        st.altair_chart(_age_chart, use_container_width=True)

    comparable = similar_players(df, player, same_position=True, n=12)
    st.markdown("<div class='workspace-label' style='font-size:.58rem;margin:14px 0 8px;'>Closest quality profiles</div>", unsafe_allow_html=True)
    similar_cols = ["PlayerName", "TeamName", "PositionGroup", "AgeYears", "MinutesPlayed", "SimilarityScore", "QualityScore", "RoleFitScore", "ProfileScore", "Archetype"]
    st.dataframe(
        comparable[[c for c in similar_cols if c in comparable.columns]].rename(columns={"PlayerName":"Player","TeamName":"Team","PositionGroup":"Role","AgeYears":"Age","MinutesPlayed":"Minutes","SimilarityScore":"Similarity","QualityScore":"Quality","RoleFitScore":"Role Fit","ProfileScore":"Impact"}).round(2),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quality":   st.column_config.ProgressColumn("Quality",   min_value=0, max_value=100, format="%.1f"),
            "Role Fit":  st.column_config.ProgressColumn("Role Fit",  min_value=0, max_value=100, format="%.1f"),
            "Impact":    st.column_config.ProgressColumn("Impact",    min_value=0, max_value=100, format="%.1f"),
            "Similarity":st.column_config.ProgressColumn("Similarity",min_value=0, max_value=1,   format="%.2f"),
            "Age":       st.column_config.NumberColumn("Age", format="%.1f"),
            "Minutes":   st.column_config.NumberColumn("Minutes", format="%d"),
        },
    )

with hradec_tab:
    st.markdown("<div class='workspace-label'>🎯 Hradec Kralove — best external targets</div>", unsafe_allow_html=True)
    targets_df = hradec_recruitment_targets(df, top_n=60)
    if targets_df.empty:
        st.info("No external targets found — check that the model data includes players outside Hradec.")
    else:
        hm_cols = st.columns(4)
        priority_count = (targets_df["PositionNeed"].str.startswith("🔴")).sum()
        with hm_cols[0]:
            metric_card("Targets ranked", f"{len(targets_df)}", "by Hradec fit score")
        with hm_cols[1]:
            metric_card("High-need positions", f"{priority_count}", "🔴 roles to fill")
        with hm_cols[2]:
            top_t = targets_df.iloc[0]
            metric_card("Top target", str(top_t["PlayerName"]), f"{top_t['PositionGroup']} · {top_t['HradecTargetScore']:.1f}")
        with hm_cols[3]:
            metric_card("Median target score", f"{targets_df['HradecTargetScore'].median():.1f}", "vs 0–100 scale")

        st.markdown(
            "<div class='note-box'>Targets scored on: quality (38%) + smart-club style fit (16%) + transfer value (18%) + age/resale (12%) + reliability (16%), "
            "with bonus weight for positions where Hradec is thin or below Czech market median.</div>",
            unsafe_allow_html=True,
        )

        # Position filter chips
        pos_options = sorted(targets_df["PositionGroup"].dropna().astype(str).unique())
        sel_pos = st.multiselect("Filter by position", pos_options, default=pos_options, key="hradec_pos_filter")
        targets_view = targets_df.loc[targets_df["PositionGroup"].astype(str).isin(sel_pos)].copy()

        board_cols = [c for c in [
            "PlayerName", "TeamName", "PositionGroup", "BundleLabel", "AgeYears",
            "HradecTargetScore", "PositionNeed", "QualityScore", "ValueRecruitmentScore",
            "AgeResaleScore", "SmartClubScore", "PerformanceReliabilityScore",
            "RiskBand", "Archetype", "WhyThisClubStyle",
        ] if c in targets_view.columns]

        st.dataframe(
            targets_view[board_cols].rename(columns={
                "PlayerName": "Player", "TeamName": "Team", "PositionGroup": "Role",
                "BundleLabel": "League", "AgeYears": "Age",
                "HradecTargetScore": "Hradec Fit ▼", "PositionNeed": "Need",
                "QualityScore": "Quality", "ValueRecruitmentScore": "Value",
                "AgeResaleScore": "Resale", "SmartClubScore": "Style Fit",
                "PerformanceReliabilityScore": "Reliability", "RiskBand": "Risk",
                "WhyThisClubStyle": "Why Hradec",
            }).round(1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hradec Fit ▼": st.column_config.ProgressColumn("Hradec Fit ▼", min_value=0, max_value=130, format="%.1f"),
                "Quality":      st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.1f"),
                "Value":        st.column_config.ProgressColumn("Value",   min_value=0, max_value=100, format="%.1f"),
                "Resale":       st.column_config.ProgressColumn("Resale",  min_value=0, max_value=100, format="%.1f"),
                "Style Fit":    st.column_config.ProgressColumn("Style Fit", min_value=0, max_value=100, format="%.1f"),
                "Reliability":  st.column_config.ProgressColumn("Reliability", min_value=0, max_value=100, format="%.1f"),
                "Age":          st.column_config.NumberColumn("Age", format="%.1f"),
            },
        )

        # Position need chart
        _need_df = (
            targets_view.groupby("PositionGroup")
            .agg(Count=("PlayerName","count"), AvgFit=("HradecTargetScore","mean"), Need=("PositionNeed","first"))
            .reset_index().sort_values("AvgFit", ascending=False)
        )
        if not _need_df.empty:
            _need_chart = (
                alt.Chart(_need_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("PositionGroup:N", sort="-x", title=None,
                            axis=alt.Axis(labelColor="#8fa3b1", labelFontSize=12)),
                    x=alt.X("AvgFit:Q", title="Average Hradec Fit Score",
                            scale=alt.Scale(domain=[0,100]),
                            axis=alt.Axis(labelColor="#8fa3b1", titleColor="#8fa3b1", gridColor="#1e2d3d")),
                    color=alt.Color("AvgFit:Q",
                                    scale=alt.Scale(domain=[30,90], range=["#1e3a5f","#10d4aa"]),
                                    legend=None),
                    tooltip=[
                        alt.Tooltip("PositionGroup:N", title="Position"),
                        alt.Tooltip("Count:Q", title="Candidates"),
                        alt.Tooltip("AvgFit:Q", format=".1f", title="Avg Fit Score"),
                        alt.Tooltip("Need:N", title="Squad need"),
                    ],
                )
                .properties(height=240, title=alt.TitleParams("Avg Hradec fit by position", color="#8fa3b1", fontSize=12))
                .configure_view(fill="#0f1623", stroke=None)
                .configure(background="#080c14")
            )
            st.altair_chart(_need_chart, use_container_width=True)

        # Best XI
        xi_order = ["CB", "CB", "FB", "FB", "DM", "CM", "AM", "W", "W", "ST"]
        xi_used: set[str] = set()
        xi_rows = []
        for slot in xi_order:
            cands = targets_view.loc[targets_view["PositionGroup"].eq(slot) & ~targets_view["PlayerName"].isin(xi_used)]
            if cands.empty:
                continue
            pick = cands.iloc[0]
            xi_used.add(pick["PlayerName"])
            xi_rows.append({
                "Role": slot, "Player": pick["PlayerName"], "Team": pick["TeamName"],
                "Age": round(pick["AgeYears"], 1), "Hradec Fit": round(pick["HradecTargetScore"], 1),
                "Quality": round(pick["QualityScore"], 1), "Need": pick.get("PositionNeed", ""),
            })
        if xi_rows:
            st.subheader("Suggested Best XI from targets")
            st.dataframe(pd.DataFrame(xi_rows), use_container_width=True, hide_index=True)

with intel_tab:
    st.markdown("<div class='workspace-label'>🌍 League & market intelligence</div>", unsafe_allow_html=True)

    # ── European market map ─────────────────────────────────────
    st.subheader("European scout priority map")
    market_df = european_market_map_frame(df, metric="QualityScore")
    if not market_df.empty:
        map_col, map_info_col = st.columns([2.2, 1])
        with map_col:
            try:
                st.altair_chart(render_european_map(market_df), use_container_width=True)
            except Exception as e:
                st.warning(f"Map could not render: {e}")
        with map_info_col:
            st.markdown("<div class='note-box'>Bubble size = player count. Colour = scout priority score (quality + high-quality share + depth). Labels show countries with priority ≥ 55.</div>", unsafe_allow_html=True)
            top_markets = market_df.head(8)[["Country", "Players", "MedianScore", "GoScore", "Recommendation"]].rename(
                columns={"MedianScore": "Median Q", "GoScore": "Priority"}
            )
            st.dataframe(top_markets.round(1), use_container_width=True, hide_index=True)

    # ── League quality bar chart ────────────────────────────────
    st.subheader("Quality by league")
    league_summary = (
        filtered.groupby("BundleLabel")
        .agg(
            Players=("PlayerName", "count"),
            MedianQuality=("QualityScore", "median"),
            MedianRoleFit=("RoleFitScore", "median"),
            EliteCount=("QualityTier", lambda x: x.isin(["Elite", "High quality"]).sum()),
            MedianAge=("AgeYears", "median"),
        )
        .round(1)
        .reset_index()
        .sort_values("MedianQuality", ascending=False)
    )
    league_summary["ElitePct"] = (league_summary["EliteCount"] / league_summary["Players"] * 100).round(1)

    if not league_summary.empty:
        top30 = league_summary.head(30)
        bar_chart = (
            alt.Chart(top30)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                y=alt.Y("BundleLabel:N", sort="-x", title=None, axis=alt.Axis(labelColor="#8fa3b1", labelFontSize=11)),
                x=alt.X("MedianQuality:Q", title="Median quality score", scale=alt.Scale(domain=[0, 80]),
                         axis=alt.Axis(labelColor="#8fa3b1", titleColor="#8fa3b1", gridColor="#1e2d3d")),
                color=alt.Color("ElitePct:Q", scale=alt.Scale(domain=[0, 30], range=["#1e2d3d", "#00d4a8"]),
                                 legend=alt.Legend(title="Elite %", labelColor="#8fa3b1", titleColor="#8fa3b1")),
                tooltip=[
                    alt.Tooltip("BundleLabel:N", title="League"),
                    alt.Tooltip("Players:Q", title="Players"),
                    alt.Tooltip("MedianQuality:Q", format=".1f", title="Median quality"),
                    alt.Tooltip("ElitePct:Q", format=".1f", title="Elite %"),
                    alt.Tooltip("MedianAge:Q", format=".1f", title="Median age"),
                ],
            )
            .properties(height=max(320, len(top30) * 20))
            .configure_view(fill="#0f1623", stroke=None)
            .configure(background="#080c14")
        )
        st.altair_chart(bar_chart, use_container_width=True)

    # ── Full league table ───────────────────────────────────────
    with st.expander("Full league table", expanded=False):
        st.dataframe(
            league_summary.rename(columns={"BundleLabel": "League", "MedianQuality": "Median Q",
                                            "MedianRoleFit": "Median Fit", "EliteCount": "Elite",
                                            "ElitePct": "Elite %", "MedianAge": "Median Age"}).round(1),
            use_container_width=True, hide_index=True,
            column_config={
                "Median Q":   st.column_config.ProgressColumn("Median Q",   min_value=0, max_value=100, format="%.1f"),
                "Median Fit": st.column_config.ProgressColumn("Median Fit", min_value=0, max_value=100, format="%.1f"),
            },
        )

    # ── Score distribution ──────────────────────────────────────
    st.subheader("Score distribution (filtered players)")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        dist_metric = st.selectbox("Metric", ["QualityScore", "RoleFitScore", "ProfileScore",
                                               "DecisionScore", "PerformanceReliabilityScore",
                                               "CompositeRecruitmentScore"], key="dist_metric_intel")
        st.pyplot(render_score_distribution(filtered, dist_metric), clear_figure=True, use_container_width=True)
    with dist_col2:
        st.pyplot(render_position_boxplot(filtered, dist_metric), clear_figure=True, use_container_width=True)

    # ── League heatmap ──────────────────────────────────────────
    with st.expander("League depth heatmap", expanded=False):
        hm_metric = st.selectbox("Heatmap metric", ["QualityScore", "RoleFitScore", "CompositeRecruitmentScore"], key="hm_metric")
        st.pyplot(render_league_heatmap(filtered, hm_metric), clear_figure=True, use_container_width=True)

with compare_tab:
    st.markdown("<div class='workspace-label'>⚖️ Side-by-side comparison</div>", unsafe_allow_html=True)
    compare_options = filtered.assign(_label=filtered["PlayerName"] + " | " + filtered["TeamName"] + " | " + filtered["PositionGroup"]).sort_values("QualityScore", ascending=False)["_label"].tolist()
    selected_compare = st.multiselect("Select 2-4 players", compare_options, default=compare_options[: min(3, len(compare_options))], max_selections=4)
    compare_names = [label.split(" | ")[0] for label in selected_compare]
    compare_df = filtered.loc[filtered["PlayerName"].isin(compare_names)].sort_values("QualityScore", ascending=False)
    if len(compare_df) < 2:
        st.info("Pick at least two players to compare.")
    else:
        # Per-player profile cards
        _card_cols = st.columns(len(compare_df))
        for _ci, (_ci_idx, _cp) in enumerate(compare_df.iterrows()):
            with _card_cols[_ci]:
                _cp_risk_cls = {"Low":"teal","Moderate":"amber","Elevated":"amber","High":"red"}.get(str(_cp.get("RiskBand","")), "")
                st.markdown(
                    f'<div class="profile-card">'
                    f'<div class="profile-name">{escape(str(_cp["PlayerName"]))}</div>'
                    f'<div class="profile-meta">{escape(str(_cp["TeamName"]))} · {escape(str(_cp["PositionGroup"]))} · {_cp["AgeYears"]:.0f} yrs</div>'
                    f'<div class="pill-row">'
                    f'<span class="pill teal">Q {_cp["QualityScore"]:.1f}</span>'
                    f'<span class="pill">Fit {_cp["RoleFitScore"]:.1f}</span>'
                    f'<span class="pill {_cp_risk_cls}">{escape(str(_cp.get("RiskBand","?")))}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        # Side-by-side radar charts for exactly 2 players
        if len(compare_df) == 2:
            _radar_cols = st.columns(2)
            for _ri, (_, _rp) in enumerate(compare_df.iterrows()):
                with _radar_cols[_ri]:
                    _rp_pos_df = df.loc[df["PositionGroup"].eq(_rp["PositionGroup"])]
                    st.pyplot(render_player_pizza(_rp_pos_df, _rp), clear_figure=True)
            st.markdown("<div class='workspace-label' style='font-size:.58rem;margin:6px 0 6px;'>Score breakdown</div>", unsafe_allow_html=True)

        compare_scores = compare_df[["PlayerName", "QualityScore", "RoleFitScore", "ProfileScore", "DecisionScore", "PerformanceReliabilityScore"]].melt(id_vars="PlayerName", var_name="Metric", value_name="Score")
        compare_chart = (
            alt.Chart(compare_scores)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=-30, labelColor="#8fa3b1", gridColor="#1e2d3d")),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelColor="#8fa3b1", gridColor="#1e2d3d")),
                color=alt.Color("PlayerName:N", title="Player",
                                scale=alt.Scale(range=["#00d4a8","#f59e0b","#8b5cf6","#ef4444"])),
                xOffset="PlayerName:N",
                tooltip=["PlayerName", "Metric", alt.Tooltip("Score:Q", format=".1f")],
            )
            .properties(height=420)
            .configure_view(fill="#0f1623", stroke=None)
            .configure(background="#080c14")
        )
        st.altair_chart(compare_chart, use_container_width=True)

with case_tab:
    st.markdown("<div class='workspace-label'>💼 Case Analysis — transfer value, resale &amp; style fit</div>", unsafe_allow_html=True)
    render_case_analysis_tab(filtered)

with export_tab:
    st.markdown("<div class='workspace-label'>📥 Export — board CSV, PDF report, shortlist</div>", unsafe_allow_html=True)
    export_cols = [c for c in ["PlayerName", "TeamName", "PositionGroup", "BundleLabel", "AgeYears", "MinutesPlayed", "QualityScore", "QualityTier", "RoleFitScore", "ProfileScore", "DecisionScore", "PerformanceReliabilityScore", "Readiness", "RiskBand", "Archetype", "QualityDrivers", "RiskFlags"] if c in filtered.columns]
    export_df = filtered[export_cols].round(3)
    shortlist_df = df.loc[df["PlayerName"].isin(st.session_state.get("shortlist_players", []))].sort_values("QualityScore", ascending=False)
    export_left, export_right = st.columns([1, 1])
    with export_left:
        st.download_button("Download board CSV", data=export_df.to_csv(index=False).encode("utf-8"), file_name="fchk_quality_board.csv", mime="text/csv", width="stretch")
    with export_right:
        st.download_button("Download board PDF", data=build_pdf(filtered, "FCHK Quality Scouting Report", scope_note=f"{len(filtered):,} outfield players · {model_preset} lens", top_n=75), file_name="fchk_quality_scouting_report.pdf", mime="application/pdf", type="primary", width="stretch")
    _sl_data = st.session_state.get("shortlist_data", {})
    _sl_names = list(_sl_data.keys())
    shortlist_df = df.loc[df["PlayerName"].isin(_sl_names)].sort_values("QualityScore", ascending=False)
    if not shortlist_df.empty:
        st.markdown(
            f'<div class="workspace-label" style="font-size:.58rem;margin:18px 0 8px;">'
            f'Shortlist — {len(shortlist_df)} player{"s" if len(shortlist_df)!=1 else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Merge in priority/notes/added from shortlist_data
        _sl_meta = pd.DataFrame([
            {"PlayerName": k, "Priority": v.get("priority","Watch"), "Notes": v.get("notes",""), "Added": v.get("added","")}
            for k, v in _sl_data.items()
        ])
        shortlist_enriched = shortlist_df.merge(_sl_meta, on="PlayerName", how="left")
        sl_cols = [c for c in ["PlayerName","TeamName","PositionGroup","AgeYears","QualityScore","QualityTier","RoleFitScore","RiskBand","Priority","Notes","Added"] if c in shortlist_enriched.columns]
        _priority_order = {"Hot": 0, "Watch": 1, "Observed": 2}
        shortlist_enriched["_pri_ord"] = shortlist_enriched["Priority"].map(_priority_order).fillna(3)
        shortlist_enriched = shortlist_enriched.sort_values(["_pri_ord", "QualityScore"], ascending=[True, False])
        st.dataframe(
            shortlist_enriched[sl_cols].rename(columns={"PlayerName":"Player","TeamName":"Team","PositionGroup":"Role","AgeYears":"Age","QualityScore":"Quality","QualityTier":"Tier","RoleFitScore":"Role Fit","RiskBand":"Risk"}).round(2),
            use_container_width=True, hide_index=True,
            column_config={
                "Quality":  st.column_config.ProgressColumn("Quality",  min_value=0, max_value=100, format="%.1f"),
                "Role Fit": st.column_config.ProgressColumn("Role Fit", min_value=0, max_value=100, format="%.1f"),
                "Age":      st.column_config.NumberColumn("Age", format="%.1f"),
                "Priority": st.column_config.SelectboxColumn("Priority", options=["Hot","Watch","Observed"], width="small"),
                "Notes":    st.column_config.TextColumn("Notes", width="medium"),
            },
        )
        sl_exp_left, sl_exp_right = st.columns(2)
        _sl_csv_cols = [c for c in export_cols if c in shortlist_enriched.columns]
        with sl_exp_left:
            st.download_button("Download shortlist CSV", data=shortlist_enriched[_sl_csv_cols + [c for c in ["Priority","Notes","Added"] if c in shortlist_enriched.columns]].to_csv(index=False).encode("utf-8"), file_name="fchk_shortlist.csv", mime="text/csv", width="stretch")
        with sl_exp_right:
            if st.button("Clear shortlist", width="stretch"):
                clear_shortlist()
                st.rerun()
