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
            Priority=("MarketTier", lambda x: x.isin(["Priority", "Must scout"]).sum()),
            MedianAge=("AgeYears", "median"),
        )
        .round(1)
        .reset_index()
    )
    market["PriorityShare"] = np.where(market["Players"].gt(0), market["Priority"] / market["Players"] * 100, 0).round(1)
    market["lat"] = market["Country"].map(lambda c: EUROPE_COUNTRY_COORDS[c][0])
    market["lon"] = market["Country"].map(lambda c: EUROPE_COUNTRY_COORDS[c][1])
    market["GoScore"] = (market["MedianScore"] * 0.65 + market["PriorityShare"] * 0.25 + np.minimum(market["Players"], 80) * 0.10).round(1)
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
    out["FitDrivers"] = out.apply(fit_drivers, axis=1)
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
    sorted_df = df.sort_values("ScoutFitScore", ascending=False).copy()
    priority_count = sorted_df["MarketTier"].isin(["Priority", "Must scout"]).sum() if "MarketTier" in sorted_df else 0
    median_fit = sorted_df["ScoutFitScore"].median() if not sorted_df.empty else 0
    median_age = sorted_df["AgeYears"].median() if not sorted_df.empty else 0

    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(scope_note, styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Executive Summary", styles["Heading2"]),
        _pdf_table(
            [
                ["Players", "Median fit", "Median age", "Priority+"],
                [f"{len(sorted_df):,}", f"{median_fit:.1f}", f"{median_age:.1f}", f"{priority_count:,}"],
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
                MedianFit=("ScoutFitScore", "median"),
                MedianAge=("AgeYears", "median"),
                Priority=("MarketTier", lambda x: x.isin(["Priority", "Must scout"]).sum()),
            )
            .round(1)
            .reset_index()
            .sort_values("MedianFit", ascending=False)
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
                    "ScoutFitScore",
                    "RoleFitScore",
                    "MarketTier",
                    "FitDrivers",
                ]
            ].rename(
                columns={
                    "PlayerName": "Player",
                    "TeamName": "Team",
                    "PositionGroup": "Pos",
                    "AgeYears": "Age",
                    "ScoutFitScore": "Fit",
                    "RoleFitScore": "Role Fit",
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
                    "Fit": pick["ScoutFitScore"],
                    "Role Fit": pick.get("RoleFitScore", 0),
                    "Drivers": pick.get("FitDrivers", ""),
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
        "ScoutFitScore",
        "Archetype",
        "CompositeRecruitmentScore",
        "ValueRecruitmentScore",
        "DecisionScore",
        "Readiness",
        "RiskBand",
        "RoleFitScore",
        "FitDrivers",
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
            "ScoutFitScore": "Fit",
            "CompositeRecruitmentScore": "Composite",
            "ValueRecruitmentScore": "Value",
            "DecisionScore": "Decision",
            "RiskBand": "Risk",
            "RoleFitScore": "Role Fit",
            "FitDrivers": "Drivers",
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
    if mode == "U23 hunt":
        st.session_state["u23_filter"] = True
        st.session_state["fit_floor"] = 35
        st.session_state["max_risk"] = 18.0
    elif mode == "Priority board":
        st.session_state["fit_floor"] = 58
        st.session_state["composite_floor"] = 45
        st.session_state["reliability_floor"] = 55
    elif mode == "Low-risk only":
        st.session_state["max_risk"] = 9.0
        st.session_state["reliability_floor"] = 70
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
        --ink: #09131c;
        --muted: #5f6b76;
        --panel: #ffffff;
        --line: #d3dde3;
        --navy: #07111a;
        --navy-2: #0f1e2b;
        --teal: #2ec4a6;
        --green: #38b000;
        --amber: #ff9f1c;
        --red: #e5383b;
        --wash: #f3f6f8;
    }

    .stApp {
        background: var(--wash);
        color: var(--ink);
    }

    .block-container {
        padding-top: .7rem;
        padding-bottom: 1.4rem;
        max-width: 94%;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
        font-weight: 950;
    }

    h2 {
        font-size: 1.05rem !important;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    section[data-testid="stSidebar"] {
        background: #f7fafb;
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: .8rem;
    }

    .workspace-nav-spacer {
        height: 24px;
    }

    .hero {
        max-width: 760px;
        margin: 16vh auto 28px auto;
        color: var(--ink);
        text-align: center;
    }

    .hero-content,
    .hero-panel {
        position: relative;
        z-index: 1;
    }

    .hero-kicker {
        color: #66737d;
        font-size: .62rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .16em;
    }

    .hero h1 {
        color: var(--ink);
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
        color: #66737d;
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
        letter-spacing: -.04em;
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
        gap: 10px;
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
        letter-spacing: -.03em;
        margin-top: 4px;
    }

    .landing-card-copy {
        color: #aebfc8;
        font-size: .68rem;
        line-height: 1.32;
        margin-top: 5px;
    }

    .workspace-label {
        color: #66737d;
        font-size: .62rem;
        font-weight: 950;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin: 6px 0 8px 0;
    }

    .analysis-panel {
        border: 1px solid #1b2b38;
        border-radius: 2px;
        background: linear-gradient(180deg, #07111a 0%, #0d1823 100%);
        color: #dce5ea;
        padding: 10px 11px;
    }

    .analysis-panel .panel-kicker {
        color: #7cead5;
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
        letter-spacing: -.04em;
    }

    .analysis-panel .panel-copy {
        color: #b7c5cd;
        font-size: .72rem;
        line-height: 1.35;
        margin-top: 5px;
    }

    .metric-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 13px;
        background: white;
        min-height: 82px;
        box-shadow: 0 1px 2px rgba(16, 33, 43, .04);
    }

    .metric-label {
        color: #66737d;
        font-size: .58rem;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-weight: 950;
    }

    .metric-value {
        margin-top: 2px;
        color: #09131c;
        font-size: 1.28rem;
        font-weight: 950;
        line-height: 1;
        letter-spacing: 0;
    }

    .metric-caption {
        color: #66737d;
        font-size: .66rem;
        margin-top: 4px;
        line-height: 1.25;
    }

    .profile-card {
        border: 1px solid var(--line);
        border-radius: 2px;
        padding: 10px;
        background: white;
        min-height: 104px;
    }

    .profile-name {
        color: #09131c;
        font-weight: 950;
        font-size: .95rem;
        line-height: 1;
        letter-spacing: -.03em;
    }

    .profile-meta {
        color: #66737d;
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
        border: 1px solid #ccd8df;
        border-radius: 2px;
        color: #09131c;
        background: #f9fbfc;
        padding: 2px 5px;
        font-size: .58rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .note-box {
        border-left: 2px solid var(--teal);
        background: #ffffff;
        padding: 7px 9px;
        border-radius: 0;
        color: #10212b;
        font-size: .74rem;
        line-height: 1.35;
    }

    .scout-label {
        display: inline-flex;
        align-items: center;
        border: 1px solid #ccd8df;
        border-radius: 2px;
        background: #fff;
        padding: 2px 5px;
        margin: 1px 3px 1px 0;
        font-size: .56rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .1em;
        color: #09131c;
    }

    .section-card {
        border: 1px solid var(--line);
        border-radius: 2px;
        background: white;
        padding: 9px;
    }

    .homepage {
        display: grid;
        grid-template-columns: 1.35fr .95fr;
        gap: 12px;
        align-items: stretch;
        margin-bottom: 12px;
    }

    .home-feature {
        border: 1px solid #1b2b38;
        border-radius: 2px;
        background:
            linear-gradient(135deg, rgba(46, 196, 166, .16) 0%, rgba(46, 196, 166, 0) 42%),
            linear-gradient(180deg, #07111a 0%, #0f1e2b 100%);
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
        border: 1px solid rgba(124, 234, 213, .16);
        background:
            linear-gradient(90deg, transparent 0 18%, rgba(255,255,255,.06) 18% 19%, transparent 19% 39%, rgba(255,255,255,.06) 39% 40%, transparent 40% 60%, rgba(255,255,255,.06) 60% 61%, transparent 61% 82%, rgba(255,255,255,.06) 82% 83%, transparent 83%),
            linear-gradient(0deg, transparent 0 22%, rgba(255,255,255,.05) 22% 23%, transparent 23% 48%, rgba(255,255,255,.05) 48% 49%, transparent 49% 74%, rgba(255,255,255,.05) 74% 75%, transparent 75%);
        transform: rotate(-8deg);
        opacity: .9;
    }

    .home-kicker {
        color: #7cead5;
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
        letter-spacing: -.05em;
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
        background: rgba(255, 255, 255, .06);
        border-radius: 2px;
        padding: 9px;
    }

    .home-stat-value {
        font-size: 1.32rem;
        font-weight: 950;
        letter-spacing: -.04em;
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
        border-radius: 2px;
        background: #ffffff;
        padding: 12px;
        min-height: 172px;
    }

    .home-pillar-label {
        color: #2a9d8f;
        font-size: .58rem;
        font-weight: 950;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    .home-pillar-title {
        color: #09131c;
        font-size: 1.18rem;
        font-weight: 950;
        letter-spacing: -.04em;
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
        border-radius: 8px;
        background: #ffffff;
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
        border: 1px solid #ccd8df;
        border-radius: 999px;
        background: #f7fafb;
        color: #09131c;
        padding: 3px 8px;
        font-size: .68rem;
        font-weight: 800;
        line-height: 1.2;
    }

    @media (max-width: 900px) {
        .hero,
        .homepage,
        .home-pillar-grid,
        .home-focus-strip,
        .home-stat-row,
        .landing-grid {
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
        border-radius: 2px;
        border: 1px solid #ccd8df;
        background: #fff;
        color: var(--ink);
        font-weight: 900;
        min-height: 34px;
        font-size: .72rem;
        letter-spacing: .04em;
        text-transform: uppercase;
        box-shadow: none;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: var(--navy);
        border-color: var(--navy);
        color: white;
    }

    div[data-testid="stTabs"] button {
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-size: .64rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 2px;
        overflow: hidden;
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
                <div class="hero-kicker">Hradeck</div>
                <h1>Football intelligence workspace</h1>
                <p>Choose a workspace to continue.</p>
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

st.markdown("<div class='workspace-label'>Scouting workspace</div>", unsafe_allow_html=True)

if "quick_mode" not in st.session_state:
    st.session_state["quick_mode"] = "Full board"
if "shortlist_players" not in st.session_state:
    st.session_state["shortlist_players"] = []

outfield_data = data.loc[~data["PositionGroup"].astype(str).eq("GK")].copy()

mode_columns = st.columns([1, 1, 1, 1, 1.6])
quick_modes = ["Full board", "U23 hunt", "Priority board", "Low-risk only"]
for idx, mode in enumerate(quick_modes):
    with mode_columns[idx]:
        st.button(
            mode,
            key=f"quick_{mode}",
            type="primary" if st.session_state["quick_mode"] == mode else "secondary",
            width="stretch",
            on_click=set_quick_mode,
            args=(mode,),
        )
with mode_columns[-1]:
    st.button("Reset filters", width="stretch", on_click=reset_filters)

with st.sidebar:
    using_v3_outputs = _model_file("recruitment").exists()
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">FCHK Scout Room</div>
            <div class="sidebar-brand-meta">{len(outfield_data):,} outfield players · {outfield_data['BundleLabel'].nunique()} leagues · {outfield_data['PositionGroup'].nunique()} roles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='menu-caption'>Use board modes for fast cuts, then refine the pool in the panels below.</div>", unsafe_allow_html=True)
    source_label = "FCHK Model V3 outputs"
    st.caption(f"Data source: {source_label} · {DEFAULT_MODEL_OUTPUT_DIR}")
    with st.expander("Loaded model files", expanded=False):
        st.caption(str(DEFAULT_MODEL_OUTPUT_DIR))
        for name, file_name in MODEL_OUTPUT_FILES.items():
            present = _model_file(name).exists()
            label = name.replace("_", " ").title()
            frame = model_metadata.get(name)
            size_note = "" if frame is None else f" · {len(frame):,} rows × {len(frame.columns):,} cols"
            st.write(f"{'[x]' if present else '[ ]'} {label}: {file_name}{size_note}")
        if "summary" in model_metadata and not model_metadata["summary"].empty:
            summary = model_metadata["summary"].copy()
            if {"Item", "Value"}.issubset(summary.columns):
                st.dataframe(summary.head(12), hide_index=True, width="stretch")

    preset_weights = {
        "Balanced": {"Composite": 3, "Decision": 2, "Value": 2, "Success": 1, "Reliability": 1, "Risk penalty": 1},
        "Value hunters": {"Composite": 2, "Decision": 1, "Value": 4, "Success": 1, "Reliability": 1, "Risk penalty": 1},
        "Ready now": {"Composite": 3, "Decision": 3, "Value": 1, "Success": 2, "Reliability": 3, "Risk penalty": 2},
        "High upside": {"Composite": 2, "Decision": 1, "Value": 3, "Success": 2, "Reliability": 1, "Risk penalty": 0},
    }
    model_preset = st.segmented_control(
        "Recruitment model",
        list(preset_weights),
        default="Balanced",
        width="stretch",
    )
    defaults = preset_weights[model_preset]
    with st.expander("Scoring weights", expanded=False):
        st.markdown("<div class='menu-caption'>Tune how the Scout Fit score balances talent, value, readiness, and risk.</div>", unsafe_allow_html=True)
        weights = {
            "Composite": st.slider("Composite", 0, 5, defaults["Composite"]),
            "Decision": st.slider("Decision", 0, 5, defaults["Decision"]),
            "Value": st.slider("Value", 0, 5, defaults["Value"]),
            "Success": st.slider("Success probability", 0, 5, defaults["Success"]),
            "Reliability": st.slider("Reliability", 0, 5, defaults["Reliability"]),
            "Risk penalty": st.slider("Risk penalty", 0, 5, defaults["Risk penalty"]),
        }

df = add_scouting_fields(outfield_data, weights)

st.subheader("Filters")
with st.container(border=True):
    position_groups = sorted(df["PositionGroup"].dropna().astype(str).unique())
    saved_positions = st.session_state.get("positions_filter")
    if saved_positions and any(position not in position_groups for position in saved_positions):
        st.session_state["positions_filter"] = position_groups
    bundle_groups = sorted(df["BundleLabel"].dropna().astype(str).unique())
    archetype_groups = sorted(df["Archetype"].dropna().astype(str).unique())
    with st.expander("Search and roles", expanded=True):
        search = st.text_input("Search player or team", key="search_filter", placeholder="Type a name or club")
        st.markdown("<div class='menu-caption'>Role shortcuts</div>", unsafe_allow_html=True)
        pos_cols = st.columns(3)
        quick_positions = [
            ("DEF", ["CB", "FB"]),
            ("MID", ["DM", "CM", "AM"]),
            ("ATT", ["W", "ST"]),
        ]
        for idx, (label, values) in enumerate(quick_positions):
            with pos_cols[idx]:
                if st.button(label, key=f"pos_{label}", width="stretch"):
                    st.session_state["positions_filter"] = values
        positions = st.multiselect(
            "Position groups",
            position_groups,
            default=position_groups,
            key="positions_filter",
        )
        st.markdown(
            "<div class='filter-chip-row'>"
            + "".join(f"<span class='filter-chip'>{p}</span>" for p in positions[:8])
            + ("<span class='filter-chip'>more</span>" if len(positions) > 8 else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    with st.expander("Market filters", expanded=True):
        u23_only = st.toggle("U23 targets only", value=False, key="u23_filter")
        age_range = st.slider(
            "Age range",
            float(np.floor(df["AgeYears"].min())),
            float(np.ceil(df["AgeYears"].max())),
            (float(np.floor(df["AgeYears"].min())), float(np.ceil(df["AgeYears"].max()))),
            step=0.5,
            key="age_filter",
        )
        minutes_range = st.slider(
            "Minutes played",
            int(df["MinutesPlayed"].min()),
            int(df["MinutesPlayed"].max()),
            (900, int(df["MinutesPlayed"].max())),
            step=100,
            key="minutes_filter",
        )
        fit_floor = st.slider("Scout fit floor", 0, 100, 35, key="fit_floor")
        composite_floor = st.slider("Composite floor", 0, 100, 35, key="composite_floor")
    with st.expander("Leagues and profiles", expanded=False):
        bundles = st.multiselect(
            "Leagues / bundles",
            bundle_groups,
            default=bundle_groups,
            key="bundles_filter",
        )
        archetypes = st.multiselect(
            "Archetypes",
            archetype_groups,
            default=archetype_groups,
            key="archetypes_filter",
        )
    with st.expander("Risk controls", expanded=False):
        reliability_floor = st.slider("Reliability floor", 0, 100, 45, key="reliability_floor")
        max_risk = st.slider("Max security risk/90", 0.0, 25.0, 18.0, step=0.5, key="max_risk")

mask = (
    df["PositionGroup"].astype(str).isin(positions)
    & df["BundleLabel"].astype(str).isin(bundles)
    & df["Archetype"].astype(str).isin(archetypes)
    & df["AgeYears"].between(age_range[0], age_range[1])
    & df["MinutesPlayed"].between(minutes_range[0], minutes_range[1])
    & df["ScoutFitScore"].ge(fit_floor)
    & df["CompositeRecruitmentScore"].ge(composite_floor)
    & df["PerformanceReliabilityScore"].ge(reliability_floor)
    & df["SecurityRisk_per90"].le(max_risk)
)
if u23_only and "IsU23Target" in df:
    mask &= df["IsU23Target"].fillna(False).astype(bool)
if search:
    haystack = (
        df["PlayerName"].fillna("").astype(str)
        + " "
        + df["TeamName"].fillna("").astype(str)
    ).str.lower()
    mask &= haystack.str.contains(search.lower(), regex=False)

filtered = df.loc[mask].sort_values(["ScoutFitScore", "CompositeRecruitmentScore"], ascending=False)

filter_tokens = [
    f"{len(positions)} roles" if len(positions) != len(position_groups) else "All roles",
    f"{len(bundles)} leagues" if len(bundles) != len(bundle_groups) else "All leagues",
    "U23 only" if u23_only else "All ages",
    f"{age_range[0]:.0f}-{age_range[1]:.0f} yrs",
    f"{minutes_range[0]:,}+ min",
    f"Fit {fit_floor}+",
    f"Reliability {reliability_floor}+",
    f"Risk <= {max_risk:.1f}/90",
]
if search:
    filter_tokens.insert(0, f"Search: {search}")
st.markdown(
    "<div class='filter-summary'><span class='filter-summary-label'>Active view</span>"
    + "".join(f"<span class='filter-token'>{escape(str(token))}</span>" for token in filter_tokens)
    + "</div>",
    unsafe_allow_html=True,
)

top = filtered.head(1)
cols = st.columns(5)
with cols[0]:
    metric_card("Players", f"{len(filtered):,}", "after active filters")
with cols[1]:
    metric_card(
        "Top Fit",
        "n/a" if top.empty else f"{top.iloc[0]['ScoutFitScore']:.1f}",
        "" if top.empty else str(top.iloc[0]["PlayerName"]),
    )
with cols[2]:
    metric_card("Median Age", "n/a" if filtered.empty else f"{filtered['AgeYears'].median():.1f}", "filtered pool")
with cols[3]:
    metric_card(
        "U23 Share",
        "n/a"
        if filtered.empty or "IsU23Target" not in filtered
        else f"{filtered['IsU23Target'].fillna(False).mean() * 100:.0f}%",
        "filtered pool",
    )
with cols[4]:
    metric_card(
        "Priority+",
        "n/a" if filtered.empty else f"{filtered['MarketTier'].isin(['Priority', 'Must scout']).sum():,}",
        "priority or must-scout tier",
    )

tab_overview, tab_board, tab_player, tab_compare, tab_reports, tab_data = st.tabs(
    ["Overview", "Board", "Player", "Compare", "Reports", "Data"]
)

with tab_overview:
    top_team = filtered.head(1)
    top_u23 = filtered.loc[filtered.get("IsU23Target", pd.Series(False, index=filtered.index)).fillna(False).astype(bool)].head(1)
    st.markdown(
        f"""
        <div class="homepage">
            <div class="home-feature">
                <div class="home-kicker">Scouting overview</div>
                <div class="home-title">Start with the shortlist decision.</div>
                <div class="home-copy">
                    Use the filters and quick modes to narrow the pool, then move through Board, Player, Compare, and Reports.
                    The overview only shows the signals needed to decide where to look next.
                </div>
                <div class="home-stat-row">
                    <div class="home-stat">
                        <div class="home-stat-value">{len(filtered):,}</div>
                        <div class="home-stat-label">Players in scope</div>
                    </div>
                    <div class="home-stat">
                        <div class="home-stat-value">{filtered['MarketTier'].isin(['Priority', 'Must scout']).sum() if not filtered.empty else 0:,}</div>
                        <div class="home-stat-label">Priority targets</div>
                    </div>
                    <div class="home-stat">
                        <div class="home-stat-value">{filtered['PositionGroup'].nunique() if not filtered.empty else 0}</div>
                        <div class="home-stat-label">Role groups</div>
                    </div>
                </div>
            </div>
            <div class="home-pillar-grid">
                <div class="home-pillar">
                    <div class="home-pillar-label">01</div>
                    <div class="home-pillar-title">1. Filter</div>
                    <div class="home-pillar-copy">Use role, age, minutes, league, reliability, and risk filters to define the current scouting pool.</div>
                </div>
                <div class="home-pillar">
                    <div class="home-pillar-label">02</div>
                    <div class="home-pillar-title">2. Shortlist</div>
                    <div class="home-pillar-copy">Open Board to tick players into a basket from the ranked table.</div>
                </div>
                <div class="home-pillar">
                    <div class="home-pillar-label">03</div>
                    <div class="home-pillar-title">3. Inspect</div>
                    <div class="home-pillar-copy">Open Player for one profile, strengths, risk notes, and similar players.</div>
                </div>
                <div class="home-pillar">
                    <div class="home-pillar-label">04</div>
                    <div class="home-pillar-title">4. Export</div>
                    <div class="home-pillar-copy">Use Reports when the shortlist is ready to share.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    home_focus_cols = st.columns(4)
    with home_focus_cols[0]:
        metric_card("Scouting lead", "n/a" if top_team.empty else str(top_team.iloc[0]["PlayerName"]), "highest Scout Fit")
    with home_focus_cols[1]:
        metric_card("Recruitment score", "n/a" if top_team.empty else f"{top_team.iloc[0]['ScoutFitScore']:.1f}", "current top target")
    with home_focus_cols[2]:
        metric_card("Top U23", "n/a" if top_u23.empty else str(top_u23.iloc[0]["PlayerName"]), "highest young outfield fit")
    with home_focus_cols[3]:
        metric_card("Team signal", "n/a" if filtered.empty else filtered.groupby("PositionGroup")["ScoutFitScore"].median().sort_values(ascending=False).index[0], "strongest role")

    st.markdown(
        """
        <div class="analysis-panel">
            <div class="panel-kicker">Market command view</div>
            <div class="panel-title">Where should FCHK send eyes next?</div>
            <div class="panel-copy">Country bubbles combine model quality, player depth, and priority-player density. Use the score selector to switch between fit, value, readiness, and decision quality.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    home_metric = st.selectbox(
        "Map score",
        [c for c in ["ScoutFitScore", "CompositeRecruitmentScore", "ValueRecruitmentScore", "DecisionScore", "PerformanceReliabilityScore"] if c in filtered.columns],
        index=0,
        help="Countries are colored by median score. Bigger bubbles mean more players in the current filter.",
    )
    market_map = european_market_map_frame(filtered, home_metric)

    if market_map.empty:
        st.info("No country coordinates were found. Check that CountryLabel exists in the Excel output, or that BundleLabel ends with a country name.")
    else:
        map_left, map_right = st.columns([1.55, 1])
        with map_left:
            europe = alt.Chart(alt.topo_feature("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json", "countries")).mark_geoshape(
                fill="#eef3f6",
                stroke="#cbd8de",
                strokeWidth=0.6,
            ).project(
                type="mercator",
                center=[12, 53],
                scale=520,
            ).properties(height=560)

            points = (
                alt.Chart(market_map)
                .mark_circle(opacity=0.86, stroke="#10212b", strokeWidth=0.8)
                .encode(
                    longitude="lon:Q",
                    latitude="lat:Q",
                    size=alt.Size("Players:Q", title="Players", scale=alt.Scale(range=[90, 1700])),
                    color=alt.Color(
                        "MedianScore:Q",
                        title=f"Median {home_metric}",
                        scale=alt.Scale(scheme="redyellowgreen", domain=[35, 70]),
                    ),
                    tooltip=[
                        "Country",
                        "Recommendation",
                        alt.Tooltip("Players:Q", format=","),
                        alt.Tooltip("MedianScore:Q", title=f"Median {home_metric}", format=".1f"),
                        alt.Tooltip("TopScore:Q", title="Top score", format=".1f"),
                        alt.Tooltip("Priority:Q", title="Priority+ players", format=","),
                        alt.Tooltip("PriorityShare:Q", title="Priority share", format=".1f"),
                        alt.Tooltip("MedianAge:Q", title="Median age", format=".1f"),
                        alt.Tooltip("GoScore:Q", title="GoScore", format=".1f"),
                    ],
                )
            )

            labels = (
                alt.Chart(market_map.head(12))
                .mark_text(dy=-16, fontSize=11, fontWeight="bold", color="#10212b")
                .encode(longitude="lon:Q", latitude="lat:Q", text="Country:N")
            )
            st.altair_chart(europe + points + labels, width="stretch")

        with map_right:
            st.subheader("Go / watch list")
            top_markets = market_map.head(8).copy()
            st.dataframe(
                top_markets[["Country", "Recommendation", "Players", "MedianScore", "Priority", "PriorityShare", "GoScore"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "MedianScore": st.column_config.ProgressColumn("Median score", min_value=0, max_value=100, format="%.1f"),
                    "PriorityShare": st.column_config.ProgressColumn("Priority %", min_value=0, max_value=100, format="%.1f%%"),
                    "GoScore": st.column_config.ProgressColumn("GoScore", min_value=0, max_value=100, format="%.1f"),
                },
            )

            if not top_markets.empty:
                best = top_markets.iloc[0]
                st.markdown(
                    f"""
                    <div class="analysis-panel">
                        <div class="panel-kicker">Best trip signal</div>
                        <div class="panel-title">{best['Country']}</div>
                        <div class="panel-copy">{best['Recommendation']} · GoScore {best['GoScore']:.1f} · {int(best['Players']):,} players in current scope</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.subheader("Board pulse")
    st.markdown(
        """
        <span class="scout-label">Signal <strong>Model-led</strong></span>
        <span class="scout-label">Lens <strong>Value + Fit</strong></span>
        <span class="scout-label">Action <strong>Trip planning</strong></span>
        """,
        unsafe_allow_html=True,
    )
    pulse_cols = st.columns(4)
    with pulse_cols[0]:
        metric_card("Must scout", f"{filtered['MarketTier'].eq('Must scout').sum():,}" if not filtered.empty else "0", "highest model tier")
    with pulse_cols[1]:
        metric_card("Priority countries", f"{market_map['Recommendation'].isin(['Scout next', 'Go now']).sum():,}" if not market_map.empty else "0", "map signals")
    with pulse_cols[2]:
        metric_card("Best median role", filtered.groupby("PositionGroup")["ScoutFitScore"].median().sort_values(ascending=False).index[0] if not filtered.empty else "n/a", "by Scout Fit")
    with pulse_cols[3]:
        metric_card("Median value", "n/a" if filtered.empty else f"{filtered['ValueRecruitmentScore'].median():.1f}", "filtered pool")

    dash_left, dash_right = st.columns([1, 1])
    with dash_left:
        st.subheader("Role strength")
        if not filtered.empty:
            role_strength = (
                filtered.groupby("PositionGroup")
                .agg(Players=("PlayerName", "count"), MedianFit=("ScoutFitScore", "median"), Priority=("MarketTier", lambda x: x.isin(["Priority", "Must scout"]).sum()))
                .round(1)
                .reset_index()
                .sort_values("MedianFit", ascending=False)
            )
            role_chart = (
                alt.Chart(role_strength)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("PositionGroup:N", title="Role"),
                    y=alt.Y("MedianFit:Q", title="Median Scout Fit", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("PositionGroup:N", legend=None, scale=alt.Scale(domain=list(POSITION_COLORS), range=list(POSITION_COLORS.values()))),
                    tooltip=["PositionGroup", "Players", alt.Tooltip("MedianFit:Q", format=".1f"), "Priority"],
                )
                .properties(height=300)
            )
            st.altair_chart(role_chart, width="stretch")
    with dash_right:
        st.subheader("Immediate targets")
        target_cols = ["PlayerName", "TeamName", "PositionGroup", "BundleLabel", "AgeYears", "ScoutFitScore", "MarketTier", "FitDrivers"]
        st.dataframe(filtered[[c for c in target_cols if c in filtered.columns]].head(10).round(2), width="stretch", hide_index=True)

with tab_overview:
    st.subheader("Recruitment board")
    if filtered.empty:
        st.info("No players match the active filters.")
    else:
        top_cards = filtered.head(3)
        card_cols = st.columns(3)
        for idx, (_, row) in enumerate(top_cards.iterrows()):
            with card_cols[idx]:
                st.markdown(
                    f"""
                    <div class="profile-card">
                        <div class="profile-name">{row['PlayerName']}</div>
                        <div class="profile-meta">{row['TeamName']} · {row['PositionGroup']} · {row['AgeYears']:.1f} yrs</div>
                        <div class="pill-row">
                            <span class="pill">Fit {row['ScoutFitScore']:.1f}</span>
                            <span class="pill">{row['MarketTier']}</span>
                            <span class="pill">{row['Archetype']}</span>
                        </div>
                        <div class="profile-meta" style="margin-top:12px;">{profile_note(row)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Fit vs value")
            scatter_source = filtered.head(750).copy()
            scatter = (
                alt.Chart(scatter_source)
                .mark_circle(size=80, opacity=0.72)
                .encode(
                    x=alt.X("ValueRecruitmentScore:Q", title="Value score"),
                    y=alt.Y("ScoutFitScore:Q", title="Scout fit"),
                    color=alt.Color(
                        "PositionGroup:N",
                        title="Position",
                        scale=alt.Scale(domain=list(POSITION_COLORS), range=list(POSITION_COLORS.values())),
                    ),
                    size=alt.Size("MinutesPlayed:Q", title="Minutes", scale=alt.Scale(range=[35, 220])),
                    tooltip=[
                        "PlayerName",
                        "TeamName",
                        "PositionGroup",
                        "Archetype",
                        alt.Tooltip("ScoutFitScore:Q", format=".1f"),
                        alt.Tooltip("CompositeRecruitmentScore:Q", format=".1f"),
                        alt.Tooltip("AgeYears:Q", format=".1f"),
                    ],
                )
                .properties(height=390)
                .interactive()
            )
            st.altair_chart(scatter, width="stretch")
        with right:
            st.subheader("Market tiers")
            tier_order = ["Must scout", "Priority", "Shortlist", "Depth", "Watch"]
            tier_counts = filtered["MarketTier"].value_counts().reindex(tier_order).dropna().reset_index()
            tier_counts.columns = ["Tier", "Players"]
            bars = (
                alt.Chart(tier_counts)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("Tier:N", sort=tier_order, title=None),
                    x=alt.X("Players:Q", title="Players"),
                    color=alt.Color(
                        "Tier:N",
                        legend=None,
                        scale=alt.Scale(domain=list(TIER_COLORS), range=list(TIER_COLORS.values())),
                    ),
                    tooltip=["Tier", "Players"],
                )
                .properties(height=260)
            )
            st.altair_chart(bars, width="stretch")
            st.dataframe(
                filtered.groupby("PositionGroup")
                .agg(
                    Players=("PlayerName", "count"),
                    MedianFit=("ScoutFitScore", "median"),
                    MedianAge=("AgeYears", "median"),
                    Priority=("MarketTier", lambda x: x.isin(["Priority", "Must scout"]).sum()),
                )
                .round(1)
                .sort_values("MedianFit", ascending=False),
                width="stretch",
            )

with tab_board:
    st.subheader("Start here")
    board_cols = [
        "PlayerName",
        "TeamName",
        "PositionGroup",
        "AgeYears",
        "MinutesPlayed",
        "ScoutFitScore",
        "MarketTier",
        "Readiness",
        "RiskBand",
        "FitDrivers",
    ]
    if filtered.empty:
        st.info("No players match the active filters.")
    else:
        quick_board = filtered[[c for c in board_cols if c in filtered.columns]].head(15).rename(
            columns={
                "PlayerName": "Player",
                "TeamName": "Team",
                "PositionGroup": "Role",
                "AgeYears": "Age",
                "MinutesPlayed": "Minutes",
                "ScoutFitScore": "Fit",
                "MarketTier": "Tier",
                "RiskBand": "Risk",
                "FitDrivers": "Drivers",
            }
        )
        st.dataframe(
            quick_board.round(2),
            width="stretch",
            hide_index=True,
            column_config={
                "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.1f"),
            },
        )

    st.subheader("Role board")
    role = st.segmented_control("Role board", position_groups, default=position_groups[0], width="stretch")
    role_df = filtered.loc[filtered["PositionGroup"].eq(role)].sort_values("RoleFitScore", ascending=False)
    if role_df.empty:
        st.info("No players match this role and the active filters.")
    else:
        role_cols = [
            "PlayerName",
            "TeamName",
            "BundleLabel",
            "AgeYears",
            "MinutesPlayed",
            "RoleFitScore",
            "ScoutFitScore",
            "MarketTier",
            "Archetype",
            "FitDrivers",
            "RiskFlags",
            "TierReason",
        ]
        st.dataframe(role_df[[c for c in role_cols if c in role_df.columns]].head(75).round(2), width="stretch", hide_index=True)
        left, right = st.columns([1, 1])
        with left:
            st.pyplot(render_score_distribution(role_df, "RoleFitScore"), clear_figure=True)
        with right:
            st.pyplot(render_score_distribution(role_df, "ScoutFitScore"), clear_figure=True)

with tab_board:
    st.subheader("Ranked shortlist")
    shortlist_cols = [
        "PlayerName",
        "TeamName",
        "PositionGroup",
        "BundleLabel",
        "AgeYears",
        "MinutesPlayed",
        "ScoutFitScore",
        "MarketTier",
        "Archetype",
        "Readiness",
        "RiskBand",
        "CompositeRecruitmentScore",
        "DecisionScore",
        "ValueRecruitmentScore",
        "SuccessProbability",
        "PerformanceReliabilityScore",
        "SecurityRisk_per90",
        "RoleFitScore",
        "FitDrivers",
        "RiskFlags",
        "TierReason",
    ]
    view_cols = [c for c in shortlist_cols if c in filtered.columns]
    if filtered.empty:
        st.info("No players match the active filters.")
    else:
        picker_cols = [
            "PlayerName",
            "TeamName",
            "PositionGroup",
            "BundleLabel",
            "AgeYears",
            "ScoutFitScore",
            "MarketTier",
            "Readiness",
            "RiskBand",
            "FitDrivers",
        ]
        shortlist_picker = filtered[[c for c in picker_cols if c in filtered.columns]].head(75).copy()
        shortlist_picker.insert(0, "Add", False)
        edited_picker = st.data_editor(
            shortlist_picker.round(2),
            width="stretch",
            hide_index=True,
            key="shortlist_picker",
            disabled=[c for c in shortlist_picker.columns if c != "Add"],
            column_config={
                "Add": st.column_config.CheckboxColumn("Add", help="Tick players to add to the shortlist basket."),
                "PlayerName": st.column_config.TextColumn("Player"),
                "TeamName": st.column_config.TextColumn("Team"),
                "PositionGroup": st.column_config.TextColumn("Role"),
                "BundleLabel": st.column_config.TextColumn("League"),
                "AgeYears": st.column_config.NumberColumn("Age", format="%.1f"),
                "ScoutFitScore": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.1f"),
            },
        )
        add_cols = st.columns([1, 1, 3])
        with add_cols[0]:
            if st.button("Add checked", type="primary", width="stretch"):
                for player_name in edited_picker.loc[edited_picker["Add"], "PlayerName"].astype(str):
                    add_to_shortlist(player_name)
        with add_cols[1]:
            st.button("Clear basket", width="stretch", on_click=clear_shortlist)
        with add_cols[2]:
            st.caption(f"{len(st.session_state.get('shortlist_players', []))} players in shortlist basket")

    shortlist_df = df.loc[df["PlayerName"].isin(st.session_state.get("shortlist_players", []))].sort_values("ScoutFitScore", ascending=False)
    if not shortlist_df.empty:
        st.subheader("Shortlist basket")
        st.dataframe(shortlist_df[[c for c in view_cols if c in shortlist_df.columns]].round(2), width="stretch", hide_index=True)
        basket_cols = st.columns([1, 1])
        with basket_cols[0]:
            st.download_button(
                "Download shortlist CSV",
                data=shortlist_df[[c for c in view_cols if c in shortlist_df.columns]].to_csv(index=False).encode("utf-8"),
                file_name="fchk_shortlist_basket.csv",
                mime="text/csv",
                width="stretch",
            )
        with basket_cols[1]:
            st.download_button(
                "Download shortlist PDF",
                data=build_pdf(shortlist_df, "FCHK Shortlist Basket", scope_note=f"{len(shortlist_df):,} manually shortlisted players", top_n=100),
                file_name="fchk_shortlist_basket.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
            )

    st.subheader("Ranked board")
    st.dataframe(
        filtered[view_cols].round(2),
        width="stretch",
        hide_index=True,
        column_config={
            "ScoutFitScore": st.column_config.ProgressColumn("Scout Fit", min_value=0, max_value=100),
            "CompositeRecruitmentScore": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100),
            "ValueRecruitmentScore": st.column_config.ProgressColumn("Value", min_value=0, max_value=100),
            "DecisionScore": st.column_config.ProgressColumn("Decision", min_value=0, max_value=100),
            "PerformanceReliabilityScore": st.column_config.ProgressColumn("Reliability", min_value=0, max_value=100),
        },
    )
    st.markdown("<div class='table-hint'>Tip: sort any score column or use the quick position buttons in the sidebar to narrow the board.</div>", unsafe_allow_html=True)

    left, mid, right = st.columns([1, 1, 1])
    with left:
        st.subheader("Archetype mix")
        if not filtered.empty:
            st.bar_chart(filtered["Archetype"].value_counts())
    with mid:
        st.subheader("Position mix")
        if not filtered.empty:
            st.bar_chart(filtered["PositionGroup"].value_counts())
    with right:
        st.subheader("Risk bands")
        if not filtered.empty:
            st.bar_chart(filtered["RiskBand"].value_counts())

with tab_compare:
    st.subheader("Player comparison")
    if filtered.empty:
        st.info("No players match the active filters.")
    else:
        compare_options = (
            filtered.assign(_label=filtered["PlayerName"] + " | " + filtered["TeamName"] + " | " + filtered["PositionGroup"])
            .sort_values("ScoutFitScore", ascending=False)["_label"]
            .tolist()
        )
        selected_compare = st.multiselect("Select 2-4 players", compare_options, default=compare_options[: min(3, len(compare_options))], max_selections=4)
        compare_names = [label.split(" | ")[0] for label in selected_compare]
        compare_df = filtered.loc[filtered["PlayerName"].isin(compare_names)].sort_values("ScoutFitScore", ascending=False)
        if len(compare_df) < 2:
            st.info("Pick at least two players to compare.")
        else:
            summary_cols = [
                "PlayerName",
                "TeamName",
                "PositionGroup",
                "AgeYears",
                "MinutesPlayed",
                "ScoutFitScore",
                "RoleFitScore",
                "MarketTier",
                "FitDrivers",
                "RiskFlags",
                "TierReason",
            ]
            st.dataframe(compare_df[[c for c in summary_cols if c in compare_df.columns]].round(2), width="stretch", hide_index=True)
            pizza_cols = st.columns(len(compare_df))
            for idx, (_, row) in enumerate(compare_df.iterrows()):
                with pizza_cols[idx]:
                    st.pyplot(render_player_pizza(df.loc[df["PositionGroup"].eq(row["PositionGroup"])], row), clear_figure=True)
            compare_scores = compare_df[["PlayerName"] + [c for c in PIZZA_METRICS.values() if c in compare_df.columns]].melt(
                id_vars="PlayerName", var_name="Metric", value_name="Score"
            )
            chart = (
                alt.Chart(compare_scores)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("PlayerName:N", title="Player"),
                    xOffset="PlayerName:N",
                    tooltip=["PlayerName", "Metric", alt.Tooltip("Score:Q", format=".1f")],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, width="stretch")

with tab_player:
    st.subheader("Player lab")
    if filtered.empty:
        st.info("No players match the active filters.")
    else:
        player_options = (
            filtered.assign(_label=filtered["PlayerName"] + " | " + filtered["TeamName"] + " | " + filtered["PositionGroup"])
            .sort_values("ScoutFitScore", ascending=False)
        )
        selected_label = st.selectbox("Player", player_options["_label"].tolist())
        player = player_options.loc[player_options["_label"].eq(selected_label)].iloc[0]

        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-name">{player['PlayerName']}</div>
                <div class="profile-meta">{player['TeamName']} · {player['BundleLabel']} · {player['PositionGroup']} · {player['AgeYears']:.1f} years · {int(player['MinutesPlayed']):,} minutes</div>
                <div class="pill-row">
                    <span class="pill">{player['Archetype']}</span>
                    <span class="pill">{player['MarketTier']}</span>
                    <span class="pill">{player['Readiness']}</span>
                    <span class="pill">{player['RiskBand']} risk</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        a, b, c, d, e = st.columns(5)
        a.metric("Scout Fit", f"{player['ScoutFitScore']:.1f}")
        b.metric("Composite", f"{player['CompositeRecruitmentScore']:.1f}")
        c.metric("Value", f"{player['ValueRecruitmentScore']:.1f}")
        d.metric("Decision", f"{player['DecisionScore']:.1f}")
        e.metric("Position pctile", f"{percentile_rank(df.loc[df['PositionGroup'].eq(player['PositionGroup']), 'ScoutFitScore'], player['ScoutFitScore']):.0f}")
        st.markdown(f"<div class='note-box'>{profile_note(player)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='note-box'>{player['TierReason']}</div>", unsafe_allow_html=True)
        action_cols = st.columns([1, 1, 3])
        with action_cols[0]:
            if st.button("Add to shortlist", type="primary", width="stretch"):
                add_to_shortlist(player["PlayerName"])
        with action_cols[1]:
            st.metric("Role Fit", f"{player['RoleFitScore']:.1f}")

        player_scores = pd.DataFrame(
            {
                "Score": [c for c in SCORE_COLUMNS if c in filtered.columns],
                "Value": [float(player[c]) for c in SCORE_COLUMNS if c in filtered.columns],
            }
        )
        pizza_ref = df.loc[df["PositionGroup"].eq(player["PositionGroup"])]
        st.subheader("mplsoccer percentile pizza")
        st.pyplot(render_player_pizza(pizza_ref, player), clear_figure=True)

        left, right = st.columns([1.15, 1])
        with left:
            score_chart = (
                alt.Chart(player_scores)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Score:N", sort=None, title=None, axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Value:Q", legend=None, scale=alt.Scale(scheme="tealblues")),
                    tooltip=["Score", alt.Tooltip("Value:Q", format=".1f")],
                )
                .properties(height=360)
            )
            st.altair_chart(score_chart, width="stretch")

        with right:
            per90_cols = [c for c in filtered.columns if c.endswith("_per90") or c.startswith("Imp_")]
            per90 = pd.DataFrame(
                {
                    "Metric": per90_cols,
                    "Value": [float(player[c]) for c in per90_cols],
                }
            ).sort_values("Value", ascending=False)
            st.dataframe(per90.round(3), width="stretch", hide_index=True)

        comparable = similar_players(df, player, same_position=True, n=12)
        st.subheader("Similarity search")
        st.dataframe(
            comparable[
                [
                    "PlayerName",
                    "TeamName",
                    "PositionGroup",
                    "AgeYears",
                    "MinutesPlayed",
                    "SimilarityScore",
                    "ScoutFitScore",
                    "RoleFitScore",
                    "Archetype",
                    "CompositeRecruitmentScore",
                    "ValueRecruitmentScore",
                ]
            ].round(2),
            width="stretch",
            hide_index=True,
        )

with tab_data:
    st.subheader("Advanced visuals")
    if filtered.empty:
        st.info("No players match the active filters.")
    else:
        visual_cols = st.columns([1, 1, 1])
        with visual_cols[0]:
            visual_metric = st.selectbox(
                "Visual metric",
                SCORE_COLUMNS,
                index=SCORE_COLUMNS.index("CompositeRecruitmentScore"),
            )
        with visual_cols[1]:
            reference_scope = st.selectbox("Reference pool", ["Filtered board", "Full database", "Same position as selected player"])
        with visual_cols[2]:
            selected_visual_player = st.selectbox(
                "Highlight player",
                filtered.assign(_label=filtered["PlayerName"] + " | " + filtered["TeamName"])
                .sort_values("ScoutFitScore", ascending=False)["_label"]
                .tolist(),
            )

        selected_row = (
            filtered.assign(_label=filtered["PlayerName"] + " | " + filtered["TeamName"])
            .loc[lambda x: x["_label"].eq(selected_visual_player)]
            .iloc[0]
        )
        if reference_scope == "Full database":
            reference_df = df
        elif reference_scope == "Same position as selected player":
            reference_df = df.loc[df["PositionGroup"].eq(selected_row["PositionGroup"])]
        else:
            reference_df = filtered

        st.markdown(
            f"""
            <div class="note-box">
                Reference pool: {reference_scope} · {len(reference_df):,} players. Highlighting {selected_row['PlayerName']} against {visual_metric}.
            </div>
            """,
            unsafe_allow_html=True,
        )

        top_left, top_right = st.columns([1, 1])
        with top_left:
            st.pyplot(render_player_pizza(reference_df, selected_row), clear_figure=True)
        with top_right:
            highlight_value = float(selected_row[visual_metric]) if visual_metric in selected_row else None
            st.pyplot(render_score_distribution(reference_df, visual_metric, highlight=highlight_value), clear_figure=True)

        bottom_left, bottom_right = st.columns([1, 1])
        with bottom_left:
            st.pyplot(render_league_heatmap(reference_df, visual_metric), clear_figure=True)
        with bottom_right:
            st.pyplot(render_position_boxplot(reference_df, visual_metric), clear_figure=True)

        st.subheader("Percentile matrix")
        percentile_rows = []
        for _, row in filtered.head(25).iterrows():
            row_ref = df.loc[df["PositionGroup"].eq(row["PositionGroup"])]
            percentile_rows.append(
                {
                    "Player": row["PlayerName"],
                    "Team": row["TeamName"],
                    "Pos": row["PositionGroup"],
                    "Fit": row["ScoutFitScore"],
                    **{
                        label: percentile_rank(row_ref[col], float(row[col]))
                        for label, col in PIZZA_METRICS.items()
                        if col in row_ref
                    },
                }
            )
        percentile_df = pd.DataFrame(percentile_rows)
        st.dataframe(percentile_df.round(1), width="stretch", hide_index=True)

with tab_data:
    st.subheader("Market map")
    if filtered.empty:
        st.info("No players match the active filters.")
    else:
        x_axis = st.selectbox("X axis", SCORE_COLUMNS, index=SCORE_COLUMNS.index("CompositeRecruitmentScore"))
        y_axis = st.selectbox("Y axis", SCORE_COLUMNS, index=SCORE_COLUMNS.index("ValueRecruitmentScore"))
        chart_df = filtered[["PlayerName", "TeamName", "PositionGroup", "Archetype", "MarketTier", x_axis, y_axis]].dropna().rename(
            columns={x_axis: "x", y_axis: "y"}
        )
        market_chart = (
            alt.Chart(chart_df.head(1000))
            .mark_circle(size=75, opacity=0.72)
            .encode(
                x=alt.X("x:Q", title=x_axis),
                y=alt.Y("y:Q", title=y_axis),
                color=alt.Color(
                    "MarketTier:N",
                    title="Tier",
                    scale=alt.Scale(domain=list(TIER_COLORS), range=list(TIER_COLORS.values())),
                ),
                shape=alt.Shape("PositionGroup:N", title="Position"),
                tooltip=["PlayerName", "TeamName", "PositionGroup", "Archetype", "MarketTier", alt.Tooltip("x:Q", format=".1f"), alt.Tooltip("y:Q", format=".1f")],
            )
            .properties(height=460)
            .interactive()
        )
        st.altair_chart(market_chart, width="stretch")

        score_summary = (
            filtered.groupby(["PositionGroup", "Archetype"])[[c for c in SCORE_COLUMNS if c in filtered.columns]]
            .median()
            .round(1)
            .reset_index()
            .sort_values("CompositeRecruitmentScore", ascending=False)
        )
        st.dataframe(score_summary, width="stretch", hide_index=True)

with tab_data:
    st.subheader("Model data room")
    st.markdown(
        """
        <div class="note-box">
            This section checks which V3 Excel outputs are loaded, how much model context is available, and where the current board has strong or weak data coverage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    inventory = file_inventory_frame(model_metadata)
    inv_cols = st.columns(4)
    with inv_cols[0]:
        metric_card("Excel files", f"{inventory['Status'].eq('Loaded').sum()}/{len(inventory)}", "loaded into the app")
    with inv_cols[1]:
        metric_card("Recruitment rows", f"{len(model_metadata.get('recruitment', df)):,}", "main board source")
    with inv_cols[2]:
        metric_card("Model input cols", f"{len(model_metadata.get('model_input', pd.DataFrame()).columns):,}", "raw feature depth")
    with inv_cols[3]:
        smart_rows = len(model_metadata.get("smart_club", pd.DataFrame()))
        metric_card("Smart-club rows", f"{smart_rows:,}", "club-fit model links")

    st.subheader("Excel inventory")
    st.dataframe(
        inventory,
        width="stretch",
        hide_index=True,
        column_config={
            "Rows": st.column_config.NumberColumn("Rows", format="%d"),
            "Columns": st.column_config.NumberColumn("Columns", format="%d"),
        },
    )

    overview_left, overview_right = st.columns([1, 1])
    with overview_left:
        st.subheader("Run summary")
        summary_df = model_metadata.get("summary", pd.DataFrame())
        if not summary_df.empty:
            st.dataframe(summary_df, width="stretch", hide_index=True)
        else:
            run_cols = [c for c in ["ModelVersion", "RunDate", "SeasonLabel"] if c in df.columns]
            if run_cols:
                st.dataframe(df[run_cols].drop_duplicates().head(10), width="stretch", hide_index=True)
            else:
                st.info("No summary workbook or run metadata columns found.")

    with overview_right:
        st.subheader("Loaded leagues")
        leagues_df = model_metadata.get("loaded_leagues", pd.DataFrame())
        if not leagues_df.empty:
            st.dataframe(leagues_df, width="stretch", hide_index=True)
        else:
            league_cols = [c for c in ["LeagueLabel", "CountryLabel", "TierLabel", "SeasonLabel"] if c in df.columns]
            st.dataframe(df[league_cols].drop_duplicates().head(50), width="stretch", hide_index=True)

    st.subheader("Coverage by model area")
    coverage = coverage_summary_frame(df)
    coverage_chart = (
        alt.Chart(coverage)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Area:N", sort="-x", title=None),
            x=alt.X("Population:Q", title="Average populated cells (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Area", "Columns present", alt.Tooltip("Population:Q", format=".1f"), "Available columns"],
        )
        .properties(height=320)
    )
    st.altair_chart(coverage_chart, width="stretch")
    st.dataframe(
        coverage.round({"Population": 1}),
        width="stretch",
        hide_index=True,
        column_config={
            "Population": st.column_config.ProgressColumn("Population", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    dq_left, dq_right = st.columns([1, 1])
    with dq_left:
        st.subheader("Confidence and flags")
        flag_cols = [c for c in ["ConfidenceBand", "MinutesRiskFlag", "DataCoverageFlag", "AvailabilityFlag", "WageRisk", "FeeRisk"] if c in df.columns]
        if flag_cols:
            selected_flag = st.selectbox("Flag / band", flag_cols)
            flag_counts = df[selected_flag].fillna("Unknown").astype(str).value_counts().head(20).reset_index()
            flag_counts.columns = [selected_flag, "Players"]
            st.dataframe(flag_counts, width="stretch", hide_index=True)
            flag_chart = (
                alt.Chart(flag_counts)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y(f"{selected_flag}:N", sort="-x", title=None),
                    x=alt.X("Players:Q", title="Players"),
                    tooltip=[selected_flag, "Players"],
                )
                .properties(height=260)
            )
            st.altair_chart(flag_chart, width="stretch")
        else:
            st.info("No confidence or risk-flag columns found in the recruitment output.")

    with dq_right:
        st.subheader("Most incomplete columns")
        missing_df = top_missing_columns(df)
        st.dataframe(
            missing_df.round({"Missing share": 1}),
            width="stretch",
            hide_index=True,
            column_config={
                "Missing share": st.column_config.ProgressColumn("Missing", min_value=0, max_value=100, format="%.1f%%"),
            },
        )

    style_df = model_metadata.get("player_styles", pd.DataFrame())
    smart_df = model_metadata.get("smart_club", pd.DataFrame())
    score_df = model_metadata.get("player_scores", pd.DataFrame())

    detail_left, detail_right = st.columns([1, 1])
    with detail_left:
        st.subheader("Style model outputs")
        if not style_df.empty:
            style_cols = [c for c in ["PrimaryPlayerStyle", "SecondaryPlayerStyle", "ClosestArchetype", "SmartClubClosenessTier"] if c in style_df.columns]
            if style_cols:
                style_choice = st.selectbox("Style dimension", style_cols)
                style_counts = style_df[style_choice].fillna("Unknown").astype(str).value_counts().head(15).reset_index()
                style_counts.columns = [style_choice, "Players"]
                st.dataframe(style_counts, width="stretch", hide_index=True)
            st.dataframe(style_df.head(25), width="stretch", hide_index=True)
        else:
            st.info("Player Styles workbook not loaded.")

    with detail_right:
        st.subheader("Smart club closeness")
        if not smart_df.empty:
            club_summary = (
                smart_df.groupby("SmartClubModel")
                .agg(
                    Links=("PlayerName", "count"),
                    MedianCloseness=("SmartClubClosenessScore", "median"),
                    BestCloseness=("SmartClubClosenessScore", "max"),
                )
                .round(1)
                .reset_index()
                .sort_values("MedianCloseness", ascending=False)
            )
            st.dataframe(club_summary, width="stretch", hide_index=True)
            selected_club_model = st.selectbox("Inspect club model", club_summary["SmartClubModel"].tolist())
            top_club_matches = smart_df.loc[smart_df["SmartClubModel"].eq(selected_club_model)].sort_values("SmartClubClosenessScore", ascending=False).head(25)
            st.dataframe(top_club_matches, width="stretch", hide_index=True)
        else:
            st.info("Smart Club Closeness workbook not loaded.")

    st.subheader("Raw score workbook preview")
    if not score_df.empty:
        st.dataframe(score_df.head(50), width="stretch", hide_index=True)
    else:
        st.info("Player Scores workbook not loaded.")


with tab_reports:
    st.subheader("Report builder")
    export_cols = [
        c
        for c in CORE_COLUMNS
        + ["MarketTier", "Readiness", "RiskBand", "ProfileScore"]
        + SCORE_COLUMNS
        + [c for c in filtered.columns if c.endswith("_per90") or c.startswith("Imp_")]
        if c in filtered.columns
    ]
    export_df = filtered[list(dict.fromkeys(export_cols))].round(3)

    report_left, report_right = st.columns([1, 1])
    with report_left:
        report_scope = st.selectbox(
            "PDF report scope",
            ["Current filtered board", "Shortlist basket", "Single league / bundle", "Single position group", "Single player"],
        )
        top_n = st.slider("Targets to include", 10, 100, 50, step=10)
    with report_right:
        report_df = filtered.copy()
        report_title = "FCHK Scouting Report"
        scope_note = f"Current filtered board · {len(report_df):,} players"
        file_stub = "filtered_board"

        if report_scope == "Single league / bundle":
            selected_league = st.selectbox("League / bundle", bundle_groups)
            report_df = df.loc[df["BundleLabel"].astype(str).eq(selected_league)].sort_values(
                ["ScoutFitScore", "CompositeRecruitmentScore"], ascending=False
            )
            report_title = f"FCHK League Report: {selected_league}"
            scope_note = f"League report for {selected_league} · {len(report_df):,} players · weighted with {model_preset} model"
            file_stub = f"league_{selected_league}"
        elif report_scope == "Shortlist basket":
            report_df = df.loc[df["PlayerName"].isin(st.session_state.get("shortlist_players", []))].sort_values(
                ["ScoutFitScore", "CompositeRecruitmentScore"], ascending=False
            )
            report_title = "FCHK Shortlist Basket Report"
            scope_note = f"Manual shortlist · {len(report_df):,} players · weighted with {model_preset} model"
            file_stub = "shortlist_basket"
        elif report_scope == "Single position group":
            selected_position = st.selectbox("Position group", position_groups)
            report_df = df.loc[df["PositionGroup"].astype(str).eq(selected_position)].sort_values(
                ["ScoutFitScore", "CompositeRecruitmentScore"], ascending=False
            )
            report_title = f"FCHK Position Report: {selected_position}"
            scope_note = f"Position report for {selected_position} · {len(report_df):,} players · weighted with {model_preset} model"
            file_stub = f"position_{selected_position}"
        elif report_scope == "Single player":
            player_choices = (
                df.assign(_label=df["PlayerName"] + " | " + df["TeamName"] + " | " + df["PositionGroup"])
                .sort_values("ScoutFitScore", ascending=False)
            )
            selected_player = st.selectbox("Player", player_choices["_label"].tolist())
            report_df = player_choices.loc[player_choices["_label"].eq(selected_player)].drop(columns=["_label"])
            player_name = str(report_df.iloc[0]["PlayerName"]) if not report_df.empty else "Player"
            report_title = f"FCHK Player Report: {player_name}"
            scope_note = f"Player report · generated with {model_preset} model"
            file_stub = f"player_{player_name}"

        st.markdown(
            f"""
            <div class="section-card">
                <div class="metric-label">Report preview</div>
                <div class="metric-value" style="font-size:1.35rem;">{len(report_df):,} players</div>
                <div class="metric-caption">{scope_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    safe_stub = "".join(ch if ch.isalnum() else "_" for ch in file_stub.lower()).strip("_")
    download_cols = st.columns([1, 1])
    with download_cols[0]:
        st.download_button(
            "Download filtered CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=download_name("shortlist", "csv"),
            mime="text/csv",
            width="stretch",
        )
    with download_cols[1]:
        st.download_button(
            "Download PDF report",
            data=build_pdf(report_df, report_title, scope_note=scope_note, top_n=top_n),
            file_name=f"fchk_{safe_stub}_report.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )

    st.subheader("PDF report contents")
    st.markdown(
        """
        <div class="note-box">
            Each PDF includes an executive summary, position summary, top archetypes, and the ranked target list for the selected scope.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        report_df[
            [
                "PlayerName",
                "TeamName",
                "PositionGroup",
                "BundleLabel",
                "AgeYears",
                "ScoutFitScore",
                "MarketTier",
                "Archetype",
                "Readiness",
                "RiskBand",
            ]
        ]
        .head(top_n)
        .round(2),
        width="stretch",
        hide_index=True,
    )

def render_transfer_market_dashboard(df: pd.DataFrame):
    st.header("Transfer Market Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 1. Market Inventory Metrics
    total_targets = len(df[df["MarketTier"].isin(["Must scout", "Priority"])])
    u23_count = len(df[df["AgeYears"] <= 23])
    
    with col1:
        metric_card("High Priority Targets", f"{total_targets}", "Must Scout / Priority Tiers")
    with col2:
        metric_card("U23 Talent Pool", f"{u23_count}", f"{(u23_count/len(df)*100):.1f}% of Database")
    with col3:
        avg_fit = df["ScoutFitScore"].mean()
        metric_card("Avg. Scout Fit", f"{avg_fit:.1f}", "Database Mean")
    with col4:
        top_league = df.groupby("BundleLabel")["ScoutFitScore"].mean().idxmax()
        metric_card("Top Value League", shorten(top_league, 20), "Highest Avg Fit Score")

    st.divider()

    left_inner, right_inner = st.columns([2, 1])

    with left_inner:
        st.subheader("Age vs. Market Value Potential")
        # Scatter plot showing Age vs ScoutFitScore, sized by Success Probability
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X("AgeYears:Q", title="Age"),
            y=alt.Y("ScoutFitScore:Q", title="Scout Fit Score"),
            color=alt.Color("PositionGroup:N", scale=alt.Scale(domain=list(POSITION_COLORS.keys()), range=list(POSITION_COLORS.values()))),
            tooltip=["PlayerName", "TeamName", "AgeYears", "ScoutFitScore", "MarketTier"]
        ).properties(height=400).interactive()
        st.altair_chart(chart, use_container_width=True)

    with right_inner:
        st.subheader("Market Tier Breakdown")
        tier_counts = df["MarketTier"].value_counts().reset_index()
        tier_chart = alt.Chart(tier_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("MarketTier:N", scale=alt.Scale(domain=list(TIER_COLORS.keys()), range=list(TIER_COLORS.values()))),
            tooltip=["MarketTier", "count"]
        ).properties(height=400)
        st.altair_chart(tier_chart, use_container_width=True)

    # 2. Targeted "Buy" List Table
    st.subheader("Priority Acquisition List")
    priority_df = df[df["MarketTier"] == "Must scout"].sort_values("ScoutFitScore", ascending=False)
    
    st.dataframe(
        priority_df[CORE_COLUMNS + ["RiskBand", "Readiness"]],
        column_config={
            "SuccessProbability": st.column_config.ProgressColumn("Success Prob", format="%.0f%%", min_value=0, max_value=100),
            "ScoutFitScore": st.column_config.NumberColumn("Fit", format="%.1f"),
            "AgeYears": "Age"
        },
        hide_index=True,
        use_container_width=True
    )
