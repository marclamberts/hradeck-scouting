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

POSITION_PROFILES: dict[str, list[str]] = {
    "ST": ["Goalscoring striker", "Target striker", "Dynamic striker", "Second striker"],
    "W":  ["Inside forward", "Wide winger", "Pressing winger", "Creative winger"],
    "AM": ["Classic playmaker", "Shadow striker", "Deep creator", "Press orchestrator"],
    "CM": ["Box-to-box", "Progressive carrier", "Press-resistant pivot", "Defensive CM"],
    "DM": ["Defensive shield", "Regista", "Press trigger", "Holding pivot"],
    "FB": ["Attacking fullback", "Defensive fullback", "Inverted fullback", "Wing-back"],
    "CB": ["Ball-playing CB", "Aggressive stopper", "Libero / sweeper", "Aerial specialist"],
    "GK": ["Sweeper keeper", "Shot-stopper", "Commanding GK", "Ball-playing GK"],
}
# Column weights per profile — mirrors the logic in assign_player_profiles
PROFILE_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
    "ST": {
        "Goalscoring striker": {"ScoringThreatScore": 2.0, "ExpectedThreatScore": 1.5},
        "Target striker":      {"DefensiveDisruptionScore": 2.0, "ScoringThreatScore": 1.0},
        "Dynamic striker":     {"PressingScore": 2.0, "CreativeProgressionScore": 1.0},
        "Second striker":      {"CreativeProgressionScore": 2.0, "ExpectedThreatScore": 1.0},
    },
    "W": {
        "Inside forward":  {"ScoringThreatScore": 2.0, "ExpectedThreatScore": 1.5},
        "Wide winger":     {"CreativeProgressionScore": 1.5, "PressingScore": 1.0},
        "Pressing winger": {"PressingScore": 2.0, "DefensiveDisruptionScore": 1.0},
        "Creative winger": {"CreativeProgressionScore": 2.0, "DecisionScore": 1.0},
    },
    "AM": {
        "Classic playmaker":  {"CreativeProgressionScore": 2.0, "DecisionScore": 1.5},
        "Shadow striker":     {"ScoringThreatScore": 2.0, "ExpectedThreatScore": 1.5},
        "Deep creator":       {"BallSecurityScore": 2.0, "CreativeProgressionScore": 1.0},
        "Press orchestrator": {"PressingScore": 2.0, "CreativeProgressionScore": 1.0},
    },
    "CM": {
        "Box-to-box":            {"PressingScore": 1.0, "ExpectedThreatScore": 1.0, "DefensiveDisruptionScore": 1.0},
        "Progressive carrier":   {"CreativeProgressionScore": 2.0, "DecisionScore": 1.0},
        "Press-resistant pivot": {"BallSecurityScore": 2.0, "CreativeProgressionScore": 1.0},
        "Defensive CM":          {"DefensiveDisruptionScore": 2.0, "PressingScore": 1.0},
    },
    "DM": {
        "Defensive shield": {"DefensiveDisruptionScore": 2.0, "BallSecurityScore": 1.5},
        "Regista":          {"CreativeProgressionScore": 2.0, "BallSecurityScore": 1.0, "DecisionScore": 1.0},
        "Press trigger":    {"PressingScore": 2.0, "DefensiveDisruptionScore": 1.0},
        "Holding pivot":    {"BallSecurityScore": 2.0, "PerformanceReliabilityScore": 1.0},
    },
    "FB": {
        "Attacking fullback":  {"CreativeProgressionScore": 2.0, "ExpectedThreatScore": 1.5},
        "Defensive fullback":  {"DefensiveDisruptionScore": 2.0, "BallSecurityScore": 1.5},
        "Inverted fullback":   {"CreativeProgressionScore": 1.5, "BallSecurityScore": 1.5},
        "Wing-back":           {"ExpectedThreatScore": 1.5, "PressingScore": 1.5},
    },
    "CB": {
        "Ball-playing CB":    {"CreativeProgressionScore": 2.0, "BallSecurityScore": 1.5},
        "Aggressive stopper": {"DefensiveDisruptionScore": 2.0, "PressingScore": 1.5},
        "Libero / sweeper":   {"CreativeProgressionScore": 1.5, "DefensiveDisruptionScore": 1.0},
        "Aerial specialist":  {"DefensiveDisruptionScore": 3.0},
    },
    "GK": {
        "Sweeper keeper":   {"DecisionScore": 2.0, "PressingScore": 1.0},
        "Shot-stopper":     {"PerformanceReliabilityScore": 2.0, "BallSecurityScore": 1.0},
        "Commanding GK":    {"DefensiveDisruptionScore": 2.0, "BallSecurityScore": 1.0},
        "Ball-playing GK":  {"CreativeProgressionScore": 2.0, "BallSecurityScore": 1.0},
    },
}

# Maps raw Wyscout position codes → PROFILE_WEIGHTS position group keys
WYSCOUT_POSITION_MAP: dict[str, str] = {
    "CF": "ST", "SS": "ST",
    "LW": "W",  "RW": "W", "LWF": "W", "RWF": "W", "WF": "W",
    "AMF": "AM", "LAMF": "AM", "RAMF": "AM",
    "CMF": "CM", "LCM": "CM", "RCM": "CM", "LCMF": "CM", "RCMF": "CM",
    "DMF": "DM", "LDM": "DM", "RDM": "DM", "LDMF": "DM", "RDMF": "DM",
    "LB": "FB",  "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CB": "CB",  "LCB": "CB", "RCB": "CB",
    "GK": "GK",
}

# Profile weights using actual Wyscout column names
WYSCOUT_PROFILE_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
    "ST": {
        "Goalscoring striker": {
            "Non-penalty goals per 90": 2.0,
            "xG per 90": 1.5,
            "Shots on target, %": 1.0,
        },
        "Target striker": {
            "Aerial duels won, %": 2.0,
            "Duels won, %": 1.0,
            "Non-penalty goals per 90": 1.0,
        },
        "Dynamic striker": {
            "Progressive runs per 90": 2.0,
            "Successful dribbles, %": 1.5,
            "Dribbles per 90": 1.0,
        },
        "Second striker": {
            "Key passes per 90": 2.0,
            "xA per 90": 2.0,
            "Touches in box per 90": 1.0,
        },
    },
    "W": {
        "Inside forward": {
            "Non-penalty goals per 90": 2.0,
            "xG per 90": 1.5,
            "Touches in box per 90": 1.0,
        },
        "Wide winger": {
            "Crosses per 90": 1.5,
            "Accurate crosses, %": 1.0,
            "Progressive runs per 90": 1.5,
            "Dribbles per 90": 1.0,
        },
        "Pressing winger": {
            "Successful defensive actions per 90": 2.0,
            "Defensive duels per 90": 1.0,
            "Dribbles per 90": 0.5,
        },
        "Creative winger": {
            "Key passes per 90": 2.0,
            "xA per 90": 2.0,
            "Smart passes per 90": 1.0,
        },
    },
    "AM": {
        "Classic playmaker": {
            "Key passes per 90": 2.0,
            "Accurate smart passes, %": 1.5,
            "Smart passes per 90": 1.5,
        },
        "Shadow striker": {
            "Non-penalty goals per 90": 2.0,
            "xG per 90": 1.5,
            "Touches in box per 90": 1.0,
        },
        "Deep creator": {
            "Accurate passes, %": 2.0,
            "Key passes per 90": 1.5,
            "Progressive passes per 90": 1.0,
        },
        "Press orchestrator": {
            "Successful defensive actions per 90": 2.0,
            "Dribbles per 90": 1.0,
            "Key passes per 90": 1.0,
        },
    },
    "CM": {
        "Box-to-box": {
            "Successful defensive actions per 90": 1.0,
            "Progressive runs per 90": 1.0,
            "xA per 90": 0.5,
            "Passes per 90": 0.5,
        },
        "Progressive carrier": {
            "Progressive runs per 90": 2.0,
            "Accurate forward passes, %": 1.0,
            "Dribbles per 90": 1.0,
        },
        "Press-resistant pivot": {
            "Accurate passes, %": 2.0,
            "Accurate short / medium passes, %": 1.5,
            "Passes per 90": 1.0,
        },
        "Defensive CM": {
            "Successful defensive actions per 90": 2.0,
            "Defensive duels won, %": 1.5,
            "Interceptions per 90": 1.0,
        },
    },
    "DM": {
        "Defensive shield": {
            "Successful defensive actions per 90": 2.0,
            "Defensive duels won, %": 1.5,
            "Interceptions per 90": 1.0,
        },
        "Regista": {
            "Progressive passes per 90": 2.0,
            "Accurate passes, %": 1.5,
            "Key passes per 90": 1.0,
        },
        "Press trigger": {
            "Successful defensive actions per 90": 2.0,
            "Defensive duels per 90": 1.5,
            "Interceptions per 90": 1.0,
        },
        "Holding pivot": {
            "Accurate passes, %": 2.0,
            "Defensive duels won, %": 1.5,
            "Passes per 90": 1.0,
        },
    },
    "FB": {
        "Attacking fullback": {
            "Crosses per 90": 2.0,
            "xA per 90": 1.5,
            "Progressive runs per 90": 1.0,
        },
        "Defensive fullback": {
            "Successful defensive actions per 90": 2.0,
            "Defensive duels won, %": 1.5,
            "Aerial duels won, %": 1.0,
        },
        "Inverted fullback": {
            "Accurate passes, %": 1.5,
            "Progressive passes per 90": 1.5,
            "Key passes per 90": 1.0,
        },
        "Wing-back": {
            "Crosses per 90": 1.5,
            "Successful defensive actions per 90": 1.5,
            "Progressive runs per 90": 1.0,
        },
    },
    "CB": {
        "Ball-playing CB": {
            "Accurate forward passes, %": 2.0,
            "Progressive passes per 90": 1.5,
            "Accurate passes, %": 1.0,
        },
        "Aggressive stopper": {
            "Successful defensive actions per 90": 2.0,
            "Defensive duels won, %": 1.5,
            "Aerial duels won, %": 1.0,
        },
        "Libero / sweeper": {
            "Accurate passes, %": 1.5,
            "Successful defensive actions per 90": 1.5,
            "Interceptions per 90": 1.0,
        },
        "Aerial specialist": {
            "Aerial duels won, %": 3.0,
            "Aerial duels per 90": 1.0,
        },
    },
    "GK": {
        "Sweeper keeper": {
            "Exits per 90": 2.0,
            "Save rate, %": 1.0,
            "Accurate passes, %": 1.0,
        },
        "Shot-stopper": {
            "Save rate, %": 2.0,
            "Prevented goals per 90": 2.0,
        },
        "Commanding GK": {
            "Aerial duels per 90.1": 2.0,
            "Save rate, %": 1.0,
            "Exits per 90": 1.0,
        },
        "Ball-playing GK": {
            "Accurate long passes, %": 2.0,
            "Passes per 90": 1.0,
            "Save rate, %": 1.0,
        },
    },
}

# Position-specific Wyscout metric sets shown in the player profile bar chart
WYSCOUT_POS_METRICS: dict[str, list[str]] = {
    "ST": [
        "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Shots on target, %",
        "Goal conversion, %", "Touches in box per 90", "Aerial duels won, %",
        "Dribbles per 90", "Successful dribbles, %", "Progressive runs per 90",
        "xA per 90", "Duels won, %",
    ],
    "W": [
        "Non-penalty goals per 90", "xG per 90", "Assists per 90", "xA per 90",
        "Key passes per 90", "Dribbles per 90", "Successful dribbles, %",
        "Crosses per 90", "Accurate crosses, %",
        "Progressive runs per 90", "Touches in box per 90",
        "Successful defensive actions per 90",
    ],
    "AM": [
        "Key passes per 90", "xA per 90", "Assists per 90",
        "Smart passes per 90", "Accurate smart passes, %",
        "Non-penalty goals per 90", "xG per 90", "Touches in box per 90",
        "Dribbles per 90", "Progressive passes per 90",
        "Through passes per 90", "Successful dribbles, %",
    ],
    "CM": [
        "Passes per 90", "Accurate passes, %", "Forward passes per 90",
        "Accurate forward passes, %", "Progressive passes per 90",
        "Key passes per 90", "xA per 90", "Progressive runs per 90",
        "Successful defensive actions per 90", "Defensive duels won, %",
        "Interceptions per 90", "Duels won, %",
    ],
    "DM": [
        "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Interceptions per 90", "PAdj Interceptions",
        "Aerial duels won, %", "Duels won, %",
        "Passes per 90", "Accurate passes, %", "Progressive passes per 90",
        "Key passes per 90", "Fouls per 90",
    ],
    "FB": [
        "Crosses per 90", "Accurate crosses, %",
        "xA per 90", "Assists per 90", "Key passes per 90",
        "Progressive runs per 90", "Dribbles per 90",
        "Successful defensive actions per 90", "Defensive duels won, %",
        "Aerial duels won, %", "Accurate passes, %", "Progressive passes per 90",
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
# Metrics where lower raw value = better performance → invert percentile for display
WYSCOUT_LOWER_IS_BETTER: frozenset[str] = frozenset({
    "Conceded goals per 90", "Fouls per 90",
    "Yellow cards per 90", "Red cards per 90",
})

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


def assign_player_profiles(df: pd.DataFrame) -> pd.Series:
    pos         = df.get("PositionGroup", pd.Series("", index=df.index)).fillna("").astype(str)
    scoring     = safe_col(df, "ScoringThreatScore")
    expected    = safe_col(df, "ExpectedThreatScore")
    creative    = safe_col(df, "CreativeProgressionScore")
    defense     = safe_col(df, "DefensiveDisruptionScore")
    press       = safe_col(df, "PressingScore")
    security    = safe_col(df, "BallSecurityScore")
    decision    = safe_col(df, "DecisionScore")
    reliability = safe_col(df, "PerformanceReliabilityScore")

    _spec: dict[str, dict[str, pd.Series]] = {
        "ST": {
            "Goalscoring striker": scoring  * 2   + expected  * 1.5,
            "Target striker":      defense  * 2   + scoring   * 1.0,
            "Dynamic striker":     press    * 2   + creative  * 1.0,
            "Second striker":      creative * 2   + expected  * 1.0,
        },
        "W": {
            "Inside forward":  scoring  * 2   + expected  * 1.5,
            "Wide winger":     creative * 1.5 + press     * 1.0,
            "Pressing winger": press    * 2   + defense   * 1.0,
            "Creative winger": creative * 2   + decision  * 1.0,
        },
        "AM": {
            "Classic playmaker":  creative * 2   + decision  * 1.5,
            "Shadow striker":     scoring  * 2   + expected  * 1.5,
            "Deep creator":       security * 2   + creative  * 1.0,
            "Press orchestrator": press    * 2   + creative  * 1.0,
        },
        "CM": {
            "Box-to-box":            press    * 1.0 + expected * 1.0 + defense * 1.0,
            "Progressive carrier":   creative * 2   + decision * 1.0,
            "Press-resistant pivot": security * 2   + creative * 1.0,
            "Defensive CM":          defense  * 2   + press    * 1.0,
        },
        "DM": {
            "Defensive shield": defense  * 2   + security   * 1.5,
            "Regista":          creative * 2   + security   * 1.0 + decision * 1.0,
            "Press trigger":    press    * 2   + defense    * 1.0,
            "Holding pivot":    security * 2   + reliability * 1.0,
        },
        "FB": {
            "Attacking fullback":  creative * 2   + expected  * 1.5,
            "Defensive fullback":  defense  * 2   + security  * 1.5,
            "Inverted fullback":   creative * 1.5 + security  * 1.5,
            "Wing-back":           expected * 1.5 + press     * 1.5,
        },
        "CB": {
            "Ball-playing CB":    creative * 2   + security * 1.5,
            "Aggressive stopper": defense  * 2   + press    * 1.5,
            "Libero / sweeper":   creative * 1.5 + defense  * 1.0,
            "Aerial specialist":  defense  * 3,
        },
        "GK": {
            "Sweeper keeper":   decision    * 2 + press    * 1.0,
            "Shot-stopper":     reliability * 2 + security * 1.0,
            "Commanding GK":    defense     * 2 + security * 1.0,
            "Ball-playing GK":  creative    * 2 + security * 1.0,
        },
    }

    result = pd.Series("", index=df.index)
    for pos_code, profile_scores in _spec.items():
        mask = pos.eq(pos_code)
        if not mask.any():
            continue
        scores_df = pd.DataFrame(profile_scores, index=df.index)
        result.loc[mask] = scores_df.loc[mask].idxmax(axis=1)
    return result


def calc_profile_fit(df: pd.DataFrame, position: str, profile: str) -> pd.Series:
    """Return a 0–100 z-score-based profile fit for every row in df.
    Scores are normalised within the position group so 50 = average,
    65 = one std above average, 35 = one std below."""
    weights = PROFILE_WEIGHTS.get(position, {}).get(profile, {})
    if not weights:
        return pd.Series(50.0, index=df.index)

    raw = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        raw = raw + safe_col(df, col) * w

    # z-score within the position pool
    pos_mask = df.get("PositionGroup", pd.Series("", index=df.index)).astype(str).eq(position)
    pos_raw = raw.loc[pos_mask]
    if len(pos_raw) < 2 or pos_raw.std() < 1e-6:
        return pd.Series(50.0, index=df.index)

    mean, std = pos_raw.mean(), pos_raw.std()
    z = (raw - mean) / std
    return (z * 15 + 50).clip(0, 100).round(1)


def calc_wyscout_profile_fit(df: pd.DataFrame, pos_group: str, profile: str) -> pd.Series:
    """Z-score profile fit using raw Wyscout column names. 50 = avg, 65 = top 16%, 80 = top 2%."""
    weights = WYSCOUT_PROFILE_WEIGHTS.get(pos_group, {}).get(profile, {})
    if not weights:
        return pd.Series(50.0, index=df.index)
    raw = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        raw = raw + safe_col(df, col) * w
    pos_mask = df.get("_PosGroup", pd.Series("", index=df.index)).astype(str).eq(pos_group)
    pos_raw = raw.loc[pos_mask]
    if len(pos_raw) < 2 or pos_raw.std() < 1e-6:
        return pd.Series(50.0, index=df.index)
    mean, std = pos_raw.mean(), pos_raw.std()
    z = (raw - mean) / std
    return (z * 15 + 50).clip(0, 100).round(1)


@st.cache_data(show_spinner=False, ttl=600)
def _load_link_db() -> pd.DataFrame:
    p = APP_DIR / "data" / "IMPECT_Wyscout_Link.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600)
def _load_wyscout_index() -> pd.DataFrame:
    p = APP_DIR / "data" / "Wyscout_Player_Index.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _render_wyscout_bars(
    ws_row: pd.Series,
    pos_pool: pd.DataFrame,
    pos_group: str,
    ws_name: str,
    ws_team: str,
    ws_file: str,
) -> "plt.Figure | None":
    """FBref-style horizontal percentile bars for Wyscout raw metrics."""
    all_metrics = WYSCOUT_POS_METRICS.get(pos_group, [])
    metrics = [m for m in all_metrics if m in ws_row.index and m in pos_pool.columns]
    if not metrics:
        return None

    n = len(metrics)
    fig_h = max(4.0, n * 0.56 + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    pcts: list[float | None] = []
    raw_strs: list[str] = []
    for m in metrics:
        val = pd.to_numeric(ws_row.get(m), errors="coerce")
        if pd.isna(val):
            pcts.append(None)
            raw_strs.append("—")
        else:
            p = percentile_rank(pd.to_numeric(pos_pool[m], errors="coerce"), float(val))
            if m in WYSCOUT_LOWER_IS_BETTER:
                p = 100.0 - p
            pcts.append(p)
            raw_strs.append(f"{val:.2f}")

    # y=n-1 → first metric (top), y=0 → last metric (bottom)
    y_pos = list(range(n - 1, -1, -1))

    for y, m, pct, raw in zip(y_pos, metrics, pcts, raw_strs):
        ax.barh(y, 100, height=0.62, color="#161b22", zorder=1)
        if pct is not None:
            clr = "#0d9e7d" if pct >= 67 else ("#f4a261" if pct >= 34 else "#e76f51")
            ax.barh(y, pct, height=0.62, color=clr, zorder=2, alpha=0.88)
            x_txt = max(pct - 1.5, 2.0)
            ha = "right" if pct > 8 else "left"
            ax.text(x_txt, y, f"{pct:.0f}", va="center", ha=ha,
                    fontsize=8, color="#ffffff", fontweight="bold", zorder=3)
        ax.text(102, y, raw, va="center", ha="left", fontsize=7.5, color="#8b949e", zorder=3)

    ax.axvline(50, color="#30363d", linewidth=1.2, linestyle="--", zorder=3)
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([m for m in reversed(metrics)], fontsize=8, color="#c9d1d9")
    ax.set_xlim(0, 116)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=7, color="#6e7681")
    ax.tick_params(axis="x", length=3, color="#30363d")
    ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="x", color="#21262d", linewidth=0.5, zorder=0)

    pool_n = len(pos_pool)
    fig.text(0.02, 0.98, f"Wyscout · {ws_file.replace('.xlsx','')}", ha="left", va="top",
             fontsize=10, fontweight="bold", color="#e6edf3")
    fig.text(0.02, 0.945, f"{pos_group} · percentile vs {pool_n} position peers · 50 = average",
             ha="left", va="top", fontsize=8, color="#8b949e")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


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


def _role_score_vec(df: pd.DataFrame) -> pd.Series:
    pos = df.get("PositionGroup", pd.Series("CM", index=df.index)).fillna("CM").astype(str)
    result = pd.Series(50.0, index=df.index)
    for pos_code, weights in {**ROLE_SCORE_WEIGHTS, "_default": ROLE_SCORE_WEIGHTS["CM"]}.items():
        mask = pos.eq(pos_code) if pos_code != "_default" else ~pos.isin(ROLE_SCORE_WEIGHTS)
        if not mask.any():
            continue
        sub = df.loc[mask]
        pos_total = sum(v for v in weights.values() if v > 0)
        neg_total = sum(abs(v) for v in weights.values() if v < 0)
        denom = max(pos_total + neg_total, 1)
        raw = pd.Series(0.0, index=sub.index)
        for col, weight in weights.items():
            vals = pd.to_numeric(sub.get(col, pd.Series(0, index=sub.index)), errors="coerce").fillna(0)
            if weight < 0:
                raw += np.maximum(0.0, 100 - vals * 6) * abs(weight)
            else:
                raw += vals * weight
        result.loc[mask] = np.clip(raw / denom, 0, 100)
    return result


def _drivers_vec(df: pd.DataFrame, col_map: dict[str, str], limit: int = 3) -> pd.Series:
    labels = list(col_map.keys())
    mat = np.column_stack([
        pd.to_numeric(df.get(col, pd.Series(0, index=df.index)), errors="coerce").fillna(0).values
        for col in col_map.values()
    ])
    top_idx    = np.argsort(-mat, axis=1)[:, :limit]
    top_scores = mat[np.arange(len(mat))[:, None], top_idx]
    top_labels = np.array(labels)[top_idx]
    return pd.Series(
        [", ".join(f"{lbl} {sc:.0f}" for lbl, sc in zip(rl, rs)) for rl, rs in zip(top_labels, top_scores)],
        index=df.index,
    )


def _risk_flags_vec(df: pd.DataFrame) -> pd.Series:
    names  = ["low minutes", "low reliability", "high security risk", "own-half losses", "danger losses", "high-possession context"]
    bools  = np.column_stack([
        safe_col(df, "MinutesPlayed").fillna(0).lt(900).values,
        safe_col(df, "PerformanceReliabilityScore").fillna(100).lt(55).values,
        safe_col(df, "SecurityRisk_per90").fillna(0).gt(13).values,
        safe_col(df, "OwnHalfLosses_per90").fillna(0).gt(3).values,
        safe_col(df, "DangerOwnHalfLosses_per90").fillna(0).gt(0.5).values,
        df.get("TeamPossProxy", pd.Series(0.0, index=df.index)).fillna(0).gt(1.25).values,
    ])
    result = []
    for row in bools:
        active = [names[i] for i in np.where(row)[0]]
        result.append(", ".join(active) if active else "clean profile")
    return pd.Series(result, index=df.index)


@st.cache_data(show_spinner=False)
def _add_static_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Archetype"]    = assign_archetypes(out)
    out["RoleFitScore"] = _role_score_vec(out)
    out["AgeYears"]     = safe_col(out, "AgeYears").round(1)
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
    out["FitDrivers"]     = _drivers_vec(out, {
        "Composite": "CompositeRecruitmentScore", "Decision": "DecisionScore",
        "Value": "ValueRecruitmentScore", "Scoring": "ScoringThreatScore",
        "Creation": "CreativeProgressionScore", "Defense": "DefensiveDisruptionScore",
        "Pressing": "PressingScore", "Security": "BallSecurityScore",
        "xThreat": "ExpectedThreatScore", "Reliability": "PerformanceReliabilityScore",
    })
    out["QualityDrivers"] = _drivers_vec(out, {
        "Role": "RoleFitScore", "Impact": "ProfileScore", "Decision": "DecisionScore",
        "Reliability": "PerformanceReliabilityScore", "Threat": "ExpectedThreatScore",
        "Security": "BallSecurityScore", "Creation": "CreativeProgressionScore",
        "Defense": "DefensiveDisruptionScore", "Pressing": "PressingScore",
    })
    out["RiskFlags"]      = _risk_flags_vec(out)
    out["PlayerProfile"]  = assign_player_profiles(out)
    return out


@st.cache_data(show_spinner=False)
def add_scouting_fields(df: pd.DataFrame, weights: dict[str, int]) -> pd.DataFrame:
    out = _add_static_fields(df)
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
    out["ScoutFitScore"] = ((out["ModelFitScore"] * 0.58) + (out["RoleFitScore"] * 0.42)).clip(0, 100)
    out["MarketTier"] = pd.cut(
        out["ScoutFitScore"],
        bins=[-np.inf, 42, 50, 58, 66, np.inf],
        labels=["Watch", "Depth", "Shortlist", "Priority", "Must scout"],
    ).astype(str)
    out["TierReason"] = (
        out["MarketTier"].astype(str)
        + " because Scout Fit is "
        + out["ScoutFitScore"].round(1).astype(str)
        + ", driven by "
        + out["FitDrivers"].astype(str)
        + "."
    )
    return out


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
    fig.patch.set_facecolor("#f5f7fa")
    ax.set_facecolor("#ffffff")
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=order, showfliers=False,
                    medianprops={"color": "#0d9e7d", "linewidth": 2},
                    whiskerprops={"color": "#c8d3df"}, capprops={"color": "#c8d3df"})
    for patch, pos in zip(bp["boxes"], order):
        patch.set_facecolor(POSITION_COLORS.get(pos, "#457b9d"))
        patch.set_alpha(0.75)
        patch.set_edgecolor("#c8d3df")
    ax.set_title(f"{metric} by position", loc="left", fontsize=13, fontweight="bold", color="#1a2332")
    ax.set_ylabel("Score", color="#4a5e75", fontsize=9)
    ax.tick_params(colors="#4a5e75")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dde3ec")
    ax.grid(axis="y", color="#dde3ec", linewidth=0.7)
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
        "profiles_filter",
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


WORKSPACES = ["Recruitment", "Scouting", "Search", "Goalkeepers", "Team", "Model"]
_WORKSPACE_ICONS = {
    "Recruitment": ("🎯", "Rankings & cases"),
    "Scouting":    ("🔍", "Wyscout database"),
    "Search":      ("🔎", "Player profile search"),
    "Goalkeepers": ("🧤", "GK boards"),
    "Team":        ("🏟", "Squad & Czech market"),
    "Model":       ("🤖", "Smart club model"),
}
WYSCOUT_DB_DIR = APP_DIR / "data" / "Wyscout DB"


def set_workspace(section: str) -> None:
    st.session_state["active_workspace"] = section
    st.session_state["show_scouting_workspace"] = True
    st.session_state.pop("landing_notice", None)


def enter_scouting_workspace() -> None:
    set_workspace("Recruitment")


def render_workspace_nav(location: str = "top") -> None:
    active = st.session_state.get("active_workspace", "Recruitment")
    brand_col, *nav_col_list = st.columns([3] + [1] * len(WORKSPACES), gap="small")
    with brand_col:
        st.markdown(
            "<div class='app-nav-brand'>"
            "<span class='app-nav-emoji'>⚽</span>"
            "<div><div class='app-nav-name'>FCHK Scouting IQ</div>"
            "<div class='app-nav-tagline'>Hradec Králové · Analytics</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
    for idx, section in enumerate(WORKSPACES):
        icon, desc = _WORKSPACE_ICONS.get(section, ("", ""))
        is_active = active == section
        with nav_col_list[idx]:
            st.button(
                f"{icon}  {section}",
                key=f"workspace_{location}_{section}",
                type="primary" if is_active else "secondary",
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


def _first_position(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first listed position when Wyscout stores comma-separated values."""
    for col in ["Position", "PositionGroup", "Pos", "position", "pos"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.split(r"[,;]")
                .str[0]
                .str.strip()
                .replace("nan", "")
            )
    return df


def _load_wyscout_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            return _first_position(_clean_columns(pd.read_excel(path)))
        except Exception:
            xl = pd.ExcelFile(path)
            frames = []
            for sheet in xl.sheet_names:
                try:
                    frames.append(_first_position(_clean_columns(pd.read_excel(xl, sheet_name=sheet))))
                except Exception:
                    pass
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elif suffix == ".csv":
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return _first_position(_clean_columns(pd.read_csv(path, encoding=enc)))
            except Exception:
                continue
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=300)
def _load_wyscout_tier(paths_and_labels: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """Load and combine multiple Wyscout files into one DataFrame.
    Adds a _League column so the result can be filtered by league."""
    frames = []
    for path_str, label in paths_and_labels:
        df = _load_wyscout_file(Path(path_str))
        if not df.empty:
            df["_League"] = label
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_leagues_overview() -> pd.DataFrame:
    path = APP_DIR / "data" / "Leagues Overview.xlsx"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "Division" in df.columns:
        df["Division"] = df["Division"].astype(str)
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
    leagues_df = load_leagues_overview()
    wyscout_files = sorted(
        [p for p in WYSCOUT_DB_DIR.glob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("~")],
        key=lambda p: p.name.lower(),
    )

    file_meta: dict[str, pd.Series | None] = {p.name: _match_league(p.name, leagues_df) for p in wyscout_files}

    def _tier_of(fname: str) -> str:
        m = file_meta.get(fname)
        return str(m["Tier Label"]) if m is not None and pd.notna(m.get("Tier Label")) else "Unknown"

    def _file_label(p: Path) -> str:
        m = file_meta.get(p.name)
        if m is not None and pd.notna(m.get("League Name")):
            return str(m["League Name"])
        return p.stem

    # Initialise filter state so all variables are defined before the sidebar block
    sel_leagues: list[str] = []
    sel_pos: list[str] = []
    sel_teams: list[str] = []
    sel_age: tuple[float, float] | None = None
    ws_search: str = ""
    sort_col_sel: str = "Default"
    _ws_numeric_cols: list[str] = []
    _ws_text_cols: list[str] = []
    _player_col = _team_col = _pos_col = _age_col = None
    ws_df: pd.DataFrame = pd.DataFrame()
    sel_tier: str = "All"
    visible_files: list = []

    with st.sidebar:
        st.markdown(
            "<div class='sidebar-brand'><div class='sidebar-brand-icon'>🔍</div>"
            "<div><div class='sidebar-brand-title'>Scouting</div>"
            "<div class='sidebar-brand-meta'>Wyscout DB browser</div></div></div>",
            unsafe_allow_html=True,
        )

        if wyscout_files:
            st.markdown("<div class='sbar-hdr'>Tier</div>", unsafe_allow_html=True)
            all_tiers = ["All", "Elite", "Top", "Strong", "Developing", "Lower", "Youth/Grassroots"]
            sel_tier = st.selectbox("Tier", all_tiers, key="ws_tier_filter", label_visibility="collapsed")

            visible_files = [p for p in wyscout_files if _tier_of(p.name) == sel_tier] if sel_tier != "All" else wyscout_files

            if visible_files:
                # Build the combined dataset for this tier (cached by the set of files)
                paths_and_labels = tuple((str(p), _file_label(p)) for p in visible_files)
                with st.spinner(f"Loading {len(visible_files)} league file{'s' if len(visible_files) != 1 else ''}…"):
                    ws_df = _load_wyscout_tier(paths_and_labels)

                if not ws_df.empty:
                    _ws_numeric_cols = [c for c in ws_df.select_dtypes(include=[np.number]).columns if c != "_League"]
                    _ws_text_cols    = [c for c in ws_df.select_dtypes(exclude=[np.number]).columns if c != "_League"]
                    _player_col = next((c for c in ["Player", "PlayerName", "Name", "player", "name"] if c in ws_df.columns), None)
                    _team_col   = next((c for c in ["Team", "TeamName", "Club", "team", "club"] if c in ws_df.columns), None)
                    _pos_col    = next((c for c in ["Position", "PositionGroup", "Pos", "position", "pos"] if c in ws_df.columns), None)
                    _age_col    = next((c for c in ["Age", "AgeYears", "age"] if c in ws_df.columns), None)
                    # Add position group mapping column for profile search
                    if _pos_col and "_PosGroup" not in ws_df.columns:
                        _pg = ws_df[_pos_col].astype(str).str.strip().map(WYSCOUT_POSITION_MAP).fillna("").rename("_PosGroup")
                        ws_df = pd.concat([ws_df, _pg], axis=1)

                    st.markdown("<div class='sbar-hdr'>Filters</div>", unsafe_allow_html=True)
                    ws_search = st.text_input("Search", key="ws_search", placeholder="Player or team…", label_visibility="collapsed")

                    league_opts = sorted(ws_df["_League"].dropna().astype(str).unique())
                    if len(league_opts) > 1:
                        sel_leagues = st.multiselect("League", league_opts, default=[], placeholder="All leagues", key="ws_league_filter")

                    if _pos_col:
                        pos_opts = sorted(ws_df[_pos_col].replace("", pd.NA).dropna().astype(str).unique())
                        sel_pos = st.multiselect("Position", pos_opts, default=[], placeholder="All positions", key="ws_pos_filter")

                    if _team_col:
                        team_opts = sorted(ws_df[_team_col].dropna().astype(str).unique())
                        sel_teams = st.multiselect("Team", team_opts, default=[], placeholder="All teams", key="ws_team_filter")

                    if _age_col:
                        _age_s = pd.to_numeric(ws_df[_age_col], errors="coerce").dropna()
                        if not _age_s.empty:
                            _amin = float(np.floor(_age_s.min()))
                            _amax = float(np.ceil(_age_s.max()))
                            if _amin < _amax:
                                sel_age = st.slider("Age", _amin, _amax, (_amin, _amax), step=1.0, key="ws_age_filter")

                    if _ws_numeric_cols:
                        st.markdown("<div class='sbar-hdr'>Sort by</div>", unsafe_allow_html=True)
                        sort_col_sel = st.selectbox("Sort column", ["Default"] + _ws_numeric_cols, key="ws_sort_col", label_visibility="collapsed")

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        "<div class='page-header'>"
        "<div class='page-header-icon'>🔍</div>"
        "<div><div class='page-header-title'>Scouting</div>"
        "<div class='page-header-sub'>Wyscout data browser — search and filter imported league files</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not wyscout_files:
        st.markdown(
            '<div class="note-box" style="border-left-color:var(--amber);">'
            '<strong style="color:var(--amber);">No Wyscout files found.</strong><br>'
            f'Upload your exported Wyscout Excel or CSV files to '
            f'<code style="color:var(--teal);background:rgba(13,158,125,.08);padding:2px 6px;border-radius:4px;">data/Wyscout DB/</code>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if ws_df.empty:
        st.info("No data loaded. Select a tier in the sidebar.")
        return

    # ── Apply sidebar filters (shared across both tabs) ───────────────────────
    ws_filtered = ws_df.copy()
    if sel_leagues:
        ws_filtered = ws_filtered.loc[ws_filtered["_League"].isin(sel_leagues)]
    if sel_pos and _pos_col:
        ws_filtered = ws_filtered.loc[ws_filtered[_pos_col].astype(str).isin(sel_pos)]
    if sel_teams and _team_col:
        ws_filtered = ws_filtered.loc[ws_filtered[_team_col].astype(str).isin(sel_teams)]
    if sel_age and _age_col:
        ws_filtered = ws_filtered.loc[pd.to_numeric(ws_filtered[_age_col], errors="coerce").between(sel_age[0], sel_age[1])]
    if ws_search:
        _hcols = [c for c in [_player_col, _team_col] if c]
        if _hcols:
            _hay = ws_filtered[_hcols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
            ws_filtered = ws_filtered.loc[_hay.str.contains(ws_search.lower(), regex=False)]

    ws_browse_tab, ws_profile_tab = st.tabs(["🔍 Browse", "🎭 Profile Search"])

    with ws_browse_tab:
        # ── Status bar ────────────────────────────────────────────────────────
        _n_leagues = ws_filtered["_League"].nunique()
        _league_summary = f"{_n_leagues} league{'s' if _n_leagues != 1 else ''}"
        row_meta_c, dl_c = st.columns([5, 1], gap="small")
        with row_meta_c:
            st.markdown(
                f'<div style="color:var(--faint);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;padding-top:6px;">'
                f'<span style="color:var(--ink);">{len(ws_filtered):,}</span> / {len(ws_df):,} players'
                f'&nbsp;·&nbsp;{_league_summary}&nbsp;·&nbsp;{sel_tier}</div>',
                unsafe_allow_html=True,
            )
        with dl_c:
            st.download_button(
                "⬇ CSV",
                data=ws_filtered.drop(columns=["_League", "_PosGroup"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name=f"wyscout_{sel_tier.lower().replace(' ','_')}_filtered.csv",
                mime="text/csv",
                width="stretch",
            )

        # ── Main table ────────────────────────────────────────────────────────
        display_cols = [c for c in ws_filtered.columns if c not in ("_League", "_PosGroup")]
        if ws_filtered["_League"].nunique() > 1:
            display_cols = ["_League"] + display_cols
            ws_display_df = ws_filtered[display_cols].rename(columns={"_League": "League"})
        else:
            ws_display_df = ws_filtered[display_cols]

        if sort_col_sel and sort_col_sel != "Default" and sort_col_sel in ws_display_df.columns:
            ws_display_df = ws_display_df.sort_values(sort_col_sel, ascending=False).reset_index(drop=True)
        else:
            ws_display_df = ws_display_df.reset_index(drop=True)

        col_config: dict = {}
        for c in _ws_numeric_cols:
            if c in ws_display_df.columns:
                _cmin = float(ws_display_df[c].min()) if not ws_display_df[c].isna().all() else 0.0
                _cmax = float(ws_display_df[c].max()) if not ws_display_df[c].isna().all() else 100.0
                if _cmax > _cmin and 0 <= _cmin and _cmax <= 100:
                    col_config[c] = st.column_config.ProgressColumn(c, min_value=_cmin, max_value=_cmax, format="%.2f")

        st.dataframe(ws_display_df, width="stretch", hide_index=True, height=780, column_config=col_config)

        # ── Column explorer ───────────────────────────────────────────────────
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
                                .mark_bar(color="#12c799", opacity=0.7, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                                .encode(
                                    x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=_explore_col,
                                            axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e", gridColor="#21262d")),
                                    y=alt.Y("count():Q", title="Players", axis=alt.Axis(labelColor="#8b949e", gridColor="#21262d")),
                                )
                                .properties(height=220)
                                .configure_view(fill="#161b22", stroke=None).configure(background="#0d1117")
                            )
                            st.altair_chart(_hist_chart, width="stretch")
                        with exp_right:
                            _stats = _series.describe()
                            _stats.index = ["Count", "Mean", "Std", "Min", "P25", "Median", "P75", "Max"]
                            st.dataframe(_stats.reset_index().rename(columns={"index": "Stat", 0: "Value"}).round(2),
                                         width="stretch", hide_index=True)

        # ── League index ──────────────────────────────────────────────────────
        if not leagues_df.empty:
            with st.expander("🌍 League index", expanded=False):
                present_names: set[str] = set()
                for p in wyscout_files:
                    m = file_meta.get(p.name)
                    if m is not None:
                        present_names.add(str(m.get("League Name", "")))
                lo_view = leagues_df.copy()
                lo_view["✓"] = lo_view["League Name"].isin(present_names).map({True: "✓", False: ""})
                lo_view["Division"] = lo_view["Division"].astype(str)
                li_tier = st.selectbox("Tier", ["All"] + sorted(lo_view["Tier Label"].dropna().unique().tolist()), key="ws_li_tier")
                if li_tier != "All":
                    lo_view = lo_view.loc[lo_view["Tier Label"].eq(li_tier)]
                st.dataframe(
                    lo_view[["League Name", "Country", "Tier Label", "Division", "✓"]],
                    width="stretch", hide_index=True, height=420,
                )

    with ws_profile_tab:
        # ── Profile target selectors ─────────────────────────────────────────
        # Show actual Wyscout position codes present in the data, filtered to
        # those that map to a known profile group
        if _pos_col and not ws_df.empty:
            _ws_all_raw_pos = sorted(
                p for p in ws_df[_pos_col].astype(str).str.strip().unique().tolist()
                if isinstance(p, str) and p and p != "nan"
            )
            _ws_mapped_pos = [p for p in _ws_all_raw_pos if WYSCOUT_POSITION_MAP.get(p)]
        else:
            _ws_mapped_pos = []

        _wppos_col, _wppro_col = st.columns([1, 2], gap="small")
        with _wppos_col:
            _ws_raw_pos_sel = st.selectbox(
                "Position", ["—"] + _ws_mapped_pos,
                key="ws_profile_pos", label_visibility="visible",
            )
        # Resolve raw Wyscout position → profile group
        _ws_target_pos = WYSCOUT_POSITION_MAP.get(_ws_raw_pos_sel, "") if _ws_raw_pos_sel != "—" else ""

        with _wppro_col:
            _ws_prof_options = list(WYSCOUT_PROFILE_WEIGHTS.get(_ws_target_pos, {}).keys()) if _ws_target_pos else []
            _ws_target_profile = st.selectbox(
                "Profile", ["—"] + _ws_prof_options,
                key="ws_profile_name", disabled=not _ws_target_pos, label_visibility="visible",
            )

        if not _ws_target_pos or _ws_target_profile == "—":
            st.markdown(
                "<div class='note-box'>Select a <strong>position</strong> and a <strong>profile</strong> above "
                "to rank every player in that position by how well they fit the profile, "
                "using z-scores calculated from Wyscout metrics within the position pool.</div>",
                unsafe_allow_html=True,
            )
        else:
            # Calculate ProfileFit on full ws_df so z-scores use the complete position pool
            _ws_scored = ws_df.copy()
            if "_PosGroup" not in _ws_scored.columns and _pos_col:
                _pg2 = _ws_scored[_pos_col].astype(str).str.strip().map(WYSCOUT_POSITION_MAP).fillna("").rename("_PosGroup")
                _ws_scored = pd.concat([_ws_scored, _pg2], axis=1)
            _ws_scored["ProfileFit"] = calc_wyscout_profile_fit(_ws_scored, _ws_target_pos, _ws_target_profile)

            # Apply sidebar filters (except position — the position selector replaces that)
            if sel_leagues:
                _ws_scored = _ws_scored.loc[_ws_scored["_League"].isin(sel_leagues)]
            if sel_teams and _team_col:
                _ws_scored = _ws_scored.loc[_ws_scored[_team_col].astype(str).isin(sel_teams)]
            if sel_age and _age_col:
                _ws_scored = _ws_scored.loc[pd.to_numeric(_ws_scored[_age_col], errors="coerce").between(sel_age[0], sel_age[1])]
            if ws_search:
                _hcols = [c for c in [_player_col, _team_col] if c]
                if _hcols:
                    _hay = _ws_scored[_hcols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                    _ws_scored = _ws_scored.loc[_hay.str.contains(ws_search.lower(), regex=False)]

            # Show players whose raw position maps to the selected group
            _ws_pos_pool = _ws_scored.loc[_ws_scored["_PosGroup"].eq(_ws_target_pos)].copy()

            if _ws_pos_pool.empty:
                st.info(f"No {_ws_raw_pos_sel} players found with current filters.")
            else:
                _ws_pos_pool = _ws_pos_pool.sort_values("ProfileFit", ascending=False)

                # Profile driver strip
                _ws_pw = WYSCOUT_PROFILE_WEIGHTS[_ws_target_pos][_ws_target_profile]
                _ws_drivers = " · ".join(f"{k} ×{v}" for k, v in _ws_pw.items())
                _ws_avail_drivers = [k for k in _ws_pw if k in _ws_pos_pool.columns]
                _ws_missing = [k for k in _ws_pw if k not in _ws_pos_pool.columns]
                _ws_note = (
                    f'<br><span style="color:var(--amber);font-size:.65rem;">Missing columns (scored as 0): '
                    f'{", ".join(_ws_missing)}</span>' if _ws_missing else ""
                )
                # Count how many position codes are included in this group
                _ws_group_codes = [k for k, v in WYSCOUT_POSITION_MAP.items() if v == _ws_target_pos]
                _ws_pool_size = len(ws_df.loc[ws_df.get("_PosGroup", pd.Series("", index=ws_df.index)).eq(_ws_target_pos)])
                st.markdown(
                    f"<div class='note-box'>"
                    f"<strong style='color:var(--teal-hi);'>{_ws_target_profile}</strong> "
                    f"<span style='color:var(--muted);'>{_ws_raw_pos_sel} · {_ws_target_pos} group "
                    f"({', '.join(_ws_group_codes)}) · z-scores vs {_ws_pool_size:,} players</span>"
                    f"<br><span style='color:var(--faint);font-size:.7rem;'>Drivers: {_ws_drivers}</span>"
                    f"{_ws_note}"
                    f"<br><span style='color:var(--faint);font-size:.7rem;'>"
                    f"50 = position average · 65 = top 16% · 80 = top 2%"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

                # Build display: id cols + league + age + raw position + ProfileFit + driver cols
                _ws_id_cols = [c for c in [_player_col, _team_col] if c]
                _ws_show = _ws_id_cols[:]
                if "_League" in _ws_pos_pool.columns and _ws_pos_pool["_League"].nunique() > 1:
                    _ws_show.append("_League")
                if _age_col and _age_col in _ws_pos_pool.columns:
                    _ws_show.append(_age_col)
                if _pos_col and _pos_col in _ws_pos_pool.columns:
                    _ws_show.append(_pos_col)
                _ws_show.append("ProfileFit")
                for _wsc in _ws_avail_drivers:
                    if _wsc not in _ws_show:
                        _ws_show.append(_wsc)

                _ws_prof_board = (
                    _ws_pos_pool[[c for c in _ws_show if c in _ws_pos_pool.columns]]
                    .rename(columns={"_League": "League"})
                    .reset_index(drop=True)
                )

                _ws_prof_cfg: dict = {
                    "ProfileFit": st.column_config.ProgressColumn("Profile Fit ▼", min_value=0, max_value=100, format="%.1f"),
                }
                for _wsc in _ws_avail_drivers:
                    _cmin = float(_ws_prof_board[_wsc].min()) if _wsc in _ws_prof_board.columns and not _ws_prof_board[_wsc].isna().all() else 0.0
                    _cmax = float(_ws_prof_board[_wsc].max()) if _wsc in _ws_prof_board.columns and not _ws_prof_board[_wsc].isna().all() else 1.0
                    if _cmax > _cmin:
                        _ws_prof_cfg[_wsc] = st.column_config.ProgressColumn(_wsc, min_value=_cmin, max_value=_cmax, format="%.2f")

                st.dataframe(_ws_prof_board.round(2), width="stretch", hide_index=True, height=780, column_config=_ws_prof_cfg)


def _norm_search(s: object) -> str:
    """Accent-stripped lowercase for robust name search."""
    import unicodedata as _ud
    return _ud.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower()


def render_search_workspace(data: pd.DataFrame) -> None:
    """Cross-database player search with IMPECT pizza + Wyscout percentile bar profile."""
    link_db = _load_link_db()
    ws_idx  = _load_wyscout_index()

    with st.sidebar:
        st.markdown(
            "<div class='sidebar-brand'><div class='sidebar-brand-icon'>🔎</div>"
            "<div><div class='sidebar-brand-title'>Player Search</div>"
            "<div class='sidebar-brand-meta'>IMPECT + Wyscout combined</div></div></div>",
            unsafe_allow_html=True,
        )
        _ws_idx_count = len(ws_idx) if not ws_idx.empty else 0
        st.markdown(
            f"<div style='color:var(--faint);font-size:.68rem;padding:6px 0 0 4px;'>"
            f"IMPECT: <strong style='color:var(--ink);'>{len(data):,}</strong> players · "
            f"Wyscout: <strong style='color:var(--ink);'>{_ws_idx_count:,}</strong> players</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='page-header'>"
        "<div class='page-header-icon'>🔎</div>"
        "<div><div class='page-header-title'>Player Search</div>"
        "<div class='page-header-sub'>Search across IMPECT model data and Wyscout — click a result for a full profile</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    search_q = st.text_input(
        "Search",
        placeholder="Type a player name (min 2 characters)…",
        key="player_search_q",
        label_visibility="collapsed",
    )
    q = search_q.strip()

    if len(q) < 2:
        st.markdown(
            "<div class='note-box'>Start typing a player name above. "
            "Results come from <strong>IMPECT model data</strong> and the <strong>Wyscout database</strong> "
            "(68 000+ players). Wyscout names use abbreviated format — search by last name for best results.</div>",
            unsafe_allow_html=True,
        )
        return

    q_norm = _norm_search(q)

    # ── IMPECT search ─────────────────────────────────────────────────────────
    impect_mask = data["PlayerName"].astype(str).apply(_norm_search).str.contains(q_norm, regex=False)
    impect_hits = data.loc[impect_mask].copy()

    results_rows: list[dict] = []
    covered_ws: set[tuple[str, str]] = set()  # (file, ws_player_name) already linked

    for idx, row in impect_hits.iterrows():
        pname = str(row.get("PlayerName", ""))
        ws_link: pd.Series | None = None
        if not link_db.empty:
            cands = link_db.loc[link_db["IMPECT_Name"].astype(str).apply(_norm_search).eq(_norm_search(pname))]
            if not cands.empty:
                for conf in ("HIGH", "MEDIUM", "LOW"):
                    sub = cands.loc[cands["MatchConfidence"] == conf]
                    if not sub.empty:
                        ws_link = sub.iloc[0]
                        covered_ws.add((str(ws_link["Wyscout_File"]), str(ws_link["Wyscout_Name"])))
                        break
        results_rows.append({
            "Player":      pname,
            "Pos":         str(row.get("PositionGroup", "")),
            "Team":        str(row.get("TeamName", "")),
            "League":      str(row.get("BundleLabel", "")),
            "Age":         round(float(row.get("AgeYears", 0) or 0), 1),
            "Quality":     round(float(row.get("QualityScore", row.get("CompositeRecruitmentScore", 0)) or 0), 1),
            "Source":      f"IMPECT+WS ({ws_link['MatchConfidence']})" if ws_link is not None else "IMPECT",
            "_data_idx":   idx,
            "_ws_link":    ws_link,
            "_ws_idx_row": None,
        })

    # ── Wyscout-only search ───────────────────────────────────────────────────
    if not ws_idx.empty:
        ws_norms = ws_idx["Player"].astype(str).apply(_norm_search)
        ws_mask = ws_norms.str.contains(q_norm, regex=False)
        # also match last word of query against Wyscout names (handles "Firstname Lastname" → "F. Lastname")
        q_parts = q_norm.split()
        if len(q_parts) > 1:
            ws_mask = ws_mask | ws_norms.str.contains(q_parts[-1], regex=False)
        ws_hits = ws_idx.loc[ws_mask].copy()

        for _, ws_row in ws_hits.iterrows():
            file_name = str(ws_row.get("File", ""))
            ws_name   = str(ws_row.get("Player", ""))
            if (file_name, ws_name) in covered_ws:
                continue
            country_raw, division = _parse_wyscout_filename(file_name)
            div_label = (
                "I" * division if isinstance(division, int)
                else str(division)
            )
            league_label = f"{country_raw} {div_label}".strip() if div_label not in ("1", "I") else country_raw
            raw_pos = str(ws_row.get("Position", "") or "")
            raw_pos = raw_pos.split(",")[0].split(";")[0].strip()  # first position only
            ws_pg   = WYSCOUT_POSITION_MAP.get(raw_pos, "")
            _age_raw = pd.to_numeric(ws_row.get("Age"), errors="coerce")
            results_rows.append({
                "Player":      ws_name,
                "Pos":         ws_pg or raw_pos,
                "Team":        str(ws_row.get("Team", "")),
                "League":      league_label,
                "Age":         float(_age_raw) if not pd.isna(_age_raw) else float("nan"),
                "Quality":     float("nan"),
                "Source":      "Wyscout",
                "_data_idx":   None,
                "_ws_link":    None,
                "_ws_idx_row": ws_row,
            })

    if not results_rows:
        st.info(f"No players found matching '{q}'. Try searching by last name.")
        return

    results_df = pd.DataFrame(results_rows)
    display_df = results_df[["Player", "Pos", "Team", "League", "Age", "Quality", "Source"]]

    st.markdown(
        f"<div style='color:var(--faint);font-size:.65rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:.08em;padding:4px 0 6px;'>"
        f"<span style='color:var(--ink);'>{len(results_df)}</span> result{'s' if len(results_df) != 1 else ''} "
        f"for <span style='color:var(--teal-hi);'>{q}</span></div>",
        unsafe_allow_html=True,
    )

    event = st.dataframe(
        display_df,
        selection_mode="single-row",
        on_select="rerun",
        key="player_search_results",
        hide_index=True,
        use_container_width=True,
        height=min(len(results_df) * 36 + 50, 380),
        column_config={
            "Quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.1f"),
            "Source":  st.column_config.TextColumn("Source"),
        },
    )

    if not event.selection.rows:
        st.markdown(
            "<div class='note-box' style='margin-top:8px;'>↑ Click a row to open the full player profile.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Player profile ────────────────────────────────────────────────────────
    sel = results_df.iloc[event.selection.rows[0]]
    is_wyscout_only = pd.isna(sel["_data_idx"])

    # Resolve IMPECT row (None for Wyscout-only players)
    impect_row: pd.Series | None = data.loc[sel["_data_idx"]] if not is_wyscout_only else None

    # Resolve Wyscout source: link DB row (IMPECT-linked) OR index row (Wyscout-only)
    # pandas converts None → NaN in mixed-type columns; pd.Series can't be bool-tested
    def _unwrap(v):
        if isinstance(v, pd.Series):
            return v
        try:
            return None if pd.isna(v) else v
        except (TypeError, ValueError):
            return v

    ws_link    = _unwrap(sel["_ws_link"])
    ws_idx_row = _unwrap(sel["_ws_idx_row"])
    has_wyscout = ws_link is not None or ws_idx_row is not None

    if impect_row is not None:
        pos_group = str(impect_row.get("PositionGroup", ""))
    else:
        raw_pos   = str(ws_idx_row.get("Position", "") or "")  # type: ignore[union-attr]
        raw_pos   = raw_pos.split(",")[0].split(";")[0].strip()  # first position only
        pos_group = WYSCOUT_POSITION_MAP.get(raw_pos, "")

    # Pre-load Wyscout data so charts and table share it
    _ws_row_loaded: pd.Series | None = None
    _ws_pos_pool_loaded: pd.DataFrame = pd.DataFrame()
    _ws_file_loaded = ""
    _ws_name_loaded = ""
    _ws_team_loaded = ""
    _ws_conf_loaded = ""

    if ws_link is not None:
        # IMPECT-linked player — use link DB metadata
        _ws_file_loaded = str(ws_link.get("Wyscout_File", ""))
        _ws_name_loaded = str(ws_link.get("Wyscout_Name", ""))
        _ws_team_loaded = str(ws_link.get("Wyscout_Team", ""))
        _ws_conf_loaded = str(ws_link.get("MatchConfidence", ""))
    elif ws_idx_row is not None:
        # Wyscout-only player — use index row metadata
        _ws_file_loaded = str(ws_idx_row.get("File", ""))
        _ws_name_loaded = str(ws_idx_row.get("Player", ""))
        _ws_team_loaded = str(ws_idx_row.get("Team", ""))
        _ws_conf_loaded = ""

    if _ws_file_loaded:
        _ws_path_loaded = WYSCOUT_DB_DIR / _ws_file_loaded
        if _ws_path_loaded.exists():
            _ws_df_loaded = _load_wyscout_file(_ws_path_loaded)
            if not _ws_df_loaded.empty:
                _ws_hit = _ws_df_loaded.loc[_ws_df_loaded["Player"].astype(str).eq(_ws_name_loaded)]
                if not _ws_hit.empty:
                    _ws_row_loaded = _ws_hit.iloc[0]
                    _pc_ws = next((c for c in ["Position", "Pos"] if c in _ws_df_loaded.columns), None)
                    _pg_col = (_ws_df_loaded[_pc_ws].astype(str).str.strip()
                               .map(WYSCOUT_POSITION_MAP).fillna("").rename("_PosGroup")
                               if _pc_ws else pd.Series("", index=_ws_df_loaded.index, name="_PosGroup"))
                    _ws_df_loaded = pd.concat([_ws_df_loaded, _pg_col], axis=1)
                    _ws_pos_pool_loaded = _ws_df_loaded.loc[_ws_df_loaded["_PosGroup"].eq(pos_group)]

    st.markdown("<hr style='border:none;border-top:1px solid #21262d;margin:1.2rem 0 0.8rem;'>",
                unsafe_allow_html=True)

    # Header card
    if impect_row is not None:
        _risk_cls = {"Low": "teal", "Moderate": "amber", "Elevated": "amber", "High": "red"}.get(
            str(impect_row.get("RiskBand", "")), "")
        _qs   = float(impect_row.get("QualityScore", impect_row.get("CompositeRecruitmentScore", 0)) or 0)
        _tier = str(impect_row.get("QualityTier", impect_row.get("ScoreBand", "")) or "")
        _arch = str(impect_row.get("Archetype", "") or "")
        _risk = str(impect_row.get("RiskBand", "") or "")
        _mins = int(impect_row.get("MinutesPlayed", 0) or 0)
        _age  = float(impect_row.get("AgeYears", 0) or 0)
        _ws_badge = (f'<span class="pill teal">Wyscout {_ws_conf_loaded}</span>' if has_wyscout else "")
        st.markdown(
            f'<div class="profile-card">'
            f'<div class="profile-name">{escape(str(impect_row.get("PlayerName","")))}</div>'
            f'<div class="profile-meta">{escape(str(impect_row.get("TeamName","")))} · '
            f'{escape(str(impect_row.get("BundleLabel","")))} · {escape(pos_group)} · '
            f'{_age:.1f} yrs · {_mins:,} min</div>'
            f'<div class="pill-row">'
            f'<span class="pill teal">Quality {_qs:.1f}</span>'
            f'{f"""<span class="pill">{escape(_tier)}</span>""" if _tier else ""}'
            f'{f"""<span class="pill">{escape(_arch)}</span>""" if _arch else ""}'
            f'{f"""<span class="pill {_risk_cls}">{escape(_risk)} risk</span>""" if _risk else ""}'
            f'{_ws_badge}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Wyscout-only header
        _ws_age_raw = ws_idx_row.get("Age", "") if ws_idx_row is not None else ""  # type: ignore[union-attr]
        _ws_age_str = f"{float(_ws_age_raw):.0f} yrs" if pd.notna(pd.to_numeric(_ws_age_raw, errors="coerce")) else ""
        _ws_raw_pos_full = str(ws_idx_row.get("Position", "") or "") if ws_idx_row is not None else ""  # type: ignore[union-attr]
        _ws_raw_pos = _ws_raw_pos_full.split(",")[0].split(";")[0].strip()
        country_raw2, _ = _parse_wyscout_filename(_ws_file_loaded)
        st.markdown(
            f'<div class="profile-card">'
            f'<div class="profile-name">{escape(_ws_name_loaded)}</div>'
            f'<div class="profile-meta">{escape(_ws_team_loaded)} · '
            f'{escape(country_raw2)} · {escape(_ws_raw_pos)} ({escape(pos_group)}) · {escape(_ws_age_str)}</div>'
            f'<div class="pill-row">'
            f'<span class="pill">Wyscout</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Charts ────────────────────────────────────────────────────────────────
    has_impect_model  = impect_row is not None and bool(pos_group)
    has_wyscout_loaded = _ws_row_loaded is not None and not _ws_pos_pool_loaded.empty

    def _render_pizza_section(container=None):
        target = container if container is not None else st
        target.markdown(
            f"<div style='color:var(--muted);font-size:.75rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.06em;margin-bottom:6px;'>IMPECT model · percentile vs {pos_group} pool</div>",
            unsafe_allow_html=True,
        )
        pos_pool_impect = data.loc[data["PositionGroup"].astype(str).eq(pos_group)].copy()
        try:
            fig_pizza = render_player_pizza(pos_pool_impect, impect_row)
            target.pyplot(fig_pizza, use_container_width=True)
            plt.close(fig_pizza)
        except Exception as _ex:
            target.info(f"Pizza chart unavailable: {_ex}")

    def _render_bars_section(container=None):
        target = container if container is not None else st
        target.markdown(
            "<div style='color:var(--muted);font-size:.75rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.06em;margin-bottom:6px;'>Wyscout raw metrics · percentile bars</div>",
            unsafe_allow_html=True,
        )
        if has_wyscout_loaded:
            if _ws_conf_loaded and _ws_conf_loaded != "HIGH":
                target.markdown(
                    f"<span style='color:var(--amber);font-size:.72rem;'>Link: {_ws_conf_loaded} — verify manually</span>",
                    unsafe_allow_html=True,
                )
            _fig_bars = _render_wyscout_bars(
                _ws_row_loaded, _ws_pos_pool_loaded, pos_group,
                _ws_name_loaded, _ws_team_loaded, _ws_file_loaded,
            )
            if _fig_bars:
                target.pyplot(_fig_bars, use_container_width=True)
                plt.close(_fig_bars)
            else:
                target.info("No Wyscout metrics available for this position.")
        elif has_wyscout:
            target.info(f"Wyscout data unavailable ({_ws_file_loaded}).")
        else:
            target.markdown(
                "<div class='note-box'>No Wyscout link for this player "
                "(league may not be in Wyscout DB, or link confidence too low).</div>",
                unsafe_allow_html=True,
            )

    if has_impect_model and has_wyscout_loaded:
        pizza_col, bars_col = st.columns(2, gap="large")
        _render_pizza_section(pizza_col)
        _render_bars_section(bars_col)
    elif has_wyscout_loaded:
        _render_bars_section()
    else:
        if has_impect_model:
            _render_pizza_section()
        _render_bars_section()

    # ── Stats tables ──────────────────────────────────────────────────────────
    _show_impect_tbl = impect_row is not None
    _show_ws_tbl     = _ws_row_loaded is not None and not _ws_pos_pool_loaded.empty

    if _show_impect_tbl and _show_ws_tbl:
        _exp_left, _exp_right = st.columns(2, gap="large")
        _impect_tbl_ctx = _exp_left
        _ws_tbl_ctx     = _exp_right
    elif _show_impect_tbl:
        _impect_tbl_ctx = st
        _ws_tbl_ctx     = None  # type: ignore[assignment]
    else:
        _impect_tbl_ctx = None  # type: ignore[assignment]
        _ws_tbl_ctx     = st

    if _show_impect_tbl:
        with _impect_tbl_ctx:  # type: ignore[union-attr]
            with st.expander("📊 IMPECT model scores", expanded=False):
                _score_rows2: list[dict] = []
                _pos_data = data.loc[data["PositionGroup"].astype(str).eq(pos_group)]
                for label, col in PIZZA_METRICS.items():
                    if col not in impect_row.index:  # type: ignore[union-attr]
                        continue
                    _sv = float(impect_row.get(col, 0) or 0)  # type: ignore[union-attr]
                    _pv = round(percentile_rank(_pos_data[col], _sv), 0) if col in _pos_data.columns else None
                    _score_rows2.append({"Metric": label, "Score": round(_sv, 1), "Pct": _pv})
                if _score_rows2:
                    _score_df2 = pd.DataFrame(_score_rows2).sort_values("Score", ascending=False)
                    st.dataframe(
                        _score_df2, hide_index=True, use_container_width=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                            "Pct":   st.column_config.NumberColumn("Pct %", format="%.0f"),
                        },
                    )

    if _show_ws_tbl:
        with _ws_tbl_ctx:  # type: ignore[union-attr]
            with st.expander("📋 Wyscout raw stats", expanded=False):
                _ws_metrics_show = [m for m in WYSCOUT_POS_METRICS.get(pos_group, [])
                                    if m in _ws_row_loaded.index]  # type: ignore[union-attr]
                _ws_table_rows2: list[dict] = []
                for m in _ws_metrics_show:
                    _v = pd.to_numeric(_ws_row_loaded.get(m), errors="coerce")  # type: ignore[union-attr]
                    if pd.isna(_v):
                        continue
                    _p2 = percentile_rank(pd.to_numeric(_ws_pos_pool_loaded[m], errors="coerce"), float(_v))
                    if m in WYSCOUT_LOWER_IS_BETTER:
                        _p2 = 100.0 - _p2
                    _ws_table_rows2.append({"Metric": m, "Value": round(float(_v), 2), "Pct": round(_p2, 0)})
                if _ws_table_rows2:
                    _ws_tbl2 = pd.DataFrame(_ws_table_rows2).sort_values("Pct", ascending=False)
                    st.dataframe(
                        _ws_tbl2, hide_index=True, use_container_width=True,
                        column_config={
                            "Pct": st.column_config.ProgressColumn("Pct %", min_value=0, max_value=100, format="%.0f"),
                        },
                    )


def render_model_workspace(data: pd.DataFrame, metadata: dict[str, pd.DataFrame]) -> None:
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

    with st.sidebar:
        st.markdown("<div class='sidebar-brand'><div class='sidebar-brand-icon'>🔬</div><div><div class='sidebar-brand-title'>Model &amp; Data</div><div class='sidebar-brand-meta'>Smart Club Closeness</div></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='sbar-hdr'>Club model</div>", unsafe_allow_html=True)
        selected_model = st.selectbox("Select model", summary[model_col].tolist(), key="model_selectbox")

    st.markdown(
        "<div class='page-header'>"
        "<div class='page-header-icon'>🔬</div>"
        "<div><div class='page-header-title'>Model &amp; Data</div>"
        "<div class='page-header-sub'>Smart club closeness scores and league data coverage</div></div>"
        "</div>",
        unsafe_allow_html=True,
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

    if keeper_df.empty:
        st.info("No goalkeeper rows were found in the loaded model outputs.")
        return

    league_options = sorted(keeper_df["BundleLabel"].dropna().astype(str).unique())

    with st.sidebar:
        st.markdown("<div class='sidebar-brand'><div class='sidebar-brand-icon'>🧤</div><div><div class='sidebar-brand-title'>Goalkeepers</div><div class='sidebar-brand-meta'>GK-specific scoring</div></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='sbar-hdr'>Filters</div>", unsafe_allow_html=True)
        selected_leagues = st.multiselect("Leagues", league_options, default=league_options, key="gk_bundles_filter")
        age_min = float(np.floor(keeper_df["AgeYears"].min()))
        age_max = float(np.ceil(keeper_df["AgeYears"].max()))
        age_range = st.slider("Age range", age_min, age_max, (age_min, age_max), step=0.5, key="gk_age_filter")

    gk_filtered = keeper_df.loc[
        keeper_df["BundleLabel"].astype(str).isin(selected_leagues)
        & keeper_df["AgeYears"].between(age_range[0], age_range[1])
    ].copy()

    st.markdown(
        f"<div class='page-header'>"
        f"<div class='page-header-icon'>🧤</div>"
        f"<div><div class='page-header-title'>Goalkeepers</div>"
        f"<div class='page-header-sub'>GK-specific scoring model · {len(gk_filtered):,} keepers</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

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
        height=780,
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
            width="stretch", hide_index=True,
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
            st.altair_chart(case_chart, width="stretch")
        else:
            _shape = rd.groupby(["PositionGroup","StyleFit"]).agg(Players=("PlayerName","count"),MedianCase=("CaseScore","median")).round(1).reset_index().sort_values("MedianCase",ascending=False)
            st.dataframe(_shape, width="stretch", hide_index=True)


def render_team_workspace(data: pd.DataFrame) -> None:
    team_df = add_scouting_fields(data, BALANCED_WEIGHTS)
    czech_df = czech_market(team_df)
    hradec_df = hradec_squad(team_df)
    external_czech = czech_df.loc[~czech_df["TeamName"].isin(hradec_df["TeamName"].unique())].copy()

    with st.sidebar:
        st.markdown("<div class='sidebar-brand'><div class='sidebar-brand-icon'>🏟</div><div><div class='sidebar-brand-title'>Team Intelligence</div><div class='sidebar-brand-meta'>Squad &amp; Czech market</div></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='note-box' style='font-size:.69rem;'>Compare Hradec&#39;s squad vs the Czech market to find recruitment priorities.</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='page-header'>"
        f"<div class='page-header-icon'>🏟</div>"
        f"<div><div class='page-header-title'>Team Intelligence</div>"
        f"<div class='page-header-sub'>Squad overview &amp; Czech market comparison · {len(hradec_df):,} Hradec players · {len(external_czech):,} Czech market</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    h_top = hradec_df.sort_values("ScoutFitScore", ascending=False).head(1)
    cz_top = external_czech.sort_values("ScoutFitScore", ascending=False).head(1)

    if hradec_df.empty:
        st.info("No FC Hradec Kralove rows were found in the loaded model outputs.")
        return

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
        height=780,
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
            height=780,
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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap" rel="stylesheet">
    <style>
    /* ── TOKENS ──────────────────────────────────────────────── */
    :root {
        --teal:      #0d9e7d;
        --teal-hi:   #12c799;
        --teal-dim:  rgba(13,158,125,.15);
        --teal-glow: rgba(13,158,125,.25);
        --ink:       #e6edf3;
        --muted:     #8b949e;
        --faint:     #6e7681;
        --border:    #21262d;
        --border-hi: #30363d;
        --surface:   #161b22;
        --raised:    #1c2128;
        --bg:        #0d1117;
        --nav-bg:    #010409;
        --amber:     #e3b341;
        --red:       #f85149;
        --green:     #3fb950;
        --shadow-sm: 0 1px 3px rgba(0,0,0,.4);
        --shadow:    0 4px 12px rgba(0,0,0,.5);
        --radius:    8px;
    }

    /* ── RESET & GLOBAL ─────────────────────────────────────── */
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background: var(--bg) !important;
        color: var(--ink) !important;
        -webkit-font-smoothing: antialiased !important;
    }
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"] { display: none !important; }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 4rem !important;
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    div[data-testid="stVerticalBlock"] { gap: 0 !important; }
    div[data-testid="stVerticalBlock"] > div.element-container { margin-bottom: .3rem !important; }
    div[data-testid="stTabsContent"] > div[data-testid="stVerticalBlock"] > div.element-container { margin-bottom: .45rem !important; }
    div[data-testid="stMetric"] { margin-bottom: 0 !important; }
    [data-testid="stDataFrame"] > div { width: 100% !important; }

    h1, h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 800; color: var(--ink) !important; }
    h2 { font-size: .62rem !important; text-transform: uppercase; letter-spacing: .16em;
         color: var(--faint) !important; border-bottom: 1px solid var(--border) !important;
         padding-bottom: 6px; margin-bottom: 10px !important; }
    h3 { font-size: .85rem !important; font-weight: 700 !important; margin-bottom: 6px !important; }

    /* ── APP NAV BAR ─────────────────────────────────────────── */
    /* Full-width nav bar — extend to viewport edges */
    [data-testid="stHorizontalBlock"]:has([class*="st-key-workspace_main"]) {
        background: var(--nav-bg) !important;
        border: none !important;
        border-bottom: 1px solid var(--border) !important;
        border-radius: 0 !important;
        margin-left: -2rem !important;
        margin-right: -2rem !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 2rem !important;
        gap: 0 !important;
        align-items: stretch !important;
    }
    /* Nav buttons — complete reset then rebuild */
    [data-testid="stHorizontalBlock"]:has([class*="st-key-workspace_main"]) .stButton > button {
        all: unset !important;
        box-sizing: border-box !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 52px !important;
        padding: 0 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: .74rem !important;
        font-weight: 600 !important;
        color: var(--muted) !important;
        border-bottom: 2px solid transparent !important;
        white-space: nowrap !important;
        letter-spacing: .01em !important;
        transition: color .12s ease, border-color .12s ease, background .12s ease !important;
        user-select: none !important;
    }
    [data-testid="stHorizontalBlock"]:has([class*="st-key-workspace_main"]) .stButton > button[kind="primary"] {
        color: var(--teal-hi) !important;
        font-weight: 700 !important;
        border-bottom-color: var(--teal) !important;
    }
    [data-testid="stHorizontalBlock"]:has([class*="st-key-workspace_main"]) .stButton > button:hover {
        color: var(--teal-hi) !important;
        background: rgba(13,158,125,.06) !important;
    }
    /* Brand area inside nav */
    .app-nav-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        height: 52px;
        padding: 0 4px;
    }
    .app-nav-emoji { font-size: 1.5rem; line-height: 1; flex-shrink: 0; }
    .app-nav-name {
        font-size: .85rem; font-weight: 900; color: var(--ink);
        letter-spacing: -.01em; line-height: 1.2;
    }
    .app-nav-tagline { font-size: .58rem; color: var(--faint); margin-top: 2px; letter-spacing: .03em; }

    /* ── PAGE HEADER ─────────────────────────────────────────── */
    .page-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 22px 0 18px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 22px;
    }
    .page-header-icon {
        font-size: 2rem; line-height: 1; flex-shrink: 0;
        background: var(--surface); border: 1px solid var(--border-hi);
        border-radius: var(--radius); width: 48px; height: 48px;
        display: flex; align-items: center; justify-content: center;
    }
    .page-header-title {
        font-size: 1.3rem; font-weight: 900; color: var(--ink);
        letter-spacing: -.02em; line-height: 1.1;
    }
    .page-header-sub { font-size: .72rem; color: var(--muted); margin-top: 4px; line-height: 1.4; }

    /* ── SIDEBAR ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--nav-bg) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }
    .sidebar-brand {
        display: flex; align-items: center; gap: 10px;
        padding: 4px 0 18px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 16px;
    }
    .sidebar-brand-icon { font-size: 1.5rem; flex-shrink: 0; }
    .sidebar-brand-title { color: var(--ink) !important; font-size: .84rem; font-weight: 800; letter-spacing: -.01em; }
    .sidebar-brand-meta  { color: var(--teal-hi) !important; font-size: .58rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: .08em; margin-top: 1px; }
    .sbar-hdr {
        color: var(--faint);
        font-size: .55rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: .2em;
        margin: 20px 0 6px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }
    .sbar-active-bar {
        background: rgba(13,158,125,.1);
        border: 1px solid rgba(13,158,125,.28);
        border-radius: 6px;
        padding: 7px 11px;
        font-size: .64rem; font-weight: 700;
        color: var(--teal-hi);
        margin: 10px 0;
        display: flex; align-items: center; gap: 6px;
    }

    /* ── BUTTONS (non-nav) ───────────────────────────────────── */
    .stButton > button, .stDownloadButton > button {
        font-family: 'Inter', sans-serif !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--border-hi) !important;
        background: var(--surface) !important;
        color: var(--muted) !important;
        font-weight: 600 !important;
        font-size: .73rem !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all .12s ease !important;
        min-height: 34px !important;
        letter-spacing: .01em !important;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: rgba(13,158,125,.14) !important;
        border-color: rgba(13,158,125,.5) !important;
        color: var(--teal-hi) !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(13,158,125,.6) !important;
        color: var(--teal-hi) !important;
        background: rgba(13,158,125,.1) !important;
    }

    /* ── CONTENT TABS ────────────────────────────────────────── */
    div[data-testid="stTabs"] {
        border-bottom: 1px solid var(--border) !important;
        margin-bottom: 2px !important;
    }
    div[data-testid="stTabs"] button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important; font-size: .72rem !important;
        color: var(--muted) !important; padding: 9px 18px !important;
        border-radius: 0 !important; background: transparent !important;
        transition: color .12s !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal-hi) !important;
        border-bottom: 2px solid var(--teal) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stTabs"] button:hover { color: var(--teal-hi) !important; }
    div[data-testid="stTabsContent"] { padding-top: 18px !important; }

    /* ── METRIC TILES ────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-top: 2px solid var(--border-hi) !important;
        border-radius: var(--radius) !important;
        padding: 14px 16px !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: .6rem !important; font-weight: 700 !important;
        text-transform: uppercase !important; letter-spacing: .12em !important;
        color: var(--faint) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.5rem !important; font-weight: 900 !important;
        color: var(--ink) !important; line-height: 1.15 !important;
    }

    /* ── INPUTS ──────────────────────────────────────────────── */
    .stTextInput input {
        border-radius: var(--radius) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: .79rem !important;
    }
    .stTextInput input:focus-within { outline: none !important; box-shadow: 0 0 0 2px var(--teal-glow) !important; }

    /* ── DATA TABLE ──────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
    }

    /* ── CUSTOM HTML COMPONENTS ─────────────────────────────── */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: 2px solid var(--teal);
        border-radius: var(--radius);
        padding: 14px 16px;
        box-shadow: var(--shadow-sm);
    }
    .metric-label  { color: var(--faint); font-size: .58rem; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; }
    .metric-value  { color: var(--ink); font-size: 1.35rem; font-weight: 900; line-height: 1.1; margin: 4px 0 2px; }
    .metric-caption{ color: var(--muted); font-size: .67rem; }

    .note-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 11px 15px;
        font-size: .77rem; color: var(--muted); line-height: 1.65;
        margin: 6px 0;
    }
    .section-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 16px 20px; box-shadow: var(--shadow-sm);
    }
    .profile-card {
        background: var(--surface); border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 16px 20px; margin-bottom: 14px;
    }
    .profile-name { color: var(--ink); font-size: 1.2rem; font-weight: 900; letter-spacing: -.01em; }
    .profile-meta { color: var(--muted); font-size: .73rem; margin-top: 4px; margin-bottom: 10px; }

    .pill-row { display: flex; flex-wrap: wrap; gap: 5px; margin: 7px 0; }
    .pill {
        background: var(--raised); border: 1px solid var(--border-hi);
        color: var(--muted); border-radius: 99px; padding: 3px 11px;
        font-size: .63rem; font-weight: 600; letter-spacing: .01em;
    }
    .pill.teal  { background: rgba(13,158,125,.12); border-color: rgba(18,199,153,.3); color: var(--teal-hi); }
    .pill.amber { background: rgba(227,179,65,.1);  border-color: rgba(227,179,65,.3); color: var(--amber); }
    .pill.red   { background: rgba(248,81,73,.1);   border-color: rgba(248,81,73,.3);  color: var(--red); }

    .intel-strip {
        border: 1px solid var(--border); border-left: 3px solid var(--teal);
        background: var(--surface); border-radius: var(--radius); padding: 10px 16px;
        margin: 0 0 10px 0; display: flex; align-items: center;
        justify-content: space-between; gap: 14px;
    }
    .intel-strip-title { color: var(--ink); font-size: .82rem; font-weight: 700; }
    .intel-strip-meta  { color: var(--faint); font-size: .61rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; }

    .workspace-label {
        color: var(--teal-hi); font-size: .57rem; font-weight: 700;
        letter-spacing: .18em; text-transform: uppercase;
        margin: 4px 0 10px 0; padding: 5px 0 5px 11px;
        border-left: 2px solid var(--teal); background: rgba(13,158,125,.06);
        border-radius: 0 5px 5px 0;
    }
    .page-title    { font-size: 1.2rem; font-weight: 900; color: var(--ink); letter-spacing: -.02em; margin: 0 0 2px 0; }
    .page-subtitle { font-size: .71rem; color: var(--faint); margin-bottom: 18px; line-height: 1.5; }

    /* spacer utility */
    .gap-sm { margin-top: 10px; }
    .gap-md { margin-top: 20px; }
</style>
    """,
    unsafe_allow_html=True,
)


# Load persistent shortlist from disk on first run
if "shortlist_data" not in st.session_state:
    st.session_state["shortlist_data"] = _load_shortlist_file()
    st.session_state["shortlist_players"] = list(st.session_state["shortlist_data"].keys())

if "active_workspace" not in st.session_state:
    st.session_state["active_workspace"] = "Recruitment"

render_workspace_nav("main")

active_workspace = st.session_state.get("active_workspace", "Recruitment")
data = load_default_data()
model_metadata = load_model_metadata()
_rec_file = _model_file("recruitment")
_data_updated = "unknown"
if _rec_file.exists():
    from datetime import datetime as _dt
    _data_updated = _dt.fromtimestamp(_rec_file.stat().st_mtime).strftime("%-d %b %Y")
if active_workspace != "Recruitment":
    if active_workspace == "Scouting":
        render_scouting_workspace()
    elif active_workspace == "Search":
        render_search_workspace(data)
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

st.markdown(
    f"<div class='page-header'>"
    f"<div class='page-header-icon'>🎯</div>"
    f"<div><div class='page-header-title'>Recruitment Board</div>"
    f"<div class='page-header-sub'>Player quality rankings, profiles and signing cases · {len(outfield_data):,} outfield players</div></div>"
    f"</div>",
    unsafe_allow_html=True,
)
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
        f"""<div class="sidebar-brand">
            <div class="sidebar-brand-icon">⚽</div>
            <div>
                <div class="sidebar-brand-title">Recruitment IQ</div>
                <div class="sidebar-brand-meta">{len(outfield_data):,} outfield players</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
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
    # ── Search ──────────────────────────────────────────────────
    search = st.text_input("Search", key="search_filter", placeholder="🔍  Search player or club…", label_visibility="collapsed")

    # ── Positions ───────────────────────────────────────────────
    st.markdown("<div class='sbar-hdr'>Positions</div>", unsafe_allow_html=True)
    saved_positions = st.session_state.get("positions_filter")
    if saved_positions and any(p not in position_groups for p in saved_positions):
        st.session_state["positions_filter"] = position_groups
    positions = st.multiselect("Roles", position_groups, default=position_groups, key="positions_filter")
    _profile_opts: list[str] = []
    for _p in positions:
        _profile_opts.extend(POSITION_PROFILES.get(_p, []))
    _profile_opts = sorted(set(_profile_opts))
    if _profile_opts:
        profiles = st.multiselect("Player profiles", _profile_opts, default=_profile_opts, key="profiles_filter")
    else:
        profiles = []

    # ── Market ──────────────────────────────────────────────────
    st.markdown("<div class='sbar-hdr'>Market</div>", unsafe_allow_html=True)
    bundles = st.multiselect("Leagues", bundle_groups, default=bundle_groups, key="bundles_filter")
    if "CountryLabel" in df.columns:
        _country_opts = sorted(df["CountryLabel"].dropna().astype(str).replace("", "Unknown").unique())
        countries = st.multiselect("Country", _country_opts, default=_country_opts, key="countries_filter")
    else:
        _country_opts: list[str] = []
        countries = []
    archetypes = st.multiselect("Archetypes", archetype_groups, default=archetype_groups, key="archetypes_filter")

    # ── Player criteria ──────────────────────────────────────────
    st.markdown("<div class='sbar-hdr'>Player criteria</div>", unsafe_allow_html=True)
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
        "Min. minutes",
        int(df["MinutesPlayed"].min()),
        int(df["MinutesPlayed"].max()),
        (900, int(df["MinutesPlayed"].max())),
        step=100,
        key="minutes_filter",
    )

    # ── Score thresholds (collapsed) ────────────────────────────
    _q_floor_val     = st.session_state.get("quality_floor", 35)
    _fit_floor_val   = st.session_state.get("fit_floor", 35)
    _rel_floor_val   = st.session_state.get("reliability_floor", 45)
    _risk_val        = st.session_state.get("max_risk", 18.0)
    _thresh_modified = sum([
        _q_floor_val != 35,
        _fit_floor_val != 35,
        _rel_floor_val != 45,
        float(_risk_val) != 18.0,
    ])
    _thresh_lbl = f"⚙ Thresholds  ({_thresh_modified} modified)" if _thresh_modified else "⚙ Thresholds"
    with st.expander(_thresh_lbl, expanded=bool(_thresh_modified)):
        quality_floor     = st.slider("Quality floor", 0, 100, 35, key="quality_floor")
        role_floor        = st.slider("Role-fit floor", 0, 100, 35, key="fit_floor")
        reliability_floor = st.slider("Reliability floor", 0, 100, 45, key="reliability_floor")
        max_risk          = st.slider("Max risk/90", 0.0, 25.0, 18.0, step=0.5, key="max_risk")

    # ── Active filters count + Reset ────────────────────────────
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    _active_count = sum([
        len(positions) < len(position_groups),
        bool(_profile_opts) and len(profiles) < len(_profile_opts),
        len(bundles) < len(bundle_groups),
        len(archetypes) < len(archetype_groups),
        bool(_country_opts) and len(countries) < len(_country_opts),
        bool(u23_only),
        age_range[0] > float(np.floor(df["AgeYears"].min())),
        age_range[1] < float(np.ceil(df["AgeYears"].max())),
        minutes_range[0] > int(df["MinutesPlayed"].min()),
        quality_floor != 35,
        role_floor != 35,
        reliability_floor != 45,
        float(max_risk) != 18.0,
        bool(search),
    ])
    if _active_count:
        st.markdown(
            f"<div class='sbar-active-bar'>⚡ {_active_count} active filter{'s' if _active_count != 1 else ''} — board is narrowed</div>",
            unsafe_allow_html=True,
        )
    st.button("✕  Reset all filters", on_click=reset_filters, type="secondary", width="stretch", key="sidebar_reset_btn")

    # ── Quick view ──────────────────────────────────────────────
    st.markdown("<div class='sbar-hdr'>Quick view</div>", unsafe_allow_html=True)
    _qm_sidebar = st.session_state.get("quick_mode", "Full board")
    _qs_cols = st.columns(2)
    with _qs_cols[0]:
        st.button("Full board",  key="qm_full",     type="primary" if _qm_sidebar == "Full board"       else "secondary", width="stretch", on_click=set_quick_mode, args=("Full board",))
        st.button("Elite",       key="qm_elite",    type="primary" if _qm_sidebar == "Elite quality"    else "secondary", width="stretch", on_click=set_quick_mode, args=("Elite quality",))
    with _qs_cols[1]:
        st.button("U23",         key="qm_u23",      type="primary" if _qm_sidebar == "U23 quality"      else "secondary", width="stretch", on_click=set_quick_mode, args=("U23 quality",))
        st.button("Reliable",    key="qm_reliable", type="primary" if _qm_sidebar == "Reliable quality" else "secondary", width="stretch", on_click=set_quick_mode, args=("Reliable quality",))

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
if profiles and "PlayerProfile" in df.columns:
    mask &= df["PlayerProfile"].isin(profiles)
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
    st.markdown(
        "<div class='intel-strip' style='border-left-color:var(--amber);margin-top:12px;'>"
        "<div>"
        "<div class='intel-strip-title' style='color:var(--amber);'>No players match the current filters</div>"
        "<div class='intel-strip-meta' style='margin-top:4px;'>Try: lowering Quality floor · broadening Roles · increasing Age range · using Reset all filters in the sidebar</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

_shortlist_count = len(st.session_state.get("shortlist_players", []))

quality_tab, profile_tab, player_tab, compare_tab, hradec_tab, intel_tab, case_tab, export_tab = st.tabs(
    ["📊 Quality Board", "🎭 Profile Search", "🔬 Player Lab", "⚖️ Compare", "🎯 Hradec Targets", "🌍 League Intel", "💼 Case Analysis", "📥 Export"]
)

with quality_tab:
    # Position focus
    _pos_options = ["All positions"] + sorted(filtered["PositionGroup"].dropna().astype(str).unique().tolist())
    _pos_focus = st.segmented_control("Position", _pos_options, default="All positions", key="pos_focus_rail", label_visibility="collapsed")
    _tab_filtered = filtered.loc[filtered["PositionGroup"].eq(_pos_focus)].copy() if _pos_focus and _pos_focus != "All positions" else filtered

    _sl_badge = f"  ·  <strong style='color:var(--teal);'>★ {_shortlist_count} shortlisted</strong>" if _shortlist_count else ""
    st.markdown(
        f"<div style='font-size:.68rem;color:var(--faint);margin-bottom:6px;'>"
        f"<strong style='color:var(--ink);'>{len(_tab_filtered):,}</strong> players{_sl_badge}"
        f"</div>",
        unsafe_allow_html=True,
    )

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
    quality_board = _tab_filtered[[c for c in quality_cols if c in _tab_filtered.columns]].rename(
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
    _tcounts    = _tab_filtered["QualityTier"].value_counts().to_dict() if "QualityTier" in _tab_filtered.columns else {}
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
        width="stretch",
        hide_index=True,
        height=780,
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

with profile_tab:
    _all_positions = sorted(PROFILE_WEIGHTS.keys())
    _prof_pos_col, _prof_pick_col = st.columns([1, 2], gap="small")
    with _prof_pos_col:
        _target_pos = st.selectbox("Position", ["—"] + _all_positions, key="profile_target_pos", label_visibility="visible")
    with _prof_pick_col:
        _profile_options = list(PROFILE_WEIGHTS.get(_target_pos, {}).keys()) if _target_pos != "—" else []
        _target_profile = st.selectbox(
            "Profile",
            ["—"] + _profile_options,
            key="profile_target_name",
            disabled=_target_pos == "—",
            label_visibility="visible",
        )

    if _target_pos == "—" or _target_profile == "—":
        st.markdown(
            "<div class='note-box'>Select a <strong>position</strong> and a <strong>player profile</strong> above "
            "to rank every player in that position by how closely they fit the profile, "
            "scored using z-scores within the position pool.</div>",
            unsafe_allow_html=True,
        )
    else:
        _pos_pool = filtered.loc[filtered["PositionGroup"].astype(str).eq(_target_pos)].copy()
        if _pos_pool.empty:
            st.info(f"No {_target_pos} players in the current filtered set.")
        else:
            _pos_pool["ProfileFit"] = calc_profile_fit(_pos_pool, _target_pos, _target_profile)
            _pos_pool = _pos_pool.sort_values("ProfileFit", ascending=False)

            # Header strip showing profile definition
            _pw = PROFILE_WEIGHTS[_target_pos][_target_profile]
            _drivers = " · ".join(f"{k.replace('Score','').replace('CreativeProgression','Creativity').replace('DefensiveDisruption','Defence').replace('ScoringThreat','Scoring').replace('ExpectedThreat','xThreat').replace('BallSecurity','Security').replace('PerformanceReliability','Reliability').replace('Pressing','Press')} ×{v}" for k, v in _pw.items())
            st.markdown(
                f"<div class='note-box'><strong style='color:var(--teal-hi);'>{_target_profile}</strong> "
                f"<span style='color:var(--muted);'>({_target_pos})</span>"
                f"<br><span style='color:var(--faint);font-size:.7rem;'>Profile drivers: {_drivers}</span><br>"
                f"<span style='color:var(--faint);font-size:.7rem;'>"
                f"Z-score scaled: 50 = position average · 65 = top 16% · 80 = top 2%"
                f"</span></div>",
                unsafe_allow_html=True,
            )

            _prof_board_cols = [
                "PlayerName", "TeamName", "BundleLabel", "AgeYears", "MinutesPlayed",
                "ProfileFit", "QualityScore", "QualityTier", "Archetype",
            ] + [c for c in _pw.keys() if c in _pos_pool.columns]
            _prof_board = _pos_pool[[c for c in _prof_board_cols if c in _pos_pool.columns]].rename(columns={
                "PlayerName": "Player", "TeamName": "Team", "BundleLabel": "League",
                "AgeYears": "Age", "MinutesPlayed": "Minutes",
                "QualityScore": "Quality", "QualityTier": "Tier",
                "ScoringThreatScore": "Scoring", "ExpectedThreatScore": "xThreat",
                "CreativeProgressionScore": "Creativity", "DefensiveDisruptionScore": "Defence",
                "PressingScore": "Press", "BallSecurityScore": "Security",
                "DecisionScore": "Decision", "PerformanceReliabilityScore": "Reliability",
            })
            _prof_col_cfg = {
                "ProfileFit": st.column_config.ProgressColumn("Profile Fit ▼", min_value=0, max_value=100, format="%.1f"),
                "Quality":    st.column_config.ProgressColumn("Quality",       min_value=0, max_value=100, format="%.1f"),
                "Age":        st.column_config.NumberColumn("Age", format="%.1f"),
                "Minutes":    st.column_config.NumberColumn("Minutes", format="%d"),
            }
            for _dc in ["Scoring", "xThreat", "Creativity", "Defence", "Press", "Security", "Decision", "Reliability"]:
                if _dc in _prof_board.columns:
                    _prof_col_cfg[_dc] = st.column_config.ProgressColumn(_dc, min_value=0, max_value=100, format="%.1f")
            st.dataframe(_prof_board.round(2), width="stretch", hide_index=True, height=780, column_config=_prof_col_cfg)

with player_tab:
    st.markdown("<div class='workspace-label'>🔬 Player Lab — deep dive profile & radar</div>", unsafe_allow_html=True)
    _plab_sorted = filtered.sort_values("QualityScore", ascending=False)
    _plab_labels = (
        _plab_sorted["PlayerName"].fillna("").astype(str) + "  ·  "
        + _plab_sorted["TeamName"].fillna("").astype(str) + "   ["
        + _plab_sorted["PositionGroup"].fillna("").astype(str) + "  "
        + _plab_sorted["AgeYears"].round(1).astype(str) + "y  Q "
        + _plab_sorted["QualityScore"].round(0).astype(int).astype(str) + "]"
    ).tolist()
    player_options = _plab_sorted.copy()
    player_options["_label"] = _plab_labels
    selected_label = st.selectbox("Select player", _plab_labels)
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
            '<div class="note-box" style="margin-top:6px;border-left-color:rgba(13,158,125,.5);">'
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
            width="stretch",
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
        st.altair_chart(_age_chart, width="stretch")

    comparable = similar_players(df, player, same_position=True, n=12)
    st.markdown("<div class='workspace-label' style='font-size:.58rem;margin:14px 0 8px;'>Closest quality profiles</div>", unsafe_allow_html=True)
    similar_cols = ["PlayerName", "TeamName", "PositionGroup", "AgeYears", "MinutesPlayed", "SimilarityScore", "QualityScore", "RoleFitScore", "ProfileScore", "Archetype"]
    st.dataframe(
        comparable[[c for c in similar_cols if c in comparable.columns]].rename(columns={"PlayerName":"Player","TeamName":"Team","PositionGroup":"Role","AgeYears":"Age","MinutesPlayed":"Minutes","SimilarityScore":"Similarity","QualityScore":"Quality","RoleFitScore":"Role Fit","ProfileScore":"Impact"}).round(2),
        width="stretch",
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
            width="stretch",
            hide_index=True,
            height=780,
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
            st.altair_chart(_need_chart, width="stretch")

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
            st.dataframe(pd.DataFrame(xi_rows), width="stretch", hide_index=True)

with intel_tab:
    st.markdown("<div class='workspace-label'>🌍 League & market intelligence</div>", unsafe_allow_html=True)

    # ── European market map ─────────────────────────────────────
    st.subheader("European scout priority map")
    market_df = european_market_map_frame(df, metric="QualityScore")
    if not market_df.empty:
        map_col, map_info_col = st.columns([2.2, 1])
        with map_col:
            try:
                st.altair_chart(render_european_map(market_df), width="stretch")
            except Exception as e:
                st.warning(f"Map could not render: {e}")
        with map_info_col:
            st.markdown("<div class='note-box'>Bubble size = player count. Colour = scout priority score (quality + high-quality share + depth). Labels show countries with priority ≥ 55.</div>", unsafe_allow_html=True)
            top_markets = market_df.head(8)[["Country", "Players", "MedianScore", "GoScore", "Recommendation"]].rename(
                columns={"MedianScore": "Median Q", "GoScore": "Priority"}
            )
            st.dataframe(top_markets.round(1), width="stretch", hide_index=True)

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
        st.altair_chart(bar_chart, width="stretch")

    # ── Full league table ───────────────────────────────────────
    with st.expander("Full league table", expanded=False):
        st.dataframe(
            league_summary.rename(columns={"BundleLabel": "League", "MedianQuality": "Median Q",
                                            "MedianRoleFit": "Median Fit", "EliteCount": "Elite",
                                            "ElitePct": "Elite %", "MedianAge": "Median Age"}).round(1),
            width="stretch", hide_index=True,
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
        st.pyplot(render_score_distribution(filtered, dist_metric), clear_figure=True, width="stretch")
    with dist_col2:
        st.pyplot(render_position_boxplot(filtered, dist_metric), clear_figure=True, width="stretch")

    # ── League heatmap ──────────────────────────────────────────
    with st.expander("League depth heatmap", expanded=False):
        hm_metric = st.selectbox("Heatmap metric", ["QualityScore", "RoleFitScore", "CompositeRecruitmentScore"], key="hm_metric")
        st.pyplot(render_league_heatmap(filtered, hm_metric), clear_figure=True, width="stretch")

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
        st.altair_chart(compare_chart, width="stretch")

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
            width="stretch", hide_index=True,
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
