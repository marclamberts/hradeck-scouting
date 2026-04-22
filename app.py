from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from textwrap import shorten

APP_DIR = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".matplotlib"))

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


DEFAULT_FILE = APP_DIR / "data" / "FCHK Scores Only.xlsx"
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


st.set_page_config(
    page_title="FCHK Scouting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if col not in {"PlayerName", "TeamName", "PositionGroup", "BundleLabel"}:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
    return df


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return _clean_columns(pd.read_excel(DEFAULT_FILE))


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
    out["ScoutFitScore"] = (numerator / denominator).clip(0, 100)
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


st.markdown(
    """
    <style>
    :root {
        --ink: #10212b;
        --muted: #667085;
        --panel: #ffffff;
        --line: #d9e2e7;
        --navy: #102a43;
        --teal: #2a9d8f;
        --sky: #457b9d;
        --coral: #e76f51;
        --amber: #f4a261;
        --wash: #f4f7f9;
    }
    .stApp { background: var(--wash); color: var(--ink); }
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: .82rem;
        text-transform: uppercase;
        letter-spacing: .07em;
        color: #536471;
        margin-bottom: .35rem;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid #dce6eb;
        border-radius: 8px;
        background: white;
        overflow: hidden;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(16, 33, 43, .035);
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details summary {
        font-weight: 850;
        color: var(--ink);
    }
    section[data-testid="stSidebar"] label {
        color: #465862;
        font-weight: 750;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    div[data-testid="stTabs"] button { font-weight: 800; color: var(--muted); }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal); }
    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #cbd8de;
        background: #ffffff;
        color: var(--ink);
        font-weight: 800;
        min-height: 42px;
        box-shadow: 0 1px 2px rgba(16, 33, 43, .06);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--teal);
        color: var(--teal);
        background: #f0fbf8;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: var(--teal);
        color: white;
        border-color: var(--teal);
    }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
        background: #21877b;
        color: white;
    }
    div[data-testid="stSidebar"] .stButton > button {
        min-height: 36px;
        font-size: .86rem;
        box-shadow: none;
    }
    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--navy);
        border-color: var(--navy);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 8px;
    }
    .hero {
        border: 1px solid #cfdde4;
        border-radius: 8px;
        padding: 24px 26px;
        background:
            linear-gradient(135deg, rgba(16, 42, 67, .96) 0%, rgba(42, 157, 143, .92) 58%, rgba(231, 111, 81, .82) 100%),
            radial-gradient(circle at top right, rgba(244, 162, 97, .7), transparent 36%);
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 14px 34px rgba(16, 42, 67, .16);
    }
    .hero-kicker {
        color: #d8f7f1;
        font-size: .78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .hero h1 {
        color: white;
        margin: 6px 0 4px 0;
        font-size: 2.35rem;
        line-height: 1.1;
    }
    .hero p { margin: 0; max-width: 900px; color: #eef7f3; }
    .sidebar-brand {
        border: 1px solid #d7e4e9;
        border-radius: 8px;
        background: linear-gradient(135deg, #102a43 0%, #225c6b 100%);
        color: white;
        padding: 14px 14px;
        margin-bottom: 12px;
    }
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 900;
        letter-spacing: 0;
        line-height: 1.15;
    }
    .sidebar-brand-meta {
        margin-top: 6px;
        color: #d4edf1;
        font-size: .78rem;
        line-height: 1.35;
    }
    .menu-caption {
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.35;
        margin: -4px 0 8px;
    }
    .filter-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 4px 0 8px;
    }
    .filter-chip {
        border: 1px solid #d8e4e8;
        background: #f7fafb;
        color: var(--ink);
        padding: 4px 8px;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 800;
    }
    .section-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: white;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(16, 33, 43, .04);
    }
    .mode-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: -2px 0 16px 0;
    }
    .mode-pill {
        border-radius: 999px;
        padding: 7px 12px;
        border: 1px solid #cbd8de;
        background: white;
        color: var(--muted);
        font-size: .82rem;
        font-weight: 800;
    }
    .mode-pill.active {
        background: var(--ink);
        color: white;
        border-color: var(--ink);
    }
    .metric-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        background: var(--panel);
        min-height: 112px;
        box-shadow: 0 1px 2px rgba(16, 33, 43, .05);
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        font-weight: 700;
    }
    .metric-value {
        margin-top: 4px;
        color: var(--ink);
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-caption {
        color: var(--muted);
        font-size: .82rem;
        margin-top: 6px;
    }
    .profile-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        background: white;
        min-height: 140px;
        box-shadow: 0 1px 2px rgba(16, 33, 43, .05);
    }
    .profile-name {
        color: var(--ink);
        font-weight: 850;
        font-size: 1.12rem;
        line-height: 1.2;
    }
    .profile-meta { color: var(--muted); font-size: .88rem; margin-top: 3px; }
    .pill-row { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
    .pill {
        border: 1px solid #d8e4e8;
        border-radius: 999px;
        color: var(--ink);
        background: #f6fafb;
        padding: 4px 9px;
        font-size: .78rem;
        font-weight: 750;
    }
    .note-box {
        border-left: 4px solid var(--teal);
        background: #ffffff;
        padding: 12px 14px;
        border-radius: 6px;
        color: var(--ink);
    }
    .table-hint {
        color: var(--muted);
        font-size: .86rem;
        margin: -6px 0 10px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">FCHK recruitment intelligence</div>
        <h1>Scouting Command Center</h1>
        <p>Rank, filter, compare, and export the player pool with configurable recruitment weights, archetypes, risk flags, and detailed player profiles.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "quick_mode" not in st.session_state:
    st.session_state["quick_mode"] = "Full board"

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
    data = load_default_data()
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">FCHK Scout Room</div>
            <div class="sidebar-brand-meta">{len(data):,} players · {data['BundleLabel'].nunique()} leagues · {data['PositionGroup'].nunique()} roles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='menu-caption'>Use board modes for fast cuts, then refine the pool in the panels below.</div>", unsafe_allow_html=True)

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

df = add_scouting_fields(data, weights)

with st.sidebar:
    position_groups = sorted(df["PositionGroup"].dropna().astype(str).unique())
    bundle_groups = sorted(df["BundleLabel"].dropna().astype(str).unique())
    archetype_groups = sorted(df["Archetype"].dropna().astype(str).unique())
    with st.expander("Search and roles", expanded=True):
        search = st.text_input("Search player or team", key="search_filter", placeholder="Type a name or club")
        st.markdown("<div class='menu-caption'>Role shortcuts</div>", unsafe_allow_html=True)
        pos_cols = st.columns(4)
        quick_positions = [
            ("GK", ["GK"]),
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

tab_overview, tab_shortlist, tab_player, tab_analytics, tab_market, tab_exports = st.tabs(
    ["Overview", "Shortlist", "Player lab", "Analytics", "Market map", "Downloads"]
)

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

with tab_shortlist:
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
    ]
    view_cols = [c for c in shortlist_cols if c in filtered.columns]
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

        comparable = (
            df.loc[df["PositionGroup"].eq(player["PositionGroup"])]
            .assign(_gap=lambda x: (x["ScoutFitScore"] - player["ScoutFitScore"]).abs())
            .sort_values(["_gap", "ScoutFitScore"], ascending=[True, False])
            .head(12)
        )
        st.subheader("Nearest positional comparables")
        st.dataframe(
            comparable[
                [
                    "PlayerName",
                    "TeamName",
                    "AgeYears",
                    "MinutesPlayed",
                    "ScoutFitScore",
                    "Archetype",
                    "CompositeRecruitmentScore",
                    "ValueRecruitmentScore",
                ]
            ].round(2),
            width="stretch",
            hide_index=True,
        )

with tab_analytics:
    st.subheader("Advanced visuals")
    if filtered.empty:
        st.info("No players match the active filters.")
    else:
        visual_cols = st.columns([1, 1, 1])
        with visual_cols[0]:
            visual_metric = st.selectbox(
                "Visual metric",
                SCORE_COLUMNS,
                index=SCORE_COLUMNS.index("ScoutFitScore") if "ScoutFitScore" in SCORE_COLUMNS else SCORE_COLUMNS.index("CompositeRecruitmentScore"),
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

with tab_market:
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

with tab_exports:
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
            ["Current filtered board", "Single league / bundle", "Single position group", "Single player"],
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
