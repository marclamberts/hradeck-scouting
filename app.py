"""
Anomaly Platform — Pure z-score outlier intelligence from Wyscout per-90 data.

Pages:
  ⚡ Anomaly Hub      — global landscape overview
  🔍 Explorer         — filterable anomaly table
  👤 Player Profile   — individual z-score deep dive
  🔗 Similarity       — cosine / Pearson / Euclidean search
  ⚽ Set Pieces       — delivery, aerial & role anomalies
  📥 Export           — download Excel anomaly report
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

APP_DIR = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/anomaly_platform_matplotlib")
matplotlib.use("Agg")

from scouting_model import (
    AnomalyEngine,
    SimilarityEngine,
    SetPieceAnalyzer,
    SET_PIECE_ROLES,
)
from wyscout_model import (
    load_wyscout_raw,
    available_leagues as wyscout_available_leagues,
)

# ── Position-specific anomaly metric catalogue ────────────────────────────────
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
        "Exits per 90", "Aerial duels per 90.1",
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

ANOMALY_COLORS: dict[str, str] = {
    "Hidden Gem":               "#00c7b7",
    "Specialist Elite":         "#e76f51",
    "Multi-dimensional":        "#f4a261",
    "Age-adjusted Gem":         "#2a9d8f",
    "Consistent Overperformer": "#457b9d",
}

POSITION_ORDER = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
POSITION_COLORS = {
    "GK": "#667085", "CB": "#2f5f98", "FB": "#00a6a6",
    "DM": "#6b8e23", "CM": "#2f855a", "AM": "#d97706",
    "W":  "#e76f51", "ST": "#c2410c",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0c0f18 !important;
    color: #dde2f0 !important;
    font-family: "Inter", system-ui, -apple-system, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #080b12 !important;
    border-right: 1px solid #151b28 !important;
}
[data-testid="stSidebar"] * { color: #b8c2d8 !important; }
[data-testid="stSidebar"] label {
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #3a4258 !important;
}
.main .block-container {
    padding: 1.6rem 2.2rem !important;
    max-width: 100% !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
.page-title {
    font-size: 1.85rem;
    font-weight: 800;
    border-left: 4px solid #ee3a27;
    padding-left: 12px;
    margin-bottom: 2px;
    color: #eef2ff;
    line-height: 1.2;
}
.page-sub {
    font-size: 0.8rem;
    color: #4a5470;
    padding-left: 16px;
    margin-bottom: 20px;
}
.kpi-card {
    background: #111520;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    border: 1px solid #1a1f2e;
}
.kpi-num  { font-size: 2rem; font-weight: 800; color: #ee3a27; line-height: 1; }
.kpi-label { font-size: 0.65rem; color: #4a5470; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; }
.type-card {
    background: #111520;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    border-top: 3px solid;
    border-left: 1px solid #1a1f2e;
    border-right: 1px solid #1a1f2e;
    border-bottom: 1px solid #1a1f2e;
}
.player-card {
    background: #111520;
    border-radius: 10px;
    padding: 20px 24px;
    border: 1px solid #1a1f2e;
    margin-bottom: 16px;
}
.player-name { font-size: 1.4rem; font-weight: 800; color: #eef2ff; }
.player-meta { font-size: 0.82rem; color: #6a7390; margin-top: 4px; }
.section-divider {
    border: none;
    border-top: 1px solid #1a1f2e;
    margin: 24px 0;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #1a1f2e !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=600)
def _load_raw(leagues_key: tuple[str, ...] | None, min_minutes: int) -> pd.DataFrame:
    leagues = list(leagues_key) if leagues_key else None
    df = load_wyscout_raw(min_minutes=min_minutes, leagues=leagues)
    if df.empty:
        return df
    if "AgeYears" not in df.columns and "Age" in df.columns:
        df["AgeYears"] = pd.to_numeric(df["Age"], errors="coerce")
    mins_col = next((c for c in ["Minutes played", "MinutesPlayed"] if c in df.columns), None)
    if mins_col and "CompositeRecruitmentScore" not in df.columns:
        mins = pd.to_numeric(df[mins_col], errors="coerce").fillna(0)
        df["CompositeRecruitmentScore"] = (
            (mins - mins.min()) / (mins.max() - mins.min() + 1e-9) * 100
        ).clip(0, 100)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=600)
def _run_anomaly_detection(data_json: str, threshold: float, method: str) -> pd.DataFrame:
    df = pd.read_json(data_json)
    frames: list[pd.DataFrame] = []
    for pos_group, grp in df.groupby("PositionGroup"):
        metrics = [m for m in POSITION_METRICS.get(str(pos_group), []) if m in grp.columns]
        if not metrics or len(grp) < 5:
            frames.append(grp)
            continue
        engine = AnomalyEngine(threshold=threshold, method=method, groupby=None)
        try:
            frames.append(engine.fit_transform(grp, metrics))
        except Exception:
            frames.append(grp)
    return pd.concat(frames, ignore_index=True) if frames else df


# ── Excel export ──────────────────────────────────────────────────────────────

def _build_anomaly_excel(anomalies: pd.DataFrame, zdf: pd.DataFrame, threshold: float) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook()
    header_fill = PatternFill("solid", fgColor="0C0F18")
    accent_fill = PatternFill("solid", fgColor="EE3A27")
    white_font  = Font(color="FFFFFF", bold=True, size=10)
    thin_border = Border(
        left=Side(style="thin", color="1A1F2E"),
        right=Side(style="thin", color="1A1F2E"),
        top=Side(style="thin", color="1A1F2E"),
        bottom=Side(style="thin", color="1A1F2E"),
    )

    def _sheet(wb, title: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        ws = wb.create_sheet(title=title[:31])
        ws.cell(row=1, column=1, value=title)
        ws.cell(row=1, column=1).font = Font(color="FFFFFF", bold=True, size=12)
        ws.cell(row=1, column=1).fill = header_fill
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df.columns))
        ws.row_dimensions[1].height = 22
        for ci, col_name in enumerate(df.columns, start=1):
            cell = ws.cell(row=2, column=ci, value=col_name)
            cell.font = white_font
            cell.fill = accent_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border
        for ri, row in enumerate(dataframe_to_rows(df.reset_index(drop=True), index=False, header=False), start=3):
            for ci, val in enumerate(row, start=1):
                cell = ws.cell(row=ri, column=ci, value=val)
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="left" if ci <= 3 else "center")
                if ri % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor="0F1420")
        for col_cells in ws.columns:
            width = max(len(str(col_cells[1].value or "")),
                        *(len(str(c.value or "")) for c in col_cells[2:8]))
            ws.column_dimensions[col_cells[0].column_letter].width = min(width + 3, 40)

    DISPLAY_COLS = [
        "Player", "Team", "Position", "PositionGroup", "Age", "_League",
        "Minutes played", "Matches played",
        "_anomaly_type", "_anomaly_score", "_peak_z", "_mean_z", "_anomaly_breadth",
        "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
        "Key passes per 90", "Passes per 90", "Accurate passes, %",
        "Successful defensive actions per 90", "Progressive passes per 90",
    ]

    rename_map = {
        "_League": "League", "_anomaly_type": "Anomaly Type",
        "_anomaly_score": "Score", "_peak_z": "Peak Z",
        "_mean_z": "Mean Z", "_anomaly_breadth": "Breadth",
    }

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in DISPLAY_COLS if c in df.columns]
        return df[cols].rename(columns=rename_map).round(3).reset_index(drop=True)

    ws = wb.active
    ws.title = "Overview"
    ws.cell(row=1, column=1, value=f"Anomaly Platform Export  |  threshold={threshold:.1f}  |  {len(anomalies):,} anomalies")
    ws.cell(row=1, column=1).font = Font(color="FFFFFF", bold=True, size=13)
    ws.cell(row=1, column=1).fill = header_fill
    ws.merge_cells("A1:H1")
    ws.row_dimensions[1].height = 26

    for i, (atype, color) in enumerate(ANOMALY_COLORS.items(), start=3):
        subset = anomalies[anomalies["_anomaly_type"] == atype] if "_anomaly_type" in anomalies.columns else pd.DataFrame()
        ws.cell(row=i, column=1, value=atype)
        ws.cell(row=i, column=2, value=len(subset))
        ws.cell(row=i, column=1).font = Font(color="FFFFFF", bold=True)
        ws.cell(row=i, column=1).fill = PatternFill("solid", fgColor=color.lstrip("#"))

    _sheet(wb, "All Anomalies",          _prep(anomalies.head(2000)))
    _sheet(wb, "Hidden Gems",            _prep(anomalies[anomalies.get("_anomaly_type", pd.Series()) == "Hidden Gem"]))
    _sheet(wb, "Specialist Elite",       _prep(anomalies[anomalies.get("_anomaly_type", pd.Series()) == "Specialist Elite"]))
    _sheet(wb, "Multi-dimensional",      _prep(anomalies[anomalies.get("_anomaly_type", pd.Series()) == "Multi-dimensional"]))
    _sheet(wb, "Age-adjusted Gems",      _prep(anomalies[anomalies.get("_anomaly_type", pd.Series()) == "Age-adjusted Gem"]))
    _sheet(wb, "Consistent Overperf",    _prep(anomalies[anomalies.get("_anomaly_type", pd.Series()) == "Consistent Overperformer"]))

    # Set piece sheet
    try:
        sp = SetPieceAnalyzer(threshold=threshold * 0.85)
        sp_enriched = sp.fit_transform(zdf)
        sp_table = sp.anomaly_table(sp_enriched, top_n=300)
        _sheet(wb, "Set Piece Anomalies", sp_table)
    except Exception:
        pass

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _player_col(df: pd.DataFrame) -> str:
    return "Player" if "Player" in df.columns else "PlayerName"


def _team_col(df: pd.DataFrame) -> str:
    return "Team" if "Team" in df.columns else "TeamName"


def _anomaly_table(anomalies: pd.DataFrame, height: int = 500) -> None:
    disp = [c for c in [
        _player_col(anomalies), _team_col(anomalies), "PositionGroup", "_League", "Age",
        "_anomaly_type", "_anomaly_score", "_peak_z", "_mean_z", "_anomaly_breadth",
        "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
        "Key passes per 90", "Successful defensive actions per 90",
    ] if c in anomalies.columns]

    rename = {
        _player_col(anomalies): "Player", _team_col(anomalies): "Team",
        "_League": "League", "_anomaly_type": "Type",
        "_anomaly_score": "Score", "_peak_z": "Peak Z",
        "_mean_z": "Mean Z", "_anomaly_breadth": "Breadth",
    }
    st.dataframe(
        anomalies[disp].rename(columns=rename).round(3),
        use_container_width=True,
        height=height,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=18, format="%.2f"),
            "Peak Z": st.column_config.NumberColumn("Peak Z", format="%.2f"),
        },
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 8px 4px;'>
        <div style='font-size:1rem;font-weight:800;color:#ee3a27;letter-spacing:.05em;'>
            ANOMALY PLATFORM
        </div>
        <div style='font-size:0.65rem;color:#2a3048;text-transform:uppercase;letter-spacing:.12em;'>
            Pure z-score intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["⚡ Anomaly Hub", "🔍 Explorer", "👤 Player Profile", "🔗 Similarity", "⚽ Set Pieces", "📥 Export"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div style='font-size:.65rem;color:#2a3048;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Data</div>", unsafe_allow_html=True)

    all_leagues = wyscout_available_leagues()
    sel_leagues = st.multiselect(
        "Leagues", all_leagues, default=[],
        placeholder="All leagues", key="sel_leagues", label_visibility="collapsed",
    )
    leagues_key: tuple[str, ...] | None = tuple(sorted(sel_leagues)) if sel_leagues else None

    min_minutes = st.slider("Min minutes", 0, 2500, 450, 50, key="min_min")

    st.markdown("<div style='font-size:.65rem;color:#2a3048;text-transform:uppercase;letter-spacing:.1em;margin-top:12px;margin-bottom:6px;'>Detection</div>", unsafe_allow_html=True)

    threshold = st.slider("Anomaly threshold (z)", 1.0, 3.0, 1.8, 0.1, key="threshold")
    method_options = {"Z-score": "z-score", "MAD (robust)": "mad", "IQR": "iqr"}
    method_label = st.selectbox("Method", list(method_options.keys()), key="method", label_visibility="collapsed")
    method = method_options[method_label]


# ── Load & process data ───────────────────────────────────────────────────────

with st.spinner("Loading Wyscout data…"):
    df = _load_raw(leagues_key, min_minutes)

if df.empty:
    st.error("No data found. Add Wyscout `.xlsx` files to `data/Wyscout DB/`.")
    st.stop()

with st.spinner("Running anomaly detection…"):
    zdf = _run_anomaly_detection(df.to_json(), threshold, method)

if "_peak_z" in zdf.columns:
    anomalies = zdf[zdf["_peak_z"] >= threshold].sort_values("_anomaly_score", ascending=False).reset_index(drop=True)
else:
    anomalies = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Anomaly Hub
# ══════════════════════════════════════════════════════════════════════════════
if page == "⚡ Anomaly Hub":
    st.markdown(
        "<div class='page-title'>Anomaly Hub</div>"
        "<div class='page-sub'>Global outlier landscape — all positions, all types, all leagues</div>",
        unsafe_allow_html=True,
    )

    n_players   = len(df)
    n_anomalies = len(anomalies)
    n_leagues   = df["_League"].nunique() if "_League" in df.columns else 0
    n_pos       = df["PositionGroup"].nunique() if "PositionGroup" in df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    for w, num, lbl in [
        (k1, f"{n_players:,}",   "Players analysed"),
        (k2, f"{n_anomalies:,}", "Anomalies detected"),
        (k3, n_leagues,          "Leagues"),
        (k4, n_pos,              "Position groups"),
    ]:
        w.markdown(
            f"<div class='kpi-card'><div class='kpi-num'>{num}</div>"
            f"<div class='kpi-label'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if anomalies.empty:
        st.info("No anomalies at the current threshold. Lower the z threshold in the sidebar.")
        st.stop()

    # ── Anomaly-type KPI row ──────────────────────────────────────────────────
    if "_anomaly_type" in anomalies.columns:
        type_counts = anomalies["_anomaly_type"].value_counts()
        tc_cols = st.columns(len(ANOMALY_COLORS))
        for i, (atype, color) in enumerate(ANOMALY_COLORS.items()):
            cnt = int(type_counts.get(atype, 0))
            tc_cols[i].markdown(
                f"<div class='type-card' style='border-top-color:{color};'>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{color};line-height:1;'>{cnt}</div>"
                f"<div style='font-size:0.6rem;color:#4a5470;text-transform:uppercase;letter-spacing:.07em;margin-top:4px;'>{atype}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Anomaly count stacked by position × type
        if "_anomaly_type" in anomalies.columns and "PositionGroup" in anomalies.columns:
            breakdown = (
                anomalies.groupby(["PositionGroup", "_anomaly_type"])
                .size()
                .reset_index(name="Count")
            )
            pos_order = [p for p in POSITION_ORDER[::-1] if p in breakdown["PositionGroup"].values]
            chart = (
                alt.Chart(breakdown)
                .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    y=alt.Y("PositionGroup:N", sort=pos_order, title=None,
                             axis=alt.Axis(labelColor="#6a7390", labelFontSize=12)),
                    x=alt.X("Count:Q", title="Anomalies detected",
                             axis=alt.Axis(labelColor="#6a7390", gridColor="#1a1f2e")),
                    color=alt.Color(
                        "_anomaly_type:N",
                        scale=alt.Scale(domain=list(ANOMALY_COLORS.keys()), range=list(ANOMALY_COLORS.values())),
                        title="Type",
                        legend=alt.Legend(labelColor="#8fa3b1", titleColor="#8fa3b1", orient="top"),
                    ),
                    tooltip=["PositionGroup", "_anomaly_type", "Count"],
                )
                .properties(height=320, title="Anomaly breakdown by position & type")
                .configure_view(fill="#111520", stroke=None)
                .configure(background="#111520")
                .configure_title(color="#6a7390", fontSize=11, anchor="start")
            )
            st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.markdown("**Top 20 anomalies**")
        top20_cols = [c for c in [
            _player_col(anomalies), "PositionGroup", "_anomaly_type", "_anomaly_score", "_peak_z"
        ] if c in anomalies.columns]
        top20 = anomalies.head(20)[top20_cols].copy()
        rename = {_player_col(anomalies): "Player", "_anomaly_type": "Type", "_anomaly_score": "Score", "_peak_z": "Peak Z"}
        st.dataframe(
            top20.rename(columns=rename).round(2),
            use_container_width=True,
            height=352,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=18, format="%.2f"),
            },
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Anomaly density by league ─────────────────────────────────────────────
    if "_League" in anomalies.columns and "_League" in df.columns:
        total_by_league  = df["_League"].value_counts().rename("Total")
        anom_by_league   = anomalies["_League"].value_counts().rename("Anomalies")
        league_df = (
            pd.concat([anom_by_league, total_by_league], axis=1)
            .dropna()
            .reset_index()
            .rename(columns={"index": "League"})
        )
        league_df.columns = ["League", "Anomalies", "Total"]
        league_df["Density"] = (league_df["Anomalies"] / league_df["Total"] * 100).round(1)
        league_df = league_df.sort_values("Density", ascending=False).head(30)

        ld1, ld2 = st.columns([3, 2])
        with ld1:
            lc = (
                alt.Chart(league_df)
                .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color="#ee3a27")
                .encode(
                    y=alt.Y("League:N", sort="-x", title=None,
                             axis=alt.Axis(labelColor="#6a7390", labelFontSize=11)),
                    x=alt.X("Density:Q", title="Anomaly density (%)",
                             axis=alt.Axis(labelColor="#6a7390", gridColor="#1a1f2e")),
                    tooltip=["League", "Anomalies", "Total", alt.Tooltip("Density:Q", format=".1f", title="Density %")],
                )
                .properties(height=max(240, len(league_df) * 18), title="Anomaly density by league")
                .configure_view(fill="#111520", stroke=None)
                .configure(background="#111520")
                .configure_title(color="#6a7390", fontSize=11, anchor="start")
            )
            st.altair_chart(lc, use_container_width=True)

        with ld2:
            st.markdown("**League table**")
            st.dataframe(
                league_df.reset_index(drop=True),
                use_container_width=True,
                height=360,
                hide_index=True,
                column_config={"Density": st.column_config.ProgressColumn("Density %", min_value=0, max_value=100, format="%.1f")},
            )

    # ── Age distribution of anomalies ─────────────────────────────────────────
    if "Age" in anomalies.columns or "AgeYears" in anomalies.columns:
        age_col = "Age" if "Age" in anomalies.columns else "AgeYears"
        age_s = pd.to_numeric(anomalies[age_col], errors="coerce").dropna()
        if not age_s.empty:
            age_df = age_s.value_counts().sort_index().reset_index()
            age_df.columns = ["Age", "Count"]
            age_chart = (
                alt.Chart(age_df)
                .mark_bar(color="#457b9d", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Age:Q", title="Age", axis=alt.Axis(labelColor="#6a7390", gridColor="#1a1f2e")),
                    y=alt.Y("Count:Q", title="Anomalies", axis=alt.Axis(labelColor="#6a7390", gridColor="#1a1f2e")),
                    tooltip=["Age", "Count"],
                )
                .properties(height=180, title="Age distribution of anomalies")
                .configure_view(fill="#111520", stroke=None)
                .configure(background="#111520")
                .configure_title(color="#6a7390", fontSize=11, anchor="start")
            )
            st.altair_chart(age_chart, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explorer":
    st.markdown(
        "<div class='page-title'>Anomaly Explorer</div>"
        "<div class='page-sub'>Filter, sort, and drill into the full anomaly dataset</div>",
        unsafe_allow_html=True,
    )

    if anomalies.empty:
        st.info("No anomalies at the current threshold. Adjust the z threshold in the sidebar.")
        st.stop()

    # ── Filters row ───────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_pos = st.multiselect("Position", POSITION_ORDER, default=[], key="exp_pos")
    with f2:
        all_types = sorted(anomalies["_anomaly_type"].unique()) if "_anomaly_type" in anomalies.columns else []
        sel_types = st.multiselect("Anomaly type", all_types, default=[], key="exp_type")
    with f3:
        leagues_avail = sorted(anomalies["_League"].unique()) if "_League" in anomalies.columns else []
        sel_exp_leagues = st.multiselect("League", leagues_avail, default=[], key="exp_leagues")
    with f4:
        age_range = st.slider("Age", 15, 45, (16, 36), key="exp_age")

    filtered = anomalies.copy()
    if sel_pos and "PositionGroup" in filtered.columns:
        filtered = filtered[filtered["PositionGroup"].isin(sel_pos)]
    if sel_types and "_anomaly_type" in filtered.columns:
        filtered = filtered[filtered["_anomaly_type"].isin(sel_types)]
    if sel_exp_leagues and "_League" in filtered.columns:
        filtered = filtered[filtered["_League"].isin(sel_exp_leagues)]
    age_col_f = "Age" if "Age" in filtered.columns else ("AgeYears" if "AgeYears" in filtered.columns else None)
    if age_col_f:
        age_s = pd.to_numeric(filtered[age_col_f], errors="coerce")
        filtered = filtered[age_s.between(age_range[0], age_range[1])]

    filtered = filtered.reset_index(drop=True)

    # Position tabs
    positions_present = [p for p in POSITION_ORDER if p in filtered.get("PositionGroup", pd.Series()).values]
    tab_labels = ["All"] + positions_present
    tabs = st.tabs(tab_labels)

    for tab_widget, label in zip(tabs, tab_labels):
        with tab_widget:
            subset = filtered if label == "All" else filtered[filtered["PositionGroup"] == label]
            st.caption(f"{len(subset):,} anomalies")
            if subset.empty:
                st.info("No results for this position.")
            else:
                _anomaly_table(subset, height=540)

    # Download
    st.markdown("---")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "⬇️ Download CSV",
            data=filtered.to_csv(index=False).encode(),
            file_name="anomalies.csv",
            mime="text/csv",
        )
    with ec2:
        st.download_button(
            "⬇️ Download Excel",
            data=_build_anomaly_excel(filtered, zdf, threshold),
            file_name="anomaly_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Player Profile
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Player Profile":
    st.markdown(
        "<div class='page-title'>Player Profile</div>"
        "<div class='page-sub'>Individual anomaly deep dive — z-score breakdown per position metric</div>",
        unsafe_allow_html=True,
    )

    if anomalies.empty:
        st.info("No anomalies available at the current threshold.")
        st.stop()

    pc = _player_col(anomalies)
    tc = _team_col(anomalies)

    options = (
        anomalies[pc].astype(str)
        + " | "
        + anomalies.get(tc, pd.Series("", index=anomalies.index)).astype(str)
        + " | "
        + anomalies.get("PositionGroup", pd.Series("", index=anomalies.index)).astype(str)
    ).tolist()

    col_sel, col_pos_filter = st.columns([3, 1])
    with col_pos_filter:
        pos_filter = st.selectbox("Filter by position", ["All"] + POSITION_ORDER, key="profile_pos_f")
    with col_sel:
        filtered_opts = options
        filtered_anom = anomalies.copy()
        if pos_filter != "All" and "PositionGroup" in anomalies.columns:
            filtered_anom = anomalies[anomalies["PositionGroup"] == pos_filter]
            filtered_opts = (
                filtered_anom[pc].astype(str)
                + " | "
                + filtered_anom.get(tc, pd.Series("", index=filtered_anom.index)).astype(str)
                + " | "
                + filtered_anom.get("PositionGroup", pd.Series("", index=filtered_anom.index)).astype(str)
            ).tolist()
        sel = st.selectbox("Select player", filtered_opts if filtered_opts else ["No anomalies"], key="profile_player")

    if not filtered_opts:
        st.info("No anomalies for this position.")
        st.stop()

    sel_name = sel.split(" | ")[0]
    row = filtered_anom.loc[filtered_anom[pc] == sel_name].iloc[0]
    pos_group = str(row.get("PositionGroup", ""))
    atype     = str(row.get("_anomaly_type", ""))
    acolor    = ANOMALY_COLORS.get(atype, "#888")
    team      = row.get(tc, "")
    league    = row.get("_League", "")
    age       = row.get("Age", row.get("AgeYears", ""))
    peak_z    = row.get("_peak_z", 0)
    score     = row.get("_anomaly_score", 0)
    breadth   = int(row.get("_anomaly_breadth", 0))
    mins      = row.get("Minutes played", row.get("MinutesPlayed", ""))

    # ── Player header card ────────────────────────────────────────────────────
    st.markdown(
        f"<div class='player-card'>"
        f"<div class='player-name'>{sel_name}</div>"
        f"<div class='player-meta'>{team}  ·  {league}  ·  {pos_group}  ·  Age {age}  ·  {mins} min</div>"
        f"<div style='margin-top:12px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;'>"
        f"<span style='background:{acolor};color:#fff;padding:3px 14px;border-radius:12px;font-size:.72rem;font-weight:700;'>{atype}</span>"
        f"<span style='color:#6a7390;font-size:.8rem;'>Score <b style='color:{acolor};'>{score:.2f}</b></span>"
        f"<span style='color:#6a7390;font-size:.8rem;'>Peak Z <b style='color:{acolor};'>{peak_z:.2f}</b></span>"
        f"<span style='color:#6a7390;font-size:.8rem;'>Anomalous metrics <b style='color:{acolor};'>{breadth}</b></span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Z-score bar chart ─────────────────────────────────────────────────────
    z_metrics = POSITION_METRICS.get(pos_group, [])
    z_data = [
        {"Metric": m, "Z": float(row[f"_z_{m}"]), "Above": float(row[f"_z_{m}"]) >= threshold}
        for m in z_metrics if f"_z_{m}" in row.index
    ]

    if z_data:
        z_df = pd.DataFrame(z_data).sort_values("Z", ascending=False)
        z_df["fill"] = z_df.apply(
            lambda r: "#ee3a27" if r["Z"] >= threshold else ("#00c7b7" if r["Z"] >= 0 else "#2a3346"),
            axis=1,
        )
        ch1, ch2 = st.columns([3, 2])
        with ch1:
            z_chart = (
                alt.Chart(z_df)
                .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    y=alt.Y("Metric:N", sort=None, title=None,
                             axis=alt.Axis(labelColor="#8fa3b1", labelFontSize=11)),
                    x=alt.X("Z:Q", title="Z-score (relative to position peers)",
                             axis=alt.Axis(labelColor="#8fa3b1", gridColor="#1a1f2e")),
                    color=alt.Color("fill:N", scale=None, legend=None),
                    tooltip=["Metric", alt.Tooltip("Z:Q", format=".3f", title="Z-score")],
                )
                .properties(
                    height=max(280, len(z_df) * 28),
                    title=f"{pos_group} metrics — threshold = {threshold:.1f}",
                )
                .configure_view(fill="#111520", stroke=None)
                .configure(background="#111520")
                .configure_title(color="#6a7390", fontSize=11, anchor="start")
            )
            st.altair_chart(z_chart, use_container_width=True)

        with ch2:
            # Raw metric values
            st.markdown("**Raw metric values**")
            raw_data = []
            for m in z_metrics:
                val = row.get(m)
                if pd.notna(val):
                    raw_data.append({"Metric": m, "Value": round(float(val), 3)})
            if raw_data:
                raw_df = pd.DataFrame(raw_data)
                st.dataframe(raw_df, use_container_width=True, height=max(280, len(raw_df) * 35 + 60), hide_index=True)
    else:
        st.info("No z-score data available for this player's position metrics.")

    # ── Same-position anomaly peers ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"**Other {pos_group} anomalies** (top 10 by score)")
    peers = anomalies[anomalies.get("PositionGroup", pd.Series()) == pos_group]
    peers = peers[peers[pc] != sel_name].head(10)
    if not peers.empty:
        _anomaly_table(peers, height=380)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Similarity Engine
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Similarity":
    st.markdown(
        "<div class='page-title'>Similarity Engine</div>"
        "<div class='page-sub'>Find statistically similar players using cosine, Pearson, or Euclidean distance</div>",
        unsafe_allow_html=True,
    )

    if anomalies.empty:
        st.info("No anomalies available.")
        st.stop()

    pc = _player_col(anomalies)
    tc = _team_col(anomalies)

    options = (
        anomalies[pc].astype(str)
        + " | "
        + anomalies.get(tc, pd.Series("", index=anomalies.index)).astype(str)
        + " | "
        + anomalies.get("PositionGroup", pd.Series("", index=anomalies.index)).astype(str)
    ).tolist()

    sc1, sc2, sc3, sc4 = st.columns([3, 1, 1, 1])
    with sc1:
        sel = st.selectbox("Target player (anomaly)", options, key="sim_player")
    with sc2:
        n_similar = st.slider("Top N", 5, 30, 10, key="sim_n")
    with sc3:
        same_pos = st.checkbox("Same position", value=True, key="sim_pos")
    with sc4:
        sim_method = st.selectbox("Method", ["cosine", "pearson", "euclidean"], key="sim_method", label_visibility="collapsed")

    sel_name = sel.split(" | ")[0]
    target   = anomalies.loc[anomalies[pc] == sel_name].iloc[0]
    pos_group = str(target.get("PositionGroup", ""))
    atype     = str(target.get("_anomaly_type", ""))
    acolor    = ANOMALY_COLORS.get(atype, "#888")

    st.markdown(
        f"<div style='background:#111520;border-radius:8px;padding:12px 18px;border:1px solid #1a1f2e;margin-bottom:16px;'>"
        f"<span style='font-weight:700;color:#eef2ff;'>{sel_name}</span>"
        f"&nbsp;&nbsp;"
        f"<span style='color:#6a7390;font-size:.8rem;'>{target.get(tc,'')}  ·  {pos_group}  ·  "
        f"<span style='background:{acolor};color:#fff;padding:1px 8px;border-radius:10px;font-size:.7rem;font-weight:700;'>{atype}</span></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    feats  = [f for f in SIMILARITY_FEATURES if f in zdf.columns]
    engine = SimilarityEngine(feats)
    results = engine.find_similar(zdf, target, method=sim_method, n=n_similar, same_position=same_pos)

    if results.empty:
        st.info("No similar players found with the current settings.")
    else:
        pc_r = _player_col(results)
        tc_r = _team_col(results)
        disp_cols = [c for c in [pc_r, tc_r, "PositionGroup", "_League", "Age", "_similarity"] + feats[:8] if c in results.columns]
        rename = {pc_r: "Player", tc_r: "Team", "_League": "League", "_similarity": "Similarity"}
        st.dataframe(
            results[disp_cols].rename(columns=rename).round(3),
            use_container_width=True,
            height=420,
            hide_index=True,
            column_config={
                "Similarity": st.column_config.ProgressColumn(
                    "Similarity", min_value=-1.0, max_value=1.0, format="%.3f"
                ),
            },
        )
        st.caption(f"{len(results)} similar players · method: {sim_method}")

    # Compare all three methods
    with st.expander("Compare all three similarity methods"):
        method_tabs = st.tabs(["Cosine", "Pearson", "Euclidean"])
        for mt, mn in zip(method_tabs, ["cosine", "pearson", "euclidean"]):
            with mt:
                res_m = engine.find_similar(zdf, target, method=mn, n=n_similar, same_position=same_pos)
                if res_m.empty:
                    st.info("No results.")
                else:
                    pc_m = _player_col(res_m)
                    disp_m = [c for c in [pc_m, "PositionGroup", "_League", "Age", "_similarity"] if c in res_m.columns]
                    st.dataframe(
                        res_m[disp_m].rename(columns={pc_m: "Player", "_League": "League", "_similarity": "Sim"}).round(3),
                        use_container_width=True,
                        height=300,
                        hide_index=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Set Pieces
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚽ Set Pieces":
    st.markdown(
        "<div class='page-title'>Set Piece Scanner</div>"
        "<div class='page-sub'>Delivery, aerial, and shooting anomalies — profiled by role</div>",
        unsafe_allow_html=True,
    )

    sp_analyzer  = SetPieceAnalyzer(threshold=threshold * 0.85)
    ws_enriched  = sp_analyzer.fit_transform(df)
    sp_anom_df   = sp_analyzer.anomaly_table(ws_enriched, top_n=300)
    role_leaders = sp_analyzer.top_players_by_role(ws_enriched, top_n=25)

    # KPIs
    n_sp_anom = len(sp_anom_df)
    n_roles   = len(role_leaders)
    sp_k1, sp_k2 = st.columns(2)
    sp_k1.markdown(
        f"<div class='kpi-card'><div class='kpi-num'>{n_sp_anom}</div>"
        f"<div class='kpi-label'>Set-piece anomalies</div></div>",
        unsafe_allow_html=True,
    )
    sp_k2.markdown(
        f"<div class='kpi-card'><div class='kpi-num'>{n_roles}</div>"
        f"<div class='kpi-label'>Roles profiled</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Overall SP anomaly table
    if not sp_anom_df.empty:
        st.markdown("**Set-piece anomaly table** (sorted by peak z)")
        st.dataframe(sp_anom_df.round(3), use_container_width=True, height=380, hide_index=True)
    else:
        st.info("No set-piece anomalies at the current threshold.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**Top players by set-piece role**")

    role_names = list(SET_PIECE_ROLES.keys())
    role_tabs  = st.tabs(role_names)
    for rtab, role_name in zip(role_tabs, role_names):
        with rtab:
            role_df = role_leaders.get(role_name, pd.DataFrame())
            if role_df.empty:
                st.info(f"No data for {role_name}.")
            else:
                st.dataframe(role_df.round(3), use_container_width=True, height=460, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Export
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Export":
    st.markdown(
        "<div class='page-title'>Export</div>"
        "<div class='page-sub'>Download the full anomaly report as Excel or CSV</div>",
        unsafe_allow_html=True,
    )

    # Summary
    n_total = len(anomalies)
    st.markdown(f"**{n_total:,}** anomalies detected at z ≥ {threshold:.1f}")

    if not anomalies.empty and "_anomaly_type" in anomalies.columns:
        tc = anomalies["_anomaly_type"].value_counts()
        tc_cols = st.columns(len(tc))
        for i, (atype, cnt) in enumerate(tc.items()):
            color = ANOMALY_COLORS.get(atype, "#888")
            tc_cols[i % len(tc_cols)].markdown(
                f"<div class='type-card' style='border-top-color:{color};'>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{color};'>{cnt}</div>"
                f"<div style='font-size:.6rem;color:#4a5470;text-transform:uppercase;'>{atype}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Excel report** — all anomaly types in separate sheets, set pieces, similarity")
        if n_total > 0:
            xlsx_bytes = _build_anomaly_excel(anomalies, zdf, threshold)
            st.download_button(
                "⬇️ Download Excel (.xlsx)",
                data=xlsx_bytes,
                file_name="anomaly_platform_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Download Excel", disabled=True, use_container_width=True)

    with ec2:
        st.markdown("**CSV** — flat anomaly table, all columns")
        if n_total > 0:
            st.download_button(
                "⬇️ Download CSV",
                data=anomalies.to_csv(index=False).encode(),
                file_name="anomaly_platform.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Download CSV", disabled=True, use_container_width=True)

    # What's in the Excel
    st.markdown("---")
    st.markdown("""
**Excel sheets included:**
- **Overview** — summary stats and anomaly type counts
- **All Anomalies** — complete ranked list (up to 2,000)
- **Hidden Gems** — low-exposure, high-signal players
- **Specialist Elite** — elite in 1–2 metrics
- **Multi-dimensional** — 4+ metrics above threshold
- **Age-adjusted Gems** — ≤23 anomalies
- **Consistent Overperformer** — broad positive signal
- **Set Piece Anomalies** — set-piece outliers
""")
