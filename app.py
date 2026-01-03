import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="RWB Scout – ASA Style",
    layout="wide",
    page_icon="⚽",
)

DATA_FILE = "RWB.xlsx"

# Column names from your file
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

METRICS = [
    "Progressive Receiving",
    "Progressive Receiving - where to? - Wide Right Zone",
    "Distance Covered Dribbles - Dribble",
    "Breaking Opponent Defence",
    "Breaking Opponent Defence - where from? - Wide Right Zone",
    "High Cross",
    "Low Cross",
    "Defensive Ball Control",
    "Number of presses during opponent build-up",
    "IMPECT",
    "Offensive IMPECT",
    "Defensive IMPECT",
]

ROLE_DEFS = {
    "Attacking Wingback Score": {
        "High Cross": 0.30,
        "Low Cross": 0.25,
        "Offensive IMPECT": 0.25,
        "Progressive Receiving - where to? - Wide Right Zone": 0.20,
    },
    "Progressor Score": {
        "Progressive Receiving": 0.30,
        "Breaking Opponent Defence": 0.30,
        "Distance Covered Dribbles - Dribble": 0.25,
        "Breaking Opponent Defence - where from? - Wide Right Zone": 0.15,
    },
    "Defensive Wingback Score": {
        "Defensive Ball Control": 0.30,
        "Number of presses during opponent build-up": 0.30,
        "Defensive IMPECT": 0.30,
        "IMPECT": 0.10,
    },
    "Balanced Score": {
        "IMPECT": 0.30,
        "Offensive IMPECT": 0.20,
        "Defensive IMPECT": 0.20,
        "Progressive Receiving": 0.15,
        "High Cross": 0.15,
    },
}

# =====================================================
# HELPERS
# =====================================================
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """In-place numeric coercion that prevents 'object' numeric columns from breaking Styler."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def percentile_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    sd = s.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean(skipna=True)) / sd

def score_from_z(z: pd.Series) -> pd.Series:
    z = pd.to_numeric(z, errors="coerce").fillna(0.0)
    return (50 + 15 * z).clip(0, 100)

def safe_num_fmt(x, decimals=2):
    """Never crashes if x is str/None."""
    try:
        if pd.isna(x):
            return "—"
    except Exception:
        pass

    try:
        v = float(x)
        if decimals == 0:
            return f"{v:,.0f}"
        return f"{v:,.{decimals}f}"
    except Exception:
        return str(x)

def style_table(df: pd.DataFrame, pct_cols: list[str] | None = None):
    """ASA-like table: percentiles heatmapped; robust formatting (no f-string crashes)."""
    pct_cols = pct_cols or []
    pct_cols = [c for c in pct_cols if c in df.columns]

    # Apply callable formatters column-by-column (safe for mixed types)
    formatters = {}
    for c in df.columns:
        if c in pct_cols:
            formatters[c] = lambda x, _d=0: safe_num_fmt(x, decimals=_d)
        elif pd.api.types.is_numeric_dtype(df[c]):
            formatters[c] = lambda x, _d=2: safe_num_fmt(x, decimals=_d)

    styler = (
        df.style
        .format(formatters)  # callable per column; safe for objects
        .set_properties(**{"font-size": "12px"})
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "700"), ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "6px 8px")]}
        ])
    )

    # Heatmap only on numeric percentiles
    if pct_cols:
        # ensure numeric for gradient
        tmp = df[pct_cols].apply(pd.to_numeric, errors="coerce")
        styler = styler.background_gradient(subset=pct_cols, cmap="Greys")

    return styler

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"Missing {file_path}. Put it next to app.py.")

    df = pd.read_excel(fp)
    df.columns = [c.strip() for c in df.columns]

    # Coerce numeric columns hard (prevents Styler crash)
    coerce_numeric(df, METRICS + [AGE_COL, SHARE_COL])

    # Ensure text columns are strings (safe searching)
    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    return df

df_raw = load_data(DATA_FILE)

# =====================================================
# BUILD MODEL (percentiles + roles)
# =====================================================
df = df_raw.copy()

# Percentiles for base metrics
for m in METRICS:
    if m in df.columns:
        df[m + " (pct)"] = percentile_rank(df[m])

# Role scores
for role, weights in ROLE_DEFS.items():
    z = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col in df.columns:
            z = z + (zscore(df[col]) * float(w))
    df[role] = score_from_z(z)

# Force role scores numeric (safety)
coerce_numeric(df, list(ROLE_DEFS.keys()))

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ RWB Scout")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Big Search", "Player Profile", "Role Leaderboards", "Distributions"]
)

min_share = st.sidebar.slider("Min Match Share", 0.0, 1.0, 0.20, 0.05)

df_f = df.copy()
if SHARE_COL in df_f.columns:
    df_f = df_f[df_f[SHARE_COL].fillna(0) >= min_share]

# =====================================================
# HEADER
# =====================================================
st.title("⚽ RWB Scout")
st.caption("ASA-style tables · role scores · percentiles · search")

# =====================================================
# DASHBOARD
# =====================================================
if page == "Dashboard":
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", len(df_f))
    c2.metric("Teams", df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0)
    c3.metric("Competitions", df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0)

    st.subheader("Top Balanced RWB")

    show_cols = [
        NAME_COL, TEAM_COL, COMP_COL, AGE_COL,
        "Balanced Score", "Attacking Wingback Score",
        "Progressor Score", "Defensive Wingback Score"
    ]
    show_cols = [c for c in show_cols if c in df_f.columns]

    top = df_f.sort_values("Balanced Score", ascending=False).head(30)[show_cols].copy()

    # add percentiles for scores
    score_cols = [c for c in top.columns if c.endswith("Score")]
    for c in score_cols:
        top[c + " (pct)"] = percentile_rank(df_f[c]).reindex(top.index)

    pct_cols = [c for c in top.columns if c.endswith("(pct)")]

    st.dataframe(
        style_table(top, pct_cols=pct_cols),
        width="stretch",
        height=520
    )

    st.subheader("Top 15 Balanced Score (Bar)")
    fig = px.bar(
        df_f.sort_values("Balanced Score", ascending=False).head(15),
        x="Balanced Score",
        y=NAME_COL,
        orientation="h",
        color="Balanced Score",
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, width="stretch")

# =====================================================
# BIG SEARCH
# =====================================================
elif page == "Big Search":
    st.subheader("Big Search")
    q = st.text_input("Search player / team / competition / nationality", placeholder="Type anything...")

    df_s = df_f.copy()
    if q:
        ql = q.lower().strip()
        mask = pd.Series(False, index=df_s.index)
        for col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            if col in df_s.columns:
                mask = mask | df_s[col].astype(str).str.lower().str.contains(ql, na=False)
        df_s = df_s[mask]

    st.caption("Sorted by Balanced Score. Use sidebar for Match Share filter.")

    base_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in df_s.columns]
    role_cols = [c for c in ROLE_DEFS.keys() if c in df_s.columns]
    out_cols = base_cols + role_cols

    out = df_s.sort_values("Balanced Score", ascending=False)[out_cols].copy()
    for c in role_cols:
        out[c + " (pct)"] = percentile_rank(df_s[c]).reindex(out.index)

    pct_cols = [c for c in out.columns if c.endswith("(pct)")]

    st.dataframe(style_table(out, pct_cols=pct_cols), width="stretch", height=720)

    st.download_button(
        "Download filtered table (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="rwb_search_export.csv",
        mime="text/csv",
    )

# =====================================================
# PLAYER PROFILE
# =====================================================
elif page == "Player Profile":
    st.subheader("Player Profile")

    players = sorted(df_f[NAME_COL].dropna().unique().tolist()) if NAME_COL in df_f.columns else []
    if not players:
        st.warning("No players available under current filters.")
        st.stop()

    player = st.selectbox("Select player", players)
    p = df_f[df_f[NAME_COL] == player].head(1)
    if p.empty:
        st.warning("Player not found.")
        st.stop()

    row = p.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Team", str(row.get(TEAM_COL, "—")))
    c2.metric("Competition", str(row.get(COMP_COL, "—")))
    c3.metric("Age", "—" if pd.isna(row.get(AGE_COL, np.nan)) else int(row.get(AGE_COL)))
    c4.metric("Match Share", "—" if pd.isna(row.get(SHARE_COL, np.nan)) else f"{row.get(SHARE_COL):.2f}")

    radar_cols = [c for c in ROLE_DEFS.keys() if c in df_f.columns]
    if radar_cols:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[float(row.get(c, np.nan)) for c in radar_cols],
            theta=[c.replace(" Score", "") for c in radar_cols],
            fill="toself",
            name=player
        ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=520)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Player Row")
    show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] + radar_cols if c in df_f.columns]
    st.dataframe(p[show], width="stretch", height=120)

# =====================================================
# ROLE LEADERBOARDS
# =====================================================
elif page == "Role Leaderboards":
    st.subheader("Role Leaderboards")

    role = st.selectbox("Role", [r for r in ROLE_DEFS.keys() if r in df_f.columns])
    n = st.slider("Rows", 10, 100, 40, 5)

    cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, role] if c in df_f.columns]
    out = df_f.sort_values(role, ascending=False).head(n)[cols].copy()
    out[role + " (pct)"] = percentile_rank(df_f[role]).reindex(out.index)

    st.dataframe(
        style_table(out, pct_cols=[role + " (pct)"]),
        width="stretch",
        height=720
    )

    fig = px.bar(
        df_f.sort_values(role, ascending=False).head(20),
        x=role,
        y=NAME_COL,
        orientation="h",
        color=role,
        hover_data=[TEAM_COL, COMP_COL] if TEAM_COL in df_f.columns and COMP_COL in df_f.columns else None,
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=650)
    st.plotly_chart(fig, width="stretch")

# =====================================================
# DISTRIBUTIONS
# =====================================================
elif page == "Distributions":
    st.subheader("Distributions")

    numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available.")
        st.stop()

    metric = st.selectbox("Metric", numeric_cols, index=(numeric_cols.index("Balanced Score") if "Balanced Score" in numeric_cols else 0))

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(df_f, x=metric, nbins=30)
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, width="stretch")

    with c2:
        fig2 = px.box(df_f, y=metric, points="all")
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, width="stretch")
