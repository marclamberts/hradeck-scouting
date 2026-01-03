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

# =====================================================
# HELPERS
# =====================================================
def percentile_rank(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index)
    mask = s.notna()
    out.loc[mask] = s[mask].rank(pct=True)
    return out * 100

def zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0 or s.isna().all():
        return pd.Series(0, index=s.index)
    return (s - s.mean()) / s.std()

def score_from_z(z: pd.Series) -> pd.Series:
    return (50 + 15 * z).clip(0, 100)

def style_table(df, value_cols, pct_cols=None):
    pct_cols = pct_cols or []
    fmt = {}

    for c in value_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            fmt[c] = "{:,.2f}"

    for c in pct_cols:
        if c in df.columns:
            fmt[c] = "{:.0f}"

    styler = (
        df.style
        .format(fmt, na_rep="—")
        .background_gradient(subset=pct_cols, cmap="Greys")
        .set_properties(**{"font-size": "12px"})
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "700"), ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "6px 8px")]}
        ])
    )
    return styler

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]

    for c in METRICS + [AGE_COL, SHARE_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

df_raw = load_data()

# =====================================================
# PERCENTILES
# =====================================================
df = df_raw.copy()
for m in METRICS:
    if m in df.columns:
        df[m + " (pct)"] = percentile_rank(df[m])

# =====================================================
# ROLE SCORES (RWB)
# =====================================================
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

for role, weights in ROLE_DEFS.items():
    z = 0
    for col, w in weights.items():
        if col in df.columns:
            z += zscore(df[col]) * w
    df[role] = score_from_z(z)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ RWB Scout")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Big Search", "Player Profile", "Role Leaderboards", "Distributions"]
)

min_share = st.sidebar.slider("Min Match Share", 0.0, 1.0, 0.20, 0.05)

df_f = df[df[SHARE_COL].fillna(0) >= min_share]

# =====================================================
# HEADER
# =====================================================
st.title("⚽ RWB Scout Dashboard")
st.caption("ASA-style scouting tables · role scores · percentiles")

# =====================================================
# DASHBOARD
# =====================================================
if page == "Dashboard":
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", len(df_f))
    c2.metric("Teams", df_f[TEAM_COL].nunique())
    c3.metric("Competitions", df_f[COMP_COL].nunique())

    st.subheader("Top Balanced RWB")

    show_cols = [
        NAME_COL, TEAM_COL, COMP_COL, AGE_COL,
        "Balanced Score", "Attacking Wingback Score",
        "Progressor Score", "Defensive Wingback Score"
    ]

    top = df_f.sort_values("Balanced Score", ascending=False).head(30)[show_cols]
    for c in show_cols:
        if c.endswith("Score"):
            top[c + " (pct)"] = percentile_rank(df_f[c]).reindex(top.index)

    pct_cols = [c for c in top.columns if c.endswith("(pct)")]
    val_cols = [c for c in top.columns if c not in pct_cols]

    st.dataframe(style_table(top, val_cols, pct_cols), use_container_width=True)

    fig = px.bar(
        df_f.sort_values("Balanced Score", ascending=False).head(15),
        x="Balanced Score",
        y=NAME_COL,
        orientation="h",
        color="Balanced Score",
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# BIG SEARCH
# =====================================================
elif page == "Big Search":
    q = st.text_input("Search player / team / competition / nationality")

    df_s = df_f.copy()
    if q:
        q = q.lower()
        mask = (
            df_s[NAME_COL].str.lower().str.contains(q, na=False)
            | df_s[TEAM_COL].str.lower().str.contains(q, na=False)
            | df_s[COMP_COL].str.lower().str.contains(q, na=False)
            | df_s[NAT_COL].str.lower().str.contains(q, na=False)
        )
        df_s = df_s[mask]

    st.dataframe(df_s.sort_values("Balanced Score", ascending=False), use_container_width=True, height=750)

# =====================================================
# PLAYER PROFILE
# =====================================================
elif page == "Player Profile":
    player = st.selectbox("Select player", sorted(df_f[NAME_COL].unique()))
    row = df_f[df_f[NAME_COL] == player].iloc[0]

    st.subheader(player)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Team", row[TEAM_COL])
    c2.metric("Competition", row[COMP_COL])
    c3.metric("Age", int(row[AGE_COL]) if not pd.isna(row[AGE_COL]) else "—")
    c4.metric("Match Share", f"{row[SHARE_COL]:.2f}")

    radar_cols = list(ROLE_DEFS.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[row[c] for c in radar_cols],
        theta=[c.replace(" Score", "") for c in radar_cols],
        fill="toself",
        name=player
    ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ROLE LEADERBOARDS
# =====================================================
elif page == "Role Leaderboards":
    role = st.selectbox("Role", list(ROLE_DEFS.keys()))
    out = df_f.sort_values(role, ascending=False).head(40)
    out[role + " (pct)"] = percentile_rank(df_f[role]).reindex(out.index)

    st.dataframe(
        style_table(out[[NAME_COL, TEAM_COL, COMP_COL, role, role + " (pct)"]],
                    [role],
                    [role + " (pct)"]),
        use_container_width=True
    )

# =====================================================
# DISTRIBUTIONS
# =====================================================
elif page == "Distributions":
    metric = st.selectbox("Metric", df_f.select_dtypes("number").columns)
    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(px.histogram(df_f, x=metric, nbins=30), use_container_width=True)
    with c2:
        st.plotly_chart(px.box(df_f, y=metric, points="all"), use_container_width=True)
