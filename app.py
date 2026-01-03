import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="RWB Scout – ASA Style", layout="wide", page_icon="⚽")
DATA_FILE = "RWB.xlsx"

# Expected columns (match your sheet)
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
# SAFE FORMATTERS (FIXES "Unknown format code 'f' for str")
# =====================================================
def safe_float(x):
    """Convert typical Excel 'number as text' to float; otherwise NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-","—"}:
        return np.nan
    s = s.replace("%", "")
    # Handle decimal comma
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # Remove thousands separators like "1,234.5"
    if s.count(",") > 1 and s.count(".") == 1:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def safe_f(x, decimals=2):
    """Safe numeric display; never throws even if x is a string."""
    try:
        if pd.isna(x):
            return "—"
    except Exception:
        pass
    v = safe_float(x)
    if np.isnan(v):
        return str(x) if str(x).strip() else "—"
    return f"{v:.{decimals}f}"

def safe_int(x):
    v = safe_float(x)
    if np.isnan(v):
        return "—"
    return f"{v:,.0f}"

# =====================================================
# HELPERS
# =====================================================
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

def percentile_rank(s: pd.Series) -> pd.Series:
    s = s.apply(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def zscore(s: pd.Series) -> pd.Series:
    s = s.apply(safe_float)
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    sd = s.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean(skipna=True)) / sd

def score_from_z(z: pd.Series) -> pd.Series:
    z = z.apply(safe_float).fillna(0.0)
    return (50 + 15 * z).clip(0, 100)

def style_table(df: pd.DataFrame, pct_cols: list[str] | None = None):
    """Crash-proof Styler (ONLY callable formatters)."""
    pct_cols = pct_cols or []
    pct_cols = [c for c in pct_cols if c in df.columns]

    formatters = {}
    for c in df.columns:
        if c in pct_cols:
            formatters[c] = lambda x: safe_f(x, 0)
        else:
            # numeric dtype OR numeric-looking objects
            if pd.api.types.is_numeric_dtype(df[c]):
                # choose int-ish vs float-ish
                s = df[c].dropna()
                if len(s) and (np.abs(s - np.round(s)) < 1e-9).mean() > 0.9:
                    formatters[c] = safe_int
                else:
                    formatters[c] = lambda x: safe_f(x, 2)

    styler = (
        df.style
        .format(formatters)
        .set_properties(**{"font-size": "12px"})
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "700"), ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "6px 8px")]}
        ])
    )

    if pct_cols:
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
    df.columns = [str(c).strip() for c in df.columns]

    # Coerce numeric columns hard
    coerce_numeric(df, METRICS + [AGE_COL, SHARE_COL])

    # Ensure text columns safe
    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    return df

df_raw = load_data(DATA_FILE)
df = df_raw.copy()

# Build metric percentiles
for m in METRICS:
    if m in df.columns:
        df[m + " (pct)"] = percentile_rank(df[m])

# Role scores
for role, weights in ROLE_DEFS.items():
    z = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col in df.columns:
            z = z + zscore(df[col]) * float(w)
    df[role] = score_from_z(z)

coerce_numeric(df, list(ROLE_DEFS.keys()))

# =====================================================
# SIDEBAR NAV + FILTERS
# =====================================================
st.sidebar.title("⚙️ RWB Scout")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Big Search", "Player Profile", "Role Leaderboards", "Distributions", "Compare Players"]
)

min_share = st.sidebar.slider("Min Match Share", 0.0, 1.0, 0.20, 0.05)

df_f = df.copy()
if SHARE_COL in df_f.columns:
    df_f = df_f[df_f[SHARE_COL].fillna(0) >= min_share]

if COMP_COL in df_f.columns:
    comps = ["All"] + sorted([c for c in df_f[COMP_COL].dropna().unique().tolist() if str(c).strip() != ""])
    comp_pick = st.sidebar.selectbox("Competition", comps)
    if comp_pick != "All":
        df_f = df_f[df_f[COMP_COL] == comp_pick]

if TEAM_COL in df_f.columns:
    teams = ["All"] + sorted([t for t in df_f[TEAM_COL].dropna().unique().tolist() if str(t).strip() != ""])
    team_pick = st.sidebar.selectbox("Team", teams)
    if team_pick != "All":
        df_f = df_f[df_f[TEAM_COL] == team_pick]

# =====================================================
# HEADER
# =====================================================
st.title("⚽ RWB Scout")
st.caption("ASA-style tables · role scores · percentiles · search")

# =====================================================
# PAGES
# =====================================================
if page == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", len(df_f))
    c2.metric("Teams", df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0)
    c3.metric("Competitions", df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0)
    c4.metric("Avg Match Share", safe_f(df_f[SHARE_COL].mean(), 2) if SHARE_COL in df_f.columns and len(df_f) else "—")

    st.markdown("---")

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Top RWB (Balanced)")
        show_cols = [c for c in [
            NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL,
            "Balanced Score", "Attacking Wingback Score", "Progressor Score", "Defensive Wingback Score",
        ] if c in df_f.columns]

        top = df_f.sort_values("Balanced Score", ascending=False).head(30)[show_cols].copy()

        score_cols = [c for c in ["Balanced Score", "Attacking Wingback Score", "Progressor Score", "Defensive Wingback Score"] if c in df_f.columns]
        for c in score_cols:
            top[c + " (pct)"] = percentile_rank(df_f[c]).reindex(top.index)

        pct_cols = [c for c in top.columns if c.endswith("(pct)")]
        st.dataframe(style_table(top, pct_cols=pct_cols), width="stretch", height=640)

    with right:
        st.subheader("Role Landscape")
        if all(c in df_f.columns for c in ["Progressor Score", "Attacking Wingback Score", "Balanced Score"]):
            fig = px.scatter(
                df_f,
                x="Progressor Score",
                y="Attacking Wingback Score",
                color="Balanced Score",
                hover_data=[c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] if c in df_f.columns],
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Top 15 Balanced (Bar)")
        fig2 = px.bar(
            df_f.sort_values("Balanced Score", ascending=False).head(15),
            x="Balanced Score",
            y=NAME_COL,
            orientation="h",
            color="Balanced Score",
        )
        fig2.update_layout(height=520, yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

elif page == "Big Search":
    st.subheader("Big Search")
    q = st.text_input("Search player / team / competition / nationality", placeholder="Type anything…")

    df_s = df_f.copy()
    if q and q.strip():
        ql = q.strip().lower()
        mask = pd.Series(False, index=df_s.index)
        for col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            if col in df_s.columns:
                mask = mask | df_s[col].astype(str).str.lower().str.contains(ql, na=False)
        df_s = df_s[mask]

    base_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in df_s.columns]
    role_cols = [c for c in ROLE_DEFS.keys() if c in df_s.columns]
    key_metrics = [c for c in ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "High Cross", "Low Cross", "Breaking Opponent Defence"] if c in df_s.columns]

    out = df_s.sort_values("Balanced Score", ascending=False)[base_cols + role_cols + key_metrics].copy()
    for c in role_cols:
        out[c + " (pct)"] = percentile_rank(df_s[c]).reindex(out.index)

    pct_cols = [c for c in out.columns if c.endswith("(pct)")]
    st.dataframe(style_table(out, pct_cols=pct_cols), width="stretch", height=740)

    st.download_button(
        "Download filtered table (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="rwb_search_export.csv",
        mime="text/csv",
    )

elif page == "Player Profile":
    st.subheader("Player Profile")

    if NAME_COL not in df_f.columns or df_f.empty:
        st.warning("No players available with current filters.")
        st.stop()

    players = sorted(df_f[NAME_COL].dropna().unique().tolist())
    player = st.selectbox("Select player", players)

    p = df_f[df_f[NAME_COL] == player].head(1)
    if p.empty:
        st.warning("Player not found.")
        st.stop()

    row = p.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Team", str(row.get(TEAM_COL, "—")))
    c2.metric("Competition", str(row.get(COMP_COL, "—")))
    c3.metric("Age", safe_int(row.get(AGE_COL, np.nan)))
    c4.metric("Match Share", safe_f(row.get(SHARE_COL, np.nan), 2))

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Role Scores (percentiles)")
        for sc in [c for c in ROLE_DEFS.keys() if c in df_f.columns]:
            pct = percentile_rank(df_f[sc]).loc[p.index[0]]
            val = row.get(sc, np.nan)
            st.write(f"**{sc.replace(' Score','')}** — {safe_f(val,1)} | {safe_f(pct,0)}p")

        st.markdown("### Key Metrics (percentiles)")
        for m in [c for c in METRICS if c in df_f.columns]:
            val = row.get(m, np.nan)
            pct = row.get(m + " (pct)", np.nan)
            st.write(f"- {m}: **{safe_f(val,2)}** | **{safe_f(pct,0)}p**")

    with right:
        st.markdown("### Radar (Role Scores)")
        radar_cols = [c for c in ROLE_DEFS.keys() if c in df_f.columns]
        if radar_cols:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[safe_float(row.get(c, np.nan)) for c in radar_cols],
                theta=[c.replace(" Score", "") for c in radar_cols],
                fill="toself",
                name=player
            ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=520)
            st.plotly_chart(fig, width="stretch")

        st.markdown("### Player Row (table)")
        show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] + radar_cols if c in df_f.columns]
        st.dataframe(p[show], width="stretch", height=140)

elif page == "Role Leaderboards":
    st.subheader("Role Leaderboards")

    available_roles = [r for r in ROLE_DEFS.keys() if r in df_f.columns]
    if not available_roles:
        st.warning("No role score columns found.")
        st.stop()

    role = st.selectbox("Role", available_roles, index=available_roles.index("Balanced Score") if "Balanced Score" in available_roles else 0)
    n = st.slider("Rows", 10, 100, 40, 5)

    cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, role] if c in df_f.columns]
    out = df_f.sort_values(role, ascending=False).head(n)[cols].copy()
    out[role + " (pct)"] = percentile_rank(df_f[role]).reindex(out.index)

    st.dataframe(style_table(out, pct_cols=[role + " (pct)"]), width="stretch", height=740)

    st.markdown("### Top 20 Bar Chart")
    fig = px.bar(
        df_f.sort_values(role, ascending=False).head(20),
        x=role,
        y=NAME_COL,
        orientation="h",
        color=role,
        hover_data=[c for c in [TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] if c in df_f.columns],
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=650)
    st.plotly_chart(fig, width="stretch")

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

    st.markdown("### Mean by Competition / Team")
    split_options = [c for c in [COMP_COL, TEAM_COL] if c in df_f.columns]
    if split_options:
        split = st.radio("Split by", split_options, horizontal=True)
        topk = st.slider("Show top groups", 5, 30, 12, 1)
        g = df_f.groupby(split, dropna=True)[metric].mean().sort_values(ascending=False).head(topk).reset_index()
        fig3 = px.bar(g, x=metric, y=split, orientation="h", color=metric)
        fig3.update_layout(height=600, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig3, width="stretch")

elif page == "Compare Players":
    st.subheader("Compare Players")

    if NAME_COL not in df_f.columns or df_f.empty:
        st.warning("No players available with current filters.")
        st.stop()

    players = sorted(df_f[NAME_COL].dropna().unique().tolist())
    picks = st.multiselect("Select 2–6 players", players, default=players[:2])

    if len(picks) < 2:
        st.info("Pick at least 2 players to compare.")
        st.stop()

    comp_df = df_f[df_f[NAME_COL].isin(picks)].copy()

    st.markdown("### Role Scores Comparison")
    role_cols = [c for c in ROLE_DEFS.keys() if c in comp_df.columns]
    if role_cols:
        melt = comp_df.melt(
            id_vars=[c for c in [NAME_COL, TEAM_COL, COMP_COL] if c in comp_df.columns],
            value_vars=role_cols,
            var_name="Role",
            value_name="Score"
        )
        fig = px.bar(melt, x="Score", y=NAME_COL, color="Role", barmode="group")
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Radar (Key Metric Percentiles)")
    radar_metrics = [m + " (pct)" for m in [
        "Progressive Receiving",
        "Breaking Opponent Defence",
        "Distance Covered Dribbles - Dribble",
        "High Cross",
        "Low Cross",
        "Defensive Ball Control",
        "Number of presses during opponent build-up",
        "IMPECT",
    ] if (m + " (pct)") in comp_df.columns]

    if radar_metrics:
        fig2 = go.Figure()
        for name in picks:
            sub = comp_df[comp_df[NAME_COL] == name].head(1)
            if sub.empty:
                continue
            r = [safe_float(sub.iloc[0].get(m, np.nan)) for m in radar_metrics]
            theta = [m.replace(" (pct)", "") for m in radar_metrics]
            fig2.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name=name))
        fig2.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=620)
        st.plotly_chart(fig2, width="stretch")

    st.markdown("### Comparison Table")
    show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] + role_cols if c in comp_df.columns]
    st.dataframe(comp_df[show].sort_values("Balanced Score", ascending=False), width="stretch", height=520)
