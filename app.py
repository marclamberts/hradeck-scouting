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
# UI THEME (ASA-ish)
# =====================================================
st.markdown(
    """
<style>
/* App bg */
.stApp { background: #f6f7f9; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }

/* Tight headings */
h1, h2, h3 { letter-spacing: -0.02em; }

/* Buttons = nav items */
div.stButton > button {
  width: 100%;
  border-radius: 10px;
  border: 1px solid #e5e7eb;
  background: #ffffff;
  padding: 10px 12px;
  font-weight: 800;
  text-align: left;
}
div.stButton > button:hover { border-color: #111827; }

/* Active nav button wrapper */
.nav-active div.stButton > button {
  background: #111827 !important;
  color: #ffffff !important;
  border-color: #111827 !important;
}

/* Card */
.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px 14px;
}

/* Slightly reduce vertical spacing */
div[data-testid="stVerticalBlock"] > div { gap: 0.65rem; }

/* Dataframe header weight */
div[data-testid="stDataFrame"] thead tr th { font-weight: 900; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# SAFE PARSING + SAFE DISPLAY
# =====================================================
def safe_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}:
        return np.nan

    s = s.replace("%", "")

    # decimal comma
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # thousands separators
    if s.count(",") >= 1 and s.count(".") == 1:
        s = s.replace(",", "")

    try:
        return float(s)
    except Exception:
        return np.nan

def safe_fmt(x, decimals=2):
    v = safe_float(x)
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"

def safe_int_fmt(x):
    v = safe_float(x)
    if np.isnan(v):
        return "—"
    return f"{int(round(v))}"

# =====================================================
# HELPERS
# =====================================================
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(safe_float)

def percentile_rank(s: pd.Series) -> pd.Series:
    s = s.map(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def zscore(s: pd.Series) -> pd.Series:
    s = s.map(safe_float)
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    sd = s.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean(skipna=True)) / sd

def score_from_z(z: pd.Series) -> pd.Series:
    z = z.map(safe_float).fillna(0.0)
    return (50 + 15 * z).clip(0, 100)

# =====================================================
# "PRO TABLE" (NO PANDAS STYLER) + NICE COLUMN NAMES
# =====================================================
DISPLAY_RENAMES = {
    "Balanced Score": "Balanced",
    "Attacking Wingback Score": "Attacking WB",
    "Progressor Score": "Progressor",
    "Defensive Wingback Score": "Defensive WB",
    "Match Share": "Share",
}

def rename_for_display(df_: pd.DataFrame) -> pd.DataFrame:
    return df_.rename(columns=DISPLAY_RENAMES)

def pro_table(df: pd.DataFrame, pct_cols: list[str] | None = None, height: int = 600):
    pct_cols = pct_cols or []
    pct_cols = [c for c in pct_cols if c in df.columns]

    col_config = {}

    # Percentiles -> bars
    for c in pct_cols:
        col_config[c] = st.column_config.ProgressColumn(
            label=c,
            min_value=0,
            max_value=100,
            format="%.0f",
            help="Percentile (0–100)",
        )

    # Numeric columns -> 2 decimals
    for c in df.columns:
        if c in pct_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            col_config[c] = st.column_config.NumberColumn(label=c, format="%.2f")

    # Make name/team wider
    if NAME_COL in df.columns:
        col_config[NAME_COL] = st.column_config.TextColumn(label=NAME_COL, width="large")
    if TEAM_COL in df.columns:
        col_config[TEAM_COL] = st.column_config.TextColumn(label=TEAM_COL, width="medium")
    if COMP_COL in df.columns:
        col_config[COMP_COL] = st.column_config.TextColumn(label=COMP_COL, width="medium")

    st.dataframe(
        df,
        width="stretch",
        height=height,
        column_config=col_config,
        hide_index=True,
    )

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

    coerce_numeric(df, METRICS + [AGE_COL, SHARE_COL])

    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    return df

df = load_data(DATA_FILE).copy()

# Percentiles for metrics
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
# NAV BUTTONS (instead of radio)
# =====================================================
PAGES = [
    ("Dashboard", "🏠 Dashboard"),
    ("Big Search", "🔎 Big Search"),
    ("Player Profile", "👤 Player Profile"),
    ("Role Leaderboards", "🏆 Leaderboards"),
    ("Distributions", "📈 Distributions"),
    ("Compare Players", "🆚 Compare"),
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

def nav_button(page_key: str, label: str):
    active = (st.session_state.page == page_key)
    if active:
        st.sidebar.markdown('<div class="nav-active">', unsafe_allow_html=True)
    clicked = st.sidebar.button(label, key=f"nav_{page_key}")
    if active:
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
    if clicked:
        st.session_state.page = page_key

st.sidebar.markdown("### 🧭 Navigation")
for key, label in PAGES:
    nav_button(key, label)

page = st.session_state.page
st.sidebar.markdown("---")

# =====================================================
# FILTERS (nice + grouped)
# =====================================================
if "filters" not in st.session_state:
    st.session_state.filters = {
        "min_share": 0.20,
        "competitions": [],
        "teams": [],
        "q": "",
    }

def reset_filters():
    st.session_state.filters = {"min_share": 0.20, "competitions": [], "teams": [], "q": ""}

st.sidebar.markdown("### 🎛️ Filters")
with st.sidebar.container():
    st.sidebar.markdown('<div class="card">', unsafe_allow_html=True)

    st.session_state.filters["q"] = st.sidebar.text_input(
        "Quick search",
        value=st.session_state.filters["q"],
        placeholder="Name / Team / Comp / Nat…",
        key="filter_q",
    )

    st.session_state.filters["min_share"] = st.sidebar.slider(
        "Min Match Share",
        0.0, 1.0,
        float(st.session_state.filters["min_share"]),
        0.05,
        key="filter_min_share",
    )

    if COMP_COL in df.columns:
        comps_all = sorted([c for c in df[COMP_COL].dropna().unique().tolist() if str(c).strip() != ""])
        st.session_state.filters["competitions"] = st.sidebar.multiselect(
            "Competitions",
            comps_all,
            default=st.session_state.filters["competitions"],
            key="filter_comps",
        )

    if TEAM_COL in df.columns:
        teams_all = sorted([t for t in df[TEAM_COL].dropna().unique().tolist() if str(t).strip() != ""])
        st.session_state.filters["teams"] = st.sidebar.multiselect(
            "Teams",
            teams_all,
            default=st.session_state.filters["teams"],
            key="filter_teams",
        )

    cA, cB = st.sidebar.columns(2)
    with cA:
        if st.button("Reset filters", key="reset_filters"):
            reset_filters()
            st.rerun()
    with cB:
        st.caption("Applies live")

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Apply filters
df_f = df.copy()

if SHARE_COL in df_f.columns:
    df_f = df_f[df_f[SHARE_COL].fillna(0) >= st.session_state.filters["min_share"]]

comps_sel = st.session_state.filters.get("competitions", [])
if comps_sel and COMP_COL in df_f.columns:
    df_f = df_f[df_f[COMP_COL].isin(comps_sel)]

teams_sel = st.session_state.filters.get("teams", [])
if teams_sel and TEAM_COL in df_f.columns:
    df_f = df_f[df_f[TEAM_COL].isin(teams_sel)]

q = st.session_state.filters.get("q", "").strip().lower()
if q:
    mask = pd.Series(False, index=df_f.index)
    for col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if col in df_f.columns:
            mask = mask | df_f[col].astype(str).str.lower().str.contains(q, na=False)
    df_f = df_f[mask]

# =====================================================
# HEADER
# =====================================================
st.title("⚽ RWB Scout")
st.caption("ASA-style tables · role scores · percentile bars · button navigation · clean filters")

# =====================================================
# DASHBOARD
# =====================================================
if page == "Dashboard":
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Players", len(df_f))
    k2.metric("Teams", df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0)
    k3.metric("Competitions", df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0)
    avg_share = df_f[SHARE_COL].mean() if SHARE_COL in df_f.columns and len(df_f) else np.nan
    k4.metric("Avg Share", safe_fmt(avg_share, 2))

    st.markdown("---")

    # Quick view chips
    st.markdown("#### Quick views")
    chip1, chip2, chip3, chip4 = st.columns(4)
    if "quick_role" not in st.session_state:
        st.session_state.quick_role = "Balanced Score"
    if chip1.button("⚡ Balanced", key="chip_bal"):
        st.session_state.quick_role = "Balanced Score"
    if chip2.button("🚀 Progressor", key="chip_prog"):
        st.session_state.quick_role = "Progressor Score"
    if chip3.button("🎯 Attacking WB", key="chip_att"):
        st.session_state.quick_role = "Attacking Wingback Score"
    if chip4.button("🧱 Defensive WB", key="chip_def"):
        st.session_state.quick_role = "Defensive Wingback Score"

    role_sort = st.session_state.quick_role if st.session_state.quick_role in df_f.columns else "Balanced Score"

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader(f"Top RWB (sorted by {DISPLAY_RENAMES.get(role_sort, role_sort)})")

        base_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in df_f.columns]
        role_cols = [c for c in ["Balanced Score", "Attacking Wingback Score", "Progressor Score", "Defensive Wingback Score"] if c in df_f.columns]
        show_cols = base_cols + role_cols

        top = df_f.sort_values(role_sort, ascending=False).head(30)[show_cols].copy()

        for c in role_cols:
            top[c + " (pct)"] = percentile_rank(df_f[c]).reindex(top.index)

        pct_cols = [c for c in top.columns if c.endswith("(pct)")]
        top_disp = rename_for_display(top)
        pro_table(top_disp, pct_cols=pct_cols, height=640)

    with right:
        st.subheader("Role landscape")
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

        st.subheader("Top 15 (bar)")
        fig2 = px.bar(
            df_f.sort_values(role_sort, ascending=False).head(15),
            x=role_sort,
            y=NAME_COL,
            orientation="h",
            color=role_sort,
        )
        fig2.update_layout(height=520, yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

# =====================================================
# BIG SEARCH
# =====================================================
elif page == "Big Search":
    st.subheader("Big Search")
    st.caption("Uses the sidebar filters + quick search. Export the current filtered view.")

    base_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in df_f.columns]
    role_cols = [c for c in ROLE_DEFS.keys() if c in df_f.columns]
    key_metrics = [c for c in ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "High Cross", "Low Cross", "Breaking Opponent Defence"] if c in df_f.columns]

    out = df_f.sort_values("Balanced Score", ascending=False)[base_cols + role_cols + key_metrics].copy()

    for c in role_cols:
        out[c + " (pct)"] = percentile_rank(df_f[c]).reindex(out.index)

    pct_cols = [c for c in out.columns if c.endswith("(pct)")]
    out_disp = rename_for_display(out)

    pro_table(out_disp, pct_cols=pct_cols, height=740)

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

    # Top card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.4, 1.4, 0.9, 0.9, 0.9])
    c1.metric("Player", str(row.get(NAME_COL, "—")))
    c2.metric("Team", str(row.get(TEAM_COL, "—")))
    c3.metric("Age", safe_int_fmt(row.get(AGE_COL, np.nan)))
    c4.metric("Share", safe_fmt(row.get(SHARE_COL, np.nan), 2))
    c5.metric("Comp", str(row.get(COMP_COL, "—")))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Role scores")
        role_cols = [c for c in ROLE_DEFS.keys() if c in df_f.columns]
        role_row = pd.DataFrame([{
            "Role": DISPLAY_RENAMES.get(c, c).replace(" Score", ""),
            "Score": safe_float(row.get(c, np.nan)),
            "Percentile": safe_float(percentile_rank(df_f[c]).loc[p.index[0]]) if c in df_f.columns else np.nan,
        } for c in role_cols])

        pro_table(role_row, pct_cols=["Percentile"], height=280)

        st.subheader("Key metrics (percentiles)")
        key = []
        for m in METRICS:
            if m in df_f.columns and (m + " (pct)") in df_f.columns:
                key.append({
                    "Metric": m,
                    "Value": safe_float(row.get(m, np.nan)),
                    "Percentile": safe_float(row.get(m + " (pct)", np.nan)),
                })
        key_df = pd.DataFrame(key)
        if len(key_df):
            pro_table(key_df, pct_cols=["Percentile"], height=420)
        else:
            st.info("No metric columns found.")

    with right:
        st.subheader("Radar (role scores)")
        radar_cols = [c for c in ROLE_DEFS.keys() if c in df_f.columns]
        if radar_cols:
            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=[safe_float(row.get(c, np.nan)) if not np.isnan(safe_float(row.get(c, np.nan))) else 0 for c in radar_cols],
                    theta=[DISPLAY_RENAMES.get(c, c).replace(" Score", "") for c in radar_cols],
                    fill="toself",
                    name=player,
                )
            )
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=560)
            st.plotly_chart(fig, width="stretch")

# =====================================================
# ROLE LEADERBOARDS
# =====================================================
elif page == "Role Leaderboards":
    st.subheader("Role Leaderboards")

    available_roles = [r for r in ROLE_DEFS.keys() if r in df_f.columns]
    if not available_roles:
        st.warning("No role columns found.")
        st.stop()

    role = st.selectbox("Role", available_roles, index=available_roles.index("Balanced Score") if "Balanced Score" in available_roles else 0)
    n = st.slider("Rows", 10, 100, 40, 5)

    cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL, role] if c in df_f.columns]
    out = df_f.sort_values(role, ascending=False).head(n)[cols].copy()
    out[role + " (pct)"] = percentile_rank(df_f[role]).reindex(out.index)

    pct = [role + " (pct)"]
    out_disp = rename_for_display(out)
    pro_table(out_disp, pct_cols=pct, height=740)

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

# =====================================================
# DISTRIBUTIONS
# =====================================================
elif page == "Distributions":
    st.subheader("Distributions")

    numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available.")
        st.stop()

    default_metric = "Balanced Score" if "Balanced Score" in numeric_cols else numeric_cols[0]
    metric = st.selectbox("Metric", numeric_cols, index=numeric_cols.index(default_metric))

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

# =====================================================
# COMPARE PLAYERS
# =====================================================
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
            value_name="Score",
        )
        melt["Role"] = melt["Role"].map(lambda x: DISPLAY_RENAMES.get(x, x).replace(" Score", ""))
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
            r = [safe_float(sub.iloc[0].get(m, np.nan)) if not np.isnan(safe_float(sub.iloc[0].get(m, np.nan))) else 0 for m in radar_metrics]
            theta = [m.replace(" (pct)", "") for m in radar_metrics]
            fig2.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name=name))
        fig2.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=620)
        st.plotly_chart(fig2, width="stretch")

    st.markdown("### Comparison Table")
    show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] + role_cols if c in comp_df.columns]
    st.dataframe(comp_df[show].sort_values("Balanced Score", ascending=False), width="stretch", height=520)
