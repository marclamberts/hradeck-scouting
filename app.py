import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="RWB Scout (ASA-style)", layout="wide", page_icon="⚽")

DATA_FILE = "RWB.xlsx"

# Columns expected from your file
ID_COL = "Player-ID"
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# Metric columns from your sheet
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

BETTER_THAN_COLS = [
    "IMPECT - BetterThan",
    "Offensive IMPECT - BetterThan",
    "Defensive IMPECT - BetterThan",
]

# =========================
# HELPERS
# =========================
def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def percentile_rank(series: pd.Series) -> pd.Series:
    # 0..1 percentile rank (ties averaged), robust to NaN
    s = series.copy()
    mask = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[mask] = s.loc[mask].rank(pct=True, method="average")
    return out

def scale_0_100(x: pd.Series) -> pd.Series:
    # expects x in 0..1
    return (x * 100).clip(0, 100)

def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = np.nanmean(s)
    sd = np.nanstd(s)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

def score_from_z(z: pd.Series) -> pd.Series:
    # map z roughly to 0..100 with a smooth squashing
    # 50 = average, 0/100 = extreme tails
    return (50 + 15 * z).clip(0, 100)

def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def style_table(df: pd.DataFrame, value_cols: list[str], pct_cols: list[str] = None):
    pct_cols = pct_cols or []
    fmt = {}
    for c in value_cols:
        fmt[c] = "{:,.2f}"
    for c in pct_cols:
        fmt[c] = "{:.0f}"

    styler = (
        df.style
        .format(fmt, na_rep="—")
        .set_properties(**{"font-size": "12px"})
    )

    # Heatmap on percentiles first (looks like ASA tables)
    if pct_cols:
        styler = styler.background_gradient(subset=pct_cols, axis=0)

    return styler

def metric_bar(percentile_value: float, label: str, value_text: str = ""):
    # ASA-like single stat bar (0..100)
    if pd.isna(percentile_value):
        st.write(f"**{label}**: —")
        return

    p = float(percentile_value)
    st.markdown(
        f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;">
            <div style="font-weight:600;">{label}</div>
            <div style="opacity:0.8;">{value_text} • {p:.0f}p</div>
          </div>
          <div style="background:#e5e7eb;border-radius:10px;height:10px;overflow:hidden;">
            <div style="width:{p:.0f}%;height:10px;background:#111827;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"Missing {file_path}. Put it next to app.py.")

    df = pd.read_excel(fp)

    # standardize
    df.columns = [c.strip() for c in df.columns]

    # numeric conversion
    df = to_numeric_safe(df, METRICS + BETTER_THAN_COLS + [AGE_COL, SHARE_COL])

    # basic cleanup
    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

df_raw = load_data(DATA_FILE)

# =========================
# BUILD PERCENTILES + ROLE SCORES
# =========================
df = df_raw.copy()

# Percentiles for metrics
for c in safe_cols(df, METRICS):
    df[c + " (pct)"] = scale_0_100(percentile_rank(df[c]))

# Role Scores (0..100) — designed for RWB
# You can tweak weights in one place.
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

def compute_role_score(df_: pd.DataFrame, role_def: dict) -> pd.Series:
    parts = []
    weights = []
    for col, w in role_def.items():
        if col in df_.columns:
            parts.append(zscore(df_[col]))
            weights.append(w)
    if not parts:
        return pd.Series(np.nan, index=df_.index)

    Z = np.vstack([p.values for p in parts])  # k x n
    w = np.array(weights).reshape(-1, 1)      # k x 1
    z = np.nanmean(Z * w, axis=0) / (np.sum(weights) if np.sum(weights) else 1)
    return pd.Series(z, index=df_.index)

for role_name, role_def in ROLE_DEFS.items():
    role_z = compute_role_score(df, role_def)
    df[role_name] = score_from_z(role_z)

# Quick “position tag” column if needed
if "Right Wing-Back" in df.columns:
    # It looks like this is a flag/label in your file; keep it but don't depend on it.
    pass

# =========================
# SIDEBAR NAV
# =========================
st.sidebar.title("⚽ RWB Scout")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Big Search", "Player Profile", "Role Leaderboards", "Distributions", "Compare Players"],
)

st.sidebar.markdown("---")
min_share = st.sidebar.slider("Min Match Share", 0.0, 1.0, 0.20, 0.05)
comps = ["All"] + sorted([c for c in df[COMP_COL].dropna().unique() if str(c) != "nan"])
teams = ["All"] + sorted([t for t in df[TEAM_COL].dropna().unique() if str(t) != "nan"])
comp_pick = st.sidebar.selectbox("Competition", comps)
team_pick = st.sidebar.selectbox("Team", teams)

df_f = df.copy()
if SHARE_COL in df_f.columns:
    df_f = df_f[df_f[SHARE_COL].fillna(0) >= min_share]
if comp_pick != "All":
    df_f = df_f[df_f[COMP_COL] == comp_pick]
if team_pick != "All":
    df_f = df_f[df_f[TEAM_COL] == team_pick]

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div style="display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:8px;">
      <div>
        <div style="font-size:28px;font-weight:800;line-height:1;">RWB Scout</div>
        <div style="opacity:0.75;">ASA-style tables • role scores • percentiles • search</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# PAGES
# =========================
if page == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", len(df_f))
    c2.metric("Competitions", df_f[COMP_COL].nunique())
    c3.metric("Teams", df_f[TEAM_COL].nunique())
    if SHARE_COL in df_f.columns:
        c4.metric("Avg Match Share", f"{df_f[SHARE_COL].mean():.2f}")

    st.markdown("---")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Top RWB by Balanced Score")
        show_cols = [
            NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL,
            "Balanced Score", "Attacking Wingback Score", "Progressor Score", "Defensive Wingback Score",
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT"
        ]
        show_cols = safe_cols(df_f, show_cols)
        top = df_f.sort_values("Balanced Score", ascending=False).head(25)[show_cols]

        # add percentiles for the 4 role scores
        for sc in ["Balanced Score", "Attacking Wingback Score", "Progressor Score", "Defensive Wingback Score"]:
            if sc in top.columns:
                top[sc + " (pct)"] = scale_0_100(percentile_rank(df_f[sc])).reindex(top.index)

        pct_cols = [c for c in top.columns if c.endswith("(pct)")]
        val_cols = [c for c in top.columns if c not in pct_cols]

        st.dataframe(style_table(top, val_cols, pct_cols), use_container_width=True, height=720)

    with right:
        st.subheader("Role Landscape (color = Balanced Score)")
        if all(c in df_f.columns for c in ["Progressor Score", "Attacking Wingback Score", "Balanced Score"]):
            fig = px.scatter(
                df_f,
                x="Progressor Score",
                y="Attacking Wingback Score",
                color="Balanced Score",
                hover_data=[NAME_COL, TEAM_COL, COMP_COL, AGE_COL],
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Role Leaders (Top 10)")
        role_choice = st.selectbox("Role", list(ROLE_DEFS.keys()), index=3)
        leaders = df_f.sort_values(role_choice, ascending=False).head(10)
        fig2 = px.bar(
            leaders,
            x=role_choice,
            y=NAME_COL,
            orientation="h",
            color=role_choice,
            hover_data=[TEAM_COL, COMP_COL, AGE_COL, SHARE_COL],
        )
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Big Search":
    st.subheader("Big Search (fast filters + text)")
    q = st.text_input("Search player / team / competition / nationality", placeholder="Type e.g. 'Ajax', 'Czech', 'Wing', 'U21'...")

    df_s = df_f.copy()
    if q.strip():
        qq = q.strip().lower()
        mask = False
        for col in safe_cols(df_s, [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]):
            mask = mask | df_s[col].astype(str).str.lower().str.contains(qq, na=False)
        df_s = df_s[mask]

    st.caption("Tip: Sort columns by clicking headers. Use sidebar to filter competition/team/match share.")
    base_cols = [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL]
    score_cols = ["Balanced Score", "Attacking Wingback Score", "Progressor Score", "Defensive Wingback Score"]
    metric_cols = safe_cols(df_s, ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "High Cross", "Low Cross", "Breaking Opponent Defence"])
    cols = safe_cols(df_s, base_cols + score_cols + metric_cols)

    df_out = df_s.sort_values("Balanced Score", ascending=False)[cols]

    # Add percentile columns for score columns for cleaner “ASA feel”
    for sc in score_cols:
        if sc in df_out.columns:
            df_out[sc + " (pct)"] = scale_0_100(percentile_rank(df_s[sc])).reindex(df_out.index)

    pct_cols = [c for c in df_out.columns if c.endswith("(pct)")]
    val_cols = [c for c in df_out.columns if c not in pct_cols]

    st.dataframe(style_table(df_out, val_cols, pct_cols), use_container_width=True, height=760)

    st.download_button(
        "Download filtered table (CSV)",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="rwb_search_export.csv",
        mime="text/csv",
    )

elif page == "Player Profile":
    st.subheader("Player Profile (percentiles + radar)")

    # player picker (searchable)
    player_list = df_f[NAME_COL].dropna().unique().tolist()
    player = st.selectbox("Select player", sorted(player_list))

    p = df_f[df_f[NAME_COL] == player].head(1)
    if p.empty:
        st.warning("No player found with current filters.")
        st.stop()

    row = p.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Team", str(row.get(TEAM_COL, "—")))
    c2.metric("Competition", str(row.get(COMP_COL, "—")))
    c3.metric("Age", "—" if pd.isna(row.get(AGE_COL, np.nan)) else int(row.get(AGE_COL)))
    c4.metric("Match Share", "—" if pd.isna(row.get(SHARE_COL, np.nan)) else f"{row.get(SHARE_COL):.2f}")

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Role Scores")
        for sc in ["Balanced Score", "Attacking Wingback Score", "Progressor Score", "Defensive Wingback Score"]:
            if sc in df_f.columns:
                pct = scale_0_100(percentile_rank(df_f[sc])).loc[p.index[0]]
                metric_bar(pct, sc.replace(" Score", ""), value_text=f"{row.get(sc, np.nan):.1f}")

        st.markdown("### Key Metrics (percentiles)")
        key_metrics = safe_cols(df_f, [
            "Progressive Receiving",
            "Breaking Opponent Defence",
            "Distance Covered Dribbles - Dribble",
            "High Cross",
            "Low Cross",
            "Defensive Ball Control",
            "Number of presses during opponent build-up",
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
        ])
        for m in key_metrics:
            pct = row.get(m + " (pct)", np.nan)
            val = row.get(m, np.nan)
            metric_bar(pct, m, value_text=("—" if pd.isna(val) else f"{val:.2f}"))

    with right:
        st.markdown("### Radar (Role Scores)")
        radar_axes = ["Attacking Wingback Score", "Progressor Score", "Defensive Wingback Score", "Balanced Score"]
        radar_axes = [a for a in radar_axes if a in df_f.columns]

        if radar_axes:
            values = [float(row.get(a, np.nan)) for a in radar_axes]
            avg_vals = [float(df_f[a].mean()) for a in radar_axes]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values, theta=[a.replace(" Score","") for a in radar_axes], fill="toself", name=player))
            fig.add_trace(go.Scatterpolar(r=avg_vals, theta=[a.replace(" Score","") for a in radar_axes], fill="toself", name="Avg (filtered)"))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Quick Table Row")
        display_cols = safe_cols(df_f, [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] + radar_axes + METRICS)
        st.dataframe(p[display_cols], use_container_width=True, height=260)

elif page == "Role Leaderboards":
    st.subheader("Role Leaderboards (ASA-style tables + color)")
    role = st.selectbox("Choose leaderboard", list(ROLE_DEFS.keys()), index=3)

    n = st.slider("Rows", 10, 100, 25, 5)

    cols = safe_cols(df_f, [
        NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL,
        role, "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
        "High Cross", "Low Cross", "Breaking Opponent Defence", "Defensive Ball Control"
    ])

    out = df_f.sort_values(role, ascending=False).head(n)[cols].copy()
    out[role + " (pct)"] = scale_0_100(percentile_rank(df_f[role])).reindex(out.index)

    pct_cols = [role + " (pct)"]
    val_cols = [c for c in out.columns if c not in pct_cols]

    st.dataframe(style_table(out, val_cols, pct_cols), use_container_width=True, height=820)

    st.markdown("### Top 20 Bar Chart (colored by value)")
    top20 = df_f.sort_values(role, ascending=False).head(20)
    fig = px.bar(
        top20,
        x=role,
        y=NAME_COL,
        orientation="h",
        color=role,
        hover_data=[TEAM_COL, COMP_COL, AGE_COL, SHARE_COL],
    )
    fig.update_layout(height=650, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Distributions":
    st.subheader("Distributions (hist / box / violin)")

    numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    metric = st.selectbox("Metric", numeric_cols, index=numeric_cols.index("Balanced Score") if "Balanced Score" in numeric_cols else 0)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.histogram(df_f, x=metric, nbins=30, marginal="rug")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.box(df_f, y=metric, points="all")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Split by Competition / Team")
    split = st.radio("Split by", [COMP_COL, TEAM_COL], horizontal=True)
    topk = st.slider("Show top groups", 5, 30, 12, 1)

    g = df_f.groupby(split)[metric].mean().sort_values(ascending=False).head(topk).reset_index()
    fig3 = px.bar(g, x=metric, y=split, orientation="h", color=metric)
    fig3.update_layout(height=600, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Compare Players":
    st.subheader("Compare Players")

    players = sorted(df_f[NAME_COL].dropna().unique().tolist())
    picks = st.multiselect("Select 2–6 players", players, default=players[:2])

    if len(picks) < 2:
        st.info("Pick at least 2 players to compare.")
        st.stop()

    comp_df = df_f[df_f[NAME_COL].isin(picks)].copy()

    st.markdown("### Role Scores Comparison")
    role_cols = [c for c in ROLE_DEFS.keys() if c in comp_df.columns]
    fig = px.bar(
        comp_df.melt(id_vars=[NAME_COL, TEAM_COL, COMP_COL], value_vars=role_cols, var_name="Score", value_name="Value"),
        x="Value", y=NAME_COL, color="Score", barmode="group",
        hover_data=[TEAM_COL, COMP_COL],
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Metric Radar (percentiles)")
    # Choose a small set of key RWB metrics for a clean radar
    radar_metrics = safe_cols(comp_df, [
        "Progressive Receiving (pct)",
        "Breaking Opponent Defence (pct)",
        "Distance Covered Dribbles - Dribble (pct)",
        "High Cross (pct)",
        "Low Cross (pct)",
        "Defensive Ball Control (pct)",
        "Number of presses during opponent build-up (pct)",
        "IMPECT (pct)",
    ])

    if radar_metrics:
        fig2 = go.Figure()
        for name in picks:
            sub = comp_df[comp_df[NAME_COL] == name].head(1)
            if sub.empty:
                continue
            r = [float(sub.iloc[0].get(m, np.nan)) for m in radar_metrics]
            theta = [m.replace(" (pct)", "") for m in radar_metrics]
            fig2.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name=name))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=620,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Comparison Table")
    show = safe_cols(comp_df, [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] + role_cols + METRICS)
    st.dataframe(comp_df[show].sort_values("Balanced Score", ascending=False), use_container_width=True, height=520)
