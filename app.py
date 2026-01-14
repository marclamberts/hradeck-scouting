import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Scout Lab", layout="wide", page_icon="⚽")

# Standard columns (expected in both sheets if possible)
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# ---- Position/Dataset configs (EDIT THESE) ----
POSITION_CONFIG = {
    "RWB": {
        "file": "RWB.xlsx",
        "title": "Right Wingback (RWB)",
        "metrics": [
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
        ],
        "role_defs": {
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
        },
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "High Cross", "Low Cross", "Breaking Opponent Defence"
        ],
        "radar_metrics": [
            "Progressive Receiving",
            "Breaking Opponent Defence",
            "Distance Covered Dribbles - Dribble",
            "High Cross",
            "Low Cross",
            "Defensive Ball Control",
            "Number of presses during opponent build-up",
            "IMPECT",
        ],
    },

    # NOTE: Replace the metric/role names below with YOUR actual CM.xlsx columns.
    "CM": {
        "file": "CM.xlsx",
        "title": "Central Midfielder (CM)",
        "metrics": [
            # Example CM metrics — change to match your CM.xlsx columns exactly
            "Progressive Passing",
            "Progressive Carrying",
            "Pass Completion %",
            "Final Third Entries",
            "Defensive Duels Won %",
            "Interceptions",
            "Pressures",
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
        ],
        "role_defs": {
            # Example roles — adjust weights + columns
            "Creator Score": {
                "Progressive Passing": 0.35,
                "Final Third Entries": 0.25,
                "Offensive IMPECT": 0.25,
                "Pass Completion %": 0.15,
            },
            "Progressor Score": {
                "Progressive Carrying": 0.35,
                "Progressive Passing": 0.25,
                "Final Third Entries": 0.20,
                "IMPECT": 0.20,
            },
            "Ball Winner Score": {
                "Interceptions": 0.30,
                "Pressures": 0.25,
                "Defensive Duels Won %": 0.25,
                "Defensive IMPECT": 0.20,
            },
            "Balanced Score": {
                "IMPECT": 0.30,
                "Offensive IMPECT": 0.20,
                "Defensive IMPECT": 0.20,
                "Progressive Passing": 0.15,
                "Progressive Carrying": 0.15,
            },
        },
        "key_metrics": ["IMPECT", "Progressive Passing", "Progressive Carrying", "Pressures", "Interceptions"],
        "radar_metrics": ["Progressive Passing", "Progressive Carrying", "Final Third Entries", "Interceptions", "Pressures", "IMPECT"],
    },
}

DISPLAY_RENAMES = {
    "Balanced Score": "Balanced",
    "Attacking Wingback Score": "Attacking WB",
    "Defensive Wingback Score": "Defensive WB",
    "Creator Score": "Creator",
    "Progressor Score": "Progressor",
    "Ball Winner Score": "Ball Winner",
    "Match Share": "Share",
}

# =====================================================
# UI THEME (ASA-ish)
# =====================================================
st.markdown(
    """
<style>
.stApp { background: #f6f7f9; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
h1, h2, h3 { letter-spacing: -0.02em; }

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
.nav-active div.stButton > button {
  background: #111827 !important;
  color: #ffffff !important;
  border-color: #111827 !important;
}
.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px 14px;
}
div[data-testid="stVerticalBlock"] > div { gap: 0.65rem; }
div[data-testid="stDataFrame"] thead tr th { font-weight: 900; }
.small-muted { color: #6b7280; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# SAFE PARSING + SCORING
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

def rename_for_display(df_: pd.DataFrame) -> pd.DataFrame:
    return df_.rename(columns=DISPLAY_RENAMES)

# =====================================================
# TABLE (NO STYLER)
# =====================================================
def pro_table(df: pd.DataFrame, pct_cols: list[str] | None = None, height: int = 600):
    pct_cols = pct_cols or []
    pct_cols = [c for c in pct_cols if c in df.columns]
    col_config = {}

    for c in pct_cols:
        col_config[c] = st.column_config.ProgressColumn(
            label=c, min_value=0, max_value=100, format="%.0f", help="Percentile (0–100)"
        )

    for c in df.columns:
        if c in pct_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            col_config[c] = st.column_config.NumberColumn(label=c, format="%.2f")

    if NAME_COL in df.columns:
        col_config[NAME_COL] = st.column_config.TextColumn(label=NAME_COL, width="large")
    if TEAM_COL in df.columns:
        col_config[TEAM_COL] = st.column_config.TextColumn(label=TEAM_COL, width="medium")
    if COMP_COL in df.columns:
        col_config[COMP_COL] = st.column_config.TextColumn(label=COMP_COL, width="medium")

    st.dataframe(df, width="stretch", height=height, column_config=col_config, hide_index=True)

# =====================================================
# LOAD + PREP DATA PER POSITION
# =====================================================
@st.cache_data(show_spinner=False)
def load_and_prepare(position_key: str) -> tuple[pd.DataFrame, dict]:
    cfg = POSITION_CONFIG[position_key]
    fp = Path(cfg["file"])
    if not fp.exists():
        raise FileNotFoundError(f"Missing {cfg['file']}. Put it next to app.py.")

    df = pd.read_excel(fp)
    df.columns = [str(c).strip() for c in df.columns]

    # Coerce key numeric fields
    coerce_numeric(df, cfg["metrics"] + [AGE_COL, SHARE_COL])

    # Clean text fields
    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    # Metric percentiles
    for m in cfg["metrics"]:
        if m in df.columns:
            df[m + " (pct)"] = percentile_rank(df[m])

    # Role scores
    for role, weights in cfg["role_defs"].items():
        z = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            if col in df.columns:
                z = z + zscore(df[col]) * float(w)
        df[role] = score_from_z(z)

    coerce_numeric(df, list(cfg["role_defs"].keys()))
    return df, cfg

# =====================================================
# SCOUTING FEATURES (SHORTLIST + SIMILARITY)
# =====================================================
def ensure_state():
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "shortlist" not in st.session_state:
        # shortlist: dict key=(position, player_name) -> {tags, notes}
        st.session_state.shortlist = {}

def shortlist_key(position_key: str, player_name: str) -> str:
    return f"{position_key}||{player_name}"

def add_to_shortlist(position_key: str, player_name: str):
    k = shortlist_key(position_key, player_name)
    if k not in st.session_state.shortlist:
        st.session_state.shortlist[k] = {"tags": "", "notes": ""}

def remove_from_shortlist(position_key: str, player_name: str):
    k = shortlist_key(position_key, player_name)
    if k in st.session_state.shortlist:
        del st.session_state.shortlist[k]

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    # X: (n, d)
    X = np.nan_to_num(X, nan=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T

def similar_players(df_f: pd.DataFrame, player_name: str, feature_cols: list[str], topk: int = 10) -> pd.DataFrame:
    if df_f.empty or player_name not in df_f[NAME_COL].values:
        return pd.DataFrame()

    cols = [c for c in feature_cols if c in df_f.columns and pd.api.types.is_numeric_dtype(df_f[c])]
    if not cols:
        return pd.DataFrame()

    X = df_f[cols].apply(zscore).to_numpy()
    sim = cosine_similarity_matrix(X)
    idx = df_f.index[df_f[NAME_COL] == player_name][0]
    scores = pd.Series(sim[df_f.index.get_loc(idx)], index=df_f.index)

    out = df_f.loc[scores.sort_values(ascending=False).index].copy()
    out["Similarity"] = scores.loc[out.index].values
    out = out[out[NAME_COL] != player_name].head(topk)

    show_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] if c in out.columns] + ["Similarity"]
    return out[show_cols]

# =====================================================
# NAV
# =====================================================
PAGES = [
    ("Dashboard", "🏠 Dashboard"),
    ("Search", "🔎 Search"),
    ("Player", "👤 Player Profile"),
    ("Leaderboards", "🏆 Leaderboards"),
    ("Distributions", "📈 Distributions"),
    ("Compare", "🆚 Compare"),
    ("Shortlist", "⭐ Shortlist"),
]

def nav_button(page_key: str, label: str):
    active = (st.session_state.page == page_key)
    if active:
        st.sidebar.markdown('<div class="nav-active">', unsafe_allow_html=True)
    clicked = st.sidebar.button(label, key=f"nav_{page_key}")
    if active:
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
    if clicked:
        st.session_state.page = page_key

# =====================================================
# APP
# =====================================================
ensure_state()

# Position selector (top of sidebar)
st.sidebar.markdown("### 🧩 Dataset")
position = st.sidebar.selectbox("Position", list(POSITION_CONFIG.keys()), index=0)
df, cfg = load_and_prepare(position)
role_cols_all = [r for r in cfg["role_defs"].keys() if r in df.columns]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Navigation")
for key, label in PAGES:
    nav_button(key, label)
page = st.session_state.page

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filters")

# Initialize filters per position
default_filters = {
    "q": "",
    "min_share": 0.20,
    "competitions": [],
    "teams": [],
    "age_range": (15, 45),
    "nats": [],
}
if position not in st.session_state.filters:
    st.session_state.filters[position] = default_filters.copy()

def reset_filters():
    st.session_state.filters[position] = default_filters.copy()

f = st.session_state.filters[position]

with st.sidebar.container():
    st.sidebar.markdown('<div class="card">', unsafe_allow_html=True)

    f["q"] = st.sidebar.text_input("Quick search", value=f["q"], placeholder="Name / Team / Comp / Nat…")

    f["min_share"] = st.sidebar.slider("Min Match Share", 0.0, 1.0, float(f["min_share"]), 0.05)

    if AGE_COL in df.columns:
        min_age, max_age = int(np.nanmin(df[AGE_COL].values) if len(df) else 15), int(np.nanmax(df[AGE_COL].values) if len(df) else 45)
        min_age = max(15, min_age)
        max_age = min(50, max_age)
        lo, hi = f["age_range"]
        lo = max(min_age, lo)
        hi = min(max_age, hi)
        f["age_range"] = st.sidebar.slider("Age range", min_age, max_age, (lo, hi), 1)

    if COMP_COL in df.columns:
        comps_all = sorted([c for c in df[COMP_COL].dropna().unique().tolist() if str(c).strip() != ""])
        f["competitions"] = st.sidebar.multiselect("Competitions", comps_all, default=f["competitions"])

    if TEAM_COL in df.columns:
        teams_all = sorted([t for t in df[TEAM_COL].dropna().unique().tolist() if str(t).strip() != ""])
        f["teams"] = st.sidebar.multiselect("Teams", teams_all, default=f["teams"])

    if NAT_COL in df.columns:
        nats_all = sorted([n for n in df[NAT_COL].dropna().unique().tolist() if str(n).strip() != ""])
        f["nats"] = st.sidebar.multiselect("Nationalities", nats_all, default=f["nats"])

    cA, cB = st.sidebar.columns(2)
    with cA:
        if st.button("Reset filters"):
            reset_filters()
            st.rerun()
    with cB:
        st.caption("Applies live")

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Apply filters
df_f = df.copy()

if SHARE_COL in df_f.columns:
    df_f = df_f[df_f[SHARE_COL].fillna(0) >= f["min_share"]]

if AGE_COL in df_f.columns:
    lo, hi = f["age_range"]
    df_f = df_f[(df_f[AGE_COL].fillna(lo) >= lo) & (df_f[AGE_COL].fillna(hi) <= hi)]

if f["competitions"] and COMP_COL in df_f.columns:
    df_f = df_f[df_f[COMP_COL].isin(f["competitions"])]

if f["teams"] and TEAM_COL in df_f.columns:
    df_f = df_f[df_f[TEAM_COL].isin(f["teams"])]

if f["nats"] and NAT_COL in df_f.columns:
    df_f = df_f[df_f[NAT_COL].isin(f["nats"])]

q = f["q"].strip().lower()
if q:
    mask = pd.Series(False, index=df_f.index)
    for col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if col in df_f.columns:
            mask = mask | df_f[col].astype(str).str.lower().str.contains(q, na=False)
    df_f = df_f[mask]

# Header
st.title("⚽ Scout Lab")
st.caption(f"TransferLab-style scouting workflow · {cfg['title']} · percentile bars · shortlist · similarity")

# =====================================================
# PAGES
# =====================================================
if page == "Dashboard":
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Players", len(df_f))
    k2.metric("Teams", df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0)
    k3.metric("Competitions", df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0)
    avg_share = df_f[SHARE_COL].mean() if SHARE_COL in df_f.columns and len(df_f) else np.nan
    k4.metric("Avg Share", safe_fmt(avg_share, 2))

    st.markdown("---")

    if "quick_role" not in st.session_state:
        st.session_state.quick_role = "Balanced Score" if "Balanced Score" in df_f.columns else (role_cols_all[0] if role_cols_all else None)

    st.markdown("#### Quick views")
    chips = st.columns(min(4, max(1, len(role_cols_all))))
    for i, r in enumerate(role_cols_all[:4]):
        if chips[i].button(f"⚡ {DISPLAY_RENAMES.get(r, r).replace(' Score','')}", key=f"chip_{position}_{r}"):
            st.session_state.quick_role = r

    role_sort = st.session_state.quick_role if st.session_state.quick_role in df_f.columns else ("Balanced Score" if "Balanced Score" in df_f.columns else None)
    if role_sort is None:
        st.warning("No role score columns found for this dataset.")
        st.stop()

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader(f"Top players (sorted by {DISPLAY_RENAMES.get(role_sort, role_sort)})")
        base_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in df_f.columns]
        role_cols = [c for c in role_cols_all if c in df_f.columns]
        show_cols = base_cols + role_cols

        top = df_f.sort_values(role_sort, ascending=False).head(30)[show_cols].copy()
        for c in role_cols:
            top[c + " (pct)"] = percentile_rank(df_f[c]).reindex(top.index)

        pct_cols = [c for c in top.columns if c.endswith("(pct)")]
        pro_table(rename_for_display(top), pct_cols=pct_cols, height=640)

    with right:
        st.subheader("Role landscape")
        # Pick two roles to plot (fallback)
        r1 = "Progressor Score" if "Progressor Score" in df_f.columns else role_cols_all[0]
        r2 = "Attacking Wingback Score" if "Attacking Wingback Score" in df_f.columns else (role_cols_all[1] if len(role_cols_all) > 1 else role_cols_all[0])

        fig = px.scatter(
            df_f,
            x=r1,
            y=r2,
            color=("Balanced Score" if "Balanced Score" in df_f.columns else r1),
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


elif page == "Search":
    st.subheader("Search")
    st.caption("Filtered scouting table + export. Add players to Shortlist from Profile page.")

    base_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in df_f.columns]
    role_cols = [c for c in role_cols_all if c in df_f.columns]
    key_metrics = [c for c in cfg["key_metrics"] if c in df_f.columns]

    sort_role = "Balanced Score" if "Balanced Score" in df_f.columns else role_cols[0]
    out = df_f.sort_values(sort_role, ascending=False)[base_cols + role_cols + key_metrics].copy()

    for c in role_cols:
        out[c + " (pct)"] = percentile_rank(df_f[c]).reindex(out.index)

    pct_cols = [c for c in out.columns if c.endswith("(pct)")]
    pro_table(rename_for_display(out), pct_cols=pct_cols, height=740)

    st.download_button(
        "Download filtered table (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"{position}_search_export.csv",
        mime="text/csv",
    )


elif page == "Player":
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

    # Header card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.4, 1.4, 0.9, 0.9, 0.9, 1.0])
    c1.metric("Player", str(row.get(NAME_COL, "—")))
    c2.metric("Team", str(row.get(TEAM_COL, "—")))
    c3.metric("Age", safe_int_fmt(row.get(AGE_COL, np.nan)))
    c4.metric("Share", safe_fmt(row.get(SHARE_COL, np.nan), 2))
    c5.metric("Comp", str(row.get(COMP_COL, "—")))
    in_sl = shortlist_key(position, player) in st.session_state.shortlist
    if c6.button("⭐ Shortlist" if not in_sl else "✅ Shortlisted", key=f"shortlist_btn_{position}_{player}"):
        if in_sl:
            remove_from_shortlist(position, player)
        else:
            add_to_shortlist(position, player)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Notes/tags if shortlisted
    if shortlist_key(position, player) in st.session_state.shortlist:
        st.markdown("#### Notes & tags")
        sl = st.session_state.shortlist[shortlist_key(position, player)]
        a, b = st.columns([1, 2])
        sl["tags"] = a.text_input("Tags (comma separated)", value=sl.get("tags", ""))
        sl["notes"] = b.text_area("Notes", value=sl.get("notes", ""), height=90)

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Role scores")
        role_cols = [c for c in role_cols_all if c in df_f.columns]
        role_row = pd.DataFrame([{
            "Role": DISPLAY_RENAMES.get(c, c).replace(" Score", ""),
            "Score": safe_float(row.get(c, np.nan)),
            "Percentile": safe_float(percentile_rank(df_f[c]).loc[p.index[0]]) if c in df_f.columns else np.nan,
        } for c in role_cols])

        pro_table(role_row, pct_cols=["Percentile"], height=280)

        st.subheader("Key metrics (percentiles)")
        key = []
        for m in cfg["metrics"]:
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
        radar_cols = [c for c in role_cols_all if c in df_f.columns]
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
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=520)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Similar players")
        st.caption("Cosine similarity on selected numeric features (z-scored).")
        sim_features = st.multiselect(
            "Similarity features",
            options=[c for c in cfg["metrics"] if c in df_f.columns],
            default=[c for c in cfg["radar_metrics"] if c in df_f.columns][:6],
            key=f"sim_feat_{position}",
        )
        topk = st.slider("Top K", 5, 25, 10, 1, key=f"sim_topk_{position}")
        sim_df = similar_players(df_f, player, sim_features, topk=topk)
        if len(sim_df):
            pro_table(sim_df, pct_cols=[], height=360)
        else:
            st.info("Not enough data/features to compute similarity.")


elif page == "Leaderboards":
    st.subheader("Role Leaderboards")

    available_roles = [r for r in role_cols_all if r in df_f.columns]
    if not available_roles:
        st.warning("No role columns found.")
        st.stop()

    default_role = "Balanced Score" if "Balanced Score" in available_roles else available_roles[0]
    role = st.selectbox("Role", available_roles, index=available_roles.index(default_role))
    n = st.slider("Rows", 10, 100, 40, 5)

    cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL, role] if c in df_f.columns]
    out = df_f.sort_values(role, ascending=False).head(n)[cols].copy()
    out[role + " (pct)"] = percentile_rank(df_f[role]).reindex(out.index)

    pro_table(rename_for_display(out), pct_cols=[role + " (pct)"], height=740)

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


elif page == "Compare":
    st.subheader("Compare Players")

    if NAME_COL not in df_f.columns or df_f.empty:
        st.warning("No players available with current filters.")
        st.stop()

    players = sorted(df_f[NAME_COL].dropna().unique().tolist())
    default = players[:2] if len(players) >= 2 else players
    picks = st.multiselect("Select 2–6 players", players, default=default)

    if len(picks) < 2:
        st.info("Pick at least 2 players to compare.")
        st.stop()

    comp_df = df_f[df_f[NAME_COL].isin(picks)].copy()

    st.markdown("### Role Scores Comparison")
    role_cols = [c for c in role_cols_all if c in comp_df.columns]
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
    radar_metrics = [m + " (pct)" for m in cfg["radar_metrics"] if (m + " (pct)") in comp_df.columns]
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
    sort_role = "Balanced Score" if "Balanced Score" in comp_df.columns else role_cols[0]
    st.dataframe(comp_df[show].sort_values(sort_role, ascending=False), width="stretch", height=520)


elif page == "Shortlist":
    st.subheader("⭐ Shortlist")
    st.caption("Your saved targets across datasets (session-only). Export at the bottom.")

    items = []
    for k, meta in st.session_state.shortlist.items():
        pos, name = k.split("||", 1)
        items.append({"Position": pos, "Name": name, "Tags": meta.get("tags", ""), "Notes": meta.get("notes", "")})

    if not items:
        st.info("Shortlist is empty. Add players from Player Profile.")
        st.stop()

    sl_df = pd.DataFrame(items)
    st.dataframe(sl_df, width="stretch", height=520)

    st.download_button(
        "Download shortlist (CSV)",
        data=sl_df.to_csv(index=False).encode("utf-8"),
        file_name="shortlist.csv",
        mime="text/csv",
    )
