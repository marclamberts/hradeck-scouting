import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import zscore
import datetime as dt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Scout Lab Pro", layout="wide", page_icon="⚽")

# =====================================================
# COLORS
# =====================================================
COLORS = {
    "yellow": "#F4C430",
    "black": "#0B0B0B",
    "white": "#F7F7F7",
    "grey": "#9AA0A6",
    "emerald": "#2ECC71",
    "background": "#0e1117",
}

# =====================================================
# CANONICAL COLUMN NAMES
# =====================================================
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# =====================================================
# POSITION/DATASET CONFIG
# =====================================================
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
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "High Cross",
            "Low Cross",
            "Breaking Opponent Defence",
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
        "default_sort": "Balanced Score",
    },
    "CM": {
        "file": "CM.xlsx",
        "title": "Central Midfielder (CM)",
        "metrics": [
            "Progressive Passing",
            "Pass Completion %",
            "Aerial Duels Won %",
            "Defensive Duels Won %",
            "Interceptions",
            "Blocks",
            "Clearances",
            "Pressures",
            "IMPECT",
            "Defensive IMPECT",
            "Offensive IMPECT",
        ],
        "role_defs": {
            "Stopper Score": {
                "Defensive Duels Won %": 0.30,
                "Blocks": 0.20,
                "Clearances": 0.20,
                "Defensive IMPECT": 0.30,
            },
            "Ball-Playing Score": {
                "Progressive Passing": 0.35,
                "Pass Completion %": 0.25,
                "Offensive IMPECT": 0.20,
                "IMPECT": 0.20,
            },
            "Aerial Score": {
                "Aerial Duels Won %": 0.45,
                "Clearances": 0.20,
                "Blocks": 0.15,
                "Defensive IMPECT": 0.20,
            },
            "Balanced Score": {
                "IMPECT": 0.30,
                "Defensive IMPECT": 0.25,
                "Progressive Passing": 0.20,
                "Defensive Duels Won %": 0.15,
                "Aerial Duels Won %": 0.10,
            },
        },
        "key_metrics": [
            "IMPECT",
            "Defensive IMPECT",
            "Progressive Passing",
            "Interceptions",
            "Aerial Duels Won %",
        ],
        "radar_metrics": [
            "Progressive Passing",
            "Pass Completion %",
            "Interceptions",
            "Blocks",
            "Aerial Duels Won %",
            "IMPECT",
        ],
        "default_sort": "Balanced Score",
    },
}

DISPLAY_RENAMES = {
    "Balanced Score": "Balanced",
    "Attacking Wingback Score": "Attacking WB",
    "Defensive Wingback Score": "Defensive WB",
    "Progressor Score": "Progressor",
    "Stopper Score": "Stopper",
    "Ball-Playing Score": "Ball-Playing",
    "Aerial Score": "Aerial",
    "Match Share": "Share",
}

# =====================================================
# UI THEME
# =====================================================
st.markdown(
    f"""
<style>
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

.stApp {{
  background: {COLORS["background"]};
  color: {COLORS["white"]};
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
}}

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(11,11,11,0.92) 0%, rgba(14,17,23,0.96) 100%);
  border-right: 1px solid rgba(244,196,48,0.12);
}}
section[data-testid="stSidebar"] * {{
  color: {COLORS["white"]};
}}

h1, h2, h3 {{
  letter-spacing: -0.03em;
  margin-top: 0.5rem;
}}
.small {{
  color: {COLORS["grey"]};
  font-size: 0.92rem;
}}
.muted {{
  color: {COLORS["grey"]};
}}
.kicker {{
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.75rem;
  color: {COLORS["grey"]};
  font-weight: 900;
  margin-bottom: 0.5rem;
}}

.card {{
  background: rgba(11,11,11,0.65);
  border: 1px solid rgba(244,196,48,0.10);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.30);
  margin-bottom: 1rem;
}}
.card-strong {{
  background: rgba(11,11,11,0.78);
  border: 1px solid rgba(244,196,48,0.18);
}}

.headerbar {{
  background: linear-gradient(135deg, rgba(11,11,11,0.85) 0%, rgba(14,17,23,0.85) 100%);
  border: 1px solid rgba(244,196,48,0.16);
  border-radius: 18px;
  padding: 16px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2rem;
  flex-wrap: wrap;
  gap: 1rem;
}}
.header-left {{
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}}
.brand {{
  font-size: 1.35rem;
  font-weight: 950;
}}
.pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid rgba(247,247,247,0.10);
  background: rgba(11,11,11,0.55);
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 0.9rem;
  color: {COLORS["white"]};
}}
.pill-accent {{
  border-color: rgba(244,196,48,0.40);
  background: rgba(244,196,48,0.10);
}}
.pill-solid {{
  border-color: rgba(244,196,48,0.65);
  background: {COLORS["yellow"]};
  color: {COLORS["black"]};
}}

.chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid rgba(247,247,247,0.10);
  background: rgba(11,11,11,0.55);
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.9rem;
  color: {COLORS["white"]};
  margin-right: 6px;
  margin-bottom: 6px;
}}
.chip strong {{ 
  font-weight: 950; 
  color: {COLORS["yellow"]}; 
}}

.player-row {{
  border: 1px solid rgba(247,247,247,0.08);
  border-radius: 14px;
  background: rgba(11,11,11,0.55);
  padding: 14px;
  margin-bottom: 0.75rem;
  transition: all 0.2s ease;
}}
.player-row:hover {{
  border-color: rgba(244,196,48,0.28);
  background: rgba(11,11,11,0.70);
}}
.player-name {{
  font-weight: 950;
  font-size: 1.05rem;
  color: {COLORS["white"]};
  margin-bottom: 4px;
}}
.player-meta {{
  color: {COLORS["grey"]};
  font-size: 0.88rem;
}}

div.stButton > button {{
  border-radius: 12px;
  border: 1px solid rgba(247,247,247,0.12);
  background: rgba(11,11,11,0.55);
  color: {COLORS["white"]};
  font-weight: 900;
  transition: all 0.2s ease;
}}
div.stButton > button:hover {{
  border-color: rgba(244,196,48,0.45);
  background: rgba(244,196,48,0.10);
}}
button[kind="primary"] {{
  border: 1px solid rgba(244,196,48,0.65) !important;
  background: {COLORS["yellow"]} !important;
  color: {COLORS["black"]} !important;
  font-weight: 950 !important;
}}

div[data-baseweb="input"] > div {{
  border-radius: 12px !important;
  background: rgba(11,11,11,0.55) !important;
  border: 1px solid rgba(247,247,247,0.10) !important;
}}
div[data-baseweb="select"] > div {{
  border-radius: 12px !important;
  background: rgba(11,11,11,0.55) !important;
  border: 1px solid rgba(247,247,247,0.10) !important;
}}
div[data-testid="stSlider"] > div {{
  border-radius: 14px;
  background: rgba(11,11,11,0.40);
  border: 1px solid rgba(247,247,247,0.08);
  padding: 8px 10px;
}}

button[data-baseweb="tab"] {{
  font-weight: 950;
}}
div[data-baseweb="tab-list"] {{
  gap: 6px;
}}

div[data-testid="stVerticalBlock"] > div {{ 
  gap: 0.75rem; 
}}

div[data-testid="stDataFrame"] thead tr th {{
  font-weight: 950;
  background: rgba(11,11,11,0.85) !important;
  border-bottom: 2px solid rgba(244,196,48,0.30) !important;
}}

div[data-testid="stDataFrame"] {{
  border: 1px solid rgba(247,247,247,0.08);
  border-radius: 12px;
}}

.metric-card {{
  text-align: center;
  padding: 1rem;
  border-radius: 12px;
  background: rgba(11,11,11,0.60);
  border: 1px solid rgba(247,247,247,0.08);
}}
.metric-value {{
  font-size: 1.8rem;
  font-weight: 950;
  color: {COLORS["yellow"]};
  margin: 0.5rem 0;
}}
.metric-label {{
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: {COLORS["grey"]};
  font-weight: 700;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# SAFE PARSING + SCORING
# =====================================================
def safe_float(x):
    """Convert value to float, handling various formats"""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}:
        return np.nan
    s = s.replace("%", "")
    # Handle European format (comma as decimal)
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # Handle thousands separator
    if s.count(",") >= 1 and s.count(".") == 1:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def safe_fmt(x, decimals=2):
    """Format number safely"""
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"


def safe_int_fmt(x):
    """Format as integer safely"""
    v = safe_float(x)
    if np.isnan(v):
        return "—"
    return f"{int(round(v))}"


def coerce_numeric(df: pd.DataFrame, cols: list):
    """Convert columns to numeric"""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(safe_float)


def percentile_rank(s: pd.Series) -> pd.Series:
    """Calculate percentile rank (0-100)"""
    s = s.map(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    if mask.sum() > 0:
        out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out


def score_from_z(z: pd.Series) -> pd.Series:
    """Convert z-scores to 0-100 scale"""
    z = z.map(safe_float).fillna(0.0)
    return (50 + 15 * z).clip(0, 100)


def rename_for_display(df_: pd.DataFrame) -> pd.DataFrame:
    """Apply display renames"""
    return df_.rename(columns=DISPLAY_RENAMES)


def player_meta(row: pd.Series) -> str:
    """Format player metadata string"""
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    nat = str(row.get(NAT_COL, "—"))
    age = safe_int_fmt(row.get(AGE_COL, np.nan))
    share = safe_fmt(row.get(SHARE_COL, np.nan), 1)
    return f"{team} · {comp} · {nat} · Age {age} · {share}% share"


def strengths_weaknesses(cfg: dict, row: pd.Series, topn: int = 6):
    """Get top and bottom metrics for player"""
    pairs = []
    for m in cfg["metrics"]:
        pct = safe_float(row.get(m + " (pct)", np.nan))
        if not np.isnan(pct):
            pairs.append((m, pct))
    
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:topn]
    bottom = list(reversed(pairs[-topn:])) if len(pairs) >= topn else list(reversed(pairs))
    return top, bottom


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between rows"""
    X = np.nan_to_num(X, nan=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def similar_players(df_f: pd.DataFrame, player_name: str, feature_cols: list, topk: int = 10) -> pd.DataFrame:
    """Find similar players using cosine similarity"""
    if df_f.empty or NAME_COL not in df_f.columns:
        return pd.DataFrame()
    if player_name not in df_f[NAME_COL].values:
        return pd.DataFrame()

    cols = [c for c in feature_cols if c in df_f.columns and pd.api.types.is_numeric_dtype(df_f[c])]
    if not cols:
        return pd.DataFrame()

    X = np.column_stack([zscore(df_f[c], nan_policy='omit').fillna(0) for c in cols])
    sim = cosine_similarity_matrix(X)

    idx = df_f.index[df_f[NAME_COL] == player_name][0]
    base_i = df_f.index.get_loc(idx)
    scores = pd.Series(sim[base_i], index=df_f.index)

    out = df_f.loc[scores.sort_values(ascending=False).index].copy()
    out["Similarity"] = scores.loc[out.index].values * 100
    out = out[out[NAME_COL] != player_name].head(topk)

    show_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in out.columns] + ["Similarity"]
    return out[show_cols]


# =====================================================
# PRO TABLE
# =====================================================
def pro_table(df: pd.DataFrame, pct_cols: list = None, height: int = 600):
    """Display dataframe with progress bars for percentile columns"""
    pct_cols = pct_cols or []
    pct_cols = [c for c in pct_cols if c in df.columns]
    col_config = {}

    for c in pct_cols:
        col_config[c] = st.column_config.ProgressColumn(
            label=c,
            min_value=0,
            max_value=100,
            format="%.0f",
            help="Percentile (0–100)",
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

    st.dataframe(df, use_container_width=True, height=height, column_config=col_config, hide_index=True)


# =====================================================
# LOAD + PREP DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_and_prepare(position_key: str) -> tuple:
    """Load and prepare position data"""
    cfg = POSITION_CONFIG[position_key]
    fp = Path(cfg["file"])
    
    if not fp.exists():
        st.error(f"❌ File not found: `{cfg['file']}`\n\nPlease place the Excel file in the same directory as this app.")
        st.stop()

    try:
        df = pd.read_excel(fp)
    except Exception as e:
        st.error(f"❌ Error reading {cfg['file']}: {e}")
        st.stop()

    df.columns = [str(c).strip() for c in df.columns]

    # Convert metrics to numeric
    coerce_numeric(df, cfg["metrics"] + [AGE_COL, SHARE_COL])

    # Clean text columns
    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    # Calculate percentiles for metrics
    for m in cfg["metrics"]:
        if m in df.columns:
            df[m + " (pct)"] = percentile_rank(df[m])

    # Calculate role scores
    for role, weights in cfg["role_defs"].items():
        z = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            if col in df.columns:
                col_z = zscore(df[col], nan_policy='omit')
                col_z = pd.Series(col_z, index=df.index).fillna(0)
                z = z + col_z * float(w)
        df[role] = score_from_z(z)

    coerce_numeric(df, list(cfg["role_defs"].keys()))
    return df, cfg


# =====================================================
# STATE + FILTERS
# =====================================================
def ensure_state():
    """Initialize session state"""
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "shortlist" not in st.session_state:
        st.session_state.shortlist = {}
    if "pinned" not in st.session_state:
        st.session_state.pinned = {}
    if "selected_player" not in st.session_state:
        st.session_state.selected_player = None
    if "compare_picks" not in st.session_state:
        st.session_state.compare_picks = {}


def shortlist_key(position_key: str, player_name: str) -> str:
    """Create unique key for shortlist"""
    return f"{position_key}||{player_name}"


def add_to_shortlist(position_key: str, player_name: str):
    """Add player to shortlist"""
    k = shortlist_key(position_key, player_name)
    if k not in st.session_state.shortlist:
        st.session_state.shortlist[k] = {"tags": "", "notes": "", "added": dt.datetime.now()}


def remove_from_shortlist(position_key: str, player_name: str):
    """Remove player from shortlist"""
    k = shortlist_key(position_key, player_name)
    if k in st.session_state.shortlist:
        del st.session_state.shortlist[k]


def default_filters_for(df: pd.DataFrame):
    """Create default filters based on data"""
    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        if len(vals):
            lo = int(max(15, np.floor(vals.min())))
            hi = int(min(50, np.ceil(vals.max())))
        else:
            lo, hi = 15, 45
    else:
        lo, hi = 15, 45

    return {
        "q": "",
        "min_share": 0.20,
        "competitions": [],
        "teams": [],
        "nats": [],
        "age_range": (lo, hi),
    }


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Apply filters to dataframe"""
    out = df.copy()

    if SHARE_COL in out.columns:
        out = out[out[SHARE_COL].fillna(0) >= float(f.get("min_share", 0.0))]

    if AGE_COL in out.columns and "age_range" in f:
        lo, hi = f["age_range"]
        out = out[(out[AGE_COL].fillna(lo) >= lo) & (out[AGE_COL].fillna(hi) <= hi)]

    if f.get("competitions") and COMP_COL in out.columns:
        out = out[out[COMP_COL].isin(f["competitions"])]

    if f.get("teams") and TEAM_COL in out.columns:
        out = out[out[TEAM_COL].isin(f["teams"])]

    if f.get("nats") and NAT_COL in out.columns:
        out = out[out[NAT_COL].isin(f["nats"])]

    q = str(f.get("q", "")).strip().lower()
    if q:
        mask = pd.Series(False, index=out.index)
        for col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            if col in out.columns:
                mask = mask | out[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
        out = out[mask]

    return out


# =====================================================
# MAIN APP
# =====================================================
ensure_state()

# Sidebar
st.sidebar.markdown("### ⚙️ Scout Control")
position = st.sidebar.selectbox("Dataset", list(POSITION_CONFIG.keys()), index=0)

# Load data
with st.spinner("Loading player database..."):
    df, cfg = load_and_prepare(position)

# Get role columns
role_cols_all = [r for r in cfg["role_defs"].keys() if r in df.columns]
default_sort = cfg.get("default_sort", "Balanced Score")
if default_sort not in df.columns and role_cols_all:
    default_sort = role_cols_all[0]

# Initialize filters
if position not in st.session_state.filters:
    st.session_state.filters[position] = default_filters_for(df)
f = st.session_state.filters[position]

# Filters in sidebar
with st.sidebar.expander("🔍 Filters", expanded=True):
    f["q"] = st.text_input("Search", value=f.get("q", ""), placeholder="Name / Team / Comp / Nat…")
    f["min_share"] = st.slider("Min Share", 0.0, 1.0, float(f.get("min_share", 0.20)), 0.05)

    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        if len(vals):
            min_age = int(max(15, np.floor(vals.min())))
            max_age = int(min(50, np.ceil(vals.max())))
        else:
            min_age, max_age = 15, 45
        lo, hi = f.get("age_range", (min_age, max_age))
        lo = max(min_age, lo)
        hi = min(max_age, hi)
        f["age_range"] = st.slider("Age", min_age, max_age, (lo, hi), 1)

    if COMP_COL in df.columns:
        comps_all = sorted([c for c in df[COMP_COL].dropna().unique().tolist() if str(c).strip() != ""])
        f["competitions"] = st.multiselect("Competitions", comps_all, default=f.get("competitions", []))

    if TEAM_COL in df.columns:
        teams_all = sorted([t for t in df[TEAM_COL].dropna().unique().tolist() if str(t).strip() != ""])
        f["teams"] = st.multiselect("Teams", teams_all, default=f.get("teams", []))

    if NAT_COL in df.columns:
        nats_all = sorted([n for n in df[NAT_COL].dropna().unique().tolist() if str(n).strip() != ""])
        f["nats"] = st.multiselect("Nationalities", nats_all, default=f.get("nats", []))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset", type="primary", key=f"reset_{position}"):
            st.session_state.filters[position] = default_filters_for(df)
            st.rerun()

# Apply filters
df_f = apply_filters(df, f)

# Initialize pinned player
if position not in st.session_state.pinned:
    if len(df_f) and NAME_COL in df_f.columns:
        st.session_state.pinned[position] = df_f.sort_values(default_sort, ascending=False).iloc[0][NAME_COL]
    else:
        st.session_state.pinned[position] = None

# Initialize compare picks
if position not in st.session_state.compare_picks:
    st.session_state.compare_picks[position] = []

# Header bar
shortlist_count = len(st.session_state.shortlist)
teams_n = df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0
comps_n = df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0

st.markdown(
    f"""
<div class="headerbar">
  <div class="header-left">
    <div class="brand">⚽ Scout Lab Pro</div>
    <span class="pill pill-accent">{cfg["title"]}</span>
    <span class="pill">Players <strong>{len(df_f)}</strong></span>
    <span class="pill">Teams <strong>{teams_n}</strong></span>
    <span class="pill">Comps <strong>{comps_n}</strong></span>
  </div>
  <div>
    <span class="pill pill-solid">⭐ Shortlist {shortlist_count}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Main tabs
tabs = st.tabs(["🔍 Search", "👤 Profile", "⚖️ Compare", "🏆 Leaderboards", "📊 Analytics", "⭐ Shortlist"])

# =====================================================
# TAB: SEARCH
# =====================================================
with tabs[0]:
    st.markdown('<div class="kicker">Scout</div>', unsafe_allow_html=True)
    st.markdown("## Player Search")

    if df_f.empty:
        st.info("🔍 No players match your filters. Try adjusting the criteria in the sidebar.")
        col1, col2, col3 = st.columns(3)
        if col1.button("Lower Min Share", key=f"lowshare_{position}", type="primary"):
            f["min_share"] = 0.10
            st.rerun()
        if col2.button("Clear Search", key=f"clearq_{position}"):
            f["q"] = ""
            st.rerun()
        if col3.button("Reset All Filters", key=f"resetall_{position}"):
            st.session_state.filters[position] = default_filters_for(df)
            st.rerun()
    else:
        # Sort controls
        col1, col2 = st.columns([2, 1])
        with col1:
            sort_options = [c for c in ([default_sort] + role_cols_all) if c in df_f.columns]
            if not sort_options:
                sort_options = [c for c in df_f.columns if pd.api.types.is_numeric_dtype(df_f[c])]
            sort_col = st.selectbox("📊 Sort by", options=sort_options, index=0, key=f"sort_{position}")
        
        with col2:
            view_count = st.selectbox("👁️ Show", [20, 40, 60, 100], index=1, key=f"view_{position}")

        st.markdown("---")

        # Two column layout: results + preview
        left_col, right_col = st.columns([1.3, 1])

        with left_col:
            st.markdown("### Results")
            results = df_f.sort_values(sort_col, ascending=False).head(view_count).copy()
            
            for idx, (_, row) in enumerate(results.iterrows()):
                name = str(row.get(NAME_COL, "—"))
                in_sl = shortlist_key(position, name) in st.session_state.shortlist

                st.markdown('<div class="player-row">', unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([3, 1, 1])
                
                with c1:
                    st.markdown(f'<div class="player-name">{name}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="player-meta">{player_meta(row)}</div>', unsafe_allow_html=True)
                
                with c2:
                    st.metric(
                        DISPLAY_RENAMES.get(sort_col, sort_col).replace(" Score", ""),
                        safe_fmt(row.get(sort_col, np.nan), 1)
                    )
                
                with c3:
                    if st.button("Open", key=f"open_{position}_{name}_{idx}", use_container_width=True):
                        st.session_state.pinned[position] = name
                        st.session_state.selected_player = name
                        st.rerun()
                    
                    star_label = "✓" if in_sl else "★"
                    if st.button(star_label, key=f"slq_{position}_{name}_{idx}", use_container_width=True):
                        if in_sl:
                            remove_from_shortlist(position, name)
                        else:
                            add_to_shortlist(position, name)
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)

        with right_col:
            st.markdown("### Pinned Player")
            pinned = st.session_state.pinned.get(position)

            if not pinned:
                st.info("Click 'Open' on a player to preview their details here.")
            else:
                p = df_f[df_f[NAME_COL] == pinned].head(1)
                if p.empty:
                    st.warning("Pinned player not in filtered results.")
                else:
                    row = p.iloc[0]
                    
                    st.markdown('<div class="card card-strong">', unsafe_allow_html=True)
                    st.markdown(f"### {pinned}")
                    st.caption(player_meta(row))
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Key metrics
                    st.markdown("#### Quick Stats")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Age", safe_int_fmt(row.get(AGE_COL, np.nan)))
                    m2.metric("Share", safe_fmt(row.get(SHARE_COL, np.nan), 1) + "%")
                    if "Balanced Score" in df_f.columns:
                        m3.metric("Balanced", safe_fmt(row.get("Balanced Score", np.nan), 1))

                    # Role scores (top 4)
                    st.markdown("#### Role Fit")
                    role_tiles = st.columns(min(4, len(role_cols_all)))
                    for i, rc in enumerate(role_cols_all[:4]):
                        role_tiles[i].metric(
                            DISPLAY_RENAMES.get(rc, rc).replace(" Score", ""),
                            safe_fmt(row.get(rc, np.nan), 1)
                        )

                    # Strengths/Weaknesses
                    st.markdown("#### Performance (Percentiles)")
                    top, bottom = strengths_weaknesses(cfg, row, topn=4)
                    
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown("**Strengths**")
                        for m, pct in top:
                            st.markdown(f"↑ {m[:20]} — **{pct:.0f}**")
                    
                    with sc2:
                        st.markdown("**Development**")
                        for m, pct in bottom:
                            st.markdown(f"↓ {m[:20]} — **{pct:.0f}**")

                    # Actions
                    st.markdown("#### Actions")
                    a1, a2 = st.columns(2)
                    
                    with a1:
                        in_sl = shortlist_key(position, pinned) in st.session_state.shortlist
                        sl_label = "✓ Shortlisted" if in_sl else "★ Shortlist"
                        if st.button(sl_label, key=f"sl_pin_{position}", use_container_width=True, type="primary"):
                            if in_sl:
                                remove_from_shortlist(position, pinned)
                            else:
                                add_to_shortlist(position, pinned)
                            st.rerun()
                    
                    with a2:
                        picks = st.session_state.compare_picks[position]
                        cmp_label = "✓ In Compare" if pinned in picks else "+ Compare"
                        if st.button(cmp_label, key=f"addcmp_{position}", use_container_width=True):
                            if pinned not in picks:
                                picks.append(pinned)
                                st.session_state.compare_picks[position] = picks[:6]
                                st.rerun()

# =====================================================
# TAB: PROFILE
# =====================================================
with tabs[1]:
    st.markdown('<div class="kicker">Report</div>', unsafe_allow_html=True)
    st.markdown("## Player Profile")

    if df_f.empty or NAME_COL not in df_f.columns:
        st.warning("No players available with current filters.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique().tolist())
        default_player = (
            st.session_state.selected_player
            or st.session_state.pinned.get(position)
            or (players[0] if players else None)
        )
        if default_player not in players and players:
            default_player = players[0]

        player = st.selectbox(
            "Select Player",
            players,
            index=players.index(default_player) if default_player in players else 0,
            key=f"profile_{position}"
        )
        
        p = df_f[df_f[NAME_COL] == player].head(1)
        row = p.iloc[0]

        # Header card
        st.markdown('<div class="card card-strong">', unsafe_allow_html=True)
        hc1, hc2, hc3, hc4, hc5 = st.columns([2.5, 1, 1, 1, 1])
        
        hc1.markdown(f"### {player}")
        hc1.caption(player_meta(row))
        hc2.metric("Age", safe_int_fmt(row.get(AGE_COL, np.nan)))
        hc3.metric("Share", safe_fmt(row.get(SHARE_COL, np.nan), 1) + "%")
        
        if "Balanced Score" in df_f.columns:
            hc4.metric("Balanced", safe_fmt(row.get("Balanced Score", np.nan), 1))
        
        in_sl = shortlist_key(position, player) in st.session_state.shortlist
        if hc5.button("★ Shortlist" if not in_sl else "✓ Shortlisted", key=f"sl_profile_{position}_{player}", type="primary"):
            if in_sl:
                remove_from_shortlist(position, player)
            else:
                add_to_shortlist(position, player)
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Notes (if shortlisted)
        if shortlist_key(position, player) in st.session_state.shortlist:
            meta = st.session_state.shortlist[shortlist_key(position, player)]
            st.markdown("### 📝 Notes")
            nc1, nc2 = st.columns([1, 2])
            meta["tags"] = nc1.text_input(
                "Tags",
                value=meta.get("tags", ""),
                placeholder="e.g. U23, target, left-footed",
                key=f"tags_{position}_{player}"
            )
            meta["notes"] = nc2.text_area(
                "Notes",
                value=meta.get("notes", ""),
                height=90,
                key=f"notes_{position}_{player}"
            )

        st.markdown("---")

        # Strengths / Weaknesses
        top, bottom = strengths_weaknesses(cfg, row, topn=8)
        swc1, swc2 = st.columns(2)
        
        with swc1:
            st.markdown("### ↑ Strengths (Percentiles)")
            for m, pct in top:
                color = COLORS["emerald"] if pct >= 80 else COLORS["yellow"]
                st.markdown(
                    f'<div style="padding:8px; margin-bottom:6px; border-left:3px solid {color}; '
                    f'background:rgba(11,11,11,0.6); border-radius:8px;">'
                    f'<strong>{m[:30]}</strong><br>'
                    f'<span style="color:{color}; font-weight:700;">{pct:.0f}th percentile</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        with swc2:
            st.markdown("### ↓ Development Areas")
            for m, pct in bottom:
                st.markdown(
                    f'<div style="padding:8px; margin-bottom:6px; border-left:3px solid {COLORS["grey"]}; '
                    f'background:rgba(11,11,11,0.6); border-radius:8px;">'
                    f'<strong>{m[:30]}</strong><br>'
                    f'<span style="color:{COLORS["grey"]}; font-weight:700;">{pct:.0f}th percentile</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # Two column layout: roles/metrics + radar/similar
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Role Suitability")
            role_cols = [c for c in role_cols_all if c in df_f.columns]
            role_data = []
            for c in role_cols:
                role_data.append({
                    "Role": DISPLAY_RENAMES.get(c, c).replace(" Score", ""),
                    "Score": safe_float(row.get(c, np.nan)),
                    "Percentile": safe_float(percentile_rank(df_f[c]).loc[p.index[0]]) if c in df_f.columns else np.nan,
                })
            
            role_df = pd.DataFrame(role_data)
            pro_table(role_df, pct_cols=["Percentile"], height=300)

            st.markdown("### Key Metrics")
            key_data = []
            for m in cfg["metrics"][:12]:  # Limit to top 12
                if m in df_f.columns and (m + " (pct)") in df_f.columns:
                    key_data.append({
                        "Metric": m[:30],
                        "Value": safe_float(row.get(m, np.nan)),
                        "Percentile": safe_float(row.get(m + " (pct)", np.nan)),
                    })
            
            if key_data:
                key_df = pd.DataFrame(key_data)
                pro_table(key_df, pct_cols=["Percentile"], height=450)
            else:
                st.info("No metric data available.")

        with right:
            st.markdown("### Role Radar")
            radar_cols = [c for c in role_cols_all if c in df_f.columns]
            if radar_cols:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=[safe_float(row.get(c, 0)) for c in radar_cols],
                        theta=[DISPLAY_RENAMES.get(c, c).replace(" Score", "") for c in radar_cols],
                        fill="toself",
                        name=player,
                        line=dict(color=COLORS["yellow"], width=2),
                        fillcolor=f"rgba(244,196,48,0.3)"
                    )
                )
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            range=[0, 100],
                            showgrid=True,
                            gridcolor="rgba(247,247,247,0.10)"
                        )
                    ),
                    height=450,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No role data for radar chart.")

            st.markdown("### Similar Players")
            st.caption("Based on cosine similarity of selected features")
            
            sim_default = [m for m in cfg["radar_metrics"] if m in df_f.columns][:6]
            sim_features = st.multiselect(
                "Select features",
                options=[m for m in cfg["metrics"] if m in df_f.columns],
                default=sim_default,
                key=f"simfeat_{position}",
            )
            topk = st.slider("Top K", 5, 20, 10, 1, key=f"simk_{position}")
            
            if sim_features:
                sim_df = similar_players(df_f, player, sim_features, topk=topk)
                if len(sim_df):
                    pro_table(sim_df, pct_cols=[], height=300)
                    
                    if st.button("➕ Add Top 3 to Compare", key=f"addsimcmp_{position}_{player}", type="primary"):
                        picks = st.session_state.compare_picks[position]
                        for nm in sim_df[NAME_COL].head(3).tolist():
                            if nm not in picks:
                                picks.append(nm)
                        st.session_state.compare_picks[position] = picks[:6]
                        st.rerun()
                else:
                    st.info("Not enough data to compute similarity.")
            else:
                st.info("Select features to find similar players.")

# =====================================================
# TAB: COMPARE
# =====================================================
with tabs[2]:
    st.markdown('<div class="kicker">Head-to-Head</div>', unsafe_allow_html=True)
    st.markdown("## Compare Players")

    if df_f.empty or NAME_COL not in df_f.columns:
        st.warning("No players available with current filters.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique().tolist())
        picks = [p for p in st.session_state.compare_picks.get(position, []) if p in players]

        default_picks = picks[:] if picks else (players[:min(3, len(players))] if len(players) >= 2 else [])
        
        chosen = st.multiselect(
            "Select 2-6 players to compare",
            players,
            default=default_picks,
            key=f"cmp_{position}"
        )
        st.session_state.compare_picks[position] = chosen

        if len(chosen) < 2:
            st.info("📌 Select at least 2 players to compare.")
        else:
            comp_df = df_f[df_f[NAME_COL].isin(chosen)].copy()

            # Quick cards
            st.markdown("### Quick Overview")
            card_cols = st.columns(min(4, len(chosen)))
            for i, nm in enumerate(chosen[:4]):
                r = comp_df[comp_df[NAME_COL] == nm].head(1).iloc[0]
                with card_cols[i]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**{nm}**")
                    st.caption(f"{r.get(TEAM_COL, '—')} · Age {safe_int_fmt(r.get(AGE_COL, np.nan))}")
                    if "Balanced Score" in comp_df.columns:
                        st.metric("Balanced", safe_fmt(r.get("Balanced Score", np.nan), 1))
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

            # Role comparison bars
            st.markdown("### Role Suitability")
            role_cols = [c for c in role_cols_all if c in comp_df.columns]
            if role_cols:
                melt = comp_df.melt(
                    id_vars=[c for c in [NAME_COL, TEAM_COL] if c in comp_df.columns],
                    value_vars=role_cols,
                    var_name="Role",
                    value_name="Score",
                )
                melt["Role"] = melt["Role"].map(lambda x: DISPLAY_RENAMES.get(x, x).replace(" Score", ""))
                
                fig = px.bar(
                    melt,
                    x="Score",
                    y=NAME_COL,
                    color="Role",
                    barmode="group",
                    orientation="h",
                    height=400
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                    yaxis=dict(categoryorder="total ascending")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No role data available.")

            st.markdown("---")

            # Radar comparison
            st.markdown("### Performance Radar (Percentiles)")
            radar_metrics = [m + " (pct)" for m in cfg["radar_metrics"] if (m + " (pct)") in comp_df.columns]
            if radar_metrics:
                fig2 = go.Figure()
                colors_list = [COLORS["yellow"], COLORS["emerald"], "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]
                
                for i, nm in enumerate(chosen):
                    sub = comp_df[comp_df[NAME_COL] == nm].head(1)
                    if not sub.empty:
                        r = [safe_float(sub.iloc[0].get(m, 0)) for m in radar_metrics]
                        theta = [m.replace(" (pct)", "")[:15] for m in radar_metrics]
                        
                        fig2.add_trace(
                            go.Scatterpolar(
                                r=r,
                                theta=theta,
                                fill="toself",
                                name=nm,
                                line=dict(color=colors_list[i % len(colors_list)], width=2)
                            )
                        )
                
                fig2.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            range=[0, 100],
                            gridcolor="rgba(247,247,247,0.10)"
                        )
                    ),
                    height=600,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No metric percentiles available for radar.")

            st.markdown("---")

            # Detailed table
            st.markdown("### Detailed Comparison")
            show_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] + role_cols if c in comp_df.columns]
            sort_role = "Balanced Score" if "Balanced Score" in comp_df.columns else (role_cols[0] if role_cols else show_cols[-1])
            
            display_df = comp_df[show_cols].sort_values(sort_role, ascending=False) if sort_role in comp_df.columns else comp_df[show_cols]
            st.dataframe(rename_for_display(display_df), use_container_width=True, height=400)

# =====================================================
# TAB: LEADERBOARDS
# =====================================================
with tabs[3]:
    st.markdown('<div class="kicker">Rankings</div>', unsafe_allow_html=True)
    st.markdown("## Leaderboards")

    if df_f.empty:
        st.info("No players available to rank with current filters.")
    else:
        available_roles = [r for r in role_cols_all if r in df_f.columns]
        if not available_roles:
            st.warning("No role score columns found.")
        else:
            lbc1, lbc2 = st.columns([2, 1])
            
            with lbc1:
                role = st.selectbox(
                    "Ranking by",
                    available_roles,
                    index=available_roles.index(default_sort) if default_sort in available_roles else 0,
                    key=f"lb_role_{position}",
                )
            
            with lbc2:
                n = st.slider("Top N", 10, 100, 30, 5, key=f"lb_n_{position}")

            st.markdown("---")

            # Leaderboard table
            cols_to_show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, role] if c in df_f.columns]
            out = df_f.sort_values(role, ascending=False).head(n)[cols_to_show].copy()
            out.insert(0, "Rank", range(1, len(out) + 1))
            out[role + " (pct)"] = percentile_rank(df_f[role]).reindex(out.index)

            pro_table(rename_for_display(out), pct_cols=[role + " (pct)"], height=600)

            st.markdown("---")

            # Top 15 bar chart
            st.markdown(f"### Top 15 - {DISPLAY_RENAMES.get(role, role)}")
            fig = px.bar(
                df_f.sort_values(role, ascending=False).head(15),
                x=role,
                y=NAME_COL,
                orientation="h",
                color=role,
                hover_data=[c for c in [TEAM_COL, COMP_COL, AGE_COL] if c in df_f.columns],
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["white"]),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB: ANALYTICS
# =====================================================
with tabs[4]:
    st.markdown('<div class="kicker">Insights</div>', unsafe_allow_html=True)
    st.markdown("## Analytics & Distributions")

    if df_f.empty:
        st.info("No data available for analysis.")
    else:
        numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            default_metric = default_sort if default_sort in numeric_cols else numeric_cols[0]
            metric = st.selectbox(
                "Select metric to analyze",
                numeric_cols,
                index=numeric_cols.index(default_metric),
                key=f"dist_{position}"
            )

            metric_values = df_f[metric].dropna()
            
            if len(metric_values) == 0:
                st.warning(f"No data available for {metric}")
            else:
                # Summary stats
                st.markdown("### Summary Statistics")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                
                stats = [
                    (safe_fmt(metric_values.mean(), 2), "Mean", sc1),
                    (safe_fmt(metric_values.median(), 2), "Median", sc2),
                    (safe_fmt(metric_values.std(), 2), "Std Dev", sc3),
                    (safe_fmt(metric_values.min(), 2), "Min", sc4),
                    (safe_fmt(metric_values.max(), 2), "Max", sc5),
                ]
                
                for value, label, col in stats:
                    col.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{value}</div>'
                        f'<div class="metric-label">{label}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("---")

                # Distribution plots
                dc1, dc2 = st.columns(2)
                
                with dc1:
                    st.markdown("#### Histogram")
                    fig1 = px.histogram(df_f, x=metric, nbins=30, color_discrete_sequence=[COLORS["yellow"]])
                    fig1.add_vline(
                        x=metric_values.mean(),
                        line_dash="dash",
                        line_color=COLORS["emerald"],
                        annotation_text=f"Mean: {metric_values.mean():.2f}",
                        annotation_position="top"
                    )
                    fig1.update_layout(
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=COLORS["white"]),
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with dc2:
                    st.markdown("#### Box Plot")
                    fig2 = px.box(df_f, y=metric, points="all", color_discrete_sequence=[COLORS["yellow"]])
                    fig2.update_layout(
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=COLORS["white"]),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("---")

                # Group analysis
                st.markdown("### Group Analysis")
                split_options = [c for c in [COMP_COL, TEAM_COL] if c in df_f.columns]
                if split_options:
                    gac1, gac2 = st.columns([2, 1])
                    
                    with gac1:
                        split = st.radio("Split by", split_options, horizontal=True, key=f"split_{position}")
                    
                    with gac2:
                        topk = st.slider("Top groups", 5, 30, 12, 1, key=f"topk_{position}")
                    
                    g = df_f.groupby(split, dropna=True)[metric].mean().sort_values(ascending=False).head(topk).reset_index()
                    
                    fig3 = px.bar(
                        g,
                        x=metric,
                        y=split,
                        orientation="h",
                        color=metric,
                        color_continuous_scale="Viridis"
                    )
                    fig3.update_layout(
                        height=500,
                        yaxis=dict(categoryorder="total ascending"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=COLORS["white"]),
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No grouping columns available (Competition or Team).")

# =====================================================
# TAB: SHORTLIST
# =====================================================
with tabs[5]:
    st.markdown('<div class="kicker">Targets</div>', unsafe_allow_html=True)
    st.markdown("## Shortlist")

    items = []
    for k, meta in st.session_state.shortlist.items():
        if "||" in k:
            pos, name = k.split("||", 1)
            added = meta.get("added", dt.datetime.now())
            added_str = added.strftime("%Y-%m-%d") if isinstance(added, dt.datetime) else "Unknown"
            items.append({
                "Position": pos,
                "Name": name,
                "Tags": meta.get("tags", ""),
                "Notes": meta.get("notes", ""),
                "Added": added_str
            })

    if not items:
        st.markdown(
            '<div class="card" style="text-align:center; padding:3rem;">'
            '<div style="font-size:4rem; opacity:0.5; margin-bottom:1rem;">⭐</div>'
            '<h2 style="color:#9AA0A6;">Your Shortlist is Empty</h2>'
            '<p style="color:#9AA0A6;">Add players from other tabs to build your target list.</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Stats
        st.markdown(f"### ⭐ {len(items)} Players Shortlisted")
        
        slc1, slc2, slc3, slc4 = st.columns(4)
        
        stats = [
            (len(items), "Total", "👥", slc1),
            (len(set(item["Position"] for item in items)), "Positions", "⚽", slc2),
            (len([item for item in items if item["Tags"].strip()]), "Tagged", "🏷️", slc3),
            (len([item for item in items if item["Notes"].strip()]), "With Notes", "📝", slc4),
        ]
        
        for value, label, icon, col in stats:
            col.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:1.5rem; margin-bottom:0.5rem;">{icon}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Editable table
        sl_df = pd.DataFrame(items)
        edited = st.data_editor(
            sl_df,
            use_container_width=True,
            height=500,
            num_rows="dynamic",
            column_config={
                "Position": st.column_config.SelectboxColumn(
                    "Position",
                    options=list(POSITION_CONFIG.keys()),
                    width="small"
                ),
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Tags": st.column_config.TextColumn("Tags", width="medium"),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
                "Added": st.column_config.TextColumn("Added", width="small"),
            },
            key="shortlist_editor",
        )

        # Update session state with edits
        new_shortlist = {}
        for _, r in edited.iterrows():
            pos = str(r.get("Position", "")).strip()
            name = str(r.get("Name", "")).strip()
            if pos and name:
                new_shortlist[shortlist_key(pos, name)] = {
                    "tags": str(r.get("Tags", "") or ""),
                    "notes": str(r.get("Notes", "") or ""),
                    "added": r.get("Added", dt.datetime.now())
                }
        st.session_state.shortlist = new_shortlist

        st.markdown("---")

        # Export options
        slec1, slec2, slec3 = st.columns(3)
        
        with slec1:
            csv_data = edited.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name=f"shortlist_{dt.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with slec2:
            json_data = edited.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                "🔧 Download JSON",
                data=json_data,
                file_name=f"shortlist_{dt.datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with slec3:
            if st.button("🗑️ Clear All", use_container_width=True, type="primary"):
                if st.session_state.get("confirm_clear"):
                    st.session_state.shortlist = {}
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm clearing shortlist.")
