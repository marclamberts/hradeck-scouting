"""
Scout Lab Pro - Ultimate Edition
Professional Football Analytics Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import zscore
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Scout Lab Pro - Ultimate Edition",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# =====================================================
# COLORS & THEME
# =====================================================
COLORS = {
    "yellow": "#F4C430",
    "black": "#0B0B0B",
    "white": "#F7F7F7",
    "grey": "#9AA0A6",
    "emerald": "#2ECC71",
    "blue": "#3498db",
    "red": "#e74c3c",
    "purple": "#9b59b6",
    "orange": "#f39c12",
    "cyan": "#1abc9c",
    "background": "#0e1117",
}

# =====================================================
# POSITION CONFIGURATIONS
# =====================================================
POSITION_CONFIG = {
    "GK": {
        "file": "Goalkeepers.xlsx",
        "title": "Goalkeepers",
        "icon": "🧤",
        "color": COLORS["purple"],
        "role_cols": ["Ball Playing GK", "Box Defender", "Shot Stopper", "Sweeper Keeper"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low pass", "Diagonal pass", "Prevented Goals Percent (based on post-shot xG)",
            "Defensive Touches outside the Box per game", "Caught Balls Percent"
        ],
    },
    "CB": {
        "file": "Central_Defenders.xlsx",
        "title": "Central Defenders",
        "icon": "🛡️",
        "color": COLORS["blue"],
        "role_cols": ["Aerially Dominant CB", "Aggressive CB", "Ball Playing CB", "Strategic CB"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low pass", "Diagonal pass", "Ground duel", "Defensive Header", "Interception"
        ],
    },
    "LB": {
        "file": "Left_Back.xlsx",
        "title": "Left Backs",
        "icon": "⬅️",
        "color": COLORS["emerald"],
        "role_cols": ["Classic Back 4 LB", "Creative LB", "Left Wing-Back"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low pass", "High Cross", "Low Cross", "Ground duel", "Interception"
        ],
    },
    "RB": {
        "file": "Right_Back.xlsx",
        "title": "Right Backs",
        "icon": "➡️",
        "color": COLORS["orange"],
        "role_cols": ["Classic Back 4 RB", "Creative RB", "Right Wing-Back"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low pass", "High Cross", "Low Cross", "Ground duel", "Interception"
        ],
    },
    "DM": {
        "file": "Defensive_Midfielder.xlsx",
        "title": "Defensive Midfielders",
        "icon": "⚓",
        "color": COLORS["red"],
        "role_cols": ["Anchorman", "Ball Winning Midfielder", "Box-to-Box Midfielder", "Central Creator", "Deep Lying Playmaker"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low pass", "Diagonal pass", "Ground duel", "Interception", "Loose ball regain"
        ],
    },
    "CM": {
        "file": "Central_Midfielder.xlsx",
        "title": "Central Midfielders",
        "icon": "⭐",
        "color": COLORS["yellow"],
        "role_cols": ["Anchorman", "Ball Winning Midfielder", "Box-to-Box Midfielder", "Central Creator", "Deep Lying Playmaker"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low pass", "Diagonal pass", "Availability Between the Lines", "Mid range shot"
        ],
    },
    "AM": {
        "file": "Attacking_Midfielder.xlsx",
        "title": "Attacking Midfielders",
        "icon": "🎯",
        "color": COLORS["cyan"],
        "role_cols": ["Central Creator", "Deep Lying Striker"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low pass", "Dribble", "Availability Between the Lines", "Mid range shot", "Availability in the Box"
        ],
    },
    "LW": {
        "file": "Left_Winger.xlsx",
        "title": "Left Wingers",
        "icon": "⚡",
        "color": COLORS["emerald"],
        "role_cols": ["Central Creator", "Classic Left Winger", "Deep Running Left Winger", "Defensive Left Winger", "Left Wing-Back"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low Cross", "Dribble", "Availability in the Box", "Close range shot"
        ],
    },
    "RW": {
        "file": "Right_Wing.xlsx",
        "title": "Right Wingers",
        "icon": "⚡",
        "color": COLORS["orange"],
        "role_cols": ["Central Creator", "Classic Right Winger", "Deep Running Right Winger", "Defensive Right Winger", "Right Wing-Back"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Low Cross", "Dribble", "Availability in the Box", "Close range shot"
        ],
    },
    "ST": {
        "file": "Strikers.xlsx",
        "title": "Strikers",
        "icon": "⚽",
        "color": COLORS["red"],
        "role_cols": ["Complete Forward", "Deep Lying Striker", "Deep Running Striker", "Poacher", "Pressing Striker", "Second Striker", "Target Man"],
        "key_metrics": [
            "IMPECT", "Offensive IMPECT", "Defensive IMPECT",
            "Availability in the Box", "Close range shot", "Header shot", "1-vs-1 against GK Shot"
        ],
    }
}

# Column names
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# =====================================================
# ENHANCED CSS
# =====================================================
def load_css():
    st.markdown(
        f"""
<style>
/* Hide Streamlit elements */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
.stDeployButton {{display: none;}}

/* Base styles */
.stApp {{
    background: {COLORS["background"]};
    color: {COLORS["white"]};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(11,11,11,0.95) 0%, rgba(14,17,23,0.98) 100%);
    border-right: 1px solid rgba(244,196,48,0.15);
}}

/* Typography */
h1, h2, h3 {{
    letter-spacing: -0.03em;
    margin-top: 0.5rem;
    font-weight: 800;
}}

.kicker {{
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-size: 0.7rem;
    color: {COLORS["grey"]};
    font-weight: 900;
    margin-bottom: 0.5rem;
}}

/* Cards */
.card {{
    background: rgba(11,11,11,0.7);
    border: 1px solid rgba(244,196,48,0.12);
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}}

.card:hover {{
    border-color: rgba(244,196,48,0.3);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}}

/* Hero section */
.hero {{
    background: linear-gradient(135deg, rgba(11,11,11,0.95) 0%, rgba(14,17,23,0.95) 100%);
    border: 2px solid rgba(244,196,48,0.25);
    border-radius: 24px;
    padding: 4rem 2rem;
    text-align: center;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
}}

.hero::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(244,196,48,0.05) 0%, transparent 70%);
    animation: pulse 15s ease-in-out infinite;
}}

@keyframes pulse {{
    0%, 100% {{transform: scale(1);}}
    50% {{transform: scale(1.1);}}
}}

.hero h1 {{
    font-size: clamp(3rem, 8vw, 5rem);
    font-weight: 950;
    background: linear-gradient(135deg, {COLORS["yellow"]} 0%, {COLORS["emerald"]} 50%, {COLORS["cyan"]} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
}}

/* Feature grid */
.feature-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}}

.feature-card {{
    background: rgba(11,11,11,0.8);
    border: 1px solid rgba(244,196,48,0.15);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}}

.feature-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, {COLORS["yellow"]}, {COLORS["emerald"]});
    transform: scaleX(0);
    transition: transform 0.3s ease;
}}

.feature-card:hover {{
    border-color: rgba(244,196,48,0.5);
    transform: translateY(-8px);
    box-shadow: 0 12px 24px rgba(244,196,48,0.2);
}}

.feature-card:hover::before {{
    transform: scaleX(1);
}}

/* Position grid */
.position-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}}

.position-card {{
    background: rgba(11,11,11,0.7);
    border: 1px solid rgba(244,196,48,0.12);
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
}}

.position-card:hover {{
    border-color: rgba(244,196,48,0.6);
    background: rgba(244,196,48,0.1);
    transform: translateY(-4px);
}}

/* Header bar */
.headerbar {{
    background: linear-gradient(135deg, rgba(11,11,11,0.9) 0%, rgba(14,17,23,0.9) 100%);
    border: 1px solid rgba(244,196,48,0.2);
    border-radius: 18px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    flex-wrap: wrap;
    gap: 1rem;
}}

/* Pills */
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid rgba(247,247,247,0.12);
    background: rgba(11,11,11,0.6);
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 900;
    font-size: 0.85rem;
}}

.pill-solid {{
    border: 1px solid rgba(244,196,48,0.7);
    background: {COLORS["yellow"]};
    color: {COLORS["black"]};
    box-shadow: 0 4px 12px rgba(244,196,48,0.3);
}}

/* Player rows */
.player-row {{
    border: 1px solid rgba(247,247,247,0.1);
    border-radius: 14px;
    background: rgba(11,11,11,0.6);
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}}

.player-row:hover {{
    border-color: rgba(244,196,48,0.4);
    background: rgba(11,11,11,0.75);
    transform: translateX(4px);
}}

.player-name {{
    font-weight: 950;
    font-size: 1.1rem;
    margin-bottom: 4px;
}}

/* Buttons */
div.stButton > button {{
    border-radius: 12px;
    font-weight: 900;
    transition: all 0.2s ease;
    border: 1px solid rgba(247,247,247,0.15);
}}

div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}}

button[kind="primary"] {{
    border: 1px solid rgba(244,196,48,0.7) !important;
    background: {COLORS["yellow"]} !important;
    color: {COLORS["black"]} !important;
    box-shadow: 0 4px 12px rgba(244,196,48,0.4) !important;
}}

button[kind="primary"]:hover {{
    box-shadow: 0 6px 20px rgba(244,196,48,0.6) !important;
}}

/* Tables */
div[data-testid="stDataFrame"] {{
    border: 1px solid rgba(247,247,247,0.1);
    border-radius: 12px;
    overflow: hidden;
}}

div[data-testid="stDataFrame"] thead tr th {{
    font-weight: 950;
    background: rgba(11,11,11,0.9) !important;
    border-bottom: 2px solid rgba(244,196,48,0.4) !important;
    padding: 1rem !important;
}}

/* Metrics */
.metric-card {{
    text-align: center;
    padding: 1.25rem;
    border-radius: 12px;
    background: rgba(11,11,11,0.7);
    border: 1px solid rgba(247,247,247,0.1);
    transition: all 0.3s ease;
}}

.metric-card:hover {{
    border-color: rgba(244,196,48,0.3);
    transform: translateY(-4px);
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 950;
    color: {COLORS["yellow"]};
    margin: 0.5rem 0;
}}

.metric-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {COLORS["grey"]};
    font-weight: 700;
}}

.stat-box {{
    background: rgba(244,196,48,0.1);
    border: 1px solid rgba(244,196,48,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}}

.stat-box:hover {{
    background: rgba(244,196,48,0.15);
    border-color: rgba(244,196,48,0.5);
    transform: scale(1.05);
}}

.stat-number {{
    font-size: 3rem;
    font-weight: 950;
    color: {COLORS["yellow"]};
    text-shadow: 0 2px 8px rgba(244,196,48,0.3);
}}

.stat-label {{
    font-size: 1rem;
    color: {COLORS["grey"]};
    margin-top: 0.5rem;
    font-weight: 700;
}}

/* Scrollbar */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: rgba(11,11,11,0.5);
}}

::-webkit-scrollbar-thumb {{
    background: rgba(244,196,48,0.5);
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: rgba(244,196,48,0.7);
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    font-weight: 900;
    border-radius: 8px;
}}

/* Loading spinner */
.stSpinner > div {{
    border-top-color: {COLORS["yellow"]} !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )

load_css()

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def safe_float(x):
    """Safely convert to float"""
    if x is None or pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}:
        return np.nan
    s = s.replace("%", "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    if s.count(",") >= 1 and s.count(".") == 1:
        s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def safe_fmt(x, decimals=2):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"

def safe_int_fmt(x):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{int(round(v))}"

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

def percentile_rank(s):
    s = s.apply(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    if mask.sum() > 0:
        out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def player_meta(row):
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    nat = str(row.get(NAT_COL, "—"))
    age = safe_int_fmt(row.get(AGE_COL, np.nan))
    share = safe_fmt(row.get(SHARE_COL, np.nan), 1)
    return f"{team} · {comp} · {nat} · Age {age} · {share}% share"

def cosine_similarity_matrix(X):
    X = np.nan_to_num(X, nan=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T

def similar_players(df_f, player_name, feature_cols, topk=10):
    """Find similar players using cosine similarity"""
    try:
        if df_f.empty or NAME_COL not in df_f.columns:
            return pd.DataFrame()
        if player_name not in df_f[NAME_COL].values:
            return pd.DataFrame()

        cols = [c for c in feature_cols if c in df_f.columns and pd.api.types.is_numeric_dtype(df_f[c])]
        if not cols:
            return pd.DataFrame()

        # Build feature matrix
        X_list = []
        for c in cols:
            col_vals = df_f[c].values
            # Handle NaN values before z-score
            col_vals_clean = np.where(np.isnan(col_vals), np.nanmean(col_vals), col_vals)
            if np.all(col_vals_clean == col_vals_clean[0]):  # Constant column
                z_vals = np.zeros_like(col_vals_clean)
            else:
                z_vals = zscore(col_vals_clean, nan_policy='omit')
                z_vals = np.nan_to_num(z_vals, nan=0.0)
            X_list.append(z_vals)
        
        X = np.column_stack(X_list)
        sim = cosine_similarity_matrix(X)

        idx = df_f.index[df_f[NAME_COL] == player_name][0]
        base_i = df_f.index.get_loc(idx)
        scores = pd.Series(sim[base_i], index=df_f.index)

        out = df_f.loc[scores.sort_values(ascending=False).index].copy()
        out["Similarity"] = scores.loc[out.index].values * 100
        out = out[out[NAME_COL] != player_name].head(topk)

        show_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] if c in out.columns] + ["Similarity"]
        return out[show_cols]
    except Exception as e:
        st.error(f"Error finding similar players: {e}")
        return pd.DataFrame()

def pro_table(df, pct_cols=None, height=600):
    """Display professional table"""
    if df.empty:
        st.info("No data to display")
        return
        
    pct_cols = pct_cols or []
    pct_cols = [c for c in pct_cols if c in df.columns]
    col_config = {}

    for c in pct_cols:
        col_config[c] = st.column_config.ProgressColumn(
            label=c, min_value=0, max_value=100, format="%.0f"
        )

    for c in df.columns:
        if c in pct_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            col_config[c] = st.column_config.NumberColumn(label=c, format="%.2f")

    st.dataframe(df, height=height, column_config=col_config, hide_index=True, use_container_width=True)

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data(show_spinner=False)
def load_and_prepare(position_key):
    """Load and prepare data for a position"""
    try:
        cfg = POSITION_CONFIG[position_key]
        
        # Try multiple file locations
        possible_paths = [
            Path(cfg["file"]),
            Path("/mnt/user-data/uploads") / cfg["file"],
            Path("uploads") / cfg["file"],
        ]
        
        fp = None
        for path in possible_paths:
            if path.exists():
                fp = path
                break
        
        if fp is None:
            st.error(f"❌ File not found: `{cfg['file']}`\n\nPlease ensure the file is in the same directory as this script.")
            st.stop()

        df = pd.read_excel(fp)
        df.columns = [str(c).strip() for c in df.columns]

        # Get numeric columns
        numeric_cols = []
        for col in df.columns:
            if col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL, "Player-ID"]:
                continue
            if "BetterThan" in col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) or col in cfg.get("key_metrics", []):
                numeric_cols.append(col)

        # Convert to numeric
        coerce_numeric(df, numeric_cols + [AGE_COL, SHARE_COL])

        # Clean text columns
        for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace("nan", "").str.strip()

        # Calculate percentiles
        for m in numeric_cols:
            if m in df.columns:
                df[m + " (pct)"] = percentile_rank(df[m])

        return df, cfg, numeric_cols
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# =====================================================
# STATE MANAGEMENT
# =====================================================
def ensure_state():
    """Initialize session state"""
    defaults = {
        "filters": {},
        "shortlist": {},
        "compare_picks": {},
        "page": "landing",
        "selected_position": None,
        "selected_player": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def shortlist_key(position_key, player_name):
    return f"{position_key}||{player_name}"

def add_to_shortlist(position_key, player_name):
    k = shortlist_key(position_key, player_name)
    if k not in st.session_state.shortlist:
        st.session_state.shortlist[k] = {
            "tags": "",
            "notes": "",
            "added": dt.datetime.now()
        }

def remove_from_shortlist(position_key, player_name):
    k = shortlist_key(position_key, player_name)
    if k in st.session_state.shortlist:
        del st.session_state.shortlist[k]

def default_filters_for(df):
    """Create default filters"""
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
        "min_share": 0.0,
        "competitions": [],
        "teams": [],
        "nats": [],
        "age_range": (lo, hi),
    }

def apply_filters(df, f):
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
# VISUALIZATION FUNCTIONS
# =====================================================
def create_radar_chart(df, player_names, metrics, title="Player Comparison"):
    """Create radar chart"""
    fig = go.Figure()
    
    colors = [COLORS["yellow"], COLORS["emerald"], COLORS["blue"], COLORS["red"], COLORS["purple"], COLORS["orange"]]
    
    for i, player in enumerate(player_names):
        player_data = df[df[NAME_COL] == player].head(1)
        if player_data.empty:
            continue
        
        values = []
        labels = []
        for m in metrics:
            pct_col = m + " (pct)"
            if pct_col in player_data.columns:
                val = safe_float(player_data.iloc[0].get(pct_col, 0))
                values.append(val if not np.isnan(val) else 0)
                labels.append(m[:20])
        
        if values:
            color = colors[i % len(colors)]
            hex_color = color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=player,
                line=dict(color=color, width=2),
                fillcolor=f"rgba({r}, {g}, {b}, 0.25)"
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(247,247,247,0.1)")
        ),
        title=title,
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["white"]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig

def create_role_comparison_chart(df, player_names, role_cols):
    """Create role comparison bar chart"""
    data = []
    for player in player_names:
        player_data = df[df[NAME_COL] == player].head(1)
        if not player_data.empty:
            for role in role_cols:
                if role in player_data.columns:
                    val = safe_float(player_data.iloc[0].get(role, 0))
                    data.append({
                        "Player": player,
                        "Role": role.replace("Score", "").strip()[:25],
                        "Score": val if not np.isnan(val) else 0
                    })
    
    if not data:
        return None
    
    df_plot = pd.DataFrame(data)
    fig = px.bar(
        df_plot,
        x="Score",
        y="Player",
        color="Role",
        barmode="group",
        orientation="h",
        height=max(400, len(player_names) * 80)
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["white"]),
        yaxis=dict(categoryorder="total ascending"),
        title="Role Suitability Comparison"
    )
    
    return fig

def create_scatter_plot(df, x_metric, y_metric, color_metric=None):
    """Create scatter plot"""
    hover_data = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL] if c in df.columns]
    
    if color_metric and color_metric in df.columns:
        fig = px.scatter(
            df, x=x_metric, y=y_metric, color=color_metric,
            hover_data=hover_data, color_continuous_scale="Viridis", height=600
        )
    else:
        fig = px.scatter(
            df, x=x_metric, y=y_metric, hover_data=hover_data,
            height=600, color_discrete_sequence=[COLORS["yellow"]]
        )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["white"]),
        title=f"{y_metric} vs {x_metric}"
    )
    
    return fig

def create_distribution_plot(df, metric):
    """Create distribution plot"""
    fig = px.histogram(df, x=metric, nbins=30, color_discrete_sequence=[COLORS["yellow"]])
    
    metric_values = df[metric].dropna()
    if len(metric_values) > 0:
        fig.add_vline(
            x=metric_values.mean(),
            line_dash="dash",
            line_color=COLORS["emerald"],
            annotation_text=f"Mean: {metric_values.mean():.2f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["white"]),
        title=f"Distribution of {metric}"
    )
    
    return fig

def create_heatmap(df, metrics, n_players=20):
    """Create performance heatmap"""
    top_players = df.nlargest(n_players, "IMPECT") if "IMPECT" in df.columns else df.head(n_players)
    
    heatmap_data = []
    for metric in metrics:
        pct_col = metric + " (pct)"
        if pct_col in top_players.columns:
            heatmap_data.append(top_players[pct_col].fillna(0).values)
    
    if not heatmap_data:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=top_players[NAME_COL].values if NAME_COL in top_players.columns else list(range(n_players)),
        y=[m[:20] for m in metrics],
        colorscale="Viridis",
        hovertemplate='Player: %{x}<br>Metric: %{y}<br>Percentile: %{z:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {n_players} Players - Performance Heatmap",
        height=max(400, len(metrics) * 30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["white"]),
        xaxis=dict(tickangle=-45)
    )
    
    return fig

# =====================================================
# LANDING PAGE
# =====================================================
def render_landing_page():
    """Render landing page"""
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:5rem;margin-bottom:1.5rem;">⚽</div>
            <h1>Scout Lab Pro</h1>
            <p style="font-size:1.6rem;color:#9AA0A6;margin-bottom:2rem;position:relative;z-index:1;">
                Ultimate Football Analytics Platform
            </p>
            <p style="font-size:1.15rem;color:#F7F7F7;max-width:750px;margin:0 auto;line-height:1.7;position:relative;z-index:1;">
                Professional scouting intelligence with 10,000+ players across 10 positions. 
                Advanced visualizations, AI-powered comparisons, and comprehensive role analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Stats
    st.markdown("### 📊 Platform Overview")
    stat_cols = st.columns(4)
    
    stats = [
        ("10,000+", "Players", "👥"),
        ("10", "Positions", "⚽"),
        ("100+", "Metrics", "📈"),
        ("50+", "Leagues", "🏆"),
    ]
    
    for i, (number, label, icon) in enumerate(stats):
        with stat_cols[i]:
            st.markdown(
                f'''
                <div class="stat-box">
                    <div style="font-size:2.5rem;margin-bottom:0.75rem;">{icon}</div>
                    <div class="stat-number">{number}</div>
                    <div class="stat-label">{label}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Features
    st.markdown("### ✨ Core Features")
    
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div style="font-size:3.5rem;margin-bottom:1.25rem;">🔍</div>
                <h3 style="margin-bottom:0.75rem;">Advanced Search</h3>
                <p style="color:#9AA0A6;line-height:1.6;">Multi-dimensional filtering by age, team, competition, match share, and performance metrics</p>
            </div>
            <div class="feature-card">
                <div style="font-size:3.5rem;margin-bottom:1.25rem;">📊</div>
                <h3 style="margin-bottom:0.75rem;">Interactive Viz</h3>
                <p style="color:#9AA0A6;line-height:1.6;">Radar charts, heatmaps, scatter plots, distributions, and correlation matrices</p>
            </div>
            <div class="feature-card">
                <div style="font-size:3.5rem;margin-bottom:1.25rem;">⚖️</div>
                <h3 style="margin-bottom:0.75rem;">Player Compare</h3>
                <p style="color:#9AA0A6;line-height:1.6;">Side-by-side comparison with up to 6 players across all metrics</p>
            </div>
            <div class="feature-card">
                <div style="font-size:3.5rem;margin-bottom:1.25rem;">🤖</div>
                <h3 style="margin-bottom:0.75rem;">AI Similarity</h3>
                <p style="color:#9AA0A6;line-height:1.6;">Find similar players using cosine similarity and machine learning</p>
            </div>
            <div class="feature-card">
                <div style="font-size:3.5rem;margin-bottom:1.25rem;">🎯</div>
                <h3 style="margin-bottom:0.75rem;">Role Analysis</h3>
                <p style="color:#9AA0A6;line-height:1.6;">Position-specific role suitability scores and tactical profiling</p>
            </div>
            <div class="feature-card">
                <div style="font-size:3.5rem;margin-bottom:1.25rem;">⭐</div>
                <h3 style="margin-bottom:0.75rem;">Target Management</h3>
                <p style="color:#9AA0A6;line-height:1.6;">Shortlist players with custom tags, notes, and multi-format export</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Positions
    st.markdown("### ⚽ Select Position")
    
    positions_html = '<div class="position-grid">'
    for key, cfg in POSITION_CONFIG.items():
        positions_html += f'''
        <div class="position-card">
            <div style="font-size:3.5rem;margin-bottom:0.75rem;">{cfg["icon"]}</div>
            <div style="font-size:1.15rem;font-weight:900;margin-bottom:0.5rem;color:{cfg["color"]};">{cfg["title"]}</div>
        </div>
        '''
    positions_html += '</div>'
    
    st.markdown(positions_html, unsafe_allow_html=True)

    # Selector
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        position_options = {k: f"{v['icon']} {v['title']}" for k, v in POSITION_CONFIG.items()}
        selected = st.selectbox(
            "Choose Your Position",
            list(position_options.keys()),
            format_func=lambda x: position_options[x],
            label_visibility="collapsed"
        )
        
        if st.button("🚀 Launch Scout Lab", type="primary", use_container_width=True):
            st.session_state.page = "app"
            st.session_state.selected_position = selected
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;padding:2.5rem;color:#9AA0A6;">
            <p style="font-size:0.95rem;margin-bottom:0.5rem;">Built with Streamlit & Plotly</p>
            <p style="font-size:0.85rem;">Professional Football Analytics & Data Science</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# MAIN APP LOGIC
# =====================================================
ensure_state()

# Route to landing or app
if st.session_state.page == "landing":
    render_landing_page()
    st.stop()

# Continue with main app
if st.session_state.selected_position is None:
    st.session_state.selected_position = "GK"

# Sidebar
st.sidebar.markdown("### ⚙️ Scout Control")

if st.sidebar.button("← Back to Home"):
    st.session_state.page = "landing"
    st.rerun()

# Position selector
position_options = {k: f"{v['icon']} {v['title']}" for k, v in POSITION_CONFIG.items()}
position = st.sidebar.selectbox(
    "Position",
    list(position_options.keys()),
    index=list(POSITION_CONFIG.keys()).index(st.session_state.selected_position),
    format_func=lambda x: position_options[x]
)

# Load data
with st.spinner("Loading player database..."):
    df, cfg, all_metrics = load_and_prepare(position)

# Initialize filters
if position not in st.session_state.filters:
    st.session_state.filters[position] = default_filters_for(df)
f = st.session_state.filters[position]

# Filters
with st.sidebar.expander("🔍 Filters", expanded=True):
    f["q"] = st.text_input("Search", value=f.get("q", ""), placeholder="Player, team, league...")
    f["min_share"] = st.slider("Min Match Share (%)", 0.0, 50.0, float(f.get("min_share", 0.0)), 0.5)

    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        if len(vals):
            min_age = int(max(15, np.floor(vals.min())))
            max_age = int(min(50, np.ceil(vals.max())))
            lo, hi = f.get("age_range", (min_age, max_age))
            f["age_range"] = st.slider("Age Range", min_age, max_age, (lo, hi))

    if COMP_COL in df.columns:
        comps = sorted([c for c in df[COMP_COL].dropna().unique() if str(c).strip()])
        f["competitions"] = st.multiselect("Competitions", comps, default=f.get("competitions", []))

    if TEAM_COL in df.columns:
        teams = sorted([t for t in df[TEAM_COL].dropna().unique() if str(t).strip()])
        f["teams"] = st.multiselect("Teams", teams, default=f.get("teams", []))

    if NAT_COL in df.columns:
        nats = sorted([n for n in df[NAT_COL].dropna().unique() if str(n).strip()])
        f["nats"] = st.multiselect("Nationalities", nats, default=f.get("nats", []))

    if st.button("🔄 Reset", type="primary", use_container_width=True):
        st.session_state.filters[position] = default_filters_for(df)
        st.rerun()

# Apply filters
df_f = apply_filters(df, f)

# Initialize compare picks
if position not in st.session_state.compare_picks:
    st.session_state.compare_picks[position] = []

# Header
shortlist_count = len(st.session_state.shortlist)
teams_n = df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0
comps_n = df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0

st.markdown(
    f"""
<div class="headerbar">
  <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
    <div style="font-size:1.6rem;font-weight:950;">{cfg['icon']} Scout Lab Pro</div>
    <span class="pill" style="border-color:{cfg['color']}50;background:{cfg['color']}25;color:{cfg['color']};">{cfg["title"]}</span>
    <span class="pill">Players <strong>{len(df_f):,}</strong></span>
    <span class="pill">Teams <strong>{teams_n}</strong></span>
    <span class="pill">Comps <strong>{comps_n}</strong></span>
  </div>
  <span class="pill pill-solid">⭐ {shortlist_count}</span>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs
tabs = st.tabs([
    "🔍 Scout",
    "👤 Profile",
    "⚖️ Compare",
    "🏆 Rankings",
    "📊 Analytics",
    "🎯 Advanced",
    "⭐ Targets"
])

# TAB 1: SCOUT
with tabs[0]:
    st.markdown('<div class="kicker">Search & Discover</div>', unsafe_allow_html=True)
    st.markdown("## Player Scout")

    if df_f.empty:
        st.info("🔍 No players match your filters. Try adjusting the criteria.")
        if st.button("Reset All Filters", type="primary"):
            st.session_state.filters[position] = default_filters_for(df)
            st.rerun()
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            sort_opts = ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"] + cfg.get("role_cols", [])
            sort_opts = [c for c in sort_opts if c in df_f.columns]
            sort_col = st.selectbox("Sort by", sort_opts)
        with c2:
            view_count = st.selectbox("Show", [20, 50, 100, 200], index=1)

        st.markdown("---")

        results = df_f.sort_values(sort_col, ascending=False).head(view_count)
        
        for idx, (_, row) in enumerate(results.iterrows()):
            name = str(row.get(NAME_COL, "—"))
            in_sl = shortlist_key(position, name) in st.session_state.shortlist

            st.markdown('<div class="player-row">', unsafe_allow_html=True)
            
            rc1, rc2, rc3 = st.columns([3.5, 1.2, 1.3])
            
            with rc1:
                st.markdown(f'<div class="player-name">{name}</div>', unsafe_allow_html=True)
                st.caption(player_meta(row))
            
            with rc2:
                score = safe_fmt(row.get(sort_col, np.nan), 1)
                pct_col = sort_col + " (pct)"
                pct = safe_fmt(row.get(pct_col, np.nan), 0) if pct_col in row else "—"
                st.metric(sort_col[:15], score, delta=f"{pct}th" if pct != "—" else None)
            
            with rc3:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("View", key=f"v{idx}", use_container_width=True):
                        st.session_state.selected_player = name
                        st.rerun()
                with b2:
                    if st.button("✓" if in_sl else "★", key=f"s{idx}", use_container_width=True):
                        (remove_from_shortlist if in_sl else add_to_shortlist)(position, name)
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

# TAB 2: PROFILE
with tabs[1]:
    st.markdown('<div class="kicker">Deep Dive</div>', unsafe_allow_html=True)
    st.markdown("## Player Profile")

    if df_f.empty:
        st.warning("No players available.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique())
        default = st.session_state.get("selected_player") or (players[0] if players else None)
        if default not in players and players:
            default = players[0]

        player = st.selectbox("Select Player", players, index=players.index(default) if default in players else 0)
        
        p_data = df_f[df_f[NAME_COL] == player].head(1)
        if not p_data.empty:
            row = p_data.iloc[0]

            st.markdown('<div class="card">', unsafe_allow_html=True)
            h1, h2, h3, h4, h5 = st.columns([2.5, 1, 1, 1, 1])
            h1.markdown(f"### {player}")
            h1.caption(player_meta(row))
            h2.metric("Age", safe_int_fmt(row.get(AGE_COL)))
            h3.metric("Share", safe_fmt(row.get(SHARE_COL), 1) + "%")
            h4.metric("IMPECT", safe_fmt(row.get("IMPECT"), 2))
            
            in_sl = shortlist_key(position, player) in st.session_state.shortlist
            if h5.button("✓ Listed" if in_sl else "★ Add", type="primary"):
                (remove_from_shortlist if in_sl else add_to_shortlist)(position, player)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Key Stats")
            s1, s2, s3, s4 = st.columns(4)
            stats = [
                ("Off. IMPECT", safe_fmt(row.get("Offensive IMPECT"), 2), s1),
                ("Def. IMPECT", safe_fmt(row.get("Defensive IMPECT"), 2), s2),
                ("Off. %ile", safe_fmt(row.get("Offensive IMPECT (pct)"), 0), s3),
                ("Def. %ile", safe_fmt(row.get("Defensive IMPECT (pct)"), 0), s4),
            ]
            for label, val, col in stats:
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            st.markdown("### Role Suitability")
            role_cols = cfg.get("role_cols", [])
            if role_cols:
                role_data = []
                for rc in role_cols:
                    if rc in df_f.columns:
                        val = safe_float(row.get(rc, 0))
                        pct = safe_float(percentile_rank(df_f[rc]).loc[p_data.index[0]])
                        role_data.append({"Role": rc[:30], "Score": val, "Percentile": pct})
                
                if role_data:
                    pro_table(pd.DataFrame(role_data), pct_cols=["Percentile"], height=250)

            st.markdown("---")

            left, right = st.columns(2)
            
            with left:
                st.markdown("### Top Metrics (Percentiles)")
                strengths = []
                for m in cfg.get("key_metrics", all_metrics[:10]):
                    pct_col = m + " (pct)"
                    if pct_col in row:
                        pct = safe_float(row.get(pct_col))
                        if not np.isnan(pct):
                            strengths.append((m, pct))
                
                strengths.sort(key=lambda x: x[1], reverse=True)
                for m, pct in strengths[:8]:
                    color = COLORS["emerald"] if pct >= 75 else COLORS["yellow"]
                    st.markdown(
                        f'<div style="padding:10px;margin-bottom:8px;border-left:3px solid {color};'
                        f'background:rgba(11,11,11,0.6);border-radius:10px;">'
                        f'<strong>{m[:30]}</strong><br>'
                        f'<span style="color:{color};font-weight:700;">{pct:.0f}th percentile</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            with right:
                st.markdown("### Performance Radar")
                radar_metrics = cfg.get("key_metrics", all_metrics[:8])
                fig = create_radar_chart(df_f, [player], radar_metrics)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.markdown("### Similar Players")
            sim_features = st.multiselect(
                "Features",
                [m for m in all_metrics if m in df_f.columns],
                default=cfg.get("key_metrics", all_metrics[:6]),
                key=f"sim{position}"
            )
            
            if sim_features:
                sim_df = similar_players(df_f, player, sim_features, topk=10)
                if not sim_df.empty:
                    pro_table(sim_df, height=300)

# TAB 3: COMPARE
with tabs[2]:
    st.markdown('<div class="kicker">Head-to-Head</div>', unsafe_allow_html=True)
    st.markdown("## Compare Players")

    if df_f.empty:
        st.warning("No players available.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique())
        picks = [p for p in st.session_state.compare_picks.get(position, []) if p in players]
        default = picks if picks else players[:min(3, len(players))]
        
        chosen = st.multiselect("Select 2-6 players", players, default=default, key=f"cmp{position}")
        st.session_state.compare_picks[position] = chosen

        if len(chosen) < 2:
            st.info("Select at least 2 players to compare.")
        else:
            comp_df = df_f[df_f[NAME_COL].isin(chosen)]

            st.markdown("### Quick View")
            cards = st.columns(min(len(chosen), 4))
            for i, name in enumerate(chosen[:4]):
                r = comp_df[comp_df[NAME_COL] == name].iloc[0]
                with cards[i]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**{name}**")
                    st.caption(f"{r.get(TEAM_COL)} · Age {safe_int_fmt(r.get(AGE_COL))}")
                    st.metric("IMPECT", safe_fmt(r.get("IMPECT"), 2))
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

            st.markdown("### Role Comparison")
            role_cols = cfg.get("role_cols", [])
            if role_cols:
                fig = create_role_comparison_chart(comp_df, chosen, role_cols)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.markdown("### Performance Radar")
            radar_metrics = cfg.get("key_metrics", all_metrics[:8])
            fig = create_radar_chart(comp_df, chosen, radar_metrics)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.markdown("### Detailed Stats")
            show_cols = [c for c in [NAME_COL, TEAM_COL, AGE_COL, "IMPECT", "Offensive IMPECT", "Defensive IMPECT"] + role_cols if c in comp_df.columns]
            st.dataframe(comp_df[show_cols].sort_values("IMPECT", ascending=False), use_container_width=True, height=400)

# TAB 4: RANKINGS
with tabs[3]:
    st.markdown('<div class="kicker">Leaderboards</div>', unsafe_allow_html=True)
    st.markdown("## Rankings")

    if not df_f.empty:
        lc1, lc2 = st.columns([2, 1])
        
        with lc1:
            sort_opts = ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"] + cfg.get("role_cols", []) + all_metrics[:10]
            sort_opts = [c for c in sort_opts if c in df_f.columns]
            metric = st.selectbox("Rank by", sort_opts)
        
        with lc2:
            n = st.slider("Top N", 10, 100, 30, 5)

        st.markdown("---")

        cols_show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, metric] if c in df_f.columns]
        ranking = df_f.sort_values(metric, ascending=False).head(n)[cols_show].copy()
        ranking.insert(0, "Rank", range(1, len(ranking) + 1))
        pct_col = metric + " (pct)"
        if pct_col in df_f.columns:
            ranking[pct_col] = percentile_rank(df_f[metric]).reindex(ranking.index)
        
        pro_table(ranking, pct_cols=[pct_col] if pct_col in ranking.columns else [], height=600)

        st.markdown("---")

        st.markdown(f"### Top 15 - {metric}")
        fig = px.bar(
            df_f.sort_values(metric, ascending=False).head(15),
            x=metric, y=NAME_COL, orientation="h", color=metric,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["white"])
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 5: ANALYTICS
with tabs[4]:
    st.markdown('<div class="kicker">Insights</div>', unsafe_allow_html=True)
    st.markdown("## Analytics")

    if not df_f.empty:
        numeric_cols = [c for c in all_metrics if c in df_f.columns]
        metric = st.selectbox("Metric", numeric_cols, key="analytics")

        metric_values = df_f[metric].dropna()
        
        if len(metric_values) > 0:
            st.markdown("### Summary")
            s1, s2, s3, s4, s5 = st.columns(5)
            
            stats = [
                (safe_fmt(metric_values.mean(), 2), "Mean", s1),
                (safe_fmt(metric_values.median(), 2), "Median", s2),
                (safe_fmt(metric_values.std(), 2), "Std", s3),
                (safe_fmt(metric_values.min(), 2), "Min", s4),
                (safe_fmt(metric_values.max(), 2), "Max", s5),
            ]
            
            for val, label, col in stats:
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            dc1, dc2 = st.columns(2)
            
            with dc1:
                st.markdown("#### Distribution")
                fig = create_distribution_plot(df_f, metric)
                st.plotly_chart(fig, use_container_width=True)
            
            with dc2:
                st.markdown("#### Box Plot")
                fig = px.box(df_f, y=metric, points="all", color_discrete_sequence=[COLORS["yellow"]])
                fig.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"])
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.markdown("### Group Analysis")
            if COMP_COL in df_f.columns or TEAM_COL in df_f.columns:
                gc1, gc2 = st.columns([2, 1])
                
                with gc1:
                    group_opts = [c for c in [COMP_COL, TEAM_COL] if c in df_f.columns]
                    group_by = st.radio("Group by", group_opts, horizontal=True)
                
                with gc2:
                    topk = st.slider("Top", 5, 30, 12, 1)
                
                grouped = df_f.groupby(group_by, dropna=True)[metric].mean().sort_values(ascending=False).head(topk).reset_index()
                
                fig = px.bar(grouped, x=metric, y=group_by, orientation="h", color=metric, color_continuous_scale="Viridis")
                fig.update_layout(
                    height=500,
                    yaxis=dict(categoryorder="total ascending"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"])
                )
                st.plotly_chart(fig, use_container_width=True)

# TAB 6: ADVANCED
with tabs[5]:
    st.markdown('<div class="kicker">Advanced</div>', unsafe_allow_html=True)
    st.markdown("## Advanced Viz")

    if not df_f.empty:
        viz_type = st.radio(
            "Type",
            ["Scatter Plot", "Performance Heatmap", "Correlation"],
            horizontal=True
        )

        st.markdown("---")

        if viz_type == "Scatter Plot":
            sc1, sc2, sc3 = st.columns(3)
            
            with sc1:
                x_metric = st.selectbox("X", all_metrics, key="scx")
            with sc2:
                y_metric = st.selectbox("Y", all_metrics, key="scy", index=min(1, len(all_metrics)-1))
            with sc3:
                color_opts = ["None"] + all_metrics
                color_metric = st.selectbox("Color", color_opts, key="scc")
            
            if x_metric and y_metric:
                fig = create_scatter_plot(df_f, x_metric, y_metric, None if color_metric == "None" else color_metric)
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Performance Heatmap":
            n_players = st.slider("Players", 10, 50, 20, 5)
            selected = st.multiselect(
                "Metrics",
                all_metrics,
                default=cfg.get("key_metrics", all_metrics[:8])
            )
            
            if selected:
                fig = create_heatmap(df_f, selected, n_players)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Correlation":
            selected = st.multiselect(
                "Metrics",
                all_metrics,
                default=cfg.get("key_metrics", all_metrics[:10])
            )
            
            if len(selected) >= 2:
                corr_data = df_f[selected].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_data.values,
                    x=[m[:20] for m in corr_data.columns],
                    y=[m[:20] for m in corr_data.index],
                    colorscale="RdBu",
                    zmid=0,
                    text=corr_data.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Correlation Matrix",
                    height=600,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                    xaxis=dict(tickangle=-45)
                )
                
                st.plotly_chart(fig, use_container_width=True)

# TAB 7: SHORTLIST
with tabs[6]:
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
            '<div class="card" style="text-align:center;padding:3rem;">'
            '<div style="font-size:4rem;opacity:0.5;margin-bottom:1rem;">⭐</div>'
            '<h2 style="color:#9AA0A6;">Shortlist Empty</h2>'
            '<p style="color:#9AA0A6;">Add players from Scout tab</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"### ⭐ {len(items)} Players")
        
        slc1, slc2, slc3, slc4 = st.columns(4)
        
        stats = [
            (len(items), "Total", "👥", slc1),
            (len(set(i["Position"] for i in items)), "Positions", "⚽", slc2),
            (len([i for i in items if i["Tags"]]), "Tagged", "🏷️", slc3),
            (len([i for i in items if i["Notes"]]), "Notes", "📝", slc4),
        ]
        
        for val, label, icon, col in stats:
            col.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:1.5rem;margin-bottom:0.5rem;">{icon}</div>'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        sl_df = pd.DataFrame(items)
        edited = st.data_editor(
            sl_df,
            use_container_width=True,
            height=500,
            num_rows="dynamic",
            column_config={
                "Position": st.column_config.SelectboxColumn("Position", options=list(POSITION_CONFIG.keys())),
                "Name": st.column_config.TextColumn("Name"),
                "Tags": st.column_config.TextColumn("Tags"),
                "Notes": st.column_config.TextColumn("Notes"),
                "Added": st.column_config.TextColumn("Added"),
            },
            key="sl_editor",
        )

        new_shortlist = {}
        for _, r in edited.iterrows():
            pos = str(r["Position"]).strip()
            name = str(r["Name"]).strip()
            if pos and name:
                new_shortlist[shortlist_key(pos, name)] = {
                    "tags": str(r.get("Tags", "") or ""),
                    "notes": str(r.get("Notes", "") or ""),
                    "added": r.get("Added", dt.datetime.now())
                }
        st.session_state.shortlist = new_shortlist

        st.markdown("---")

        slec1, slec2, slec3 = st.columns(3)
        
        with slec1:
            csv_data = edited.to_csv(index=False).encode("utf-8")
            st.download_button("📄 CSV", csv_data, f"shortlist_{dt.datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
        
        with slec2:
            json_data = edited.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button("🔧 JSON", json_data, f"shortlist_{dt.datetime.now().strftime('%Y%m%d')}.json", "application/json", use_container_width=True)
        
        with slec3:
            if st.button("🗑️ Clear", use_container_width=True, type="primary"):
                st.session_state.shortlist = {}
                st.rerun()
