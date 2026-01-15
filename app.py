import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import hashlib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Scout Lab Pro", layout="wide", page_icon="⚽")

# =====================================================
# MODERN COLOR PALETTE
# =====================================================
COLORS = {
    "primary": "#00D9FF",      # Cyan accent
    "secondary": "#FF6B9D",    # Pink accent
    "success": "#00F5A0",      # Green
    "warning": "#FFD93D",      # Yellow
    "dark": "#0A0E27",         # Deep navy
    "darker": "#050816",       # Almost black
    "card": "#151B3B",         # Card background
    "border": "#1E2749",       # Subtle borders
    "text": "#E8EAED",         # Light text
    "muted": "#8B92B0",        # Muted text
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
ID_COL = "Player-ID"

# =====================================================
# POSITION CONFIGURATIONS - ALL 10 POSITIONS
# =====================================================
POSITION_CONFIG = {
    "GK": {
        "file": "/mnt/user-data/uploads/Goalkeepers.xlsx",
        "title": "Goalkeepers",
        "icon": "🧤",
        "role_prefix": ["Ball Playing GK", "Box Defender", "Shot Stopper", "Sweeper Keeper"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "CB": {
        "file": "/mnt/user-data/uploads/Central_Defenders.xlsx",
        "title": "Central Defenders",
        "icon": "🛡️",
        "role_prefix": ["Aerially Dominant CB", "Aggressive CB", "Ball Playing CB", "Strategic CB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "LB": {
        "file": "/mnt/user-data/uploads/Left_Back.xlsx",
        "title": "Left Backs",
        "icon": "⬅️",
        "role_prefix": ["Attacking FB", "Defensive FB", "Progressive FB", "Inverted FB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "RB": {
        "file": "/mnt/user-data/uploads/Right_Back.xlsx",
        "title": "Right Backs",
        "icon": "➡️",
        "role_prefix": ["Attacking FB", "Defensive FB", "Progressive FB", "Inverted FB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "DM": {
        "file": "/mnt/user-data/uploads/Defensive_Midfielder.xlsx",
        "title": "Defensive Midfielders",
        "icon": "⚓",
        "role_prefix": ["Anchorman", "Ball Winning Midfielder", "Deep Lying Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "CM": {
        "file": "/mnt/user-data/uploads/Central_Midfielder.xlsx",
        "title": "Central Midfielders",
        "icon": "⭐",
        "role_prefix": ["Anchorman", "Ball Winning Midfielder", "Box-to-Box Midfielder", "Central Creator", "Deep Lying Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "AM": {
        "file": "/mnt/user-data/uploads/Attacking_Midfielder.xlsx",
        "title": "Attacking Midfielders",
        "icon": "🎯",
        "role_prefix": ["Advanced Playmaker", "Central Creator", "Shadow Striker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "LW": {
        "file": "/mnt/user-data/uploads/Left_Winger.xlsx",
        "title": "Left Wingers",
        "icon": "⚡",
        "role_prefix": ["Inside Forward", "Touchline Winger", "Wide Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "RW": {
        "file": "/mnt/user-data/uploads/Right_Wing.xlsx",
        "title": "Right Wingers",
        "icon": "⚡",
        "role_prefix": ["Inside Forward", "Touchline Winger", "Wide Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "ST": {
        "file": "/mnt/user-data/uploads/Strikers.xlsx",
        "title": "Strikers",
        "icon": "⚽",
        "role_prefix": ["Complete Forward", "Deep Lying Striker", "Deep Running Striker", "Poacher", "Pressing Striker", "Second Striker", "Target Man"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
}

# =====================================================
# ULTRA-MODERN CSS
# =====================================================
st.markdown(
    f"""
<style>
/* Reset & Hide Streamlit UI */
#MainMenu, footer, header {{visibility: hidden;}}
.stDeployButton {{display: none;}}

/* Root Variables */
:root {{
    --primary: {COLORS["primary"]};
    --secondary: {COLORS["secondary"]};
    --success: {COLORS["success"]};
    --dark: {COLORS["dark"]};
    --darker: {COLORS["darker"]};
    --card: {COLORS["card"]};
    --border: {COLORS["border"]};
    --text: {COLORS["text"]};
    --muted: {COLORS["muted"]};
}}

/* Global Background */
.stApp {{
    background: linear-gradient(135deg, {COLORS["darker"]} 0%, {COLORS["dark"]} 100%);
    color: {COLORS["text"]};
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
}}

/* Container */
.block-container {{
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1400px;
    margin: 0 auto;
}}

/* Sidebar Redesign */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {COLORS["darker"]} 0%, {COLORS["dark"]} 100%);
    border-right: 1px solid {COLORS["border"]};
    padding-top: 1rem;
}}

section[data-testid="stSidebar"] .block-container {{
    padding: 1rem !important;
}}

/* Typography */
h1, h2, h3, h4, h5, h6 {{
    color: {COLORS["text"]};
    font-weight: 700;
    letter-spacing: -0.025em;
}}

h1 {{font-size: 2.5rem; margin-bottom: 0.5rem;}}
h2 {{font-size: 1.75rem; margin-bottom: 0.5rem;}}
h3 {{font-size: 1.35rem; margin-bottom: 0.4rem;}}
h4 {{font-size: 1.1rem; margin-bottom: 0.3rem;}}

/* Custom Header Bar */
.header-bar {{
    background: linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["dark"]} 100%);
    border: 1px solid {COLORS["border"]};
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    position: sticky;
    top: 0;
    z-index: 100;
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: 1.5rem;
}}

.brand {{
    font-size: 1.75rem;
    font-weight: 900;
    background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}}

.position-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, {COLORS["primary"]}15 0%, {COLORS["secondary"]}15 100%);
    border: 1px solid {COLORS["primary"]}40;
    padding: 0.5rem 1.25rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    color: {COLORS["text"]};
}}

.stat-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 600;
    color: {COLORS["text"]};
    transition: all 0.3s ease;
}}

.stat-pill:hover {{
    border-color: {COLORS["primary"]};
    transform: translateY(-2px);
}}

.stat-pill strong {{
    color: {COLORS["primary"]};
    font-weight: 800;
}}

/* Cards */
.modern-card {{
    background: linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["dark"]} 100%);
    border: 1px solid {COLORS["border"]};
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}}

.modern-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 217, 255, 0.1);
    border-color: {COLORS["primary"]}40;
}}

.player-card {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    cursor: pointer;
}}

.player-card:hover {{
    border-color: {COLORS["primary"]};
    background: {COLORS["dark"]};
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(0, 217, 255, 0.15);
}}

.player-name {{
    font-size: 1.1rem;
    font-weight: 800;
    color: {COLORS["text"]};
    margin-bottom: 0.3rem;
    letter-spacing: -0.01em;
}}

.player-meta {{
    font-size: 0.85rem;
    color: {COLORS["muted"]};
    line-height: 1.4;
}}

/* Metric Cards */
.metric-card {{
    background: linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["dark"]} 100%);
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}}

.metric-card:hover {{
    border-color: {COLORS["primary"]};
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 217, 255, 0.15);
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
}}

.metric-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS["muted"]};
    font-weight: 700;
}}

/* Buttons */
div.stButton > button {{
    background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
    color: {COLORS["darker"]};
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
}}

div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0, 217, 255, 0.5);
}}

button[kind="secondary"] {{
    background: {COLORS["card"]} !important;
    color: {COLORS["text"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    box-shadow: none !important;
}}

button[kind="secondary"]:hover {{
    border-color: {COLORS["primary"]} !important;
    background: {COLORS["dark"]} !important;
}}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {{
    background: {COLORS["card"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 10px !important;
    color: {COLORS["text"]} !important;
    transition: all 0.3s ease !important;
}}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {{
    border-color: {COLORS["primary"]} !important;
    box-shadow: 0 0 0 2px {COLORS["primary"]}20 !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: {COLORS["muted"]} !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s ease !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: {COLORS["primary"]} !important;
    border-bottom-color: {COLORS["primary"]} !important;
}}

button[data-baseweb="tab"]:hover {{
    color: {COLORS["text"]} !important;
}}

/* DataFrames */
div[data-testid="stDataFrame"] {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    overflow: hidden;
}}

div[data-testid="stDataFrame"] thead tr th {{
    background: {COLORS["darker"]} !important;
    color: {COLORS["text"]} !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    border-bottom: 2px solid {COLORS["primary"]} !important;
}}

div[data-testid="stDataFrame"] tbody tr:hover {{
    background: {COLORS["dark"]} !important;
}}

/* Section Headers */
.section-header {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: {COLORS["muted"]};
    font-weight: 900;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {COLORS["border"]};
}}

/* Strength/Weakness Lists */
.strength-item {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}}

.strength-item:hover {{
    border-color: {COLORS["success"]};
    background: {COLORS["dark"]};
}}

.weakness-item {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}}

.weakness-item:hover {{
    border-color: {COLORS["secondary"]};
    background: {COLORS["dark"]};
}}

/* Sliders */
div[data-baseweb="slider"] {{
    padding-top: 0.5rem;
}}

div[data-baseweb="slider"] > div > div {{
    background: {COLORS["border"]} !important;
}}

div[data-baseweb="slider"] > div > div > div {{
    background: linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%) !important;
}}

/* Progress Bars */
div[data-testid="stProgress"] > div > div {{
    background: {COLORS["border"]} !important;
}}

div[data-testid="stProgress"] > div > div > div {{
    background: linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["success"]} 100%) !important;
}}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {COLORS["muted"]} !important;
    font-size: 0.85rem;
}}

/* Info/Warning boxes */
div[data-testid="stNotification"] {{
    background: {COLORS["card"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 12px !important;
    color: {COLORS["text"]} !important;
}}

/* Expander */
div[data-testid="stExpander"] {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
}}

/* Scrollbar */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: {COLORS["darker"]};
}}

::-webkit-scrollbar-thumb {{
    background: {COLORS["border"]};
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {COLORS["primary"]};
}}

/* Sticky Pinned Section */
.sticky-panel {{
    position: sticky;
    top: 120px;
}}

/* Loading Animation */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

.loading {{
    animation: pulse 2s ease-in-out infinite;
}}

/* Responsive */
@media (max-width: 768px) {{
    .header-bar {{
        flex-direction: column;
        gap: 1rem;
        align-items: flex-start;
    }}
    
    .block-container {{
        padding: 1rem !important;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# UTILITY FUNCTIONS
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
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
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

def make_rowid(row: pd.Series, position: str) -> str:
    parts = [position, str(row.get(NAME_COL, "")), str(row.get(TEAM_COL, "")), str(row.get(COMP_COL, "")), str(row.name)]
    raw = "||".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

def player_meta(row: pd.Series) -> str:
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    nat = str(row.get(NAT_COL, "—"))
    age = safe_int_fmt(row.get(AGE_COL, np.nan))
    share = safe_fmt(row.get(SHARE_COL, np.nan), 1)
    return f"{team} • {comp} • {nat} • Age {age} • {share}% share"

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data(show_spinner=False)
def load_position_data(position_key: str) -> pd.DataFrame:
    cfg = POSITION_CONFIG[position_key]
    fp = Path(cfg["file"])
    
    if not fp.exists():
        raise FileNotFoundError(f"Missing {cfg['file']}")
    
    df = pd.read_excel(fp)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Identify role columns (first columns after Match Share that are numeric)
    role_cols = []
    metric_cols = []
    
    for col in df.columns:
        if col in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, NAT_COL, SHARE_COL, ID_COL]:
            continue
        # Role columns are typically the first batch before IMPECT
        if any(prefix in col for prefix in cfg["role_prefix"]):
            role_cols.append(col)
        elif col not in ["IMPECT - BetterThan", "Offensive IMPECT - BetterThan", "Defensive IMPECT - BetterThan"]:
            if "IMPECT" in col or pd.api.types.is_numeric_dtype(df[col]):
                metric_cols.append(col)
    
    # Coerce numeric columns
    all_numeric = role_cols + metric_cols + [AGE_COL, SHARE_COL]
    coerce_numeric(df, all_numeric)
    
    # Clean text columns
    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()
    
    # Calculate percentiles for all numeric metrics
    for m in metric_cols:
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
            df[m + " (pct)"] = percentile_rank(df[m])
    
    # Store configuration
    cfg["role_cols"] = role_cols
    cfg["metric_cols"] = metric_cols
    cfg["all_metrics"] = role_cols + metric_cols
    
    return df

# =====================================================
# FILTERS
# =====================================================
def default_filters_for(df: pd.DataFrame):
    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        if len(vals):
            lo = int(max(15, np.floor(vals.min())))
            hi = int(min(50, np.ceil(vals.max())))
        else:
            lo, hi = 15, 45
    else:
        lo, hi = 15, 45
    return {"q": "", "min_share": 0.0, "competitions": [], "teams": [], "nats": [], "age_range": (lo, hi)}

def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
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
                mask = mask | out[col].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]
    
    return out

def strengths_weaknesses(cfg: dict, row: pd.Series, topn: int = 5):
    pairs = []
    for m in cfg["metric_cols"]:
        if m in ["IMPECT - BetterThan", "Offensive IMPECT - BetterThan", "Defensive IMPECT - BetterThan"]:
            continue
        pct = safe_float(row.get(m + " (pct)", np.nan))
        if not np.isnan(pct):
            pairs.append((m, pct))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:topn]
    bottom = list(reversed(pairs[-topn:])) if len(pairs) >= topn else list(reversed(pairs))
    return top, bottom

# =====================================================
# STATE MANAGEMENT
# =====================================================
def ensure_state():
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "shortlist" not in st.session_state:
        st.session_state.shortlist = {}
    if "pinned" not in st.session_state:
        st.session_state.pinned = {}
    if "selected_player" not in st.session_state:
        st.session_state.selected_player = {}
    if "compare_picks" not in st.session_state:
        st.session_state.compare_picks = {}

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

# =====================================================
# MAIN APP
# =====================================================
ensure_state()

# Sidebar
with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Control Panel</div>', unsafe_allow_html=True)
    
    position = st.selectbox(
        "Position",
        list(POSITION_CONFIG.keys()),
        format_func=lambda x: f"{POSITION_CONFIG[x]['icon']} {POSITION_CONFIG[x]['title']}",
        index=0
    )

# Load data
with st.spinner("Loading data..."):
    df = load_position_data(position)
    cfg = POSITION_CONFIG[position]

# Initialize position-specific state
if position not in st.session_state.filters:
    st.session_state.filters[position] = default_filters_for(df)
if position not in st.session_state.pinned:
    st.session_state.pinned[position] = None
if position not in st.session_state.selected_player:
    st.session_state.selected_player[position] = None
if position not in st.session_state.compare_picks:
    st.session_state.compare_picks[position] = []

f = st.session_state.filters[position]

# Sidebar Filters
with st.sidebar:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    
    f["q"] = st.text_input("🔍 Search", value=f.get("q", ""), placeholder="Player, team, competition...")
    
    f["min_share"] = st.slider("📊 Min Match Share (%)", 0.0, 100.0, float(f.get("min_share", 0.0)), 5.0)
    
    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        min_age = int(max(15, np.floor(vals.min()))) if len(vals) else 15
        max_age = int(min(50, np.ceil(vals.max()))) if len(vals) else 45
        lo, hi = f.get("age_range", (min_age, max_age))
        lo = max(min_age, lo)
        hi = min(max_age, hi)
        f["age_range"] = st.slider("🎂 Age Range", min_age, max_age, (lo, hi), 1)
    
    if COMP_COL in df.columns:
        comps_all = sorted([c for c in df[COMP_COL].dropna().unique().tolist() if str(c).strip() != ""])
        f["competitions"] = st.multiselect("🏆 Competitions", comps_all, default=f.get("competitions", []))
    
    if TEAM_COL in df.columns:
        teams_all = sorted([t for t in df[TEAM_COL].dropna().unique().tolist() if str(t).strip() != ""])
        f["teams"] = st.multiselect("⚽ Teams", teams_all, default=f.get("teams", []))
    
    if NAT_COL in df.columns:
        nats_all = sorted([n for n in df[NAT_COL].dropna().unique().tolist() if str(n).strip() != ""])
        f["nats"] = st.multiselect("🌍 Nationalities", nats_all, default=f.get("nats", []))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.filters[position] = default_filters_for(df)
            st.rerun()
    with col2:
        st.caption(f"✓ Live")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
df_f = apply_filters(df, f)
if not df_f.empty:
    df_f = df_f.copy()
    df_f["_rowid"] = df_f.apply(lambda r: make_rowid(r, position), axis=1)
else:
    df_f["_rowid"] = []

# Set default pinned player
if st.session_state.pinned[position] is None and len(df_f) and NAME_COL in df_f.columns:
    # Sort by IMPECT if available
    if "IMPECT" in df_f.columns:
        st.session_state.pinned[position] = df_f.sort_values("IMPECT", ascending=False).iloc[0][NAME_COL]
    else:
        st.session_state.pinned[position] = df_f.iloc[0][NAME_COL]

if st.session_state.selected_player[position] is None and st.session_state.pinned[position] is not None:
    st.session_state.selected_player[position] = st.session_state.pinned[position]

# Header Bar
shortlist_count = len(st.session_state.shortlist)
teams_n = df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0
comps_n = df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0

st.markdown(
    f"""
<div class="header-bar">
    <div class="header-left">
        <div class="brand">Scout Lab Pro</div>
        <div class="position-badge">{cfg["icon"]} {cfg["title"]}</div>
        <div class="stat-pill">Players <strong>{len(df_f)}</strong></div>
        <div class="stat-pill">Teams <strong>{teams_n}</strong></div>
        <div class="stat-pill">Competitions <strong>{comps_n}</strong></div>
    </div>
    <div style="display:flex;gap:1rem;align-items:center;">
        <div class="stat-pill" style="background: linear-gradient(135deg, {COLORS["primary"]}20 0%, {COLORS["secondary"]}20 100%); border-color: {COLORS["primary"]}60;">
            ⭐ Shortlist <strong>{shortlist_count}</strong>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs
tabs = st.tabs(["🔍 Scout", "📊 Profile", "⚖️ Compare", "🏆 Leaderboards", "📈 Analytics", "⭐ Shortlist"])

# =====================================================
# TAB 1: SCOUT (Search with pinned player)
# =====================================================
with tabs[0]:
    if df_f.empty:
        st.info("🔍 No players match your current filters. Try adjusting the criteria in the sidebar.")
    else:
        # Sort options
        sort_options = ["IMPECT"] + cfg["role_cols"] if "IMPECT" in df_f.columns else cfg["role_cols"]
        sort_options = [c for c in sort_options if c in df_f.columns]
        
        if not sort_options and cfg["metric_cols"]:
            sort_options = [c for c in cfg["metric_cols"] if c in df_f.columns][:5]
        
        sort_col = st.selectbox("📊 Sort by", options=sort_options, index=0, key=f"sort_{position}")
        
        left_col, right_col = st.columns([1.2, 0.8], gap="large")
        
        with left_col:
            st.markdown('<div class="section-header">Search Results</div>', unsafe_allow_html=True)
            
            results = df_f.sort_values(sort_col, ascending=False).head(50).copy()
            
            for _, r in results.iterrows():
                name = str(r.get(NAME_COL, "—"))
                rid = str(r.get("_rowid", r.name))
                in_sl = shortlist_key(position, name) in st.session_state.shortlist
                
                st.markdown('<div class="player-card">', unsafe_allow_html=True)
                
                col_a, col_b, col_c = st.columns([3, 1, 1])
                
                with col_a:
                    st.markdown(f'<div class="player-name">{name}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="player-meta">{player_meta(r)}</div>', unsafe_allow_html=True)
                
                with col_b:
                    score_val = safe_fmt(r.get(sort_col, np.nan), 1)
                    st.markdown(
                        f'<div class="metric-card" style="padding: 0.5rem;">'
                        f'<div class="metric-label">{sort_col.replace(" Score", "")[:15]}</div>'
                        f'<div class="metric-value" style="font-size: 1.5rem; margin: 0.25rem 0;">{score_val}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col_c:
                    if st.button("👁️ View", key=f"view_{position}_{rid}", use_container_width=True):
                        st.session_state.pinned[position] = name
                        st.session_state.selected_player[position] = name
                        st.rerun()
                    
                    if st.button("⭐" if not in_sl else "✓", key=f"sl_{position}_{rid}", use_container_width=True, type="secondary"):
                        if not in_sl:
                            add_to_shortlist(position, name)
                        else:
                            remove_from_shortlist(position, name)
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with right_col:
            st.markdown('<div class="sticky-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📌 Pinned Player</div>', unsafe_allow_html=True)
            
            pinned = st.session_state.pinned.get(position)
            
            if not pinned:
                st.info("👆 Click 'View' on any player to pin them here")
            else:
                p = df_f[df_f[NAME_COL] == pinned].head(1)
                if p.empty:
                    st.warning("Pinned player not in current filter results")
                else:
                    row = p.iloc[0]
                    
                    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                    st.markdown(f"### {pinned}")
                    st.caption(player_meta(row))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Role scores
                    if cfg["role_cols"]:
                        st.markdown("#### 🎯 Role Fit")
                        role_display_cols = st.columns(min(2, len(cfg["role_cols"])))
                        for i, rc in enumerate(cfg["role_cols"][:4]):
                            with role_display_cols[i % 2]:
                                val = safe_fmt(row.get(rc, np.nan), 0)
                                st.markdown(
                                    f'<div class="metric-card">'
                                    f'<div class="metric-label">{rc[:20]}</div>'
                                    f'<div class="metric-value" style="font-size: 1.5rem;">{val}%</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                    
                    # Top strengths
                    st.markdown("#### ⬆️ Strengths")
                    top, _ = strengths_weaknesses(cfg, row, topn=3)
                    for m, pct in top:
                        st.markdown(
                            f'<div class="strength-item">'
                            f'<span style="color: {COLORS["success"]}; font-size: 1.2rem;">↑</span>'
                            f'<div style="flex: 1;">'
                            f'<div style="font-weight: 700; font-size: 0.9rem;">{m[:30]}</div>'
                            f'<div style="color: {COLORS["muted"]}; font-size: 0.75rem;">{pct:.0f}th percentile</div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Actions
                    ac1, ac2 = st.columns(2)
                    in_sl = shortlist_key(position, pinned) in st.session_state.shortlist
                    with ac1:
                        if st.button("⭐ Shortlist" if not in_sl else "✓ Shortlisted", key=f"sl_pin_{position}", use_container_width=True):
                            if not in_sl:
                                add_to_shortlist(position, pinned)
                            else:
                                remove_from_shortlist(position, pinned)
                            st.rerun()
                    with ac2:
                        if st.button("➕ Compare", key=f"cmp_pin_{position}", use_container_width=True, type="secondary"):
                            picks = st.session_state.compare_picks[position]
                            if pinned not in picks:
                                picks.append(pinned)
                                st.session_state.compare_picks[position] = picks[:6]
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# TAB 2: PROFILE
# =====================================================
with tabs[1]:
    if df_f.empty or NAME_COL not in df_f.columns:
        st.warning("No players available with current filters.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique().tolist())
        default_player = st.session_state.selected_player.get(position) or st.session_state.pinned.get(position) or (players[0] if players else None)
        if default_player not in players and players:
            default_player = players[0]
        
        col_sel, col_act = st.columns([2.5, 1])
        
        with col_sel:
            player = st.selectbox("🎯 Select Player", players, index=players.index(default_player) if default_player in players else 0, key=f"profile_{position}")
            st.session_state.selected_player[position] = player
        
        with col_act:
            in_sl = shortlist_key(position, player) in st.session_state.shortlist
            if st.button("⭐ Shortlist" if not in_sl else "✓ Shortlisted", key=f"sl_profile_{position}", use_container_width=True):
                if not in_sl:
                    add_to_shortlist(position, player)
                else:
                    remove_from_shortlist(position, player)
                st.rerun()
        
        p = df_f[df_f[NAME_COL] == player].head(1)
        row = p.iloc[0]
        
        # Header Card
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        hc1, hc2, hc3, hc4 = st.columns([2.5, 1, 1, 1])
        
        with hc1:
            st.markdown(f"### {player}")
            st.caption(player_meta(row))
        
        with hc2:
            age_val = safe_int_fmt(row.get(AGE_COL, np.nan))
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Age</div>'
                f'<div class="metric-value" style="font-size: 1.75rem;">{age_val}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        with hc3:
            share_val = safe_fmt(row.get(SHARE_COL, np.nan), 1)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Share</div>'
                f'<div class="metric-value" style="font-size: 1.75rem;">{share_val}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        with hc4:
            impect_val = safe_fmt(row.get("IMPECT", np.nan), 2)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">IMPECT</div>'
                f'<div class="metric-value" style="font-size: 1.75rem;">{impect_val}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Strengths & Weaknesses
        top, bottom = strengths_weaknesses(cfg, row, topn=6)
        
        str_col, weak_col = st.columns(2, gap="large")
        
        with str_col:
            st.markdown("#### ⬆️ Strengths")
            for m, pct in top:
                st.markdown(
                    f'<div class="strength-item">'
                    f'<span style="color: {COLORS["success"]}; font-size: 1.3rem; font-weight: 900;">↑</span>'
                    f'<div style="flex: 1;">'
                    f'<div style="font-weight: 700;">{m}</div>'
                    f'<div style="color: {COLORS["muted"]}; font-size: 0.85rem;">{pct:.0f}th percentile</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        with weak_col:
            st.markdown("#### ⬇️ Development Areas")
            for m, pct in bottom:
                st.markdown(
                    f'<div class="weakness-item">'
                    f'<span style="color: {COLORS["secondary"]}; font-size: 1.3rem; font-weight: 900;">↓</span>'
                    f'<div style="flex: 1;">'
                    f'<div style="font-weight: 700;">{m}</div>'
                    f'<div style="color: {COLORS["muted"]}; font-size: 0.85rem;">{pct:.0f}th percentile</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        
        # Role Scores & Radar
        radar_col, data_col = st.columns([1.2, 0.8], gap="large")
        
        with radar_col:
            st.markdown("#### 🎯 Role Suitability Radar")
            if cfg["role_cols"]:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[safe_float(row.get(c, np.nan)) if not np.isnan(safe_float(row.get(c, np.nan))) else 0 for c in cfg["role_cols"]],
                    theta=[c.replace(" Score", "")[:25] for c in cfg["role_cols"]],
                    fill="toself",
                    name=player,
                    line=dict(color=COLORS["primary"], width=2),
                    fillcolor=f"{COLORS['primary']}40"
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(range=[0, 100], gridcolor=COLORS["border"], showticklabels=True),
                        angularaxis=dict(gridcolor=COLORS["border"])
                    ),
                    height=500,
                    margin=dict(l=80, r=80, t=40, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text"], size=11),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with data_col:
            st.markdown("#### 📊 Role Scores")
            if cfg["role_cols"]:
                for rc in cfg["role_cols"]:
                    val = safe_float(row.get(rc, np.nan))
                    if not np.isnan(val):
                        pct = val
                        st.markdown(
                            f'<div style="margin-bottom: 1rem;">'
                            f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">'
                            f'<span style="font-weight: 700; font-size: 0.9rem;">{rc[:25]}</span>'
                            f'<span style="font-weight: 900; color: {COLORS["primary"]};">{pct:.0f}%</span>'
                            f'</div>'
                            f'<div style="width: 100%; height: 8px; background: {COLORS["border"]}; border-radius: 4px; overflow: hidden;">'
                            f'<div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);"></div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

# =====================================================
# TAB 3: COMPARE
# =====================================================
with tabs[2]:
    if df_f.empty or NAME_COL not in df_f.columns:
        st.warning("No players available with current filters.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique().tolist())
        picks = [p for p in st.session_state.compare_picks.get(position, []) if p in players]
        default = picks[:] if len(picks) else (players[:3] if len(players) >= 3 else players[:])
        
        chosen = st.multiselect("🎯 Select players to compare (2-6)", players, default=default, key=f"cmp_{position}")
        st.session_state.compare_picks[position] = chosen
        
        if len(chosen) < 2:
            st.info("📊 Select at least 2 players to generate comparison charts")
        else:
            comp_df = df_f[df_f[NAME_COL].isin(chosen)].copy()
            
            # Role comparison
            if cfg["role_cols"]:
                st.markdown("#### 🎯 Role Suitability Comparison")
                melt = comp_df.melt(
                    id_vars=[c for c in [NAME_COL] if c in comp_df.columns],
                    value_vars=cfg["role_cols"],
                    var_name="Role",
                    value_name="Score"
                )
                
                fig = px.bar(
                    melt,
                    x="Score",
                    y=NAME_COL,
                    color="Role",
                    barmode="group",
                    orientation="h"
                )
                fig.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text"]),
                    xaxis=dict(gridcolor=COLORS["border"], range=[0, 100]),
                    yaxis=dict(gridcolor=COLORS["border"]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Radar comparison
            radar_metrics = [m + " (pct)" for m in cfg["key_metrics"] if (m + " (pct)") in comp_df.columns]
            if radar_metrics:
                st.markdown("#### 📊 Key Metrics Radar (Percentiles)")
                fig2 = go.Figure()
                
                colors_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]]
                
                for idx, nm in enumerate(chosen):
                    sub = comp_df[comp_df[NAME_COL] == nm].head(1)
                    r = [safe_float(sub.iloc[0].get(m, np.nan)) if not np.isnan(safe_float(sub.iloc[0].get(m, np.nan))) else 0 for m in radar_metrics]
                    theta = [m.replace(" (pct)", "") for m in radar_metrics]
                    color = colors_cycle[idx % len(colors_cycle)]
                    fig2.add_trace(go.Scatterpolar(
                        r=r,
                        theta=theta,
                        fill="toself",
                        name=nm,
                        line=dict(color=color, width=2),
                        fillcolor=f"{color}30"
                    ))
                
                fig2.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100], gridcolor=COLORS["border"])),
                    height=550,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text"]),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Data table
            st.markdown("#### 📋 Detailed Comparison")
            show_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] + cfg["role_cols"] + cfg["key_metrics"] if c in comp_df.columns]
            st.dataframe(
                comp_df[show_cols].sort_values(cfg["role_cols"][0] if cfg["role_cols"] else show_cols[-1], ascending=False),
                use_container_width=True,
                height=400
            )

# =====================================================
# TAB 4: LEADERBOARDS
# =====================================================
with tabs[3]:
    if df_f.empty:
        st.info("No players to rank with current filters.")
    else:
        all_sortable = ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"] + cfg["role_cols"] + cfg["metric_cols"]
        all_sortable = [c for c in all_sortable if c in df_f.columns and c not in ["IMPECT - BetterThan", "Offensive IMPECT - BetterThan", "Defensive IMPECT - BetterThan"]]
        
        if not all_sortable:
            st.warning("No sortable columns found.")
        else:
            col_sort, col_n = st.columns([2, 1])
            
            with col_sort:
                metric = st.selectbox("📊 Rank by", all_sortable, index=0 if "IMPECT" in all_sortable else 0, key=f"lb_{position}")
            
            with col_n:
                n = st.slider("Top N", 10, 100, 30, 5, key=f"lb_n_{position}")
            
            cols_show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL, metric] if c in df_f.columns]
            out = df_f.sort_values(metric, ascending=False).head(n)[cols_show].copy()
            
            # Add percentile
            out[metric + " (pct)"] = percentile_rank(df_f[metric]).reindex(out.index)
            
            # Add rank
            out.insert(0, "Rank", range(1, len(out) + 1))
            
            st.dataframe(
                out,
                use_container_width=True,
                height=600,
                column_config={
                    "Rank": st.column_config.NumberColumn("🏆 Rank", width="small"),
                    metric + " (pct)": st.column_config.ProgressColumn(
                        "Percentile",
                        min_value=0,
                        max_value=100,
                        format="%.0f"
                    )
                }
            )

# =====================================================
# TAB 5: ANALYTICS
# =====================================================
with tabs[4]:
    if df_f.empty:
        st.info("No players with current filters.")
    else:
        numeric_cols = [c for c in df_f.select_dtypes(include=[np.number]).columns.tolist() 
                       if c not in ["IMPECT - BetterThan", "Offensive IMPECT - BetterThan", "Defensive IMPECT - BetterThan"]]
        
        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            default_metric = "IMPECT" if "IMPECT" in numeric_cols else numeric_cols[0]
            metric = st.selectbox("📊 Select Metric", numeric_cols, index=numeric_cols.index(default_metric), key=f"dist_{position}")
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("#### 📊 Distribution")
                fig1 = px.histogram(df_f, x=metric, nbins=30, color_discrete_sequence=[COLORS["primary"]])
                fig1.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text"]),
                    xaxis=dict(gridcolor=COLORS["border"]),
                    yaxis=dict(gridcolor=COLORS["border"]),
                    showlegend=False
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("#### 📦 Box Plot")
                fig2 = px.box(df_f, y=metric, points="all", color_discrete_sequence=[COLORS["secondary"]])
                fig2.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text"]),
                    yaxis=dict(gridcolor=COLORS["border"]),
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Summary statistics
            st.markdown("#### 📈 Summary Statistics")
            stats_data = {
                "Metric": ["Mean", "Median", "Std Dev", "Min", "25th %ile", "75th %ile", "Max"],
                "Value": [
                    f"{df_f[metric].mean():.2f}",
                    f"{df_f[metric].median():.2f}",
                    f"{df_f[metric].std():.2f}",
                    f"{df_f[metric].min():.2f}",
                    f"{df_f[metric].quantile(0.25):.2f}",
                    f"{df_f[metric].quantile(0.75):.2f}",
                    f"{df_f[metric].max():.2f}",
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

# =====================================================
# TAB 6: SHORTLIST
# =====================================================
with tabs[5]:
    items = []
    for k, meta in st.session_state.shortlist.items():
        pos, name = k.split("||", 1)
        items.append({
            "Position": pos,
            "Player": name,
            "Tags": meta.get("tags", ""),
            "Notes": meta.get("notes", "")
        })
    
    if not items:
        st.info("⭐ Your shortlist is empty. Add players from the Scout or Profile tabs.")
    else:
        st.markdown(f"### ⭐ Your Shortlist ({len(items)} players)")
        
        sl_df = pd.DataFrame(items)
        edited = st.data_editor(
            sl_df,
            use_container_width=True,
            height=500,
            num_rows="dynamic",
            column_config={
                "Position": st.column_config.TextColumn("Position", width="small"),
                "Player": st.column_config.TextColumn("Player", width="medium"),
                "Tags": st.column_config.TextColumn("Tags", width="medium"),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            },
            key="shortlist_editor"
        )
        
        # Update state
        new_shortlist = {}
        for _, r in edited.iterrows():
            pos = str(r.get("Position", "")).strip()
            name = str(r.get("Player", "")).strip()
            if not pos or not name:
                continue
            new_shortlist[shortlist_key(pos, name)] = {
                "tags": str(r.get("Tags", "") or ""),
                "notes": str(r.get("Notes", "") or "")
            }
        st.session_state.shortlist = new_shortlist
        
        # Actions
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.download_button(
                "📥 Download CSV",
                data=edited.to_csv(index=False).encode("utf-8"),
                file_name=f"shortlist_{position}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
                st.session_state.shortlist = {}
                st.rerun()
