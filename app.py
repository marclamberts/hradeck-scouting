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
st.set_page_config(page_title="Scout Lab", layout="wide", page_icon="⚽")

# =====================================================
# COLORS (YOUR PALETTE)
# =====================================================
COLORS = {
    "yellow": "#F4C430",
    "black": "#0B0B0B",
    "white": "#F7F7F7",
    "grey": "#9AA0A6",
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
# POSITION CONFIG
# IMPORTANT: CM metric names must match your CM.xlsx headers EXACTLY.
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
        "title": "Central Defender / CM Dataset (CM.xlsx)",
        "metrics": [
            # Replace with exact CM.xlsx headers
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
# ULTRA TIGHT "WEBAPP" CSS
# =====================================================
st.markdown(
    f"""
<style>
/* Hide Streamlit chrome */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

/* Reduce outer padding massively */
.block-container {{
  padding-top: 0.65rem !important;
  padding-bottom: 1.0rem !important;
  padding-left: 1.0rem !important;
  padding-right: 1.0rem !important;
  max-width: 1180px;   /* webapp feel */
  margin: 0 auto;
}}

.stApp {{
  background: {COLORS["background"]};
  color: {COLORS["white"]};
}}

/* Sidebar compact */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(11,11,11,0.92) 0%, rgba(14,17,23,0.96) 100%);
  border-right: 1px solid rgba(244,196,48,0.12);
}}
section[data-testid="stSidebar"] * {{ color: {COLORS["white"]}; }}

/* Typography tighter */
h1,h2,h3 {{
  letter-spacing: -0.03em;
  margin-bottom: 0.25rem !important;
}}
/* make st.caption less "spaced" */
[data-testid="stCaptionContainer"] {{
  margin-top: -0.35rem;
}}

.kicker {{
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  color: {COLORS["grey"]};
  font-weight: 900;
  margin-bottom: 0.2rem;
}}

/* Reduce vertical spacing between elements */
div[data-testid="stVerticalBlock"] > div {{ gap: 0.45rem; }}

/* Cards */
.card {{
  background: rgba(11,11,11,0.62);
  border: 1px solid rgba(244,196,48,0.10);
  border-radius: 14px;
  padding: 12px 12px;
}}
.card-strong {{
  background: rgba(11,11,11,0.78);
  border: 1px solid rgba(244,196,48,0.18);
}}

/* Sticky header */
.topbar {{
  position: sticky;
  top: 0;
  z-index: 99;
  background: linear-gradient(135deg, rgba(11,11,11,0.92) 0%, rgba(14,17,23,0.92) 100%);
  border: 1px solid rgba(244,196,48,0.16);
  border-radius: 16px;
  padding: 12px 14px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  backdrop-filter: blur(8px);
}}

.topbar-left {{
  display:flex;
  gap:10px;
  align-items:baseline;
  flex-wrap: wrap;
}}
.brand {{
  font-size: 1.20rem;
  font-weight: 950;
}}
.pill {{
  display:inline-flex;
  align-items:center;
  gap:8px;
  border: 1px solid rgba(247,247,247,0.10);
  background: rgba(11,11,11,0.55);
  padding: 5px 10px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 0.86rem;
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

/* Player row compact */
.player-row {{
  border: 1px solid rgba(247,247,247,0.07);
  border-radius: 12px;
  background: rgba(11,11,11,0.55);
  padding: 10px 10px;
}}
.player-row:hover {{ border-color: rgba(244,196,48,0.28); }}

.player-name {{
  font-weight: 950;
  font-size: 0.98rem;
  line-height: 1.1;
}}
.player-meta {{
  color: {COLORS["grey"]};
  font-size: 0.86rem;
  line-height: 1.2;
  margin-top: 2px;
}}

/* Buttons compact */
div.stButton > button {{
  border-radius: 11px;
  border: 1px solid rgba(247,247,247,0.12);
  background: rgba(11,11,11,0.55);
  color: {COLORS["white"]};
  font-weight: 900;
  padding: 0.35rem 0.55rem;
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

/* Inputs compact */
div[data-baseweb="input"] > div {{
  border-radius: 11px !important;
  background: rgba(11,11,11,0.55) !important;
  border: 1px solid rgba(247,247,247,0.10) !important;
}}
div[data-baseweb="select"] > div {{
  border-radius: 11px !important;
  background: rgba(11,11,11,0.55) !important;
  border: 1px solid rgba(247,247,247,0.10) !important;
}}

/* Tabs tighter */
button[data-baseweb="tab"] {{
  font-weight: 950 !important;
  padding-top: 6px !important;
  padding-bottom: 6px !important;
}}
div[data-baseweb="tab-list"] {{ gap: 4px; }}

/* Sticky pinned panel (right side) */
.sticky {{
  position: sticky;
  top: 78px; /* below topbar */
}}

/* Dataframe header weight */
div[data-testid="stDataFrame"] thead tr th {{ font-weight: 950; }}
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

# =====================================================
# TABLE
# =====================================================
def pro_table(df: pd.DataFrame, pct_cols: list[str] | None = None, height: int = 520):
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

    st.dataframe(df, width="stretch", height=height, column_config=col_config, hide_index=True)

# =====================================================
# LOAD + PREP
# =====================================================
@st.cache_data(show_spinner=False)
def load_and_prepare(position_key: str) -> tuple[pd.DataFrame, dict]:
    cfg = POSITION_CONFIG[position_key]
    fp = Path(cfg["file"])
    if not fp.exists():
        raise FileNotFoundError(f"Missing {cfg['file']}. Put it next to app.py.")

    df = pd.read_excel(fp)
    df.columns = [str(c).strip() for c in df.columns]

    coerce_numeric(df, cfg["metrics"] + [AGE_COL, SHARE_COL])

    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    for m in cfg["metrics"]:
        if m in df.columns:
            df[m + " (pct)"] = percentile_rank(df[m])

    for role, weights in cfg["role_defs"].items():
        z = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            if col in df.columns:
                z = z + zscore(df[col]) * float(w)
        df[role] = score_from_z(z)

    coerce_numeric(df, list(cfg["role_defs"].keys()))
    return df, cfg

# =====================================================
# STATE
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

def player_meta(row: pd.Series) -> str:
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    nat = str(row.get(NAT_COL, "—"))
    age = safe_int_fmt(row.get(AGE_COL, np.nan))
    share = safe_fmt(row.get(SHARE_COL, np.nan), 2)
    return f"{team} · {comp} · {nat} · Age {age} · Share {share}"

def strengths_weaknesses(cfg: dict, row: pd.Series, topn: int = 6):
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
    X = np.nan_to_num(X, nan=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T

def similar_players(df_f: pd.DataFrame, player_name: str, feature_cols: list[str], topk: int = 10) -> pd.DataFrame:
    if df_f.empty or NAME_COL not in df_f.columns:
        return pd.DataFrame()
    if player_name not in df_f[NAME_COL].values:
        return pd.DataFrame()

    cols = [c for c in feature_cols if c in df_f.columns and pd.api.types.is_numeric_dtype(df_f[c])]
    if not cols:
        return pd.DataFrame()

    X = np.column_stack([zscore(df_f[c]).to_numpy() for c in cols])
    sim = cosine_similarity_matrix(X)

    idx = df_f.index[df_f[NAME_COL] == player_name][0]
    base_i = df_f.index.get_loc(idx)
    scores = pd.Series(sim[base_i], index=df_f.index)

    out = df_f.loc[scores.sort_values(ascending=False).index].copy()
    out["Similarity"] = scores.loc[out.index].values
    out = out[out[NAME_COL] != player_name].head(topk)

    show_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in out.columns] + ["Similarity"]
    return out[show_cols]

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
    return {"q": "", "min_share": 0.20, "competitions": [], "teams": [], "nats": [], "age_range": (lo, hi)}

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

# =====================================================
# APP BOOT
# =====================================================
ensure_state()

st.sidebar.markdown("### ⚙️ Control Room")
position = st.sidebar.selectbox("Dataset", list(POSITION_CONFIG.keys()), index=0)

df, cfg = load_and_prepare(position)

# per-position state
if position not in st.session_state.filters:
    st.session_state.filters[position] = default_filters_for(df)
if position not in st.session_state.pinned:
    st.session_state.pinned[position] = None
if position not in st.session_state.selected_player:
    st.session_state.selected_player[position] = None
if position not in st.session_state.compare_picks:
    st.session_state.compare_picks[position] = []

f = st.session_state.filters[position]
role_cols_all = [r for r in cfg["role_defs"].keys() if r in df.columns]
default_sort = cfg.get("default_sort", "Balanced Score")
if default_sort not in df.columns and role_cols_all:
    default_sort = role_cols_all[0]

# sidebar filters (tight)
with st.sidebar.container():
    st.sidebar.markdown('<div class="card">', unsafe_allow_html=True)
    f["q"] = st.sidebar.text_input("Search", value=f.get("q", ""), placeholder="Name / Team / Comp / Nat…")
    f["min_share"] = st.sidebar.slider("Min Share", 0.0, 1.0, float(f.get("min_share", 0.20)), 0.05)

    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        min_age = int(max(15, np.floor(vals.min()))) if len(vals) else 15
        max_age = int(min(50, np.ceil(vals.max()))) if len(vals) else 45
        lo, hi = f.get("age_range", (min_age, max_age))
        lo = max(min_age, lo)
        hi = min(max_age, hi)
        f["age_range"] = st.sidebar.slider("Age", min_age, max_age, (lo, hi), 1)

    if COMP_COL in df.columns:
        comps_all = sorted([c for c in df[COMP_COL].dropna().unique().tolist() if str(c).strip() != ""])
        f["competitions"] = st.sidebar.multiselect("Competitions", comps_all, default=f.get("competitions", []))

    if TEAM_COL in df.columns:
        teams_all = sorted([t for t in df[TEAM_COL].dropna().unique().tolist() if str(t).strip() != ""])
        f["teams"] = st.sidebar.multiselect("Teams", teams_all, default=f.get("teams", []))

    if NAT_COL in df.columns:
        nats_all = sorted([n for n in df[NAT_COL].dropna().unique().tolist() if str(n).strip() != ""])
        f["nats"] = st.sidebar.multiselect("Nationalities", nats_all, default=f.get("nats", []))

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Reset", type="primary", key=f"reset_{position}"):
            st.session_state.filters[position] = default_filters_for(df)
            st.rerun()
    with c2:
        st.caption("Live")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# apply filters + row ids
df_f = apply_filters(df, f)
if not df_f.empty:
    df_f = df_f.copy()
    df_f["_rowid"] = df_f.apply(lambda r: make_rowid(r, position), axis=1)
else:
    df_f["_rowid"] = []

# default pinned
if st.session_state.pinned[position] is None and len(df_f) and NAME_COL in df_f.columns:
    st.session_state.pinned[position] = df_f.sort_values(default_sort, ascending=False).iloc[0][NAME_COL]
if st.session_state.selected_player[position] is None and st.session_state.pinned[position] is not None:
    st.session_state.selected_player[position] = st.session_state.pinned[position]

# topbar
shortlist_count = len(st.session_state.shortlist)
teams_n = df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0
comps_n = df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0

st.markdown(
    f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="brand">Scout Lab</div>
    <span class="pill pill-accent">{cfg["title"]}</span>
    <span class="pill">Players <strong>{len(df_f)}</strong></span>
    <span class="pill">Teams <strong>{teams_n}</strong></span>
    <span class="pill">Comps <strong>{comps_n}</strong></span>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <span class="pill pill-solid">⭐ {shortlist_count}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Search", "Profile", "Compare", "Leaderboards", "Distributions", "Shortlist"])

# =====================================================
# SEARCH (two pane, pinned sticky)
# =====================================================
with tabs[0]:
    st.markdown('<div class="kicker">Scout</div>', unsafe_allow_html=True)

    if df_f.empty:
        st.info("No players match your filters.")
    else:
        sort_options = [c for c in ([default_sort] + role_cols_all) if c in df_f.columns]
        if not sort_options:
            sort_options = [c for c in df_f.columns if pd.api.types.is_numeric_dtype(df_f[c])]
        sort_col = st.selectbox("Sort", options=sort_options, index=0, key=f"sort_{position}")

        left, right = st.columns([1.10, 0.90], gap="medium")

        with left:
            # Compact results header
            h1, h2, h3 = st.columns([1.4, 1.0, 0.6])
            h1.markdown("### Results")
            h2.caption("Click Open → updates pinned")
            h3.caption(f"{len(df_f)} rows")

            results = df_f.sort_values(sort_col, ascending=False).head(60).copy()
            for _, r in results.iterrows():
                name = str(r.get(NAME_COL, "—"))
                rid = str(r.get("_rowid", r.name))
                in_sl = shortlist_key(position, name) in st.session_state.shortlist

                st.markdown('<div class="player-row">', unsafe_allow_html=True)
                a, b, c = st.columns([3.2, 1.1, 1.2], gap="small")
                with a:
                    st.markdown(f'<div class="player-name">{name}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="player-meta">{player_meta(r)}</div>', unsafe_allow_html=True)
                with b:
                    st.metric(DISPLAY_RENAMES.get(sort_col, sort_col).replace(" Score", ""), safe_fmt(r.get(sort_col, np.nan), 1))
                with c:
                    if st.button("Open", key=f"open_{position}_{rid}"):
                        st.session_state.pinned[position] = name
                        st.session_state.selected_player[position] = name
                        st.rerun()
                    if st.button("★" if not in_sl else "✓", key=f"sl_{position}_{rid}"):
                        add_to_shortlist(position, name) if not in_sl else remove_from_shortlist(position, name)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="sticky">', unsafe_allow_html=True)
            st.markdown("### Pinned")
            pinned = st.session_state.pinned.get(position)

            if not pinned:
                st.caption("Open a player in Results.")
            else:
                p = df_f[df_f[NAME_COL] == pinned].head(1)
                if p.empty:
                    st.caption("Pinned player not found.")
                else:
                    row = p.iloc[0]
                    st.markdown('<div class="card card-strong">', unsafe_allow_html=True)
                    st.markdown(f"**{pinned}**")
                    st.caption(player_meta(row))
                    st.markdown("</div>", unsafe_allow_html=True)

                    # compact role tiles
                    st.markdown("#### Roles")
                    tiles = st.columns(min(4, max(1, len(role_cols_all))), gap="small")
                    for i, rc in enumerate(role_cols_all[:4]):
                        tiles[i].metric(DISPLAY_RENAMES.get(rc, rc).replace(" Score", ""), safe_fmt(row.get(rc, np.nan), 1))

                    st.markdown("#### Signals (pct)")
                    top, bottom = strengths_weaknesses(cfg, row, topn=4)
                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        for m, pct in top:
                            st.write(f"↑ {m} — {pct:.0f}")
                    with c2:
                        for m, pct in bottom:
                            st.write(f"↓ {m} — {pct:.0f}")

                    a1, a2 = st.columns(2, gap="small")
                    in_sl = shortlist_key(position, pinned) in st.session_state.shortlist
                    if a1.button("Shortlist" if not in_sl else "Shortlisted", key=f"sl_pin_{position}", type="primary"):
                        add_to_shortlist(position, pinned) if not in_sl else remove_from_shortlist(position, pinned)
                        st.rerun()
                    if a2.button("Add Compare", key=f"addcmp_{position}"):
                        picks = st.session_state.compare_picks[position]
                        if pinned not in picks:
                            picks.append(pinned)
                            st.session_state.compare_picks[position] = picks[:6]
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PROFILE
# =====================================================
with tabs[1]:
    st.markdown('<div class="kicker">Report</div>', unsafe_allow_html=True)

    if df_f.empty or NAME_COL not in df_f.columns:
        st.warning("No players available with current filters.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique().tolist())
        default_player = st.session_state.selected_player.get(position) or st.session_state.pinned.get(position) or (players[0] if players else None)
        if default_player not in players and players:
            default_player = players[0]

        # compact row: select + actions
        csel, cact = st.columns([2.3, 1.0], gap="small")
        player = csel.selectbox("Player", players, index=players.index(default_player) if default_player in players else 0, key=f"profile_{position}")
        st.session_state.selected_player[position] = player

        p = df_f[df_f[NAME_COL] == player].head(1)
        row = p.iloc[0]

        in_sl = shortlist_key(position, player) in st.session_state.shortlist
        if cact.button("Shortlist" if not in_sl else "Shortlisted", type="primary", key=f"sl_profile_{position}_{player}"):
            add_to_shortlist(position, player) if not in_sl else remove_from_shortlist(position, player)
            st.rerun()

        # header card
        st.markdown('<div class="card card-strong">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1], gap="small")
        c1.markdown(f"### {player}")
        c1.caption(player_meta(row))
        c2.metric("Age", safe_int_fmt(row.get(AGE_COL, np.nan)))
        c3.metric("Share", safe_fmt(row.get(SHARE_COL, np.nan), 2))
        c4.metric("Balanced" if "Balanced Score" in df_f.columns else "Score", safe_fmt(row.get("Balanced Score", np.nan), 1) if "Balanced Score" in df_f.columns else "—")
        st.markdown("</div>", unsafe_allow_html=True)

        # strengths / risks compact
        top, bottom = strengths_weaknesses(cfg, row, topn=6)
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("#### Strengths (pct)")
            for m, pct in top:
                st.write(f"↑ **{m}** — {pct:.0f}")
        with right:
            st.markdown("#### Risks (pct)")
            for m, pct in bottom:
                st.write(f"↓ **{m}** — {pct:.0f}")

        st.markdown("---")

        colA, colB = st.columns([1.0, 1.0], gap="medium")
        with colA:
            st.markdown("#### Role scores")
            role_cols = [c for c in cfg["role_defs"].keys() if c in df_f.columns]
            role_row = pd.DataFrame(
                [{
                    "Role": DISPLAY_RENAMES.get(c, c).replace(" Score", ""),
                    "Score": safe_float(row.get(c, np.nan)),
                    "Percentile": safe_float(percentile_rank(df_f[c]).loc[p.index[0]]) if c in df_f.columns else np.nan,
                } for c in role_cols]
            )
            pro_table(role_row, pct_cols=["Percentile"], height=270)

            st.markdown("#### Key metrics (pct)")
            key = []
            for m in cfg["metrics"]:
                if m in df_f.columns and (m + " (pct)") in df_f.columns:
                    key.append({"Metric": m, "Value": safe_float(row.get(m, np.nan)), "Percentile": safe_float(row.get(m + " (pct)", np.nan))})
            key_df = pd.DataFrame(key)
            if len(key_df):
                pro_table(key_df, pct_cols=["Percentile"], height=420)
            else:
                st.info("No metric columns found.")

        with colB:
            st.markdown("#### Radar (roles)")
            radar_cols = [c for c in cfg["role_defs"].keys() if c in df_f.columns]
            if radar_cols:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[safe_float(row.get(c, np.nan)) if not np.isnan(safe_float(row.get(c, np.nan))) else 0 for c in radar_cols],
                    theta=[DISPLAY_RENAMES.get(c, c).replace(" Score", "") for c in radar_cols],
                    fill="toself",
                    name=player,
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100], gridcolor="rgba(247,247,247,0.10)")),
                    height=520,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Similar players")
            sim_default = [m for m in cfg["radar_metrics"] if m in df_f.columns][:6]
            sim_features = st.multiselect("Features", options=[m for m in cfg["metrics"] if m in df_f.columns], default=sim_default, key=f"simfeat_{position}")
            topk = st.slider("Top K", 5, 25, 10, 1, key=f"simk_{position}")

            sim_df = similar_players(df_f, player, sim_features, topk=topk)
            if len(sim_df):
                pro_table(sim_df, pct_cols=[], height=320)
            else:
                st.info("Not enough data/features to compute similarity.")

# =====================================================
# COMPARE
# =====================================================
with tabs[2]:
    st.markdown('<div class="kicker">Decision</div>', unsafe_allow_html=True)
    if df_f.empty or NAME_COL not in df_f.columns:
        st.warning("No players available with current filters.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique().tolist())
        picks = [p for p in st.session_state.compare_picks.get(position, []) if p in players]
        default = picks[:] if len(picks) else (players[:2] if len(players) >= 2 else players[:])

        chosen = st.multiselect("Players (2–6)", players, default=default, key=f"cmp_{position}")
        st.session_state.compare_picks[position] = chosen

        if len(chosen) < 2:
            st.info("Pick at least 2 players to compare.")
        else:
            comp_df = df_f[df_f[NAME_COL].isin(chosen)].copy()
            role_cols = [c for c in cfg["role_defs"].keys() if c in comp_df.columns]

            if role_cols:
                melt = comp_df.melt(
                    id_vars=[c for c in [NAME_COL, TEAM_COL, COMP_COL] if c in comp_df.columns],
                    value_vars=role_cols,
                    var_name="Role",
                    value_name="Score",
                )
                melt["Role"] = melt["Role"].map(lambda x: DISPLAY_RENAMES.get(x, x).replace(" Score", ""))
                fig = px.bar(melt, x="Score", y=NAME_COL, color="Role", barmode="group")
                fig.update_layout(
                    height=520,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig, use_container_width=True)

            radar_metrics = [m + " (pct)" for m in cfg["radar_metrics"] if (m + " (pct)") in comp_df.columns]
            if radar_metrics:
                fig2 = go.Figure()
                for nm in chosen:
                    sub = comp_df[comp_df[NAME_COL] == nm].head(1)
                    r = [safe_float(sub.iloc[0].get(m, np.nan)) if not np.isnan(safe_float(sub.iloc[0].get(m, np.nan))) else 0 for m in radar_metrics]
                    theta = [m.replace(" (pct)", "") for m in radar_metrics]
                    fig2.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name=nm))
                fig2.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100], gridcolor="rgba(247,247,247,0.10)")),
                    height=560,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig2, use_container_width=True)

            show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] + role_cols if c in comp_df.columns]
            sort_role = "Balanced Score" if "Balanced Score" in comp_df.columns else (role_cols[0] if role_cols else show[-1])
            st.dataframe(comp_df[show].sort_values(sort_role, ascending=False), use_container_width=True, height=460)

# =====================================================
# LEADERBOARDS
# =====================================================
with tabs[3]:
    st.markdown('<div class="kicker">Market</div>', unsafe_allow_html=True)
    if df_f.empty:
        st.info("No players to rank with current filters.")
    else:
        role_cols = [c for c in cfg["role_defs"].keys() if c in df_f.columns]
        if not role_cols:
            st.warning("No role score columns found.")
        else:
            role = st.selectbox("Role", role_cols, index=role_cols.index(default_sort) if default_sort in role_cols else 0, key=f"lb_role_{position}")
            n = st.slider("Rows", 10, 100, 40, 5, key=f"lb_n_{position}")

            cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL, role] if c in df_f.columns]
            out = df_f.sort_values(role, ascending=False).head(n)[cols].copy()
            out[role + " (pct)"] = percentile_rank(df_f[role]).reindex(out.index)
            pro_table(out, pct_cols=[role + " (pct)"], height=560)

# =====================================================
# DISTRIBUTIONS
# =====================================================
with tabs[4]:
    st.markdown('<div class="kicker">Context</div>', unsafe_allow_html=True)
    if df_f.empty:
        st.info("No players with current filters.")
    else:
        numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            default_metric = default_sort if default_sort in numeric_cols else numeric_cols[0]
            metric = st.selectbox("Metric", numeric_cols, index=numeric_cols.index(default_metric), key=f"dist_{position}")

            c1, c2 = st.columns(2, gap="small")
            with c1:
                fig1 = px.histogram(df_f, x=metric, nbins=30)
                fig1.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["white"]))
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.box(df_f, y=metric, points="all")
                fig2.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["white"]))
                st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# SHORTLIST
# =====================================================
with tabs[5]:
    st.markdown('<div class="kicker">Targets</div>', unsafe_allow_html=True)
    items = []
    for k, meta in st.session_state.shortlist.items():
        pos, name = k.split("||", 1)
        items.append({"Position": pos, "Name": name, "Tags": meta.get("tags", ""), "Notes": meta.get("notes", "")})

    if not items:
        st.info("Shortlist is empty. Add players from Search or Profile.")
    else:
        sl_df = pd.DataFrame(items)
        edited = st.data_editor(
            sl_df,
            use_container_width=True,
            height=480,
            num_rows="dynamic",
            column_config={
                "Position": st.column_config.TextColumn(width="small"),
                "Name": st.column_config.TextColumn(width="medium"),
                "Tags": st.column_config.TextColumn(width="medium"),
                "Notes": st.column_config.TextColumn(width="large"),
            },
            key="shortlist_editor",
        )

        new_shortlist = {}
        for _, r in edited.iterrows():
            pos = str(r.get("Position", "")).strip()
            name = str(r.get("Name", "")).strip()
            if not pos or not name:
                continue
            new_shortlist[shortlist_key(pos, name)] = {"tags": str(r.get("Tags", "") or ""), "notes": str(r.get("Notes", "") or "")}
        st.session_state.shortlist = new_shortlist

        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.download_button(
                "Download shortlist (CSV)",
                data=edited.to_csv(index=False).encode("utf-8"),
                file_name="shortlist.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("Clear shortlist", key="clear_shortlist", type="primary"):
                st.session_state.shortlist = {}
                st.rerun()
