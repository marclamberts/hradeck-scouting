# app.py — Scout Lab Pro + IMPECT LIVE (API → in-app dataframe)
# ============================================================
# ✅ Pulls iteration data directly from IMPECT API (no CSV needed)
# ✅ Stores wide KPI table in session_state + caches (TTL)
# ✅ Adds Landing “IMPECT Live (API)” loader
# ✅ Adds Dashboard tab “🌐 IMPECT Live” with KPI browser + radar (percentile-based)
#
# SECURITY:
# - DO NOT hardcode credentials.
# - Put them in .streamlit/secrets.toml:
#     IMPECT_USERNAME="your@email.com"
#     IMPECT_PASSWORD="your-rotated-password"
#
# Run:
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import datetime as dt
import warnings

# NEW: IMPECT live imports
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Scout Lab Pro",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="collapsed",
)

# =====================================================
# POSITION CONFIGURATIONS
# =====================================================
POSITION_CONFIG = {
    "GK": {
        "file": "Goalkeepers.xlsx",
        "title": "Goalkeepers",
        "icon": "🧤",
        "role_cols": ["Ball Playing GK", "Box Defender", "Shot Stopper", "Sweeper Keeper"],
        "key_metrics": [
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "Low pass",
            "Diagonal pass",
            "Chipped/Lofted ball",
            "Goal kick",
            "Free kick",
        ],
        "categories": {
            "Passing": ["Low pass", "Diagonal pass", "Chipped/Lofted ball", "Goal kick", "Free kick"],
            "Shot Stopping": [
                "Prevented Goals Percent (based on post-shot xG) - Long Range Shot saved",
                "Prevented Goals Percent (based on post-shot xG) - Mid Range Shot saved",
                "Prevented Goals Percent (based on post-shot xG) - Close Range Shot saved",
                "Prevented Goals Percent (based on post-shot xG) - Header saved",
            ],
            "Positioning": ["Defensive Touches outside the Box per game", "Caught Balls Percent"],
        },
    },
    "CB": {
        "file": "Central Defenders.xlsx",
        "title": "Central Defenders",
        "icon": "🛡️",
        "role_cols": ["Aerially Dominant CB", "Aggressive CB", "Ball Playing CB", "Strategic CB"],
        "key_metrics": [
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "Low pass",
            "Diagonal pass",
            "Ground duel",
            "Defensive Header",
            "Interception",
        ],
        "categories": {
            "Passing": ["Low pass", "Diagonal pass", "Chipped/Lofted ball"],
            "Defending": ["Ground duel", "Defensive Header", "Interception", "Loose ball regain", "Block", "Clearance with foot"],
            "Attacking": ["Header shot", "Dribble"],
        },
    },
    "LB": {
        "file": "Left Back.xlsx",
        "title": "Left Backs",
        "icon": "⬅️",
        "role_cols": ["Classic Back 4 LB", "Creative LB", "Left Wing-Back"],
        "key_metrics": [
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "Low pass",
            "High Cross",
            "Low Cross",
            "Ground duel",
            "Interception",
        ],
        "categories": {
            "Passing": ["Low pass", "Chipped/Lofted ball"],
            "Crossing": ["High Cross", "Low Cross"],
            "Attacking": ["Dribble", "Availability Between the Lines", "Mid range shot", "Availability in the Box"],
            "Defending": ["Ground duel", "Defensive Header", "Interception", "Loose ball regain"],
        },
    },
    "RB": {
        "file": "Right Back.xlsx",
        "title": "Right Backs",
        "icon": "➡️",
        "role_cols": ["Classic Back 4 RB", "Creative RB", "Right Wing-Back"],
        "key_metrics": [
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "Low pass",
            "High Cross",
            "Low Cross",
            "Ground duel",
            "Interception",
        ],
        "categories": {
            "Passing": ["Low pass", "Chipped/Lofted ball"],
            "Crossing": ["High Cross", "Low Cross"],
            "Attacking": ["Dribble", "Availability Between the Lines", "Mid range shot", "Availability in the Box"],
            "Defending": ["Ground duel", "Defensive Header", "Interception", "Loose ball regain"],
        },
    },
    "DM": {
        "file": "Defensive Midfielder.xlsx",
        "title": "Defensive Midfielders",
        "icon": "⚓",
        "role_cols": ["Anchorman", "Ball Winning Midfielder", "Box-to-Box Midfielder", "Central Creator", "Deep Lying Playmaker"],
        "key_metrics": [
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "Low pass",
            "Diagonal pass",
            "Dribble",
            "Interception",
            "Loose ball regain",
        ],
        "categories": {
            "Passing": ["Low pass", "Diagonal pass", "Chipped/Lofted ball"],
            "Attacking": ["Dribble", "Availability Between the Lines", "Mid range shot", "Availability in the Box"],
            "Defending": ["Ground duel", "Defensive Header", "Interception", "Loose ball regain", "Block"],
        },
    },
    "CM": {
        "file": "Central Midfielder.xlsx",
        "title": "Central Midfielders",
        "icon": "⭐",
        "role_cols": ["Anchorman", "Ball Winning Midfielder", "Box-to-Box Midfielder", "Central Creator", "Deep Lying Playmaker"],
        "key_metrics": [
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "Low pass",
            "Diagonal pass",
            "Dribble",
            "Availability Between the Lines",
            "Mid range shot",
        ],
        "categories": {
            "Passing": ["Low pass", "Diagonal pass", "Chipped/Lofted ball"],
            "Attacking": ["Dribble", "Availability Between the Lines", "Mid range shot", "Availability from Deep Runs", "Availability in the Box"],
            "Defending": ["Ground duel", "Defensive Header", "Interception", "Loose ball regain"],
        },
    },
    "AM": {
        "file": "Attacking Midfielder.xlsx",
        "title": "Attacking Midfielders",
        "icon": "🎯",
        "role_cols": ["Central Creator", "Deep Lying Striker"],
        "key_metrics": [
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "Low pass",
            "Dribble",
            "Availability Between the Lines",
            "Mid range shot",
            "Availability in the Box",
        ],
        "categories": {
            "Passing": ["Low pass", "High Cross", "Low Cross"],
            "Dribbling": ["Dribble"],
            "Movement": ["Availability Between the Lines", "Availability from Deep Runs", "Availability in the Box"],
            "Shooting": ["Mid range shot", "Close range shot", "Header shot", "Offensive header"],
        },
    },
    "LW": {
        "file": "Left Winger.xlsx",
        "title": "Left Wingers",
        "icon": "⚡",
        "role_cols": ["Central Creator", "Classic Left Winger", "Deep Running Left Winger", "Defensive Left Winger", "Left Wing-Back"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low Cross", "Dribble", "Availability in the Box", "Close range shot", "Header shot"],
        "categories": {
            "Crossing": ["High Cross", "Low Cross"],
            "Dribbling": ["Dribble"],
            "Movement": ["Availability Between the Lines", "Availability from Deep Runs", "Availability in the Box"],
            "Shooting": ["Mid range shot", "Close range shot", "Header shot", "Offensive header"],
        },
    },
    "RW": {
        "file": "Right Wing.xlsx",
        "title": "Right Wingers",
        "icon": "⚡",
        "role_cols": ["Central Creator", "Classic Right Winger", "Deep Running Right Winger", "Defensive Right Winger", "Right Wing-Back"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low Cross", "Dribble", "Availability in the Box", "Close range shot", "Header shot"],
        "categories": {
            "Crossing": ["High Cross", "Low Cross"],
            "Dribbling": ["Dribble"],
            "Movement": ["Availability Between the Lines", "Availability from Deep Runs", "Availability in the Box"],
            "Shooting": ["Mid range shot", "Close range shot", "Header shot", "Offensive header"],
        },
    },
    "ST": {
        "file": "Strikers.xlsx",
        "title": "Strikers",
        "icon": "⚽",
        "role_cols": ["Complete Forward", "Deep Lying Striker", "Deep Running Striker", "Poacher", "Pressing Striker", "Second Striker", "Target Man"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Dribble", "Availability in the Box", "Close range shot", "Header shot", "Hold-Up play"],
        "categories": {
            "Movement": ["Availability Between the Lines", "Availability from Deep Runs", "Availability in the Box"],
            "Shooting": ["Mid range shot", "Close range shot", "Header shot", "Offensive header"],
            "Physical": ["Hold-Up play", "Ground duel"],
        },
    },
}

NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# =====================================================
# CSS (your original styling)
# =====================================================
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --info: #06b6d4;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --bg-hover: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
    }

    .stApp { background: var(--bg-dark); color: var(--text-primary); }

    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    [data-testid="stMetric"] {
        background: var(--bg-dark);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 700;
    }

    .top-nav {
        background: var(--bg-card);
        border-bottom: 2px solid var(--primary);
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .nav-brand {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-primary);
    }

    .landing-hero {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #3b82f6 100%);
        padding: 4rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(37, 99, 235, 0.3);
    }

    .landing-title {
        font-size: 4rem;
        font-weight: 900;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .landing-subtitle {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .landing-tagline {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
        max-width: 800px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }

    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        border-color: var(--primary);
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
    }

    .feature-icon { font-size: 3rem; margin-bottom: 1rem; }
    .feature-title { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.75rem; }
    .feature-desc { font-size: 0.95rem; color: var(--text-secondary); line-height: 1.5; }

    .stats-showcase {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
    }

    .dashboard-header {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }

    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .chart-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--text-primary);
    }

    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.5rem;
        font-weight: 700;
        transition: all 0.2s ease;
        width: 100%;
        margin-bottom: 1rem;
    }

    .stButton > button:hover {
        background: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }

    button[kind="primary"] { background: var(--primary) !important; }
    button[kind="secondary"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; }

    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiselect > div > div > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
    }

    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border);
    }

    .results-count {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-success { background: var(--success); color: white; }
    .badge-warning { background: var(--warning); color: white; }
    .badge-danger  { background: var(--danger);  color: white; }
    .badge-info    { background: var(--info);    color: white; }

    .empty-state { text-align: center; padding: 4rem 2rem; color: var(--text-secondary); }

    hr { border-color: var(--border); margin: 1rem 0; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: var(--bg-card); padding: 0.5rem; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; color: var(--text-secondary); font-weight: 600; }
    .stTabs [aria-selected="true"] { background: var(--primary); color: white; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def safe_float(x):
    if x is None or pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("%", "").replace(",", "")
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def safe_fmt(x, decimals=1):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"


def percentile_rank_series(s: pd.Series) -> pd.Series:
    s = s.apply(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    if mask.sum() > 0:
        out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out


def get_percentile_badge(pct):
    if np.isnan(pct):
        return ""
    if pct >= 90:
        return '<span class="badge badge-success">Top 10%</span>'
    elif pct >= 75:
        return '<span class="badge badge-info">Top 25%</span>'
    elif pct >= 50:
        return '<span class="badge badge-warning">Above Avg</span>'
    else:
        return '<span class="badge badge-danger">Below Avg</span>'


# =====================================================
# DATA LOADING (EXCEL)
# =====================================================
@st.cache_data(show_spinner=False)
def load_data(position_key):
    cfg = POSITION_CONFIG[position_key]

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
        st.error(f"❌ File not found: `{cfg['file']}`")
        st.stop()

    df = pd.read_excel(fp)
    df.columns = [str(c).strip() for c in df.columns]

    numeric_cols = []
    for col in df.columns:
        if col in ["Player-ID", NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            continue
        if "BetterThan" in col:
            continue
        numeric_cols.append(col)

    for c in numeric_cols + [AGE_COL, SHARE_COL]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace("nan", "").str.strip()

    for m in numeric_cols:
        if m in df.columns:
            df[m + " (pct)"] = percentile_rank_series(df[m])

    return df, cfg, numeric_cols


# =====================================================
# IMPECT LIVE (API) — in-app wide dataframe
# =====================================================
TOKEN_URL = "https://login.impect.com/auth/realms/production/protocol/openid-connect/token"
BASE_API_URL = "https://api.impect.com"

TIMEOUT_SECONDS = 30
MAX_RETRIES = 8
BACKOFF_BASE = 0.75
BACKOFF_CAP = 30.0


def unwrap_data(obj):
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj


def get_access_token(session: requests.Session, username: str, password: str) -> str:
    payload = {
        "client_id": "api",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = session.post(TOKEN_URL, data=payload, headers=headers, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()["access_token"]


def get_with_retry(session: requests.Session, url: str, token: str):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    for attempt in range(MAX_RETRIES + 1):
        r = session.get(url, headers=headers, timeout=TIMEOUT_SECONDS)

        if r.status_code < 400:
            return r.json()

        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt))
            wait += random.uniform(0, 0.35 * wait)
            if attempt >= MAX_RETRIES:
                r.raise_for_status()
            time.sleep(wait)
            continue

        r.raise_for_status()

    raise RuntimeError("Retry loop exited unexpectedly")


def fetch_kpi_defs(session, token, language="en"):
    url = f"{BASE_API_URL}/v5/customerapi/kpis?language={language}"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_iteration_players(session, iteration_id, token):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/players"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_iteration_squads(session, iteration_id, token):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/squads"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_player_kpis_for_squad(session, iteration_id, squad_id, token):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/squads/{squad_id}/player-kpis"
    return unwrap_data(get_with_retry(session, url, token))


def extract_long_rows(iteration_id, squad_id, squad_name, items):
    out = []
    for item in items or []:
        matches = item.get("matches") or 0
        player_obj = item.get("player") if isinstance(item, dict) else None
        p = player_obj if isinstance(player_obj, dict) else item
        player_id = (p or {}).get("id") or item.get("playerId")

        for kv in (item.get("kpis") or []):
            out.append(
                {
                    "iterationId": iteration_id,
                    "squadId": squad_id,
                    "squadName": squad_name,
                    "playerId": player_id,
                    "matches": matches,
                    "kpiId": kv.get("kpiId"),
                    "value_raw": kv.get("value"),
                }
            )
    return out


@st.cache_data(show_spinner=False, ttl=60 * 30)  # cache 30 minutes
def load_impect_iteration_wide(username: str, password: str, iteration_id: int, language: str, max_workers: int, base_sleep: float):
    session = requests.Session()
    token = get_access_token(session, username, password)

    kpi_defs = fetch_kpi_defs(session, token, language)
    kpi_df = pd.json_normalize(kpi_defs if isinstance(kpi_defs, list) else [])
    label_col = "details.label" if "details.label" in kpi_df.columns else "name"

    kpi_lookup = (
        kpi_df.rename(columns={"id": "kpiId", label_col: "kpiLabel", "name": "kpiTechnicalName"})[
            ["kpiId", "kpiLabel", "kpiTechnicalName"]
        ]
        .drop_duplicates()
    )
    kpi_lookup["kpiId"] = pd.to_numeric(kpi_lookup["kpiId"], errors="coerce")

    rename_map = dict(
        zip(
            kpi_lookup["kpiId"],
            (
                kpi_lookup["kpiLabel"]
                .fillna(kpi_lookup["kpiTechnicalName"])
                .fillna("kpi")
                + " (" + kpi_lookup["kpiId"].astype("Int64").astype(str) + ")"
            ),
        )
    )

    players = fetch_iteration_players(session, iteration_id, token)
    players_df = pd.json_normalize(players if isinstance(players, list) else [])
    if "id" in players_df.columns:
        players_df = players_df.rename(columns={"id": "playerId"})
    players_df["playerId"] = pd.to_numeric(players_df["playerId"], errors="coerce")

    squads = fetch_iteration_squads(session, iteration_id, token)
    squads_lookup = [{"squadId": s.get("id"), "squadName": s.get("name")} for s in (squads or [])]

    long_rows = []

    def worker(squad):
        sid = squad["squadId"]
        sname = squad["squadName"]
        if sid is None:
            return []
        if base_sleep:
            time.sleep(base_sleep + random.uniform(0, base_sleep))
        items = fetch_player_kpis_for_squad(session, iteration_id, int(sid), token)
        if not isinstance(items, list):
            return []
        return extract_long_rows(iteration_id, sid, sname, items)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, s) for s in squads_lookup]
        for fut in as_completed(futures):
            long_rows.extend(fut.result())

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        raise RuntimeError("No KPI rows returned from IMPECT.")

    # decimal comma -> dot
    s = (
        long_df["value_raw"]
        .astype("string")
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    long_df["value"] = pd.to_numeric(s, errors="coerce")

    long_df["playerId"] = pd.to_numeric(long_df["playerId"], errors="coerce")
    long_df["kpiId"] = pd.to_numeric(long_df["kpiId"], errors="coerce")
    long_df["matches"] = pd.to_numeric(long_df["matches"], errors="coerce").fillna(0)

    long_df = long_df.dropna(subset=["playerId", "kpiId"])

    # ✅ CRITICAL FIX:
    # Aggregate duplicates so each index+kpiId is unique before pivoting.
    key_cols = ["iterationId", "squadId", "squadName", "playerId", "matches", "kpiId"]
    long_df = (
        long_df.groupby(key_cols, as_index=False)["value"]
        .mean()   # or .first() if you prefer
    )
    
    wide = (
        long_df.pivot_table(
            index=["iterationId", "squadId", "squadName", "playerId", "matches"],
            columns="kpiId",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None

    wide.columns.name = None

    wide = wide.rename(columns={k: v for k, v in rename_map.items() if k in wide.columns})
    wide = wide.merge(players_df, how="left", on="playerId")

    return wide, players_df, kpi_lookup


def norm_name(x):
    return str(x or "").strip().lower()


def find_impect_player(im_df: pd.DataFrame, name_query: str) -> pd.DataFrame:
    if im_df is None or im_df.empty:
        return im_df.iloc[0:0]
    q = norm_name(name_query)
    for c in ["commonname", "firstname", "lastname"]:
        if c not in im_df.columns:
            im_df[c] = ""

    cn = im_df["commonname"].map(norm_name)
    fn = im_df["firstname"].map(norm_name)
    ln = im_df["lastname"].map(norm_name)
    full = (fn + " " + ln).str.strip()

    m = (cn == q) | (full == q)
    if m.any():
        return im_df[m]

    m2 = cn.str.contains(q, na=False) | fn.str.contains(q, na=False) | ln.str.contains(q, na=False) | full.str.contains(q, na=False)
    return im_df[m2]


def compute_percentile(series: pd.Series, value: float, invert: bool = False) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return np.nan
    rank = (s < value).sum() + 0.5 * (s == value).sum()
    pct = rank / len(s) * 100
    return 100 - pct if invert else pct


# =====================================================
# STATE
# =====================================================
def init_state():
    if "view" not in st.session_state:
        st.session_state.view = "landing"
    if "selected_player" not in st.session_state:
        st.session_state.selected_player = None
    if "position" not in st.session_state:
        st.session_state.position = "ST"
    if "comparison_list" not in st.session_state:
        st.session_state.comparison_list = []
    if "impect_loaded" not in st.session_state:
        st.session_state.impect_loaded = False
    if "impect_wide" not in st.session_state:
        st.session_state.impect_wide = None
    if "impect_kpis" not in st.session_state:
        st.session_state.impect_kpis = None
    if "impect_iteration_id" not in st.session_state:
        st.session_state.impect_iteration_id = 1421
    if "impect_language" not in st.session_state:
        st.session_state.impect_language = "en"

init_state()

# =====================================================
# NAVIGATION
# =====================================================
def render_nav(show_back=False):
    if show_back:
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Home", use_container_width=True):
                st.session_state.view = "landing"
                st.session_state.selected_player = None
                st.rerun()
        with col2:
            cfg = POSITION_CONFIG.get(st.session_state.position, {})
            st.markdown(
                f"""
            <div class="top-nav">
                <div class="nav-brand">⚽ Scout Lab Pro</div>
                <div style="display:flex; align-items:center; gap:1rem;">
                    <span style="font-size:1.5rem;">{cfg.get('icon','⚽')}</span>
                    <span style="font-weight:600; font-size:1.1rem;">{cfg.get('title','Scouting')}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
        <div class="top-nav">
            <div class="nav-brand">⚽ Scout Lab Pro</div>
            <div style="font-weight:600; font-size:1.1rem;">Professional Football Analytics</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# =====================================================
# LANDING VIEW
# =====================================================
def render_landing_view():
    st.markdown(
        """
    <div class="landing-hero">
        <div class="landing-title">⚽ Scout Lab Pro</div>
        <div class="landing-subtitle">Advanced Football Analytics Platform</div>
        <div class="landing-tagline">
            Comprehensive player scouting with IMPECT data, detailed performance metrics,
            role suitability analysis, and professional visualization tools across 10 positions.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🎯 Platform Features")
    st.markdown(
        """
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">IMPECT Analytics</div>
            <div class="feature-desc">Industry-leading performance metrics with offensive and defensive breakdowns</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎭</div>
            <div class="feature-title">Role Suitability</div>
            <div class="feature-desc">Analyze player fit for multiple tactical roles with detailed scoring</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Performance Trends</div>
            <div class="feature-desc">Distribution analysis and percentile rankings across all metrics</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚖️</div>
            <div class="feature-title">Player Comparison</div>
            <div class="feature-desc">Side-by-side comparison of up to 6 players with radar visualizations</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Advanced Filters</div>
            <div class="feature-desc">Multi-parameter search by age, competition, team, nationality, and metrics</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Scouting Reports</div>
            <div class="feature-desc">Automated report generation with strengths and development areas</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # IMPECT LIVE LOADER
    st.markdown("---")
    st.markdown("## 🌐 IMPECT Live (API)")

    with st.expander("Load iteration into app (no CSV)", expanded=False):
        st.caption("Credentials are read from `.streamlit/secrets.toml` (server-side).")

        st.session_state.impect_iteration_id = st.number_input(
            "Iteration ID",
            min_value=1,
            value=int(st.session_state.impect_iteration_id),
            step=1,
        )
        st.session_state.impect_language = st.selectbox(
            "KPI language",
            options=["en", "nl", "de", "fr"],
            index=["en", "nl", "de", "fr"].index(st.session_state.impect_language) if st.session_state.impect_language in ["en","nl","de","fr"] else 0,
        )
        max_workers = st.slider("MAX_WORKERS (avoid 429)", 1, 6, 3)
        base_sleep = st.slider("Base sleep (throttle)", 0.0, 1.0, 0.2, 0.05)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Fetch from IMPECT API", use_container_width=True):
                username = st.secrets.get("IMPECT_USERNAME", "")
                password = st.secrets.get("IMPECT_PASSWORD", "")
                if not username or not password:
                    st.error("Missing IMPECT_USERNAME/IMPECT_PASSWORD in .streamlit/secrets.toml")
                else:
                    try:
                        with st.spinner("Fetching IMPECT iteration..."):
                            wide, players_df, kpi_lookup = load_impect_iteration_wide(
                                username=username,
                                password=password,
                                iteration_id=int(st.session_state.impect_iteration_id),
                                language=st.session_state.impect_language,
                                max_workers=int(max_workers),
                                base_sleep=float(base_sleep),
                            )
                        st.session_state.impect_wide = wide
                        st.session_state.impect_kpis = kpi_lookup
                        st.session_state.impect_loaded = True
                        st.success(f"Loaded IMPECT: {wide.shape[0]} rows × {wide.shape[1]} columns")
                    except Exception as e:
                        st.error(f"IMPECT load failed: {e}")

        with c2:
            if st.button("Clear IMPECT cache + state", use_container_width=True, type="secondary"):
                st.session_state.impect_loaded = False
                st.session_state.impect_wide = None
                st.session_state.impect_kpis = None
                load_impect_iteration_wide.clear()
                st.success("Cleared IMPECT cached data.")

        if st.session_state.impect_loaded and isinstance(st.session_state.impect_wide, pd.DataFrame):
            st.caption("Preview (first 10 rows):")
            st.dataframe(st.session_state.impect_wide.head(10), use_container_width=True)

    # Position selection
    st.markdown("---")
    st.markdown("## 🎯 Select Position to Begin Scouting")
    st.markdown("### Choose a position to access player database and analytics")

    position_groups = {
        "Goalkeepers": ["GK"],
        "Defenders": ["CB", "LB", "RB"],
        "Midfielders": ["DM", "CM", "AM"],
        "Forwards": ["LW", "RW", "ST"],
    }

    for group_name, positions in position_groups.items():
        st.markdown(f"### {group_name}")
        cols = st.columns(len(positions))
        for idx, pos_key in enumerate(positions):
            cfg = POSITION_CONFIG[pos_key]
            with cols[idx]:
                if st.button(
                    f"{cfg['icon']}\n\n**{cfg['title']}**\n\nView Database",
                    key=f"landing_pos_{pos_key}",
                    use_container_width=True,
                    help=f"Scout {cfg['title'].lower()}",
                ):
                    st.session_state.position = pos_key
                    st.session_state.view = "search"
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Database Coverage")
    st.markdown(
        """
    <div class="stats-showcase">
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px,1fr));
                    gap:2rem; text-align:center;">
            <div><div class="stat-value">10</div><div class="stat-label">Positions Covered</div></div>
            <div><div class="stat-value">100+</div><div class="stat-label">Metrics per Position</div></div>
            <div><div class="stat-value">Multiple</div><div class="stat-label">Role Profiles</div></div>
            <div><div class="stat-value">Global</div><div class="stat-label">Competition Coverage</div></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center; color: var(--text-secondary); padding:2rem 0;">
        <p style="margin-bottom:0.5rem;">Built with advanced analytics and IMPECT data</p>
        <p style="font-size:0.875rem;">Professional football scouting and analysis platform</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# =====================================================
# SEARCH VIEW
# =====================================================
def render_search_view(df, cfg, all_metrics):
    st.markdown("### Select Position")
    cols = st.columns(5)

    positions = list(POSITION_CONFIG.keys())
    for i, pos_key in enumerate(positions):
        pos_cfg = POSITION_CONFIG[pos_key]
        col_idx = i % 5
        with cols[col_idx]:
            if st.button(
                f"{pos_cfg['icon']} {pos_cfg['title']}",
                key=f"pos_{pos_key}",
                use_container_width=True,
                type="primary" if pos_key == st.session_state.position else "secondary",
            ):
                st.session_state.position = pos_key
                st.rerun()

    st.markdown("---")

    st.markdown("### Search Filters")

    with st.container():
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            search = st.text_input("🔍 Search Player", placeholder="Name, team, league...")

        with fc2:
            min_age, max_age = 15, 45
            if AGE_COL in df.columns:
                vals = df[AGE_COL].dropna()
                if len(vals):
                    min_age = int(max(15, np.floor(vals.min())))
                    max_age = int(min(45, np.ceil(vals.max())))
            age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))

        with fc3:
            min_share = st.slider("Min Match Share %", 0.0, 50.0, 0.0, 1.0)

        with fc4:
            if COMP_COL in df.columns:
                comps = sorted([c for c in df[COMP_COL].dropna().unique() if str(c).strip()])
                selected_comps = st.multiselect("Competition", comps, key="comp_filter")
            else:
                selected_comps = []

    with st.expander("🎯 Advanced Filters", expanded=False):
        afc1, afc2, afc3 = st.columns(3)

        with afc1:
            if TEAM_COL in df.columns:
                teams = sorted([t for t in df[TEAM_COL].dropna().unique() if str(t).strip()])
                selected_teams = st.multiselect("Teams", teams, key="team_filter")
            else:
                selected_teams = []

        with afc2:
            if NAT_COL in df.columns:
                nats = sorted([n for n in df[NAT_COL].dropna().unique() if str(n).strip()])
                selected_nats = st.multiselect("Nationalities", nats, key="nat_filter")
            else:
                selected_nats = []

        with afc3:
            if "IMPECT" in df.columns:
                min_impect = st.number_input("Min IMPECT", min_value=0.0, value=0.0, step=0.1)
            else:
                min_impect = 0.0

    df_filtered = df.copy()

    if search:
        mask = pd.Series(False, index=df_filtered.index)
        for col in [NAME_COL, TEAM_COL, COMP_COL]:
            if col in df_filtered.columns:
                mask = mask | df_filtered[col].astype(str).str.lower().str.contains(search.lower(), na=False, regex=False)
        df_filtered = df_filtered[mask]

    if AGE_COL in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered[AGE_COL].fillna(age_range[0]) >= age_range[0]) & (df_filtered[AGE_COL].fillna(age_range[1]) <= age_range[1])
        ]

    if SHARE_COL in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[SHARE_COL].fillna(0) >= min_share]

    if selected_comps and COMP_COL in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[COMP_COL].isin(selected_comps)]

    if selected_teams and TEAM_COL in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[TEAM_COL].isin(selected_teams)]

    if selected_nats and NAT_COL in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[NAT_COL].isin(selected_nats)]

    if "IMPECT" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["IMPECT"].fillna(0) >= min_impect]

    sort_col = st.selectbox("Sort by", ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Age", "Match Share"], key="sort")
    if sort_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(sort_col, ascending=(sort_col == "Age"))

    st.markdown("---")

    st.markdown(
        f"""
    <div class="results-header">
        <div class="results-count">{len(df_filtered)} Players Found</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if df_filtered.empty:
        st.markdown(
            """
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <h2>No players found</h2>
            <p>Try adjusting your filters</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        for _, row in df_filtered.head(50).iterrows():
            render_player_card(row, cfg)

def render_player_card(row, cfg):
    name = str(row.get(NAME_COL, "—"))
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    age = safe_fmt(row.get(AGE_COL, 0), 0)
    nat = str(row.get(NAT_COL, "—"))
    share = safe_fmt(row.get(SHARE_COL, 0), 1)

    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"### {name}")
            st.markdown(f"🏟️ {team} • 🏆 {comp} • 🌍 {nat} • 👤 {age} years • ⏱️ {share}% share")

        with col2:
            impect_val = safe_fmt(row.get("IMPECT", 0), 2)
            impect_pct = safe_float(row.get("IMPECT (pct)", 0))
            if np.isnan(impect_pct):
                st.metric("IMPECT", impect_val)
            else:
                st.metric("IMPECT", impect_val, delta=f"{impect_pct:.0f}th %ile")

        metric_cols = st.columns(6)
        display_count = 0
        for metric in cfg.get("key_metrics", []):
            if metric in row and metric + " (pct)" in row and display_count < 6:
                val = safe_fmt(row.get(metric, 0), 1)
                pct = safe_float(row.get(metric + " (pct)", 0))
                with metric_cols[display_count % 6]:
                    if np.isnan(pct):
                        st.metric(label=metric[:20], value=val)
                    else:
                        st.metric(label=metric[:20], value=val, delta=f"{pct:.0f}th")
                display_count += 1

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("👁️ View Dashboard", key=f"view_{name.replace(' ', '_').replace('.', '_')}", use_container_width=True):
                st.session_state.selected_player = name
                st.session_state.view = "dashboard"
                st.rerun()

        with bc2:
            in_comparison = name in st.session_state.comparison_list
            btn_text = "✓ In Comparison" if in_comparison else "➕ Add to Compare"
            if st.button(btn_text, key=f"comp_{name.replace(' ', '_').replace('.', '_')}", use_container_width=True):
                if in_comparison:
                    st.session_state.comparison_list.remove(name)
                else:
                    if len(st.session_state.comparison_list) < 6:
                        st.session_state.comparison_list.append(name)
                    else:
                        st.warning("Maximum 6 players for comparison")
                st.rerun()

        st.markdown("---")

# =====================================================
# DASHBOARD VIEW
# =====================================================
def render_dashboard_view(df, cfg, all_metrics):
    player_name = st.session_state.selected_player

    if st.button("← Back to Search", key="back_btn"):
        st.session_state.view = "search"
        st.rerun()

    player_data = df[df[NAME_COL] == player_name]
    if player_data.empty:
        st.error("Player not found")
        return

    row = player_data.iloc[0]

    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    age = safe_fmt(row.get(AGE_COL, 0), 0)
    nat = str(row.get(NAT_COL, "—"))
    share = safe_fmt(row.get(SHARE_COL, 0), 1)

    st.markdown(
        f"""
    <div class="dashboard-header">
        <div class="dashboard-title">{player_name}</div>
        <div class="dashboard-subtitle" style="font-size: 1.25rem; opacity: 0.95;">
            {team} • {comp} • {nat} • {age} years old • {share}% match share
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["📊 Overview", "🎯 Detailed Stats", "📈 Performance Trends", "⚖️ Comparison", "📝 Report", "🌐 IMPECT Live"])

    with tabs[0]:
        render_overview_tab(row, cfg, df)

    with tabs[1]:
        render_detailed_stats_tab(row, cfg, all_metrics)

    with tabs[2]:
        render_performance_tab(row, cfg, df)

    with tabs[3]:
        render_comparison_tab(df, cfg, all_metrics)

    with tabs[4]:
        render_report_tab(row, cfg, all_metrics)

    # NEW: IMPECT LIVE TAB
    with tabs[5]:
        st.markdown("### 🌐 IMPECT Live (API) — KPI browser + radar")

        if not st.session_state.impect_loaded or not isinstance(st.session_state.impect_wide, pd.DataFrame):
            st.info("IMPECT is not loaded yet. Go to Home → IMPECT Live (API) and click **Fetch from IMPECT API**.")
            return

        im_df = st.session_state.impect_wide.copy()

        # Find player by name (Excel name -> IMPECT commonname/firstname+lastname)
        matches = find_impect_player(im_df, player_name)

        if matches.empty:
            st.error(f"Couldn’t match this player in IMPECT iteration data: {player_name!r}")
            st.caption("Tip: IMPECT uses `commonname` or firstname+lastname. Try searching below.")
            q = st.text_input("Search IMPECT player name", value=player_name)
            matches2 = find_impect_player(im_df, q)
            if not matches2.empty:
                matches = matches2
            else:
                st.caption("Sample IMPECT commonname values:")
                if "commonname" in im_df.columns:
                    st.write(sorted(im_df["commonname"].dropna().astype(str).unique())[:40])
                return

        # If multiple rows (e.g., multiple squads/matches), pick highest matches
        if "matches" in matches.columns:
            matches = matches.sort_values("matches", ascending=False)

        player_im = matches.iloc[0]

        # Show identity
        id_cols = ["commonname", "firstname", "lastname", "squadName", "positions", "matches", "playerId", "iterationId"]
        present = [c for c in id_cols if c in matches.columns]
        st.dataframe(matches[present].head(10), use_container_width=True)

        # Choose KPI columns (numeric) for radar/table
        meta_cols = set(["iterationId", "squadId", "squadName", "playerId", "matches", "commonname", "firstname", "lastname", "positions"])
        numeric_cols = [c for c in im_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(im_df[c])]

        if not numeric_cols:
            st.warning("No numeric KPI columns found in IMPECT dataframe.")
            return

        # Default KPI pick: try to prefer common/shorter columns
        default_kpis = numeric_cols[:12]

        chosen = st.multiselect(
            "Pick KPIs for radar (6–12 recommended)",
            options=numeric_cols,
            default=default_kpis[:10],
        )

        if len(chosen) < 3:
            st.info("Select at least 3 KPIs to draw a radar.")
            return

        # Compute percentiles across the iteration pool for selected KPIs
        pct_vals = []
        labels = []
        raw_vals = []

        # Optional: invert "bad" metrics if they contain keywords
        invert_keywords = ["loss", "lost", "conceded", "error", "own goal", "foul", "yellow", "red"]
        for c in chosen:
            v = safe_float(player_im.get(c, np.nan))
            inv = any(k in str(c).lower() for k in invert_keywords)
            pct = compute_percentile(im_df[c], v, invert=inv)
            pct_vals.append(0 if np.isnan(pct) else float(pct))
            raw_vals.append(v)
            labels.append(c[:28] + ("…" if len(c) > 28 else ""))

        # Radar (percentiles)
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=pct_vals,
                theta=labels,
                fill="toself",
                name="Percentile",
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100], showgrid=True)),
            showlegend=False,
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            margin=dict(t=30, b=20, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        out_rows = []
        for c, lab, v, p in zip(chosen, labels, raw_vals, pct_vals):
            out_rows.append({"KPI": c, "Value": v, "Percentile": p})

        out_df = pd.DataFrame(out_rows).sort_values("Percentile", ascending=False)
        st.dataframe(out_df, use_container_width=True)

# =====================================================
# OVERVIEW TAB
# =====================================================
def render_overview_tab(row, cfg, df):
    impect = safe_fmt(row.get("IMPECT", 0), 2)
    off_impect = safe_fmt(row.get("Offensive IMPECT", 0), 2)
    def_impect = safe_fmt(row.get("Defensive IMPECT", 0), 2)

    impect_pct = safe_float(row.get("IMPECT (pct)", 0))
    off_pct = safe_float(row.get("Offensive IMPECT (pct)", 0))
    def_pct = safe_float(row.get("Defensive IMPECT (pct)", 0))

    st.markdown("### 🏆 Key Performance Indicators")

    cols = st.columns(5)
    stats = [
        ("IMPECT", impect, impect_pct, cols[0]),
        ("Offensive", off_impect, off_pct, cols[1]),
        ("Defensive", def_impect, def_pct, cols[2]),
        ("Age", safe_fmt(row.get(AGE_COL, 0), 0), None, cols[3]),
        ("Share", f"{safe_fmt(row.get(SHARE_COL, 0), 1)}%", None, cols[4]),
    ]

    for label, val, pct, col in stats:
        with col:
            st.markdown(
                f"""
            <div class="stat-card">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
                {f'<div style="margin-top:0.5rem;">{get_percentile_badge(pct)}</div>' if pct is not None and not np.isnan(pct) else ''}
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📊 Performance Radar</div>', unsafe_allow_html=True)

        metrics = cfg.get("key_metrics", [])
        values, labels = [], []
        for m in metrics:
            pct_col = m + " (pct)"
            if pct_col in row:
                pct = safe_float(row.get(pct_col, np.nan))
                if not np.isnan(pct):
                    values.append(pct)
                    labels.append(m[:25])

        if values:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself"))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100], showgrid=True)),
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", size=12),
                showlegend=False,
                margin=dict(t=20, b=20, l=40, r=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Metric Breakdown</div>', unsafe_allow_html=True)

        metric_data = []
        for m in metrics[:8]:
            if m in row:
                val = safe_float(row.get(m, np.nan))
                pct = safe_float(row.get(m + " (pct)", np.nan))
                if not np.isnan(val):
                    metric_data.append({"Metric": m[:25], "Value": val, "Percentile": pct})

        if metric_data:
            df_metrics = pd.DataFrame(metric_data)
            fig = px.bar(df_metrics, x="Value", y="Metric", color="Percentile", orientation="h", range_color=[0, 100], text="Percentile")
            fig.update_traces(texttemplate="%{text:.0f}th", textposition="outside")
            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                yaxis=dict(categoryorder="total ascending"),
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    role_cols = cfg.get("role_cols", [])
    if role_cols:
        st.markdown("---")
        st.markdown("### 🎭 Role Suitability Analysis")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        role_data = []
        for rc in role_cols:
            if rc in row:
                val = safe_float(row.get(rc, np.nan))
                if not np.isnan(val):
                    role_data.append({"Role": rc[:40], "Score": val})

        if role_data:
            df_roles = pd.DataFrame(role_data).sort_values("Score", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df_roles["Role"], x=df_roles["Score"], orientation="h", text=df_roles["Score"].apply(lambda x: f"{x:.1f}"), textposition="outside"))
            fig.update_layout(
                height=max(300, len(role_data) * 50),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", size=12),
                margin=dict(t=20, b=20, l=10, r=10),
                xaxis=dict(range=[0, 110], title="Suitability Score"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# DETAILED STATS TAB
# =====================================================
def render_detailed_stats_tab(row, cfg, all_metrics):
    st.markdown("### 📋 Complete Statistics Breakdown")

    categories = cfg.get("categories", {})
    role_cols = cfg.get("role_cols", [])

    if categories:
        for category, metrics in categories.items():
            st.markdown(f"#### {category}")

            cat_data = []
            for m in metrics:
                if m in row and m + " (pct)" in row:
                    val = safe_float(row.get(m, np.nan))
                    pct = safe_float(row.get(m + " (pct)", np.nan))
                    if not np.isnan(val):
                        cat_data.append({"Metric": m, "Value": round(val, 2), "Percentile": round(pct, 0) if not np.isnan(pct) else np.nan})

            if cat_data:
                df_cat = pd.DataFrame(cat_data)
                st.dataframe(df_cat, use_container_width=True, height=min(400, len(cat_data) * 50 + 100))

            st.markdown("---")

    st.markdown("#### All Metrics")

    all_stats = []
    for col in all_metrics:
        if col in role_cols:
            continue
        if col in row and col + " (pct)" in row:
            val = safe_float(row.get(col, np.nan))
            pct = safe_float(row.get(col + " (pct)", np.nan))
            if not np.isnan(val):
                all_stats.append({"Metric": col, "Value": round(val, 2), "Percentile": round(pct, 0) if not np.isnan(pct) else np.nan})

    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        st.dataframe(df_stats, use_container_width=True, height=500)

# =====================================================
# PERFORMANCE TAB
# =====================================================
def render_performance_tab(row, cfg, df):
    st.markdown("### 📈 Performance Distribution Analysis")
    key_metrics = cfg.get("key_metrics", [])
    col1, col2 = st.columns(2)

    for idx, metric in enumerate(key_metrics[:6]):
        if metric not in df.columns:
            continue
        col = col1 if idx % 2 == 0 else col2
        with col:
            st.markdown(f"#### {metric}")

            player_val = safe_float(row.get(metric, np.nan))
            player_pct = safe_float(row.get(metric + " (pct)", np.nan))
            metric_vals = df[metric].dropna()
            if len(metric_vals) > 0 and not np.isnan(player_val):
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=metric_vals, nbinsx=30, name="Distribution"))
                fig.add_vline(x=player_val, line_dash="dash", line_width=3, annotation_text=f"Player: {player_val:.1f} ({player_pct:.0f}th)" if not np.isnan(player_pct) else f"Player: {player_val:.1f}")
                fig.add_vline(x=metric_vals.mean(), line_dash="dot", line_width=2, annotation_text=f"Avg: {metric_vals.mean():.1f}")
                fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# =====================================================
# COMPARISON TAB
# =====================================================
def render_comparison_tab(df, cfg, all_metrics):
    st.markdown("### ⚖️ Player Comparison")

    if not st.session_state.comparison_list:
        st.info("Add players from the search page to compare them here")
        return

    st.markdown(f"**{len(st.session_state.comparison_list)} players in comparison**")

    comp_df = df[df[NAME_COL].isin(st.session_state.comparison_list)]
    if comp_df.empty:
        st.warning("No valid players in comparison list")
        return

    st.markdown("#### Quick Stats")
    comp_data = []
    for _, row in comp_df.iterrows():
        comp_data.append(
            {
                "Player": row[NAME_COL],
                "Team": row.get(TEAM_COL, "—"),
                "Age": safe_fmt(row.get(AGE_COL, 0), 0),
                "IMPECT": safe_fmt(row.get("IMPECT", 0), 2),
                "Off. IMPECT": safe_fmt(row.get("Offensive IMPECT", 0), 2),
                "Def. IMPECT": safe_fmt(row.get("Defensive IMPECT", 0), 2),
            }
        )
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    st.markdown("#### Performance Comparison")
    metrics = cfg.get("key_metrics", [])[:8]
    fig = go.Figure()

    for player_name in st.session_state.comparison_list:
        player_row = comp_df[comp_df[NAME_COL] == player_name].iloc[0]
        values, labels = [], []
        for m in metrics:
            pct_col = m + " (pct)"
            if pct_col in player_row:
                pct = safe_float(player_row.get(pct_col, np.nan))
                if not np.isnan(pct):
                    values.append(pct)
                    labels.append(m[:20])
        if values:
            fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself", name=player_name))

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], showgrid=True)),
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🗑️ Clear Comparison List", type="secondary"):
        st.session_state.comparison_list = []
        st.rerun()

# =====================================================
# REPORT TAB
# =====================================================
def render_report_tab(row, cfg, all_metrics):
    st.markdown("### 📝 Scouting Report")

    player_name = row[NAME_COL]
    team = row.get(TEAM_COL, "—")
    age = safe_fmt(row.get(AGE_COL, 0), 0)

    report = f"""
## Player Profile: {player_name}

**Club:** {team}  
**Age:** {age} years  
**Position:** {cfg['title']}  
**Competition:** {row.get(COMP_COL, "—")}  
**Nationality:** {row.get(NAT_COL, "—")}  

---

### Overall Assessment

**IMPECT Score:** {safe_fmt(row.get('IMPECT', 0), 2)} ({safe_fmt(row.get('IMPECT (pct)', 0), 0)}th percentile)  
**Offensive Contribution:** {safe_fmt(row.get('Offensive IMPECT', 0), 2)} ({safe_fmt(row.get('Offensive IMPECT (pct)', 0), 0)}th percentile)  
**Defensive Contribution:** {safe_fmt(row.get('Defensive IMPECT', 0), 2)} ({safe_fmt(row.get('Defensive IMPECT (pct)', 0), 0)}th percentile)  

---

### Key Strengths
"""

    strengths = []
    for m in all_metrics:
        pct_col = m + " (pct)"
        if pct_col in row:
            pct = safe_float(row.get(pct_col, np.nan))
            if not np.isnan(pct) and pct >= 75:
                strengths.append((m, pct))
    strengths.sort(key=lambda x: x[1], reverse=True)

    for metric, pct in strengths[:5]:
        report += f"- **{metric}**: {safe_fmt(row.get(metric, 0), 1)} ({pct:.0f}th percentile)\n"

    report += "\n---\n\n### Development Areas\n\n"

    weaknesses = []
    for m in all_metrics:
        pct_col = m + " (pct)"
        if pct_col in row:
            pct = safe_float(row.get(pct_col, np.nan))
            if not np.isnan(pct) and pct < 40:
                weaknesses.append((m, pct))
    weaknesses.sort(key=lambda x: x[1])

    for metric, pct in weaknesses[:5]:
        report += f"- **{metric}**: {safe_fmt(row.get(metric, 0), 1)} ({pct:.0f}th percentile)\n"

    report += f"\n---\n\n**Report Generated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    st.markdown(report)

    st.markdown("---")
    st.markdown("### 📥 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download as Text",
            report,
            file_name=f"scout_report_{player_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.info("PDF export: add reportlab/weasyprint if needed.")

# =====================================================
# MAIN APP
# =====================================================
def main():
    if st.session_state.view == "landing":
        render_nav(show_back=False)
        render_landing_view()
    else:
        render_nav(show_back=True)
        with st.spinner("Loading data..."):
            df, cfg, all_metrics = load_data(st.session_state.position)

        if st.session_state.view == "search":
            render_search_view(df, cfg, all_metrics)
        elif st.session_state.view == "dashboard":
            render_dashboard_view(df, cfg, all_metrics)

if __name__ == "__main__":
    main()
