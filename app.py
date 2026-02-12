"""
IMPECT Stats Table — Multi-League with TOP BUTTONS + PROPER DARK FILTERS
========================================================================
What you asked:
✅ League switch buttons on top (filename = league name from ./data)
✅ Filters in the LEFT SIDEBAR with matching StatsBomb-esque colors
✅ Table matches theme, sticky cols, subtle pct/z bars
✅ Positions abbreviated

Folder:
./data/
  Keuken Kampioen Divisie.xlsx
  Eredivisie.xlsx
  Premier League.xlsx
  ...

Each file must contain sheets: "Standard" and "xG"
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IMPECT Stats | Multi-League",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"

# ─────────────────────────────────────────────
# COLOUR PALETTE (StatsBomb-esque)
# ─────────────────────────────────────────────
BG = "#1A1A1A"
RADAR_BG = "#222222"
GRID_COLOR = "#333333"
PLAYER_COL = "#F5C518"      # Gold
LEAGUE_COL = "#5A5A5A"      # Grey baseline
TEXT_LIGHT = "#E0E0E0"
TEXT_DIM = "#888888"
TABLE_ROW_ALT = "#2A2A2A"
TABLE_BORDER = "#3A3A3A"
PCT_HIGH = "#4CAF50"        # Green
PCT_MID = "#F5C518"         # Yellow
PCT_LOW = "#E53935"         # Red
ACCENT_LINE = "#F5C518"

INVERTED = ["foul", "lost", "unsuccessful", "failed", "off target", "red", "yellow"]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def is_inverted(col: str) -> bool:
    c = (col or "").lower()
    return any(x in c for x in INVERTED)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def clean_stat_name(stat: str) -> str:
    return stat.split(" (")[0] if " (" in stat else stat


# ─────────────────────────────────────────────
# Position abbreviations
# ─────────────────────────────────────────────
def _abbr_single(token: str) -> str:
    t = (token or "").strip().upper()
    if not t:
        return ""
    t = t.replace("CENTRE", "CENTER")

    direct = {
        "GOALKEEPER": "GK",
        "KEEPER": "GK",
        "CENTER_FORWARD": "CF",
        "CENTRAL_FORWARD": "CF",
        "STRIKER": "ST",
        "CENTER_STRIKER": "ST",
        "SECOND_STRIKER": "SS",
        "ATTACKING_MIDFIELD": "AM",
        "DEFENSE_MIDFIELD": "DM",
        "DEFENSIVE_MIDFIELD": "DM",
        "CENTRAL_MIDFIELD": "CM",
        "WIDE_MIDFIELD": "WM",
        "CENTRAL_DEFENDER": "CB",
        "CENTER_BACK": "CB",
        "FULLBACK": "FB",
    }
    if t in direct:
        return direct[t]

    side = ""
    if t.startswith("LEFT_"):
        side = "L"
        t = t[len("LEFT_"):]
    elif t.startswith("RIGHT_"):
        side = "R"
        t = t[len("RIGHT_"):]

    if "WINGBACK" in t:
        return f"{side}WB" if side else "WB"
    if "FULLBACK" in t or t == "BACK":
        return f"{side}B" if side else "FB"
    if "WINGER" in t:
        return f"{side}W" if side else "W"
    if "WIDE" in t and "MIDFIELD" in t:
        return f"{side}M" if side else "WM"

    if "DEFENDER" in t or "DEFENCE" in t or "DEFENSE" in t:
        if "CENTRAL" in t or "CENTER" in t:
            return "CB"
        if side:
            return f"{side}B"
        return "DEF"

    if "MIDFIELD" in t:
        if "ATTACK" in t:
            return "AM"
        if "DEFENSE" in t or "DEFENCE" in t:
            return "DM"
        if "CENTRAL" in t or "CENTER" in t:
            return "CM"
        if side:
            return f"{side}M"
        return "MF"

    if "FORWARD" in t:
        if "CENTER" in t or "CENTRAL" in t:
            return "CF"
        if side:
            return f"{side}F"
        return "F"

    parts = [p for p in token.split("_") if p]
    initials = "".join(p[0] for p in parts[:3]).upper()
    return initials or token[:3].upper()

def abbreviate_positions(pos_str: str) -> str:
    if pd.isna(pos_str) or not str(pos_str).strip():
        return ""
    tokens = [p.strip() for p in str(pos_str).split(",") if p.strip()]
    out, seen = [], set()
    for tok in tokens:
        ab = _abbr_single(tok)
        if ab and ab not in seen:
            seen.add(ab)
            out.append(ab)
    return ", ".join(out)


# ─────────────────────────────────────────────
# League discovery
# ─────────────────────────────────────────────
def list_league_files(data_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(data_dir):
        return []
    out = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith((".xlsx", ".xls")) and not fn.startswith("~$"):
            league = os.path.splitext(fn)[0]
            out.append((league, os.path.join(data_dir, fn)))
    return sorted(out, key=lambda x: x[0].lower())

league_files = list_league_files(DATA_DIR)
if not league_files:
    st.error(f"No Excel files found in '{DATA_DIR}/'. Put league files there (e.g., '{DATA_DIR}/Eredivisie.xlsx').")
    st.stop()

league_names = [x[0] for x in league_files]
league_to_path = {lg: path for lg, path in league_files}


# ─────────────────────────────────────────────
# Data loading + transforms
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    standard = pd.read_excel(file_path, sheet_name="Standard")
    xg = pd.read_excel(file_path, sheet_name="xG")

    base = [
        "iterationId", "squadId", "squadName", "playerId", "positions",
        "commonname", "firstname", "lastname", "birthdate", "birthplace",
        "leg", "countryIds", "gender", "season", "dataVersion",
        "lastChangeTimestamp", "competition_name", "competition_type", "competition_gender"
    ]

    xg_kpis = [c for c in xg.columns if c not in base and c != "playerId"]
    merged = standard.merge(xg[["playerId"] + xg_kpis], on="playerId", how="left")

    cn = merged.get("commonname", "").fillna("").astype(str).str.strip()
    fallback = (
        merged.get("firstname", "").fillna("").astype(str).str.strip()
        + " "
        + merged.get("lastname", "").fillna("").astype(str).str.strip()
    ).str.strip()
    merged["displayName"] = np.where(cn == "", fallback, cn)
    merged["posAbbr"] = merged.get("positions", "").apply(abbreviate_positions) if "positions" in merged.columns else ""
    return merged

def get_kpis(df: pd.DataFrame) -> list[str]:
    base = {
        "iterationId", "squadId", "squadName", "playerId", "positions", "posAbbr",
        "commonname", "firstname", "lastname", "birthdate", "birthplace",
        "leg", "countryIds", "gender", "season", "dataVersion",
        "lastChangeTimestamp", "competition_name", "competition_type", "competition_gender",
        "displayName"
    }
    return [c for c in df.columns if c not in base and pd.api.types.is_numeric_dtype(df[c])]

def calc_percentiles(df: pd.DataFrame, kpis: list[str]) -> pd.DataFrame:
    for col in kpis:
        v = pd.to_numeric(df[col], errors="coerce")
        pct = v.rank(pct=True, method="average") * 100
        df[f"{col}_pct"] = 100 - pct if is_inverted(col) else pct
    return df

def calc_zscores(df: pd.DataFrame, kpis: list[str]) -> pd.DataFrame:
    for col in kpis:
        x = pd.to_numeric(df[col], errors="coerce")
        if is_inverted(col):
            x = -x
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=0)
        df[f"{col}_z"] = np.nan if (sd == 0 or pd.isna(sd)) else (x - mu) / sd
    return df


# ─────────────────────────────────────────────
# Cell styles
# ─────────────────────────────────────────────
def pct_cell_style(val):
    if pd.isna(val):
        return f"color:{TEXT_DIM}; background-color: transparent;"
    v = clamp(float(val), 0.0, 100.0)
    if v >= 85:
        col, weight, glow = PCT_HIGH, 900, "rgba(76,175,80,.10)"
    elif v <= 25:
        col, weight, glow = PCT_LOW, 900, "rgba(229,57,53,.10)"
    else:
        col, weight, glow = PCT_MID, 850, "rgba(245,197,24,.10)"
    return f"""
      background-color: transparent;
      background-image:
        linear-gradient(to right, rgba(255,255,255,.06) 0%, rgba(255,255,255,.06) 100%),
        linear-gradient(to right, {col} {v:.1f}%, transparent {v:.1f}%);
      box-shadow: inset 0 0 0 9999px {glow};
      color: {TEXT_LIGHT};
      font-weight: {weight};
    """

def z_cell_style(val):
    if pd.isna(val):
        return f"color:{TEXT_DIM}; background-color: transparent;"
    z = clamp(float(val), -3.0, 3.0)
    pos = (z + 3.0) / 6.0 * 100.0
    mid = 50.0
    col = PCT_HIGH if z >= 0 else PCT_LOW
    weight = 900 if abs(z) >= 1.25 else 850
    glow = "rgba(76,175,80,.10)" if z >= 0 else "rgba(229,57,53,.10)"
    center = f"""
      linear-gradient(to right,
        transparent 49.6%,
        {LEAGUE_COL} 49.6%,
        {LEAGUE_COL} 50.4%,
        transparent 50.4%)
    """
    if pos >= mid:
        bar = f"""
        linear-gradient(to right,
          transparent 0%,
          transparent {mid:.1f}%,
          {col} {mid:.1f}%,
          {col} {pos:.1f}%,
          transparent {pos:.1f}%,
          transparent 100%)
        """
    else:
        bar = f"""
        linear-gradient(to right,
          transparent 0%,
          transparent {pos:.1f}%,
          {col} {pos:.1f}%,
          {col} {mid:.1f}%,
          transparent {mid:.1f}%,
          transparent 100%)
        """
    track = "linear-gradient(to right, rgba(255,255,255,.06) 0%, rgba(255,255,255,.06) 100%)"
    return f"""
      background-color: transparent;
      background-image: {center}, {track}, {bar};
      box-shadow: inset 0 0 0 9999px {glow};
      color: {TEXT_LIGHT};
      font-weight: {weight};
    """


# ─────────────────────────────────────────────
# Global CSS — SIDEBAR CONTROLS PROPERLY DARK
# ─────────────────────────────────────────────
st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  background: {BG} !important;
  color: {TEXT_LIGHT} !important;
}}
#MainMenu, footer, header, .stDeployButton {{ visibility:hidden !important; display:none !important; }}

html, body, [class*="css"], .stMarkdown, .stText, .stCaption {{
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial !important;
}}

.block-container {{
  max-width: 1550px !important;
  padding-top: 1.0rem !important;
  padding-bottom: 2.0rem !important;
}}

/* SIDEBAR panel */
section[data-testid="stSidebar"] {{
  background: {RADAR_BG} !important;
  border-right: 1px solid {TABLE_BORDER} !important;
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5 {{
  color: {TEXT_LIGHT} !important;
}}

/* Make all widget labels consistent */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] p {{
  color: {TEXT_DIM} !important;
  font-weight: 650 !important;
}}

/* Inputs (text/select) */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {{
  background: {BG} !important;
  color: {TEXT_LIGHT} !important;
  border: 1px solid {TABLE_BORDER} !important;
  border-radius: 12px !important;
}}
section[data-testid="stSidebar"] input:focus,
section[data-testid="stSidebar"] textarea:focus {{
  border-color: {ACCENT_LINE} !important;
  box-shadow: 0 0 0 3px rgba(245,197,24,.18) !important;
}}

/* Baseweb Select (Selectbox + Multiselect) */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
  background: {BG} !important;
  border: 1px solid {TABLE_BORDER} !important;
  border-radius: 12px !important;
}}
section[data-testid="stSidebar"] [data-baseweb="select"] * {{
  color: {TEXT_LIGHT} !important;
}}
/* Dropdown menu */
div[data-baseweb="popover"] * {{
  color: {TEXT_LIGHT} !important;
}}
div[data-baseweb="menu"] {{
  background: {RADAR_BG} !important;
  border: 1px solid {TABLE_BORDER} !important;
  border-radius: 12px !important;
}}
div[data-baseweb="menu"] li {{
  background: transparent !important;
}}
div[data-baseweb="menu"] li:hover {{
  background: {GRID_COLOR} !important;
}}

/* Multiselect chips */
section[data-testid="stSidebar"] [data-baseweb="tag"] {{
  background: {BG} !important;
  border: 1px solid {TABLE_BORDER} !important;
  color: {TEXT_LIGHT} !important;
}}

/* Toggle switch */
section[data-testid="stSidebar"] [data-testid="stToggleSwitch"] label {{
  color: {TEXT_LIGHT} !important;
}}
section[data-testid="stSidebar"] [data-testid="stToggleSwitch"] span {{
  color: {TEXT_LIGHT} !important;
}}

/* Cards + badges */
.card {{
  background: {RADAR_BG};
  border: 1px solid {TABLE_BORDER};
  border-radius: 14px;
  padding: 14px 16px;
}}
.badge {{
  display:inline-flex;
  align-items:center;
  gap:.45rem;
  padding:.32rem .6rem;
  border: 1px solid {TABLE_BORDER};
  border-radius: 999px;
  font-size: .78rem;
  font-weight: 800;
  color: {TEXT_LIGHT};
  background: {BG};
}}
.badge .dim {{ color: {TEXT_DIM}; font-weight: 700; }}
.small-muted {{ color: {TEXT_DIM} !important; }}
hr {{ border-color: {TABLE_BORDER} !important; }}

/* League buttons */
.league-row {{
  display:flex;
  flex-wrap:wrap;
  gap:.5rem;
  margin-top: .9rem;
  margin-bottom: .6rem;
}}

/* Make Streamlit buttons look like pills */
div.stButton > button {{
  border-radius: 12px !important;
  border: 1px solid {TABLE_BORDER} !important;
  background: {RADAR_BG} !important;
  color: {TEXT_LIGHT} !important;
  font-weight: 900 !important;
  padding: .5rem .85rem !important;
}}
div.stButton > button:hover {{
  border-color: {ACCENT_LINE} !important;
}}
/* Active league button: we set via label prefix ✓ and extra styling using :has when supported */
div.stButton > button:focus {{
  border-color: {ACCENT_LINE} !important;
  box-shadow: 0 0 0 3px rgba(245,197,24,.12) !important;
}}

/* Downloads */
.stDownloadButton button {{
  background: {ACCENT_LINE} !important;
  color: #000 !important;
  font-weight: 950 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
}}
.stDownloadButton button:hover {{ filter: brightness(0.95) !important; }}

/* TABLE */
.table-wrap {{
  max-height: 720px;
  overflow: auto;
  border-radius: 14px;
  border: 1px solid {TABLE_BORDER};
  background: {BG};
}}

table.dataframe {{
  width: 100% !important;
  table-layout: fixed;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  background: transparent !important;
}}

table.dataframe thead th {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: {RADAR_BG} !important;
  color: {TEXT_LIGHT} !important;
  border-bottom: 1px solid {TABLE_BORDER} !important;
  border-right: 1px solid {TABLE_BORDER} !important;
  font-size: 11px !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
  padding: 12px 10px !important;
  text-align: left;
}}
table.dataframe thead th:last-child {{ border-right: none !important; }}

table.dataframe tbody tr {{ background: {BG} !important; }}
table.dataframe tbody tr:nth-child(even) {{ background: {TABLE_ROW_ALT} !important; }}
table.dataframe tbody tr:hover {{ background: {GRID_COLOR} !important; }}

table.dataframe tbody td {{
  background-color: transparent !important;
  color: {TEXT_LIGHT} !important;
  border-bottom: 1px solid {TABLE_BORDER} !important;
  border-right: 1px solid {TABLE_BORDER} !important;
  font-size: 13px !important;
  padding: 10px 10px !important;
  text-align: center;
  font-variant-numeric: tabular-nums !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
table.dataframe tbody td:last-child {{ border-right: none !important; }}

/* Column sizing */
table.dataframe thead th:nth-child(1),
table.dataframe tbody td:nth-child(1) {{ width: 60px; }}   /* Rank */
table.dataframe thead th:nth-child(2),
table.dataframe tbody td:nth-child(2) {{ width: 190px; text-align:left; font-weight: 900; }} /* Player */
table.dataframe thead th:nth-child(3),
table.dataframe tbody td:nth-child(3) {{ width: 170px; text-align:left; color:{TEXT_DIM} !important; font-weight: 750; }} /* Squad */
table.dataframe thead th:nth-child(4),
table.dataframe tbody td:nth-child(4) {{ width: 160px; text-align:left; font-weight: 850; }} /* Positions */
table.dataframe thead th:nth-child(n+5),
table.dataframe tbody td:nth-child(n+5) {{ width: 115px; }}

/* Sticky first 4 columns */
table.dataframe thead th:nth-child(1),
table.dataframe tbody td:nth-child(1) {{ position: sticky; left: 0; z-index: 9; background: inherit !important; }}
table.dataframe thead th:nth-child(2),
table.dataframe tbody td:nth-child(2) {{ position: sticky; left: 60px; z-index: 9; background: inherit !important; }}
table.dataframe thead th:nth-child(3),
table.dataframe tbody td:nth-child(3) {{ position: sticky; left: 250px; z-index: 9; background: inherit !important; }}
table.dataframe thead th:nth-child(4),
table.dataframe tbody td:nth-child(4) {{ position: sticky; left: 420px; z-index: 9; background: inherit !important; }}

table.dataframe tbody td:nth-child(4) {{ box-shadow: 8px 0 12px rgba(0,0,0,.25); }}
table.dataframe thead th:nth-child(4) {{ box-shadow: 8px 0 12px rgba(0,0,0,.35); }}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    f"""
<div class="card" style="display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;">
  <div>
    <div class="badge"><span>IMPECT</span> <span class="dim">•</span> <span>Multi-League</span></div>
    <h1 style="margin:.45rem 0 0 0; font-size: 1.9rem;">Player Stats Explorer</h1>
    <div class="small-muted">Buttons on top • Filters left • per 90 • percentiles + z-scores • subtle bars • positions abbreviated</div>
  </div>
  <div class="badge"><span class="dim">Leagues</span> {len(league_names)}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# League selector (TOP BUTTONS)
# ─────────────────────────────────────────────
if "league_selected" not in st.session_state:
    st.session_state["league_selected"] = league_names[0]

st.markdown('<div class="league-row">', unsafe_allow_html=True)

def chunks(lst, n=6):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

for row in chunks(league_names, 6):
    cols = st.columns(len(row))
    for lg, c in zip(row, cols):
        active = (lg == st.session_state["league_selected"])
        label = f"✓ {lg}" if active else lg
        if c.button(label, use_container_width=True, key=f"league_btn_{lg}"):
            st.session_state["league_selected"] = lg

st.markdown("</div>", unsafe_allow_html=True)

league_name = st.session_state["league_selected"]
file_path = league_to_path[league_name]

# ─────────────────────────────────────────────
# Load selected league
# ─────────────────────────────────────────────
try:
    df = load_data(file_path)
    kpis = get_kpis(df)
    df = calc_percentiles(df, kpis)
    df = calc_zscores(df, kpis)
except Exception as e:
    st.error(f"Failed to load '{league_name}': {e}")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar — FILTERS LEFT (proper dark styling)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🎛 Filters")
    st.markdown(f"<div class='small-muted'>League: <b style='color:{ACCENT_LINE}'>{league_name}</b></div>", unsafe_allow_html=True)
    st.divider()

    name_filter = st.text_input("Player search", placeholder="Type player name…", key=f"name_{league_name}")
    st.divider()

    squads = ["All Squads"] + sorted(df["squadName"].dropna().unique().tolist())
    squad = st.selectbox("Squad", squads, key=f"squad_{league_name}")
    st.divider()

    pos_group = st.selectbox(
        "Position group",
        ["All Positions", "Defenders", "Midfielders", "Forwards"],
        key=f"posgrp_{league_name}",
    )
    st.divider()

    categories = {
        "⚽ Goals & Assists": ["Goals", "Assists", "Pre Assist", "Shot-Creating Actions", "Shot xG from Passes"],
        "🎯 Shooting": ["Total Shots", "Total Shots On Target", "Shot-based xG", "Post-Shot xG"],
        "📤 Passing": ["Successful Passes", "Unsuccessful Passes", "Progressive passes", "Pass Accuracy"],
        "🤼 Duels": ["Won Ground Duels", "Lost Ground Duels", "Won Aerial Duels", "Lost Aerial Duels"],
        "📈 xG Metrics": ["Shot-based xG", "Post-Shot xG", "Expected Goal Assists", "Expected Shot Assists", "Packing non-shot-based xG"],
    }
    selected_cat = st.selectbox("Stat category", list(categories.keys()), key=f"cat_{league_name}")
    keywords = categories[selected_cat]
    st.divider()

    display_mode = st.selectbox("Display mode", ["Percentiles", "Raw values", "Both"], index=0, key=f"display_{league_name}")
    metric_mode = st.selectbox("Metric mode", ["Percentile", "Z-score"], index=0, key=f"metric_{league_name}")

    show_bars = st.toggle("Show distribution bars", value=True, key=f"bars_{league_name}")
    show_rank = st.toggle("Show rank", value=True, key=f"rank_{league_name}")

    st.divider()

    matching = [k for k in kpis if any(kw in k for kw in keywords)]
    if not matching:
        matching = kpis[:]
    stats_options = matching[:40] if len(matching) > 40 else matching

    selected_stats = st.multiselect(
        "Select stats",
        options=stats_options,
        default=stats_options[:8] if len(stats_options) >= 8 else stats_options,
        key=f"stats_{league_name}",
    )

    if not selected_stats:
        st.warning("Select at least one stat.")
        st.stop()

# ─────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────
df_filtered = df.copy()
if name_filter:
    df_filtered = df_filtered[df_filtered["displayName"].str.contains(name_filter, case=False, na=False)]
if squad != "All Squads":
    df_filtered = df_filtered[df_filtered["squadName"] == squad]
if pos_group != "All Positions":
    pos_map = {
        "Defenders": ["DEFENDER", "BACK"],
        "Midfielders": ["MIDFIELD"],
        "Forwards": ["FORWARD", "WINGER"],
    }
    tokens = pos_map[pos_group]
    mask = pd.Series(False, index=df_filtered.index)
    for t in tokens:
        mask |= df_filtered["positions"].str.contains(t, case=False, na=False)
    df_filtered = df_filtered[mask]

# ─────────────────────────────────────────────
# Status badges
# ─────────────────────────────────────────────
st.markdown(
    f"""
<div style="margin-top: .8rem; display:flex; gap:.5rem; flex-wrap:wrap;">
  <div class="badge"><span class="dim">League</span> {league_name}</div>
  <div class="badge"><span class="dim">Players</span> {len(df_filtered)}</div>
  <div class="badge"><span class="dim">Teams</span> {df_filtered["squadName"].nunique()}</div>
  <div class="badge"><span class="dim">Stats</span> {len(selected_stats)}</div>
  <div class="badge"><span class="dim">Display</span> {display_mode}</div>
  <div class="badge"><span class="dim">Metric</span> {metric_mode}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Build display dataframe
# ─────────────────────────────────────────────
base_cols = ["displayName", "squadName", "posAbbr"]
cols = base_cols.copy()

metric_suffix = "_pct" if metric_mode == "Percentile" else "_z"
metric_label = "(PCT)" if metric_mode == "Percentile" else "(Z)"

if display_mode == "Percentiles":
    for stat in selected_stats:
        mcol = f"{stat}{metric_suffix}"
        if mcol in df_filtered.columns:
            cols.append(mcol)
elif display_mode == "Raw values":
    for stat in selected_stats:
        if stat in df_filtered.columns:
            cols.append(stat)
else:
    for stat in selected_stats:
        if stat in df_filtered.columns:
            cols.append(stat)
        mcol = f"{stat}{metric_suffix}"
        if mcol in df_filtered.columns:
            cols.append(mcol)

cols = [c for c in cols if c in df_filtered.columns]
df_display = df_filtered[cols].copy()

rename_map = {"posAbbr": "positions"}
seen = set()
for col in df_display.columns:
    if col in base_cols:
        continue
    if col.endswith("_pct"):
        nice = f"{clean_stat_name(col[:-4])} {metric_label}"
    elif col.endswith("_z"):
        nice = f"{clean_stat_name(col[:-2])} {metric_label}"
    else:
        nice = clean_stat_name(col)

    original = nice
    i = 1
    while nice in seen:
        i += 1
        nice = f"{original} [{i}]"
    seen.add(nice)
    rename_map[col] = nice

df_display = df_display.rename(columns=rename_map)

# Sort
sort_col = None
for c in df_display.columns:
    if c.endswith(metric_label):
        sort_col = c
        break
if sort_col is None and len(df_display.columns) > 3:
    sort_col = df_display.columns[3]
if sort_col:
    df_display = df_display.sort_values(sort_col, ascending=False, na_position="last")

# Rank
if show_rank and sort_col:
    df_display.insert(0, "Rank", df_display[sort_col].rank(ascending=False, method="min").astype("Int64"))

# ─────────────────────────────────────────────
# Style + render
# ─────────────────────────────────────────────
styled = df_display.style.hide(axis="index")

fmt = {}
for c in df_display.columns:
    if c in ["Rank", "displayName", "squadName", "positions"]:
        continue
    if c.endswith("(PCT)"):
        fmt[c] = "{:.0f}"
    elif c.endswith("(Z)"):
        fmt[c] = "{:+.2f}"
    else:
        fmt[c] = "{:.2f}"
styled = styled.format(fmt, na_rep="-")

if "Rank" in df_display.columns:
    styled = styled.applymap(lambda v: f"color:{TEXT_DIM}; font-weight:950; background-color: transparent;", subset=["Rank"])

if show_bars:
    for c in df_display.columns:
        if c.endswith("(PCT)"):
            styled = styled.applymap(pct_cell_style, subset=[c])
        if c.endswith("(Z)"):
            styled = styled.applymap(z_cell_style, subset=[c])

st.markdown(
    f"""
<div style="margin-top: .9rem; margin-bottom: .5rem; display:flex; align-items:center; justify-content:space-between; gap:1rem;">
  <h3 style="margin:0;">Player Table</h3>
  <div class="badge"><span class="dim">Sorted by</span> {sort_col if sort_col else "—"}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(f'<div class="table-wrap">{styled.to_html()}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="card" style="margin-top: 1.2rem;">
  <h3 style="margin:0 0 .35rem 0;">Export</h3>
  <div class="small-muted">Exports the current view (filters + columns shown).</div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ CSV",
        csv,
        f"{league_name}_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True,
        key=f"csv_{league_name}",
    )

with c2:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_display.to_excel(writer, index=False)
    st.download_button(
        "⬇ Excel",
        buffer.getvalue(),
        f"{league_name}_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key=f"xlsx_{league_name}",
    )

with c3:
    legend = (
        "Percentiles: green=high, yellow=mid, red=low. Bars fill 0→100."
        if metric_mode == "Percentile"
        else "Z-scores: green=positive, red=negative. Bars diverge from 0 (center line)."
    )
    st.markdown(
        f"""
<div class="card" style="height:100%; display:flex; flex-direction:column; justify-content:space-between;">
  <div>
    <div class="badge"><span class="dim">Legend</span> {metric_mode}</div>
    <div class="small-muted" style="margin-top:.6rem; line-height:1.35;">
      {legend}<br/>
      <span style="color:{TEXT_DIM};">Lower-is-better metrics are automatically inverted.</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
