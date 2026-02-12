"""
IMPECT Stats Table — Multi-League + Player Profiles (ALWAYS-VISIBLE LEFT FILTER PANEL)
====================================================================================
✅ League switch buttons on top (from ./data filenames)
✅ Real left menu (not Streamlit sidebar) => always visible
✅ Filters: Club, Position group, Position abbreviations, Player search, Stat category, Stats selection
✅ Two views:
   1) Table
   2) Player Profile (cards + top attributes + similar players)
✅ Percentile / Z-score mode toggle
✅ Subtle distribution bars
✅ Dark StatsBomb-esque palette
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
from datetime import date
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
    initial_sidebar_state="collapsed",
)

DATA_DIR = "data"

# ─────────────────────────────────────────────
# COLOUR PALETTE (StatsBomb-esque)
# ─────────────────────────────────────────────
BG = "#1A1A1A"
RADAR_BG = "#222222"
GRID_COLOR = "#333333"
PLAYER_COL = "#F5C518"
LEAGUE_COL = "#5A5A5A"
TEXT_LIGHT = "#E0E0E0"
TEXT_DIM = "#888888"
TABLE_ROW_ALT = "#2A2A2A"
TABLE_BORDER = "#3A3A3A"
PCT_HIGH = "#4CAF50"
PCT_MID = "#F5C518"
PCT_LOW = "#E53935"
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

def safe_dt(x):
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

def calc_age(birthdate) -> str:
    bd = safe_dt(birthdate)
    if pd.isna(bd):
        return ""
    bd = bd.date()
    today = date.today()
    years = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    return str(years)

def html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

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

def explode_posabbr(series: pd.Series) -> list[str]:
    vals = set()
    for x in series.dropna().astype(str):
        for t in [p.strip() for p in x.split(",")]:
            if t:
                vals.add(t)
    order = ["GK","CB","LB","RB","LWB","RWB","DM","CM","AM","LW","RW","CF","ST","SS","FB","W","MF","DEF","F","WM"]
    return sorted(vals, key=lambda v: (order.index(v) if v in order else 999, v))


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
# Cell styles for table bars
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
# Profile UI helpers (cards + bars)
# ─────────────────────────────────────────────
def pct_bar(pct: float) -> str:
    if pd.isna(pct):
        return f"<div class='barwrap'><div class='barbg'></div><div class='bartext dim'>-</div></div>"
    p = clamp(float(pct), 0, 100)
    if p >= 85:
        col = PCT_HIGH
    elif p <= 25:
        col = PCT_LOW
    else:
        col = PCT_MID
    return f"""
    <div class="barwrap">
      <div class="barbg"></div>
      <div class="barfill" style="width:{p:.1f}%; background:{col};"></div>
      <div class="bartext">{p:.0f}</div>
    </div>
    """

def z_bar(z: float) -> str:
    if pd.isna(z):
        return f"<div class='zwrap'><div class='zcenter'></div><div class='bartext dim'>-</div></div>"
    z = clamp(float(z), -3, 3)
    pos = (z + 3) / 6 * 100
    col = PCT_HIGH if z >= 0 else PCT_LOW
    # bar from center (50) to pos
    left = min(50, pos)
    width = abs(pos - 50)
    return f"""
    <div class="zwrap">
      <div class="zcenter"></div>
      <div class="zfill" style="left:{left:.2f}%; width:{width:.2f}%; background:{col};"></div>
      <div class="bartext">{z:+.2f}</div>
    </div>
    """


# ─────────────────────────────────────────────
# CSS (left panel + table + profile)
# ─────────────────────────────────────────────
st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  background: {BG} !important;
  color: {TEXT_LIGHT} !important;
}}
#MainMenu, footer, header, .stDeployButton {{ visibility:hidden !important; display:none !important; }}

.block-container {{
  max-width: 1680px !important;
  padding-top: 1.0rem !important;
  padding-bottom: 2.0rem !important;
}}

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
  font-weight: 850;
  color: {TEXT_LIGHT};
  background: {BG};
}}
.badge .dim {{ color: {TEXT_DIM}; font-weight: 750; }}
.small-muted {{ color: {TEXT_DIM} !important; }}
hr {{ border-color: {TABLE_BORDER} !important; }}

.filter-panel {{
  background: {RADAR_BG};
  border: 1px solid {TABLE_BORDER};
  border-radius: 14px;
  padding: 12px 12px;
  position: sticky;
  top: 12px;
}}
.filter-title {{
  font-size: 1.05rem;
  font-weight: 900;
  margin: 0 0 6px 0;
}}
.filter-sub {{
  color: {TEXT_DIM};
  font-size: .82rem;
  margin-bottom: 8px;
}}

[data-baseweb="select"] > div {{
  background: {BG} !important;
  border: 1px solid {TABLE_BORDER} !important;
  border-radius: 12px !important;
}}
[data-baseweb="select"] * {{ color: {TEXT_LIGHT} !important; }}

input {{
  background: {BG} !important;
  color: {TEXT_LIGHT} !important;
  border: 1px solid {TABLE_BORDER} !important;
  border-radius: 12px !important;
}}
input:focus {{
  border-color: {ACCENT_LINE} !important;
  box-shadow: 0 0 0 3px rgba(245,197,24,.18) !important;
}}

div[data-baseweb="menu"] {{
  background: {RADAR_BG} !important;
  border: 1px solid {TABLE_BORDER} !important;
  border-radius: 12px !important;
}}
div[data-baseweb="menu"] li:hover {{ background: {GRID_COLOR} !important; }}

[data-baseweb="tag"] {{
  background: {BG} !important;
  border: 1px solid {TABLE_BORDER} !important;
  color: {TEXT_LIGHT} !important;
}}

/* Buttons */
div.stButton > button {{
  border-radius: 12px !important;
  border: 1px solid {TABLE_BORDER} !important;
  background: {RADAR_BG} !important;
  color: {TEXT_LIGHT} !important;
  font-weight: 900 !important;
  padding: .5rem .85rem !important;
}}
div.stButton > button:hover {{ border-color: {ACCENT_LINE} !important; }}

/* Table wrapper */
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
table.dataframe tbody td:nth-child(1) {{ width: 60px; }}
table.dataframe thead th:nth-child(2),
table.dataframe tbody td:nth-child(2) {{ width: 190px; text-align:left; font-weight: 900; }}
table.dataframe thead th:nth-child(3),
table.dataframe tbody td:nth-child(3) {{ width: 170px; text-align:left; color:{TEXT_DIM} !important; font-weight: 750; }}
table.dataframe thead th:nth-child(4),
table.dataframe tbody td:nth-child(4) {{ width: 160px; text-align:left; font-weight: 850; }}
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

/* Profile bars */
.barwrap {{
  position: relative;
  height: 18px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255,255,255,.06);
  border: 1px solid {TABLE_BORDER};
}}
.barbg {{
  position:absolute; inset:0;
  background: rgba(255,255,255,.06);
}}
.barfill {{
  position:absolute; left:0; top:0; bottom:0;
  opacity: .85;
}}
.zwrap {{
  position: relative;
  height: 18px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255,255,255,.06);
  border: 1px solid {TABLE_BORDER};
}}
.zcenter {{
  position:absolute; top:0; bottom:0; left:50%;
  width:2px;
  background: {LEAGUE_COL};
  opacity: .9;
}}
.zfill {{
  position:absolute; top:0; bottom:0;
  opacity: .85;
}}
.bartext {{
  position:absolute; inset:0;
  display:flex; align-items:center; justify-content:center;
  font-size: 11px;
  font-weight: 900;
  color: {TEXT_LIGHT};
}}
.dim {{ color: {TEXT_DIM} !important; }}
.profile-grid {{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}}
.metric-row {{
  display:grid;
  grid-template-columns: 1.2fr 1.0fr;
  gap: 10px;
  align-items:center;
  padding: 8px 0;
  border-bottom: 1px dashed rgba(255,255,255,.10);
}}
.metric-row:last-child {{ border-bottom: none; }}
.metric-name {{
  font-weight: 850;
  font-size: 12.5px;
  color: {TEXT_LIGHT};
}}
.metric-sub {{
  font-size: 11px;
  color: {TEXT_DIM};
  margin-top: 2px;
}}
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
    <div class="small-muted">Left menu = filters • Top buttons = leagues • Table + Player Profiles</div>
  </div>
  <div class="badge"><span class="dim">Leagues</span> {len(league_names)}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# League buttons (top)
# ─────────────────────────────────────────────
if "league_selected" not in st.session_state:
    st.session_state["league_selected"] = league_names[0]

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

league_name = st.session_state["league_selected"]
file_path = league_to_path[league_name]

# ─────────────────────────────────────────────
# Load league
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
# Layout: LEFT FILTER PANEL + RIGHT CONTENT
# ─────────────────────────────────────────────
left, right = st.columns([0.28, 0.72], gap="large")

# ─────────────────────────────────────────────
# LEFT FILTER PANEL
# ─────────────────────────────────────────────
with left:
    st.markdown(
        f"""
<div class="filter-panel">
  <div class="filter-title">🎛 Filters</div>
  <div class="filter-sub">League: <b style="color:{ACCENT_LINE}">{league_name}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

    name_filter = st.text_input("Player", placeholder="Search player name…", key=f"name_{league_name}")

    squads = ["All Clubs"] + sorted(df["squadName"].dropna().unique().tolist())
    squad = st.selectbox("Club", squads, key=f"club_{league_name}")

    pos_group = st.selectbox(
        "Position Group",
        ["All Positions", "Defenders", "Midfielders", "Forwards", "Goalkeepers"],
        key=f"posgrp_{league_name}",
    )

    all_posabbr = explode_posabbr(df.get("posAbbr", pd.Series([], dtype=str)))
    selected_posabbr = st.multiselect(
        "Positions (abbr.)",
        options=all_posabbr,
        default=[],
        help="Example: CB, DM, ST. Leave empty to ignore.",
        key=f"posabbr_{league_name}",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    categories = {
        "⚽ Goals & Assists": ["Goals", "Assists", "Pre Assist", "Shot-Creating Actions", "Shot xG from Passes"],
        "🎯 Shooting": ["Total Shots", "Total Shots On Target", "Shot-based xG", "Post-Shot xG"],
        "📤 Passing": ["Successful Passes", "Unsuccessful Passes", "Progressive passes", "Pass Accuracy"],
        "🤼 Duels": ["Won Ground Duels", "Lost Ground Duels", "Won Aerial Duels", "Lost Aerial Duels"],
        "📈 xG Metrics": ["Shot-based xG", "Post-Shot xG", "Expected Goal Assists", "Expected Shot Assists", "Packing non-shot-based xG"],
    }
    selected_cat = st.selectbox("Stat Category", list(categories.keys()), key=f"cat_{league_name}")
    keywords = categories[selected_cat]

    matching = [k for k in kpis if any(kw in k for kw in keywords)]
    if not matching:
        matching = kpis[:]
    stats_options = matching[:40] if len(matching) > 40 else matching

    selected_stats = st.multiselect(
        "Stats",
        options=stats_options,
        default=stats_options[:8] if len(stats_options) >= 8 else stats_options,
        key=f"stats_{league_name}",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    metric_mode = st.selectbox("Metric Mode", ["Percentile", "Z-score"], index=0, key=f"metric_{league_name}")
    display_mode = st.selectbox("Display", ["Metrics only", "Raw only", "Raw + Metrics"], index=0, key=f"display_{league_name}")
    show_bars = st.toggle("Distribution bars", value=True, key=f"bars_{league_name}")
    show_rank = st.toggle("Show rank", value=True, key=f"rank_{league_name}")

    if not selected_stats:
        st.warning("Select at least one stat.")
        st.stop()

# ─────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────
df_filtered = df.copy()

if name_filter:
    df_filtered = df_filtered[df_filtered["displayName"].str.contains(name_filter, case=False, na=False)]

if squad != "All Clubs":
    df_filtered = df_filtered[df_filtered["squadName"] == squad]

if pos_group != "All Positions":
    pos_map = {
        "Defenders": ["DEFENDER", "BACK"],
        "Midfielders": ["MIDFIELD"],
        "Forwards": ["FORWARD", "WINGER"],
        "Goalkeepers": ["GOALKEEPER"],
    }
    tokens = pos_map[pos_group]
    mask = pd.Series(False, index=df_filtered.index)
    for t in tokens:
        mask |= df_filtered["positions"].astype(str).str.contains(t, case=False, na=False)
    df_filtered = df_filtered[mask]

if selected_posabbr:
    pos_mask = pd.Series(False, index=df_filtered.index)
    s = df_filtered.get("posAbbr", "").fillna("").astype(str)
    for p in selected_posabbr:
        pos_mask |= s.str.contains(rf"\b{p}\b", case=False, na=False)
    df_filtered = df_filtered[pos_mask]


# ─────────────────────────────────────────────
# Build display dataframe (for TABLE view)
# ─────────────────────────────────────────────
base_cols = ["displayName", "squadName", "posAbbr"]
cols = base_cols.copy()

metric_suffix = "_pct" if metric_mode == "Percentile" else "_z"
metric_label = "(PCT)" if metric_mode == "Percentile" else "(Z)"

def metric_col(stat: str) -> str:
    return f"{stat}{metric_suffix}"

if display_mode == "Metrics only":
    for stat in selected_stats:
        mc = metric_col(stat)
        if mc in df_filtered.columns:
            cols.append(mc)
elif display_mode == "Raw only":
    for stat in selected_stats:
        if stat in df_filtered.columns:
            cols.append(stat)
else:
    for stat in selected_stats:
        if stat in df_filtered.columns:
            cols.append(stat)
        mc = metric_col(stat)
        if mc in df_filtered.columns:
            cols.append(mc)

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

# Sort by first metric column (if present)
sort_col = None
for c in df_display.columns:
    if c.endswith(metric_label):
        sort_col = c
        break
if sort_col is None and len(df_display.columns) > 3:
    sort_col = df_display.columns[3]
if sort_col:
    df_display = df_display.sort_values(sort_col, ascending=False, na_position="last")

if show_rank and sort_col:
    df_display.insert(0, "Rank", df_display[sort_col].rank(ascending=False, method="min").astype("Int64"))

# Style table
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
    styled = styled.applymap(
        lambda v: f"color:{TEXT_DIM}; font-weight:950; background-color: transparent;",
        subset=["Rank"],
    )

if show_bars:
    for c in df_display.columns:
        if c.endswith("(PCT)"):
            styled = styled.applymap(pct_cell_style, subset=[c])
        if c.endswith("(Z)"):
            styled = styled.applymap(z_cell_style, subset=[c])

# ─────────────────────────────────────────────
# RIGHT: Tabs (Table / Player Profile)
# ─────────────────────────────────────────────
with right:
    tabs = st.tabs(["📊 Table", "👤 Player Profile"])

    # ---------- TABLE TAB ----------
    with tabs[0]:
        st.markdown(
            f"""
<div style="display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.2rem; margin-bottom:.4rem;">
  <div class="badge"><span class="dim">League</span> {league_name}</div>
  <div class="badge"><span class="dim">Players</span> {len(df_filtered)}</div>
  <div class="badge"><span class="dim">Teams</span> {df_filtered["squadName"].nunique()}</div>
  <div class="badge"><span class="dim">Stats</span> {len(selected_stats)}</div>
  <div class="badge"><span class="dim">Mode</span> {metric_mode}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(f'<div class="table-wrap">{styled.to_html()}</div>', unsafe_allow_html=True)

        # Export
        csv = df_display.to_csv(index=False).encode("utf-8")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_display.to_excel(writer, index=False)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card'><h3 style='margin:0 0 .35rem 0;'>Export</h3>"
            "<div class='small-muted'>Exports the current view (filters + columns shown).</div></div>",
            unsafe_allow_html=True,
        )
        e1, e2 = st.columns([1, 1])
        with e1:
            st.download_button(
                "⬇ CSV",
                csv,
                f"{league_name}_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True,
            )
        with e2:
            st.download_button(
                "⬇ Excel",
                buffer.getvalue(),
                f"{league_name}_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    # ---------- PROFILE TAB ----------
    with tabs[1]:
        if len(df_filtered) == 0:
            st.warning("No players match your current filters.")
            st.stop()

        # Player picker respects filters
        player_list = df_filtered["displayName"].dropna().astype(str).sort_values().unique().tolist()
        default_player = player_list[0]
        player = st.selectbox("Select player", player_list, index=0, key=f"profile_player_{league_name}")

        row = df_filtered[df_filtered["displayName"] == player].iloc[0]

        player_club = row.get("squadName", "")
        player_pos = row.get("posAbbr", row.get("positions", ""))
        player_age = calc_age(row.get("birthdate", pd.NaT))
        foot = row.get("leg", "")

        # Header card
        st.markdown(
            f"""
<div class="card">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
    <div>
      <div class="badge"><span style="color:{PLAYER_COL}; font-weight:950;">{html_escape(player)}</span></div>
      <div style="margin-top:.55rem; display:flex; gap:.5rem; flex-wrap:wrap;">
        <div class="badge"><span class="dim">Club</span> {html_escape(player_club)}</div>
        <div class="badge"><span class="dim">Pos</span> {html_escape(player_pos)}</div>
        {"<div class='badge'><span class='dim'>Age</span> " + html_escape(player_age) + "</div>" if player_age else ""}
        {"<div class='badge'><span class='dim'>Foot</span> " + html_escape(foot) + "</div>" if str(foot).strip() else ""}
      </div>
      <div class="small-muted" style="margin-top:.55rem;">
        Profile shows selected stats + top attributes + similar players (based on z-scores of selected stats).
      </div>
    </div>
    <div class="badge"><span class="dim">League</span> {html_escape(league_name)}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Prepare metric columns for selected stats
        metric_suffix = "_pct" if metric_mode == "Percentile" else "_z"
        metric_is_pct = (metric_mode == "Percentile")

        # Cards: Selected stats snapshot
        # We will display: Stat name + bar (pct or z) + raw value underneath (always)
        rows_html = []
        for stat in selected_stats[:14]:  # keep readable; you can increase
            raw = row.get(stat, np.nan)
            m = row.get(f"{stat}{metric_suffix}", np.nan)

            stat_label = html_escape(clean_stat_name(stat))
            raw_txt = "-" if pd.isna(raw) else (f"{raw:.2f}" if isinstance(raw, (int, float, np.floating)) else html_escape(raw))

            bar = pct_bar(m) if metric_is_pct else z_bar(m)
            sub = f"<div class='metric-sub'>Raw: {raw_txt}</div>"

            rows_html.append(
                f"""
<div class="metric-row">
  <div>
    <div class="metric-name">{stat_label}</div>
    {sub}
  </div>
  <div>{bar}</div>
</div>
"""
            )

        snapshot_html = "\n".join(rows_html) if rows_html else "<div class='small-muted'>No stats selected.</div>"

        # Top attributes (best percentiles across ALL KPIs)
        # Always uses percentiles for "top attributes" (even if you view z-score)
        pct_cols = [f"{k}_pct" for k in kpis if f"{k}_pct" in df_filtered.columns]
        top_attrs = []
        for k in kpis:
            pc = f"{k}_pct"
            if pc in df_filtered.columns:
                top_attrs.append((k, row.get(pc, np.nan)))
        top_attrs = [x for x in top_attrs if not pd.isna(x[1])]
        top_attrs.sort(key=lambda x: x[1], reverse=True)
        top_attrs = top_attrs[:10]

        top_html_rows = []
        for k, p in top_attrs:
            top_html_rows.append(
                f"""
<div class="metric-row">
  <div>
    <div class="metric-name">{html_escape(clean_stat_name(k))}</div>
    <div class="metric-sub">Percentile</div>
  </div>
  <div>{pct_bar(p)}</div>
</div>
"""
            )
        top_html = "\n".join(top_html_rows) if top_html_rows else "<div class='small-muted'>No percentile data available.</div>"

        # Similar players (cosine similarity on selected stats z-scores)
        sim_table = None
        z_cols = []
        for s in selected_stats:
            zc = f"{s}_z"
            if zc in df_filtered.columns:
                z_cols.append(zc)

        if len(z_cols) >= 2:
            mat = df_filtered[["displayName", "squadName", "posAbbr"] + z_cols].copy()
            X = mat[z_cols].to_numpy(dtype=float)
            # impute NaNs with column means
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
            # normalize
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            Xn = X / norms

            # target index
            target_idx = mat.index[mat["displayName"] == player]
            if len(target_idx) > 0:
                ti = mat.index.get_loc(target_idx[0])
                sims = Xn @ Xn[ti].reshape(-1, 1)
                sims = sims.ravel()

                mat2 = mat[["displayName", "squadName", "posAbbr"]].copy()
                mat2["similarity"] = sims
                mat2 = mat2.sort_values("similarity", ascending=False)
                mat2 = mat2[mat2["displayName"] != player].head(12)
                mat2["similarity"] = (mat2["similarity"] * 100).round(1)
                sim_table = mat2.rename(
                    columns={"displayName": "Player", "squadName": "Club", "posAbbr": "Pos", "similarity": "Similarity %"}
                )

        # Layout: two columns of profile content
        cA, cB = st.columns([1, 1], gap="large")

        with cA:
            st.markdown(
                "<div class='card'><h3 style='margin:0 0 .4rem 0;'>Selected stats snapshot</h3>"
                "<div class='small-muted'>Bars reflect your current metric mode (Percentile or Z-score).</div>"
                f"<div style='margin-top:.6rem;'>{snapshot_html}</div></div>",
                unsafe_allow_html=True,
            )

        with cB:
            st.markdown(
                "<div class='card'><h3 style='margin:0 0 .4rem 0;'>Top attributes (percentiles)</h3>"
                "<div class='small-muted'>Best percentiles across all available KPIs.</div>"
                f"<div style='margin-top:.6rem;'>{top_html}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

        # Similar players table
        st.markdown(
            "<div class='card'><h3 style='margin:0 0 .4rem 0;'>Similar players</h3>"
            "<div class='small-muted'>Cosine similarity using z-scores of your selected stats (within current filters).</div></div>",
            unsafe_allow_html=True,
        )
        if sim_table is None:
            st.info("Select at least 2 stats (with z-scores available) to compute similar players.")
        else:
            st.dataframe(sim_table, use_container_width=True, hide_index=True)
