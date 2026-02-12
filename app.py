"""
IMPECT Stats Table — StatsBomb Edition
=====================================
- Dark StatsBomb-style UI
- Percentile & Z-score toggle
- Distribution bars in cells
- Everything per 90
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# -------------------------
# Config
# -------------------------
st.set_page_config(
    page_title="IMPECT Stats | KKD",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "Keuken Kampioen Divisie.xlsx"

# -------------------------
# COLOUR PALETTE (StatsBomb-esque)
# -------------------------
BG = "#1A1A1A"
RADAR_BG = "#222222"
GRID_COLOR = "#333333"
PLAYER_COL = "#F5C518"      # Gold
LEAGUE_COL = "#5A5A5A"
TEXT_LIGHT = "#E0E0E0"
TEXT_DIM = "#888888"
TABLE_ROW_ALT = "#2A2A2A"
TABLE_BORDER = "#3A3A3A"
PCT_HIGH = "#4CAF50"        # Green
PCT_MID = "#F5C518"         # Yellow
PCT_LOW = "#E53935"         # Red
ACCENT_LINE = "#F5C518"

INVERTED = ["foul", "lost", "unsuccessful", "failed", "off target", "red", "yellow"]

def is_inverted(col):
    return any(x in col.lower() for x in INVERTED)

# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_data(path):
    standard = pd.read_excel(path, sheet_name="Standard")
    xg = pd.read_excel(path, sheet_name="xG")

    merged = standard.merge(
        xg.drop(columns=[c for c in xg.columns if c in standard.columns and c != "playerId"]),
        on="playerId",
        how="left"
    )

    merged["displayName"] = merged["commonname"].fillna("").str.strip()
    fallback = merged["firstname"].fillna("") + " " + merged["lastname"].fillna("")
    merged["displayName"] = np.where(merged["displayName"] == "", fallback, merged["displayName"])

    return merged

def get_kpis(df):
    base = {
        "playerId","displayName","squadName","positions",
        "firstname","lastname","commonname"
    }
    return [c for c in df.columns if c not in base and pd.api.types.is_numeric_dtype(df[c])]

def calc_percentiles(df, kpis):
    for col in kpis:
        v = pd.to_numeric(df[col], errors="coerce")
        pct = v.rank(pct=True) * 100
        df[f"{col}_pct"] = 100 - pct if is_inverted(col) else pct
    return df

def calc_zscores(df, kpis):
    for col in kpis:
        x = pd.to_numeric(df[col], errors="coerce")
        if is_inverted(col):
            x = -x
        df[f"{col}_z"] = (x - x.mean()) / x.std(ddof=0)
    return df

# -------------------------
# Styling helpers
# -------------------------
def pct_style(v):
    if pd.isna(v):
        return f"color:{TEXT_DIM};"
    v = float(v)
    if v >= 85:
        col = PCT_HIGH
    elif v <= 25:
        col = PCT_LOW
    else:
        col = PCT_MID
    return f"""
    background: linear-gradient(to right, {col} {v}%, transparent {v}%);
    color: {TEXT_LIGHT};
    font-weight: 700;
    """

def z_style(v):
    if pd.isna(v):
        return f"color:{TEXT_DIM};"
    z = max(-3, min(3, float(v)))
    mid = 50
    pos = (z + 3) / 6 * 100
    col = PCT_HIGH if z >= 0 else PCT_LOW
    return f"""
    background:
      linear-gradient(to right,
        transparent 0%,
        transparent {min(mid,pos)}%,
        {col} {min(mid,pos)}%,
        {col} {max(mid,pos)}%,
        transparent {max(mid,pos)}%,
        transparent 100%);
    color:{TEXT_LIGHT};
    font-weight:700;
    """

# -------------------------
# Global CSS (StatsBomb)
# -------------------------
st.markdown(f"""
<style>
html, body {{
    background:{BG};
    color:{TEXT_LIGHT};
}}

.block-container {{
    max-width: 1450px;
    padding-top: 1.2rem;
}}

#MainMenu, footer, header {{ display:none; }}

section[data-testid="stSidebar"] {{
    background:{RADAR_BG};
    border-right:1px solid {TABLE_BORDER};
}}

.stDataFrame {{
    border:1px solid {TABLE_BORDER};
    background:{BG};
}}

.dataframe thead th {{
    background:{RADAR_BG};
    color:{TEXT_LIGHT};
    border-bottom:1px solid {TABLE_BORDER};
    font-size:11px;
    text-transform:uppercase;
}}

.dataframe tbody td {{
    border-bottom:1px solid {TABLE_BORDER};
    color:{TEXT_LIGHT};
    font-size:13px;
}}

.dataframe tbody tr:nth-child(even) {{
    background:{TABLE_ROW_ALT};
}}

.dataframe tbody tr:hover {{
    background:#333333;
}}

.stDownloadButton button {{
    background:{ACCENT_LINE};
    color:#000;
    font-weight:800;
    border:none;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load + compute
# -------------------------
df = load_data(DATA_FILE)
kpis = get_kpis(df)
df = calc_percentiles(df, kpis)
df = calc_zscores(df, kpis)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("### Filters")
    name = st.text_input("Player")
    metric_mode = st.radio("Metric", ["Percentile", "Z-score"])
    stats = st.multiselect("Stats", kpis, default=kpis[:6])

# -------------------------
# Filter
# -------------------------
if name:
    df = df[df["displayName"].str.contains(name, case=False)]

# -------------------------
# Build table
# -------------------------
cols = ["displayName","squadName","positions"]
for s in stats:
    cols.append(f"{s}_pct" if metric_mode=="Percentile" else f"{s}_z")

df_show = df[cols].copy()

rename = {}
for c in df_show.columns:
    if c.endswith("_pct"):
        rename[c] = c.replace("_pct"," (PCT)")
    elif c.endswith("_z"):
        rename[c] = c.replace("_z"," (Z)")
df_show = df_show.rename(columns=rename)

# -------------------------
# Style
# -------------------------
styler = df_show.style

for c in df_show.columns:
    if c.endswith("(PCT)"):
        styler = styler.applymap(pct_style, subset=[c])
    if c.endswith("(Z)"):
        styler = styler.applymap(z_style, subset=[c])

styler = styler.format(na_rep="-")

# -------------------------
# Display
# -------------------------
st.dataframe(styler, use_container_width=True, height=650, hide_index=True)

# -------------------------
# Export
# -------------------------
csv = df_show.to_csv(index=False).encode()
st.download_button("⬇ CSV", csv, "kkd_stats.csv")
