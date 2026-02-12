"""
IMPECT Stats Table — StatsBomb Edition (Full, Consistent, Readable)
===================================================================
- Dark StatsBomb-esque palette (your exact colors) applied everywhere
- Filters: player search, squad, position group, category, stat selection
- Display mode: Percentiles / Raw values / Both
- Metric mode: Percentile / Z-score (for the metric columns)
- Distribution bars: Pct fill + Z diverging with center line
- Per-90 assumed (no per-90 toggle)
- Table font + colors forced consistently (no random dark text)
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IMPECT Stats | KKD",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "Keuken Kampioen Divisie.xlsx"

# ─────────────────────────────────────────────
# COLOUR PALETTE (StatsBomb-esque) — USER PROVIDED
# ─────────────────────────────────────────────
BG = "#1A1A1A"
RADAR_BG = "#222222"
GRID_COLOR = "#333333"
PLAYER_COL = "#F5C518"      # Gold for player / accents
LEAGUE_COL = "#5A5A5A"      # Grey for baseline
TEXT_LIGHT = "#E0E0E0"
TEXT_DIM = "#888888"
TABLE_ROW_ALT = "#2A2A2A"
TABLE_BORDER = "#3A3A3A"
PCT_HIGH = "#4CAF50"        # Green for high percentile
PCT_MID = "#F5C518"         # Yellow for mid
PCT_LOW = "#E53935"         # Red for low
ACCENT_LINE = "#F5C518"

# Metrics where lower is better (auto-invert in pct + z)
INVERTED = ["foul", "lost", "unsuccessful", "failed", "off target", "red", "yellow"]


def is_inverted(col: str) -> bool:
    c = (col or "").lower()
    return any(x in c for x in INVERTED)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clean_stat_name(stat: str) -> str:
    return stat.split(" (")[0] if " (" in stat else stat


# ─────────────────────────────────────────────
# Data loading
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

    # Merge only non-base KPI cols from xG
    xg_kpis = [c for c in xg.columns if c not in base and c != "playerId"]
    merged = standard.merge(xg[["playerId"] + xg_kpis], on="playerId", how="left")

    # Display name
    cn = merged.get("commonname", "").fillna("").astype(str).str.strip()
    fallback = (
        merged.get("firstname", "").fillna("").astype(str).str.strip()
        + " "
        + merged.get("lastname", "").fillna("").astype(str).str.strip()
    ).str.strip()
    merged["displayName"] = np.where(cn == "", fallback, cn)

    return merged


def get_kpis(df: pd.DataFrame) -> list[str]:
    base = {
        "iterationId", "squadId", "squadName", "playerId", "positions",
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
    """
    Z-score per KPI, after flipping inverted metrics so higher is always better.
    """
    for col in kpis:
        x = pd.to_numeric(df[col], errors="coerce")
        if is_inverted(col):
            x = -x
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=0)
        df[f"{col}_z"] = np.nan if (sd == 0 or pd.isna(sd)) else (x - mu) / sd
    return df


# ─────────────────────────────────────────────
# Cell styles (bars) — table-friendly (doesn't break row bg/font)
# ─────────────────────────────────────────────
def pct_cell_style(val):
    """
    Percentiles:
    - fill bar 0→100
    - color band: green high / yellow mid / red low
    """
    if pd.isna(val):
        return f"color:{TEXT_DIM}; background-color: transparent;"

    v = clamp(float(val), 0.0, 100.0)

    if v >= 85:
        col = PCT_HIGH
        weight = 900
    elif v <= 25:
        col = PCT_LOW
        weight = 900
    else:
        col = PCT_MID
        weight = 800

    return f"""
      background-color: transparent;
      background-image: linear-gradient(to right, {col} {v:.1f}%, transparent {v:.1f}%);
      color: {TEXT_LIGHT};
      font-weight: {weight};
    """


def z_cell_style(val):
    """
    Z-scores:
    - diverging bar around 0 (center line)
    - green positive, red negative
    - clamp to [-3, 3]
    """
    if pd.isna(val):
        return f"color:{TEXT_DIM}; background-color: transparent;"

    z = clamp(float(val), -3.0, 3.0)
    pos = (z + 3.0) / 6.0 * 100.0
    mid = 50.0

    col = PCT_HIGH if z >= 0 else PCT_LOW
    weight = 900 if abs(z) >= 1.25 else 800

    center = f"""
      linear-gradient(to right,
        transparent 49.4%,
        {LEAGUE_COL} 49.4%,
        {LEAGUE_COL} 50.6%,
        transparent 50.6%)
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

    return f"""
      background-color: transparent;
      background-image: {center}, {bar};
      color: {TEXT_LIGHT};
      font-weight: {weight};
    """


# ─────────────────────────────────────────────
# Global CSS (forces table font + colors; matches sidebar + app)
# ─────────────────────────────────────────────
st.markdown(
    f"""
<style>
/* App background */
html, body, [data-testid="stAppViewContainer"] {{
  background: {BG} !important;
  color: {TEXT_LIGHT} !important;
}}
/* Remove Streamlit chrome */
#MainMenu, footer, header, .stDeployButton {{ visibility: hidden !important; display:none !important; }}

/* Layout */
.block-container {{
  max-width: 1500px !important;
  padding-top: 1.1rem !important;
  padding-bottom: 2.2rem !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: {RADAR_BG} !important;
  border-right: 1px solid {TABLE_BORDER} !important;
}}
section[data-testid="stSidebar"] * {{
  color: {TEXT_LIGHT} !important;
}}

/* Typography */
html, body, [class*="css"], .stMarkdown, .stText, .stCaption {{
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial !important;
}}

/* Inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {{
  background: {BG} !important;
  color: {TEXT_LIGHT} !important;
  border: 1px solid {TABLE_BORDER} !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}}
section[data-testid="stSidebar"] input:focus,
section[data-testid="stSidebar"] select:focus,
section[data-testid="stSidebar"] textarea:focus {{
  border-color: {ACCENT_LINE} !important;
  box-shadow: 0 0 0 3px rgba(245,197,24,.18) !important;
}}

/* Multiselect chips */
section[data-testid="stSidebar"] [data-baseweb="tag"] {{
  background: {BG} !important;
  border: 1px solid {TABLE_BORDER} !important;
  color: {TEXT_LIGHT} !important;
}}

/* Cards / badges */
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
.badge .dim {{
  color: {TEXT_DIM};
  font-weight: 700;
}}
.small-muted {{
  color: {TEXT_DIM} !important;
}}

/* Dividers */
hr {{ border-color: {TABLE_BORDER} !important; }}

/* ─────────────────────────────
   TABLE: force StatsBomb font + colors
   ───────────────────────────── */
.stDataFrame, .stDataFrame * {{
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial !important;
}}
/* Wrapper */
.stDataFrame {{
  background: {BG} !important;
  border: 1px solid {TABLE_BORDER} !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}}
/* Table & cells */
.stDataFrame .dataframe,
.stDataFrame .dataframe th,
.stDataFrame .dataframe td {{
  color: {TEXT_LIGHT} !important;
}}
/* Header */
.stDataFrame .dataframe thead th {{
  background: {RADAR_BG} !important;
  color: {TEXT_LIGHT} !important;
  border-bottom: 1px solid {TABLE_BORDER} !important;
  border-right: 1px solid {TABLE_BORDER} !important;
  font-size: 11px !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
  padding: 12px 12px !important;
  position: sticky !important;
  top: 0 !important;
  z-index: 2 !important;
}}
.stDataFrame .dataframe thead th:last-child {{
  border-right: none !important;
}}
/* Rows */
.stDataFrame .dataframe tbody tr {{
  background: {BG} !important;
}}
.stDataFrame .dataframe tbody tr:nth-child(even) {{
  background: {TABLE_ROW_ALT} !important;
}}
.stDataFrame .dataframe tbody tr:hover {{
  background: {GRID_COLOR} !important;
}}
/* Body cells */
.stDataFrame .dataframe tbody td {{
  background: transparent !important; /* inherit row bg; allow bars via background-image */
  border-bottom: 1px solid {TABLE_BORDER} !important;
  border-right: 1px solid {TABLE_BORDER} !important;
  font-size: 13px !important;
  padding: 10px 12px !important;
  text-align: center !important;
  font-variant-numeric: tabular-nums !important;
}}
.stDataFrame .dataframe tbody td:last-child {{
  border-right: none !important;
}}
/* Left columns */
.stDataFrame .dataframe tbody td:nth-child(1) {{
  text-align: left !important;
  font-weight: 900 !important;
  color: {TEXT_LIGHT} !important;
}}
.stDataFrame .dataframe tbody td:nth-child(2),
.stDataFrame .dataframe tbody td:nth-child(3) {{
  text-align: left !important;
  font-weight: 750 !important;
  color: {TEXT_DIM} !important;
}}

/* Download buttons */
.stDownloadButton button {{
  background: {ACCENT_LINE} !important;
  color: #000 !important;
  font-weight: 950 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
}}
.stDownloadButton button:hover {{
  filter: brightness(0.95) !important;
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
    <div class="badge"><span>IMPECT</span> <span class="dim">•</span> <span>Keuken Kampioen Divisie</span></div>
    <h1 style="margin:.45rem 0 0 0; font-size: 1.9rem;">Player Stats Explorer</h1>
    <div class="small-muted">StatsBomb-style table • per 90 • percentiles + z-scores • bars</div>
  </div>
  <div class="badge"><span class="dim">Season</span> 25/26</div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Load + compute
# ─────────────────────────────────────────────
try:
    df = load_data(DATA_FILE)
    kpis = get_kpis(df)
    df = calc_percentiles(df, kpis)
    df = calc_zscores(df, kpis)
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")

    name_filter = st.text_input("Player search", placeholder="Type player name…")

    st.divider()

    squads = ["All Squads"] + sorted(df["squadName"].dropna().unique().tolist())
    squad = st.selectbox("Squad", squads)

    st.divider()

    pos_group = st.selectbox("Position group", ["All Positions", "Defenders", "Midfielders", "Forwards"])

    st.divider()

    categories = {
        "⚽ Goals & Assists": ["Goals", "Assists", "Pre Assist", "Shot-Creating Actions", "Shot xG from Passes"],
        "🎯 Shooting": ["Total Shots", "Total Shots On Target", "Shot-based xG", "Post-Shot xG"],
        "📤 Passing": ["Successful Passes", "Unsuccessful Passes", "Progressive passes", "Pass Accuracy"],
        "🤼 Duels": ["Won Ground Duels", "Lost Ground Duels", "Won Aerial Duels", "Lost Aerial Duels"],
        "📈 xG Metrics": ["Shot-based xG", "Post-Shot xG", "Expected Goal Assists", "Expected Shot Assists", "Packing non-shot-based xG"],
    }
    selected_cat = st.selectbox("Stat category", list(categories.keys()))
    keywords = categories[selected_cat]

    st.divider()

    display_mode = st.selectbox("Display mode", ["Percentiles", "Raw values", "Both"], index=0)
    metric_mode = st.selectbox("Metric mode", ["Percentile", "Z-score"], index=0)

    show_bars = st.toggle("Show distribution bars", value=True)
    show_rank = st.toggle("Show rank", value=True)

    st.divider()

    matching = [k for k in kpis if any(kw in k for kw in keywords)]
    if not matching:
        matching = kpis[:]

    # Keep list manageable; multiselect itself supports search
    stats_options = matching[:40] if len(matching) > 40 else matching

    selected_stats = st.multiselect(
        "Select stats",
        options=stats_options,
        default=stats_options[:8] if len(stats_options) >= 8 else stats_options,
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
<div style="margin-top: 1rem; display:flex; gap:.5rem; flex-wrap:wrap;">
  <div class="badge"><span class="dim">Players</span> {len(df_filtered)}</div>
  <div class="badge"><span class="dim">Stats</span> {len(selected_stats)}</div>
  <div class="badge"><span class="dim">Teams</span> {df_filtered["squadName"].nunique()}</div>
  <div class="badge"><span class="dim">Display</span> {display_mode}</div>
  <div class="badge"><span class="dim">Metric</span> {metric_mode}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Build display dataframe
# ─────────────────────────────────────────────
base_cols = ["displayName", "squadName", "positions"]
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

else:  # Both
    for stat in selected_stats:
        if stat in df_filtered.columns:
            cols.append(stat)
        mcol = f"{stat}{metric_suffix}"
        if mcol in df_filtered.columns:
            cols.append(mcol)

cols = [c for c in cols if c in df_filtered.columns]
df_display = df_filtered[cols].copy()

# Rename columns
rename_map = {}
seen = set()
for col in df_display.columns:
    if col in base_cols:
        continue

    if col.endswith("_pct"):
        base = clean_stat_name(col[:-4])
        nice = f"{base} {metric_label}"
    elif col.endswith("_z"):
        base = clean_stat_name(col[:-2])
        nice = f"{base} {metric_label}"
    else:
        nice = clean_stat_name(col)

    orig = nice
    i = 1
    while nice in seen:
        i += 1
        nice = f"{orig} [{i}]"
    seen.add(nice)
    rename_map[col] = nice

df_display = df_display.rename(columns=rename_map)

# Sort by first metric column if present, else first stat column
sort_col = None
for c in df_display.columns:
    if c.endswith(metric_label):
        sort_col = c
        break
if sort_col is None and len(df_display.columns) > len(base_cols):
    sort_col = df_display.columns[len(base_cols)]

if sort_col:
    df_display = df_display.sort_values(sort_col, ascending=False, na_position="last")

# Rank
if show_rank and sort_col:
    df_display.insert(0, "Rank", df_display[sort_col].rank(ascending=False, method="min").astype("Int64"))

# ─────────────────────────────────────────────
# Style table
# ─────────────────────────────────────────────
styled = df_display.style

# Format
fmt = {}
for c in df_display.columns:
    if c in ["Rank"] + base_cols:
        continue
    if c.endswith("(PCT)"):
        fmt[c] = "{:.0f}"
    elif c.endswith("(Z)"):
        fmt[c] = "{:+.2f}"
    else:
        fmt[c] = "{:.2f}"
styled = styled.format(fmt, na_rep="-")

# Rank style
if "Rank" in df_display.columns:
    styled = styled.applymap(
        lambda v: f"color:{TEXT_DIM}; font-weight:950; background-color: transparent;",
        subset=["Rank"],
    )

# Metric styles
for c in df_display.columns:
    if c.endswith("(PCT)"):
        if show_bars:
            styled = styled.applymap(pct_cell_style, subset=[c])
        else:
            styled = styled.applymap(
                lambda v: f"color:{PCT_HIGH}; font-weight:950;" if pd.notna(v) and float(v) >= 85
                else (f"color:{PCT_LOW}; font-weight:950;" if pd.notna(v) and float(v) <= 25 else f"color:{PCT_MID}; font-weight:900;"),
                subset=[c],
            )

    if c.endswith("(Z)"):
        if show_bars:
            styled = styled.applymap(z_cell_style, subset=[c])
        else:
            styled = styled.applymap(
                lambda v: f"color:{PCT_HIGH}; font-weight:950;" if pd.notna(v) and float(v) >= 1.25
                else (f"color:{PCT_LOW}; font-weight:950;" if pd.notna(v) and float(v) <= -1.25 else f"color:{PCT_MID}; font-weight:900;"),
                subset=[c],
            )

# ─────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────
st.markdown(
    f"""
<div style="margin-top: 1.1rem; margin-bottom: .5rem; display:flex; align-items:center; justify-content:space-between; gap:1rem;">
  <h3 style="margin:0;">Player Table</h3>
  <div class="badge"><span class="dim">Sorted by</span> {sort_col if sort_col else "—"}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.dataframe(styled, use_container_width=True, height=700, hide_index=True)

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
        f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True,
    )

with c2:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_display.to_excel(writer, index=False)
    st.download_button(
        "⬇ Excel",
        buffer.getvalue(),
        f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
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
  <div style="margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap;">
    <div class="badge" style="border-color:{PCT_HIGH};"><span style="color:{PCT_HIGH}; font-weight:950;">■</span> High</div>
    <div class="badge" style="border-color:{PCT_MID};"><span style="color:{PCT_MID}; font-weight:950;">■</span> Mid</div>
    <div class="badge" style="border-color:{PCT_LOW};"><span style="color:{PCT_LOW}; font-weight:950;">■</span> Low</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
