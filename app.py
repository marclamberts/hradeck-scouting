"""
IMPECT Stats Table — Editorial Edition
======================================
Data-first (StatsHub / DunksAndThrees / 538-ish) Streamlit table:
- Clean editorial theme (neutral, minimal gradients/shadows)
- Subtle percentile highlighting (only extremes pop)
- Display mode: Percentiles / Raw / Both
- Optional rank column (based on first selected stat)
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

# Metrics where lower is better
INVERTED = ["foul", "lost", "unsuccessful", "failed", "off target", "red", "yellow"]


def is_inverted(col: str) -> bool:
    c = (col or "").lower()
    return any(inv in c for inv in INVERTED)


# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    standard = pd.read_excel(file_path, sheet_name="Standard")
    xg = pd.read_excel(file_path, sheet_name="xG")

    base = [
        "iterationId",
        "squadId",
        "squadName",
        "playerId",
        "positions",
        "commonname",
        "firstname",
        "lastname",
        "birthdate",
        "birthplace",
        "leg",
        "countryIds",
        "gender",
        "season",
        "dataVersion",
        "lastChangeTimestamp",
        "competition_name",
        "competition_type",
        "competition_gender",
    ]

    xg_kpis = [c for c in xg.columns if c not in base]
    merged = standard.merge(xg[["playerId"] + xg_kpis], on="playerId", how="left")

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
        "iterationId",
        "squadId",
        "squadName",
        "playerId",
        "positions",
        "commonname",
        "firstname",
        "lastname",
        "birthdate",
        "birthplace",
        "leg",
        "countryIds",
        "gender",
        "season",
        "dataVersion",
        "lastChangeTimestamp",
        "competition_name",
        "competition_type",
        "competition_gender",
        "displayName",
    }
    return [
        c
        for c in df.columns
        if c not in base and pd.api.types.is_numeric_dtype(df[c])
    ]


def calc_percentiles(df: pd.DataFrame, kpis: list[str]) -> pd.DataFrame:
    # percentile rank per metric; invert if "lower is better"
    for col in kpis:
        vals = pd.to_numeric(df[col], errors="coerce")
        pct = vals.rank(pct=True, method="average") * 100
        df[f"{col}_pct"] = 100 - pct if is_inverted(col) else pct
    return df


# -------------------------
# Styling (editorial + subtle highlights)
# -------------------------
def metric_style(val):
    """
    Subtle “editorial” highlight:
    - most cells transparent
    - strong positive: soft blue wash
    - strong negative: soft red wash
    """
    if pd.isna(val):
        return "background-color: transparent; color: #111827;"
    v = float(val)

    if v >= 90:
        return "background-color: rgba(37,99,235,.20); color: #0b1220; font-weight: 750;"
    if v >= 75:
        return "background-color: rgba(37,99,235,.12); color: #0b1220; font-weight: 650;"
    if v >= 60:
        return "background-color: rgba(37,99,235,.07); color: #111827;"
    if v <= 10:
        return "background-color: rgba(185,28,28,.16); color: #0b1220; font-weight: 700;"
    if v <= 25:
        return "background-color: rgba(185,28,28,.08); color: #111827;"
    return "background-color: transparent; color: #111827;"


# -------------------------
# Global CSS theme (replace your old massive CSS block)
# -------------------------
st.markdown(
    """
<style>
/* =========================
   Editorial / Data-first UI
   ========================= */
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji" !important;
}
:root{
  --bg: #ffffff;
  --panel: #f6f7f9;
  --panel-2: #fbfbfc;
  --text: #111827;
  --muted: #6b7280;
  --border: #e5e7eb;
  --accent: #2563eb;
}

/* Remove Streamlit chrome */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; display:none !important; }

/* Layout */
.block-container{
  max-width: 1400px !important;
  padding-top: 1.25rem !important;
  padding-bottom: 2.5rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: var(--panel) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container{
  padding-top: 1.25rem !important;
}

/* Inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea{
  background: #fff !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] input:focus,
section[data-testid="stSidebar"] select:focus{
  border-color: rgba(37, 99, 235, .65) !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, .12) !important;
}

/* Headings */
h1, h2, h3{
  color: var(--text) !important;
  letter-spacing: -0.01em !important;
}
.small-muted{ color: var(--muted); font-size: 0.9rem; }

/* Cards */
.card{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
}
.badge{
  display:inline-flex;
  align-items:center;
  gap:.5rem;
  padding:.35rem .6rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: .8rem;
  color: var(--muted);
  background: var(--panel-2);
}

/* Dataframe wrapper */
.stDataFrame{
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
  overflow: hidden !important;
}

/* Pandas styler table */
.dataframe{
  border-collapse: separate !important;
  border-spacing: 0 !important;
  width: 100% !important;
}

/* Header: clean, not glossy */
.dataframe thead th{
  position: sticky !important;
  top: 0 !important;
  z-index: 2 !important;
  background: #0b1220 !important;
  color: #f9fafb !important;
  font-size: 11px !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
  padding: 12px 12px !important;
  border-bottom: 1px solid rgba(255,255,255,.12) !important;
  border-right: 1px solid rgba(255,255,255,.08) !important;
}
.dataframe thead th:last-child{ border-right: none !important; }

/* Body cells */
.dataframe tbody td{
  padding: 11px 12px !important;
  border-bottom: 1px solid var(--border) !important;
  border-right: 1px solid #f0f1f3 !important;
  font-size: 13px !important;
  color: var(--text) !important;
  text-align: center !important;
  font-variant-numeric: tabular-nums !important;
}
.dataframe tbody td:last-child{ border-right:none !important; }

/* Zebra */
.dataframe tbody tr:nth-child(odd){ background: #fff !important; }
.dataframe tbody tr:nth-child(even){ background: var(--panel-2) !important; }

/* Hover: subtle */
.dataframe tbody tr:hover{ background: rgba(37,99,235,.06) !important; }

/* Left columns */
.dataframe tbody td:nth-child(1){
  font-weight: 650 !important;
  text-align: left !important;
  white-space: nowrap !important;
}
.dataframe tbody td:nth-child(2),
.dataframe tbody td:nth-child(3){
  text-align: left !important;
  color: var(--muted) !important;
  white-space: nowrap !important;
}

/* Slim scrollbars */
.stDataFrame ::-webkit-scrollbar{ height: 10px; width: 10px; }
.stDataFrame ::-webkit-scrollbar-track{ background: #eef0f3; }
.stDataFrame ::-webkit-scrollbar-thumb{ background: #c7cbd1; border-radius: 999px; }
.stDataFrame ::-webkit-scrollbar-thumb:hover{ background: #aeb4bd; }

/* Download buttons */
.stDownloadButton button{
  background: #0b1220 !important;
  color: #fff !important;
  border: 1px solid #0b1220 !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
  box-shadow: none !important;
  font-weight: 650 !important;
}
.stDownloadButton button:hover{
  background: #111b33 !important;
  border-color: #111b33 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Masthead
# -------------------------
st.markdown(
    """
<div class="card" style="display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;">
  <div>
    <div class="badge">IMPECT • Keuken Kampioen Divisie</div>
    <h1 style="margin:.35rem 0 0 0; font-size: 1.85rem;">Player Stats Explorer</h1>
    <div class="small-muted">Standard + xG • Percentile ranks + exports</div>
  </div>
  <div class="badge">Season 2025/26</div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Load data
# -------------------------
try:
    df = load_data(DATA_FILE)
    kpis = get_kpis(df)
    df = calc_percentiles(df, kpis)
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    name_filter = st.text_input(
        "Player name", placeholder="Type player name…", label_visibility="collapsed"
    )

    st.markdown("#### 🏟️ Squad")
    squads = ["All Squads"] + sorted(df["squadName"].dropna().unique().tolist())
    squad = st.selectbox("Squad", squads, label_visibility="collapsed")

    st.markdown("#### ⚽ Position Group")
    positions = ["All Positions", "Defenders", "Midfielders", "Forwards"]
    pos_group = st.selectbox("Position", positions, label_visibility="collapsed")

    st.markdown("#### 📊 Category")
    categories = {
        "⚽ Goals & Assists": [
            "Goals",
            "Assists",
            "Pre Assist",
            "Shot-Creating Actions",
            "Shot xG from Passes",
        ],
        "🎯 Shooting": [
            "Total Shots",
            "Total Shots On Target",
            "Shot-based xG",
            "Post-Shot xG",
        ],
        "📤 Passing": [
            "Successful Passes",
            "Unsuccessful Passes",
            "Progressive passes",
            "Pass Accuracy",
        ],
        "🤼 Duels": [
            "Won Ground Duels",
            "Lost Ground Duels",
            "Won Aerial Duels",
            "Lost Aerial Duels",
        ],
        "📈 xG Metrics": [
            "Shot-based xG",
            "Post-Shot xG",
            "Expected Goal Assists",
            "Expected Shot Assists",
            "Packing non-shot-based xG",
        ],
    }
    selected_cat = st.selectbox("Category", list(categories.keys()), label_visibility="collapsed")
    keywords = categories[selected_cat]

    st.markdown("#### 📌 Display")
    display_mode = st.selectbox(
        "Display mode",
        ["Percentiles", "Raw values", "Both"],
        index=0,
        label_visibility="collapsed",
    )
    show_rank = st.toggle("Show Rank (based on first stat)", value=True)

    st.markdown("#### 📈 Select Stats")
    matching = [k for k in kpis if any(kw in k for kw in keywords)][:30]
    selected_stats = st.multiselect(
        "Statistics",
        options=matching,
        default=matching[:8] if len(matching) >= 8 else matching,
        label_visibility="collapsed",
    )

    if not selected_stats:
        st.warning("Select at least one stat.")
        st.stop()

# -------------------------
# Apply filters
# -------------------------
df_filtered = df.copy()

if name_filter:
    df_filtered = df_filtered[
        df_filtered["displayName"].str.contains(name_filter, case=False, na=False)
    ]

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

st.markdown(
    f"""
<div style="margin-top: 1rem;">
  <span class="badge">Players: <strong>{len(df_filtered)}</strong></span>
  <span class="badge">Metrics: <strong>{len(selected_stats)}</strong></span>
  <span class="badge">Teams: <strong>{df_filtered["squadName"].nunique()}</strong></span>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Build display dataframe
# -------------------------
base_cols = ["displayName", "squadName", "positions"]
cols = base_cols.copy()

# helper to make safe/clean display names
def clean_stat_name(stat: str) -> str:
    # remove " (ID)" style suffix
    return stat.split(" (")[0] if " (" in stat else stat


# Add columns depending on display_mode
if display_mode == "Percentiles":
    for stat in selected_stats:
        pct_col = f"{stat}_pct"
        if pct_col in df_filtered.columns:
            cols.append(pct_col)

elif display_mode == "Raw values":
    for stat in selected_stats:
        if stat in df_filtered.columns:
            cols.append(stat)

else:  # Both
    for stat in selected_stats:
        if stat in df_filtered.columns:
            cols.append(stat)
        pct_col = f"{stat}_pct"
        if pct_col in df_filtered.columns:
            cols.append(pct_col)

cols = [c for c in cols if c in df_filtered.columns]
df_display = df_filtered[cols].copy()

# Rename percentile columns (and raw columns) nicely, avoiding duplicates
rename_map = {}
seen = set()
for col in df_display.columns:
    if col in base_cols:
        continue

    if col.endswith("_pct"):
        stat = col[:-4]
        nice = f"{clean_stat_name(stat)} (Pct)"
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

# Sort by first *percentile* if available; else by first stat shown
sort_col = None
if display_mode == "Percentiles":
    sort_col = df_display.columns[3] if len(df_display.columns) > 3 else None
elif display_mode == "Raw values":
    sort_col = df_display.columns[3] if len(df_display.columns) > 3 else None
else:  # Both: prefer first Pct column if present
    stat_names = [clean_stat_name(s) for s in selected_stats]
    for s in stat_names:
        candidate = f"{s} (Pct)"
        if candidate in df_display.columns:
            sort_col = candidate
            break
    if sort_col is None and len(df_display.columns) > 3:
        sort_col = df_display.columns[3]

if sort_col:
    # Descending makes sense for percentiles; for raw it depends, but acceptable default.
    df_display = df_display.sort_values(sort_col, ascending=False, na_position="last")

# Add Rank column
if show_rank and sort_col:
    df_display.insert(
        0, "Rank", df_display[sort_col].rank(ascending=False, method="min").astype("Int64")
    )

# -------------------------
# Style table
# -------------------------
styled = df_display.style

# numeric formatting
fmt = {}
for col in df_display.columns:
    if col in ["Rank"] + base_cols:
        continue
    # Percentiles as integers; raw as compact
    if "(Pct)" in col:
        fmt[col] = "{:.0f}"
    else:
        fmt[col] = "{:.2f}"

styled = styled.format(fmt, na_rep="-")

# apply subtle highlight ONLY to percentile columns
for col in df_display.columns:
    if "(Pct)" in col:
        styled = styled.applymap(metric_style, subset=[col])

# Slightly mute Rank column
if "Rank" in df_display.columns:
    styled = styled.applymap(
        lambda v: "color:#6b7280; font-weight:650; background-color: transparent;",
        subset=["Rank"],
    )

# -------------------------
# Table header
# -------------------------
st.markdown(
    """
<div style="margin-top: 1.25rem; margin-bottom: .5rem; display:flex; align-items:center; justify-content:space-between;">
  <h3 style="margin:0; font-size: 1.05rem;">Table</h3>
  <div class="badge">Tip: filter left • sort by first selected stat</div>
</div>
""",
    unsafe_allow_html=True,
)

st.dataframe(styled, use_container_width=True, height=650, hide_index=True)

# -------------------------
# Export
# -------------------------
st.markdown(
    """
<div style="margin-top: 1.25rem;" class="card">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
    <div>
      <div class="badge">Export</div>
      <div class="small-muted" style="margin-top:.35rem;">Exports current view (filtered + columns shown).</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ CSV",
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
        "⬇️ Excel",
        buffer.getvalue(),
        f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with c3:
    st.markdown(
        f"""
<div class="card" style="text-align:center;">
  <div class="small-muted" style="text-transform:uppercase; letter-spacing:.08em; font-weight:650;">Current view</div>
  <div style="display:flex; justify-content:space-around; gap:1rem; margin-top:.75rem;">
    <div>
      <div style="font-size:1.6rem; font-weight:800; color:#111827;">{len(df_display)}</div>
      <div class="small-muted">Rows</div>
    </div>
    <div>
      <div style="font-size:1.6rem; font-weight:800; color:#111827;">{len(selected_stats)}</div>
      <div class="small-muted">Stats</div>
    </div>
    <div>
      <div style="font-size:1.6rem; font-weight:800; color:#111827;">{df_display['squadName'].nunique() if 'squadName' in df_display.columns else 0}</div>
      <div class="small-muted">Teams</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# -------------------------
# Legend (compact)
# -------------------------
st.markdown(
    """
<div class="card" style="margin-top: 1.25rem;">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
    <div>
      <div class="badge">Legend</div>
      <div class="small-muted" style="margin-top:.35rem;">
        Percentile columns use subtle highlighting: top performers (blue) and bottom performers (red).
        “Lower is better” metrics are inverted automatically.
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
