"""
IMPECT Stats Table — Editorial Edition (with Z-score + Distribution Bars)
=========================================================================
- Clean, modern sports-analytics palette (appealing + readable)
- Display modes: Percentiles / Raw values / Both
- Metric scale toggle: Percentile vs Z-score
- Small in-cell distribution bars (Pct bars + Z-score diverging bars)
- Everything assumed per 90 already (no per-90 toggle)
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

# Metrics where lower is better (auto-invert so "higher is better" for Pct & Z)
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
    for col in kpis:
        vals = pd.to_numeric(df[col], errors="coerce")
        pct = vals.rank(pct=True, method="average") * 100
        df[f"{col}_pct"] = 100 - pct if is_inverted(col) else pct
    return df


def calc_zscores(df: pd.DataFrame, kpis: list[str]) -> pd.DataFrame:
    """
    Z-score per metric, after flipping 'lower is better' metrics so that
    higher Z always means better.
    """
    for col in kpis:
        x = pd.to_numeric(df[col], errors="coerce")
        # flip "bad when high" metrics
        if is_inverted(col):
            x = -x

        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=0)

        if sd == 0 or pd.isna(sd):
            df[f"{col}_z"] = np.nan
        else:
            df[f"{col}_z"] = (x - mu) / sd
    return df


# -------------------------
# Helpers: names + styling
# -------------------------
def clean_stat_name(stat: str) -> str:
    return stat.split(" (")[0] if " (" in stat else stat


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pct_cell_style(val):
    """
    Percentile cell:
    - subtle green tint at high percentiles
    - subtle red tint at low percentiles
    - distribution bar (left->right fill)
    """
    if pd.isna(val):
        return "background-color: transparent; color: var(--text);"

    v = float(val)
    v = clamp(v, 0, 100)

    # Distribution bar fill
    bar = f"""
    background-image:
      linear-gradient(to right,
        rgba(6,118,71,.18) 0%,
        rgba(6,118,71,.18) {v:.1f}%,
        transparent {v:.1f}%,
        transparent 100%);
    """

    # Subtle emphasis on extremes
    if v >= 95:
        return bar + " color: #054f31; font-weight: 800;"
    if v >= 85:
        return bar + " color: #065f46; font-weight: 700;"
    if v >= 70:
        return bar + " color: var(--text);"
    if v <= 15:
        return """
        background-image:
          linear-gradient(to right,
            rgba(180,35,24,.20) 0%,
            rgba(180,35,24,.20) """ + f"""{v:.1f}%,
            transparent """ + f"""{v:.1f}%,
            transparent 100%);
        color: #7a271a; font-weight: 700;
        """
    if v <= 30:
        return """
        background-image:
          linear-gradient(to right,
            rgba(180,35,24,.12) 0%,
            rgba(180,35,24,.12) """ + f"""{v:.1f}%,
            transparent """ + f"""{v:.1f}%,
            transparent 100%);
        color: var(--text);
        """
    return bar + " color: var(--text);"


def z_cell_style(val):
    """
    Z-score cell:
    - diverging bar around center (0)
    - center line
    - green for positive, red for negative
    """
    if pd.isna(val):
        return "background-color: transparent; color: var(--text);"

    z = float(val)
    zc = clamp(z, -3.0, 3.0)
    # map [-3..3] => [0..100]
    pos = (zc + 3.0) / 6.0 * 100.0  # 0..100
    mid = 50.0

    # Bar: negative fills from pos->mid; positive fills from mid->pos
    if pos >= mid:
        bar_layer = f"""
        linear-gradient(to right,
          transparent 0%,
          transparent {mid:.1f}%,
          rgba(6,118,71,.18) {mid:.1f}%,
          rgba(6,118,71,.18) {pos:.1f}%,
          transparent {pos:.1f}%,
          transparent 100%)
        """
        text = "color: #065f46; font-weight: 700;" if z >= 1.25 else "color: var(--text);"
    else:
        bar_layer = f"""
        linear-gradient(to right,
          transparent 0%,
          transparent {pos:.1f}%,
          rgba(180,35,24,.18) {pos:.1f}%,
          rgba(180,35,24,.18) {mid:.1f}%,
          transparent {mid:.1f}%,
          transparent 100%)
        """
        text = "color: #7a271a; font-weight: 700;" if z <= -1.25 else "color: var(--text);"

    # Center line at 0
    center_line = """
    linear-gradient(to right,
      transparent 49.4%,
      rgba(102,112,133,.35) 49.4%,
      rgba(102,112,133,.35) 50.6%,
      transparent 50.6%)
    """

    return f"""
    background-image: {center_line}, {bar_layer};
    {text}
    """


# -------------------------
# Global CSS theme (appealing + editorial)
# -------------------------
st.markdown(
    """
<style>
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji" !important;
}
:root{
  /* Surfaces */
  --bg: #fcfcfd;
  --panel: #f3f4f6;
  --panel-2: #f9fafb;

  /* Text */
  --text: #0f172a;
  --muted: #667085;

  /* Borders */
  --border: #e4e7ec;

  /* Accent */
  --accent: #1f4fd8;
  --accent-soft: rgba(31,79,216,.08);

  /* Performance */
  --good: #067647;
  --good-soft: rgba(6,118,71,.14);
  --bad: #b42318;
  --bad-soft: rgba(180,35,24,.14);
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
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(31,79,216,.15) !important;
}

/* Cards + badges */
.card{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
}
.badge{
  display:inline-flex;
  align-items:center;
  gap:.45rem;
  padding:.32rem .6rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: .78rem;
  font-weight: 600;
  color: var(--muted);
  background: var(--panel-2);
}
.small-muted{ color: var(--muted); font-size: 0.9rem; }

h1, h2, h3{
  color: var(--text) !important;
  letter-spacing: -0.01em !important;
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

/* Header */
.dataframe thead th{
  position: sticky !important;
  top: 0 !important;
  z-index: 2 !important;
  background: #101828 !important;
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
  background-color: transparent !important; /* allow background-image bars to show cleanly */
}
.dataframe tbody td:last-child{ border-right:none !important; }

/* Zebra */
.dataframe tbody tr:nth-child(odd){ background: #fff !important; }
.dataframe tbody tr:nth-child(even){ background: var(--panel-2) !important; }

/* Hover */
.dataframe tbody tr:hover{ background: rgba(31,79,216,.06) !important; }

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
  background: #101828 !important;
  color: #fff !important;
  border: 1px solid #101828 !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
  box-shadow: none !important;
  font-weight: 650 !important;
}
.stDownloadButton button:hover{
  background: #18223a !important;
  border-color: #18223a !important;
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
    <div class="small-muted">Standard + xG • per 90 • percentile + z-score modes</div>
  </div>
  <div class="badge">Season 2025/26</div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Load + compute
# -------------------------
try:
    df = load_data(DATA_FILE)
    kpis = get_kpis(df)
    df = calc_percentiles(df, kpis)
    df = calc_zscores(df, kpis)
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
    pos_group = st.selectbox(
        "Position", ["All Positions", "Defenders", "Midfielders", "Forwards"], label_visibility="collapsed"
    )

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

    metric_mode = st.selectbox(
        "Metric mode",
        ["Percentile", "Z-score"],
        index=0,
        label_visibility="collapsed",
        help="Affects the metric columns in Percentiles/Both modes.",
    )

    show_bars = st.toggle("Show distribution bars", value=True)
    show_rank = st.toggle("Show Rank (based on first metric column)", value=True)

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
  <span class="badge">Stats selected: <strong>{len(selected_stats)}</strong></span>
  <span class="badge">Teams: <strong>{df_filtered["squadName"].nunique()}</strong></span>
  <span class="badge">Mode: <strong>{display_mode}</strong></span>
  <span class="badge">Metric: <strong>{metric_mode}</strong></span>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Build display dataframe
# -------------------------
base_cols = ["displayName", "squadName", "positions"]
cols = base_cols.copy()

metric_suffix = "_pct" if metric_mode == "Percentile" else "_z"
metric_label = "(Pct)" if metric_mode == "Percentile" else "(Z)"

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

# Rename columns nicely
rename_map = {}
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

# Sort by first metric column when available, else first stat column
sort_col = None
for c in df_display.columns:
    if c.endswith(metric_label):
        sort_col = c
        break
if sort_col is None and len(df_display.columns) > len(base_cols):
    sort_col = df_display.columns[len(base_cols)]

if sort_col:
    df_display = df_display.sort_values(sort_col, ascending=False, na_position="last")

# Add Rank
if show_rank and sort_col:
    df_display.insert(
        0, "Rank", df_display[sort_col].rank(ascending=False, method="min").astype("Int64")
    )

# -------------------------
# Style table
# -------------------------
styled = df_display.style

# Format numbers
fmt = {}
for col in df_display.columns:
    if col in ["Rank"] + base_cols:
        continue
    if col.endswith("(Pct)"):
        fmt[col] = "{:.0f}"
    elif col.endswith("(Z)"):
        fmt[col] = "{:+.2f}"
    else:
        fmt[col] = "{:.2f}"

styled = styled.format(fmt, na_rep="-")

# Style Rank
if "Rank" in df_display.columns:
    styled = styled.applymap(
        lambda v: "color:#667085; font-weight:800; background-color: transparent;",
        subset=["Rank"],
    )

# Apply metric styling (with optional bars)
for col in df_display.columns:
    if col.endswith("(Pct)"):
        if show_bars:
            styled = styled.applymap(pct_cell_style, subset=[col])
        else:
            # fallback: just subtle extremes (no bar)
            styled = styled.applymap(
                lambda v: "color:#054f31; font-weight:800;" if pd.notna(v) and float(v) >= 95
                else ("color:#7a271a; font-weight:700;" if pd.notna(v) and float(v) <= 15 else "color: var(--text);"),
                subset=[col],
            )

    if col.endswith("(Z)"):
        if show_bars:
            styled = styled.applymap(z_cell_style, subset=[col])
        else:
            styled = styled.applymap(
                lambda v: "color:#065f46; font-weight:800;" if pd.notna(v) and float(v) >= 1.25
                else ("color:#7a271a; font-weight:800;" if pd.notna(v) and float(v) <= -1.25 else "color: var(--text);"),
                subset=[col],
            )

# -------------------------
# Table header
# -------------------------
st.markdown(
    """
<div style="margin-top: 1.25rem; margin-bottom: .5rem; display:flex; align-items:center; justify-content:space-between;">
  <h3 style="margin:0; font-size: 1.05rem;">Table</h3>
  <div class="badge">Tip: use sidebar filters • sorted by first metric column</div>
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
      <div style="font-size:1.6rem; font-weight:900; color:var(--text);">{len(df_display)}</div>
      <div class="small-muted">Rows</div>
    </div>
    <div>
      <div style="font-size:1.6rem; font-weight:900; color:var(--text);">{len(selected_stats)}</div>
      <div class="small-muted">Stats</div>
    </div>
    <div>
      <div style="font-size:1.6rem; font-weight:900; color:var(--text);">{df_display['squadName'].nunique() if 'squadName' in df_display.columns else 0}</div>
      <div class="small-muted">Teams</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# -------------------------
# Legend
# -------------------------
if metric_mode == "Percentile":
    legend = "Percentiles: green = high, red = low. Bars show percentile fill (0→100)."
else:
    legend = "Z-scores: green = positive, red = negative. Bars diverge from 0 (center line)."

st.markdown(
    f"""
<div class="card" style="margin-top: 1.25rem;">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;">
    <div>
      <div class="badge">Legend</div>
      <div class="small-muted" style="margin-top:.35rem;">{legend} “Lower is better” stats are auto-inverted.</div>
    </div>
    <div class="badge">Bars: {"On" if show_bars else "Off"}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
