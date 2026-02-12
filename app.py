"""
IMPECT Stats Table - FBref/Baseball Savant Style
=================================================
Automatically loads Keuken Kampioen Divisie data (Standard + xG combined)

Features:
- Baseball Savant-style percentile color coding
- Position filtering
- Per 90 calculations
- Advanced stat selection
- Export to CSV/Excel
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="IMPECT Stats", page_icon="⚽", layout="wide")

# Hard-coded file path - change this to your actual file location
DATA_FILE = "/Users/user/IMPECT/season_25-26_leagues/Keuken Kampioen Divisie.xlsx"

# Color scales for percentile rankings (Baseball Savant style)
def percentile_color(pct):
    """Return color based on percentile"""
    if pd.isna(pct):
        return "#FFFFFF"
    if pct >= 90:
        return "#08519c"  # Dark blue - Elite
    if pct >= 75:
        return "#3182bd"  # Blue - Excellent
    if pct >= 60:
        return "#6baed6"  # Light blue - Above average
    if pct >= 40:
        return "#9ecae1"  # Very light blue - Average
    if pct >= 25:
        return "#c6dbef"  # Pale blue - Below average
    return "#eff3ff"  # Almost white - Poor

# Metrics where lower is better (inverted color scale)
INVERTED_METRICS = [
    "Number of Fouls", "Lost Ground Duels", "Lost Aerial Duels",
    "Unsuccessful Passes", "Failed passes", "Failed dribbles",
    "Total Shots Off Target", "Red Card", "Yellow Card",
    "Ball losses", "Critical Ball Losses"
]

def is_inverted_metric(col_name: str) -> bool:
    """Check if metric should use inverted color scale"""
    return any(inv.lower() in col_name.lower() for inv in INVERTED_METRICS)

# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_data(file_path: str):
    """Load and merge Standard + xG sheets"""
    
    # Load both sheets
    standard_df = pd.read_excel(file_path, sheet_name="Standard")
    xg_df = pd.read_excel(file_path, sheet_name="xG")
    
    # Base columns (metadata)
    base_cols = [
        'iterationId', 'squadId', 'squadName', 'playerId', 'matches', 'positions',
        'commonname', 'firstname', 'lastname', 'birthdate', 'birthplace',
        'leg', 'countryIds', 'gender', 'season', 'dataVersion',
        'lastChangeTimestamp', 'competition_name', 'competition_type', 'competition_gender'
    ]
    
    # Get KPI columns (non-base columns)
    standard_kpis = [col for col in standard_df.columns if col not in base_cols]
    xg_kpis = [col for col in xg_df.columns if col not in base_cols]
    
    # Merge on playerId
    merged_df = standard_df.merge(
        xg_df[['playerId'] + xg_kpis],
        on='playerId',
        how='left',
        suffixes=('', '_xg')
    )
    
    # Create display name
    cn = merged_df.get("commonname", pd.Series([""] * len(merged_df))).fillna("").astype(str).str.strip()
    fallback = (
        merged_df.get("firstname", pd.Series([""] * len(merged_df))).fillna("").astype(str).str.strip()
        + " "
        + merged_df.get("lastname", pd.Series([""] * len(merged_df))).fillna("").astype(str).str.strip()
    ).str.strip()
    merged_df["displayName"] = np.where(cn == "", fallback, cn)
    
    # Ensure matches is numeric
    merged_df["matches"] = pd.to_numeric(merged_df["matches"], errors="coerce").fillna(0)
    
    return merged_df, standard_kpis, xg_kpis

def get_kpi_columns(df: pd.DataFrame) -> list:
    """Extract all KPI columns"""
    base_cols = {
        'iterationId', 'squadId', 'squadName', 'playerId', 'matches', 'positions',
        'commonname', 'firstname', 'lastname', 'birthdate', 'birthplace',
        'leg', 'countryIds', 'gender', 'season', 'dataVersion',
        'lastChangeTimestamp', 'competition_name', 'competition_type',
        'competition_gender', 'displayName'
    }
    
    kpi_cols = []
    for col in df.columns:
        if col not in base_cols and pd.api.types.is_numeric_dtype(df[col]):
            kpi_cols.append(col)
    
    return sorted(kpi_cols)

def calculate_percentiles(df: pd.DataFrame, kpi_cols: list) -> pd.DataFrame:
    """Calculate percentile rankings for all KPI columns"""
    df_pct = df.copy()
    
    for col in kpi_cols:
        pct_col = f"{col}_pct"
        values = pd.to_numeric(df[col], errors="coerce")
        
        if is_inverted_metric(col):
            # For "bad" metrics, invert percentiles
            df_pct[pct_col] = 100 - values.rank(pct=True, method='average') * 100
        else:
            df_pct[pct_col] = values.rank(pct=True, method='average') * 100
    
    return df_pct

def calculate_per90(df: pd.DataFrame, kpi_cols: list) -> pd.DataFrame:
    """Calculate per-90 minutes stats"""
    df_p90 = df.copy()
    
    # Assume 90 minutes per match
    df_p90["minutes"] = df_p90["matches"] * 90
    
    for col in kpi_cols:
        # Don't convert rates/percentages to per-90
        if "%" not in col and " %" not in col.lower() and df_p90["minutes"].sum() > 0:
            p90_col = f"{col}_p90"
            df_p90[p90_col] = (pd.to_numeric(df[col], errors="coerce") / df_p90["minutes"]) * 90
    
    return df_p90

# -------------------------
# Position filtering
# -------------------------
POSITION_GROUPS = {
    "All Outfield": [],
    "Defenders": ["DEFENDER", "CENTRE_BACK", "CENTRAL_DEFENDER", "CENTER_BACK"],
    "Fullbacks": ["WINGBACK", "LEFT_WINGBACK", "RIGHT_WINGBACK"],
    "Midfielders": ["MIDFIELD", "CENTRAL_MIDFIELD", "ATTACKING_MIDFIELD", "DEFENSE_MIDFIELD"],
    "Forwards": ["FORWARD", "WINGER", "CENTER_FORWARD", "LEFT_WINGER", "RIGHT_WINGER"],
}

def filter_by_position(df: pd.DataFrame, position_group: str):
    """Filter dataframe by position group"""
    if position_group == "All Outfield" or not position_group:
        return df
    
    tokens = POSITION_GROUPS.get(position_group, [])
    if not tokens:
        return df
    
    mask = pd.Series(False, index=df.index)
    for token in tokens:
        mask |= df["positions"].astype(str).str.contains(token, case=False, na=False)
    
    return df[mask]

# -------------------------
# Styling
# -------------------------
def create_color_map(df: pd.DataFrame, col: str):
    """Create color map for a column based on percentiles"""
    pct_col = f"{col}_pct"
    if pct_col not in df.columns:
        return ['background-color: #FFFFFF'] * len(df)
    
    colors = []
    for pct in df[pct_col]:
        if is_inverted_metric(col):
            color = percentile_color(100 - pct if not pd.isna(pct) else pct)
        else:
            color = percentile_color(pct)
        colors.append(f'background-color: {color}')
    
    return colors

# -------------------------
# Main app
# -------------------------
st.title("⚽ Keuken Kampioen Divisie Stats")
st.caption("FBref / Baseball Savant style - Standard + xG combined")

# Load data
try:
    df_raw, standard_kpis, xg_kpis = load_data(DATA_FILE)
    st.success(f"✅ Loaded {len(df_raw)} players from {DATA_FILE.split('/')[-1]}")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info(f"Expected file at: {DATA_FILE}")
    st.stop()

# Get all KPI columns
kpi_cols = get_kpi_columns(df_raw)

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    # Position filter
    position_group = st.selectbox(
        "Position Group",
        options=list(POSITION_GROUPS.keys()),
        index=0
    )
    
    # Minimum matches
    min_matches = st.number_input(
        "Minimum matches",
        min_value=0,
        value=3,
        step=1
    )
    
    st.divider()
    
    # Squad filter
    squads = sorted(df_raw["squadName"].dropna().unique().tolist())
    squad_filter = st.multiselect("Squad", squads)
    
    # Name search
    name_filter = st.text_input("Player name contains", placeholder="e.g., Van")
    
    st.divider()
    
    # Display options
    st.header("📊 Display")
    show_per90 = st.checkbox("Show per-90 stats", value=False)
    show_percentiles = st.checkbox("Show percentile columns", value=False)

# Calculate percentiles
df_with_pct = calculate_percentiles(df_raw, kpi_cols)

# Calculate per-90 if requested
if show_per90:
    df_with_pct = calculate_per90(df_with_pct, kpi_cols)
    display_kpi_cols = [col for col in df_with_pct.columns if col.endswith("_p90")]
    table_suffix = " (per 90)"
else:
    display_kpi_cols = kpi_cols
    table_suffix = ""

# Apply position filter first
df_filtered = filter_by_position(df_with_pct, position_group)

# Apply other filters
if min_matches > 0:
    df_filtered = df_filtered[df_filtered["matches"] >= min_matches]

if squad_filter:
    df_filtered = df_filtered[df_filtered["squadName"].isin(squad_filter)]

if name_filter:
    df_filtered = df_filtered[
        df_filtered["displayName"].astype(str).str.contains(name_filter, case=False, na=False)
    ]

st.info(f"🔍 **{len(df_filtered)}** players match filters | Position: **{position_group}**")

if df_filtered.empty:
    st.warning("No players match your filters")
    st.stop()

# Stat category selection
st.subheader("📋 Select Statistics")

# Create tabs for stat categories
tab1, tab2, tab3 = st.tabs(["⚽ Standard Stats", "📈 xG Stats", "🎯 Custom Selection"])

with tab1:
    st.caption("Standard event-based statistics")
    
    # Smart defaults for standard
    default_standard = [
        col for col in display_kpi_cols 
        if any(x in col for x in standard_kpis)
        and any(key in col.lower() for key in [
            "goal", "assist", "pass", "shot", "duel", "touch", "dribble"
        ])
    ][:12]
    
    selected_stats_standard = st.multiselect(
        "Standard statistics",
        options=[col for col in display_kpi_cols if any(x in col for x in standard_kpis)],
        default=default_standard
    )
    
    if st.button("Show Standard Stats Table", use_container_width=True):
        selected_stats = selected_stats_standard

with tab2:
    st.caption("Expected goals and threat metrics")
    
    # Smart defaults for xG
    default_xg = [
        col for col in display_kpi_cols 
        if any(x in col for x in xg_kpis)
    ][:12]
    
    selected_stats_xg = st.multiselect(
        "xG statistics",
        options=[col for col in display_kpi_cols if any(x in col for x in xg_kpis)],
        default=default_xg
    )
    
    if st.button("Show xG Stats Table", use_container_width=True):
        selected_stats = selected_stats_xg

with tab3:
    st.caption("Mix and match any stats")
    
    # Create categories for easier selection
    categories = {
        "Goals & Assists": ["goal", "assist"],
        "Passing": ["pass"],
        "Shooting": ["shot"],
        "Duels": ["duel", "aerial", "ground"],
        "Possession": ["touch", "dribble"],
        "xG Metrics": ["xg", "expected"],
    }
    
    selected_category = st.selectbox("Quick filter", ["All"] + list(categories.keys()))
    
    if selected_category == "All":
        filtered_kpis = display_kpi_cols
    else:
        keywords = categories[selected_category]
        filtered_kpis = [
            col for col in display_kpi_cols 
            if any(kw in col.lower() for kw in keywords)
        ]
    
    selected_stats_custom = st.multiselect(
        "Select any statistics",
        options=filtered_kpis,
        default=filtered_kpis[:10] if filtered_kpis else []
    )
    
    if st.button("Show Custom Stats Table", use_container_width=True):
        selected_stats = selected_stats_custom

# Default to standard stats
if 'selected_stats' not in locals():
    selected_stats = default_standard if default_standard else display_kpi_cols[:10]

if not selected_stats:
    st.warning("⚠️ Select at least one statistic from the tabs above")
    st.stop()

st.metric("📊 Statistics displayed", len(selected_stats))

# Build display dataframe
display_cols = ["displayName", "squadName", "positions", "matches"] + selected_stats

if show_percentiles:
    # Add percentile columns interleaved
    display_cols_with_pct = ["displayName", "squadName", "positions", "matches"]
    for stat in selected_stats:
        display_cols_with_pct.append(stat)
        pct_col = f"{stat}_pct"
        if pct_col in df_filtered.columns:
            display_cols_with_pct.append(pct_col)
    display_cols = display_cols_with_pct

# Filter to existing columns
display_cols = [col for col in display_cols if col in df_filtered.columns]

df_display = df_filtered[display_cols].copy()

# Sort by first stat (descending)
if selected_stats:
    df_display = df_display.sort_values(selected_stats[0], ascending=False)

# Display table
st.subheader(f"📊 Stats Table{table_suffix}")

# Apply styling with color coding
styled_df = df_display.style

for stat in selected_stats:
    if stat in df_display.columns:
        styled_df = styled_df.apply(
            lambda _: create_color_map(df_display, stat),
            subset=[stat],
            axis=0
        )

# Format numbers
format_dict = {}
for stat in selected_stats:
    if stat in df_display.columns:
        # Check if values are generally small (< 1) for more decimals
        max_val = df_display[stat].max()
        if pd.notna(max_val) and max_val < 1:
            format_dict[stat] = '{:.3f}'
        elif pd.notna(max_val) and max_val < 10:
            format_dict[stat] = '{:.2f}'
        else:
            format_dict[stat] = '{:.1f}'

if format_dict:
    styled_df = styled_df.format(format_dict, na_rep="-")

# Display
st.dataframe(styled_df, use_container_width=True, height=600)

# Summary stats
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Players", len(df_display))
with col2:
    avg_matches = df_display["matches"].mean()
    st.metric("Avg Matches", f"{avg_matches:.1f}")
with col3:
    if selected_stats:
        first_stat_avg = df_display[selected_stats[0]].mean()
        st.metric(f"Avg {selected_stats[0][:25]}", f"{first_stat_avg:.2f}")
with col4:
    if len(selected_stats) > 1:
        second_stat_avg = df_display[selected_stats[1]].mean()
        st.metric(f"Avg {selected_stats[1][:25]}", f"{second_stat_avg:.2f}")

# Export
st.divider()
st.subheader("💾 Export")

col1, col2 = st.columns(2)

with col1:
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name=f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_display.to_excel(writer, index=False, sheet_name='Stats')
    
    st.download_button(
        "📥 Download Excel",
        data=buffer.getvalue(),
        file_name=f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# Legend
with st.expander("🎨 Color Scale Legend"):
    st.markdown("""
    **Percentile Color Scale** (Baseball Savant style)
    
    - 🟦 **90-100th**: Elite (Dark Blue) `#08519c`
    - 🟦 **75-89th**: Excellent (Blue) `#3182bd`
    - 🟦 **60-74th**: Above Average (Light Blue) `#6baed6`
    - 🟦 **40-59th**: Average (Very Light Blue) `#9ecae1`
    - ⬜ **25-39th**: Below Average (Pale Blue) `#c6dbef`
    - ⬜ **0-24th**: Poor (Almost White) `#eff3ff`
    
    *For "bad" metrics (fouls, turnovers, unsuccessful passes), the scale is automatically inverted.*
    """)