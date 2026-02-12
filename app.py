"""
IMPECT Stats Table - FBref/Baseball Savant Style
=================================================
Modern, interactive stats tables with percentile rankings and color coding.

Features:
- Position-specific stat tables
- Percentile-based color coding
- Advanced filtering
- Per 90 calculations
- Export to CSV/Excel
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="IMPECT Stats", page_icon="📊", layout="wide")

# Default templates directory (change to your actual path)
DEFAULT_TEMPLATES_DIR = "/Users/user/Documents/GitHub//hradeck-scouting/"

# Position templates available
POSITION_TEMPLATES = {
    "GK": "Goalkeeper",
    "CB": "Center Back",
    "FB": "Fullback",
    "DM": "Defensive Midfielder",
    "CM": "Central Midfielder",
    "AM": "Attacking Midfielder",
    "W": "Winger",
    "ST": "Striker",
}

# Color scales for percentile rankings
def percentile_color(pct):
    """Return color based on percentile (Baseball Savant style)"""
    if pd.isna(pct):
        return "#FFFFFF"
    if pct >= 90:
        return "#08519c"  # Dark blue
    if pct >= 75:
        return "#3182bd"  # Blue
    if pct >= 60:
        return "#6baed6"  # Light blue
    if pct >= 40:
        return "#9ecae1"  # Very light blue
    if pct >= 25:
        return "#c6dbef"  # Pale blue
    if pct >= 10:
        return "#eff3ff"  # Almost white
    return "#FFFFFF"  # White

def percentile_color_inverted(pct):
    """Inverted color scale (for metrics where lower is better)"""
    if pd.isna(pct):
        return "#FFFFFF"
    return percentile_color(100 - pct)

# Metrics where lower is better
INVERTED_METRICS = {
    "Number of Fouls", "fouls", "red", "yellow", "card",
    "lost", "unsuccessful", "failed", "off target"
}

def is_inverted_metric(col_name: str) -> bool:
    """Check if metric should use inverted color scale"""
    col_lower = col_name.lower()
    return any(inv in col_lower for inv in INVERTED_METRICS)

# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_template(template_path: str) -> pd.DataFrame:
    """Load and prepare template data"""
    df = pd.read_excel(template_path)
    
    # Ensure display name
    if "displayName" not in df.columns:
        cn = df.get("commonname", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        fallback = (
            df.get("firstname", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
            + " "
            + df.get("lastname", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        ).str.strip()
        df["displayName"] = np.where(cn == "", fallback, cn)
    
    # Ensure numeric matches column
    if "matches" in df.columns:
        df["matches"] = pd.to_numeric(df["matches"], errors="coerce").fillna(0)
    else:
        df["matches"] = 0
    
    return df

def get_kpi_columns(df: pd.DataFrame) -> list:
    """Extract KPI columns (those with IDs in parentheses)"""
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
    
    # Assume 90 minutes per match (could be refined with actual minutes)
    df_p90["minutes"] = df_p90["matches"] * 90
    
    for col in kpi_cols:
        if "%" not in col and "win" not in col.lower():  # Don't convert rates
            p90_col = f"{col}_p90"
            df_p90[p90_col] = (pd.to_numeric(df[col], errors="coerce") / df_p90["minutes"]) * 90
    
    return df_p90

# -------------------------x
# Styling functions
# -------------------------
def style_dataframe(df: pd.DataFrame, stat_cols: list, show_percentiles: bool = True):
    """Apply FBref/Savant-style formatting"""
    
    def color_percentile(val, col_name):
        """Color cell based on percentile"""
        if pd.isna(val):
            return 'background-color: #FFFFFF'
        
        pct_col = f"{col_name}_pct"
        if pct_col not in df.columns:
            return 'background-color: #FFFFFF'
        
        # Get percentile for this row
        idx = val.name if hasattr(val, 'name') else 0
        if idx not in df.index:
            return 'background-color: #FFFFFF'
        
        pct = df.loc[idx, pct_col]
        
        if is_inverted_metric(col_name):
            color = percentile_color_inverted(pct)
        else:
            color = percentile_color(pct)
        
        return f'background-color: {color}'
    
    # Start with base styling
    styler = df.style
    
    # Apply color coding to stat columns
    for col in stat_cols:
        if col in df.columns:
            styler = styler.applymap(
                lambda val: color_percentile(val, col),
                subset=[col]
            )
    
    # Format numbers
    format_dict = {}
    for col in stat_cols:
        if col in df.columns:
            format_dict[col] = '{:.2f}'
    
    if format_dict:
        styler = styler.format(format_dict, na_rep="-")
    
    return styler

# -------------------------
# Main app
# -------------------------
st.title("📊 IMPECT Stats Tables")
st.caption("FBref / Baseball Savant style - Percentile-ranked player statistics")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Templates directory input
    templates_dir = st.text_input(
        "Templates Directory",
        value=DEFAULT_TEMPLATES_DIR,
        help="Path to folder containing template Excel files"
    )
    
    # Check if directory exists
    if not os.path.exists(templates_dir):
        st.error(f"Directory not found: {templates_dir}")
        st.info("Upload templates manually below")
        use_upload = True
    else:
        use_upload = False
        st.success("✅ Templates directory found")
    
    st.divider()
    
    # Position selection
    st.subheader("Position")
    position_code = st.selectbox(
        "Select position",
        options=list(POSITION_TEMPLATES.keys()),
        format_func=lambda x: f"{x} - {POSITION_TEMPLATES[x]}"
    )
    
    st.divider()
    
    # Display options
    st.subheader("Display")
    show_per90 = st.checkbox("Show per-90 stats", value=False)
    show_percentiles = st.checkbox("Show percentile columns", value=False)
    min_matches = st.number_input("Minimum matches", min_value=0, value=1, step=1)
    
    st.divider()
    
    # Advanced filters
    with st.expander("🔍 Advanced Filters"):
        squad_filter = st.text_input("Squad contains", placeholder="e.g., Ajax")
        name_filter = st.text_input("Name contains", placeholder="e.g., Van")

# Load data
if use_upload:
    uploaded_file = st.file_uploader(
        f"Upload {position_code} template",
        type=["xlsx"],
        help=f"Upload {POSITION_TEMPLATES[position_code]} template Excel file"
    )
    
    if not uploaded_file:
        st.info("⬆️ Upload a template file to begin")
        st.stop()
    
    df_raw = load_template(uploaded_file)
else:
    template_file = os.path.join(templates_dir, f"{position_code}_template.xlsx")
    
    if not os.path.exists(template_file):
        st.error(f"Template not found: {template_file}")
        st.stop()
    
    df_raw = load_template(template_file)

# Show data info
st.info(f"📁 Loaded: **{len(df_raw)}** players | Position: **{POSITION_TEMPLATES[position_code]}**")

# Get KPI columns
kpi_cols = get_kpi_columns(df_raw)

if not kpi_cols:
    st.error("No KPI columns found in template")
    st.stop()

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

# Apply filters
df_filtered = df_with_pct.copy()

if min_matches > 0:
    df_filtered = df_filtered[df_filtered["matches"] >= min_matches]

if squad_filter:
    if "squadName" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["squadName"].astype(str).str.contains(squad_filter, case=False, na=False)
        ]

if name_filter:
    df_filtered = df_filtered[
        df_filtered["displayName"].astype(str).str.contains(name_filter, case=False, na=False)
    ]

st.info(f"🔍 Filtered: **{len(df_filtered)}** players match criteria")

if df_filtered.empty:
    st.warning("No players match your filters")
    st.stop()

# Column selector
st.subheader("📋 Select Stats to Display")

col1, col2 = st.columns([3, 1])

with col1:
    # Smart defaults based on position
    if position_code == "GK":
        default_stats = [col for col in display_kpi_cols if any(
            x in col.lower() for x in ["save", "catch", "shot", "pass"]
        )][:10]
    elif position_code in ["CB", "FB"]:
        default_stats = [col for col in display_kpi_cols if any(
            x in col.lower() for x in ["duel", "pass", "aerial", "touch"]
        )][:10]
    elif position_code in ["DM", "CM"]:
        default_stats = [col for col in display_kpi_cols if any(
            x in col.lower() for x in ["pass", "duel", "touch", "progressive"]
        )][:10]
    elif position_code in ["AM", "W", "ST"]:
        default_stats = [col for col in display_kpi_cols if any(
            x in col.lower() for x in ["goal", "assist", "shot", "xg", "touch", "dribble"]
        )][:10]
    else:
        default_stats = display_kpi_cols[:10]
    
    selected_stats = st.multiselect(
        "Statistics to show",
        options=display_kpi_cols,
        default=default_stats,
        help="Select which statistics to display in the table"
    )

with col2:
    st.metric("Stats selected", len(selected_stats))
    st.metric("Max recommended", 15)

if not selected_stats:
    st.warning("Select at least one statistic")
    st.stop()

# Build display dataframe
display_cols = ["displayName", "squadName", "matches"] + selected_stats

if show_percentiles:
    # Add percentile columns
    pct_cols = [f"{col}_pct" for col in selected_stats if f"{col}_pct" in df_filtered.columns]
    # Interleave stats and percentiles
    display_cols_with_pct = ["displayName", "squadName", "matches"]
    for stat in selected_stats:
        display_cols_with_pct.append(stat)
        pct_col = f"{stat}_pct"
        if pct_col in df_filtered.columns:
            display_cols_with_pct.append(pct_col)
    display_cols = display_cols_with_pct

# Filter to display columns that exist
display_cols = [col for col in display_cols if col in df_filtered.columns]

df_display = df_filtered[display_cols].copy()

# Sort by first stat column (descending)
if len(selected_stats) > 0:
    df_display = df_display.sort_values(selected_stats[0], ascending=False)

# Display the table
st.subheader(f"📊 {POSITION_TEMPLATES[position_code]} Stats{table_suffix}")

# Style and display
styled_df = style_dataframe(df_display, selected_stats, show_percentiles)

# Convert to HTML with custom CSS
html = styled_df.to_html(index=False, escape=False)

# Add custom CSS for table styling
table_css = """
<style>
    table {
        width: 100%;
        border-collapse: collapse;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 13px;
    }
    th {
        background-color: #1e40af;
        color: white;
        padding: 12px 8px;
        text-align: left;
        font-weight: 600;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    td {
        padding: 10px 8px;
        border-bottom: 1px solid #e5e7eb;
    }
    tr:hover {
        background-color: #f9fafb;
    }
    tr:nth-child(even) {
        background-color: #f3f4f6;
    }
    tr:nth-child(even):hover {
        background-color: #f9fafb;
    }
</style>
"""

# Display table
st.markdown(table_css + html, unsafe_allow_html=True)

# Stats summary
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Players shown", len(df_display))
with col2:
    avg_matches = df_display["matches"].mean()
    st.metric("Avg matches", f"{avg_matches:.1f}")
with col3:
    if selected_stats:
        avg_first_stat = df_display[selected_stats[0]].mean()
        st.metric(f"Avg {selected_stats[0][:20]}", f"{avg_first_stat:.2f}")
with col4:
    if len(selected_stats) > 1:
        avg_second_stat = df_display[selected_stats[1]].mean()
        st.metric(f"Avg {selected_stats[1][:20]}", f"{avg_second_stat:.2f}")

# Export options
st.divider()
st.subheader("💾 Export")

col1, col2 = st.columns(2)

with col1:
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name=f"{position_code}_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    # Excel export with styling
    from io import BytesIO
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_display.to_excel(writer, index=False, sheet_name='Stats')
    
    st.download_button(
        "📥 Download Excel",
        data=buffer.getvalue(),
        file_name=f"{position_code}_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# Show legend
with st.expander("🎨 Color Scale Legend"):
    st.markdown("""
    **Percentile Color Scale** (Baseball Savant style)
    
    - 🟦 **90-100th**: Elite (Dark Blue)
    - 🟦 **75-89th**: Excellent (Blue)
    - 🟦 **60-74th**: Above Average (Light Blue)
    - 🟦 **40-59th**: Average (Very Light Blue)
    - ⬜ **25-39th**: Below Average (Pale Blue)
    - ⬜ **10-24th**: Poor (Almost White)
    - ⬜ **0-9th**: Very Poor (White)
    
    *For metrics where lower is better (fouls, turnovers), the scale is inverted.*
    """)

# Data preview tab
with st.expander("📊 Full Dataset Preview"):
    st.dataframe(df_filtered, use_container_width=True, height=400)