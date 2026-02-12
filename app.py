"""
IMPECT Stats Table - Premium Edition
=====================================
Ultra-polished stats table for Keuken Kampioen Divisie
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
    initial_sidebar_state="expanded"  # Show sidebar by default
)

# File path - change to your actual location
DATA_FILE = "Keuken Kampioen Divisie.xlsx"

# Baseball Savant color scale
def percentile_color(pct):
    if pd.isna(pct): return "#FFFFFF"
    if pct >= 90: return "#08519c"  # Dark blue
    if pct >= 75: return "#3182bd"  # Blue
    if pct >= 60: return "#6baed6"  # Light blue
    if pct >= 40: return "#9ecae1"  # Very light blue
    if pct >= 25: return "#c6dbef"  # Pale blue
    return "#eff3ff"

# Metrics where lower is better
INVERTED = ["foul", "lost", "unsuccessful", "failed", "off target", "red", "yellow"]

def is_inverted(col):
    return any(inv in col.lower() for inv in INVERTED)

# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_data(file_path):
    standard = pd.read_excel(file_path, sheet_name="Standard")
    xg = pd.read_excel(file_path, sheet_name="xG")
    
    base = ['iterationId', 'squadId', 'squadName', 'playerId', 'positions',
            'commonname', 'firstname', 'lastname', 'birthdate', 'birthplace',
            'leg', 'countryIds', 'gender', 'season', 'dataVersion',
            'lastChangeTimestamp', 'competition_name', 'competition_type', 'competition_gender']
    
    xg_kpis = [c for c in xg.columns if c not in base]
    merged = standard.merge(xg[['playerId'] + xg_kpis], on='playerId', how='left')
    
    cn = merged.get("commonname", "").fillna("").astype(str).str.strip()
    fallback = (merged.get("firstname", "").fillna("").astype(str).str.strip() + " " +
                merged.get("lastname", "").fillna("").astype(str).str.strip()).str.strip()
    merged["displayName"] = np.where(cn == "", fallback, cn)
    
    return merged

def get_kpis(df):
    base = {'iterationId', 'squadId', 'squadName', 'playerId', 'positions',
            'commonname', 'firstname', 'lastname', 'birthdate', 'birthplace',
            'leg', 'countryIds', 'gender', 'season', 'dataVersion',
            'lastChangeTimestamp', 'competition_name', 'competition_type',
            'competition_gender', 'displayName'}
    return [c for c in df.columns if c not in base and pd.api.types.is_numeric_dtype(df[c])]

def calc_percentiles(df, kpis):
    for col in kpis:
        vals = pd.to_numeric(df[col], errors="coerce")
        pct = vals.rank(pct=True, method='average') * 100
        df[f"{col}_pct"] = 100 - pct if is_inverted(col) else pct
    return df

def color_map(df, col):
    pct_col = f"{col}_pct"
    if pct_col not in df.columns:
        return ['background-color: #FFFFFF'] * len(df)
    
    colors = []
    for pct in df[pct_col]:
        color = percentile_color(100 - pct if is_inverted(col) and not pd.isna(pct) else pct)
        colors.append(f'background-color: {color}')
    return colors

# -------------------------
# UI
# -------------------------

# Premium header
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
        <h1 style="margin: 0;">⚽ Keuken Kampioen Divisie</h1>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                         color: white; 
                         padding: 0.5rem 1rem; 
                         border-radius: 8px; 
                         font-weight: 600; 
                         font-size: 0.875rem;
                         box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);">
                Season 25/26
            </span>
        </div>
    </div>
    <p style="color: #64748b; font-size: 0.875rem; margin: 0;">
        Premium stats explorer with percentile rankings • Standard + xG metrics
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
try:
    df = load_data(DATA_FILE)
    kpis = get_kpis(df)
    df = calc_percentiles(df, kpis)
    
    # Show data badge
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                border-left: 4px solid #22c55e; 
                padding: 0.75rem 1rem; 
                border-radius: 8px; 
                margin-bottom: 1.5rem;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
        <span style="color: #166534; font-weight: 600; font-size: 0.875rem;">
            ✓ Data Loaded
        </span>
        <span style="color: #16a34a; margin-left: 0.5rem; font-size: 0.875rem;">
            {len(df)} players • {len(kpis)} metrics
        </span>
    </div>
    """, unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# -------------------------
# SIDEBAR - All Filters
# -------------------------
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.25rem; 
                   font-weight: 700; 
                   color: #0f172a; 
                   margin: 0;">
            🎯 Filters & Settings
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Player search
    st.markdown("##### 🔍 Player Search")
    name_filter = st.text_input(
        "Player name",
        placeholder="Type player name...",
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Squad filter
    st.markdown("##### 🏟️ Squad")
    squads = ["All Squads"] + sorted(df["squadName"].dropna().unique().tolist())
    squad = st.selectbox("Squad", squads, label_visibility="collapsed")
    
    st.divider()
    
    # Position filter
    st.markdown("##### ⚽ Position Group")
    positions = ["All Positions", "Defenders", "Midfielders", "Forwards"]
    pos_group = st.selectbox("Position", positions, label_visibility="collapsed")
    
    st.divider()
    
    # Stat category
    st.markdown("##### 📊 Stat Category")
    categories = {
        "⚽ Goals & Assists": ["Goals", "Assists", "Pre Assist", "Shot-Creating Actions", "Shot xG from Passes"],
        "🎯 Shooting": ["Total Shots", "Total Shots On Target", "Shot-based xG", "Post-Shot xG"],
        "📤 Passing": ["Successful Passes", "Unsuccessful Passes", "Progressive passes", "Pass Accuracy"],
        "🤼 Duels": ["Won Ground Duels", "Lost Ground Duels", "Won Aerial Duels", "Lost Aerial Duels"],
        "📊 xG Metrics": ["Shot-based xG", "Post-Shot xG", "Expected Goal Assists", "Expected Shot Assists", "Packing non-shot-based xG"],
    }
    
    selected_cat = st.selectbox(
        "Category",
        list(categories.keys()),
        label_visibility="collapsed"
    )
    keywords = categories[selected_cat]
    
    st.divider()
    
    # Stat selection
    st.markdown("##### 📈 Select Stats")
    matching = [k for k in kpis if any(kw in k for kw in keywords)][:20]
    
    selected_stats = st.multiselect(
        "Statistics",
        options=matching,
        default=matching[:8] if len(matching) >= 8 else matching,
        label_visibility="collapsed"
    )
    
    if not selected_stats:
        st.warning("⚠️ Select at least one stat")
        st.stop()
    
    st.divider()
    
    # Info box
    st.markdown(f"""
    <div style="background: #f8fafc; 
                padding: 1rem; 
                border-radius: 8px; 
                border: 1px solid #e2e8f0;
                margin-top: 1rem;">
        <div style="font-size: 0.75rem; 
                   font-weight: 600; 
                   color: #475569; 
                   text-transform: uppercase; 
                   letter-spacing: 0.05em;
                   margin-bottom: 0.5rem;">
            Current Selection
        </div>
        <div style="font-size: 0.875rem; color: #64748b;">
            <div>📊 {len(selected_stats)} stats</div>
            <div style="margin-top: 0.25rem;">📋 {selected_cat.split()[1]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Apply filters
df_filtered = df.copy()

if name_filter:
    df_filtered = df_filtered[df_filtered["displayName"].str.contains(name_filter, case=False, na=False)]

if squad != "All Squads":
    df_filtered = df_filtered[df_filtered["squadName"] == squad]

if pos_group != "All Positions":
    pos_map = {
        "Defenders": ["DEFENDER", "BACK"],
        "Midfielders": ["MIDFIELD"],
        "Forwards": ["FORWARD", "WINGER"]
    }
    tokens = pos_map[pos_group]
    mask = pd.Series(False, index=df_filtered.index)
    for t in tokens:
        mask |= df_filtered["positions"].str.contains(t, case=False, na=False)
    df_filtered = df_filtered[mask]

# Results counter with better styling
st.markdown(f"""
<div style="display: inline-block;
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-left: 3px solid #3b82f6;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            margin: 1rem 0;
            font-weight: 600;
            font-size: 0.875rem;
            color: #1e40af;">
    📊 {len(df_filtered)} players match filters
</div>
""", unsafe_allow_html=True)

# Build table - ONLY PERCENTILES
cols = ["displayName", "squadName", "positions"]

# Add percentile columns only
for stat in selected_stats:
    pct_col = f"{stat}_pct"
    if pct_col in df_filtered.columns:
        cols.append(pct_col)

cols = [c for c in cols if c in df_filtered.columns]
df_display = df_filtered[cols].copy()

# Rename percentile columns to show cleaner names
rename_map = {}
seen_names = set()
for col in df_display.columns:
    if col.endswith("_pct"):
        # Extract the stat name without " (ID)" and "_pct"
        stat_name = col.replace("_pct", "")
        # Clean up the name - remove ID in parentheses
        clean_name = stat_name.split(" (")[0] if " (" in stat_name else stat_name
        
        # Handle duplicates by adding suffix
        original_clean_name = clean_name
        counter = 1
        while clean_name in seen_names:
            clean_name = f"{original_clean_name} [{counter}]"
            counter += 1
        
        seen_names.add(clean_name)
        rename_map[col] = clean_name

df_display = df_display.rename(columns=rename_map)

# Sort by first percentile stat (descending)
if len(df_display.columns) > 3:  # If we have stat columns beyond the base 3
    first_stat_col = df_display.columns[3]
    df_display = df_display.sort_values(first_stat_col, ascending=False)

# Style table - color by percentile values
styled = df_display.style

# Color the renamed columns
for col in df_display.columns:
    if col not in ["displayName", "squadName", "positions"]:
        # Map colors based on percentile values
        def color_by_percentile(val):
            if pd.isna(val):
                return 'background-color: #FFFFFF'
            color = percentile_color(val)
            # Add white text for dark backgrounds
            if val >= 60:
                return f'background-color: {color}; color: white; font-weight: 700'
            return f'background-color: {color}'
        
        styled = styled.applymap(
            color_by_percentile,
            subset=[col]
        )

# Format percentiles as integers
fmt = {}
for col in df_display.columns:
    if col not in ["displayName", "squadName", "positions"]:
        fmt[col] = '{:.0f}'  # Show as whole numbers (90, 75, etc.)

if fmt:
    styled = styled.format(fmt, na_rep="-")

# Table header section
st.markdown("""
<div style="margin-top: 2rem; margin-bottom: 1rem;">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <h3 style="font-size: 1.125rem; 
                   font-weight: 700; 
                   color: #0f172a; 
                   margin: 0;">
            📊 Player Percentile Rankings
        </h3>
        <div style="display: flex; gap: 0.5rem; align-items: center;">
            <span style="background: #f1f5f9; 
                         padding: 0.375rem 0.75rem; 
                         border-radius: 6px; 
                         font-size: 0.75rem; 
                         font-weight: 600;
                         color: #475569;">
                Percentiles Only • Sorted by Best
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Display
st.markdown("""
<style>
    /* Global Design System */
    :root {
        --primary-blue: #1e40af;
        --primary-blue-dark: #1e3a8a;
        --accent-gold: #f59e0b;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-light: #64748b;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-hover: #f1f5f9;
        --border-light: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Remove Streamlit branding completely */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Page Layout */
    .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 100% !important;
    }
    
    /* Title Section */
    h1 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.25rem !important;
        background: linear-gradient(135deg, var(--primary-blue-dark) 0%, var(--primary-blue) 50%, #2563eb 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 0.25rem !important;
        letter-spacing: -0.025em !important;
    }
    
    /* Filter Bar Enhancement */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1.5px solid var(--border-light) !important;
        border-radius: 8px !important;
        padding: 0.625rem 1rem !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        background: white !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1), var(--shadow-md) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-light) !important;
        font-weight: 400 !important;
    }
    
    /* Checkbox Enhancement */
    .stCheckbox {
        padding: 0.5rem !important;
        border-radius: 8px !important;
        background: var(--bg-secondary) !important;
        border: 1.5px solid var(--border-light) !important;
    }
    
    .stCheckbox:hover {
        background: var(--bg-hover) !important;
        border-color: var(--primary-blue) !important;
    }
    
    /* Multiselect Enhancement */
    .stMultiSelect > div {
        border: 1.5px solid var(--border-light) !important;
        border-radius: 8px !important;
        background: white !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stMultiSelect > div:focus-within {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1), var(--shadow-md) !important;
    }
    
    /* Selectbox Enhancement */
    .stSelectbox label {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Caption Enhancement */
    .stCaption {
        font-size: 0.875rem !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        padding: 0.5rem 0.75rem !important;
        background: var(--bg-secondary) !important;
        border-radius: 6px !important;
        display: inline-block !important;
        margin-top: 0.5rem !important;
    }
    
    /* Divider Enhancement */
    hr {
        margin: 2rem 0 !important;
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, var(--border-light), transparent) !important;
    }
    
    /* TABLE DESIGN - Premium Stats Website Style */
    .stDataFrame {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
        border: 1px solid var(--border-light) !important;
    }
    
    /* Table structure */
    .dataframe {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        font-size: 13px !important;
        border: none !important;
    }
    
    /* Header Row - Premium Blue Design */
    .dataframe thead tr {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%) !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 10 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    .dataframe thead th {
        padding: 16px 14px !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        color: #f1f5f9 !important;
        text-align: left !important;
        border: none !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        white-space: nowrap !important;
        position: relative !important;
    }
    
    .dataframe thead th:first-child {
        padding-left: 20px !important;
        border-left: none !important;
    }
    
    .dataframe thead th:last-child {
        border-right: none !important;
    }
    
    /* Add subtle gradient underline to headers */
    .dataframe thead th::after {
        content: '' !important;
        position: absolute !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent) !important;
    }
    
    /* Body Rows - Enhanced */
    .dataframe tbody tr {
        border-bottom: 1px solid #e2e8f0 !important;
        transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: white !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background: #f8fafc !important;
    }
    
    .dataframe tbody tr:hover {
        background: linear-gradient(90deg, #eff6ff 0%, #dbeafe 50%, #eff6ff 100%) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1) !important;
        transform: scale(1.002) !important;
        position: relative !important;
        z-index: 1 !important;
        border-color: #93c5fd !important;
    }
    
    /* Table Cells - Enhanced */
    .dataframe tbody td {
        padding: 14px 14px !important;
        border: none !important;
        border-right: 1px solid #f1f5f9 !important;
        font-size: 13px !important;
        color: var(--text-primary) !important;
        vertical-align: middle !important;
        transition: all 0.15s ease !important;
    }
    
    .dataframe tbody td:first-child {
        padding-left: 20px !important;
        border-left: none !important;
    }
    
    .dataframe tbody td:last-child {
        border-right: none !important;
    }
    
    /* Player Name Column - Sticky & Bold */
    .dataframe tbody td:first-child {
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
        position: sticky !important;
        left: 0 !important;
        z-index: 2 !important;
        background: inherit !important;
        box-shadow: 2px 0 4px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Squad Column - Secondary Info */
    .dataframe tbody td:nth-child(2) {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 12.5px !important;
    }
    
    /* Position Column - Badge Style */
    .dataframe tbody td:nth-child(3) {
        color: var(--text-light) !important;
        font-size: 11.5px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.3px !important;
    }
    
    /* Stat Columns - Right-aligned with tabular nums */
    .dataframe tbody td:nth-child(n+4) {
        text-align: center !important;
        font-variant-numeric: tabular-nums !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace !important;
    }
    
    /* Percentile columns styling */
    .dataframe tbody td[title*="pct"],
    .dataframe tbody td[title*="percentile"] {
        font-size: 10.5px !important;
        color: var(--text-light) !important;
        font-weight: 500 !important;
        font-style: italic !important;
        opacity: 0.75 !important;
    }
    
    /* Enhanced color cell borders for better definition */
    .dataframe tbody td[style*="background-color: #08519c"],
    .dataframe tbody td[style*="background-color: #3182bd"],
    .dataframe tbody td[style*="background-color: #6baed6"] {
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Download Buttons - Premium Style */
    .stDownloadButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.625rem 1.5rem !important;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3), 0 2px 4px -1px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.025em !important;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.4), 0 4px 8px -2px rgba(37, 99, 235, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    .stDownloadButton button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Scrollbar Styling */
    .stDataFrame ::-webkit-scrollbar {
        height: 10px !important;
        width: 10px !important;
    }
    
    .stDataFrame ::-webkit-scrollbar-track {
        background: #f1f5f9 !important;
        border-radius: 10px !important;
    }
    
    .stDataFrame ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #94a3b8, #64748b) !important;
        border-radius: 10px !important;
        border: 2px solid #f1f5f9 !important;
    }
    
    .stDataFrame ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #64748b, #475569) !important;
    }
    
    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .dataframe thead th {
            font-size: 10px !important;
            padding: 12px 10px !important;
        }
        
        .dataframe tbody td {
            font-size: 12px !important;
            padding: 12px 10px !important;
        }
    }
    
    /* Add subtle animation on page load */
    .stDataFrame {
        animation: fadeIn 0.4s ease-in !important;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

st.dataframe(styled, use_container_width=True, height=650, hide_index=True)

# Export section with premium styling
st.markdown("""
<div style="margin-top: 2rem; 
            padding: 1.5rem; 
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
            border-radius: 12px;
            border: 1px solid #e2e8f0;">
    <h3 style="font-size: 0.875rem; 
               font-weight: 700; 
               text-transform: uppercase; 
               letter-spacing: 0.05em; 
               color: #475569; 
               margin: 0 0 1rem 0;">
        💾 Export Data
    </h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 CSV",
        csv,
        f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True
    )

with col2:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_display.to_excel(writer, index=False)
    st.download_button(
        "📥 Excel",
        buffer.getvalue(),
        f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with col3:
    # Stats summary
    st.markdown(f"""
    <div style="background: white; 
                padding: 1rem; 
                border-radius: 8px; 
                border: 1px solid #e2e8f0;
                text-align: center;">
        <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            Current View
        </div>
        <div style="display: flex; justify-content: space-around; gap: 1rem;">
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">{len(df_display)}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Players</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">{len(selected_stats)}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Stats</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">{df_display['squadName'].nunique()}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Teams</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer with legend
st.markdown("""
<div style="margin-top: 2rem; 
            padding: 1.5rem; 
            background: white; 
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
    <h4 style="font-size: 0.875rem; 
               font-weight: 700; 
               color: #0f172a; 
               margin: 0 0 1rem 0;">
        🎨 Color Scale Guide
    </h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 24px; height: 24px; background: #08519c; border-radius: 4px;"></div>
            <span style="font-size: 0.813rem; color: #475569;"><strong>90-100th</strong> Elite</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 24px; height: 24px; background: #3182bd; border-radius: 4px;"></div>
            <span style="font-size: 0.813rem; color: #475569;"><strong>75-89th</strong> Excellent</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 24px; height: 24px; background: #6baed6; border-radius: 4px;"></div>
            <span style="font-size: 0.813rem; color: #475569;"><strong>60-74th</strong> Above Avg</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 24px; height: 24px; background: #9ecae1; border-radius: 4px;"></div>
            <span style="font-size: 0.813rem; color: #475569;"><strong>40-59th</strong> Average</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 24px; height: 24px; background: #c6dbef; border-radius: 4px;"></div>
            <span style="font-size: 0.813rem; color: #475569;"><strong>25-39th</strong> Below Avg</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 24px; height: 24px; background: #eff3ff; border-radius: 4px; border: 1px solid #e2e8f0;"></div>
            <span style="font-size: 0.813rem; color: #475569;"><strong>0-24th</strong> Poor</span>
        </div>
    </div>
    <p style="margin-top: 1rem; font-size: 0.75rem; color: #64748b; font-style: italic;">
        * For negative metrics (fouls, turnovers), the color scale is automatically inverted.
    </p>
</div>
""", unsafe_allow_html=True)