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
    /* ============================================
       GLOBAL DESIGN SYSTEM
       ============================================ */
    :root {
        --primary: #0f172a;
        --primary-light: #1e293b;
        --accent: #3b82f6;
        --accent-dark: #1d4ed8;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --text: #0f172a;
        --text-light: #64748b;
        --border: #e2e8f0;
        --bg: #ffffff;
        --bg-alt: #f8fafc;
    }
    
    /* Remove all Streamlit branding */
    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }
    
    .block-container {
        padding: 1rem 2rem !important;
        max-width: 100% !important;
    }
    
    /* ============================================
       PREMIUM TABLE DESIGN
       ============================================ */
    
    /* Table wrapper - card style */
    .stDataFrame {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 
            0 0 0 1px rgba(0, 0, 0, 0.05),
            0 4px 6px -1px rgba(0, 0, 0, 0.08),
            0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        border: none !important;
        background: white !important;
    }
    
    .dataframe {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border: none !important;
    }
    
    /* ============================================
       HEADER DESIGN - Dark & Modern
       ============================================ */
    .dataframe thead {
        position: sticky !important;
        top: 0 !important;
        z-index: 100 !important;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%) !important;
        box-shadow: 
            0 1px 0 0 rgba(255, 255, 255, 0.1) inset,
            0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    
    .dataframe thead th {
        padding: 18px 16px !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        color: #f8fafc !important;
        text-align: center !important;
        border: none !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        white-space: nowrap !important;
        background: transparent !important;
        position: relative !important;
    }
    
    /* First 3 columns - left aligned */
    .dataframe thead th:nth-child(1),
    .dataframe thead th:nth-child(2),
    .dataframe thead th:nth-child(3) {
        text-align: left !important;
    }
    
    /* Player name column - wider */
    .dataframe thead th:nth-child(1) {
        padding-left: 24px !important;
        min-width: 200px !important;
        border-left: none !important;
    }
    
    /* Remove right border from last column */
    .dataframe thead th:last-child {
        border-right: none !important;
    }
    
    /* Subtle gradient underline effect */
    .dataframe thead th::after {
        content: '' !important;
        position: absolute !important;
        bottom: 0 !important;
        left: 10% !important;
        right: 10% !important;
        height: 2px !important;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(59, 130, 246, 0.6) 50%, 
            transparent 100%) !important;
        opacity: 0 !important;
        transition: opacity 0.3s ease !important;
    }
    
    .dataframe thead th:hover::after {
        opacity: 1 !important;
    }
    
    /* ============================================
       BODY ROWS - Enhanced Alternating
       ============================================ */
    .dataframe tbody tr {
        border: none !important;
        border-bottom: 1px solid #f1f5f9 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: white !important;
    }
    
    .dataframe tbody tr:nth-child(odd) {
        background: #fafbfc !important;
    }
    
    .dataframe tbody tr:hover {
        background: linear-gradient(90deg, 
            #f0f9ff 0%, 
            #e0f2fe 50%, 
            #f0f9ff 100%) !important;
        transform: scale(1.002) translateY(-1px) !important;
        box-shadow: 
            0 0 0 1px #bae6fd,
            0 8px 16px -4px rgba(59, 130, 246, 0.15) !important;
        border-color: #7dd3fc !important;
        z-index: 10 !important;
        position: relative !important;
    }
    
    /* Hover effect on cells */
    .dataframe tbody tr:hover td {
        border-color: rgba(186, 230, 253, 0.5) !important;
    }
    
    /* ============================================
       TABLE CELLS - Premium Typography
       ============================================ */
    .dataframe tbody td {
        padding: 16px 16px !important;
        font-size: 13.5px !important;
        color: var(--text) !important;
        border: none !important;
        border-right: 1px solid #f1f5f9 !important;
        text-align: center !important;
        vertical-align: middle !important;
        transition: all 0.15s ease !important;
        font-variant-numeric: tabular-nums !important;
    }
    
    .dataframe tbody td:last-child {
        border-right: none !important;
    }
    
    /* ============================================
       PLAYER INFO COLUMNS - Left Side
       ============================================ */
    
    /* Player Name - Bold & Sticky */
    .dataframe tbody td:nth-child(1) {
        font-weight: 700 !important;
        font-size: 14px !important;
        color: #0f172a !important;
        text-align: left !important;
        padding-left: 24px !important;
        position: sticky !important;
        left: 0 !important;
        z-index: 20 !important;
        background: inherit !important;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.04) !important;
        min-width: 200px !important;
        max-width: 250px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Squad Name - Secondary */
    .dataframe tbody td:nth-child(2) {
        font-weight: 500 !important;
        font-size: 12.5px !important;
        color: #64748b !important;
        text-align: left !important;
        max-width: 180px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Position - Badge Style */
    .dataframe tbody td:nth-child(3) {
        font-weight: 600 !important;
        font-size: 10.5px !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        text-align: left !important;
        opacity: 0.85 !important;
    }
    
    /* ============================================
       PERCENTILE CELLS - Premium Badges
       ============================================ */
    .dataframe tbody td:nth-child(n+4) {
        font-weight: 700 !important;
        font-size: 14px !important;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace !important;
        border-radius: 6px !important;
        margin: 2px !important;
        position: relative !important;
    }
    
    /* Elite percentiles - Add subtle glow */
    .dataframe tbody td[style*="background-color: #08519c"],
    .dataframe tbody td[style*="background-color: #3182bd"],
    .dataframe tbody td[style*="background-color: #6baed6"] {
        box-shadow: 
            inset 0 1px 0 0 rgba(255, 255, 255, 0.25),
            0 2px 8px rgba(59, 130, 246, 0.3) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
    }
    
    /* Add shine effect on hover for colored cells */
    .dataframe tbody tr:hover td[style*="background-color: #08519c"],
    .dataframe tbody tr:hover td[style*="background-color: #3182bd"],
    .dataframe tbody tr:hover td[style*="background-color: #6baed6"] {
        box-shadow: 
            inset 0 1px 0 0 rgba(255, 255, 255, 0.4),
            0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Lower percentiles - subtle styling */
    .dataframe tbody td[style*="background-color: #9ecae1"],
    .dataframe tbody td[style*="background-color: #c6dbef"],
    .dataframe tbody td[style*="background-color: #eff3ff"] {
        border: 1px solid rgba(59, 130, 246, 0.15) !important;
        font-weight: 600 !important;
    }
    
    /* ============================================
       SCROLLBAR - Custom Design
       ============================================ */
    .stDataFrame ::-webkit-scrollbar {
        height: 12px !important;
        width: 12px !important;
    }
    
    .stDataFrame ::-webkit-scrollbar-track {
        background: #f1f5f9 !important;
        border-radius: 100px !important;
        margin: 8px !important;
    }
    
    .stDataFrame ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #94a3b8, #64748b) !important;
        border-radius: 100px !important;
        border: 3px solid #f1f5f9 !important;
        transition: all 0.2s ease !important;
    }
    
    .stDataFrame ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #64748b, #475569) !important;
        border-width: 2px !important;
    }
    
    /* ============================================
       SIDEBAR ENHANCEMENTS
       ============================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h5 {
        color: #0f172a !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stSelectbox select {
        background: white !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input:focus,
    section[data-testid="stSidebar"] .stSelectbox select:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    section[data-testid="stSidebar"] .stMultiSelect {
        background: white !important;
        border-radius: 8px !important;
    }
    
    /* ============================================
       BUTTON ENHANCEMENTS
       ============================================ */
    .stDownloadButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px !important;
        box-shadow: 
            0 1px 0 0 rgba(255, 255, 255, 0.2) inset,
            0 4px 12px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 
            0 1px 0 0 rgba(255, 255, 255, 0.3) inset,
            0 6px 20px rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    .stDownloadButton button:active {
        transform: translateY(0) !important;
        box-shadow: 
            0 1px 0 0 rgba(255, 255, 255, 0.2) inset,
            0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* ============================================
       ANIMATIONS
       ============================================ */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stDataFrame {
        animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Add subtle pulse to elite cells */
    @keyframes pulse {
        0%, 100% {
            box-shadow: 
                inset 0 1px 0 0 rgba(255, 255, 255, 0.25),
                0 2px 8px rgba(59, 130, 246, 0.3);
        }
        50% {
            box-shadow: 
                inset 0 1px 0 0 rgba(255, 255, 255, 0.35),
                0 4px 16px rgba(59, 130, 246, 0.4);
        }
    }
    
    .dataframe tbody tr:hover td[style*="background-color: #08519c"] {
        animation: pulse 2s ease-in-out infinite !important;
    }
    
    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */
    @media (max-width: 1200px) {
        .dataframe thead th,
        .dataframe tbody td {
            padding: 12px 10px !important;
            font-size: 12px !important;
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