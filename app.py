"""
Scout Lab Pro - Modern Scouting Platform
Clean, professional UI focused on the scouting workflow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.stats import zscore
import datetime as dt
import warnings
import io

warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Scout Lab Pro",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="collapsed"
)

# =====================================================
# POSITION CONFIGURATIONS
# =====================================================
POSITION_CONFIG = {
    "GK": {
        "file": "Goalkeepers.xlsx",
        "title": "Goalkeepers",
        "icon": "🧤",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low pass", "Diagonal pass", "Prevented Goals Percent (based on post-shot xG)"],
    },
    "CB": {
        "file": "Central_Defenders.xlsx",
        "title": "Central Defenders",
        "icon": "🛡️",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low pass", "Diagonal pass", "Ground duel", "Defensive Header", "Interception"],
    },
    "LB": {
        "file": "Left_Back.xlsx",
        "title": "Left Backs",
        "icon": "⬅️",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low pass", "High Cross", "Low Cross"],
    },
    "RB": {
        "file": "Right_Back.xlsx",
        "title": "Right Backs",
        "icon": "➡️",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low pass", "High Cross", "Low Cross"],
    },
    "DM": {
        "file": "Defensive_Midfielder.xlsx",
        "title": "Defensive Midfielders",
        "icon": "⚓",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low pass", "Diagonal pass", "Ground duel", "Interception"],
    },
    "CM": {
        "file": "Central_Midfielder.xlsx",
        "title": "Central Midfielders",
        "icon": "⭐",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low pass", "Diagonal pass", "Availability Between the Lines"],
    },
    "AM": {
        "file": "Attacking_Midfielder.xlsx",
        "title": "Attacking Midfielders",
        "icon": "🎯",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low pass", "Dribble", "Availability Between the Lines"],
    },
    "LW": {
        "file": "Left_Winger.xlsx",
        "title": "Left Wingers",
        "icon": "⚡",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low Cross", "Dribble", "Availability in the Box"],
    },
    "RW": {
        "file": "Right_Wing.xlsx",
        "title": "Right Wingers",
        "icon": "⚡",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Low Cross", "Dribble", "Availability in the Box"],
    },
    "ST": {
        "file": "Strikers.xlsx",
        "title": "Strikers",
        "icon": "⚽",
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Availability in the Box", "Close range shot", "Header shot"],
    }
}

NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# =====================================================
# MODERN CSS
# =====================================================
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern color palette */
    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --bg-hover: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
    }
    
    /* Main app */
    .stApp {
        background: var(--bg-dark);
        color: var(--text-primary);
    }
    
    /* Container styling */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: var(--bg-dark);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.875rem;
    }
    
    /* Top navigation bar */
    .top-nav {
        background: var(--bg-card);
        border-bottom: 2px solid var(--primary);
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 0;
    }
    
    .nav-brand {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-position {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: var(--bg-dark);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    
    /* Dashboard */
    .dashboard-header {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Chart container */
    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .chart-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--text-primary);
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.5rem;
        font-weight: 700;
        transition: all 0.2s ease;
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .stButton > button:hover {
        background: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    
    button[kind="primary"] {
        background: var(--primary) !important;
    }
    
    button[kind="secondary"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiselect > div > div > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--bg-card);
    }
    
    /* Results header */
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border);
    }
    
    .results-count {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: var(--text-secondary);
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    /* Markdown headers in containers */
    h3 {
        color: var(--text-primary);
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    /* Horizontal rule */
    hr {
        border-color: var(--border);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def safe_float(x):
    if x is None or pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("%", "").replace(",", "")
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def safe_fmt(x, decimals=1):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"

def percentile_rank(s):
    s = s.apply(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    if mask.sum() > 0:
        out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def get_percentile_color(pct):
    if pct >= 80:
        return "#10b981"  # green
    elif pct >= 60:
        return "#3b82f6"  # blue
    elif pct >= 40:
        return "#f59e0b"  # orange
    else:
        return "#ef4444"  # red

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data(show_spinner=False)
def load_data(position_key):
    cfg = POSITION_CONFIG[position_key]
    
    possible_paths = [
        Path(cfg["file"]),
        Path("/mnt/user-data/uploads") / cfg["file"],
        Path("uploads") / cfg["file"],
    ]
    
    fp = None
    for path in possible_paths:
        if path.exists():
            fp = path
            break
    
    if fp is None:
        st.error(f"❌ File not found: `{cfg['file']}`")
        st.stop()

    df = pd.read_excel(fp)
    df.columns = [str(c).strip() for c in df.columns]

    # Get numeric columns
    numeric_cols = []
    for col in df.columns:
        if col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL, "Player-ID"]:
            continue
        if "BetterThan" in col:
            continue
        numeric_cols.append(col)

    # Convert to numeric
    for c in numeric_cols + [AGE_COL, SHARE_COL]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    # Clean text
    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace("nan", "").str.strip()

    # Percentiles
    for m in numeric_cols:
        if m in df.columns:
            df[m + " (pct)"] = percentile_rank(df[m])

    return df, cfg, numeric_cols

# =====================================================
# STATE
# =====================================================
def init_state():
    if "view" not in st.session_state:
        st.session_state.view = "search"
    if "selected_player" not in st.session_state:
        st.session_state.selected_player = None
    if "position" not in st.session_state:
        st.session_state.position = "ST"
    if "filters" not in st.session_state:
        st.session_state.filters = {}

init_state()

# =====================================================
# NAVIGATION BAR
# =====================================================
def render_nav():
    cfg = POSITION_CONFIG[st.session_state.position]
    
    st.markdown(f"""
    <div class="top-nav">
        <div class="nav-brand">
            ⚽ Scout Lab Pro
        </div>
        <div class="nav-position">
            <span style="font-size: 1.5rem;">{cfg['icon']}</span>
            <span style="font-weight: 600;">{cfg['title']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# SEARCH VIEW
# =====================================================
def render_search_view(df, cfg, all_metrics):
    # Position selector
    st.markdown("### Select Position")
    cols = st.columns(5)
    
    positions = list(POSITION_CONFIG.keys())
    for i, pos_key in enumerate(positions):
        pos_cfg = POSITION_CONFIG[pos_key]
        col_idx = i % 5
        
        with cols[col_idx]:
            if st.button(
                f"{pos_cfg['icon']} {pos_cfg['title']}", 
                key=f"pos_{pos_key}",
                use_container_width=True,
                type="primary" if pos_key == st.session_state.position else "secondary"
            ):
                st.session_state.position = pos_key
                st.rerun()
    
    st.markdown("---")
    
    # Filters
    st.markdown("### Search Filters")
    
    with st.container():
        fc1, fc2, fc3, fc4 = st.columns(4)
        
        with fc1:
            search = st.text_input("🔍 Search Player", placeholder="Name, team, league...")
        
        with fc2:
            min_age, max_age = 15, 45
            if AGE_COL in df.columns:
                vals = df[AGE_COL].dropna()
                if len(vals):
                    min_age = int(max(15, np.floor(vals.min())))
                    max_age = int(min(45, np.ceil(vals.max())))
            age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
        
        with fc3:
            min_share = st.slider("Min Match Share %", 0.0, 50.0, 0.0, 1.0)
        
        with fc4:
            if COMP_COL in df.columns:
                comps = sorted([c for c in df[COMP_COL].dropna().unique() if str(c).strip()])
                selected_comps = st.multiselect("Competition", comps, key="comp_filter")
            else:
                selected_comps = []
    
    # Apply filters
    df_filtered = df.copy()
    
    if search:
        mask = pd.Series(False, index=df_filtered.index)
        for col in [NAME_COL, TEAM_COL, COMP_COL]:
            if col in df_filtered.columns:
                mask = mask | df_filtered[col].astype(str).str.lower().str.contains(search.lower(), na=False, regex=False)
        df_filtered = df_filtered[mask]
    
    if AGE_COL in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered[AGE_COL].fillna(age_range[0]) >= age_range[0]) &
            (df_filtered[AGE_COL].fillna(age_range[1]) <= age_range[1])
        ]
    
    if SHARE_COL in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[SHARE_COL].fillna(0) >= min_share]
    
    if selected_comps and COMP_COL in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[COMP_COL].isin(selected_comps)]
    
    # Sort
    sort_col = st.selectbox("Sort by", ["IMPECT", "Offensive IMPECT", "Defensive IMPECT", "Age", "Match Share"], key="sort")
    if sort_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(sort_col, ascending=(sort_col == "Age"))
    
    st.markdown("---")
    
    # Results
    st.markdown(f"""
    <div class="results-header">
        <div class="results-count">
            {len(df_filtered)} Players Found
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if df_filtered.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <h2>No players found</h2>
            <p>Try adjusting your filters</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display players
        for idx, (_, row) in enumerate(df_filtered.head(50).iterrows()):
            render_player_card(row, cfg)

def render_player_card(row, cfg):
    name = str(row.get(NAME_COL, "—"))
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    age = safe_fmt(row.get(AGE_COL, 0), 0)
    nat = str(row.get(NAT_COL, "—"))
    share = safe_fmt(row.get(SHARE_COL, 0), 1)
    
    impect = safe_float(row.get("IMPECT", 0))
    impect_str = safe_fmt(impect, 2)
    
    off_impect = safe_fmt(row.get("Offensive IMPECT", 0), 2)
    def_impect = safe_fmt(row.get("Defensive IMPECT", 0), 2)
    
    # Create container for this player
    with st.container():
        # Player info section
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {name}")
            st.markdown(f"🏟️ {team} • 🏆 {comp} • 🌍 {nat} • 👤 {age} years • ⏱️ {share}% share")
        
        with col2:
            st.metric("IMPECT", impect_str)
        
        # Metrics grid
        metric_cols = st.columns(3)
        
        for idx, metric in enumerate(cfg.get("key_metrics", [])[:6]):
            if metric in row and metric + " (pct)" in row:
                val = safe_fmt(row.get(metric, 0), 1)
                pct = safe_float(row.get(metric + " (pct)", 0))
                
                col_idx = idx % 3
                with metric_cols[col_idx]:
                    st.metric(
                        label=metric[:20],
                        value=val,
                        delta=f"{pct:.0f}th %ile"
                    )
        
        # View button
        if st.button("👁️ View Player Dashboard", key=f"view_{name.replace(' ', '_').replace('.', '_')}", use_container_width=True):
            st.session_state.selected_player = name
            st.session_state.view = "dashboard"
            st.rerun()
        
        st.markdown("---")

# =====================================================
# DASHBOARD VIEW
# =====================================================
def render_dashboard_view(df, cfg, all_metrics):
    player_name = st.session_state.selected_player
    
    # Back button
    if st.button("← Back to Search", key="back_btn"):
        st.session_state.view = "search"
        st.rerun()
    
    # Get player data
    player_data = df[df[NAME_COL] == player_name]
    if player_data.empty:
        st.error("Player not found")
        return
    
    row = player_data.iloc[0]
    
    # Header
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    age = safe_fmt(row.get(AGE_COL, 0), 0)
    nat = str(row.get(NAT_COL, "—"))
    share = safe_fmt(row.get(SHARE_COL, 0), 1)
    
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="dashboard-title">{player_name}</div>
        <div class="dashboard-subtitle">
            {team} • {comp} • {nat} • {age} years old • {share}% match share
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key stats
    impect = safe_fmt(row.get("IMPECT", 0), 2)
    off_impect = safe_fmt(row.get("Offensive IMPECT", 0), 2)
    def_impect = safe_fmt(row.get("Defensive IMPECT", 0), 2)
    
    impect_pct = safe_fmt(row.get("IMPECT (pct)", 0), 0)
    off_pct = safe_fmt(row.get("Offensive IMPECT (pct)", 0), 0)
    def_pct = safe_fmt(row.get("Defensive IMPECT (pct)", 0), 0)
    
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{impect}</div>
            <div class="stat-label">IMPECT</div>
            <div style="color: #94a3b8; margin-top: 0.5rem;">{impect_pct}th percentile</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{off_impect}</div>
            <div class="stat-label">Offensive IMPECT</div>
            <div style="color: #94a3b8; margin-top: 0.5rem;">{off_pct}th percentile</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{def_impect}</div>
            <div class="stat-label">Defensive IMPECT</div>
            <div style="color: #94a3b8; margin-top: 0.5rem;">{def_pct}th percentile</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{age}</div>
            <div class="stat-label">Age</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{share}%</div>
            <div class="stat-label">Match Share</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Performance Metrics (Percentiles)</div>', unsafe_allow_html=True)
        
        # Radar chart
        metrics = cfg.get("key_metrics", all_metrics[:8])
        values = []
        labels = []
        
        for m in metrics:
            pct_col = m + " (pct)"
            if pct_col in row:
                pct = safe_float(row.get(pct_col, 0))
                if not np.isnan(pct):
                    values.append(pct)
                    labels.append(m[:25])
        
        if values:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                fillcolor='rgba(37, 99, 235, 0.3)',
                line=dict(color='#2563eb', width=2)
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 100], showgrid=True, gridcolor='#334155'),
                    bgcolor='#0f172a'
                ),
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc', size=11),
                showlegend=False,
                margin=dict(t=20, b=20, l=40, r=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Metric Breakdown</div>', unsafe_allow_html=True)
        
        # Bar chart
        metric_data = []
        for m in metrics:
            if m in row:
                val = safe_float(row.get(m, 0))
                pct = safe_float(row.get(m + " (pct)", 0))
                if not np.isnan(val):
                    metric_data.append({
                        "Metric": m[:25],
                        "Value": val,
                        "Percentile": pct
                    })
        
        if metric_data:
            df_metrics = pd.DataFrame(metric_data)
            
            fig = px.bar(
                df_metrics,
                x="Value",
                y="Metric",
                color="Percentile",
                orientation='h',
                color_continuous_scale=[[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#10b981']],
                range_color=[0, 100]
            )
            
            fig.update_layout(
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                yaxis=dict(categoryorder='total ascending'),
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Role metrics (if available)
    role_cols = []
    for col in df.columns:
        if any(x in col for x in ["Score", "GK", "CB", "FB", "Midfielder", "Winger", "Forward", "Striker"]):
            if col not in ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"] and "BetterThan" not in col:
                role_cols.append(col)
    
    if role_cols:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Role Suitability</div>', unsafe_allow_html=True)
        
        role_data = []
        for rc in role_cols[:7]:
            if rc in row:
                val = safe_float(row.get(rc, 0))
                if not np.isnan(val):
                    role_data.append({
                        "Role": rc.replace("Score", "").strip()[:30],
                        "Score": val
                    })
        
        if role_data:
            df_roles = pd.DataFrame(role_data).sort_values("Score", ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_roles["Role"],
                x=df_roles["Score"],
                orientation='h',
                marker=dict(
                    color=df_roles["Score"],
                    colorscale=[[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#10b981']],
                    cmin=0,
                    cmax=100,
                    showscale=False
                )
            ))
            
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                margin=dict(t=20, b=20, l=10, r=10),
                xaxis=dict(range=[0, 100], gridcolor='#334155'),
                yaxis=dict(gridcolor='#334155')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # All metrics table
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Complete Statistics</div>', unsafe_allow_html=True)
    
    all_stats = []
    for col in all_metrics:
        if col in row and col + " (pct)" in row:
            val = safe_float(row.get(col, 0))
            pct = safe_float(row.get(col + " (pct)", 0))
            if not np.isnan(val):
                all_stats.append({
                    "Metric": col,
                    "Value": round(val, 2),
                    "Percentile": round(pct, 0) if not np.isnan(pct) else 0
                })
    
    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        st.dataframe(
            df_stats,
            use_container_width=True,
            height=400,
            column_config={
                "Percentile": st.column_config.ProgressColumn(
                    "Percentile",
                    min_value=0,
                    max_value=100,
                    format="%d%%"
                )
            }
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button
    st.markdown("---")
    if st.button("📥 Download Player Report (PDF)", type="primary", use_container_width=True):
        st.info("PDF download functionality - integrate with reportlab or weasyprint for production")

# =====================================================
# MAIN APP
# =====================================================
def main():
    render_nav()
    
    # Load data
    with st.spinner("Loading data..."):
        df, cfg, all_metrics = load_data(st.session_state.position)
    
    # Route to view
    if st.session_state.view == "search":
        render_search_view(df, cfg, all_metrics)
    elif st.session_state.view == "dashboard":
        render_dashboard_view(df, cfg, all_metrics)

if __name__ == "__main__":
    main()
