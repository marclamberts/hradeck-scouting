import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import hashlib
import datetime as dt
from typing import List, Dict, Tuple, Optional

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Scout Lab Pro",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# =====================================================
# NAVIGATION STATE
# =====================================================
def ensure_navigation_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "landing"
    if "show_scout_app" not in st.session_state:
        st.session_state.show_scout_app = False

ensure_navigation_state()

# =====================================================
# ENHANCED COLOR PALETTE & DESIGN TOKENS
# =====================================================
COLORS = {
    "primary": "#00D9FF",      # Electric cyan
    "secondary": "#FF6B9D",    # Vibrant pink
    "success": "#00F5A0",      # Neon green
    "warning": "#FFD93D",      # Electric yellow
    "danger": "#FF4757",       # Red accent
    "dark": "#0A0E27",         # Deep navy
    "darker": "#050816",       # Almost black
    "card": "#151B3B",         # Card background
    "card_hover": "#1A2142",   # Card hover state
    "border": "#1E2749",       # Subtle borders
    "border_hover": "#2A3458", # Hover borders
    "text": "#E8EAED",         # Light text
    "text_muted": "#8B92B0",   # Muted text
    "text_accent": "#B8BCC8",  # Accent text
    "overlay": "rgba(5, 8, 22, 0.95)",
    "glass": "rgba(21, 27, 59, 0.8)",
}

# =====================================================
# POSITION CONFIGURATIONS
# =====================================================
POSITION_CONFIG = {
    "GK": {
        "file": "Goalkeepers.xlsx",
        "title": "Goalkeepers",
        "icon": "🧤",
        "color": "#FF6B9D",
        "role_prefix": ["Ball Playing GK", "Box Defender", "Shot Stopper", "Sweeper Keeper"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "CB": {
        "file": "Central Defenders.xlsx",
        "title": "Central Defenders",
        "icon": "🛡️",
        "color": "#00F5A0",
        "role_prefix": ["Aerially Dominant CB", "Aggressive CB", "Ball Playing CB", "Strategic CB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "LB": {
        "file": "Left Back.xlsx",
        "title": "Left Backs",
        "icon": "⬅️",
        "color": "#FFD93D",
        "role_prefix": ["Attacking FB", "Defensive FB", "Progressive FB", "Inverted FB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "RB": {
        "file": "Right Back.xlsx",
        "title": "Right Backs",
        "icon": "➡️",
        "color": "#FF4757",
        "role_prefix": ["Attacking FB", "Defensive FB", "Progressive FB", "Inverted FB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "DM": {
        "file": "Defensive Midfielder.xlsx",
        "title": "Defensive Midfielders",
        "icon": "⚓",
        "color": "#9C88FF",
        "role_prefix": ["Anchorman", "Ball Winning Midfielder", "Deep Lying Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "CM": {
        "file": "Central Midfielder.xlsx",
        "title": "Central Midfielders",
        "icon": "⭐",
        "color": "#00D9FF",
        "role_prefix": ["Anchorman", "Ball Winning Midfielder", "Box-to-Box Midfielder", "Central Creator", "Deep Lying Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "AM": {
        "file": "Attacking Midfielder.xlsx",
        "title": "Attacking Midfielders",
        "icon": "🎯",
        "color": "#FFA502",
        "role_prefix": ["Advanced Playmaker", "Central Creator", "Shadow Striker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "LW": {
        "file": "Left Winger.xlsx",
        "title": "Left Wingers",
        "icon": "⚡",
        "color": "#7bed9f",
        "role_prefix": ["Inside Forward", "Touchline Winger", "Wide Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "RW": {
        "file": "Right Wing.xlsx",
        "title": "Right Wingers",
        "icon": "⚡",
        "color": "#70a1ff",
        "role_prefix": ["Inside Forward", "Touchline Winger", "Wide Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "ST": {
        "file": "Strikers.xlsx",
        "title": "Strikers",
        "icon": "⚽",
        "color": "#ff7675",
        "role_prefix": ["Complete Forward", "Deep Lying Striker", "Deep Running Striker", "Poacher", "Pressing Striker", "Second Striker", "Target Man"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
}

# Column names
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"
ID_COL = "Player-ID"

# =====================================================
# LANDING PAGE
# =====================================================
def render_landing_page():
    st.markdown(generate_enhanced_css(), unsafe_allow_html=True)
    
    # Hero Section
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 50%, var(--darker) 100%);
        padding: 4rem 2rem;
        text-align: center;
        border-radius: var(--radius-lg);
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        border: 1px solid var(--border);
        backdrop-filter: blur(20px);
    ">
        <div style="
            position: absolute;
            inset: 0;
            background: 
                radial-gradient(circle at 20% 20%, rgba(0,217,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(255,107,157,0.1) 0%, transparent 50%);
            animation: float 20s ease-in-out infinite;
        "></div>
        <div style="position: relative; z-index: 2;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">⚽</div>
            <h1 style="
                font-size: clamp(3rem, 6vw, 5rem);
                font-weight: 900;
                background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
                letter-spacing: -0.02em;
            ">Scout Lab Pro</h1>
            <p style="
                font-size: 1.25rem;
                color: var(--text-accent);
                margin-bottom: 2rem;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.6;
            ">
                Advanced Football Analytics Platform for Professional Scouting and Player Analysis
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.markdown(f'''
            <div class="metric-card hover-glow">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{len(POSITION_CONFIG)}</div>
                <div class="metric-label">Positions Covered</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown(f'''
            <div class="metric-card hover-glow">
                <div style="font-size: 2rem; margin-bottom: 0.5rem; color: {COLORS["success"]};">50+</div>
                <div class="metric-label">Performance Metrics</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown(f'''
            <div class="metric-card hover-glow">
                <div style="font-size: 2rem; margin-bottom: 0.5rem; color: {COLORS["warning"]};">∞</div>
                <div class="metric-label">Player Comparisons</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with stat_col4:
        st.markdown(f'''
            <div class="metric-card hover-glow">
                <div style="font-size: 2rem; margin-bottom: 0.5rem; color: {COLORS["secondary"]};">AI</div>
                <div class="metric-label">Powered Analysis</div>
            </div>
        ''', unsafe_allow_html=True)
    
    # Features Section
    st.markdown("## 🚀 Platform Features")
    
    feature_col1, feature_col2 = st.columns(2, gap="large")
    
    with feature_col1:
        st.markdown(f'''
            <div class="modern-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="width: 60px; height: 60px; background: linear-gradient(135deg, {COLORS["primary"]}40, {COLORS["secondary"]}40); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                        🔍
                    </div>
                    <h3 style="margin: 0; color: var(--text);">Advanced Player Search</h3>
                </div>
                <p style="color: var(--text-muted); line-height: 1.6;">
                    Powerful filtering system to discover players by position, age, performance metrics, 
                    teams, competitions, and custom criteria. Find your next signing with precision.
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
            <div class="modern-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="width: 60px; height: 60px; background: linear-gradient(135deg, {COLORS["success"]}40, {COLORS["primary"]}40); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                        📊
                    </div>
                    <h3 style="margin: 0; color: var(--text);">Performance Analytics</h3>
                </div>
                <p style="color: var(--text-muted); line-height: 1.6;">
                    Comprehensive statistical analysis with IMPECT scores, role suitability ratings, 
                    strengths/weaknesses identification, and percentile rankings vs. peers.
                </p>
            </div>
        ''', unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown(f'''
            <div class="modern-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="width: 60px; height: 60px; background: linear-gradient(135deg, {COLORS["warning"]}40, {COLORS["success"]}40); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                        ⚖️
                    </div>
                    <h3 style="margin: 0; color: var(--text);">Player Comparisons</h3>
                </div>
                <p style="color: var(--text-muted); line-height: 1.6;">
                    Side-by-side analysis of multiple players with interactive radar charts, 
                    head-to-head statistics, and detailed performance breakdowns.
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
            <div class="modern-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="width: 60px; height: 60px; background: linear-gradient(135deg, {COLORS["secondary"]}40, {COLORS["warning"]}40); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                        ⭐
                    </div>
                    <h3 style="margin: 0; color: var(--text);">Smart Shortlisting</h3>
                </div>
                <p style="color: var(--text-muted); line-height: 1.6;">
                    Create and manage player shortlists with notes, tags, and export capabilities. 
                    Never lose track of promising talent again.
                </p>
            </div>
        ''', unsafe_allow_html=True)
    
    # Positions Grid
    st.markdown("## ⚽ Available Positions")
    
    pos_cols = st.columns(5)
    for idx, (pos_key, pos_config) in enumerate(POSITION_CONFIG.items()):
        with pos_cols[idx % 5]:
            st.markdown(f'''
                <div style="
                    padding: 1.5rem;
                    background: var(--glass);
                    border: 1px solid var(--border);
                    border-radius: var(--radius-lg);
                    text-align: center;
                    transition: all var(--duration-normal) var(--easing);
                    cursor: pointer;
                    backdrop-filter: blur(20px);
                " onmouseover="this.style.borderColor='{pos_config["color"]}'; this.style.transform='translateY(-4px)'" onmouseout="this.style.borderColor='var(--border)'; this.style.transform='translateY(0px)'">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{pos_config["icon"]}</div>
                    <div style="font-weight: 700; color: var(--text); margin-bottom: 0.25rem;">{pos_key}</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">{pos_config["title"]}</div>
                </div>
            ''', unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f'''
            <div style="text-align: center; padding: 2rem;">
                <h3 style="margin-bottom: 1.5rem; color: var(--text);">Ready to Start Scouting?</h3>
                <p style="color: var(--text-muted); margin-bottom: 2rem; max-width: 400px; margin-left: auto; margin-right: auto;">
                    Launch the full scouting platform and discover your next star player with advanced analytics and AI-powered insights.
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Launch Scout Lab Pro", width="stretch", type="primary"):
            st.session_state.current_page = "scout_app"
            st.session_state.show_scout_app = True
            st.rerun()

# =====================================================
# NAVIGATION
# =====================================================
def render_navigation():
    """Render top navigation bar"""
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        position: sticky;
        top: 0;
        z-index: 1000;
    ">
        <div style="display: flex; align-items: center; gap: 2rem;">
            <div style="font-size: 1.5rem; font-weight: 900; background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚽ Scout Lab Pro
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="padding: 0.5rem 1rem; background: rgba(0,217,255,0.2); border: 1px solid rgba(0,217,255,0.4); border-radius: 20px; color: #00D9FF; font-weight: 700; font-size: 0.8rem;">
                v2.0 Enhanced
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col2:
        nav_buttons = st.columns(3)
        
        with nav_buttons[0]:
            if st.button("🏠 Home", width="stretch", type="secondary" if st.session_state.current_page != "landing" else "primary"):
                st.session_state.current_page = "landing"
                st.session_state.show_scout_app = False
                st.rerun()
        
        with nav_buttons[1]:
            if st.button("⚽ Scout Platform", width="stretch", type="secondary" if st.session_state.current_page != "scout_app" else "primary"):
                st.session_state.current_page = "scout_app"
                st.session_state.show_scout_app = True
                st.rerun()
        
        with nav_buttons[2]:
            if st.button("📖 About", width="stretch", type="secondary" if st.session_state.current_page != "about" else "primary"):
                st.session_state.current_page = "about"
                st.session_state.show_scout_app = False
                st.rerun()

# =====================================================
# ABOUT PAGE
# =====================================================
def render_about_page():
    st.markdown(generate_enhanced_css(), unsafe_allow_html=True)
    
    st.markdown(f'''
        <div class="modern-card" style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚽</div>
            <h1 style="background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">About Scout Lab Pro</h1>
            <p style="color: var(--text-muted); font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                Professional Football Analytics Platform for Modern Scouting
            </p>
        </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('''
            <div class="modern-card">
                <h3>🎯 Mission</h3>
                <p style="color: var(--text-muted); line-height: 1.6;">
                    Scout Lab Pro revolutionizes football scouting by providing advanced analytics, 
                    AI-powered insights, and comprehensive player analysis tools for professionals 
                    in the beautiful game.
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
            <div class="modern-card">
                <h3>🔬 Technology</h3>
                <p style="color: var(--text-muted); line-height: 1.6;">
                    Built with cutting-edge web technologies including Streamlit, Plotly for 
                    interactive visualizations, and advanced statistical analysis algorithms 
                    for meaningful player insights.
                </p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
            <div class="modern-card">
                <h3>📊 Features</h3>
                <ul style="color: var(--text-muted); line-height: 1.6;">
                    <li>Advanced player search and filtering</li>
                    <li>Comprehensive performance analytics</li>
                    <li>Interactive player comparisons</li>
                    <li>Role suitability analysis</li>
                    <li>Smart shortlisting system</li>
                    <li>Export and reporting tools</li>
                </ul>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
            <div class="modern-card">
                <h3>🎨 Design</h3>
                <p style="color: var(--text-muted); line-height: 1.6;">
                    Features a modern glass morphism design with smooth animations, 
                    responsive layout, and position-specific color schemes for 
                    an intuitive user experience.
                </p>
            </div>
        ''', unsafe_allow_html=True)
# =====================================================
def generate_enhanced_css():
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&display=swap');

:root {{
    --primary: {COLORS["primary"]};
    --secondary: {COLORS["secondary"]};
    --success: {COLORS["success"]};
    --warning: {COLORS["warning"]};
    --danger: {COLORS["danger"]};
    --dark: {COLORS["dark"]};
    --darker: {COLORS["darker"]};
    --card: {COLORS["card"]};
    --card-hover: {COLORS["card_hover"]};
    --border: {COLORS["border"]};
    --border-hover: {COLORS["border_hover"]};
    --text: {COLORS["text"]};
    --text-muted: {COLORS["text_muted"]};
    --text-accent: {COLORS["text_accent"]};
    --glass: {COLORS["glass"]};
    
    --easing: cubic-bezier(0.4, 0, 0.2, 1);
    --duration-normal: 0.25s;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-full: 50px;
    --shadow-medium: 0 4px 20px rgba(0, 0, 0, 0.2);
    --shadow-glow: 0 0 20px rgba(0, 217, 255, 0.3);
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
div[data-testid="collapsedControl"] {{ display: none; }}

* {{ box-sizing: border-box; }}

html, body {{ scroll-behavior: smooth; }}

.stApp {{
    background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 50%, var(--darker) 100%);
    color: var(--text);
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    min-height: 100vh;
    position: relative;
}}

.stApp::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(circle at 25% 25%, rgba(0,217,255,0.03) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(255,107,157,0.03) 0%, transparent 50%);
    animation: float 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}}

@keyframes float {{
    0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
    25% {{ transform: translate(20px, -20px) rotate(1deg); }}
    50% {{ transform: translate(-10px, 10px) rotate(-1deg); }}
    75% {{ transform: translate(-20px, -10px) rotate(0.5deg); }}
}}

.block-container {{
    padding: 1.5rem 2rem !important;
    max-width: 1600px !important;
    margin: 0 auto;
}}

h1, h2, h3, h4, h5, h6 {{
    color: var(--text);
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin: 0;
}}

h1 {{
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 900;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}}

h3 {{ font-size: 1.35rem; margin-bottom: 0.5rem; }}
h4 {{ font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--text-accent); }}

section[data-testid="stSidebar"] {{
    background: var(--glass);
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
    box-shadow: var(--shadow-medium);
}}

section[data-testid="stSidebar"] .block-container {{
    padding: 1rem !important;
}}

.header-bar {{
    background: var(--glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-medium);
    position: sticky;
    top: 0;
    z-index: 100;
    transition: all var(--duration-normal) var(--easing);
}}

.header-bar:hover {{
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: 1.5rem;
}}

.brand {{
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}}

.position-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(0,217,255,0.15) 0%, rgba(255,107,157,0.15) 100%);
    border: 1px solid rgba(0,217,255,0.4);
    padding: 0.5rem 1.25rem;
    border-radius: var(--radius-full);
    font-weight: 700;
    font-size: 1rem;
    color: var(--text);
    transition: all var(--duration-normal) var(--easing);
}}

.position-badge:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow);
}}

.stat-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--card);
    border: 1px solid var(--border);
    padding: 0.4rem 1rem;
    border-radius: var(--radius-full);
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    transition: all var(--duration-normal) var(--easing);
    backdrop-filter: blur(10px);
}}

.stat-pill:hover {{
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow);
}}

.stat-pill strong {{
    color: var(--primary);
    font-weight: 800;
}}

.modern-card {{
    background: var(--glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-medium);
    transition: all var(--duration-normal) var(--easing);
    position: relative;
    overflow: hidden;
}}

.modern-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    opacity: 0;
    transition: opacity var(--duration-normal) var(--easing);
}}

.modern-card:hover {{
    transform: translateY(-4px);
    box-shadow: var(--shadow-glow);
    border-color: var(--border-hover);
}}

.modern-card:hover::before {{
    opacity: 1;
}}

div.stButton > button {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: var(--darker);
    border: none;
    border-radius: var(--radius-md);
    padding: 0.6rem 1.5rem;
    font-weight: 800;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
    transition: all var(--duration-normal) var(--easing);
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
}}

div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 217, 255, 0.6);
}}

button[kind="secondary"] {{
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-medium) !important;
}}

button[kind="secondary"]:hover {{
    border-color: var(--primary) !important;
    background: var(--card-hover) !important;
    box-shadow: var(--shadow-glow) !important;
}}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text) !important;
    transition: all var(--duration-normal) var(--easing) !important;
}}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {{
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.2) !important;
    transform: translateY(-1px);
}}

button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    border-bottom: 3px solid transparent !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    transition: all var(--duration-normal) var(--easing) !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}}

button[data-baseweb="tab"]:hover {{
    color: var(--text) !important;
}}

div[data-testid="stDataFrame"] {{
    background: var(--glass) !important;
    backdrop-filter: blur(20px);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden;
    box-shadow: var(--shadow-medium);
}}

div[data-testid="stDataFrame"] thead tr th {{
    background: var(--darker) !important;
    color: var(--text) !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    border-bottom: 2px solid var(--primary) !important;
    padding: 1rem !important;
}}

div[data-testid="stDataFrame"] tbody tr {{
    transition: all 0.15s var(--easing);
}}

div[data-testid="stDataFrame"] tbody tr:hover {{
    background: var(--card-hover) !important;
    transform: scale(1.005);
}}

.section-header {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    font-weight: 900;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
    position: relative;
}}

.section-header::after {{
    content: '';
    position: absolute;
    bottom: -2px; left: 0;
    width: 60px; height: 2px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
}}

.metric-card {{
    background: var(--glass);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem;
    text-align: center;
    transition: all var(--duration-normal) var(--easing);
    position: relative;
    overflow: hidden;
}}

.metric-card:hover {{
    transform: translateY(-6px);
    box-shadow: var(--shadow-glow);
    border-color: var(--primary);
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
    line-height: 1;
}}

.metric-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 700;
}}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: var(--darker); border-radius: 4px; }}
::-webkit-scrollbar-thumb {{ 
    background: linear-gradient(180deg, var(--primary) 0%, var(--secondary) 100%); 
    border-radius: 4px; 
}}

@media (max-width: 768px) {{
    .header-bar {{ flex-direction: column; align-items: flex-start; gap: 1rem; }}
    .block-container {{ padding: 1rem !important; }}
}}
</style>
"""

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def safe_float(x):
    if x is None: return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}: return np.nan
    s = s.replace("%", "")
    if s.count(",") == 1 and s.count(".") == 0: s = s.replace(",", ".")
    if s.count(",") >= 1 and s.count(".") == 1: s = s.replace(",", "")
    try: return float(s)
    except: return np.nan

def safe_fmt(x, decimals=2):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"

def safe_int_fmt(x):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{int(round(v))}"

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(safe_float)

def percentile_rank(s: pd.Series) -> pd.Series:
    s = s.map(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def make_rowid(row: pd.Series, position: str) -> str:
    parts = [position, str(row.get(NAME_COL, "")), str(row.get(TEAM_COL, "")), str(row.get(COMP_COL, "")), str(row.name)]
    raw = "||".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

def player_meta(row: pd.Series) -> str:
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    nat = str(row.get(NAT_COL, "—"))
    age = safe_int_fmt(row.get(AGE_COL, np.nan))
    share = safe_fmt(row.get(SHARE_COL, np.nan), 1)
    return f"{team} • {comp} • {nat} • Age {age} • {share}% share"

def create_enhanced_plotly_theme():
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': COLORS["text"], 'size': 12, 'family': 'Plus Jakarta Sans, system-ui, sans-serif'},
            'colorway': [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"], COLORS["danger"]],
            'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60},
            'xaxis': {
                'gridcolor': COLORS["border"], 
                'linecolor': COLORS["border"], 
                'tickfont': {'color': COLORS["text_muted"]},
                'title': {'font': {'color': COLORS["text"]}}
            },
            'yaxis': {
                'gridcolor': COLORS["border"], 
                'linecolor': COLORS["border"], 
                'tickfont': {'color': COLORS["text_muted"]},
                'title': {'font': {'color': COLORS["text"]}}
            },
            'legend': {'bgcolor': COLORS["glass"], 'bordercolor': COLORS["border"], 'borderwidth': 1, 'font': {'color': COLORS["text"]}}
        }
    }

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data(show_spinner=False)
def load_position_data(position_key: str) -> tuple[pd.DataFrame, dict]:
    cfg = POSITION_CONFIG[position_key].copy()
    fp = Path(cfg["file"])
    
    if not fp.exists():
        raise FileNotFoundError(f"Missing {cfg['file']}")
    
    df = pd.read_excel(fp)
    df.columns = [str(c).strip() for c in df.columns]
    
    role_cols = []
    metric_cols = []
    
    for col in df.columns:
        if col in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, NAT_COL, SHARE_COL, ID_COL] or 'BetterThan' in col:
            continue
            
        if "IMPECT" in col:
            metric_cols.append(col)
            continue
        
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if sample and isinstance(sample, str) and '%' in sample:
                col_idx = df.columns.get_loc(col)
                if col_idx < 20 and any(prefix in col for prefix in cfg.get("role_prefix", [])):
                    role_cols.append(col)
                else:
                    metric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            metric_cols.append(col)
    
    all_numeric = role_cols + metric_cols + [AGE_COL, SHARE_COL]
    coerce_numeric(df, all_numeric)
    
    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()
    
    for m in metric_cols:
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
            df[m + " (pct)"] = percentile_rank(df[m])
    
    cfg["role_cols"] = role_cols
    cfg["metric_cols"] = metric_cols
    cfg["all_metrics"] = role_cols + metric_cols
    
    return df, cfg

# =====================================================
# FILTERS & STATE
# =====================================================
def default_filters_for(df: pd.DataFrame):
    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        if len(vals):
            lo = int(max(15, np.floor(vals.min())))
            hi = int(min(50, np.ceil(vals.max())))
        else:
            lo, hi = 15, 45
    else:
        lo, hi = 15, 45
    return {"q": "", "min_share": 0.0, "competitions": [], "teams": [], "nats": [], "age_range": (lo, hi)}

def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df.copy()
    
    if SHARE_COL in out.columns:
        out = out[out[SHARE_COL].fillna(0) >= float(f.get("min_share", 0.0))]
    
    if AGE_COL in out.columns and "age_range" in f:
        lo, hi = f["age_range"]
        out = out[(out[AGE_COL].fillna(lo) >= lo) & (out[AGE_COL].fillna(hi) <= hi)]
    
    if f.get("competitions") and COMP_COL in out.columns:
        out = out[out[COMP_COL].isin(f["competitions"])]
    
    if f.get("teams") and TEAM_COL in out.columns:
        out = out[out[TEAM_COL].isin(f["teams"])]
    
    if f.get("nats") and NAT_COL in out.columns:
        out = out[out[NAT_COL].isin(f["nats"])]
    
    q = str(f.get("q", "")).strip().lower()
    if q:
        mask = pd.Series(False, index=out.index)
        for col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            if col in out.columns:
                mask = mask | out[col].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]
    
    return out

def strengths_weaknesses(cfg: dict, row: pd.Series, topn: int = 5):
    pairs = []
    for m in cfg.get("metric_cols", []):
        if m in ["IMPECT - BetterThan", "Offensive IMPECT - BetterThan", "Defensive IMPECT - BetterThan"]:
            continue
        pct = safe_float(row.get(m + " (pct)", np.nan))
        if not np.isnan(pct):
            pairs.append((m, pct))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:topn]
    bottom = list(reversed(pairs[-topn:])) if len(pairs) >= topn else list(reversed(pairs))
    return top, bottom

def ensure_state():
    for key in ["filters", "shortlist", "pinned", "selected_player", "compare_picks"]:
        if key not in st.session_state:
            st.session_state[key] = {}

def shortlist_key(position_key: str, player_name: str) -> str:
    return f"{position_key}||{player_name}"

def add_to_shortlist(position_key: str, player_name: str):
    k = shortlist_key(position_key, player_name)
    if k not in st.session_state.shortlist:
        st.session_state.shortlist[k] = {"tags": "", "notes": "", "added": dt.datetime.now()}

def remove_from_shortlist(position_key: str, player_name: str):
    k = shortlist_key(position_key, player_name)
    if k in st.session_state.shortlist:
        del st.session_state.shortlist[k]

# =====================================================
# MAIN APP WITH NAVIGATION
# =====================================================
def main():
    # Always render navigation
    if st.session_state.current_page != "landing":
        render_navigation()
    
    # Route to appropriate page
    if st.session_state.current_page == "landing":
        render_landing_page()
    elif st.session_state.current_page == "about":
        render_about_page()
    elif st.session_state.current_page == "scout_app":
        render_scout_app()

def render_scout_app():
    st.markdown(generate_enhanced_css(), unsafe_allow_html=True)
    ensure_state()

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">🎯 Scout Control Center</div>', unsafe_allow_html=True)
        
        position = st.selectbox(
            "🏟️ Position",
            list(POSITION_CONFIG.keys()),
            format_func=lambda x: f"{POSITION_CONFIG[x]['icon']} {POSITION_CONFIG[x]['title']}",
            index=0
        )

    # Load data
    with st.spinner("🔄 Loading player database..."):
        try:
            df, cfg = load_position_data(position)
        except FileNotFoundError as e:
            st.error(f"📁 Data file not found: {e}")
            st.info("💡 Please ensure the Excel files are in the same directory as this script")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()

    # Initialize state
    if position not in st.session_state.filters:
        st.session_state.filters[position] = default_filters_for(df)
    if position not in st.session_state.pinned:
        st.session_state.pinned[position] = None
    if position not in st.session_state.selected_player:
        st.session_state.selected_player[position] = None
    if position not in st.session_state.compare_picks:
        st.session_state.compare_picks[position] = []

    f = st.session_state.filters[position]
    position_color = POSITION_CONFIG[position].get("color", COLORS["primary"])

    # Enhanced Sidebar Filters
    with st.sidebar:
        st.markdown('<div class="modern-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("##### 🔍 Search & Filters")
        
        f["q"] = st.text_input("🔍 Search", value=f.get("q", ""), placeholder="Player, team, competition...")
        f["min_share"] = st.slider("📊 Min Match Share (%)", 0.0, 100.0, float(f.get("min_share", 0.0)), 5.0)
        
        if AGE_COL in df.columns and len(df):
            vals = df[AGE_COL].dropna()
            min_age = int(max(15, np.floor(vals.min()))) if len(vals) else 15
            max_age = int(min(50, np.ceil(vals.max()))) if len(vals) else 45
            lo, hi = f.get("age_range", (min_age, max_age))
            f["age_range"] = st.slider("🎂 Age Range", min_age, max_age, (lo, hi), 1)
        
        if COMP_COL in df.columns:
            comps_all = sorted([c for c in df[COMP_COL].dropna().unique().tolist() if str(c).strip() != ""])
            f["competitions"] = st.multiselect("🏆 Competitions", comps_all, default=f.get("competitions", []))
        
        if TEAM_COL in df.columns:
            teams_all = sorted([t for t in df[TEAM_COL].dropna().unique().tolist() if str(t).strip() != ""])
            f["teams"] = st.multiselect("⚽ Teams", teams_all, default=f.get("teams", []))
        
        if NAT_COL in df.columns:
            nats_all = sorted([n for n in df[NAT_COL].dropna().unique().tolist() if str(n).strip() != ""])
            f["nats"] = st.multiselect("🌍 Nationalities", nats_all, default=f.get("nats", []))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", width="stretch"):
                st.session_state.filters[position] = default_filters_for(df)
                st.rerun()
        with col2:
            filter_count = sum([bool(f.get("q", "")), f.get("min_share", 0) > 0, bool(f.get("competitions", [])), bool(f.get("teams", [])), bool(f.get("nats", []))])
            st.markdown(f'<div style="text-align: center; padding: 0.5rem; background: rgba(0,217,255,0.2); border: 1px solid rgba(0,217,255,0.4); border-radius: 50px; color: #00D9FF; font-weight: 700; font-size: 0.8rem;">{filter_count} filters</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    df_f = apply_filters(df, f)
    if not df_f.empty:
        df_f = df_f.copy()
        df_f["_rowid"] = df_f.apply(lambda r: make_rowid(r, position), axis=1)

    # Set defaults
    if st.session_state.pinned[position] is None and len(df_f) and NAME_COL in df_f.columns:
        if "IMPECT" in df_f.columns:
            st.session_state.pinned[position] = df_f.sort_values("IMPECT", ascending=False).iloc[0][NAME_COL]
        else:
            st.session_state.pinned[position] = df_f.iloc[0][NAME_COL]

    if st.session_state.selected_player[position] is None and st.session_state.pinned[position] is not None:
        st.session_state.selected_player[position] = st.session_state.pinned[position]

    # Enhanced Header
    shortlist_count = len(st.session_state.shortlist)
    teams_n = df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0
    comps_n = df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0

    st.markdown(f"""
    <div class="header-bar">
        <div class="header-left">
            <div class="brand">Scout Lab Pro</div>
            <div class="position-badge" style="border-color: {position_color}40;">
                {cfg["icon"]} {cfg["title"]}
            </div>
            <div class="stat-pill">
                <span style="color: {COLORS["text_muted"]};">Players</span> 
                <strong>{len(df_f):,}</strong>
            </div>
            <div class="stat-pill">
                <span style="color: {COLORS["text_muted"]};">Teams</span> 
                <strong>{teams_n}</strong>
            </div>
        </div>
        <div style="display:flex;gap:1rem;align-items:center;">
            <div class="stat-pill" style="background: linear-gradient(135deg, {COLORS["warning"]}20 0%, {COLORS["success"]}20 100%); border-color: {COLORS["warning"]}60;">
                ⭐ Shortlist <strong>{shortlist_count}</strong>
            </div>
            <div style="padding: 0.5rem; background: rgba(0,217,255,0.2); border: 1px solid rgba(0,217,255,0.4); border-radius: 50px; color: #00D9FF; font-weight: 700; font-size: 0.8rem;">
                🕐 {dt.datetime.now().strftime("%H:%M")}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["🔍 Scout Browser", "👤 Player Profile", "⚖️ Head-to-Head", "🏆 Rankings", "📊 Analytics", "⭐ Shortlist"])

    # =====================================================
    # TAB 1: SCOUT BROWSER
    # =====================================================
    with tabs[0]:
        if df_f.empty:
            st.markdown('''
                <div class="modern-card" style="text-align: center; padding: 3rem;">
                    <h3 style="color: var(--text-muted);">🔍 No Players Found</h3>
                    <p style="color: var(--text-muted); margin-bottom: 2rem;">
                        Try adjusting your filters in the sidebar to discover more players.
                    </p>
                </div>
            ''', unsafe_allow_html=True)
        else:
            sort_options = ["IMPECT"] + cfg.get("role_cols", []) if "IMPECT" in df_f.columns else cfg.get("role_cols", [])
            sort_options = [c for c in sort_options if c in df_f.columns]
            
            if not sort_options and cfg.get("metric_cols", []):
                sort_options = [c for c in cfg.get("metric_cols", []) if c in df_f.columns][:5]
            
            if not sort_options:
                numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
                sort_options = numeric_cols[:1] if numeric_cols else [NAME_COL]
            
            col_sort, col_view = st.columns([2, 1])
            
            with col_sort:
                sort_col = st.selectbox("📊 Sort by Metric", options=sort_options, index=0)
            
            with col_view:
                view_count = st.selectbox("👁️ Show Results", options=[10, 20, 30, 50], index=2)
            
            left_col, right_col = st.columns([1.4, 0.6], gap="large")
            
            with left_col:
                st.markdown('<div class="section-header">🔍 Player Discovery</div>', unsafe_allow_html=True)
                
                results = df_f.sort_values(sort_col, ascending=False).head(view_count).copy()
                
                for rank, (_, r) in enumerate(results.iterrows(), 1):
                    name = str(r.get(NAME_COL, "—"))
                    rid = str(r.get("_rowid", r.name))
                    in_sl = shortlist_key(position, name) in st.session_state.shortlist
                    score_val = safe_fmt(r.get(sort_col, np.nan), 1)
                    
                    st.markdown(f'''
                        <div style="background: var(--glass); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1rem; margin-bottom: 1rem; transition: all var(--duration-normal) var(--easing);" onmouseover="this.style.borderColor='{COLORS["primary"]}'" onmouseout="this.style.borderColor='var(--border)'">
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div style="flex: 1;">
                                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                                        <div style="width: 2rem; height: 2rem; background: {position_color}; color: var(--darker); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 0.8rem;">#{rank}</div>
                                        <div style="font-weight: 800; font-size: 1.1rem; color: var(--text);">{name}</div>
                                    </div>
                                    <div style="color: var(--text-muted); font-size: 0.8rem; margin-left: 3rem;">{player_meta(r)}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase;">{sort_col[:15]}</div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-weight: 900; font-size: 1.5rem; color: {position_color};">{score_val}</div>
                                </div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
                    with action_col1:
                        if st.button("👁️ View", key=f"view_{position}_{rid}", width="stretch"):
                            st.session_state.pinned[position] = name
                            st.session_state.selected_player[position] = name
                            st.rerun()
                    
                    with action_col2:
                        if st.button("⭐" if not in_sl else "✓", key=f"sl_{position}_{rid}", width="stretch", type="secondary"):
                            if not in_sl:
                                add_to_shortlist(position, name)
                            else:
                                remove_from_shortlist(position, name)
                            st.rerun()
                    
                    with action_col3:
                        if st.button("➕ Compare", key=f"cmp_{position}_{rid}", width="stretch", type="secondary"):
                            picks = st.session_state.compare_picks[position]
                            if name not in picks:
                                picks.append(name)
                                st.session_state.compare_picks[position] = picks[:6]
                            st.rerun()
            
            with right_col:
                st.markdown('<div style="position: sticky; top: 140px;">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📌 Featured Player</div>', unsafe_allow_html=True)
                
                pinned = st.session_state.pinned.get(position)
                
                if not pinned:
                    st.markdown('''
                        <div class="modern-card" style="text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">👆</div>
                            <h4 style="color: var(--text-muted); margin-bottom: 0.5rem;">No Player Selected</h4>
                            <p style="color: var(--text-muted); font-size: 0.9rem;">
                                Click "View" on any player to see detailed insights here
                            </p>
                        </div>
                    ''', unsafe_allow_html=True)
                else:
                    p = df_f[df_f[NAME_COL] == pinned].head(1)
                    if p.empty:
                        st.warning("📍 Pinned player not in current filter results")
                    else:
                        row = p.iloc[0]
                        
                        st.markdown(f'''
                            <div class="modern-card">
                                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                    <div style="width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, {position_color}40, {COLORS["primary"]}40); display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                                        {cfg["icon"]}
                                    </div>
                                    <div>
                                        <h3 style="margin: 0; color: var(--text);">{pinned}</h3>
                                        <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">{player_meta(row)}</p>
                                    </div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        if "IMPECT" in row:
                            impect_val = safe_fmt(row.get("IMPECT", np.nan), 1)
                            st.markdown(f'''
                                <div class="metric-card">
                                    <div class="metric-label">IMPECT Score</div>
                                    <div class="metric-value">{impect_val}</div>
                                </div>
                            ''', unsafe_allow_html=True)
                        
                        st.markdown("#### ⬆️ Key Strengths")
                        top, _ = strengths_weaknesses(cfg, row, topn=3)
                        for m, pct in top:
                            st.markdown(f'''
                                <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem; background: var(--glass); border-radius: var(--radius-md); margin-bottom: 0.5rem; border-left: 3px solid {COLORS["success"]};">
                                    <div style="width: 8px; height: 8px; border-radius: 50%; background: {COLORS["success"]}; flex-shrink: 0;"></div>
                                    <div>
                                        <div style="font-weight: 700; font-size: 0.9rem; color: var(--text);">{m[:30]}</div>
                                        <div style="color: var(--text-muted); font-size: 0.75rem;">{pct:.0f}th percentile</div>
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)
                        
                        ac1, ac2 = st.columns(2)
                        in_sl = shortlist_key(position, pinned) in st.session_state.shortlist
                        with ac1:
                            if st.button("⭐ Shortlist" if not in_sl else "✓ Shortlisted", key="sl_pin", width="stretch"):
                                if not in_sl:
                                    add_to_shortlist(position, pinned)
                                else:
                                    remove_from_shortlist(position, pinned)
                                st.rerun()
                        with ac2:
                            if st.button("➕ Compare", key="cmp_pin", width="stretch", type="secondary"):
                                picks = st.session_state.compare_picks[position]
                                if pinned not in picks:
                                    picks.append(pinned)
                                    st.session_state.compare_picks[position] = picks[:6]
                                st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # TAB 2: PLAYER PROFILE
    # =====================================================
    with tabs[1]:
        if df_f.empty or NAME_COL not in df_f.columns:
            st.warning("⚠️ No players available with current filters.")
        else:
            players = sorted(df_f[NAME_COL].dropna().unique().tolist())
            default_player = st.session_state.selected_player.get(position) or st.session_state.pinned.get(position) or (players[0] if players else None)
            if default_player not in players and players:
                default_player = players[0]
            
            player = st.selectbox("🎯 Select Player", players, index=players.index(default_player) if default_player in players else 0)
            st.session_state.selected_player[position] = player
            
            p = df_f[df_f[NAME_COL] == player].head(1)
            row = p.iloc[0]
            
            st.markdown(f'''
                <div class="modern-card" style="margin-bottom: 2rem;">
                    <div style="display: flex; align-items: center; gap: 2rem;">
                        <div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, {position_color}60, {COLORS["primary"]}60); display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                            {cfg["icon"]}
                        </div>
                        <div>
                            <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, {COLORS["primary"]} 0%, {position_color} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{player}</h1>
                            <p style="margin: 0.5rem 0 0 0; color: var(--text-accent); font-size: 1.1rem;">{player_meta(row)}</p>
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                age_val = safe_int_fmt(row.get(AGE_COL, np.nan))
                st.markdown(f'<div class="metric-card"><div class="metric-label">Age</div><div class="metric-value">{age_val}</div></div>', unsafe_allow_html=True)
            
            with col2:
                share_val = safe_fmt(row.get(SHARE_COL, np.nan), 1)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Match Share</div><div class="metric-value">{share_val}%</div></div>', unsafe_allow_html=True)
            
            with col3:
                impect_val = safe_fmt(row.get("IMPECT", np.nan), 2)
                st.markdown(f'<div class="metric-card"><div class="metric-label">IMPECT</div><div class="metric-value">{impect_val}</div></div>', unsafe_allow_html=True)
            
            with col4:
                in_sl = shortlist_key(position, player) in st.session_state.shortlist
                if st.button("⭐ Shortlist" if not in_sl else "✓ Shortlisted", width="stretch"):
                    if not in_sl:
                        add_to_shortlist(position, player)
                    else:
                        remove_from_shortlist(position, player)
                    st.rerun()
            
            # Enhanced Strengths & Weaknesses Analysis
            st.markdown("---")
            top, bottom = strengths_weaknesses(cfg, row, topn=8)
            
            str_col, weak_col = st.columns(2, gap="large")
            
            with str_col:
                st.markdown("#### ⬆️ Standout Strengths")
                if not top:
                    st.info("No strength metrics available with current data")
                else:
                    for i, (m, pct) in enumerate(top):
                        # Color based on percentile
                        if pct >= 90:
                            color = COLORS["success"]
                            icon = "🔥"
                            level = "Elite"
                        elif pct >= 75:
                            color = COLORS["primary"] 
                            icon = "⭐"
                            level = "Strong"
                        else:
                            color = COLORS["warning"]
                            icon = "↑"
                            level = "Above Avg"
                            
                        st.markdown(f'''
                            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: var(--glass); border-radius: var(--radius-md); margin-bottom: 0.75rem; border-left: 4px solid {color}; backdrop-filter: blur(10px); transition: all var(--duration-normal) var(--easing);" onmouseover="this.style.transform='translateX(4px)'" onmouseout="this.style.transform='translateX(0px)'">
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <span style="font-size: 1.2rem;">{icon}</span>
                                    <div style="flex: 1;">
                                        <div style="font-weight: 700; font-size: 1rem; color: var(--text); margin-bottom: 0.25rem;">{m}</div>
                                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                                            <div style="color: {color}; font-weight: 800; font-size: 0.9rem; font-family: 'JetBrains Mono', monospace;">{pct:.0f}th percentile</div>
                                            <div style="background: {color}20; color: {color}; border: 1px solid {color}40; padding: 0.2rem 0.5rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700;">
                                                {level}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)
            
            with weak_col:
                st.markdown("#### ⬇️ Development Areas")
                if not bottom:
                    st.info("No development area metrics available with current data")
                else:
                    for i, (m, pct) in enumerate(bottom):
                        # Color based on how low the percentile is
                        if pct <= 25:
                            color = COLORS["danger"]
                            icon = "⚠️"
                            level = "Weak"
                        elif pct <= 50:
                            color = COLORS["secondary"]
                            icon = "↓"
                            level = "Below Avg"
                        else:
                            color = COLORS["warning"]
                            icon = "≈"
                            level = "Average"
                            
                        st.markdown(f'''
                            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: var(--glass); border-radius: var(--radius-md); margin-bottom: 0.75rem; border-left: 4px solid {color}; backdrop-filter: blur(10px); transition: all var(--duration-normal) var(--easing);" onmouseover="this.style.transform='translateX(4px)'" onmouseout="this.style.transform='translateX(0px)'">
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <span style="font-size: 1.2rem;">{icon}</span>
                                    <div style="flex: 1;">
                                        <div style="font-weight: 700; font-size: 1rem; color: var(--text); margin-bottom: 0.25rem;">{m}</div>
                                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                                            <div style="color: {color}; font-weight: 800; font-size: 0.9rem; font-family: 'JetBrains Mono', monospace;">{pct:.0f}th percentile</div>
                                            <div style="background: {color}20; color: {color}; border: 1px solid {color}40; padding: 0.2rem 0.5rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700;">
                                                {level}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)
            
            # Role Analysis if available
            if cfg.get("role_cols", []):
                st.markdown("---")
                
                # Role scores section with radar chart
                radar_col, data_col = st.columns([1.3, 0.7], gap="large")
                
                with radar_col:
                    st.markdown("#### 🎯 Role Suitability Radar")
                    
                    # Create radar chart
                    role_values = []
                    role_labels = []
                    for rc in cfg.get("role_cols", []):
                        val = safe_float(row.get(rc, np.nan))
                        if not np.isnan(val):
                            role_values.append(val)
                            role_labels.append(rc.replace(" Score", "")[:20])
                    
                    if role_values:
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=role_values,
                            theta=role_labels,
                            fill='toself',
                            name=player,
                            line=dict(color=position_color, width=3),
                            fillcolor=f"rgba({int(position_color[1:3], 16)}, {int(position_color[3:5], 16)}, {int(position_color[5:7], 16)}, 0.25)",
                            marker=dict(size=8, color=position_color)
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    range=[0, 100],
                                    gridcolor=COLORS["border"],
                                    showticklabels=True,
                                    tickfont=dict(size=10, color=COLORS["text_muted"]),
                                    tickmode='linear',
                                    tick0=0,
                                    dtick=20
                                ),
                                angularaxis=dict(
                                    gridcolor=COLORS["border"],
                                    tickfont=dict(size=11, color=COLORS["text"]),
                                    linecolor=COLORS["border"]
                                ),
                                bgcolor="rgba(0,0,0,0)"
                            ),
                            height=400,
                            margin=dict(l=80, r=80, t=50, b=50),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color=COLORS["text"], size=11, family='Plus Jakarta Sans, system-ui, sans-serif'),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No role data available for radar chart")
                
                with data_col:
                    st.markdown("#### 📊 Role Scores")
                    for rc in cfg.get("role_cols", []):
                        val = safe_float(row.get(rc, np.nan))
                        if not np.isnan(val):
                            # Determine fit level and color
                            if val >= 80:
                                fit_level = "Excellent"
                                color = COLORS["success"]
                            elif val >= 65:
                                fit_level = "Good"
                                color = COLORS["primary"]
                            elif val >= 50:
                                fit_level = "Average"
                                color = COLORS["warning"]
                            else:
                                fit_level = "Poor"
                                color = COLORS["danger"]
                                
                            st.markdown(f'''
                                <div style="margin-bottom: 1rem; padding: 1rem; background: var(--glass); border: 1px solid var(--border); border-radius: var(--radius-md); backdrop-filter: blur(10px); transition: all var(--duration-normal) var(--easing);" onmouseover="this.style.borderColor='{color}'; this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='var(--border)'; this.style.transform='translateY(0px)'">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                        <span style="font-weight: 700; font-size: 0.9rem; color: var(--text);">{rc[:25]}</span>
                                        <div style="background: {color}20; color: {color}; border: 1px solid {color}40; padding: 0.25rem 0.6rem; border-radius: 12px; font-size: 0.7rem; font-weight: 700;">
                                            {fit_level}
                                        </div>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                                        <div style="flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden;">
                                            <div style="width: {val}%; height: 100%; background: linear-gradient(90deg, {color} 0%, {color}80 100%); transition: width 0.5s ease;"></div>
                                        </div>
                                        <div style="font-weight: 900; color: {color}; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; min-width: 3rem;">
                                            {val:.0f}%
                                        </div>
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Role comparison grid
                st.markdown("#### 🔄 Quick Role Comparison")
                role_comparison_cols = st.columns(min(4, len(cfg.get("role_cols", []))))
                for idx, rc in enumerate(cfg.get("role_cols", [])[:4]):
                    with role_comparison_cols[idx % 4]:
                        val = safe_float(row.get(rc, np.nan))
                        if not np.isnan(val):
                            if val >= 80:
                                color = COLORS["success"]
                            elif val >= 65:
                                color = COLORS["primary"]
                            elif val >= 50:
                                color = COLORS["warning"]
                            else:
                                color = COLORS["danger"]
                            
                            st.markdown(f'''
                                <div class="metric-card" style="border-left: 3px solid {color};">
                                    <div class="metric-label">{rc[:15]}</div>
                                    <div class="metric-value" style="font-size: 1.5rem; color: {color};">{val:.0f}%</div>
                                </div>
                            ''', unsafe_allow_html=True)

    # =====================================================
    # TAB 3: HEAD-TO-HEAD
    # =====================================================
    with tabs[2]:
        if df_f.empty:
            st.warning("⚠️ No players available.")
        else:
            players = sorted(df_f[NAME_COL].dropna().unique().tolist())
            picks = [p for p in st.session_state.compare_picks.get(position, []) if p in players]
            default = picks[:] if len(picks) else (players[:3] if len(players) >= 3 else players[:])
            
            chosen = st.multiselect("🎯 Select Players to Compare", players, default=default)
            st.session_state.compare_picks[position] = chosen
            
            if len(chosen) < 2:
                st.info("📊 Select at least 2 players to generate comparison charts")
            else:
                comp_df = df_f[df_f[NAME_COL].isin(chosen)].copy()
                quick_cols = [c for c in [NAME_COL, TEAM_COL, AGE_COL, SHARE_COL] + cfg.get("key_metrics", []) if c in comp_df.columns]
                st.dataframe(comp_df[quick_cols], width="stretch")

    # =====================================================
    # TAB 4: RANKINGS
    # =====================================================
    with tabs[3]:
        if df_f.empty:
            st.info("📊 No players available for ranking")
        else:
            all_sortable = ["IMPECT"] + cfg.get("role_cols", []) + cfg.get("metric_cols", [])
            all_sortable = [c for c in all_sortable if c in df_f.columns and "BetterThan" not in c]
            
            if all_sortable:
                metric = st.selectbox("📊 Ranking Metric", all_sortable)
                n = st.slider("Top N Players", 10, min(50, len(df_f)), 20)
                
                out = df_f.sort_values(metric, ascending=False).head(n)[[NAME_COL, TEAM_COL, metric]].copy()
                out.insert(0, "Rank", range(1, len(out) + 1))
                
                st.dataframe(out, width="stretch")

    # =====================================================
    # TAB 5: ANALYTICS
    # =====================================================
    with tabs[4]:
        if df_f.empty:
            st.info("📊 No data available for analysis")
        else:
            numeric_cols = [c for c in df_f.select_dtypes(include=[np.number]).columns.tolist() if "BetterThan" not in c]
            
            if numeric_cols:
                metric = st.selectbox("📊 Metric to Analyze", numeric_cols)
                
                col1, col2, col3, col4 = st.columns(4)
                metric_values = df_f[metric].dropna()
                
                with col1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Mean</div><div class="metric-value" style="font-size: 1.5rem;">{safe_fmt(metric_values.mean(), 2)}</div></div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Median</div><div class="metric-value" style="font-size: 1.5rem;">{safe_fmt(metric_values.median(), 2)}</div></div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Max</div><div class="metric-value" style="font-size: 1.5rem;">{safe_fmt(metric_values.max(), 2)}</div></div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Std Dev</div><div class="metric-value" style="font-size: 1.5rem;">{safe_fmt(metric_values.std(), 2)}</div></div>', unsafe_allow_html=True)
                
                fig = px.histogram(df_f, x=metric, nbins=25, color_discrete_sequence=[COLORS["primary"]])
                theme = create_enhanced_plotly_theme()
                fig.update_layout(height=400, showlegend=False, **theme['layout'])
                st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TAB 6: SHORTLIST
    # =====================================================
    with tabs[5]:
        items = []
        for k, meta in st.session_state.shortlist.items():
            pos, name = k.split("||", 1)
            items.append({
                "Position": pos,
                "Player": name,
                "Tags": meta.get("tags", ""),
                "Notes": meta.get("notes", ""),
                "Added": meta.get("added", dt.datetime.now()).strftime("%Y-%m-%d %H:%M") if isinstance(meta.get("added"), dt.datetime) else "Unknown"
            })
        
        if not items:
            st.markdown('''
                <div class="modern-card" style="text-align: center; padding: 4rem;">
                    <div style="font-size: 4rem; margin-bottom: 1.5rem; opacity: 0.5;">⭐</div>
                    <h2 style="color: var(--text-muted); margin-bottom: 1rem;">Your Shortlist is Empty</h2>
                    <p style="color: var(--text-muted); margin-bottom: 2rem; max-width: 400px; margin-left: auto; margin-right: auto;">
                        Start building your player shortlist by adding players from the Scout Browser or Player Profile tabs.
                    </p>
                    <div style="padding: 0.75rem 1.5rem; background: rgba(0,217,255,0.2); border: 1px solid rgba(0,217,255,0.4); border-radius: 50px; color: #00D9FF; font-weight: 700;">
                        🔍 Browse players to get started
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f"### ⭐ Your Shortlist ({len(items)} players)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Players</div><div class="metric-value">{len(items)}</div></div>', unsafe_allow_html=True)
            
            with col2:
                positions = set(item["Position"] for item in items)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Positions</div><div class="metric-value">{len(positions)}</div></div>', unsafe_allow_html=True)
            
            with col3:
                tagged = len([item for item in items if item["Tags"].strip()])
                st.markdown(f'<div class="metric-card"><div class="metric-label">With Tags</div><div class="metric-value">{tagged}</div></div>', unsafe_allow_html=True)
            
            with col4:
                with_notes = len([item for item in items if item["Notes"].strip()])
                st.markdown(f'<div class="metric-card"><div class="metric-label">With Notes</div><div class="metric-value">{with_notes}</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            sl_df = pd.DataFrame(items)
            edited = st.data_editor(
                sl_df,
                width="stretch",
                height=400,
                num_rows="dynamic",
                column_config={
                    "Position": st.column_config.SelectboxColumn("Position", width="small", options=list(POSITION_CONFIG.keys())),
                    "Player": st.column_config.TextColumn("Player", width="medium"),
                    "Tags": st.column_config.TextColumn("Tags", width="medium"),
                    "Notes": st.column_config.TextColumn("Notes", width="large"),
                    "Added": st.column_config.TextColumn("Added", width="small")
                },
                key="shortlist_editor"
            )
            
            new_shortlist = {}
            for _, r in edited.iterrows():
                pos = str(r.get("Position", "")).strip()
                name = str(r.get("Player", "")).strip()
                if not pos or not name:
                    continue
                
                existing_key = shortlist_key(pos, name)
                existing_meta = st.session_state.shortlist.get(existing_key, {})
                
                new_shortlist[existing_key] = {
                    "tags": str(r.get("Tags", "") or ""),
                    "notes": str(r.get("Notes", "") or ""),
                    "added": existing_meta.get("added", dt.datetime.now())
                }
            
            st.session_state.shortlist = new_shortlist
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                csv_data = pd.DataFrame(items).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export CSV",
                    data=csv_data,
                    file_name=f"shortlist_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    width="stretch"
                )
            
            with col2:
                if st.button("🗑️ Clear All", width="stretch", type="secondary"):
                    st.session_state.shortlist = {}
                    st.rerun()
            
            with col3:
                if items:
                    pos_counts = {}
                    for item in items:
                        pos = item["Position"]
                        pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    
                    breakdown = " • ".join([f"{POSITION_CONFIG[pos]['icon']} {pos}: {count}" for pos, count in pos_counts.items()])
                    st.markdown(f'<div style="padding: 0.5rem 1rem; background: rgba(255,107,157,0.2); border: 1px solid rgba(255,107,157,0.4); border-radius: 50px; color: #FF6B9D; font-weight: 700; font-size: 0.8rem; text-align: center;">{breakdown}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
