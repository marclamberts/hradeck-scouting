import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
LOGO_FILE = "FCHK.png"  

st.set_page_config(page_title="FCHK Pro Scout", layout="wide", page_icon="⚽")

# --- 1. THEME STATE & DYNAMIC COLORS ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

if st.session_state.theme == 'Dark':
    B_COLOR, T_COLOR = "#0E1117", "#FFFFFF"
    PILL_TRACK, PILL_HANDLE = "#1E1E1E", "#00FF00"
    GRID = "#31333F"
else:
    B_COLOR, T_COLOR = "#DDE1E6", "#000000"
    PILL_TRACK, PILL_HANDLE = "#BDC1C6", "#000000"
    GRID = "#BBBBBB"

# --- 2. CSS: UI UNIFICATION & ISOLATED PILL ---
st.markdown(f"""
    <style>
    /* Global Backgrounds */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main, 
    [data-testid="stSidebarNav"], .stAppHeader, [data-testid="stDecoration"],
    div[data-testid="stToolbar"], [data-testid="stSidebar"] div, .stAppViewContainer {{
        background-color: {B_COLOR} !important;
    }}

    /* THE PILL TOGGLE (Strictly isolated to Theme Switch) */
    div[data-testid="stCheckbox"] > label > div:first-child {{
        background-color: {PILL_TRACK} !important;
        border: 2px solid {T_COLOR} !important;
        width: 46px !important; height: 24px !important; border-radius: 12px !important;
        display: flex !important; align-items: center !important;
    }}
    div[data-testid="stCheckbox"] > label > div:first-child > div {{
        background-color: {PILL_HANDLE} !important;
        width: 18px !important; height: 18px !important; border-radius: 50% !important;
    }}

    /* UNIFIED FILTERS: Background same as App, Text is Opposite */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div, 
    .stTextInput input, 
    div[role="combobox"],
    div[data-baseweb="popover"] div {{
        background-color: {B_COLOR} !important;
        color: {T_COLOR} !important;
        border: 1.5px solid {T_COLOR} !important;
    }}

    /* Global Text Colors */
    html, body, .stMarkdown, p, h1, h2, h3, h4, span, label, li, td, th, 
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] *, .stSelectbox label, .stTextInput label,
    div[role="listbox"] div, .stDataFrame div, button, .stTab, .stCaption, 
    .stSlider label {{
        color: {T_COLOR} !important;
        -webkit-text-fill-color: {T_COLOR} !important;
    }}

    /* Navigation Buttons */
    .stButton>button {{ 
        width: 100%; border-radius: 4px; background-color: transparent !important; 
        color: {T_COLOR} !important; border: 1.5px solid {T_COLOR} !important;
        font-weight: bold; margin-bottom: 8px;
    }}

    header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. POSITION MAPPING & DATA ---
POS_MAPPING = {
    'Goalkeeper': ['GK'],
    'Defender': ['CB', 'LCB', 'RCB', 'LB', 'RB', 'LWB', 'RWB', 'DF'],
    'Midfielder': ['DMF', 'LDMF', 'RDMF', 'CMF', 'LCMF', 'RCMF', 'AMF', 'LAMF', 'RAMF', 'LM', 'RM', 'MF'],
    'Attacker': ['CF', 'LW', 'RW', 'LWF', 'RWF', 'SS', 'ST', 'FW']
}

def get_group(pos_string):
    if not pos_string or pos_string in ['nan', 'None', 'NULL', '']: return "Unknown"
    tags = [t.strip().upper() for t in str(pos_string).split(',')]
    for group, codes in POS_MAPPING.items():
        if any(code in tags for code in codes): return group
    return "Other"

@st.cache_data
def load_data(table):
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'position' in df.columns:
            df['position_group'] = df['position'].apply(get_group)
        
        # Convert numeric stats
        for c in df.columns:
            if any(k in c for k in ['value', 'age', 'goal', 'xg', 'match', 'minutes', 'xa', 'assist']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # Performance Calculations (Delta = Expected - Actual)
        if 'xg' in df.columns and 'goals' in df.columns:
            df['goals_diff'] = df['xg'] - df['goals']
        if 'xa' in df.columns and 'assists' in df.columns:
            df['assists_diff'] = df['xa'] - df['assists']
        return df

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=T_COLOR, size=12),
        xaxis=dict(title_font=dict(color=T_COLOR), tickfont=dict(color=T_COLOR), gridcolor=GRID, linecolor=T_COLOR),
        yaxis=dict(title_font=dict(color=T_COLOR), tickfont=dict(color=T_COLOR), gridcolor=GRID, linecolor=T_COLOR)
    )
    return fig

# --- 4. AUTHENTICATION ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated: return True
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        try: st.image(LOGO_FILE, width=120)
        except: pass
        st.title("FCHK LOGIN")
        with st.form("login"):
            u, p = st.text_input("USER"), st.text_input("PASSWORD", type="password")
            if st.form_submit_button("ENTER"):
                if u == VALID_USERNAME and p == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("ACCESS DENIED")
    return False

# --- 5. MAIN APP ---
if check_password():
    with st.sidebar:
        st.write("### ⚙️ SYSTEM")
        theme_toggle = st.toggle("DARK MODE", value=(st.session_state.theme == 'Dark'), key="theme_switch")
        st.session_state.theme = 'Dark' if theme_toggle else 'Light'
        st.write("---")
        st.write("### 🧭 PAGES")
        if st.button("🏠 DASHBOARD"): st.session_state.view = 'Dashboard'
        if st.button("🔍 SEARCH"): st.session_state.view = 'Search'
        if st.button("🏆 PERFORMANCE"): st.session_state.view = 'Perf'
        if st.button("📊 RANKINGS"): st.session_state.view = 'Bar'
        if st.button("📈 DISTRIBUTIONS"): st.session_state.view = 'Dist'
        st.write("---")
        if st.button("LOGOUT"):
            st.session_state.authenticated = False
            st.rerun()

    with sqlite3.connect(DB_FILE) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    
    selected_table = st.sidebar.selectbox("DATASET", tables, key="table_select")
    df_raw = load_data(selected_table)

    if 'view' not in st.session_state: st.session_state.view = 'Dashboard'

    # UI Helper for shared filters
    def filter_ui(key):
        c1, c2, c3 = st.columns(3)
        with c1:
            teams = ["ALL TEAMS"] + sorted([x for x in df_raw["team"].unique() if x])
            st.session_state.f_team = st.selectbox("TEAM", teams, index=0, key=f"{key}_t")
        with c2:
            groups = ["ALL GROUPS", "Goalkeeper", "Defender", "Midfielder", "Attacker"]
            st.session_state.f_group = st.selectbox("POSITION GROUP", groups, index=0, key=f"{key}_g")
        with c3:
            st.session_state.f_search = st.text_input("PLAYER SEARCH", key=f"{key}_s")

    # --- PAGES ---
    if st.session_state.view == 'Perf':
        st.title("🏆 Over/Under Performance Leaderboards")
        tab1, tab2 = st.tabs(["Goal Finishing (xG vs Goals)", "Playmaking (xA vs Assists)"])
        with tab1:
            col_l, col_r = st.columns(2)
            with col_l:
                st.write("🔥 **Top 10 Clinical Finishers**")
                st.dataframe(df_raw.sort_values('goals_diff').head(10)[['player','team','goals','xg','goals_diff']], width='stretch')
            with col_r:
                st.write("📉 **Top 10 Underperformers**")
                st.dataframe(df_raw.sort_values('goals_diff', ascending=False).head(10)[['player','team','goals','xg','goals_diff']], width='stretch')
            st.plotly_chart(style_fig(px.scatter(df_raw, x="xg", y="goals", hover_name="player", trendline="ols", color_discrete_sequence=[T_COLOR])), width='stretch')
        
        with tab2:
            col_l, col_r = st.columns(2)
            with col_l:
                st.write("🧙‍♂️ **Top 10 Creative Overperformers**")
                st.dataframe(df_raw.sort_values('assists_diff').head(10)[['player','team','assists','xa','assists_diff']], width='stretch')
            with col_r:
                st.write("⚠️ **Top 10 Creative Underperformers**")
                st.dataframe(df_raw.sort_values('assists_diff', ascending=False).head(10)[['player','team','assists','xa','assists_diff']], width='stretch')

    elif st.session_state.view == 'Dist':
        st.title("📈 Statistical Distributions")
        filter_ui("dist_view")
        num_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
        d_col = st.selectbox("Select Metric", num_cols)
        st.plotly_chart(style_fig(px.histogram(df_raw, x=d_col, nbins=30, color_discrete_sequence=[T_COLOR])), width='stretch')

    elif st.session_state.view == 'Dashboard':
        st.title("📊 Scout Dashboard")
        filter_ui("dash_view")
        m1, m2 = st.columns(2)
        m1.metric("TOTAL PLAYERS", len(df_raw))
        if 'market_value' in df_raw.columns:
            m2.metric("AVG VALUE", f"€{int(df_raw['market_value'].mean()):,}")
        st.plotly_chart(style_fig(px.box(df_raw, x="position_group", y="market_value", color_discrete_sequence=[T_COLOR])), width='stretch')

    elif st.session_state.view == 'Search':
        st.title("🔍 Advanced Database Search")
        filter_ui("search_view")
        st.dataframe(df_raw, width='stretch', height=600)

    elif st.session_state.view == 'Bar':
        st.title("📊 Player Rankings")
        num_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("Rank By Metric", num_cols)
        st.plotly_chart(style_fig(px.bar(df_raw.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, color_discrete_sequence=[T_COLOR])), width='stretch')