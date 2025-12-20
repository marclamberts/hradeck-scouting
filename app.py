import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import re

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
LOGO_FILE = "FCHK.png"  
THEME_COLOR = "#DDE1E6" # Unified Slate Grey

st.set_page_config(page_title="FCHK Pro Scout", layout="wide", page_icon="⚽")

# --- ULTIMATE CSS: TOTAL UNIFICATION ---
st.markdown(f"""
    <style>
    /* 1. Global Background (App, Sidebar, Header) */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main, 
    [data-testid="stSidebarNav"], .stAppHeader, [data-testid="stDecoration"],
    div[data-testid="stToolbar"], [data-testid="stSidebar"] div, .stAppViewContainer {{
        background-color: {THEME_COLOR} !important;
    }}

    /* 2. Absolute Black Text & Font Unification */
    html, body, .stMarkdown, p, h1, h2, h3, h4, span, label, li, td, th, 
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] *, .stSelectbox label, .stTextInput label,
    div[role="listbox"] div, .stDataFrame div, button, .stTab, .stCaption, 
    .stSlider label, [data-testid="stMetricDelta"] {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* 3. Dropdowns & Inputs: Background and Border */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="popover"] div, 
    ul[role="listbox"], 
    li[role="option"],
    div[data-baseweb="input"] > div, 
    .stTextInput input, 
    div[role="combobox"] {{
        background-color: {THEME_COLOR} !important;
        color: #000000 !important;
        border: 1.5px solid #000000 !important;
    }}

    /* Dropdown hover selection */
    li[role="option"]:hover {{
        background-color: #CCCCCC !important;
    }}

    /* 4. Buttons: Minimalist Bordered Style (No Hover Change) */
    .stButton>button {{ 
        width: 100%; border-radius: 4px; background-color: transparent !important; 
        color: #000000 !important; font-weight: bold; border: 1.5px solid #000000 !important;
        transition: none !important; text-transform: uppercase;
    }}
    .stButton>button:hover, .stButton>button:active, .stButton>button:focus {{ 
        background-color: transparent !important; color: #000000 !important; 
        border-color: #000000 !important; box-shadow: none !important; 
    }}

    /* 5. Metrics & Containers */
    div[data-testid="metric-container"] {{
        background-color: transparent !important; border: 1.5px solid #000000 !important;
        border-radius: 8px; padding: 15px;
    }}
    hr {{ border: 0.5px solid #000000 !important; }}
    header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated: return True

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        st.write("#")
        try: st.image(LOGO_FILE, width=120)
        except: st.warning(f"File '{LOGO_FILE}' not found.")
        st.title("FCHK LOGIN")
        with st.form("login"):
            u, p = st.text_input("USER"), st.text_input("PASSWORD", type="password")
            if st.form_submit_button("ENTER SYSTEM"):
                if u == VALID_USERNAME and p == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("ACCESS DENIED")
    return False

# --- DATA LOADING ---
@st.cache_data
def get_tables():
    with sqlite3.connect(DB_FILE) as conn:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

def load_data(table):
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
        for c in df.columns:
            if any(k in c.lower() for k in ['value', 'age', 'goal', 'xg', 'match', 'minutes', 'assist']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = df[c].astype(str).replace(['nan', 'None'], '')
        return df

# --- CHART STYLING (FORCE BLACK TICKS/FACECOLOR) ---
def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=THEME_COLOR,
        plot_bgcolor=THEME_COLOR,
        font=dict(color="black", size=12),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor="#BBBBBB", linecolor="black"),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor="#BBBBBB", linecolor="black"),
        legend=dict(font=dict(color="black"), bgcolor="rgba(0,0,0,0)")
    )
    return fig

# --- FILTER LOGIC (FIXED MULTI-POSITION MATCHING) ---
def apply_global_filters(data):
    df_f = data.copy()
    if st.session_state.f_team != "ALL TEAMS":
        df_f = df_f[df_f["team"] == st.session_state.f_team]
    
    # FIXED POSITION LOGIC: Checks if ANY selected position exists in the player's list
    if st.session_state.f_pos and "ALL POSITIONS" not in st.session_state.f_pos:
        df_f = df_f[df_f["position"].apply(lambda x: any(p.strip() in [s.strip() for s in str(x).split(',')] for p in st.session_state.f_pos))]
        
    if st.session_state.f_search:
        df_f = df_f[df_f["player"].str.contains(st.session_state.f_search, case=False, na=False)]
    
    if 'age' in df_f.columns:
        df_f = df_f[df_f['age'].between(st.session_state.f_age[0], st.session_state.f_age[1])]
    
    if 'minutes_played' in df_f.columns:
        df_f = df_f[df_f['minutes_played'].between(st.session_state.f_mins[0], st.session_state.f_mins[1])]
    return df_f

# --- MAIN APP ---
if check_password():
    # Sidebar
    try: st.sidebar.image(LOGO_FILE, width=80)
    except: pass
    
    selected_table = st.sidebar.selectbox("DATASET", get_tables(), key="table_select")
    
    # Initialize Page State
    if 'view' not in st.session_state: st.session_state.view = 'Dashboard'
    
    # Initialize Persistent Filters
    if 'f_team' not in st.session_state: st.session_state.f_team = "ALL TEAMS"
    if 'f_pos' not in st.session_state: st.session_state.f_pos = []
    if 'f_search' not in st.session_state: st.session_state.f_search = ""
    if 'f_age' not in st.session_state: st.session_state.f_age = (15, 45)
    if 'f_mins' not in st.session_state: st.session_state.f_mins = (0, 5000)

    st.sidebar.write("##")
    if st.sidebar.button("🏠 DASHBOARD"): st.session_state.view = 'Dashboard'
    if st.sidebar.button("🔍 SEARCH"): st.session_state.view = 'Search'
    if st.sidebar.button("📊 BAR RANKING"): st.session_state.view = 'Bar'
    if st.sidebar.button("📈 DISTRIBUTIONS"): st.session_state.view = 'Dist'
    
    st.sidebar.write("---")
    if st.sidebar.button("LOGOUT"):
        st.session_state.authenticated = False
        st.rerun()

    df_raw = load_data(selected_table)

    # Filter Component used on both Search and Dashboard
    def draw_filters(key_prefix):
        c1, c2, c3 = st.columns(3)
        with c1:
            t_opts = ["ALL TEAMS"] + sorted([str(x) for x in df_raw["team"].unique() if x])
            st.session_state.f_team = st.selectbox("TEAM", t_opts, index=t_opts.index(st.session_state.f_team) if st.session_state.f_team in t_opts else 0, key=f"{key_prefix}_t")
        with c2:
            p_set = set()
            for p in df_raw["position"].unique():
                if p: [p_set.add(s.strip()) for s in str(p).split(',')]
            all_p = sorted(list(p_set))
            st.session_state.f_pos = st.multiselect("POSITIONS (Select Multiple)", all_p, default=st.session_state.f_pos, key=f"{key_prefix}_p")
        with c3:
            st.session_state.f_search = st.text_input("NAME SEARCH", value=st.session_state.f_search, key=f"{key_prefix}_s")

    # --- VIEWS ---

    if st.session_state.view == 'Search':
        st.title("🔍 Advanced Scout Search")
        draw_filters("search")
        
        s1, s2 = st.columns(2)
        with s1:
            st.session_state.f_age = st.slider("AGE LIMITS", 15, 50, st.session_state.f_age, key="s_age_sl")
        with s2:
            st.session_state.f_mins = st.slider("MINUTES PLAYED", 0, 10000, st.session_state.f_mins, key="s_min_sl")
        
        df_filtered = apply_global_filters(df_raw)
        st.divider()
        st.dataframe(df_filtered, use_container_width=True, height=500)

    elif st.session_state.view == 'Dashboard':
        st.title(f"📊 Dashboard: {selected_table}")
        draw_filters("dash")
        
        df_dash = apply_global_filters(df_raw)
        st.divider()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PLAYERS", len(df_dash))
        if 'market_value' in df_dash.columns: m2.metric("AVG VALUE", f"€{int(df_dash['market_value'].mean()):,}")
        if 'goals' in df_dash.columns: m3.metric("GOALS", int(df_dash['goals'].sum()))
        if 'age' in df_dash.columns: m4.metric("AVG AGE", round(df_dash['age'].mean(), 1))

        l, r = st.columns(2)
        with l:
            fig_box = style_fig(px.box(df_dash, x="team", y="market_value", template="simple_white", color_discrete_sequence=['black']))
            st.plotly_chart(fig_box, use_container_width=True)
        with r:
            fig_scat = style_fig(px.scatter(df_dash, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=['black']))
            st.plotly_chart(fig_scat, use_container_width=True)

    elif st.session_state.view == 'Bar':
        st.title("📊 Player Rankings")
        df_bar = apply_global_filters(df_raw)
        num_cols = df_bar.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("METRIC", num_cols, index=num_cols.index('market_value') if 'market_value' in num_cols else 0)
        fig_bar = style_fig(px.bar(df_bar.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white", color_discrete_sequence=['black']))
        st.plotly_chart(fig_bar, use_container_width=True)

    elif st.session_state.view == 'Dist':
        st.title("📈 Statistical Spread")
        df_dist = apply_global_filters(df_raw)
        num_cols = df_dist.select_dtypes(include=['number']).columns.tolist()
        d_col = st.selectbox("METRIC", num_cols)
        fig_hist = style_fig(px.histogram(df_dist, x=d_col, template="simple_white", color_discrete_sequence=['black']))
        st.plotly_chart(fig_hist, use_container_width=True)