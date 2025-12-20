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
THEME_COLOR = "#DDE1E6" # Unified Grey Background

st.set_page_config(page_title="FCHK Pro Scout", layout="wide", page_icon="⚽")

# --- CSS: TOTAL UNIFICATION & BLACK TEXT ---
st.markdown(f"""
    <style>
    /* Global Background */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main, 
    [data-testid="stSidebarNav"], .stAppHeader, [data-testid="stDecoration"],
    div[data-testid="stToolbar"], [data-testid="stSidebar"] div, .stAppViewContainer {{
        background-color: {THEME_COLOR} !important;
    }}

    /* Force Black Text Everywhere */
    html, body, .stMarkdown, p, h1, h2, h3, h4, span, label, li, td, th, 
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] *, .stSelectbox label, .stTextInput label,
    div[role="listbox"] div, .stDataFrame div, button, .stTab, .stCaption, .stSlider label {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* Inputs & Dropdowns - Facecolor match */
    div[data-baseweb="select"] > div, div[data-baseweb="popover"] div, 
    ul[role="listbox"], li[role="option"], div[data-baseweb="input"] > div, 
    .stTextInput input, div[role="combobox"] {{
        background-color: {THEME_COLOR} !important;
        color: #000000 !important;
        border: 1.5px solid #000000 !important;
    }}

    /* Buttons: No color change on hover */
    .stButton>button {{ 
        width: 100%; border-radius: 4px; background-color: transparent !important; 
        color: #000000 !important; font-weight: bold; border: 1.5px solid #000000 !important;
        transition: none !important;
    }}
    .stButton>button:hover {{ 
        background-color: transparent !important; 
        color: #000000 !important; 
        border-color: #000000 !important; 
        box-shadow: none !important; 
    }}

    /* Metric Cards */
    div[data-testid="metric-container"] {{
        background-color: transparent !important; border: 1.5px solid #000000 !important;
        border-radius: 8px; padding: 15px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated: return True

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        try: st.image(LOGO_FILE, width=120)
        except: st.warning("FCHK.png not found.")
        st.title("FCHK LOGIN")
        with st.form("login"):
            u, p = st.text_input("USER"), st.text_input("PASSWORD", type="password")
            if st.form_submit_button("ENTER"):
                if u == VALID_USERNAME and p == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("DENIED")
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
            if any(k in c.lower() for k in ['value', 'age', 'goal', 'xg', 'match', 'minutes']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = df[c].astype(str).replace(['nan', 'None'], '')
        return df

# --- FILTER LOGIC (FIXED POSITION MATCHING) ---
def apply_filters(data, f_team, f_pos, f_search, f_age=(0,50), f_mins=(0,10000)):
    df_f = data.copy()
    if f_team != "ALL TEAMS":
        df_f = df_f[df_f["team"] == f_team]
    
    # Exact Position Match: CB will not match LCB
    if f_pos != "ALL POSITIONS" and "position" in df_f.columns:
        pattern = r'\b' + re.escape(f_pos) + r'\b'
        df_f = df_f[df_f["position"].str.contains(pattern, regex=True, na=False)]
        
    if f_search:
        df_f = df_f[df_f["player"].str.contains(f_search, case=False, na=False)]
    if 'age' in df_f.columns:
        df_f = df_f[df_f['age'].between(f_age[0], f_age[1])]
    if 'minutes_played' in df_f.columns:
        df_f = df_f[df_f['minutes_played'].between(f_mins[0], f_mins[1])]
    return df_f

# --- MAIN APP ---
if check_password():
    st.sidebar.image(LOGO_FILE, width=80)
    st.sidebar.title("SCOUTING MENU")
    selected_table = st.sidebar.selectbox("DATASET", get_all_tables(), key="table_select")
    
    if 'view' not in st.session_state: st.session_state.view = 'Dashboard'

    st.sidebar.write("##")
    if st.sidebar.button("🏠 DASHBOARD"): st.session_state.view = 'Dashboard'
    if st.sidebar.button("🔍 SEARCH"): st.session_state.view = 'Search'
    if st.sidebar.button("📄 TABLE VIEW"): st.session_state.view = 'Table'
    if st.sidebar.button("📊 BAR GRAPHS"): st.session_state.view = 'Bar'
    
    df_raw = load_data(selected_table)

    # Persistence of filters across pages
    if 'f_team' not in st.session_state: st.session_state.f_team = "ALL TEAMS"
    if 'f_pos' not in st.session_state: st.session_state.f_pos = "ALL POSITIONS"
    if 'f_search' not in st.session_state: st.session_state.f_search = ""
    if 'f_age' not in st.session_state: st.session_state.f_age = (15, 45)
    if 'f_mins' not in st.session_state: st.session_state.f_mins = (0, 5000)

    # --- VIEWS ---
    
    if st.session_state.view == 'Search':
        st.title("🔍 Advanced Player Search")
        c1, c2, c3 = st.columns(3)
        with c1:
            opts = ["ALL TEAMS"] + sorted([str(x) for x in df_raw["team"].unique() if x])
            st.session_state.f_team = st.selectbox("TEAM", opts, index=opts.index(st.session_state.f_team) if st.session_state.f_team in opts else 0)
        with c2:
            p_opts = set()
            for p in df_raw["position"].unique():
                if p: [p_opts.add(s.strip()) for s in str(p).split(',')]
            all_p = ["ALL POSITIONS"] + sorted(list(p_opts))
            st.session_state.f_pos = st.selectbox("POSITION", all_p, index=all_p.index(st.session_state.f_pos) if st.session_state.f_pos in all_p else 0)
        with c3:
            st.session_state.f_search = st.text_input("NAME", value=st.session_state.f_search)

        s1, s2 = st.columns(2)
        with s1:
            st.session_state.f_age = st.slider("AGE RANGE", 15, 50, st.session_state.f_age)
        with s2:
            st.session_state.f_mins = st.slider("MINUTES PLAYED", 0, 10000, st.session_state.f_mins)

        df_filtered = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, 
                                   st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        
        st.divider()
        st.subheader(f"Results ({len(df_filtered)} players)")
        st.dataframe(df_filtered, use_container_width=True, height=500)

    elif st.session_state.view == 'Dashboard':
        st.title(f"📊 {selected_table} Analytics")
        # Shared Filters applied here too
        df_dash = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, 
                               st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PLAYERS", len(df_dash))
        if 'market_value' in df_dash.columns:
            m2.metric("AVG VALUE", f"€{int(df_dash['market_value'].mean()):,}")
        if 'goals' in df_dash.columns:
            m3.metric("TOTAL GOALS", int(df_dash['goals'].sum()))
        if 'age' in df_dash.columns:
            m4.metric("AVG AGE", round(df_dash['age'].mean(), 1))

        fig = px.scatter(df_dash, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=['black'])
        fig.update_layout(paper_bgcolor=THEME_COLOR, plot_bgcolor=THEME_COLOR, font=dict(color="black"))
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Table':
        st.title("📄 Global Table View")
        st.dataframe(df_raw, use_container_width=True, height=700)

    elif st.session_state.view == 'Bar':
        st.title("📊 Ranking Metrics")
        num_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("SELECT METRIC", num_cols, index=num_cols.index('market_value') if 'market_value' in num_cols else 0)
        fig = px.bar(df_raw.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white", color_discrete_sequence=['black'])
        fig.update_layout(paper_bgcolor=THEME_COLOR, plot_bgcolor=THEME_COLOR, font=dict(color="black"))
        st.plotly_chart(fig, use_container_width=True)