import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/5329/5329945.png" 

st.set_page_config(page_title="Hradeck Pro Scout", layout="wide", page_icon="⚽")

# --- GLOBAL STYLING: UNIFIED GREY & PURE BLACK ---
st.markdown("""
    <style>
    /* 1. UNIFY ALL BACKGROUNDS TO ONE COLOR */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main, 
    [data-testid="stSidebarNav"], .stAppHeader, [data-testid="stDecoration"] {
        background-color: #DDE1E6 !important;
    }

    /* 2. FORCE PURE BLACK TEXT EVERYWHERE (Aggressive Overrides) */
    html, body, .stMarkdown, p, h1, h2, h3, h4, span, label, li, td, th, 
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] *, .stSelectbox label, .stTextInput label,
    div[role="listbox"] div, .stDataFrame div, button, .stTab {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* 3. INPUTS: Clean boxes with black borders */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, 
    .stTextInput input, div[role="combobox"] {
        background-color: #DDE1E6 !important;
        color: #000000 !important;
        border: 1.5px solid #000000 !important;
        border-radius: 4px !important;
    }

    /* 4. BUTTONS: Classic High-Contrast Style */
    .stButton>button { 
        width: 100%; 
        border-radius: 4px; 
        background-color: transparent !important; 
        color: #000000 !important; 
        font-weight: bold;
        border: 1.5px solid #000000 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover { 
        background-color: #000000 !important; 
        color: #FFFFFF !important; 
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* 5. METRIC CARDS: Flat Bordered Style */
    div[data-testid="metric-container"] {
        background-color: transparent !important;
        border: 1.5px solid #000000 !important;
        border-radius: 8px;
        padding: 15px;
    }

    /* 6. HIDE ELEMENTS THAT ADD NOISE */
    hr { border: 0.5px solid #000000 !important; }
    [data-testid="stMetricDelta"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. AUTHENTICATION ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        st.write("#")
        st.image(LOGO_URL, width=70)
        st.title("HRADECK LOGIN")
        with st.form("login_form"):
            user = st.text_input("USER")
            pwd = st.text_input("PASSWORD", type="password")
            if st.form_submit_button("ENTER SYSTEM"):
                if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("ACCESS DENIED")
    return False

# --- 2. DATA LOADING ---
@st.cache_data
def get_tables():
    with sqlite3.connect(DB_FILE) as conn:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

def load_data(table):
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
        # Handle columns
        for c in df.columns:
            if any(k in c.lower() for k in ['value', 'age', 'goal', 'xg', 'match', 'per_90']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = df[c].astype(str).replace(['nan', 'None'], '')
        return df

# --- 3. MAIN APP ---
if check_password():
    st.sidebar.title("SCOUTING MENU")
    
    tables = get_tables()
    selected_table = st.sidebar.selectbox("DATASET", tables)
    
    if 'view' not in st.session_state:
        st.session_state.view = 'Dashboard'

    st.sidebar.write("##")
    if st.sidebar.button("DASHBOARD"): st.session_state.view = 'Dashboard'
    if st.sidebar.button("TABLE"): st.session_state.view = 'Table'
    if st.sidebar.button("BAR GRAPHS"): st.session_state.view = 'Bar'
    if st.sidebar.button("DISTRIBUTION"): st.session_state.view = 'Dist'
    
    st.sidebar.write("---")
    if st.sidebar.button("LOGOUT"):
        st.session_state.authenticated = False
        st.rerun()

    df_raw = load_data(selected_table)

    # PAGE HEADER
    st.title(f"DATA: {selected_table}")
    
    # FILTERS
    c1, c2, c3 = st.columns(3)
    with c1:
        teams = ["ALL TEAMS"] + sorted([str(x) for x in df_raw["team"].unique() if x])
        f_team = st.selectbox("TEAM", teams)
    with c2:
        pos = ["ALL POSITIONS"]
        if "position" in df_raw.columns:
            p_set = set()
            for p in df_raw["position"].unique():
                if p: [p_set.add(s.strip()) for s in str(p).split(',')]
            pos += sorted(list(p_set))
        f_pos = st.selectbox("POSITION", pos)
    with c3:
        search = st.text_input("PLAYER SEARCH")

    # LOGIC
    df = df_raw.copy()
    if f_team != "ALL TEAMS": df = df[df["team"] == f_team]
    if f_pos != "ALL POSITIONS" and "position" in df.columns:
        df = df[df["position"].str.contains(f_pos, na=False)]
    if search: df = df[df["player"].str.contains(search, case=False, na=False)]

    st.write("---")

    # VIEWS
    if st.session_state.view == 'Dashboard':
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PLAYERS", len(df))
        if 'market_value' in df.columns: m2.metric("AVG VALUE", f"€{int(df['market_value'].mean()):,}")
        if 'goals' in df.columns: m3.metric("TOTAL GOALS", int(df['goals'].sum()))
        if 'age' in df.columns: m4.metric("AVG AGE", round(df['age'].mean(), 1))

        st.write("### MARKET VALUE TRENDS")
        l, r = st.columns(2)
        with l:
            fig = px.box(df, x="team", y="market_value", template="simple_white", color_discrete_sequence=['black'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="black"))
            st.plotly_chart(fig, use_container_width=True)
        with r:
            fig = px.scatter(df, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=['black'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="black"))
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Table':
        st.dataframe(df, use_container_width=True, height=600)

    elif st.session_state.view == 'Bar':
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("METRIC", num_cols, index=num_cols.index('market_value') if 'market_value' in num_cols else 0)
        fig = px.bar(df.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white", color_discrete_sequence=['black'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="black"))
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Dist':
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        d_col = st.selectbox("METRIC", num_cols)
        fig = px.histogram(df, x=d_col, template="simple_white", color_discrete_sequence=['black'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="black"))
        st.plotly_chart(fig, use_container_width=True)