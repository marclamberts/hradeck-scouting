import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import re

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
LOGO_FILE = "FCHK.png"  # Local logo file
THEME_COLOR = "#DDE1E6" # Unified background color

st.set_page_config(page_title="Hradeck Pro Scout", layout="wide", page_icon="⚽")

# --- ULTIMATE CSS: TOTAL UNIFICATION ---
st.markdown(f"""
    <style>
    /* 1. Global Background for every container */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main, 
    [data-testid="stSidebarNav"], .stAppHeader, [data-testid="stDecoration"],
    div[data-testid="stToolbar"], [data-testid="stSidebar"] div, .stAppViewContainer {{
        background-color: {THEME_COLOR} !important;
    }}

    /* 2. Absolute Black Text for every element */
    html, body, .stMarkdown, p, h1, h2, h3, h4, span, label, li, td, th, 
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] *, .stSelectbox label, .stTextInput label,
    div[role="listbox"] div, .stDataFrame div, button, .stTab, .stCaption {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* 3. Inputs & DROPDOWN MENUS: Force background color on the pop-over lists */
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

    /* Fix for the hover state inside the dropdown menu to stay grey */
    li[role="option"]:hover {{
        background-color: #CCCCCC !important;
    }}

    /* 4. Buttons: No Hover Change */
    .stButton>button {{ 
        width: 100%; 
        border-radius: 4px; 
        background-color: transparent !important; 
        color: #000000 !important; 
        font-weight: bold;
        border: 1.5px solid #000000 !important;
        transition: none !important;
    }}
    .stButton>button:hover, .stButton>button:active, .stButton>button:focus {{ 
        background-color: transparent !important; 
        color: #000000 !important; 
        border-color: #000000 !important;
        box-shadow: none !important;
    }}

    /* 5. Metrics & Containers */
    div[data-testid="metric-container"] {{
        background-color: transparent !important;
        border: 1.5px solid #000000 !important;
        border-radius: 8px;
        padding: 15px;
    }}
    hr {{ border: 0.5px solid #000000 !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        st.write("#")
        try:
            st.image(LOGO_FILE, width=120)
        except:
            st.warning("Logo file 'FCHK.png' not found.")
        st.title("HRADECK LOGIN")
        with st.form("login_form"):
            user = st.text_input("USER")
            pwd = st.text_input("PASSWORD", type="password")
            if st.form_submit_button("ENTER"):
                if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("ACCESS DENIED")
    return False

# --- DATA HELPERS ---
@st.cache_data
def get_tables():
    with sqlite3.connect(DB_FILE) as conn:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

def load_data(table):
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
        for c in df.columns:
            if any(k in c.lower() for k in ['value', 'age', 'goal', 'xg', 'match', 'per_90', 'assist', 'won']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = df[c].astype(str).replace(['nan', 'None'], '')
        return df

# Helper for Chart Styling
def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=THEME_COLOR,
        plot_bgcolor=THEME_COLOR,
        font=dict(color="black", size=12),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor="#CCCCCC", linecolor="black"),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor="#CCCCCC", linecolor="black"),
        legend=dict(font=dict(color="black"), bgcolor="rgba(0,0,0,0)")
    )
    return fig

# --- MAIN APP ---
if check_password():
    # Sidebar
    try:
        st.sidebar.image(LOGO_FILE, width=80)
    except:
        pass
    st.sidebar.title("SCOUTING MENU")
    
    tables = get_tables()
    selected_table = st.sidebar.selectbox("DATASET", tables, key="table_select")
    
    if 'view' not in st.session_state:
        st.session_state.view = 'Dashboard'

    st.sidebar.write("##")
    if st.sidebar.button("🏠 DASHBOARD"): st.session_state.view = 'Dashboard'
    if st.sidebar.button("🔍 SEARCH"): st.session_state.view = 'Search'
    if st.sidebar.button("📄 TABLE VIEW"): st.session_state.view = 'Table'
    if st.sidebar.button("📊 BAR GRAPHS"): st.session_state.view = 'Bar'
    if st.sidebar.button("📈 DISTRIBUTIONS"): st.session_state.view = 'Dist'
    
    st.sidebar.write("---")
    if st.sidebar.button("LOGOUT"):
        st.session_state.authenticated = False
        st.rerun()

    # Load Full Data for Table
    df_raw = load_data(selected_table)

    # Persistent filter state initialization
    if 'f_team' not in st.session_state: st.session_state.f_team = "ALL TEAMS"
    if 'f_pos' not in st.session_state: st.session_state.f_pos = "ALL POSITIONS"
    if 'f_search' not in st.session_state: st.session_state.f_search = ""

    # Shared Filtering Logic (applied globally)
    df = df_raw.copy()
    if st.session_state.f_team != "ALL TEAMS":
        df = df[df["team"] == st.session_state.f_team]
    if st.session_state.f_pos != "ALL POSITIONS" and "position" in df.columns:
        df = df[df["position"].apply(lambda x: st.session_state.f_pos in [s.strip() for s in str(x).split(',')])]
    if st.session_state.f_search:
        df = df[df["player"].str.contains(st.session_state.f_search, case=False, na=False)]

    # --- VIEWS ---
    
    # 1. SEARCH PAGE (The page with filters)
    if st.session_state.view == 'Search':
        st.title("🔍 Advanced Scout Search")
        st.write("Apply filters below to update the system globally.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            team_options = ["ALL TEAMS"] + sorted([str(x) for x in df_raw["team"].unique() if x])
            st.session_state.f_team = st.selectbox("TEAM FILTER", team_options, index=team_options.index(st.session_state.f_team) if st.session_state.f_team in team_options else 0)
        with c2:
            pos_options = set()
            if "position" in df_raw.columns:
                for p in df_raw["position"].unique():
                    if p: [pos_options.add(s.strip()) for s in str(p).split(',')]
            all_pos = ["ALL POSITIONS"] + sorted(list(pos_options))
            st.session_state.f_pos = st.selectbox("POSITION FILTER", all_pos, index=all_pos.index(st.session_state.f_pos) if st.session_state.f_pos in all_pos else 0)
        with c3:
            st.session_state.f_search = st.text_input("NAME SEARCH", value=st.session_state.f_search)

        st.divider()
        st.subheader(f"Results ({len(df)} players found)")
        st.dataframe(df, use_container_width=True, height=500)

    # 2. DASHBOARD
    elif st.session_state.view == 'Dashboard':
        st.title(f"📊 Dashboard: {selected_table}")
        st.caption(f"Currently filtered by: {st.session_state.f_team} | {st.session_state.f_pos}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PLAYERS", len(df))
        if 'market_value' in df.columns: m2.metric("AVG VALUE", f"€{int(df['market_value'].mean()):,}")
        if 'goals' in df.columns: m3.metric("TOTAL GOALS", int(df['goals'].sum()))
        if 'age' in df.columns: m4.metric("AVG AGE", round(df['age'].mean(), 1))

        l, r = st.columns(2)
        with l:
            st.subheader("Value by Team")
            fig = style_fig(px.box(df, x="team", y="market_value", template="simple_white", color_discrete_sequence=['black']))
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.subheader("Goals vs xG")
            fig = style_fig(px.scatter(df, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=['black']))
            st.plotly_chart(fig, use_container_width=True)

    # 3. TABLE VIEW
    elif st.session_state.view == 'Table':
        st.title(f"📄 Data Table: {selected_table}")
        st.dataframe(df, use_container_width=True, height=600)

    # 4. BAR GRAPHS
    elif st.session_state.view == 'Bar':
        st.title("📊 Bar Analysis")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("METRIC", num_cols, index=num_cols.index('market_value') if 'market_value' in num_cols else 0)
        fig = style_fig(px.bar(df.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white", color_discrete_sequence=['black']))
        st.plotly_chart(fig, use_container_width=True)

    # 5. DISTRIBUTIONS
    elif st.session_state.view == 'Dist':
        st.title("📈 Distributions")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        d_col = st.selectbox("METRIC", num_cols)
        fig = style_fig(px.histogram(df, x=d_col, template="simple_white", color_discrete_sequence=['black']))
        st.plotly_chart(fig, use_container_width=True)