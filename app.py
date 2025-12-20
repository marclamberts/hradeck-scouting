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
THEME_COLOR = "#DDE1E6" # The unified background color

st.set_page_config(page_title="FCHK Pro Scout", layout="wide", page_icon="⚽")

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
    div[role="listbox"] div, .stDataFrame div, button, .stTab, .stCaption, .stSlider label {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* 3. Inputs & DROPDOWN MENUS */
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

    /* Fix for dropdown hover */
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
        st.title("FCHK LOGIN")
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
            if any(k in c.lower() for k in ['value', 'age', 'goal', 'xg', 'match', 'per_90', 'minutes']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = df[c].astype(str).replace(['nan', 'None'], '')
        return df

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

# --- MAIN APP ---
if check_password():
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

    # Load Full Data
    df_raw = load_data(selected_table)

    # --- FILTER FUNCTIONS ---
    def apply_scout_filters(data, f_team, f_pos, f_search, f_age=(0, 100), f_mins=(0, 10000)):
        df_f = data.copy()
        if f_team != "ALL TEAMS":
            df_f = df_f[df_f["team"] == f_team]
        if f_pos != "ALL POSITIONS" and "position" in df_f.columns:
            df_f = df_f[df_f["position"].apply(lambda x: f_pos in [s.strip() for s in str(x).split(',')])]
        if f_search:
            df_f = df_f[df_f["player"].str.contains(f_search, case=False, na=False)]
        
        # Age Filter
        if 'age' in df_f.columns:
            df_f = df_f[df_f['age'].between(f_age[0], f_age[1])]
        
        # Minutes Played Filter
        if 'minutes_played' in df_f.columns:
            df_f = df_f[df_f['minutes_played'].between(f_mins[0], f_mins[1])]
            
        return df_f

    # --- VIEWS ---

    if st.session_state.view == 'Search':
        st.title("🔍 Advanced Scout Search")
        
        # Row 1: Basic
        c1, c2, c3 = st.columns(3)
        with c1:
            team_options = ["ALL TEAMS"] + sorted([str(x) for x in df_raw["team"].unique() if x])
            f_team = st.selectbox("TEAM", team_options, key="search_team")
        with c2:
            pos_options = set()
            if "position" in df_raw.columns:
                for p in df_raw["position"].unique():
                    if p: [pos_options.add(s.strip()) for s in str(p).split(',')]
            all_pos = ["ALL POSITIONS"] + sorted(list(pos_options))
            f_pos = st.selectbox("POSITION", all_pos, key="search_pos")
        with c3:
            f_search = st.text_input("NAME SEARCH", key="search_name")

        # Row 2: Advanced Sliders
        st.write("## Advanced Filtering")
        s1, s2 = st.columns(2)
        with s1:
            if 'age' in df_raw.columns:
                min_a, max_a = int(df_raw['age'].min()), int(df_raw['age'].max())
                f_age = st.slider("AGE RANGE", min_a, max_a, (min_a, max_a), key="search_age")
            else:
                f_age = (0, 100)
        with s2:
            if 'minutes_played' in df_raw.columns:
                min_m, max_m = int(df_raw['minutes_played'].min()), int(df_raw['minutes_played'].max())
                f_mins = st.slider("MINUTES PLAYED", min_m, max_m, (min_m, max_m), key="search_mins")
            else:
                f_mins = (0, 100000)

        df_filtered = apply_scout_filters(df_raw, f_team, f_pos, f_search, f_age, f_mins)
        
        st.write("---")
        st.subheader(f"Results ({len(df_filtered)} players)")
        st.dataframe(df_filtered, use_container_width=True, height=500)

    elif st.session_state.view == 'Dashboard':
        st.title(f"📊 Dashboard: {selected_table}")
        
        # Dashboard maintains standard filters
        c1, c2, c3 = st.columns(3)
        with c1:
            team_opts = ["ALL TEAMS"] + sorted([str(x) for x in df_raw["team"].unique() if x])
            f_team = st.selectbox("TEAM", team_opts, key="dash_team")
        with c2:
            p_opts = set()
            if "position" in df_raw.columns:
                for p in df_raw["position"].unique():
                    if p: [p_opts.add(s.strip()) for s in str(p).split(',')]
            all_p = ["ALL POSITIONS"] + sorted(list(p_opts))
            f_pos = st.selectbox("POSITION", all_p, key="dash_pos")
        with c3:
            f_search = st.text_input("PLAYER SEARCH", key="dash_name")
        
        df_dash = apply_scout_filters(df_raw, f_team, f_pos, f_search)
        
        st.write("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PLAYERS", len(df_dash))
        if 'market_value' in df_dash.columns: m2.metric("AVG VALUE", f"€{int(df_dash['market_value'].mean()):,}")
        if 'goals' in df_dash.columns: m3.metric("TOTAL GOALS", int(df_dash['goals'].sum()))
        if 'age' in df_dash.columns: m4.metric("AVG AGE", round(df_dash['age'].mean(), 1))

        l, r = st.columns(2)
        with l:
            st.subheader("Value by Team")
            fig = style_fig(px.box(df_dash, x="team", y="market_value", template="simple_white", color_discrete_sequence=['black']))
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.subheader("Goals vs xG")
            fig = style_fig(px.scatter(df_dash, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=['black']))
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Table':
        st.title(f"📄 Data Table: {selected_table}")
        st.dataframe(df_raw, use_container_width=True, height=600)

    elif st.session_state.view == 'Bar':
        st.title("📊 Bar Analysis")
        num_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("METRIC", num_cols, index=num_cols.index('market_value') if 'market_value' in num_cols else 0)
        fig = style_fig(px.bar(df_raw.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white", color_discrete_sequence=['black']))
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Dist':
        st.title("📈 Distributions")
        num_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
        d_col = st.selectbox("METRIC", num_cols)
        fig = style_fig(px.histogram(df_raw, x=d_col, template="simple_white", color_discrete_sequence=['black']))
        st.plotly_chart(fig, use_container_width=True)