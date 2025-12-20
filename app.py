import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
LOGO_FILE = "FCHK.png"  
THEME_COLOR = "#DDE1E6" # Unified Slate Grey background

st.set_page_config(page_title="FCHK Pro Scout", layout="wide", page_icon="⚽")

# --- CSS: TOTAL UNIFICATION & BLACK TEXT ---
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
    div[role="listbox"] div, .stDataFrame div, button, .stTab, .stCaption, 
    .stSlider label, [data-testid="stMetricDelta"] {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* 3. Inputs & Dropdowns - Uniform Grey Facecolor */
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

    /* 4. Buttons: Minimalist Bordered Style */
    .stButton>button {{ 
        width: 100%; border-radius: 4px; background-color: transparent !important; 
        color: #000000 !important; font-weight: bold; border: 1.5px solid #000000 !important;
        transition: none !important; text-transform: uppercase;
    }}
    .stButton>button:hover {{ 
        background-color: transparent !important; color: #000000 !important; 
        border-color: #000000 !important; box-shadow: none !important; 
    }}

    /* 5. Metrics & Containers */
    div[data-testid="metric-container"] {{
        background-color: transparent !important; border: 1.5px solid #000000 !important;
        border-radius: 8px; padding: 15 (px);
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
        try: st.image(LOGO_FILE, width=120)
        except: st.warning("Logo 'FCHK.png' not found.")
        st.title("FCHK LOGIN")
        with st.form("login"):
            u, p = st.text_input("USER"), st.text_input("PASSWORD", type="password")
            if st.form_submit_button("ENTER"):
                if u == VALID_USERNAME and p == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("ACCESS DENIED")
    return False

# --- DATA UTILITIES ---
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

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=THEME_COLOR, plot_bgcolor=THEME_COLOR,
        font=dict(color="black", size=12),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor="#BBBBBB", linecolor="black"),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor="#BBBBBB", linecolor="black"),
        legend=dict(font=dict(color="black"), bgcolor="rgba(0,0,0,0)")
    )
    return fig

# --- THE FILTER ENGINE ---
def apply_filters(data, f_team, f_pos_list, f_search, f_age, f_mins):
    df_f = data.copy()
    
    if f_team != "ALL TEAMS":
        df_f = df_f[df_f["team"] == f_team]
    
    # ADVANCED POSITION LOGIC: 
    # Splits the player's positions and checks if any selected tag is present.
    if f_pos_list:
        def match_pos(cell):
            player_positions = [p.strip() for p in str(cell).split(',')]
            return any(pos in player_positions for pos in f_pos_list)
        df_f = df_f[df_f["position"].apply(match_pos)]
        
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
    selected_table = st.sidebar.selectbox("DATASET", get_tables(), key="table_select")
    
    if 'view' not in st.session_state: st.session_state.view = 'Dashboard'
    
    # Filter States
    if 'f_team' not in st.session_state: st.session_state.f_team = "ALL TEAMS"
    if 'f_pos' not in st.session_state: st.session_state.f_pos = []
    if 'f_search' not in st.session_state: st.session_state.f_search = ""
    if 'f_age' not in st.session_state: st.session_state.f_age = (15, 45)
    if 'f_mins' not in st.session_state: st.session_state.f_mins = (0, 5000)

    st.sidebar.write("##")
    if st.sidebar.button("🏠 DASHBOARD"): st.session_state.view = 'Dashboard'
    if st.sidebar.button("🔍 SEARCH"): st.session_state.view = 'Search'
    if st.sidebar.button("📊 BAR RANKING"): st.session_state.view = 'Bar'
    
    df_raw = load_data(selected_table)

    # --- SHARED FILTER UI ---
    def render_filters(key):
        c1, c2, c3 = st.columns(3)
        with c1:
            teams = ["ALL TEAMS"] + sorted([str(x) for x in df_raw["team"].unique() if x])
            st.session_state.f_team = st.selectbox("TEAM", teams, index=teams.index(st.session_state.f_team) if st.session_state.f_team in teams else 0, key=f"{key}_t")
        with c2:
            # Extract tags directly from the data to ensure correct spelling
            all_tags = set()
            for p in df_raw["position"].unique():
                if p: [all_tags.add(s.strip()) for s in str(p).split(',')]
            st.session_state.f_pos = st.multiselect("POSITIONS", sorted(list(all_tags)), default=st.session_state.f_pos, key=f"{key}_p")
        with c3:
            st.session_state.f_search = st.text_input("NAME", value=st.session_state.f_search, key=f"{key}_s")

    # --- VIEWS ---
    if st.session_state.view == 'Search':
        st.title("🔍 Advanced Scout Search")
        render_filters("search_pg")
        
        s1, s2 = st.columns(2)
        with s1:
            st.session_state.f_age = st.slider("AGE", 15, 50, st.session_state.f_age, key="s_age")
        with s2:
            st.session_state.f_mins = st.slider("MINS", 0, 10000, st.session_state.f_mins, key="s_mins")
        
        df_f = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        st.divider()
        st.dataframe(df_f, use_container_width=True, height=500)

    elif st.session_state.view == 'Dashboard':
        st.title(f"📊 {selected_table} Summary")
        render_filters("dash_pg")
        
        df_d = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        st.divider()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PLAYERS", len(df_d))
        if 'market_value' in df_d.columns: m2.metric("AVG VALUE", f"€{int(df_d['market_value'].mean()):,}")
        if 'goals' in df_d.columns: m3.metric("TOTAL GOALS", int(df_d['goals'].sum()))
        if 'age' in df_d.columns: m4.metric("AVG AGE", round(df_d['age'].mean(), 1))

        fig = style_fig(px.scatter(df_d, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=['black']))
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Bar':
        st.title("📊 Bar Ranking")
        df_b = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        num_cols = df_b.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("METRIC", num_cols)
        fig = style_fig(px.bar(df_b.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white", color_discrete_sequence=['black']))
        st.plotly_chart(fig, use_container_width=True)