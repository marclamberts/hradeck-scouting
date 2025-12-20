import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
LOGO_FILE = "FCHK.png"  
THEME_COLOR = "#DDE1E6" # Unified Slate Grey

st.set_page_config(page_title="FCHK Pro Scout", layout="wide", page_icon="⚽")

# --- ULTIMATE CSS: TOTAL UNIFICATION & BLACK TEXT ---
st.markdown(f"""
    <style>
    /* Global Background */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main, 
    [data-testid="stSidebarNav"], .stAppHeader, [data-testid="stDecoration"],
    div[data-testid="stToolbar"], [data-testid="stSidebar"] div, .stAppViewContainer {{
        background-color: {THEME_COLOR} !important;
    }}

    /* Force Pure Black Text & Labels */
    html, body, .stMarkdown, p, h1, h2, h3, h4, span, label, li, td, th, 
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] *, .stSelectbox label, .stTextInput label,
    div[role="listbox"] div, .stDataFrame div, button, .stTab, .stCaption, 
    .stSlider label, [data-testid="stMetricDelta"] {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Inputs & Dropdowns */
    div[data-baseweb="select"] > div, div[data-baseweb="popover"] div, 
    ul[role="listbox"], li[role="option"], div[data-baseweb="input"] > div, 
    .stTextInput input, div[role="combobox"] {{
        background-color: {THEME_COLOR} !important;
        color: #000000 !important;
        border: 1.5px solid #000000 !important;
    }}

    /* Buttons: No Hover Change */
    .stButton>button {{ 
        width: 100%; border-radius: 4px; background-color: transparent !important; 
        color: #000000 !important; font-weight: bold; border: 1.5px solid #000000 !important;
        transition: none !important; text-transform: uppercase;
    }}

    /* Metric Cards */
    div[data-testid="metric-container"] {{
        background-color: transparent !important; border: 1.5px solid #000000 !important;
        border-radius: 8px; padding: 15px;
    }}
    header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# --- 1. AUTHENTICATION ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated: return True
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        try: st.image(LOGO_FILE, width=120)
        except: st.warning("Logo file 'FCHK.png' not found.")
        st.title("FCHK LOGIN")
        with st.form("login"):
            u, p = st.text_input("USER"), st.text_input("PASSWORD", type="password")
            if st.form_submit_button("ENTER SYSTEM"):
                if u == VALID_USERNAME and p == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("ACCESS DENIED")
    return False

# --- 2. DATA UTILITIES ---
def load_data(table):
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        for c in df.columns:
            if any(k in c for k in ['value', 'age', 'goal', 'xg', 'match', 'minutes']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                # String cleaning to prevent 'NULL' strings showing in UI
                df[c] = df[c].astype(str).replace(['nan', 'None', 'NULL', 'null'], '')
        return df

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=THEME_COLOR, plot_bgcolor=THEME_COLOR,
        font=dict(color="black", size=12),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor="#BBBBBB", linecolor="black"),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor="#BBBBBB", linecolor="black")
    )
    return fig

# --- 3. THE FIXED POSITION FILTER ENGINE ---
def apply_filters(data, f_team, f_pos_list, f_search, f_age, f_mins):
    df_f = data.copy()
    if f_team != "ALL TEAMS":
        df_f = df_f[df_f["team"] == f_team]
    
    # EXACT POSITION MATCHING
    if f_pos_list:
        def match_pos(cell_val):
            if not cell_val or cell_val == '': return False
            # Normalize and split database string: "CB, LCB" -> ["CB", "LCB"]
            player_positions = [p.strip().upper() for p in str(cell_val).split(',')]
            # Check for overlap with user selection
            return any(pos.upper() in player_positions for pos in f_pos_list)
        df_f = df_f[df_f["position"].apply(match_pos)]
        
    if f_search:
        df_f = df_f[df_f["player"].str.contains(f_search, case=False, na=False)]
    if 'age' in df_f.columns:
        df_f = df_f[df_f['age'].between(f_age[0], f_age[1])]
    if 'minutes_played' in df_f.columns:
        df_f = df_f[df_f['minutes_played'].between(f_mins[0], f_mins[1])]
    return df_f

# --- 4. MAIN APP ---
if check_password():
    with sqlite3.connect(DB_FILE) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    
    selected_table = st.sidebar.selectbox("DATASET", tables, key="table_select")
    df_raw = load_data(selected_table)

    # State initialization for global filters
    for key, val in [('view','Dashboard'), ('f_team','ALL TEAMS'), ('f_pos',[]), 
                     ('f_search',''), ('f_age',(15,45)), ('f_mins',(0,5000))]:
        if key not in st.session_state: st.session_state[key] = val

    # Sidebar Navigation
    st.sidebar.write("##")
    if st.sidebar.button("🏠 DASHBOARD"): st.session_state.view = 'Dashboard'
    if st.sidebar.button("🔍 SEARCH"): st.session_state.view = 'Search'
    if st.sidebar.button("📊 BAR RANKING"): st.session_state.view = 'Bar'
    if st.sidebar.button("📈 DISTRIBUTIONS"): st.session_state.view = 'Dist'

    def filter_ui(key):
        c1, c2, c3 = st.columns(3)
        with c1:
            teams = ["ALL TEAMS"] + sorted([x for x in df_raw["team"].unique() if x])
            st.session_state.f_team = st.selectbox("TEAM", teams, index=teams.index(st.session_state.f_team) if st.session_state.f_team in teams else 0, key=f"{key}_t")
        with c2:
            tags = set()
            for p in df_raw["position"].unique():
                if p and p != '': [tags.add(s.strip()) for s in str(p).split(',')]
            st.session_state.f_pos = st.multiselect("POSITIONS", sorted(list(tags)), default=st.session_state.f_pos, key=f"{key}_p")
        with c3:
            st.session_state.f_search = st.text_input("NAME", value=st.session_state.f_search, key=f"{key}_s")

    # --- VIEWS ---
    if st.session_state.view == 'Search':
        st.title("🔍 Scout Search")
        filter_ui("search_view")
        s1, s2 = st.columns(2)
        with s1: st.session_state.f_age = st.slider("AGE", 15, 50, st.session_state.f_age, key="sl_age")
        with s2: st.session_state.f_mins = st.slider("MINUTES", 0, 10000, st.session_state.f_mins, key="sl_mins")
        df_f = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        st.dataframe(df_f, width='stretch', height=600)

    elif st.session_state.view == 'Dashboard':
        st.title("📊 Analytics Dashboard")
        filter_ui("dash_view")
        df_f = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PLAYERS", len(df_f))
        if 'market_value' in df_f.columns: m2.metric("AVG VALUE", f"€{int(df_f['market_value'].mean()):,}")
        if 'goals' in df_f.columns: m3.metric("GOALS", int(df_f['goals'].sum()))
        if 'age' in df_f.columns: m4.metric("AVG AGE", round(df_f['age'].mean(), 1))
        
        l, r = st.columns(2)
        with l: 
            fig_box = px.box(df_f, x="team", y="market_value", template="simple_white", color_discrete_sequence=['black'])
            st.plotly_chart(style_fig(fig_box), width='stretch')
        with r: 
            fig_scat = px.scatter(df_f, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=['black'])
            st.plotly_chart(style_fig(fig_scat), width='stretch')

    elif st.session_state.view == 'Bar':
        st.title("📊 Rankings")
        filter_ui("bar_view")
        df_f = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        num_cols = df_f.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("METRIC", num_cols)
        fig_bar = px.bar(df_f.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white", color_discrete_sequence=['black'])
        st.plotly_chart(style_fig(fig_bar), width='stretch')

    elif st.session_state.view == 'Dist':
        st.title("📈 Variable Distributions")
        filter_ui("dist_view")
        df_f = apply_filters(df_raw, st.session_state.f_team, st.session_state.f_pos, st.session_state.f_search, st.session_state.f_age, st.session_state.f_mins)
        d_col = st.selectbox("METRIC", df_f.select_dtypes(include=['number']).columns.tolist())
        fig_hist = px.histogram(df_f, x=d_col, template="simple_white", color_discrete_sequence=['black'])
        st.plotly_chart(style_fig(fig_hist), width='stretch')