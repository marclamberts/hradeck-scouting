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
    B_COLOR = "#0E1117"  # Deep Dark
    T_COLOR = "#FFFFFF"  # White Text
    PILL_TRACK = "#1E1E1E" 
    PILL_HANDLE = "#00FF00" 
    GRID = "#31333F"
else:
    B_COLOR = "#DDE1E6"  # FCHK Slate Grey
    T_COLOR = "#000000"  # Black Text
    PILL_TRACK = "#BDC1C6" 
    PILL_HANDLE = "#000000" 
    GRID = "#BBBBBB"

# --- 2. CSS: UNIFIED FILTERS & ISOLATED PILL ---
st.markdown(f"""
    <style>
    /* Global Backgrounds */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main, 
    [data-testid="stSidebarNav"], .stAppHeader, [data-testid="stDecoration"],
    div[data-testid="stToolbar"], [data-testid="stSidebar"] div, .stAppViewContainer {{
        background-color: {B_COLOR} !important;
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

    /* THE PILL TOGGLE (Isolated to Theme Switch) */
    div[data-testid="stCheckbox"] > label > div:first-child {{
        background-color: {PILL_TRACK} !important;
        border: 2px solid {T_COLOR} !important;
        width: 46px !important;
        height: 24px !important;
        border-radius: 12px !important;
        display: flex !important;
        align-items: center !important;
    }}
    div[data-testid="stCheckbox"] > label > div:first-child > div {{
        background-color: {PILL_HANDLE} !important;
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
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

# --- 3. MAPPING & DATA ---
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
        for c in df.columns:
            if any(k in c for k in ['value', 'age', 'goal', 'xg', 'match', 'minutes']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
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
                else: st.error("DENIED")
    return False

# --- 5. MAIN APP ---
if check_password():
    with st.sidebar:
        st.write("### ⚙️ SYSTEM")
        theme_toggle = st.toggle("DARK MODE", value=(st.session_state.theme == 'Dark'), key="theme_switch")
        st.session_state.theme = 'Dark' if theme_toggle else 'Light'
        st.write("---")
        st.write("### 🧭 PAGES")
        if st.button("📊 DASHBOARD"): st.session_state.view = 'Dashboard'
        if st.button("🔍 SEARCH"): st.session_state.view = 'Search'
        if st.button("🏆 RANKINGS"): st.session_state.view = 'Bar'
        if st.button("📈 DISTRIBUTIONS"): st.session_state.view = 'Dist'
        st.write("---")
        if st.button("LOGOUT"):
            st.session_state.authenticated = False
            st.rerun()

    with sqlite3.connect(DB_FILE) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    
    selected_table = st.sidebar.selectbox("DATASET", tables, key="table_select")
    df_raw = load_data(selected_table)

    for key, val in [('view','Dashboard'), ('f_team','ALL TEAMS'), ('f_group','ALL GROUPS'), 
                     ('f_search',''), ('f_age',(15,45)), ('f_mins',(0,5000))]:
        if key not in st.session_state: st.session_state[key] = val

    def filter_ui(key):
        c1, c2, c3 = st.columns(3)
        with c1:
            teams = ["ALL TEAMS"] + sorted([x for x in df_raw["team"].unique() if x])
            st.session_state.f_team = st.selectbox("TEAM", teams, index=teams.index(st.session_state.f_team) if st.session_state.f_team in teams else 0, key=f"{key}_t")
        with c2:
            groups = ["ALL GROUPS", "Goalkeeper", "Defender", "Midfielder", "Attacker"]
            st.session_state.f_group = st.selectbox("GROUP", groups, index=groups.index(st.session_state.f_group) if st.session_state.f_group in groups else 0, key=f"{key}_g")
        with c3:
            st.session_state.f_search = st.text_input("PLAYER NAME", value=st.session_state.f_search, key=f"{key}_s")

    def apply_filters(data):
        df_f = data.copy()
        if st.session_state.f_team != "ALL TEAMS": df_f = df_f[df_f["team"] == st.session_state.f_team]
        if st.session_state.f_group != "ALL GROUPS": df_f = df_f[df_f["position_group"] == st.session_state.f_group]
        if st.session_state.f_search: df_f = df_f[df_f["player"].str.contains(st.session_state.f_search, case=False, na=False)]
        if 'age' in df_f.columns: df_f = df_f[df_f['age'].between(st.session_state.f_age[0], st.session_state.f_age[1])]
        if 'minutes_played' in df_f.columns: df_f = df_f[df_f['minutes_played'].between(st.session_state.f_mins[0], st.session_state.f_mins[1])]
        return df_f

    df_f = apply_filters(df_raw)

    # --- PAGE VIEWS ---
    if st.session_state.view == 'Search':
        st.title("🔍 Scout Search")
        filter_ui("search")
        s1, s2 = st.columns(2)
        with s1: st.session_state.f_age = st.slider("AGE", 15, 50, st.session_state.f_age, key="sl1")
        with s2: st.session_state.f_mins = st.slider("MINUTES", 0, 10000, st.session_state.f_mins, key="sl2")
        st.dataframe(df_f, width='stretch', height=600)

    elif st.session_state.view == 'Dashboard':
        st.title("📊 Analytics Dashboard")
        filter_ui("dash")
        m1, m2 = st.columns(2)
        m1.metric("PLAYERS", len(df_f))
        if 'market_value' in df_f.columns: m2.metric("AVG VALUE", f"€{int(df_f['market_value'].mean()):,}")
        l, r = st.columns(2)
        with l: st.plotly_chart(style_fig(px.box(df_f, x="team", y="market_value", template="simple_white", color_discrete_sequence=[T_COLOR])), width='stretch')
        with r: st.plotly_chart(style_fig(px.scatter(df_f, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=[T_COLOR])), width='stretch')

    elif st.session_state.view == 'Bar':
        st.title("🏆 Player Rankings")
        filter_ui("bar")
        num_cols = df_f.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            y_col = st.selectbox("RANK BY", num_cols)
            st.plotly_chart(style_fig(px.bar(df_f.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white", color_discrete_sequence=[T_COLOR])), width='stretch')

    elif st.session_state.view == 'Dist':
        st.title("📈 Statistical Distributions")
        filter_ui("dist")
        d_col = st.selectbox("METRIC SPREAD", df_f.select_dtypes(include=['number']).columns.tolist())
        st.plotly_chart(style_fig(px.histogram(df_f, x=d_col, template="simple_white", color_discrete_sequence=[T_COLOR])), width='stretch')