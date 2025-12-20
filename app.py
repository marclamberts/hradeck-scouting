import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

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

    /* Fix for the hover state inside the dropdown menu */
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
            if any(k in c.lower() for k in ['value', 'age', 'goal', 'xg', 'match', 'per_90']):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = df[c].astype(str).replace(['nan', 'None'], '')
        return df

# Chart Styling Helper
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

    # Load Full Data
    df_raw = load_data(selected_table)

    # --- SHARED FILTER LOGIC (Global Filters) ---
    def render_filters(key_prefix):
        c1, c2, c3 = st.columns(3)
        with c1:
            team_options = ["ALL TEAMS"] + sorted([str(x) for x in df_raw["team"].unique() if x])
            f_team = st.selectbox("TEAM", team_options, key=f"{key_prefix}_team")
        with c2:
            pos_options = set()
            if "position" in df_raw.columns:
                for p in df_raw["position"].unique():
                    if p:
                        for s in str(p).split(','):
                            tag = s.strip()
                            if tag: pos_options.add(tag)
            pos_list = ["ALL POSITIONS"] + sorted(list(pos_options))
            f_pos = st.selectbox("POSITION", pos_list, key=f"{key_prefix}_pos")
        with c3:
            f_search = st.text_input("PLAYER SEARCH", key=f"{key_prefix}_search")
        return f_team, f_pos, f_search

    def apply_filters(data, f_team, f_pos, f_search):
        df_f = data.copy()
        if f_team != "ALL TEAMS":
            df_f = df_f[df_f["team"] == f_team]
        if f_pos != "ALL POSITIONS" and "position" in df_f.columns:
            # Fixing the Position Filter: checks if selected tag exists in the comma-separated list
            df_f = df_f[df_f["position"].apply(lambda x: f_pos in [s.strip() for s in str(x).split(',')])]
        if f_search:
            df_f = df_f[df_f["player"].str.contains(f_search, case=False, na=False)]
        return df_f

    # --- VIEWS ---

    if st.session_state.view == 'Search':
        st.title("🔍 Advanced Player Search")
        # Filters specific to this page
        f_team, f_pos, f_search = render_filters("search_page")
        df_filtered = apply_filters(df_raw, f_team, f_pos, f_search)
        
        st.write("---")
        st.subheader(f"Results ({len(df_filtered)} players)")
        st.dataframe(df_filtered, use_container_width=True, height=500)

    elif st.session_state.view == 'Dashboard':
        st.title(f"📊 Dashboard: {selected_table}")
        # Filters also on the dashboard
        f_team, f_pos, f_search = render_filters("dashboard_page")
        df_filtered = apply_filters(df_raw, f_team, f_pos, f_search)
        
        st.write("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PLAYERS", len(df_filtered))
        if 'market_value' in df_filtered.columns: 
            m2.metric("AVG VALUE", f"€{int(df_filtered['market_value'].mean()):,}")
        if 'goals' in df_filtered.columns: 
            m3.metric("TOTAL GOALS", int(df_filtered['goals'].sum()))
        if 'age' in df_filtered.columns: 
            m4.metric("AVG AGE", round(df_filtered['age'].mean(), 1))

        l, r = st.columns(2)
        with l:
            st.subheader("Value by Team")
            fig = style_fig(px.box(df_filtered, x="team", y="market_value", template="simple_white", color_discrete_sequence=['black']))
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.subheader("Goals vs xG")
            fig = style_fig(px.scatter(df_filtered, x="xg", y="goals", hover_name="player", template="simple_white", color_discrete_sequence=['black']))
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Table':
        st.title(f"📄 Data Table: {selected_table}")
        st.dataframe(df_raw, use_container_width=True, height=600)

    elif st.session_state.view == 'Bar':
        st.title("📊 Performance Bars")
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