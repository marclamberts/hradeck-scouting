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

# --- GLOBAL STYLING: ONE COLOR BACKGROUND + BLACK TEXT EVERYWHERE ---
# Background Color: #E0E2E6 (Consistent Grey)
# Font Color: #000000 (Pure Black)
st.markdown("""
    <style>
    /* 1. FORCE GLOBAL BACKGROUND AND FONT */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main, div[data-testid="stToolbar"] {
        background-color: #E0E2E6 !important;
    }

    /* 2. FORCE BLACK TEXT ON EVERYTHING */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, span, label, li, td, th {
        color: #000000 !important;
        font-family: 'Inter', sans-serif;
    }

    /* 3. UNIFY INPUT BOXES AND DROPDOWNS */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stTextInput input {
        background-color: #E0E2E6 !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
    }

    /* 4. UNIFY METRIC CARDS (No white background) */
    div[data-testid="metric-container"] {
        background-color: #E0E2E6 !important;
        border: 1px solid #000000 !important;
        border-radius: 10px;
        padding: 10px;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    /* 5. UNIFY BUTTONS */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        background-color: #E0E2E6 !important; 
        color: #000000 !important; 
        font-weight: bold;
        border: 2px solid #000000 !important;
    }
    .stButton>button:hover { 
        background-color: #000000 !important; 
        color: #FFFFFF !important; 
    }

    /* 6. UNIFY SIDEBAR ITEMS */
    [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stMarkdown p {
        color: #000000 !important;
    }
    
    /* 7. TABLE STYLING */
    .stDataFrame {
        border: 1px solid #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. AUTHENTICATION ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    _, col2, _ = st.columns([1, 1.2, 1])
    with col2:
        st.image(LOGO_URL, width=80)
        st.title("HRADECK LOGIN")
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In"):
                if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access Denied.")
    return False

# --- 2. DATA UTILITIES ---
def get_all_tables():
    with sqlite3.connect(DB_FILE) as conn:
        return [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]

def load_clean_data(table_name):
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        df = df.fillna(0)
        for col in df.columns:
            if any(k in col.lower() for k in ['value', 'age', 'goal', 'xg', 'match', 'per_90']):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

# --- 3. MAIN APP ---
if check_password():
    # Sidebar Setup
    st.sidebar.title("SCOUT MENU")
    tables = get_all_tables()
    selected_table = st.sidebar.selectbox("📂 DATASET", tables)
    
    if 'view' not in st.session_state:
        st.session_state.view = 'Dashboard'

    st.sidebar.markdown("---")
    if st.sidebar.button("🏠 DASHBOARD"): st.session_state.view = 'Dashboard'
    if st.sidebar.button("📄 TABLE VIEW"): st.session_state.view = 'Table'
    if st.sidebar.button("📊 BAR GRAPHS"): st.session_state.view = 'Bar'
    if st.sidebar.button("📈 DISTRIBUTIONS"): st.session_state.view = 'Dist'
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # Load Data
    df_raw = load_clean_data(selected_table)

    # Top Filters
    st.title(f"EXPLORING: {selected_table}")
    c1, c2, c3 = st.columns(3)
    with c1:
        teams = ["All Teams"] + sorted(df_raw["team"].unique().tolist()) if "team" in df_raw.columns else ["N/A"]
        filt_team = st.selectbox("Team Filter", teams)
    with c2:
        pos = ["All Positions"]
        if "position" in df_raw.columns:
            pos += sorted(list(set([x.strip() for p in df_raw["position"].unique() for x in str(p).split(',')])))
        filt_pos = st.selectbox("Position Filter", pos)
    with c3:
        search = st.text_input("🔍 Player Search")

    # Filter Logic
    df = df_raw.copy()
    if filt_team != "All Teams": df = df[df["team"] == filt_team]
    if filt_pos != "All Positions" and "position" in df.columns:
        df = df[df["position"].astype(str).str.contains(filt_pos, na=False)]
    if search: df = df[df["player"].astype(str).str.contains(search, case=False, na=False)]

    st.divider()

    # --- VIEWS ---
    if st.session_state.view == 'Dashboard':
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Players", len(df))
        if 'market_value' in df.columns: m2.metric("Avg Value", f"€{int(df['market_value'].mean()):,}")
        if 'goals' in df.columns: m3.metric("Goals", int(df['goals'].sum()))
        if 'age' in df.columns: m4.metric("Avg Age", round(df['age'].mean(), 1))

        # Visuals
        l, r = st.columns(2)
        with l:
            st.subheader("Team Values")
            fig = px.box(df, x="team", y="market_value", template="simple_white")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.subheader("xG vs Goals")
            fig = px.scatter(df, x="xg", y="goals", hover_name="player", template="simple_white")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Table':
        st.dataframe(df, use_container_width=True, height=600)

    elif st.session_state.view == 'Bar':
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("Metric", num_cols, index=num_cols.index('market_value') if 'market_value' in num_cols else 0)
        fig = px.bar(df.sort_values(y_col, ascending=False).head(20), x="player", y=y_col, template="simple_white")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Dist':
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        d_col = st.selectbox("Metric", num_cols)
        fig = px.histogram(df, x=d_col, template="simple_white", color_discrete_sequence=['black'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)