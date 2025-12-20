import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/5329/5329945.png" 

# Set Page Config
st.set_page_config(page_title="Hradeck Pro Scout", layout="wide", page_icon="📈")

# --- UNIFIED CSS (Ensuring same background and font color everywhere) ---
# Background: #F0F2F6 (Light Grey) | Font: #1C1C1C (Dark Grey/Black)
st.markdown("""
    <style>
    /* 1. Global Background & Font Color */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"], .main {
        background-color: #F0F2F6 !important;
    }

    /* 2. Unified Font Color for all elements */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, span, label, .stMetric div {
        color: #1C1C1C !important;
    }

    /* 3. Sidebar specific font and background override */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label {
        color: #1C1C1C !important;
    }

    /* 4. Professional Sidebar Buttons */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        background-color: #FFFFFF; 
        color: #1C1C1C !important; 
        font-weight: 600;
        border: 1px solid #D1D5DB;
        transition: 0.2s;
    }
    .stButton>button:hover { 
        background-color: #004B91; 
        color: #FFFFFF !important; 
        border-color: #004B91;
    }

    /* 5. Clean Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #D1D5DB !important;
    }

    /* 6. Fix for Selectbox/Input focus and background */
    .stSelectbox div, .stTextInput div {
        background-color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. AUTHENTICATION SYSTEM ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Login UI
    _, col2, _ = st.columns([1, 1.2, 1])
    with col2:
        st.write("#")
        st.image(LOGO_URL, width=80)
        st.title("Hradeck Scouting")
        st.subheader("Professional Data Access")
        
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In")

            if submit:
                if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    return False

# --- 2. DATA UTILITIES ---
@st.cache_data
def get_all_tables():
    with sqlite3.connect(DB_FILE) as conn:
        return [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]

def load_clean_data(table_name):
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        df = df.fillna('')
        # Standardize numeric columns for scouting
        numeric_keywords = ['value', 'age', 'goal', 'xg', 'match', 'assist', 'won', 'accurate', 'per_90']
        for col in df.columns:
            if any(key in col.lower() for key in numeric_keywords):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

# --- 3. MAIN APPLICATION ---
if check_password():
    # Sidebar Logo and Title
    st.sidebar.image(LOGO_URL, width=50)
    st.sidebar.markdown("### Navigation")
    
    # Dataset Selector
    tables = get_all_tables()
    selected_table = st.sidebar.selectbox("📂 Select Dataset", tables)
    st.sidebar.markdown("---")
    
    # Navigation View State
    if 'view' not in st.session_state:
        st.session_state.view = 'Dashboard'

    # Navigation Buttons
    if st.sidebar.button("🏠 Executive Dashboard"):
        st.session_state.view = 'Dashboard'
    if st.sidebar.button("📄 Raw Data Table"):
        st.session_state.view = 'Table'
    if st.sidebar.button("📊 Bar Graphs"):
        st.session_state.view = 'Bar'
    if st.sidebar.button("📈 Distribution Plots"):
        st.session_state.view = 'Dist'
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # Load Data
    df_raw = load_clean_data(selected_table)

    # --- TOP FILTERS (Horizontal Row) ---
    st.title(f"⚽ {selected_table} Intel")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        team_list = ["All Teams"] + sorted(df_raw["team"].unique().tolist()) if "team" in df_raw.columns else ["N/A"]
        filt_team = st.selectbox("Team Filter", team_list)
    with c2:
        pos_list = ["All Positions"]
        if "position" in df_raw.columns:
            pos_set = set()
            for p in df_raw["position"].unique():
                for sub_p in str(p).split(','):
                    pos_set.add(sub_p.strip())
            pos_list += sorted(list(pos_set))
        filt_pos = st.selectbox("Position Filter", pos_list)
    with c3:
        search = st.text_input("🔍 Player Search", placeholder="Type name...")

    # Filter Logic
    df = df_raw.copy()
    if filt_team != "All Teams":
        df = df[df["team"] == filt_team]
    if filt_pos != "All Positions" and "position" in df.columns:
        df = df[df["position"].astype(str).str.contains(filt_pos, na=False)]
    if search:
        df = df[df["player"].astype(str).str.contains(search, case=False, na=False)]

    st.divider()

    # --- VIEW: DASHBOARD ---
    if st.session_state.view == 'Dashboard':
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pool Size", len(df))
        if 'market_value' in df.columns:
            m2.metric("Avg Value", f"€{int(df['market_value'].mean()):,}")
        if 'goals' in df.columns:
            m3.metric("Total Goals", int(df['goals'].sum()))
        if 'age' in df.columns:
            m4.metric("Avg Age", round(df['age'].mean(), 1))

        st.markdown("### 🏆 Top 5 Market Value")
        if 'market_value' in df.columns:
            top_players = df.sort_values("market_value", ascending=False).head(5)
            cols = st.columns(5)
            for i, (_, player) in enumerate(top_players.iterrows()):
                with cols[i]:
                    st.markdown(f"**{player['player']}**")
                    st.caption(player['team'])
                    st.markdown(f"**€{player['market_value']:,}**")

        st.write("##")
        dl, dr = st.columns(2)
        with dl:
            st.subheader("Team Value Spread")
            fig = px.box(df, x="team", y="market_value", color="team", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with dr:
            st.subheader("Goal Accuracy")
            fig = px.scatter(df, x="xg", y="goals", hover_name="player", color="team", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # --- VIEW: TABLE ---
    elif st.session_state.view == 'Table':
        st.subheader("Detailed Records")
        st.dataframe(df, use_container_width=True, height=600)

    # --- VIEW: BAR GRAPHS ---
    elif st.session_state.view == 'Bar':
        st.subheader("Performance Comparison")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            y_col = st.selectbox("Select Metric", numeric_cols, index=numeric_cols.index('market_value') if 'market_value' in numeric_cols else 0)
        with b_c2:
            sort = st.radio("Display", ["Top 20", "Bottom 20"], horizontal=True)

        is_asc = True if sort == "Bottom 20" else False
        fig_bar = px.bar(df.sort_values(y_col, ascending=is_asc).head(20), 
                         x="player", y=y_col, color=y_col, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- VIEW: DISTRIBUTIONS ---
    elif st.session_state.view == 'Dist':
        st.subheader("Statistical Distributions")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        d_col = st.selectbox("Select Metric", numeric_cols, index=numeric_cols.index('age') if 'age' in numeric_cols else 0)
        
        fig_dist = px.histogram(df, x=d_col, marginal="box", nbins=30, template="plotly_white", color_discrete_sequence=['#004B91'])
        st.plotly_chart(fig_dist, use_container_width=True)