import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
# Professional scouting/analytics icon
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/5329/5329945.png" 

st.set_page_config(page_title="Hradeck Pro Scout", layout="wide", page_icon="📈")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8f9fa; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #262730; }
    
    /* Button styling */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.2em; 
        background-color: #004b91; 
        color: white; 
        font-weight: 500;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #003366; border: none; color: #ffcc00; }
    
    /* Metric Card styling */
    [data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #004b91; }
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef0f2;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. AUTHENTICATION SYSTEM ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Center-aligned Login UI
    _, col2, _ = st.columns([1, 1.2, 1])
    with col2:
        st.write("#")
        st.image(LOGO_URL, width=90)
        st.title("Hradeck Scouting")
        st.subheader("Professional Data Portal")
        
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In")

            if submit:
                if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Access Denied.")
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
        for col in df.columns:
            # Force numeric conversion for known scouting stats
            if any(key in col.lower() for key in ['value', 'age', 'goal', 'xg', 'match', 'assist', 'won', 'accurate', 'per_90']):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

# --- 3. MAIN APPLICATION ---
if check_password():
    # --- SIDEBAR NAVIGATION ---
    st.sidebar.image(LOGO_URL, width=60)
    st.sidebar.title("Scout Menu")
    
    tables = get_all_tables()
    selected_table = st.sidebar.selectbox("📂 Dataset Select", tables)
    
    st.sidebar.markdown("---")
    
    # Init View State
    if 'view' not in st.session_state:
        st.session_state.view = 'Dashboard'

    # Navigation Buttons
    if st.sidebar.button("🏠 Executive Dashboard", use_container_width=True):
        st.session_state.view = 'Dashboard'
    if st.sidebar.button("📄 Raw Data Table", use_container_width=True):
        st.session_state.view = 'Table'
    if st.sidebar.button("📊 Bar Graphs", use_container_width=True):
        st.session_state.view = 'Bar'
    if st.sidebar.button("📈 Distribution Plots", use_container_width=True):
        st.session_state.view = 'Dist'
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    # Load Data
    df_raw = load_clean_data(selected_table)

    # --- TOP FILTERS (Next to each other) ---
    st.title(f"⚽ {selected_table} Intel")
    
    # Layout 3 columns for filters
    c1, c2, c3 = st.columns(3)
    with c1:
        team_list = ["All Teams"] + sorted(df_raw["team"].unique().tolist()) if "team" in df_raw.columns else ["N/A"]
        filt_team = st.selectbox("Filter Team", team_list)
    with c2:
        pos_list = ["All Positions"]
        if "position" in df_raw.columns:
            pos_set = set()
            for p in df_raw["position"].unique():
                for sub_p in str(p).split(','):
                    pos_set.add(sub_p.strip())
            pos_list += sorted(list(pos_set))
        filt_pos = st.selectbox("Filter Position", pos_list)
    with c3:
        search = st.text_input("🔍 Search Player", placeholder="Type name...")

    # Shared Filtering Logic
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
            m2.metric("Avg Market Value", f"€{int(df['market_value'].mean()):,}")
        if 'goals' in df.columns:
            m3.metric("Goals Scored", int(df['goals'].sum()))
        if 'age' in df.columns:
            m4.metric("Avg Age", round(df['age'].mean(), 1))

        st.markdown("### 🏆 Top 5 Value Players")
        if 'market_value' in df.columns:
            top_players = df.sort_values("market_value", ascending=False).head(5)
            cols = st.columns(5)
            for i, (_, player) in enumerate(top_players.iterrows()):
                with cols[i]:
                    st.markdown(f"**{player['player']}**")
                    st.caption(f"{player['team']}")
                    st.markdown(f"**€{player['market_value']:,}**")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Market Value by Team")
            fig = px.box(df, x="team", y="market_value", color="team")
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            st.subheader("Efficiency: xG vs Goals")
            fig = px.scatter(df, x="xg", y="goals", hover_name="player", color="team", size="market_value" if 'market_value' in df.columns else None)
            st.plotly_chart(fig, use_container_width=True)

    # --- VIEW: TABLE ---
    elif st.session_state.view == 'Table':
        st.subheader("Database Records")
        st.dataframe(df, use_container_width=True, height=600)

    # --- VIEW: BAR GRAPHS ---
    elif st.session_state.view == 'Bar':
        st.subheader("Comparative Bar Analysis")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        bc1, bc2 = st.columns(2)
        with bc1:
            y_metric = st.selectbox("Select Metric", numeric_cols, index=numeric_cols.index('market_value') if 'market_value' in numeric_cols else 0)
        with bc2:
            sort_order = st.radio("Display", ["Top 20", "Bottom 20"], horizontal=True)

        is_asc = True if sort_order == "Bottom 20" else False
        fig_bar = px.bar(df.sort_values(y_metric, ascending=is_asc).head(20), 
                         x="player", y=y_metric, color=y_metric, 
                         color_continuous_scale="Viridis",
                         title=f"{sort_order} Players for {y_metric}")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- VIEW: DISTRIBUTIONS ---
    elif st.session_state.view == 'Dist':
        st.subheader("Statistical Distributions")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        dist_metric = st.selectbox("Select Metric to Analyze", numeric_cols, index=numeric_cols.index('age') if 'age' in numeric_cols else 0)
        
        fig_dist = px.histogram(df, x=dist_metric, marginal="box", 
                                title=f"Spread of {dist_metric}",
                                color_discrete_sequence=['#ff4b4b'],
                                nbins=30)
        st.plotly_chart(fig_dist, use_container_width=True)