import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/5329/5329945.png" # Professional soccer/data icon

st.set_page_config(page_title="Hradeck Pro Scout", layout="wide", page_icon="📈")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004b91; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #0e1117; color: white; }
    </style>
    """, unsafe_allow_value=True)

# --- 1. AUTHENTICATION SYSTEM ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Professional Login Page with Logo
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.write("##")
        st.image(LOGO_URL, width=100)
        st.title("Hradeck Scouting")
        st.subheader("Professional Data Access Portal")
        
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In")

            if submit:
                if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
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
            converted = pd.to_numeric(df[col], errors='coerce')
            if not converted.isna().all():
                df[col] = converted.fillna(0)
        return df

# --- 3. MAIN APPLICATION ---
if check_password():
    # Sidebar Navigation
    st.sidebar.image(LOGO_URL, width=60)
    st.sidebar.title("Hradeck Scout")
    
    tables = get_all_tables()
    selected_table = st.sidebar.selectbox("📂 Database Table", tables)
    
    st.sidebar.markdown("---")
    
    # Navigation Buttons
    if 'view' not in st.session_state:
        st.session_state.view = 'Dashboard' # Default to Dashboard

    if st.sidebar.button("🏠 Executive Dashboard"):
        st.session_state.view = 'Dashboard'
    if st.sidebar.button("📄 Raw Data Table"):
        st.session_state.view = 'Table'
    if st.sidebar.button("📊 Performance Charts"):
        st.session_state.view = 'Charts'
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # Load Data
    df_raw = load_clean_data(selected_table)

    # Top Filter Row (Always visible for context)
    f1, f2 = st.columns([2, 1])
    with f1:
        st.title(f"⚽ {selected_table} Intel")
    with f2:
        # Search functionality
        search = st.text_input("🔍 Quick Player Search", placeholder="Name...")

    # Shared Filtering Logic
    df = df_raw.copy()
    if search:
        df = df[df["player"].astype(str).str.contains(search, case=False, na=False)]

    # --- VIEW: DASHBOARD ---
    if st.session_state.view == 'Dashboard':
        # Top Row Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Players", len(df))
        m2.metric("Avg Market Value", f"€{int(df['market_value'].mean()):,}" if 'market_value' in df.columns else "N/A")
        m3.metric("Avg Age", round(df['age'].mean(), 1) if 'age' in df.columns else "N/A")
        m4.metric("Total Goals", int(df['goals'].sum()) if 'goals' in df.columns else "N/A")

        st.markdown("### 🏆 Top Prospects by Market Value")
        if 'market_value' in df.columns:
            top_players = df.sort_values("market_value", ascending=False).head(5)
            cols = st.columns(5)
            for i, (index, player) in enumerate(top_players.iterrows()):
                with cols[i]:
                    st.markdown(f"**{player['player']}**")
                    st.caption(f"{player['team']}")
                    st.markdown(f"**€{player['market_value']:,}**")
                    st.progress(0.8) # Design flourish
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Market Value Distribution")
            fig = px.box(df, y="market_value", x="team", color="team", points="all")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Efficiency: xG vs Goals")
            fig = px.scatter(df, x="xg", y="goals", hover_name="player", size="market_value", color="team")
            st.plotly_chart(fig, use_container_width=True)

    # --- VIEW: TABLE ---
    elif st.session_state.view == 'Table':
        st.subheader("Data Explorer")
        st.dataframe(df, use_container_width=True, height=600)

    # --- VIEW: CHARTS ---
    elif st.session_state.view == 'Charts':
        st.subheader("Advanced Analytics")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            y_axis = st.selectbox("Select Metric", num_cols, index=num_cols.index('market_value') if 'market_value' in num_cols else 0)
        with c2:
            chart_type = st.radio("Chart Type", ["Bar", "Distribution"], horizontal=True)

        if chart_type == "Bar":
            fig = px.bar(df.sort_values(y_axis, ascending=False).head(20), x="player", y=y_axis, color=y_axis)
        else:
            fig = px.histogram(df, x=y_axis, nbins=30, marginal="box")
        
        st.plotly_chart(fig, use_container_width=True)