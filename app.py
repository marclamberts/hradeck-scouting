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

# --- CUSTOM CSS (Fixed parameter here) ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004b91; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #eee; }
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. AUTHENTICATION SYSTEM ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Professional Login Page with Logo
    _, col2, _ = st.columns([1, 1.2, 1])
    with col2:
        st.write("#")
        st.image(LOGO_URL, width=80)
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
                    st.error("Invalid credentials.")
    return False

# --- 2. DATA UTILITIES ---
def get_all_tables():
    with sqlite3.connect(DB_FILE) as conn:
        return [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]

def load_clean_data(table_name):
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        df = df.fillna('')
        # Clean numeric columns to prevent plotting errors
        for col in df.columns:
            if col in ['market_value', 'age', 'goals', 'xg', 'matches_played', 'assists']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

# --- 3. MAIN APPLICATION ---
if check_password():
    # Sidebar
    st.sidebar.image(LOGO_URL, width=50)
    st.sidebar.title("Hradeck Scout")
    
    tables = get_all_tables()
    selected_table = st.sidebar.selectbox("📂 Database Table", tables)
    
    st.sidebar.markdown("---")
    
    # Navigation Buttons
    if 'view' not in st.session_state:
        st.session_state.view = 'Dashboard'

    if st.sidebar.button("🏠 Executive Dashboard", use_container_width=True):
        st.session_state.view = 'Dashboard'
    if st.sidebar.button("📄 Raw Data Table", use_container_width=True):
        st.session_state.view = 'Table'
    if st.sidebar.button("📊 Performance Charts", use_container_width=True):
        st.session_state.view = 'Charts'
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    # Load Data
    df_raw = load_clean_data(selected_table)

    # Top Filter Area
    st.title(f"⚽ {selected_table} Intel")
    
    # Shared Search Bar for all views
    search = st.text_input("🔍 Quick Player Search", placeholder="Type player name...", key="main_search")
    
    df = df_raw.copy()
    if search:
        df = df[df["player"].astype(str).str.contains(search, case=False, na=False)]

    st.divider()

    # --- VIEW: DASHBOARD ---
    if st.session_state.view == 'Dashboard':
        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Players", len(df))
        
        if 'market_value' in df.columns:
            m2.metric("Avg Market Value", f"€{int(df['market_value'].mean()):,}")
        
        if 'goals' in df.columns:
            m3.metric("League Goals", int(df['goals'].sum()))
            
        if 'age' in df.columns:
            m4.metric("Average Age", round(df['age'].mean(), 1))

        st.markdown("### 🏆 Market Leaders")
        if 'market_value' in df.columns:
            top_5 = df.sort_values("market_value", ascending=False).head(5)
            cols = st.columns(5)
            for i, (_, player) in enumerate(top_5.iterrows()):
                with cols[i]:
                    st.markdown(f"**{player['player']}**")
                    st.caption(player['team'])
                    st.markdown(f"**€{player['market_value']:,}**")
        
        st.write("##")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Team Value Distribution")
            fig_box = px.box(df, x="team", y="market_value", color="team")
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            st.subheader("Goal Efficiency (xG vs Goals)")
            fig_scat = px.scatter(df, x="xg", y="goals", hover_name="player", color="team", size="market_value" if 'market_value' in df.columns else None)
            st.plotly_chart(fig_scat, use_container_width=True)

    # --- VIEW: TABLE ---
    elif st.session_state.view == 'Table':
        st.subheader("Spreadsheet View")
        st.dataframe(df, use_container_width=True, height=600)

    # --- VIEW: CHARTS ---
    elif st.session_state.view == 'Charts':
        st.subheader("Comparison Analytics")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        ch1, ch2 = st.columns(2)
        with ch1:
            y_axis = st.selectbox("Metric to Compare", numeric_cols, index=numeric_cols.index('market_value') if 'market_value' in numeric_cols else 0)
        with ch2:
            sort_by = st.radio("Show:", ["Top 15", "Bottom 15"], horizontal=True)

        is_asc = True if sort_by == "Bottom 15" else False
        fig_bar = px.bar(df.sort_values(y_axis, ascending=is_asc).head(15), 
                         x="player", y=y_axis, color=y_axis, 
                         color_continuous_scale="Blues")
        st.plotly_chart(fig_bar, use_container_width=True)