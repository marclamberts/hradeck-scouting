import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"

st.set_page_config(page_title="Hradeck Scouting Portal", layout="wide", page_icon="⚽")

# --- 1. AUTHENTICATION SYSTEM ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Scouting Database Login")
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)

                if submit:
                    if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
    return False

# --- 2. DATA UTILITIES ---
def get_all_tables():
    """Fetch all table names from the SQLite database."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            return [row[0] for row in conn.execute(query).fetchall()]
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return []

def load_clean_data(table_name):
    """Load table and perform cleaning to prevent Traceback errors."""
    with sqlite3.connect(DB_FILE) as conn:
        # Quote table name for SQL safety (handles '2. Buli')
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        
        # 1. Fill NaN/None with empty string to prevent comparison errors (NoneType vs str)
        df = df.fillna('')
        
        # 2. Automatically detect and convert numeric columns
        # This ensures df.describe() and charts work correctly
        for col in df.columns:
            # Try to convert to numeric, if it's mostly numbers, it stays numeric
            # We use errors='ignore' to leave actual text columns alone
            converted = pd.to_numeric(df[col], errors='coerce')
            if not converted.isna().all():
                df[col] = converted.fillna(0)
        
        return df

# --- 3. MAIN APPLICATION ---
if check_password():
    # --- SIDEBAR NAVIGATION ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3364/3364819.png", width=100)
    st.sidebar.title("Navigation")
    
    tables = get_all_tables()
    selected_table = st.sidebar.selectbox("📂 Select League/Table", tables if tables else ["No Tables Found"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("App View")
    
    # Initialize view state
    if 'view' not in st.session_state:
        st.session_state.view = 'Table'

    # Sidebar Buttons for Switching Views
    if st.sidebar.button("📄 Data Table", use_container_width=True):
        st.session_state.view = 'Table'
    if st.sidebar.button("📊 Bar Graphs", use_container_width=True):
        st.session_state.view = 'Bar'
    if st.sidebar.button("📈 Distribution Plots", use_container_width=True):
        st.session_state.view = 'Dist'

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    # --- LOAD AND FILTER DATA ---
    if tables:
        raw_df = load_clean_data(selected_table)
        
        # --- TOP FILTERS (Horizontal Row) ---
        st.title(f"⚽ {selected_table} Explorer")
        
        f1, f2, f3 = st.columns(3)
        
        with f1:
            team_list = ["All Teams"] + sorted(raw_df["team"].unique().tolist()) if "team" in raw_df.columns else ["N/A"]
            filt_team = st.selectbox("Team Filter", team_list)
            
        with f2:
            if "position" in raw_df.columns:
                # Handle comma separated positions
                pos_set = set()
                for p in raw_df["position"].unique():
                    for sub_p in str(p).split(','):
                        pos_set.add(sub_p.strip())
                pos_list = ["All Positions"] + sorted(list(pos_set))
            else:
                pos_list = ["N/A"]
            filt_pos = st.selectbox("Position Filter", pos_list)
            
        with f3:
            search = st.text_input("🔍 Search Player", placeholder="Type name...")

        # Apply logic
        df = raw_df.copy()
        if filt_team != "All Teams":
            df = df[df["team"] == filt_team]
        if filt_pos != "All Positions" and "position" in df.columns:
            df = df[df["position"].astype(str).str.contains(filt_pos, na=False)]
        if search:
            df = df[df["player"].astype(str).str.contains(search, case=False, na=False)]

        st.divider()

        # --- VIEW CONTENT ---
        if st.session_state.view == 'Table':
            tab_data, tab_stats = st.tabs(["📋 Table View", "🔢 Statistical Summary"])
            
            with tab_data:
                st.dataframe(df, use_container_width=True, height=500)
                st.caption(f"Showing {len(df)} entries")
            
            with tab_stats:
                st.subheader("Numeric Overview")
                st.write(df.describe())

        elif st.session_state.view == 'Bar':
            st.subheader("Rankings & Comparison")
            
            # Select only numeric columns for Y-axis
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 0:
                c1, c2 = st.columns(2)
                with c1:
                    y_metric = st.selectbox("Select Metric (Y-Axis)", numeric_cols, 
                                            index=numeric_cols.index('market_value') if 'market_value' in numeric_cols else 0)
                with c2:
                    sort_order = st.radio("Sort", ["Top 20", "Bottom 20"], horizontal=True)

                plot_df = df.sort_values(y_metric, ascending=(sort_order == "Bottom 20")).head(20)
                
                fig = px.bar(plot_df, x="player", y=y_metric, color=y_metric, 
                             hover_data=["team", "position", "age"],
                             color_continuous_scale="Viridis",
                             title=f"{sort_order} Players by {y_metric}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for graphing.")

        elif st.session_state.view == 'Dist':
            st.subheader("Data Distributions")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                dist_metric = st.selectbox("Select Metric for Distribution", numeric_cols)
                
                fig = px.histogram(df, x=dist_metric, marginal="rug", 
                                   title=f"Spread of {dist_metric} across {selected_table}",
                                   color_discrete_sequence=['#00CC96'],
                                   nbins=25)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for distributions.")
    else:
        st.info("No data found. Please check your webapp_database.db file.")