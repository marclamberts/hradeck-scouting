import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIG ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"

st.set_page_config(page_title="Football Scout Pro", layout="wide")

# --- AUTH ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated: return True
    
    st.title("🔐 Pro Scout Portal")
    with st.form("login"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    return False

# --- DATA HELPERS ---
def get_tables():
    with sqlite3.connect(DB_FILE) as conn:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]

def load_data(table_name):
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql(f'SELECT * FROM "{table_name}"', conn)

# --- MAIN APP ---
if check_password():
    # 1. SIDEBAR SETUP
    st.sidebar.title("📁 Navigation")
    
    # Table Selector
    all_tables = get_tables()
    selected_table = st.sidebar.selectbox("Select Database Table", all_tables)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("View Mode")
    
    # Navigation Buttons (using session state)
    if 'view' not in st.session_state:
        st.session_state.view = 'Table'

    if st.sidebar.button("📄 Table View", use_container_width=True):
        st.session_state.view = 'Table'
    if st.sidebar.button("📊 Bar Graphs", use_container_width=True):
        st.session_state.view = 'Bar'
    if st.sidebar.button("📈 Distributions", use_container_width=True):
        st.session_state.view = 'Dist'

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # 2. LOAD DATA
    df = load_data(selected_table)

    # 3. TOP FILTERS (Above main page, next to each other)
    st.title(f"Viewing: {selected_table}")
    
    f1, f2, f3 = st.columns(3)
    
    with f1:
        teams = ["All"] + sorted(df["team"].unique().tolist())
        selected_team = st.selectbox("Filter by Team", teams)
    
    with f2:
        # Handling positions which might be comma-separated like "LCB, RCB"
        unique_pos = set()
        for p in df["position"].dropna().unique():
            for sub_p in p.split(','):
                unique_pos.add(sub_p.strip())
        positions = ["All"] + sorted(list(unique_pos))
        selected_pos = st.selectbox("Filter by Position", positions)
        
    with f3:
        search_query = st.text_input("🔍 Search Player Name", "")

    # Apply Filters to Dataframe
    filtered_df = df.copy()
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["team"] == selected_team]
    if selected_pos != "All":
        filtered_df = filtered_df[filtered_df["position"].str.contains(selected_pos, na=False)]
    if search_query:
        filtered_df = filtered_df[filtered_df["player"].str.contains(search_query, case=False, na=False)]

    st.markdown("---")

    # 4. MAIN CONTENT AREA (Based on Sidebar Buttons)
    if st.session_state.view == 'Table':
        st.subheader("Data Table")
        st.dataframe(filtered_df, use_container_width=True, height=600)

    elif st.session_state.view == 'Bar':
        st.subheader("Bar Graph Analysis")
        col_x, col_y = st.columns(2)
        
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        
        with col_x:
            x_axis = st.selectbox("X Axis (Category)", ["player", "team", "position"])
        with col_y:
            y_axis = st.selectbox("Y Axis (Metric)", numeric_cols, index=numeric_cols.index('market_value') if 'market_value' in numeric_cols else 0)
            
        fig = px.bar(filtered_df.sort_values(y_axis, ascending=False).head(20), 
                     x=x_axis, y=y_axis, color=y_axis, 
                     title=f"Top 20 Players by {y_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.view == 'Dist':
        st.subheader("Value Distributions")
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        dist_col = st.selectbox("Select Metric to view Distribution", numeric_cols, index=numeric_cols.index('age') if 'age' in numeric_cols else 0)
        
        fig = px.histogram(filtered_df, x=dist_col, nbins=30, marginal="box", 
                           title=f"Distribution of {dist_col}",
                           color_discrete_sequence=['indianred'])
        st.plotly_chart(fig, use_container_width=True)