import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- CONFIG ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"

st.set_page_config(page_title="Pro Scout Dashboard", layout="wide")

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
def run_query(q):
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql(q, conn)

# --- MAIN APP ---
if check_password():
    # Load Main Data
    df = run_query('SELECT * FROM "Champ"')
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard Overview", "Advanced Scout Search", "Player Profiles"])
    
    if page == "Dashboard Overview":
        st.title("📊 League Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Value by Team")
            # Aggregating market value from the 'market_value' column 
            team_val = df.groupby("team")["market_value"].sum().reset_index()
            fig1 = px.bar(team_val, x="team", y="market_value", color="market_value")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Goals vs Expected Goals (xG)")
            # Using 'goals' and 'xg' columns found in your DB 
            fig2 = px.scatter(df, x="xg", y="goals", hover_name="player", color="team", 
                              size="market_value", title="Performance Efficiency")
            st.plotly_chart(fig2, use_container_width=True)

    elif page == "Advanced Scout Search":
        st.title("🔍 Advanced Scout Search")
        
        # Sidebar Filters
        st.sidebar.subheader("Filters")
        positions = df["position"].unique().tolist()
        selected_pos = st.sidebar.multiselect("Positions", positions, default=positions[:3])
        
        min_val, max_val = int(df["market_value"].min()), int(df["market_value"].max())
        val_range = st.sidebar.slider("Market Value Range", min_val, max_val, (min_val, max_val))
        
        # Filtering the dataframe
        filtered_df = df[
            (df["position"].isin(selected_pos)) & 
            (df["market_value"].between(val_range[0], val_range[1]))
        ]
        
        st.write(f"Showing {len(filtered_df)} players matching criteria")
        st.dataframe(filtered_df, use_container_width=True)

    elif page == "Player Profiles":
        st.title("👤 Player Detailed Profile")
        
        player_list = df["player"].unique()
        selected_player = st.selectbox("Select a player to analyze", player_list)
        
        p_data = df[df["player"] == selected_player].iloc[0]
        
        # Layout for Profile
        c1, c2, c3 = st.columns(3)
        c1.metric("Market Value", f"€{p_data['market_value']:,}")
        c2.metric("Matches Played", p_data["matches_played"])
        c3.metric("Goal Conversion %", f"{p_data['goal_conversion,_%']}%")
        
        st.divider()
        
        # Technical Stats
        st.subheader("Technical Performance")
        cols = st.columns(4)
        cols[0].write(f"**Foot:** {p_data['foot']}")
        cols[1].write(f"**Height:** {p_data['height']}cm")
        cols[2].write(f"**Age:** {p_data['age']}")
        cols[3].write(f"**Contract Ends:** {p_data['contract_expires']}")
        
        # Radar Chart for stats
        stats_cols = ["duels_won,_%", "successful_dribbles,_%", "accurate_crosses,_%"]
        radar_df = pd.DataFrame(dict(r=p_data[stats_cols].values, theta=stats_cols))
        fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True)
        st.plotly_chart(fig_radar)

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()