import streamlit as st
import sqlite3
import pandas as pd

# --- CONFIGURATION ---
DB_FILE = "webapp_database.db"
VALID_USERNAME = "kralove"
VALID_PASSWORD = "CZ2526"

st.set_page_config(page_title="SQL Database Portal", layout="wide")

# --- AUTHENTICATION FUNCTION ---
def check_password():
    """Returns True if the user had the correct credentials."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Show login form
    st.title("🔐 Database Login")
    with st.form("login_form"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if user == VALID_USERNAME and pwd == VALID_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")
    return False

# --- DATABASE FUNCTIONS ---
def get_table_names():
    with sqlite3.connect(DB_FILE) as conn:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        return [row[0] for row in conn.execute(query).fetchall()]

def load_data(table_name):
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

# --- MAIN APP ---
if check_password():
    # Sidebar Logout
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    st.title("📊 Data Explorer")
    
    # Get all tables from your db (e.g., 'Champ')
    try:
        tables = get_table_names()
        
        if not tables:
            st.warning("No tables found in the database.")
        else:
            selected_table = st.sidebar.selectbox("Select Table", tables)
            
            # Display Data
            st.subheader(f"Table: {selected_table}")
            df = load_data(selected_table)
            
            # Simple filters
            if not df.empty:
                search = st.text_input("Filter data by any value...")
                if search:
                    mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                    df = df[mask]
                
                st.dataframe(df, use_container_width=True)
                
                # Show summary stats for numeric columns
                if st.checkbox("Show Summary Statistics"):
                    st.write(df.describe())
            else:
                st.info("This table is empty.")
                
    except Exception as e:
        st.error(f"Error connecting to database: {e}")