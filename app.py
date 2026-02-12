"""
IMPECT Stats Table - Clean & Simple
====================================
FBref/Baseball Savant style table for Keuken Kampioen Divisie
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="IMPECT Stats", page_icon="⚽", layout="wide")

# File path - change to your actual location
DATA_FILE = "Keuken Kampioen Divisie.xlsx"

# Baseball Savant color scale
def percentile_color(pct):
    if pd.isna(pct): return "#FFFFFF"
    if pct >= 90: return "#08519c"  # Dark blue
    if pct >= 75: return "#3182bd"  # Blue
    if pct >= 60: return "#6baed6"  # Light blue
    if pct >= 40: return "#9ecae1"  # Very light blue
    if pct >= 25: return "#c6dbef"  # Pale blue
    return "#eff3ff"

# Metrics where lower is better
INVERTED = ["foul", "lost", "unsuccessful", "failed", "off target", "red", "yellow"]

def is_inverted(col):
    return any(inv in col.lower() for inv in INVERTED)

# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_data(file_path):
    standard = pd.read_excel(file_path, sheet_name="Standard")
    xg = pd.read_excel(file_path, sheet_name="xG")
    
    base = ['iterationId', 'squadId', 'squadName', 'playerId', 'positions',
            'commonname', 'firstname', 'lastname', 'birthdate', 'birthplace',
            'leg', 'countryIds', 'gender', 'season', 'dataVersion',
            'lastChangeTimestamp', 'competition_name', 'competition_type', 'competition_gender']
    
    xg_kpis = [c for c in xg.columns if c not in base]
    merged = standard.merge(xg[['playerId'] + xg_kpis], on='playerId', how='left')
    
    cn = merged.get("commonname", "").fillna("").astype(str).str.strip()
    fallback = (merged.get("firstname", "").fillna("").astype(str).str.strip() + " " +
                merged.get("lastname", "").fillna("").astype(str).str.strip()).str.strip()
    merged["displayName"] = np.where(cn == "", fallback, cn)
    
    return merged

def get_kpis(df):
    base = {'iterationId', 'squadId', 'squadName', 'playerId', 'positions',
            'commonname', 'firstname', 'lastname', 'birthdate', 'birthplace',
            'leg', 'countryIds', 'gender', 'season', 'dataVersion',
            'lastChangeTimestamp', 'competition_name', 'competition_type',
            'competition_gender', 'displayName'}
    return [c for c in df.columns if c not in base and pd.api.types.is_numeric_dtype(df[c])]

def calc_percentiles(df, kpis):
    for col in kpis:
        vals = pd.to_numeric(df[col], errors="coerce")
        pct = vals.rank(pct=True, method='average') * 100
        df[f"{col}_pct"] = 100 - pct if is_inverted(col) else pct
    return df

def color_map(df, col):
    pct_col = f"{col}_pct"
    if pct_col not in df.columns:
        return ['background-color: #FFFFFF'] * len(df)
    
    colors = []
    for pct in df[pct_col]:
        color = percentile_color(100 - pct if is_inverted(col) and not pd.isna(pct) else pct)
        colors.append(f'background-color: {color}')
    return colors

# -------------------------
# UI
# -------------------------
st.title("⚽ Keuken Kampioen Divisie Stats")

# Load data
try:
    df = load_data(DATA_FILE)
    kpis = get_kpis(df)
    df = calc_percentiles(df, kpis)
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# Filters in columns at top
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    name_filter = st.text_input("🔍 Player name", placeholder="Search...")

with col2:
    squads = ["All"] + sorted(df["squadName"].dropna().unique().tolist())
    squad = st.selectbox("🏟️ Squad", squads)

with col3:
    positions = ["All", "Defenders", "Midfielders", "Forwards"]
    pos_group = st.selectbox("⚽ Position", positions)

with col4:
    show_pct = st.checkbox("Show %ile", value=False)

# Apply filters
df_filtered = df.copy()

if name_filter:
    df_filtered = df_filtered[df_filtered["displayName"].str.contains(name_filter, case=False, na=False)]

if squad != "All":
    df_filtered = df_filtered[df_filtered["squadName"] == squad]

if pos_group != "All":
    pos_map = {
        "Defenders": ["DEFENDER", "BACK"],
        "Midfielders": ["MIDFIELD"],
        "Forwards": ["FORWARD", "WINGER"]
    }
    tokens = pos_map[pos_group]
    mask = pd.Series(False, index=df_filtered.index)
    for t in tokens:
        mask |= df_filtered["positions"].str.contains(t, case=False, na=False)
    df_filtered = df_filtered[mask]

st.caption(f"**{len(df_filtered)}** players")

# Stat selection
st.divider()

# Quick categories
categories = {
    "Goals & Assists": ["Goals", "Assists", "Pre Assist", "Shot-Creating Actions", "Shot xG from Passes"],
    "Shooting": ["Total Shots", "Total Shots On Target", "Shot-based xG", "Post-Shot xG"],
    "Passing": ["Successful Passes", "Unsuccessful Passes", "Progressive passes", "Pass Accuracy"],
    "Duels": ["Won Ground Duels", "Lost Ground Duels", "Won Aerial Duels", "Lost Aerial Duels"],
    "xG Suite": ["Shot-based xG", "Post-Shot xG", "Expected Goal Assists", "Expected Shot Assists", "Packing non-shot-based xG"],
}

selected_cat = st.selectbox("📊 Stat Category", list(categories.keys()))
keywords = categories[selected_cat]

# Find matching KPIs
matching = [k for k in kpis if any(kw in k for kw in keywords)][:15]

selected_stats = st.multiselect(
    "Select stats",
    options=matching,
    default=matching[:8] if len(matching) >= 8 else matching,
    help="Statistics to display"
)

if not selected_stats:
    st.warning("Select at least one stat")
    st.stop()

# Build table
cols = ["displayName", "squadName", "positions"] + selected_stats

if show_pct:
    cols_with_pct = ["displayName", "squadName", "positions"]
    for stat in selected_stats:
        cols_with_pct.append(stat)
        if f"{stat}_pct" in df_filtered.columns:
            cols_with_pct.append(f"{stat}_pct")
    cols = cols_with_pct

cols = [c for c in cols if c in df_filtered.columns]
df_display = df_filtered[cols].copy()

# Sort by first stat
if selected_stats:
    df_display = df_display.sort_values(selected_stats[0], ascending=False)

# Style table
styled = df_display.style

for stat in selected_stats:
    if stat in df_display.columns:
        styled = styled.apply(lambda _: color_map(df_display, stat), subset=[stat], axis=0)

# Format numbers
fmt = {}
for stat in selected_stats:
    if stat in df_display.columns:
        max_val = df_display[stat].max()
        if pd.notna(max_val):
            if max_val < 1:
                fmt[stat] = '{:.3f}'
            elif max_val < 10:
                fmt[stat] = '{:.2f}'
            else:
                fmt[stat] = '{:.1f}'

if fmt:
    styled = styled.format(fmt, na_rep="-")

# Display
st.dataframe(styled, use_container_width=True, height=650)

# Export
col1, col2 = st.columns(2)

with col1:
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 CSV",
        csv,
        f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True
    )

with col2:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_display.to_excel(writer, index=False)
    st.download_button(
        "📥 Excel",
        buffer.getvalue(),
        f"kkd_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )