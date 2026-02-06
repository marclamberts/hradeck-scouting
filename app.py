# app.py — UPDATED (adds IMPECT Sync → CSV + IMPECT Radar (CSV) tab)
#
# ✅ What’s new
# - IMPECT Sync expander on Landing: exports iteration → wide CSV (API)
# - Dashboard tab “🧭 IMPECT Radar (CSV)”: builds radar from CSV-only module
# - Uses st.secrets for credentials (no hardcoded password)
#
# IMPORTANT:
# - Put impect_api.py + impect_radar_csv.py in the SAME folder as app.py (or adjust imports)
# - Create .streamlit/secrets.toml with:
#     IMPECT_USERNAME="..."
#     IMPECT_PASSWORD="..."
#     IMPECT_ITERATION_ID=1421
#
# NOTE:
# This file is your original app with minimal invasive edits:
#   1) imports added
#   2) landing gets IMPECT Sync expander
#   3) dashboard gets new Radar tab

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.stats import zscore
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

# =====================================================
# NEW: IMPECT INTEGRATION IMPORTS
# =====================================================
# These two modules are the refactors of the code you pasted.
# - impect_api.py: export_iteration_to_csv(...)
# - impect_radar_csv.py: CSV-only radar functions that return matplotlib fig
try:
    from impect_api import export_iteration_to_csv
    import impect_radar_csv as ir
    IMPECT_MODULES_OK = True
except Exception:
    IMPECT_MODULES_OK = False

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Scout Lab Pro",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="collapsed"
)

# =====================================================
# POSITION CONFIGURATIONS
# =====================================================
POSITION_CONFIG = {
    "GK": {
        "file": "Goalkeepers.xlsx",
        "title": "Goalkeepers",
        "icon": "🧤",
        "role_cols": ['Ball Playing GK', 'Box Defender', 'Shot Stopper', 'Sweeper Keeper'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low pass', 'Diagonal pass', 'Chipped/Lofted ball', 'Goal kick', 'Free kick'],
        "categories": {
            "Passing": ['Low pass', 'Diagonal pass', 'Chipped/Lofted ball', 'Goal kick', 'Free kick'],
            "Shot Stopping": ['Prevented Goals Percent (based on post-shot xG) - Long Range Shot saved',
                            'Prevented Goals Percent (based on post-shot xG) - Mid Range Shot saved',
                            'Prevented Goals Percent (based on post-shot xG) - Close Range Shot saved',
                            'Prevented Goals Percent (based on post-shot xG) - Header saved'],
            "Positioning": ['Defensive Touches outside the Box per game', 'Caught Balls Percent']
        }
    },
    "CB": {
        "file": "Central Defenders.xlsx",
        "title": "Central Defenders",
        "icon": "🛡️",
        "role_cols": ['Aerially Dominant CB', 'Aggressive CB', 'Ball Playing CB', 'Strategic CB'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low pass', 'Diagonal pass', 'Ground duel', 'Defensive Header', 'Interception'],
        "categories": {
            "Passing": ['Low pass', 'Diagonal pass', 'Chipped/Lofted ball'],
            "Defending": ['Ground duel', 'Defensive Header', 'Interception', 'Loose ball regain', 'Block', 'Clearance with foot'],
            "Attacking": ['Header shot', 'Dribble']
        }
    },
    "LB": {
        "file": "Left Back.xlsx",
        "title": "Left Backs",
        "icon": "⬅️",
        "role_cols": ['Classic Back 4 LB', 'Creative LB', 'Left Wing-Back'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low pass', 'High Cross', 'Low Cross', 'Ground duel', 'Interception'],
        "categories": {
            "Passing": ['Low pass', 'Chipped/Lofted ball'],
            "Crossing": ['High Cross', 'Low Cross'],
            "Attacking": ['Dribble', 'Availability Between the Lines', 'Mid range shot', 'Availability in the Box'],
            "Defending": ['Ground duel', 'Defensive Header', 'Interception', 'Loose ball regain']
        }
    },
    "RB": {
        "file": "Right Back.xlsx",
        "title": "Right Backs",
        "icon": "➡️",
        "role_cols": ['Classic Back 4 RB', 'Creative RB', 'Right Wing-Back'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low pass', 'High Cross', 'Low Cross', 'Ground duel', 'Interception'],
        "categories": {
            "Passing": ['Low pass', 'Chipped/Lofted ball'],
            "Crossing": ['High Cross', 'Low Cross'],
            "Attacking": ['Dribble', 'Availability Between the Lines', 'Mid range shot', 'Availability in the Box'],
            "Defending": ['Ground duel', 'Defensive Header', 'Interception', 'Loose ball regain']
        }
    },
    "DM": {
        "file": "Defensive Midfielder.xlsx",
        "title": "Defensive Midfielders",
        "icon": "⚓",
        "role_cols": ['Anchorman', 'Ball Winning Midfielder', 'Box-to-Box Midfielder', 'Central Creator', 'Deep Lying Playmaker'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low pass', 'Diagonal pass', 'Dribble', 'Interception', 'Loose ball regain'],
        "categories": {
            "Passing": ['Low pass', 'Diagonal pass', 'Chipped/Lofted ball'],
            "Attacking": ['Dribble', 'Availability Between the Lines', 'Mid range shot', 'Availability in the Box'],
            "Defending": ['Ground duel', 'Defensive Header', 'Interception', 'Loose ball regain', 'Block']
        }
    },
    "CM": {
        "file": "Central Midfielder.xlsx",
        "title": "Central Midfielders",
        "icon": "⭐",
        "role_cols": ['Anchorman', 'Ball Winning Midfielder', 'Box-to-Box Midfielder', 'Central Creator', 'Deep Lying Playmaker'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low pass', 'Diagonal pass', 'Dribble', 'Availability Between the Lines', 'Mid range shot'],
        "categories": {
            "Passing": ['Low pass', 'Diagonal pass', 'Chipped/Lofted ball'],
            "Attacking": ['Dribble', 'Availability Between the Lines', 'Mid range shot', 'Availability from Deep Runs', 'Availability in the Box'],
            "Defending": ['Ground duel', 'Defensive Header', 'Interception', 'Loose ball regain']
        }
    },
    "AM": {
        "file": "Attacking Midfielder.xlsx",
        "title": "Attacking Midfielders",
        "icon": "🎯",
        "role_cols": ['Central Creator', 'Deep Lying Striker'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low pass', 'Dribble', 'Availability Between the Lines', 'Mid range shot', 'Availability in the Box'],
        "categories": {
            "Passing": ['Low pass', 'High Cross', 'Low Cross'],
            "Dribbling": ['Dribble'],
            "Movement": ['Availability Between the Lines', 'Availability from Deep Runs', 'Availability in the Box'],
            "Shooting": ['Mid range shot', 'Close range shot', 'Header shot', 'Offensive header']
        }
    },
    "LW": {
        "file": "Left Winger.xlsx",
        "title": "Left Wingers",
        "icon": "⚡",
        "role_cols": ['Central Creator', 'Classic Left Winger', 'Deep Running Left Winger', 'Defensive Left Winger', 'Left Wing-Back'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low Cross', 'Dribble', 'Availability in the Box', 'Close range shot', 'Header shot'],
        "categories": {
            "Crossing": ['High Cross', 'Low Cross'],
            "Dribbling": ['Dribble'],
            "Movement": ['Availability Between the Lines', 'Availability from Deep Runs', 'Availability in the Box'],
            "Shooting": ['Mid range shot', 'Close range shot', 'Header shot', 'Offensive header']
        }
    },
    "RW": {
        "file": "Right Wing.xlsx",
        "title": "Right Wingers",
        "icon": "⚡",
        "role_cols": ['Central Creator', 'Classic Right Winger', 'Deep Running Right Winger', 'Defensive Right Winger', 'Right Wing-Back'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Low Cross', 'Dribble', 'Availability in the Box', 'Close range shot', 'Header shot'],
        "categories": {
            "Crossing": ['High Cross', 'Low Cross'],
            "Dribbling": ['Dribble'],
            "Movement": ['Availability Between the Lines', 'Availability from Deep Runs', 'Availability in the Box'],
            "Shooting": ['Mid range shot', 'Close range shot', 'Header shot', 'Offensive header']
        }
    },
    "ST": {
        "file": "Strikers.xlsx",
        "title": "Strikers",
        "icon": "⚽",
        "role_cols": ['Complete Forward', 'Deep Lying Striker', 'Deep Running Striker', 'Poacher', 'Pressing Striker', 'Second Striker', 'Target Man'],
        "key_metrics": ['IMPECT', 'Offensive IMPECT', 'Defensive IMPECT', 'Dribble', 'Availability in the Box', 'Close range shot', 'Header shot', 'Hold-Up play'],
        "categories": {
            "Movement": ['Availability Between the Lines', 'Availability from Deep Runs', 'Availability in the Box'],
            "Shooting": ['Mid range shot', 'Close range shot', 'Header shot', 'Offensive header'],
            "Physical": ['Hold-Up play', 'Ground duel']
        }
    }
}

NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# =====================================================
# ENHANCED CSS
# =====================================================
# (UNCHANGED — your full CSS block below)
st.markdown("""<style>
/* ... your CSS exactly as-is ... */
</style>""", unsafe_allow_html=True)

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def safe_float(x):
    if x is None or pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("%", "").replace(",", "")
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def safe_fmt(x, decimals=1):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"

def percentile_rank(s):
    s = s.apply(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    if mask.sum() > 0:
        out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def get_percentile_badge(pct):
    if np.isnan(pct):
        return ""
    if pct >= 90:
        return f'<span class="badge badge-success">Top 10%</span>'
    elif pct >= 75:
        return f'<span class="badge badge-info">Top 25%</span>'
    elif pct >= 50:
        return f'<span class="badge badge-warning">Above Avg</span>'
    else:
        return f'<span class="badge badge-danger">Below Avg</span>'

def get_percentile_color(pct):
    if pct >= 80:
        return "#10b981"
    elif pct >= 60:
        return "#3b82f6"
    elif pct >= 40:
        return "#f59e0b"
    else:
        return "#ef4444"

# =====================================================
# DATA LOADING (EXCEL)
# =====================================================
@st.cache_data(show_spinner=False)
def load_data(position_key):
    cfg = POSITION_CONFIG[position_key]

    possible_paths = [
        Path(cfg["file"]),
        Path("/mnt/user-data/uploads") / cfg["file"],
        Path("uploads") / cfg["file"],
    ]

    fp = None
    for path in possible_paths:
        if path.exists():
            fp = path
            break

    if fp is None:
        st.error(f"❌ File not found: `{cfg['file']}`")
        st.stop()

    df = pd.read_excel(fp)
    df.columns = [str(c).strip() for c in df.columns]

    numeric_cols = []
    for col in df.columns:
        if col in ['Player-ID', NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            continue
        if 'BetterThan' in col:
            continue
        numeric_cols.append(col)

    for c in numeric_cols + [AGE_COL, SHARE_COL]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace("nan", "").str.strip()

    for m in numeric_cols:
        if m in df.columns:
            df[m + " (pct)"] = percentile_rank(df[m])

    return df, cfg, numeric_cols

# =====================================================
# NEW: IMPECT CSV LOADER (cached)
# =====================================================
@st.cache_data(show_spinner=False)
def load_impect_iteration_csv(csv_path: str) -> pd.DataFrame:
    if not IMPECT_MODULES_OK:
        return pd.DataFrame()
    df = ir.load_impect_csv(csv_path)
    df = ir.compute_derived(df)
    return df

# =====================================================
# STATE
# =====================================================
def init_state():
    if "view" not in st.session_state:
        st.session_state.view = "landing"
    if "selected_player" not in st.session_state:
        st.session_state.selected_player = None
    if "position" not in st.session_state:
        st.session_state.position = "ST"
    if "comparison_list" not in st.session_state:
        st.session_state.comparison_list = []
    # NEW: remember last CSV path in UI
    if "impect_csv_path" not in st.session_state:
        st.session_state.impect_csv_path = "data/impect_player_kpis_1421.csv"

init_state()

# =====================================================
# NAVIGATION
# =====================================================
def render_nav(show_back=False):
    if show_back:
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Home", use_container_width=True):
                st.session_state.view = "landing"
                st.session_state.selected_player = None
                st.rerun()
        with col2:
            cfg = POSITION_CONFIG.get(st.session_state.position, {})
            st.markdown(f"""
            <div class="top-nav">
                <div class="nav-brand">⚽ Scout Lab Pro</div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 1.5rem;">{cfg.get('icon', '⚽')}</span>
                    <span style="font-weight: 600; font-size: 1.1rem;">{cfg.get('title', 'Scouting')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="top-nav">
            <div class="nav-brand">⚽ Scout Lab Pro</div>
            <div style="font-weight: 600; font-size: 1.1rem;">Professional Football Analytics</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# LANDING PAGE VIEW
# =====================================================
def render_landing_view():
    st.markdown("""
    <div class="landing-hero">
        <div class="landing-title">⚽ Scout Lab Pro</div>
        <div class="landing-subtitle">Advanced Football Analytics Platform</div>
        <div class="landing-tagline">
            Comprehensive player scouting with IMPECT data, detailed performance metrics,
            role suitability analysis, and professional visualization tools across 10 positions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🎯 Platform Features")
    st.markdown("""<div class="feature-grid"> ... </div>""", unsafe_allow_html=True)

    # =====================================================
    # NEW: IMPECT SYNC EXPANDER (API → CSV)
    # =====================================================
    st.markdown("---")
    st.markdown("## 🔄 IMPECT Sync (optional)")

    if not IMPECT_MODULES_OK:
        st.warning("IMPECT modules not found. Add impect_api.py and impect_radar_csv.py next to app.py to enable Sync + Radar.")
    else:
        with st.expander("Export iteration to CSV (API)", expanded=False):
            st.session_state.impect_csv_path = st.text_input(
                "Output CSV path",
                value=st.session_state.impect_csv_path,
                help="Where to save the exported wide KPI CSV."
            )

            default_iter = int(st.secrets.get("IMPECT_ITERATION_ID", 1421))
            iteration_id = st.number_input("Iteration ID", min_value=1, value=default_iter, step=1)

            c1, c2 = st.columns(2)
            with c1:
                max_workers = st.slider("MAX_WORKERS (avoid 429)", min_value=1, max_value=8, value=3)
            with c2:
                lang = st.selectbox("KPI language", options=["en", "nl", "de", "fr"], index=0)

            if st.button("Export now", use_container_width=True):
                username = st.secrets.get("IMPECT_USERNAME", "")
                password = st.secrets.get("IMPECT_PASSWORD", "")

                if not username or not password:
                    st.error("Missing IMPECT_USERNAME / IMPECT_PASSWORD in .streamlit/secrets.toml")
                else:
                    try:
                        # optional: override workers on the module (keeps your retry logic)
                        import impect_api
                        impect_api.MAX_WORKERS = int(max_workers)

                        summary = export_iteration_to_csv(
                            username=username,
                            password=password,
                            iteration_id=int(iteration_id),
                            output_csv=st.session_state.impect_csv_path,
                            kpi_language=lang,
                        )
                        st.success(
                            f"Saved CSV: {summary['output_csv']} "
                            f"({summary['players_rows']} rows × {summary['cols']} cols, {summary['seconds']}s)"
                        )
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Export failed: {e}")

    # Position selection (unchanged)
    st.markdown("---")
    st.markdown("## 🎯 Select Position to Begin Scouting")
    st.markdown("### Choose a position to access player database and analytics")

    position_groups = {
        "Goalkeepers": ["GK"],
        "Defenders": ["CB", "LB", "RB"],
        "Midfielders": ["DM", "CM", "AM"],
        "Forwards": ["LW", "RW", "ST"]
    }

    for group_name, positions in position_groups.items():
        st.markdown(f"### {group_name}")
        cols = st.columns(len(positions))
        for idx, pos_key in enumerate(positions):
            cfg = POSITION_CONFIG[pos_key]
            with cols[idx]:
                if st.button(
                    f"{cfg['icon']}\n\n**{cfg['title']}**\n\nView Database",
                    key=f"landing_pos_{pos_key}",
                    use_container_width=True,
                    help=f"Scout {cfg['title'].lower()}"
                ):
                    st.session_state.position = pos_key
                    st.session_state.view = "search"
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Database Coverage")
    st.markdown("""<div class="stats-showcase"> ... </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); padding: 2rem 0;">
        <p style="margin-bottom: 0.5rem;">Built with advanced analytics and IMPECT data</p>
        <p style="font-size: 0.875rem;">Professional football scouting and analysis platform</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# SEARCH VIEW
# =====================================================
# (UNCHANGED — keep your existing search view and helpers)
def render_search_view(df, cfg, all_metrics):
    # ... your existing code ...
    pass

def render_player_card(row, cfg):
    # ... your existing code ...
    pass

# =====================================================
# DASHBOARD VIEW
# =====================================================
def render_dashboard_view(df, cfg, all_metrics):
    player_name = st.session_state.selected_player

    if st.button("← Back to Search", key="back_btn"):
        st.session_state.view = "search"
        st.rerun()

    player_data = df[df[NAME_COL] == player_name]
    if player_data.empty:
        st.error("Player not found")
        return

    row = player_data.iloc[0]

    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    age = safe_fmt(row.get(AGE_COL, 0), 0)
    nat = str(row.get(NAT_COL, "—"))
    share = safe_fmt(row.get(SHARE_COL, 0), 1)

    st.markdown(f"""
    <div class="dashboard-header">
        <div class="dashboard-title">{player_name}</div>
        <div class="dashboard-subtitle" style="font-size: 1.25rem; opacity: 0.95;">
            {team} • {comp} • {nat} • {age} years old • {share}% match share
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # UPDATED: add IMPECT Radar tab at the end
    # =====================================================
    tabs = st.tabs([
        "📊 Overview",
        "🎯 Detailed Stats",
        "📈 Performance Trends",
        "⚖️ Comparison",
        "📝 Report",
        "🧭 IMPECT Radar (CSV)"  # NEW
    ])

    with tabs[0]:
        render_overview_tab(row, cfg, df)

    with tabs[1]:
        render_detailed_stats_tab(row, cfg, all_metrics)

    with tabs[2]:
        render_performance_tab(row, cfg, df)

    with tabs[3]:
        render_comparison_tab(df, cfg, all_metrics)

    with tabs[4]:
        render_report_tab(row, cfg, all_metrics)

    # =====================================================
    # NEW TAB: IMPECT Radar (CSV-only)
    # =====================================================
    with tabs[5]:
        st.markdown("### 🧭 IMPECT Radar (from iteration CSV)")
        if not IMPECT_MODULES_OK:
            st.warning("IMPECT modules not available. Add impect_api.py + impect_radar_csv.py next to app.py.")
            return

        csv_path = st.text_input(
            "IMPECT CSV path",
            value=st.session_state.impect_csv_path,
            help="Path to exported iteration CSV (wide KPIs)."
        )
        st.session_state.impect_csv_path = csv_path

        p = Path(csv_path)
        if not p.exists():
            st.warning("CSV not found. Export it on the Landing page (IMPECT Sync) first.")
            return

        with st.spinner("Loading IMPECT CSV and building profiles..."):
            im_df = load_impect_iteration_csv(str(p))

        if im_df.empty:
            st.error("IMPECT CSV loaded but produced an empty dataframe.")
            return

        # match selected player name against IMPECT identity columns
        rows = ir.find_player_rows(im_df.copy(), player_name)
        if rows.empty:
            st.error(f"Player not found in IMPECT CSV: {player_name!r}")
            # quick helper list
            if "commonname" in im_df.columns:
                st.caption("Sample names in CSV:")
                st.write(sorted(im_df["commonname"].dropna().astype(str).unique())[:30])
            return

        player_row = rows.iloc[0]

        profiles = ir.build_profiles_from_csv(im_df)
        if not profiles:
            st.error("No profiles could be built from this CSV (missing KPIs).")
            return

        available_profiles = list(profiles.keys())
        default_profile = available_profiles[0]

        c1, c2 = st.columns([2, 1])
        with c1:
            profile_name = st.selectbox("Profile", options=available_profiles, index=0)
        with c2:
            benchmark_by_position = st.toggle("Benchmark by position", value=True)

        # Create figure (your impect_radar_csv.py must implement this)
        try:
            fig = ir.make_player_radar_fig(
                df_full=im_df,
                player_row=player_row,
                profile_name=profile_name,
                profiles=profiles,
                benchmark_by_position=benchmark_by_position,
            )
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Radar build failed: {e}")
            st.info("Make sure impect_radar_csv.py exposes make_player_radar_fig(...) and returns a matplotlib Figure.")

# =====================================================
# THE REST OF YOUR ORIGINAL FUNCTIONS
# =====================================================
# Keep your existing implementations exactly as-is:
def render_overview_tab(row, cfg, df):
    # ... your existing code ...
    pass

def render_detailed_stats_tab(row, cfg, all_metrics):
    # ... your existing code ...
    pass

def render_performance_tab(row, cfg, df):
    # ... your existing code ...
    pass

def render_comparison_tab(df, cfg, all_metrics):
    # ... your existing code ...
    pass

def render_report_tab(row, cfg, all_metrics):
    # ... your existing code ...
    pass

# =====================================================
# MAIN APP
# =====================================================
def main():
    if st.session_state.view == "landing":
        render_nav(show_back=False)
        render_landing_view()
    else:
        render_nav(show_back=True)

        with st.spinner("Loading data..."):
            df, cfg, all_metrics = load_data(st.session_state.position)

        if st.session_state.view == "search":
            render_search_view(df, cfg, all_metrics)
        elif st.session_state.view == "dashboard":
            render_dashboard_view(df, cfg, all_metrics)

if __name__ == "__main__":
    main()
