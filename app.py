import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import hashlib
import datetime as dt
import io
import json
import zipfile

# Excel support with graceful fallback
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# =====================================================
# PAGE CONFIG & CONSTANTS
# =====================================================
st.set_page_config(
    page_title="Scout Lab Pro - Ultimate Edition",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# Color palette
COLORS = {
    "primary": "#00D9FF", "secondary": "#FF6B9D", "success": "#00F5A0",
    "warning": "#FFD93D", "danger": "#FF4757", "dark": "#0A0E27",
    "darker": "#050816", "card": "#151B3B", "border": "#1E2749",
    "text": "#E8EAED", "text_muted": "#8B92B0", "glass": "rgba(21, 27, 59, 0.8)",
}

# Position configurations
POSITION_CONFIG = {
    "GK": {"file": "Goalkeepers.xlsx", "title": "Goalkeepers", "icon": "🧤", "color": "#FF6B9D",
           "role_prefix": ["Ball Playing GK", "Box Defender", "Shot Stopper", "Sweeper Keeper"]},
    "CB": {"file": "Central Defenders.xlsx", "title": "Central Defenders", "icon": "🛡️", "color": "#00F5A0",
           "role_prefix": ["Aerially Dominant CB", "Aggressive CB", "Ball Playing CB", "Strategic CB"]},
    "LB": {"file": "Left Back.xlsx", "title": "Left Backs", "icon": "⬅️", "color": "#FFD93D",
           "role_prefix": ["Attacking FB", "Defensive FB", "Progressive FB", "Inverted FB"]},
    "RB": {"file": "Right Back.xlsx", "title": "Right Backs", "icon": "➡️", "color": "#FF4757",
           "role_prefix": ["Attacking FB", "Defensive FB", "Progressive FB", "Inverted FB"]},
    "DM": {"file": "Defensive Midfielder.xlsx", "title": "Defensive Midfielders", "icon": "⚓", "color": "#9C88FF",
           "role_prefix": ["Anchorman", "Ball Winning Midfielder", "Deep Lying Playmaker"]},
    "CM": {"file": "Central Midfielder.xlsx", "title": "Central Midfielders", "icon": "⭐", "color": "#00D9FF",
           "role_prefix": ["Anchorman", "Ball Winning Midfielder", "Box-to-Box Midfielder", "Central Creator"]},
    "AM": {"file": "Attacking Midfielder.xlsx", "title": "Attacking Midfielders", "icon": "🎯", "color": "#FFA502",
           "role_prefix": ["Advanced Playmaker", "Central Creator", "Shadow Striker"]},
    "LW": {"file": "Left Winger.xlsx", "title": "Left Wingers", "icon": "⚡", "color": "#7bed9f",
           "role_prefix": ["Inside Forward", "Touchline Winger", "Wide Playmaker"]},
    "RW": {"file": "Right Wing.xlsx", "title": "Right Wingers", "icon": "⚡", "color": "#70a1ff",
           "role_prefix": ["Inside Forward", "Touchline Winger", "Wide Playmaker"]},
    "ST": {"file": "Strikers.xlsx", "title": "Strikers", "icon": "⚽", "color": "#ff7675",
           "role_prefix": ["Complete Forward", "Deep Lying Striker", "Poacher", "Target Man"]}
}

# Column names
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# =====================================================
# ENHANCED EXPORT UTILITIES
# =====================================================
def download_plotly_chart(fig, filename, format="png"):
    """Enhanced chart export with error handling"""
    try:
        if format.lower() == "png":
            return fig.to_image(format="png", width=1200, height=800, scale=2)
        elif format.lower() == "svg":
            return fig.to_image(format="svg", width=1200, height=800)
        elif format.lower() == "html":
            return fig.to_html(include_plotlyjs=True).encode('utf-8')
    except Exception as e:
        st.error(f"Error exporting chart: {e}")
        return None

def create_professional_excel_report(df, cfg, position_key):
    """Create comprehensive Excel report with professional formatting"""
    if not EXCEL_AVAILABLE:
        return df.to_csv(index=False).encode('utf-8')
    
    try:
        wb = Workbook()
        
        # Main data sheet
        ws = wb.active
        ws.title = "Player Data"
        
        # Professional styling
        header_font = Font(name="Arial", bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        border = Border(left=Side(style='thin'), right=Side(style='thin'), 
                       top=Side(style='thin'), bottom=Side(style='thin'))
        
        # Add data with formatting
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
        
        # Format headers
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 50)
        
        # Statistics sheet
        ws_stats = wb.create_sheet("Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats_data = []
        
        for col in numeric_cols[:10]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats_data.append({
                    'Metric': col, 'Count': len(col_data), 'Mean': round(col_data.mean(), 2),
                    'Median': round(col_data.median(), 2), 'Std Dev': round(col_data.std(), 2),
                    'Min': round(col_data.min(), 2), 'Max': round(col_data.max(), 2)
                })
        
        if stats_data:
            for r in dataframe_to_rows(pd.DataFrame(stats_data), index=False, header=True):
                ws_stats.append(r)
            
            for cell in ws_stats[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.border = border
        
        # Team analysis sheet
        ws_teams = wb.create_sheet("Team Analysis")
        if TEAM_COL in df.columns and len(df) > 0:
            team_stats = df.groupby(TEAM_COL).agg({
                NAME_COL: 'count', AGE_COL: 'mean', SHARE_COL: 'mean'
            }).round(2)
            team_stats.columns = ['Player Count', 'Avg Age', 'Avg Share']
            
            for r in dataframe_to_rows(team_stats, index=True, header=True):
                ws_teams.append(r)
            
            for cell in ws_teams[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.border = border
        
        # Metadata sheet
        ws_meta = wb.create_sheet("Report Info")
        metadata = [
            ["Report Title", f"Scout Lab Pro - {cfg.get('title', position_key)} Analysis"],
            ["Generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Position", cfg.get("title", position_key)],
            ["Total Players", len(df)],
            ["Data Source", cfg.get("file", "Sample Data")],
            ["Version", "Ultimate Edition v2.0"]
        ]
        
        for row in metadata:
            ws_meta.append(row)
        
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error creating Excel report: {e}")
        return df.to_csv(index=False).encode('utf-8')

def create_ultimate_package(df, cfg, position_key):
    """Create comprehensive download package"""
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Excel report
            excel_data = create_professional_excel_report(df, cfg, position_key)
            zip_file.writestr(f"{position_key}_professional_report.xlsx", excel_data)
            
            # CSV data
            csv_data = df.to_csv(index=False).encode('utf-8')
            zip_file.writestr(f"data/{position_key}_complete_data.csv", csv_data)
            
            # Top performers subset
            if 'IMPECT' in df.columns and len(df) > 10:
                top_performers = df.nlargest(min(50, len(df)//2), 'IMPECT')
                top_csv = top_performers.to_csv(index=False).encode('utf-8')
                zip_file.writestr(f"data/{position_key}_top_performers.csv", top_csv)
            
            # JSON exports
            json_data = df.to_json(orient='records', indent=2).encode('utf-8')
            zip_file.writestr(f"data/{position_key}_data.json", json_data)
            
            # Comprehensive metadata
            metadata = {
                "export_info": {
                    "title": f"Scout Lab Pro Ultimate - {cfg.get('title', position_key)}",
                    "position": position_key,
                    "generated_at": dt.datetime.now().isoformat(),
                    "version": "Ultimate Edition v2.0",
                    "total_players": len(df),
                    "data_completeness": f"{(df.notna().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%"
                },
                "included_files": [
                    f"{position_key}_professional_report.xlsx",
                    f"data/{position_key}_complete_data.csv", 
                    f"data/{position_key}_data.json",
                    "README.md"
                ],
                "technical_specs": {
                    "excel_sheets": ["Player Data", "Statistics", "Team Analysis", "Report Info"],
                    "formats": ["XLSX", "CSV", "JSON"],
                    "features": ["Professional formatting", "Multi-sheet analysis", "Comprehensive metadata"]
                }
            }
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            # README
            readme = f"""
# Scout Lab Pro Ultimate - {cfg.get('title', position_key)} Analysis Package

## Package Contents
- **{position_key}_professional_report.xlsx**: Multi-sheet Excel workbook with professional formatting
- **data/{position_key}_complete_data.csv**: Complete dataset in CSV format
- **data/{position_key}_data.json**: JSON format for programmatic access
- **metadata.json**: Package information and technical specifications

## Key Statistics
- **Total Players**: {len(df):,}
- **Data Quality**: {(df.notna().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}% complete
- **Generated**: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Generated by Scout Lab Pro Ultimate Edition - Professional Football Analytics Platform
            """
            zip_file.writestr("README.md", readme.encode('utf-8'))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating package: {e}")
        return None

# =====================================================
# ENHANCED CSS STYLING
# =====================================================
def generate_enhanced_css():
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&display=swap');

:root {{
    --primary: {COLORS["primary"]}; --secondary: {COLORS["secondary"]}; --success: {COLORS["success"]};
    --warning: {COLORS["warning"]}; --danger: {COLORS["danger"]}; --dark: {COLORS["dark"]};
    --darker: {COLORS["darker"]}; --card: {COLORS["card"]}; --border: {COLORS["border"]};
    --text: {COLORS["text"]}; --text-muted: {COLORS["text_muted"]}; --glass: {COLORS["glass"]};
    --easing: cubic-bezier(0.4, 0, 0.2, 1); --duration: 0.25s; --radius: 14px;
    --shadow: 0 4px 20px rgba(0, 0, 0, 0.2); --glow: 0 0 20px rgba(0, 217, 255, 0.3);
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
div[data-testid="collapsedControl"] {{ display: none; }}

.stApp {{
    background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 50%, var(--darker) 100%);
    color: var(--text); font-family: 'Plus Jakarta Sans', system-ui, sans-serif; min-height: 100vh;
}}

.stApp::before {{
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: -1;
    background: radial-gradient(circle at 25% 25%, rgba(0,217,255,0.03) 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, rgba(255,107,157,0.03) 0%, transparent 50%);
    animation: float 20s ease-in-out infinite;
}}

@keyframes float {{
    0%, 100% {{ transform: translate(0, 0); }}
    25% {{ transform: translate(20px, -20px); }}
    50% {{ transform: translate(-10px, 10px); }}
    75% {{ transform: translate(-20px, -10px); }}
}}

.block-container {{ padding: 1.5rem 2rem !important; max-width: 1600px !important; margin: 0 auto; }}

h1, h2, h3, h4 {{ color: var(--text); font-weight: 700; letter-spacing: -0.025em; margin: 0; }}
h1 {{ font-size: clamp(2rem, 4vw, 3rem); font-weight: 900;
     background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}

section[data-testid="stSidebar"] {{
    background: var(--glass); backdrop-filter: blur(20px); border-right: 1px solid var(--border);
}}

.modern-card {{
    background: var(--glass); backdrop-filter: blur(20px); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: var(--shadow);
    transition: all var(--duration) var(--easing); position: relative; overflow: hidden;
}}

.modern-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; opacity: 0;
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    transition: opacity var(--duration) var(--easing);
}}

.modern-card:hover {{ transform: translateY(-4px); box-shadow: var(--glow); }}
.modern-card:hover::before {{ opacity: 1; }}

.metric-card {{
    background: var(--glass); backdrop-filter: blur(10px); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem; text-align: center; transition: all var(--duration) var(--easing);
}}

.metric-card:hover {{ transform: translateY(-6px); box-shadow: var(--glow); border-color: var(--primary); }}

.metric-value {{
    font-size: 2rem; font-weight: 900; font-family: 'Inter', monospace; margin: 0.5rem 0;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}

.metric-label {{
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--text-muted); font-weight: 700;
}}

div.stButton > button {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: var(--darker); border: none; border-radius: 10px; padding: 0.6rem 1.5rem;
    font-weight: 800; transition: all var(--duration) var(--easing);
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
}}

div.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 217, 255, 0.6); }}

button[kind="secondary"] {{
    background: var(--card) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important;
}}

div[data-testid="stDataFrame"] {{
    background: var(--glass) !important; backdrop-filter: blur(20px);
    border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
}}

div[data-testid="stDataFrame"] thead tr th {{
    background: var(--darker) !important; color: var(--text) !important; font-weight: 800 !important;
    border-bottom: 2px solid var(--primary) !important;
}}

::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: var(--darker); border-radius: 4px; }}
::-webkit-scrollbar-thumb {{ 
    background: linear-gradient(180deg, var(--primary) 0%, var(--secondary) 100%); border-radius: 4px; 
}}
</style>
"""

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def safe_float(x):
    if x is None: return np.nan
    try: return float(str(x).strip().replace("%", "").replace(",", ""))
    except: return np.nan

def safe_fmt(x, decimals=2):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"

def percentile_rank(s):
    s = s.map(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def player_meta(row):
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—")) 
    age = safe_fmt(row.get(AGE_COL, np.nan), 0)
    share = safe_fmt(row.get(SHARE_COL, np.nan), 1)
    return f"{team} • {comp} • Age {age} • {share}% share"

def create_enhanced_plotly_theme():
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': COLORS["text"], 'size': 12, 'family': 'Plus Jakarta Sans'},
            'colorway': [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]],
            'xaxis': {'gridcolor': COLORS["border"], 'tickfont': {'color': COLORS["text_muted"]}},
            'yaxis': {'gridcolor': COLORS["border"], 'tickfont': {'color': COLORS["text_muted"]}},
        }
    }

# =====================================================
# DATA LOADING & SAMPLE DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_position_data(position_key):
    cfg = POSITION_CONFIG[position_key].copy()
    fp = Path(cfg["file"])
    
    if not fp.exists():
        return create_sample_data(position_key), cfg
    
    try:
        df = pd.read_excel(fp)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Process columns
        role_cols, metric_cols = [], []
        for col in df.columns:
            if col in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, NAT_COL, SHARE_COL] or 'BetterThan' in col:
                continue
            elif "IMPECT" in col or pd.api.types.is_numeric_dtype(df[col]):
                metric_cols.append(col)
            elif any(prefix in col for prefix in cfg.get("role_prefix", [])):
                role_cols.append(col)
        
        # Convert to numeric
        for c in role_cols + metric_cols + [AGE_COL, SHARE_COL]:
            if c in df.columns:
                df[c] = df[c].map(safe_float)
        
        # Add percentiles
        for m in metric_cols:
            if m in df.columns:
                df[m + " (pct)"] = percentile_rank(df[m])
        
        cfg["role_cols"] = role_cols
        cfg["metric_cols"] = metric_cols
        return df, cfg
        
    except Exception as e:
        st.warning(f"Could not load {cfg['file']}: {e}. Using sample data.")
        return create_sample_data(position_key), cfg

def create_sample_data(position_key):
    """Create realistic sample data"""
    np.random.seed(42)
    
    names = [f"{np.random.choice(['James', 'Lucas', 'Diego', 'Marco', 'Paulo', 'Carlos', 'Alex', 'David', 'Miguel'])}"
             f" {np.random.choice(['Silva', 'Santos', 'Garcia', 'Lopez', 'Martinez', 'Rodriguez', 'Brown', 'Wilson'])}"
             for _ in range(150)]
    
    teams = ["Barcelona", "Real Madrid", "Man City", "Liverpool", "Bayern", "PSG", "Juventus", "Chelsea", 
             "Arsenal", "Milan", "Atletico", "Dortmund", "Inter", "Napoli", "Sevilla"]
    
    data = {
        NAME_COL: names,
        TEAM_COL: np.random.choice(teams, 150),
        COMP_COL: np.random.choice(["La Liga", "Premier League", "Bundesliga", "Serie A", "Ligue 1"], 150),
        AGE_COL: np.random.normal(26, 4, 150).clip(18, 35),
        NAT_COL: np.random.choice(["Spain", "England", "Germany", "Italy", "France", "Brazil"], 150),
        SHARE_COL: np.random.beta(3, 2, 150) * 100,
        "IMPECT": np.random.normal(55, 18, 150).clip(10, 95),
        "Offensive IMPECT": np.random.normal(50, 15, 150).clip(5, 95),
        "Defensive IMPECT": np.random.normal(52, 16, 150).clip(5, 95),
    }
    
    # Add role scores
    cfg = POSITION_CONFIG[position_key]
    for role in cfg.get("role_prefix", [])[:4]:
        data[f"{role} Score"] = np.random.normal(58, 22, 150).clip(0, 98)
    
    df = pd.DataFrame(data)
    
    # Add percentiles
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != AGE_COL:
            df[f"{col} (pct)"] = percentile_rank(df[col])
    
    return df

# =====================================================
# FILTERS & STATE MANAGEMENT
# =====================================================
def ensure_state():
    for key in ["filters", "shortlist", "pinned", "selected_player", "compare_picks", "current_page"]:
        if key not in st.session_state:
            st.session_state[key] = {} if key != "current_page" else "landing"

def apply_filters(df, filters):
    """Apply user filters to dataframe"""
    out = df.copy()
    
    if filters.get("min_share", 0) > 0 and SHARE_COL in out.columns:
        out = out[out[SHARE_COL].fillna(0) >= filters["min_share"]]
    
    if "age_range" in filters and AGE_COL in out.columns:
        lo, hi = filters["age_range"]
        out = out[(out[AGE_COL] >= lo) & (out[AGE_COL] <= hi)]
    
    for filter_key, col in [("competitions", COMP_COL), ("teams", TEAM_COL), ("nats", NAT_COL)]:
        if filters.get(filter_key) and col in out.columns:
            out = out[out[col].isin(filters[filter_key])]
    
    q = str(filters.get("q", "")).strip().lower()
    if q:
        mask = pd.Series(False, index=out.index)
        for col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            if col in out.columns:
                mask |= out[col].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]
    
    return out

def strengths_weaknesses(cfg, row, topn=5):
    """Get top and bottom metrics for player"""
    pairs = []
    for m in cfg.get("metric_cols", []):
        if " - BetterThan" not in m:
            pct = safe_float(row.get(m + " (pct)", np.nan))
            if not np.isnan(pct):
                pairs.append((m, pct))
    
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:topn], list(reversed(pairs[-topn:]))

# =====================================================
# UI COMPONENTS
# =====================================================
def render_landing_page():
    st.markdown(generate_enhanced_css(), unsafe_allow_html=True)
    
    # Hero section
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 50%, var(--darker) 100%);
                padding: 4rem 2rem; text-align: center; border-radius: var(--radius);
                margin-bottom: 3rem; border: 1px solid var(--border); backdrop-filter: blur(20px);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">⚽</div>
        <h1 style="font-size: clamp(3rem, 6vw, 5rem); font-weight: 900; 
                   background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem;">
            Scout Lab Pro
        </h1>
        <div style="background: linear-gradient(135deg, {COLORS["success"]}40, {COLORS["warning"]}40);
                    border: 1px solid {COLORS["success"]}60; border-radius: 50px; padding: 0.5rem 1.5rem;
                    display: inline-block; margin-bottom: 1rem; font-weight: 800; color: {COLORS["success"]};">
            ✨ ULTIMATE ENHANCED EDITION ✨
        </div>
        <p style="font-size: 1.25rem; color: var(--text-muted); max-width: 600px; 
                  margin: 0 auto 2rem auto; line-height: 1.6;">
            The Complete Professional Football Analytics Platform with Ultimate Export Capabilities
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3, col4, col5 = st.columns(5)
    features = [
        (len(POSITION_CONFIG), "Positions", "⚽", COLORS["primary"]),
        ("100+", "Metrics", "📊", COLORS["success"]),
        ("∞", "Comparisons", "⚖️", COLORS["warning"]),
        ("AI", "Analytics", "🤖", COLORS["secondary"]),
        ("ALL", "Export Formats", "📥", COLORS["danger"])
    ]
    
    for i, (value, label, icon, color) in enumerate(features):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f'''
                <div class="metric-card" style="border-left: 3px solid {color};">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 1.8rem; margin-bottom: 0.5rem; color: {color}; font-weight: 900;">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
            ''', unsafe_allow_html=True)
    
    # Launch button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Ultimate Scout Lab Pro", width="stretch", type="primary"):
            st.session_state.current_page = "scout_app"
            st.rerun()

def render_scout_app():
    st.markdown(generate_enhanced_css(), unsafe_allow_html=True)
    ensure_state()

    # Sidebar
    with st.sidebar:
        st.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); font-weight: 900; margin-bottom: 1.5rem;">🎯 Ultimate Scout Control</div>')
        
        position = st.selectbox(
            "🏟️ Position",
            list(POSITION_CONFIG.keys()),
            format_func=lambda x: f"{POSITION_CONFIG[x]['icon']} {POSITION_CONFIG[x]['title']}"
        )

    # Load data
    with st.spinner("Loading player database..."):
        df, cfg = load_position_data(position)

    # Initialize state
    if position not in st.session_state.filters:
        st.session_state.filters[position] = {
            "q": "", "min_share": 0.0, "competitions": [], "teams": [], "nats": [],
            "age_range": (int(df[AGE_COL].min()) if AGE_COL in df.columns else 18,
                         int(df[AGE_COL].max()) if AGE_COL in df.columns else 35)
        }

    f = st.session_state.filters[position]
    position_color = cfg.get("color", COLORS["primary"])

    # Enhanced sidebar filters
    with st.sidebar:
        st.markdown('<div class="modern-card">')
        st.markdown("##### 🔍 Advanced Filters")
        
        f["q"] = st.text_input("Search", value=f.get("q", ""), placeholder="Player, team...")
        f["min_share"] = st.slider("Min Match Share (%)", 0.0, 100.0, f.get("min_share", 0.0), 5.0)
        
        if AGE_COL in df.columns:
            min_age, max_age = int(df[AGE_COL].min()), int(df[AGE_COL].max())
            f["age_range"] = st.slider("Age Range", min_age, max_age, f.get("age_range", (min_age, max_age)))
        
        for filter_key, col, label in [("teams", TEAM_COL, "Teams"), ("competitions", COMP_COL, "Competitions"), ("nats", NAT_COL, "Nationalities")]:
            if col in df.columns:
                options = sorted([x for x in df[col].dropna().unique() if str(x).strip()])
                f[filter_key] = st.multiselect(label, options, default=f.get(filter_key, []))
        
        if st.button("🔄 Reset Filters", width="stretch"):
            st.session_state.filters[position] = {"q": "", "min_share": 0.0, "competitions": [], "teams": [], "nats": [], "age_range": (18, 35)}
            st.rerun()
        
        st.markdown('</div>')

        # Export section
        st.markdown('<div class="modern-card">')
        st.markdown("##### 📥 Ultimate Exports")
        
        df_filtered = apply_filters(df, f)
        
        if len(df_filtered) > 0:
            # Professional Excel
            if st.button("📊 Professional Excel", width="stretch"):
                excel_data = create_professional_excel_report(df_filtered, cfg, position)
                st.download_button(
                    "Download Excel Report",
                    data=excel_data,
                    file_name=f"{position}_professional_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Ultimate package
            if st.button("📦 Ultimate Package", width="stretch"):
                with st.spinner("Creating ultimate package..."):
                    package_data = create_ultimate_package(df_filtered, cfg, position)
                    if package_data:
                        st.download_button(
                            "Download Ultimate Package",
                            data=package_data,
                            file_name=f"{position}_ultimate_package.zip",
                            mime="application/zip"
                        )
        
        st.markdown('</div>')

    # Apply filters
    df_f = apply_filters(df, f)

    # Header
    st.markdown(f"""
    <div style="background: var(--glass); backdrop-filter: blur(20px); border: 1px solid var(--border);
                border-radius: var(--radius); padding: 1.5rem 2rem; margin-bottom: 2rem;
                display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 1.5rem;">
            <div style="font-size: 1.8rem; font-weight: 900; background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚽ Scout Lab Pro Ultimate
            </div>
            <div style="background: linear-gradient(135deg, rgba(0,217,255,0.15) 0%, rgba(255,107,157,0.15) 100%);
                        border: 1px solid rgba(0,217,255,0.4); padding: 0.5rem 1.25rem; border-radius: 50px;
                        font-weight: 700; color: var(--text);">
                {cfg["icon"]} {cfg["title"]}
            </div>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <div style="padding: 0.4rem 1rem; background: var(--card); border: 1px solid var(--border);
                        border-radius: 50px; font-weight: 600; color: var(--text);">
                Players: <strong style="color: var(--primary);">{len(df_f):,}</strong>
            </div>
            <div style="padding: 0.5rem; background: linear-gradient(135deg, rgba(0,245,160,0.2), rgba(255,215,61,0.2));
                        border: 1px solid rgba(0,245,160,0.4); border-radius: 50px; color: #00F5A0; font-weight: 700;">
                ✨ Ultimate
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main tabs
    tabs = st.tabs([
        "🔍 Scout Browser",
        "👤 Player Profiles", 
        "⚖️ Head-to-Head",
        "🏆 Rankings",
        "📊 Analytics",
        "⭐ Shortlist",
        "📈 Market Insights",
        "📥 Export Center"
    ])

    with tabs[0]:  # Scout Browser
        if df_f.empty:
            st.info("🔍 No players match your current filters. Try adjusting the criteria in the sidebar.")
        else:
            # Controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                sort_options = ["IMPECT"] + cfg.get("role_cols", []) + cfg.get("metric_cols", [])
                sort_options = [c for c in sort_options if c in df_f.columns and " - BetterThan" not in c]
                sort_col = st.selectbox("📊 Sort by", sort_options, index=0 if sort_options else None)
            
            with col2:
                view_count = st.selectbox("👁️ Show", [10, 20, 30, 50], index=2)
            
            with col3:
                show_distribution = st.checkbox("📈 Show Distribution")

            if sort_col and sort_col in df_f.columns:
                # Distribution chart
                if show_distribution:
                    dist_fig = px.histogram(df_f, x=sort_col, nbins=25, color_discrete_sequence=[position_color],
                                          title=f"Distribution of {sort_col}")
                    dist_fig.update_layout(height=300, **create_enhanced_plotly_theme()['layout'])
                    
                    chart_col, export_col = st.columns([3, 1])
                    with chart_col:
                        st.plotly_chart(dist_fig, use_container_width=True)
                    with export_col:
                        st.markdown("**Export Chart**")
                        for fmt in ["PNG", "SVG", "HTML"]:
                            if st.button(fmt, key=f"dist_{fmt}"):
                                chart_data = download_plotly_chart(dist_fig, "distribution", fmt.lower())
                                if chart_data:
                                    st.download_button(
                                        f"Download {fmt}",
                                        data=chart_data,
                                        file_name=f"{position}_{sort_col}_distribution.{fmt.lower()}",
                                        mime=f"image/{fmt.lower()}" if fmt != "HTML" else "text/html"
                                    )

                # Player list
                results = df_f.sort_values(sort_col, ascending=False).head(view_count)
                
                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    name = str(row.get(NAME_COL, "Unknown"))
                    score = safe_fmt(row.get(sort_col, 0), 1)
                    
                    st.markdown(f'''
                        <div class="modern-card">
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div style="flex: 1;">
                                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                                        <div style="width: 2rem; height: 2rem; background: {position_color}; color: var(--darker);
                                                    border-radius: 50%; display: flex; align-items: center; justify-content: center;
                                                    font-weight: 900;">#{rank}</div>
                                        <div style="font-weight: 800; font-size: 1.1rem;">{name}</div>
                                    </div>
                                    <div style="color: var(--text-muted); margin-left: 3rem;">{player_meta(row)}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: var(--text-muted); font-size: 0.8rem;">{sort_col[:15]}</div>
                                    <div style="font-weight: 900; font-size: 1.5rem; color: {position_color};">{score}</div>
                                </div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

    with tabs[1]:  # Player Profiles
        if df_f.empty:
            st.warning("No players available with current filters.")
        else:
            players = sorted(df_f[NAME_COL].dropna().unique().tolist())
            player = st.selectbox("🎯 Select Player", players)
            
            if player:
                p_row = df_f[df_f[NAME_COL] == player].iloc[0]
                
                # Player header
                st.markdown(f'''
                    <div class="modern-card">
                        <div style="display: flex; align-items: center; gap: 2rem;">
                            <div style="width: 80px; height: 80px; border-radius: 50%;
                                        background: linear-gradient(135deg, {position_color}60, {COLORS["primary"]}60);
                                        display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                                {cfg["icon"]}
                            </div>
                            <div>
                                <h1 style="margin: 0; font-size: 2.5rem;">{player}</h1>
                                <p style="margin: 0.5rem 0 0 0; color: var(--text-muted);">{player_meta(p_row)}</p>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                metrics = [
                    (safe_fmt(p_row.get(AGE_COL, 0), 0), "Age", "🎂"),
                    (safe_fmt(p_row.get(SHARE_COL, 0), 1) + "%", "Match Share", "⏱️"),
                    (safe_fmt(p_row.get("IMPECT", 0), 1), "IMPECT", "⚡"),
                    (safe_fmt(p_row.get("Offensive IMPECT", 0), 1), "Off. IMPECT", "⚽")
                ]
                
                for i, (value, label, icon) in enumerate(metrics):
                    with [col1, col2, col3, col4][i]:
                        st.markdown(f'''
                            <div class="metric-card">
                                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">{icon}</div>
                                <div class="metric-value">{value}</div>
                                <div class="metric-label">{label}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                
                # Strengths and weaknesses
                st.markdown("#### Performance Analysis")
                top, bottom = strengths_weaknesses(cfg, p_row)
                
                str_col, weak_col = st.columns(2)
                
                with str_col:
                    st.markdown("**Top Strengths**")
                    for metric, pct in top[:5]:
                        color = COLORS["success"] if pct >= 80 else COLORS["primary"]
                        st.markdown(f'''
                            <div style="padding: 0.75rem; background: var(--glass); border-radius: 10px; 
                                        margin-bottom: 0.5rem; border-left: 3px solid {color};">
                                <div style="font-weight: 700; margin-bottom: 0.25rem;">{metric[:25]}</div>
                                <div style="color: {color}; font-weight: 600;">{pct:.0f}th percentile</div>
                            </div>
                        ''', unsafe_allow_html=True)
                
                with weak_col:
                    st.markdown("**Development Areas**")
                    for metric, pct in bottom[:5]:
                        color = COLORS["danger"] if pct <= 30 else COLORS["warning"]
                        st.markdown(f'''
                            <div style="padding: 0.75rem; background: var(--glass); border-radius: 10px;
                                        margin-bottom: 0.5rem; border-left: 3px solid {color};">
                                <div style="font-weight: 700; margin-bottom: 0.25rem;">{metric[:25]}</div>
                                <div style="color: {color}; font-weight: 600;">{pct:.0f}th percentile</div>
                            </div>
                        ''', unsafe_allow_html=True)

    with tabs[2]:  # Head-to-Head
        if df_f.empty:
            st.warning("No players available.")
        else:
            players = sorted(df_f[NAME_COL].dropna().unique().tolist())
            chosen = st.multiselect("🎯 Select Players to Compare", players, default=players[:3] if len(players) >= 3 else players)
            
            if len(chosen) >= 2:
                comp_df = df_f[df_f[NAME_COL].isin(chosen)]
                
                # Quick comparison table
                quick_cols = [c for c in [NAME_COL, TEAM_COL, AGE_COL, "IMPECT"] if c in comp_df.columns]
                st.dataframe(comp_df[quick_cols], width="stretch")
                
                # Radar chart for role comparison
                if cfg.get("role_cols") and len(cfg["role_cols"]) >= 3:
                    st.markdown("#### Role Suitability Comparison")
                    
                    radar_fig = go.Figure()
                    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]]
                    
                    for i, (_, player_row) in enumerate(comp_df.iterrows()):
                        player_name = str(player_row.get(NAME_COL, f"Player {i+1}"))
                        
                        radar_values = []
                        radar_labels = []
                        
                        for role in cfg["role_cols"][:6]:  # Limit to 6 for readability
                            val = safe_float(player_row.get(role, 0))
                            if not np.isnan(val):
                                radar_values.append(val)
                                radar_labels.append(role.replace("Score", "").strip()[:15])
                        
                        if radar_values:
                            color = colors[i % len(colors)]
                            radar_fig.add_trace(go.Scatterpolar(
                                r=radar_values,
                                theta=radar_labels,
                                fill='toself',
                                name=player_name,
                                line=dict(color=color, width=2)
                            ))
                    
                    radar_fig.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 100])),
                        height=500,
                        title="Role Suitability Comparison",
                        **create_enhanced_plotly_theme()['layout']
                    )
                    
                    chart_col, export_col = st.columns([3, 1])
                    
                    with chart_col:
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    with export_col:
                        st.markdown("**Export Comparison**")
                        for fmt in ["PNG", "SVG", "HTML"]:
                            if st.button(fmt, key=f"radar_{fmt}"):
                                chart_data = download_plotly_chart(radar_fig, "comparison", fmt.lower())
                                if chart_data:
                                    st.download_button(
                                        f"Download {fmt}",
                                        data=chart_data,
                                        file_name=f"{position}_comparison.{fmt.lower()}",
                                        mime=f"image/{fmt.lower()}" if fmt != "HTML" else "text/html"
                                    )
            else:
                st.info("Select at least 2 players for comparison")

    with tabs[3]:  # Rankings
        if df_f.empty:
            st.info("No data available for ranking")
        else:
            all_sortable = [c for c in ["IMPECT"] + cfg.get("role_cols", []) + cfg.get("metric_cols", []) 
                           if c in df_f.columns and " - BetterThan" not in c]
            
            if all_sortable:
                rank_col1, rank_col2 = st.columns([2, 1])
                
                with rank_col1:
                    metric = st.selectbox("📊 Ranking Metric", all_sortable)
                
                with rank_col2:
                    n = st.slider("Top N Players", 10, min(50, len(df_f)), 20)
                
                if metric:
                    ranking_df = df_f.sort_values(metric, ascending=False).head(n).copy()
                    ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))
                    
                    display_cols = [c for c in ["Rank", NAME_COL, TEAM_COL, metric] if c in ranking_df.columns]
                    st.dataframe(ranking_df[display_cols], width="stretch", height=400)
                    
                    # Export rankings
                    st.markdown("#### Export Rankings")
                    csv_data = ranking_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📄 Download Rankings CSV",
                        data=csv_data,
                        file_name=f"{position}_top_{n}_{metric}_rankings.csv",
                        mime="text/csv"
                    )

    with tabs[4]:  # Analytics
        if df_f.empty:
            st.info("No data available for analysis")
        else:
            numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                metric = st.selectbox("📊 Metric to Analyze", numeric_cols)
                
                if metric in df_f.columns:
                    metric_values = df_f[metric].dropna()
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    stats = [
                        (safe_fmt(metric_values.mean(), 2), "Mean", "📊"),
                        (safe_fmt(metric_values.median(), 2), "Median", "📈"),
                        (safe_fmt(metric_values.max(), 2), "Max", "⬆️"),
                        (safe_fmt(metric_values.std(), 2), "Std Dev", "📏")
                    ]
                    
                    for i, (value, label, icon) in enumerate(stats):
                        with [col1, col2, col3, col4][i]:
                            st.markdown(f'''
                                <div class="metric-card">
                                    <div style="font-size: 1rem; margin-bottom: 0.5rem;">{icon}</div>
                                    <div class="metric-value" style="font-size: 1.5rem;">{value}</div>
                                    <div class="metric-label">{label}</div>
                                </div>
                            ''', unsafe_allow_html=True)
                    
                    # Distribution
                    fig = px.histogram(df_f, x=metric, nbins=25, color_discrete_sequence=[COLORS["primary"]])
                    fig.add_vline(x=metric_values.mean(), line_dash="dash", line_color=COLORS["success"],
                                 annotation_text=f"Mean: {metric_values.mean():.2f}")
                    fig.update_layout(height=400, **create_enhanced_plotly_theme()['layout'])
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:  # Shortlist
        if "shortlist" not in st.session_state:
            st.session_state.shortlist = {}
        
        items = []
        for k, meta in st.session_state.shortlist.items():
            if "||" in k:
                pos, name = k.split("||", 1)
                items.append({
                    "Position": pos, "Player": name, 
                    "Tags": meta.get("tags", ""), "Notes": meta.get("notes", ""),
                    "Added": meta.get("added", dt.datetime.now()).strftime("%Y-%m-%d") if isinstance(meta.get("added"), dt.datetime) else "Unknown"
                })
        
        if not items:
            st.markdown('''
                <div class="modern-card" style="text-align: center; padding: 4rem;">
                    <div style="font-size: 4rem; margin-bottom: 1.5rem; opacity: 0.5;">⭐</div>
                    <h2 style="color: var(--text-muted);">Your Shortlist is Empty</h2>
                    <p style="color: var(--text-muted);">Add players from other tabs to build your shortlist.</p>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f"### ⭐ Your Shortlist ({len(items)} players)")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            shortlist_stats = [
                (len(items), "Total Players", "👥"),
                (len(set(item["Position"] for item in items)), "Positions", "⚽"),
                (len([item for item in items if item["Tags"].strip()]), "Tagged", "🏷️"),
                (len([item for item in items if item["Notes"].strip()]), "With Notes", "📝")
            ]
            
            for i, (value, label, icon) in enumerate(shortlist_stats):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f'''
                        <div class="metric-card">
                            <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">{icon}</div>
                            <div class="metric-value">{value}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                    ''', unsafe_allow_html=True)
            
            # Editable shortlist
            st.markdown("---")
            edited = st.data_editor(
                pd.DataFrame(items),
                width="stretch",
                height=400,
                num_rows="dynamic",
                column_config={
                    "Position": st.column_config.SelectboxColumn("Position", options=list(POSITION_CONFIG.keys())),
                    "Player": st.column_config.TextColumn("Player"),
                    "Tags": st.column_config.TextColumn("Tags"),
                    "Notes": st.column_config.TextColumn("Notes"),
                    "Added": st.column_config.TextColumn("Added")
                },
                key="shortlist_editor"
            )
            
            # Export shortlist
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = edited.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📄 Export CSV",
                    data=csv_data,
                    file_name=f"shortlist_{dt.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if EXCEL_AVAILABLE:
                    excel_data = create_professional_excel_report(edited, {"title": "Shortlist"}, "shortlist")
                    st.download_button(
                        "📊 Export Excel",
                        data=excel_data,
                        file_name=f"shortlist_{dt.datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                if st.button("🗑️ Clear All"):
                    st.session_state.shortlist = {}
                    st.rerun()

    with tabs[6]:  # Market Insights
        st.markdown("### 📈 Market Intelligence")
        
        if not df_f.empty:
            # Team analysis
            if TEAM_COL in df_f.columns:
                st.markdown("#### Team Distribution")
                
                team_stats = df_f.groupby(TEAM_COL).agg({
                    NAME_COL: 'count',
                    AGE_COL: 'mean',
                    'IMPECT': 'mean' if 'IMPECT' in df_f.columns else AGE_COL
                }).round(2)
                team_stats.columns = ['Player Count', 'Avg Age', 'Avg IMPECT' if 'IMPECT' in df_f.columns else 'Avg Age 2']
                team_stats = team_stats.sort_values('Player Count', ascending=False).head(10)
                
                # Team chart
                team_fig = px.bar(
                    x=team_stats.index,
                    y=team_stats['Player Count'],
                    color=team_stats['Player Count'],
                    color_continuous_scale="Viridis",
                    title="Top 10 Teams by Player Count"
                )
                team_fig.update_layout(height=400, **create_enhanced_plotly_theme()['layout'])
                
                insight_col1, insight_col2 = st.columns([3, 1])
                
                with insight_col1:
                    st.plotly_chart(team_fig, use_container_width=True)
                
                with insight_col2:
                    st.markdown("**Export Team Analysis**")
                    team_csv = team_stats.to_csv().encode('utf-8')
                    st.download_button(
                        "📄 Team Data CSV",
                        data=team_csv,
                        file_name=f"{position}_team_analysis.csv",
                        mime="text/csv"
                    )
                
                # Display team stats
                st.dataframe(team_stats, use_container_width=True)
            
            # Age distribution
            if AGE_COL in df_f.columns:
                st.markdown("#### Age Distribution Analysis")
                
                age_fig = px.histogram(
                    df_f, 
                    x=AGE_COL, 
                    nbins=15,
                    color_discrete_sequence=[COLORS["warning"]],
                    title=f"Age Distribution - {cfg['title']}"
                )
                age_fig.update_layout(height=350, **create_enhanced_plotly_theme()['layout'])
                st.plotly_chart(age_fig, use_container_width=True)

    with tabs[7]:  # Export Center
        st.markdown("### 📥 Ultimate Export Center")
        
        if not df_f.empty:
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.markdown("#### 📊 Data Exports")
                
                # Professional Excel
                if st.button("📊 Generate Professional Excel Report", width="stretch"):
                    with st.spinner("Creating comprehensive Excel report..."):
                        excel_data = create_professional_excel_report(df_f, cfg, position)
                        st.download_button(
                            "📥 Download Excel Report",
                            data=excel_data,
                            file_name=f"{position}_professional_report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="main_excel_export"
                        )
                        st.success("✅ Excel report generated!")
                
                # CSV Export
                if st.button("📄 Export Complete CSV Data", width="stretch"):
                    csv_data = df_f.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download CSV Data",
                        data=csv_data,
                        file_name=f"{position}_complete_data_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="main_csv_export"
                    )
                    st.success("✅ CSV data ready!")
                
                # JSON Export
                if st.button("🔧 Export JSON Data", width="stretch"):
                    json_data = df_f.to_json(orient='records', indent=2).encode('utf-8')
                    st.download_button(
                        "📥 Download JSON Data",
                        data=json_data,
                        file_name=f"{position}_data_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        key="main_json_export"
                    )
                    st.success("✅ JSON data ready!")
            
            with export_col2:
                st.markdown("#### 📦 Ultimate Packages")
                
                # Ultimate Package
                if st.button("📦 Generate Ultimate Analysis Package", width="stretch"):
                    with st.spinner("Creating ultimate analysis package..."):
                        package_data = create_ultimate_package(df_f, cfg, position)
                        if package_data:
                            st.download_button(
                                "📥 Download Ultimate Package",
                                data=package_data,
                                file_name=f"{position}_ultimate_package_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                                mime="application/zip",
                                key="main_package_export"
                            )
                            st.success("✅ Ultimate package created!")
                
                # Export summary
                st.markdown("---")
                st.markdown("#### 📋 Export Summary")
                st.info(f"""
                **Available in Ultimate Edition:**
                
                📊 **Professional Excel**: Multi-sheet workbook with statistics, team analysis, and professional formatting
                
                📄 **Complete CSV**: Raw data in universal format for analysis tools
                
                🔧 **JSON Data**: Structured format for developers and APIs
                
                📦 **Ultimate Package**: Complete ZIP archive with all formats, documentation, and metadata
                
                **Current Dataset:**
                - 👥 {len(df_f):,} players (filtered from {len(df):,} total)
                - 📊 {len(df_f.columns)} data columns
                - 🏟️ Position: {cfg['title']}
                - 📈 Data Quality: {(df_f.notna().sum().sum() / (len(df_f) * len(df_f.columns)) * 100):.1f}% complete
                """)

# =====================================================
# MAIN APPLICATION
# =====================================================
def main():
    ensure_state()
    
    # Navigation
    if st.session_state.current_page != "landing":
        # Simple navigation
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col2:
            nav1, nav2, nav3 = st.columns(3)
            
            with nav1:
                if st.button("🏠 Home", width="stretch", type="secondary" if st.session_state.current_page != "landing" else "primary"):
                    st.session_state.current_page = "landing"
                    st.rerun()
            
            with nav2:
                if st.button("⚽ Scout Platform", width="stretch", type="secondary" if st.session_state.current_page != "scout_app" else "primary"):
                    st.session_state.current_page = "scout_app"
                    st.rerun()
            
            with nav3:
                if st.button("📖 About", width="stretch", type="secondary"):
                    st.info("Scout Lab Pro Ultimate - The most advanced football analytics platform with complete export capabilities and professional-grade features.")
    
    # Route to appropriate page
    if st.session_state.current_page == "landing":
        render_landing_page()
    elif st.session_state.current_page == "scout_app":
        render_scout_app()

if __name__ == "__main__":
    main()
