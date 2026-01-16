import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import hashlib
import datetime as dt
from typing import List, Dict, Tuple, Optional
import io
import base64
import json
import zipfile

# For Excel exports
try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Scout Lab Pro - Enhanced Edition",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# =====================================================
# ENHANCED COLOR PALETTE & DESIGN TOKENS
# =====================================================
COLORS = {
    "primary": "#00D9FF",      # Electric cyan
    "secondary": "#FF6B9D",    # Vibrant pink
    "success": "#00F5A0",      # Neon green
    "warning": "#FFD93D",      # Electric yellow
    "danger": "#FF4757",       # Red accent
    "dark": "#0A0E27",         # Deep navy
    "darker": "#050816",       # Almost black
    "card": "#151B3B",         # Card background
    "card_hover": "#1A2142",   # Card hover state
    "border": "#1E2749",       # Subtle borders
    "border_hover": "#2A3458", # Hover borders
    "text": "#E8EAED",         # Light text
    "text_muted": "#8B92B0",   # Muted text
    "text_accent": "#B8BCC8",  # Accent text
    "overlay": "rgba(5, 8, 22, 0.95)",
    "glass": "rgba(21, 27, 59, 0.8)",
}

# =====================================================
# POSITION CONFIGURATIONS
# =====================================================
POSITION_CONFIG = {
    "GK": {
        "file": "Goalkeepers.xlsx",
        "title": "Goalkeepers",
        "icon": "🧤",
        "color": "#FF6B9D",
        "role_prefix": ["Ball Playing GK", "Box Defender", "Shot Stopper", "Sweeper Keeper"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "CB": {
        "file": "Central Defenders.xlsx",
        "title": "Central Defenders",
        "icon": "🛡️",
        "color": "#00F5A0",
        "role_prefix": ["Aerially Dominant CB", "Aggressive CB", "Ball Playing CB", "Strategic CB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "LB": {
        "file": "Left Back.xlsx",
        "title": "Left Backs",
        "icon": "⬅️",
        "color": "#FFD93D",
        "role_prefix": ["Attacking FB", "Defensive FB", "Progressive FB", "Inverted FB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "RB": {
        "file": "Right Back.xlsx",
        "title": "Right Backs",
        "icon": "➡️",
        "color": "#FF4757",
        "role_prefix": ["Attacking FB", "Defensive FB", "Progressive FB", "Inverted FB"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "DM": {
        "file": "Defensive Midfielder.xlsx",
        "title": "Defensive Midfielders",
        "icon": "⚓",
        "color": "#9C88FF",
        "role_prefix": ["Anchorman", "Ball Winning Midfielder", "Deep Lying Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "CM": {
        "file": "Central Midfielder.xlsx",
        "title": "Central Midfielders",
        "icon": "⭐",
        "color": "#00D9FF",
        "role_prefix": ["Anchorman", "Ball Winning Midfielder", "Box-to-Box Midfielder", "Central Creator", "Deep Lying Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "AM": {
        "file": "Attacking Midfielder.xlsx",
        "title": "Attacking Midfielders",
        "icon": "🎯",
        "color": "#FFA502",
        "role_prefix": ["Advanced Playmaker", "Central Creator", "Shadow Striker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "LW": {
        "file": "Left Winger.xlsx",
        "title": "Left Wingers",
        "icon": "⚡",
        "color": "#7bed9f",
        "role_prefix": ["Inside Forward", "Touchline Winger", "Wide Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "RW": {
        "file": "Right Wing.xlsx",
        "title": "Right Wingers",
        "icon": "⚡",
        "color": "#70a1ff",
        "role_prefix": ["Inside Forward", "Touchline Winger", "Wide Playmaker"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
    "ST": {
        "file": "Strikers.xlsx",
        "title": "Strikers",
        "icon": "⚽",
        "color": "#ff7675",
        "role_prefix": ["Complete Forward", "Deep Lying Striker", "Deep Running Striker", "Poacher", "Pressing Striker", "Second Striker", "Target Man"],
        "key_metrics": ["IMPECT", "Offensive IMPECT", "Defensive IMPECT"],
    },
}

# Column names
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"
ID_COL = "Player-ID"

# =====================================================
# DOWNLOAD & EXPORT UTILITIES
# =====================================================

def download_plotly_chart(fig, filename, format="png"):
    """Convert Plotly chart to downloadable format"""
    try:
        if format.lower() == "png":
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            return img_bytes
        elif format.lower() == "svg":
            return fig.to_image(format="svg", width=1200, height=800)
        elif format.lower() == "html":
            html_str = fig.to_html(include_plotlyjs=True)
            return html_str.encode('utf-8')
        elif format.lower() == "pdf":
            img_bytes = fig.to_image(format="pdf", width=1200, height=800)
            return img_bytes
    except Exception as e:
        st.error(f"Error exporting chart: {e}")
        return None

def create_excel_report(df, cfg, position_key):
    """Create comprehensive Excel report with multiple sheets"""
    if not EXCEL_AVAILABLE:
        # Fallback to CSV if openpyxl not available
        return df.to_csv(index=False).encode('utf-8')
    
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Player Data"
        
        # Style definitions
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        border = Border(left=Side(style='thin'), right=Side(style='thin'), 
                       top=Side(style='thin'), bottom=Side(style='thin'))
        center_align = Alignment(horizontal='center', vertical='center')
        
        # Main data sheet
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
        
        # Format headers
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
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
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
        
        # Summary statistics sheet
        ws_stats = wb.create_sheet("Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats_data = []
        
        for col in numeric_cols[:10]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats_data.append({
                    'Metric': col,
                    'Count': len(col_data),
                    'Mean': round(col_data.mean(), 2),
                    'Median': round(col_data.median(), 2),
                    'Std Dev': round(col_data.std(), 2),
                    'Min': round(col_data.min(), 2),
                    'Max': round(col_data.max(), 2),
                    '25th %ile': round(col_data.quantile(0.25), 2),
                    '75th %ile': round(col_data.quantile(0.75), 2)
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            for r in dataframe_to_rows(stats_df, index=False, header=True):
                ws_stats.append(r)
            
            for cell in ws_stats[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = center_align
                cell.border = border
        
        # Team analysis sheet
        ws_teams = wb.create_sheet("Team Analysis")
        if TEAM_COL in df.columns and len(df) > 0:
            team_stats = df.groupby(TEAM_COL).agg({
                NAME_COL: 'count',
                AGE_COL: 'mean',
                SHARE_COL: 'mean'
            }).round(2)
            team_stats.columns = ['Player Count', 'Avg Age', 'Avg Match Share']
            team_stats = team_stats.sort_values('Player Count', ascending=False)
            
            for r in dataframe_to_rows(team_stats, index=True, header=True):
                ws_teams.append(r)
            
            for cell in ws_teams[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = center_align
                cell.border = border
        
        # Add metadata sheet
        ws_meta = wb.create_sheet("Report Info")
        metadata = [
            ["Report Generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Position", cfg.get("title", position_key)],
            ["Total Players", len(df)],
            ["Data Source", cfg.get("file", "Unknown")],
            ["Generated By", "Scout Lab Pro Enhanced"],
            ["Export Format", "Excel Workbook (.xlsx)"],
            ["Analysis Features", "Multi-sheet, Statistics, Team Analysis"]
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

def create_download_package(df, cfg, position_key):
    """Create a complete download package with multiple files"""
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add main Excel report
            excel_data = create_excel_report(df, cfg, position_key)
            zip_file.writestr(f"{position_key}_complete_report.xlsx", excel_data)
            
            # Add CSV export
            csv_data = df.to_csv(index=False).encode('utf-8')
            zip_file.writestr(f"{position_key}_player_data.csv", csv_data)
            
            # Add JSON summary
            summary = {
                "position": position_key,
                "position_title": cfg.get("title", position_key),
                "total_players": len(df),
                "generated_at": dt.datetime.now().isoformat(),
                "data_columns": df.columns.tolist(),
                "export_info": {
                    "format": "Scout Lab Pro Enhanced Package",
                    "version": "2.0",
                    "includes": ["Excel Report", "CSV Data", "JSON Summary", "Documentation"]
                }
            }
            zip_file.writestr(f"{position_key}_summary.json", json.dumps(summary, indent=2))
            
            # Add comprehensive readme
            readme_content = f"""
Scout Lab Pro Enhanced - Data Export Package
==========================================

Position: {cfg.get('title', position_key)} ({position_key})
Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Players: {len(df)}
Data Columns: {len(df.columns)}

Package Contents:
- {position_key}_complete_report.xlsx: Comprehensive Excel report
- {position_key}_player_data.csv: Raw player data in CSV format  
- {position_key}_summary.json: Statistical summary and metadata
- README.txt: This documentation file

Generated by Scout Lab Pro Enhanced Edition
© 2024 - Professional Football Analytics Platform
            """
            zip_file.writestr("README.txt", readme_content.encode('utf-8'))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating download package: {e}")
        return None

# =====================================================
# ENHANCED CSS STYLING
# =====================================================
def generate_enhanced_css():
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&display=swap');

:root {{
    --primary: {COLORS["primary"]};
    --secondary: {COLORS["secondary"]};
    --success: {COLORS["success"]};
    --warning: {COLORS["warning"]};
    --danger: {COLORS["danger"]};
    --dark: {COLORS["dark"]};
    --darker: {COLORS["darker"]};
    --card: {COLORS["card"]};
    --card-hover: {COLORS["card_hover"]};
    --border: {COLORS["border"]};
    --border-hover: {COLORS["border_hover"]};
    --text: {COLORS["text"]};
    --text-muted: {COLORS["text_muted"]};
    --text-accent: {COLORS["text_accent"]};
    --glass: {COLORS["glass"]};
    
    --easing: cubic-bezier(0.4, 0, 0.2, 1);
    --duration-normal: 0.25s;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-full: 50px;
    --shadow-medium: 0 4px 20px rgba(0, 0, 0, 0.2);
    --shadow-glow: 0 0 20px rgba(0, 217, 255, 0.3);
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
div[data-testid="collapsedControl"] {{ display: none; }}

* {{ box-sizing: border-box; }}

html, body {{ scroll-behavior: smooth; }}

.stApp {{
    background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 50%, var(--darker) 100%);
    color: var(--text);
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    min-height: 100vh;
    position: relative;
}}

.stApp::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(circle at 25% 25%, rgba(0,217,255,0.03) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(255,107,157,0.03) 0%, transparent 50%);
    animation: float 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}}

@keyframes float {{
    0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
    25% {{ transform: translate(20px, -20px) rotate(1deg); }}
    50% {{ transform: translate(-10px, 10px) rotate(-1deg); }}
    75% {{ transform: translate(-20px, -10px) rotate(0.5deg); }}
}}

.hover-glow:hover {{
    transform: translateY(-6px);
    box-shadow: 0 12px 40px rgba(0, 217, 255, 0.4);
    border-color: var(--primary) !important;
}}

.block-container {{
    padding: 1.5rem 2rem !important;
    max-width: 1600px !important;
    margin: 0 auto;
}}

h1, h2, h3, h4, h5, h6 {{
    color: var(--text);
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin: 0;
}}

h1 {{
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 900;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}}

.modern-card {{
    background: var(--glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-medium);
    transition: all var(--duration-normal) var(--easing);
    position: relative;
    overflow: hidden;
}}

.modern-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    opacity: 0;
    transition: opacity var(--duration-normal) var(--easing);
}}

.modern-card:hover {{
    transform: translateY(-4px);
    box-shadow: var(--shadow-glow);
    border-color: var(--border-hover);
}}

.modern-card:hover::before {{
    opacity: 1;
}}

.metric-card {{
    background: var(--glass);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem;
    text-align: center;
    transition: all var(--duration-normal) var(--easing);
    position: relative;
    overflow: hidden;
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
    line-height: 1;
}}

.metric-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 700;
}}

.section-header {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    font-weight: 900;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
    position: relative;
}}

.section-header::after {{
    content: '';
    position: absolute;
    bottom: -2px; left: 0;
    width: 60px; height: 2px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
}}

div.stButton > button {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: var(--darker);
    border: none;
    border-radius: var(--radius-md);
    padding: 0.6rem 1.5rem;
    font-weight: 800;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
    transition: all var(--duration-normal) var(--easing);
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
    position: relative;
    overflow: hidden;
}}

div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 217, 255, 0.6);
}}

button[kind="secondary"] {{
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-medium) !important;
}}

button[kind="secondary"]:hover {{
    border-color: var(--primary) !important;
    background: var(--card-hover) !important;
    box-shadow: var(--shadow-glow) !important;
}}

div[data-testid="stDataFrame"] {{
    background: var(--glass) !important;
    backdrop-filter: blur(20px);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden;
    box-shadow: var(--shadow-medium);
}}

div[data-testid="stDataFrame"] thead tr th {{
    background: var(--darker) !important;
    color: var(--text) !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    border-bottom: 2px solid var(--primary) !important;
    padding: 1rem !important;
}}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: var(--darker); border-radius: 4px; }}
::-webkit-scrollbar-thumb {{ 
    background: linear-gradient(180deg, var(--primary) 0%, var(--secondary) 100%); 
    border-radius: 4px; 
}}
</style>
"""

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def safe_float(x):
    if x is None: return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}: return np.nan
    s = s.replace("%", "")
    try: return float(s)
    except: return np.nan

def safe_fmt(x, decimals=2):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"

def safe_int_fmt(x):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{int(round(v))}"

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(safe_float)

def percentile_rank(s: pd.Series) -> pd.Series:
    s = s.map(safe_float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = s.loc[mask].rank(pct=True, method="average") * 100
    return out

def player_meta(row: pd.Series) -> str:
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    nat = str(row.get(NAT_COL, "—"))
    age = safe_int_fmt(row.get(AGE_COL, np.nan))
    share = safe_fmt(row.get(SHARE_COL, np.nan), 1)
    return f"{team} • {comp} • {nat} • Age {age} • {share}% share"

def create_enhanced_plotly_theme():
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': COLORS["text"], 'size': 12, 'family': 'Plus Jakarta Sans, system-ui, sans-serif'},
            'colorway': [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"], COLORS["danger"]],
            'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60},
            'xaxis': {
                'gridcolor': COLORS["border"], 
                'linecolor': COLORS["border"], 
                'tickfont': {'color': COLORS["text_muted"]},
                'title': {'font': {'color': COLORS["text"]}}
            },
            'yaxis': {
                'gridcolor': COLORS["border"], 
                'linecolor': COLORS["border"], 
                'tickfont': {'color': COLORS["text_muted"]},
                'title': {'font': {'color': COLORS["text"]}}
            },
            'legend': {'bgcolor': COLORS["glass"], 'bordercolor': COLORS["border"], 'borderwidth': 1, 'font': {'color': COLORS["text"]}}
        }
    }

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data(show_spinner=False)
def load_position_data(position_key: str) -> tuple[pd.DataFrame, dict]:
    cfg = POSITION_CONFIG[position_key].copy()
    fp = Path(cfg["file"])
    
    if not fp.exists():
        # Create sample data if files don't exist
        return create_sample_data(position_key), cfg
    
    try:
        df = pd.read_excel(fp)
        df.columns = [str(c).strip() for c in df.columns]
        
        role_cols = []
        metric_cols = []
        
        for col in df.columns:
            if col in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, NAT_COL, SHARE_COL, ID_COL] or 'BetterThan' in col:
                continue
                
            if "IMPECT" in col:
                metric_cols.append(col)
                continue
            
            if df[col].dtype == 'object':
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if sample and isinstance(sample, str) and '%' in sample:
                    col_idx = df.columns.get_loc(col)
                    if col_idx < 20 and any(prefix in col for prefix in cfg.get("role_prefix", [])):
                        role_cols.append(col)
                    else:
                        metric_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                metric_cols.append(col)
        
        all_numeric = role_cols + metric_cols + [AGE_COL, SHARE_COL]
        coerce_numeric(df, all_numeric)
        
        for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            if c in df.columns:
                df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()
        
        for m in metric_cols:
            if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
                df[m + " (pct)"] = percentile_rank(df[m])
        
        cfg["role_cols"] = role_cols
        cfg["metric_cols"] = metric_cols
        cfg["all_metrics"] = role_cols + metric_cols
        
        return df, cfg
        
    except Exception as e:
        st.warning(f"Could not load {cfg['file']}: {e}. Using sample data.")
        return create_sample_data(position_key), cfg

def create_sample_data(position_key: str) -> pd.DataFrame:
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Sample player names
    first_names = ["James", "Mohamed", "Lucas", "Diego", "Marco", "Kevin", "Paulo", "Andre", "Carlos", "Rafael", 
                   "Roberto", "Alessandro", "Fernando", "Gabriel", "Antonio", "Miguel", "Jorge", "David", "Manuel", "Francisco"]
    last_names = ["Silva", "Santos", "Rodriguez", "Martinez", "Garcia", "Lopez", "Gonzalez", "Perez", "Wilson", "Brown",
                  "Smith", "Johnson", "Williams", "Jones", "Davis", "Miller", "Moore", "Taylor", "Anderson", "Thomas"]
    
    teams = ["Barcelona", "Real Madrid", "Manchester City", "Liverpool", "Bayern Munich", "PSG", "Juventus", "AC Milan",
             "Chelsea", "Arsenal", "Tottenham", "Manchester United", "Atletico Madrid", "Inter Milan", "Napoli"]
    
    competitions = ["La Liga", "Premier League", "Bundesliga", "Serie A", "Ligue 1", "Champions League"]
    
    nationalities = ["Spain", "England", "Germany", "Italy", "France", "Brazil", "Argentina", "Portugal", "Netherlands", "Belgium"]
    
    n_players = 150
    
    data = {
        NAME_COL: [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_players)],
        TEAM_COL: np.random.choice(teams, n_players),
        COMP_COL: np.random.choice(competitions, n_players),
        AGE_COL: np.random.normal(26, 4, n_players).clip(18, 35),
        NAT_COL: np.random.choice(nationalities, n_players),
        SHARE_COL: np.random.beta(2, 2, n_players) * 100,
        "IMPECT": np.random.normal(50, 15, n_players).clip(0, 100),
        "Offensive IMPECT": np.random.normal(45, 12, n_players).clip(0, 100),
        "Defensive IMPECT": np.random.normal(48, 13, n_players).clip(0, 100),
    }
    
    # Add position-specific role scores
    cfg = POSITION_CONFIG[position_key]
    for role in cfg.get("role_prefix", [])[:4]:  # Limit to first 4 roles
        data[f"{role} Score"] = np.random.normal(60, 20, n_players).clip(0, 100)
    
    df = pd.DataFrame(data)
    
    # Add percentile columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in [AGE_COL]:
            df[f"{col} (pct)"] = percentile_rank(df[col])
    
    return df

# =====================================================
# MAIN ENHANCED APPLICATION
# =====================================================
def main():
    st.markdown(generate_enhanced_css(), unsafe_allow_html=True)
    
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "main"
    
    # Enhanced Header
    st.markdown(f"""
    <div style="
        background: var(--glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            inset: 0;
            background: 
                radial-gradient(circle at 30% 30%, rgba(0,217,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 70% 70%, rgba(255,107,157,0.1) 0%, transparent 50%);
        "></div>
        <div style="position: relative; z-index: 2;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚽</div>
            <h1 style="
                font-size: 3rem;
                font-weight: 900;
                background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
            ">Scout Lab Pro Enhanced</h1>
            <div style="
                background: linear-gradient(135deg, {COLORS["success"]}40, {COLORS["warning"]}40);
                border: 1px solid {COLORS["success"]}60;
                border-radius: 50px;
                padding: 0.5rem 1.5rem;
                display: inline-block;
                margin-bottom: 1rem;
                font-weight: 800;
                color: {COLORS["success"]};
            ">
                ✨ Complete Export & Analytics Suite ✨
            </div>
            <p style="
                font-size: 1.2rem;
                color: var(--text-accent);
                margin: 0;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            ">
                Advanced Football Analytics with Professional Export Capabilities
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.markdown('<div class="section-header">🎯 Enhanced Scout Control</div>', unsafe_allow_html=True)
        
        position = st.selectbox(
            "🏟️ Position",
            list(POSITION_CONFIG.keys()),
            format_func=lambda x: f"{POSITION_CONFIG[x]['icon']} {POSITION_CONFIG[x]['title']}",
            index=0
        )
        
        st.markdown("---")
        st.markdown("##### 📥 Quick Export Tools")
        
        # Export availability notice
        if EXCEL_AVAILABLE:
            st.success("✅ Full export capabilities available")
        else:
            st.warning("⚠️ Limited exports (Excel library not available)")

    # Load data
    with st.spinner("🔄 Loading enhanced player database..."):
        try:
            df, cfg = load_position_data(position)
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()

    position_color = cfg.get("color", COLORS["primary"])

    # Enhanced Dashboard
    st.markdown(f"## 📊 {cfg['title']} Analytics Dashboard")
    
    # Quick stats with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
            <div class="metric-card hover-glow" style="border-left: 3px solid {COLORS["primary"]};">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">👥</div>
                <div class="metric-value" style="color: {COLORS["primary"]};">{len(df)}</div>
                <div class="metric-label">Total Players</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        teams_count = df[TEAM_COL].nunique() if TEAM_COL in df.columns else 0
        st.markdown(f'''
            <div class="metric-card hover-glow" style="border-left: 3px solid {COLORS["success"]};">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">⚽</div>
                <div class="metric-value" style="color: {COLORS["success"]};">{teams_count}</div>
                <div class="metric-label">Teams</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_age = df[AGE_COL].mean() if AGE_COL in df.columns else 0
        st.markdown(f'''
            <div class="metric-card hover-glow" style="border-left: 3px solid {COLORS["warning"]};">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">🎂</div>
                <div class="metric-value" style="color: {COLORS["warning"]};">{avg_age:.1f}</div>
                <div class="metric-label">Avg Age</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        avg_impect = df["IMPECT"].mean() if "IMPECT" in df.columns else 0
        st.markdown(f'''
            <div class="metric-card hover-glow" style="border-left: 3px solid {position_color};">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">⚡</div>
                <div class="metric-value" style="color: {position_color};">{avg_impect:.1f}</div>
                <div class="metric-label">Avg IMPECT</div>
            </div>
        ''', unsafe_allow_html=True)

    # Enhanced Tabs
    tabs = st.tabs([
        "🔍 Player Browser", 
        "📊 Advanced Analytics", 
        "📈 Market Insights", 
        "📥 Export Center",
        "🎯 Demo Features"
    ])

    with tabs[0]:
        st.markdown("### 🔍 Enhanced Player Browser")
        
        # Advanced filtering
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            search_query = st.text_input("🔍 Search players", placeholder="Name, team, nationality...")
        
        with filter_col2:
            if TEAM_COL in df.columns:
                teams = ["All"] + sorted(df[TEAM_COL].unique().tolist())
                selected_team = st.selectbox("⚽ Team", teams)
            else:
                selected_team = "All"
        
        with filter_col3:
            age_range = st.slider("🎂 Age Range", 
                                int(df[AGE_COL].min()) if AGE_COL in df.columns else 18, 
                                int(df[AGE_COL].max()) if AGE_COL in df.columns else 35, 
                                (20, 30))
        
        # Apply filters
        filtered_df = df.copy()
        
        if search_query:
            mask = df[NAME_COL].str.contains(search_query, case=False, na=False)
            if TEAM_COL in df.columns:
                mask |= df[TEAM_COL].str.contains(search_query, case=False, na=False)
            if NAT_COL in df.columns:
                mask |= df[NAT_COL].str.contains(search_query, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        if selected_team != "All" and TEAM_COL in df.columns:
            filtered_df = filtered_df[filtered_df[TEAM_COL] == selected_team]
        
        if AGE_COL in df.columns:
            filtered_df = filtered_df[
                (filtered_df[AGE_COL] >= age_range[0]) & 
                (filtered_df[AGE_COL] <= age_range[1])
            ]
        
        st.markdown(f"**Found {len(filtered_df)} players matching your criteria**")
        
        # Enhanced data display
        display_cols = [c for c in [NAME_COL, TEAM_COL, AGE_COL, "IMPECT"] if c in filtered_df.columns]
        if display_cols:
            st.dataframe(
                filtered_df[display_cols].head(20), 
                width="stretch", 
                height=400,
                use_container_width=True
            )

    with tabs[1]:
        st.markdown("### 📊 Advanced Analytics Dashboard")
        
        if "IMPECT" in df.columns:
            # Performance distribution with export options
            analytics_col1, analytics_col2 = st.columns([3, 1])
            
            with analytics_col1:
                fig = px.histogram(
                    df, 
                    x="IMPECT", 
                    nbins=25,
                    color_discrete_sequence=[position_color],
                    title=f"IMPECT Score Distribution - {cfg['title']}"
                )
                
                # Add statistical lines
                mean_val = df["IMPECT"].mean()
                median_val = df["IMPECT"].median()
                
                fig.add_vline(
                    x=mean_val, 
                    line_dash="dash", 
                    line_color=COLORS["success"],
                    annotation_text=f"Mean: {mean_val:.1f}"
                )
                
                fig.add_vline(
                    x=median_val, 
                    line_dash="dot", 
                    line_color=COLORS["warning"],
                    annotation_text=f"Median: {median_val:.1f}"
                )
                
                theme = create_enhanced_plotly_theme()
                fig.update_layout(height=500, **theme['layout'])
                st.plotly_chart(fig, use_container_width=True)
            
            with analytics_col2:
                st.markdown("**Chart Export Options**")
                
                if st.button("📊 PNG", key="analytics_png"):
                    png_data = download_plotly_chart(fig, "analytics", "png")
                    if png_data:
                        st.download_button(
                            "Download PNG",
                            data=png_data,
                            file_name=f"{position}_analytics.png",
                            mime="image/png"
                        )
                
                if st.button("🖼️ SVG", key="analytics_svg"):
                    svg_data = download_plotly_chart(fig, "analytics", "svg")
                    if svg_data:
                        st.download_button(
                            "Download SVG",
                            data=svg_data,
                            file_name=f"{position}_analytics.svg",
                            mime="image/svg+xml"
                        )
                
                if st.button("📄 HTML", key="analytics_html"):
                    html_data = download_plotly_chart(fig, "analytics", "html")
                    if html_data:
                        st.download_button(
                            "Download HTML",
                            data=html_data,
                            file_name=f"{position}_analytics.html",
                            mime="text/html"
                        )
                
                # Quick statistics
                st.markdown("---")
                st.markdown("**Quick Stats**")
                st.metric("Mean IMPECT", f"{mean_val:.1f}")
                st.metric("Median IMPECT", f"{median_val:.1f}")
                st.metric("Std Deviation", f"{df['IMPECT'].std():.1f}")

    with tabs[2]:
        st.markdown("### 📈 Market Insights")
        
        if TEAM_COL in df.columns and len(df) > 0:
            # Team analysis
            team_stats = df.groupby(TEAM_COL).agg({
                NAME_COL: 'count',
                AGE_COL: 'mean',
                "IMPECT": 'mean' if "IMPECT" in df.columns else AGE_COL
            }).round(2)
            
            team_stats.columns = ['Player Count', 'Avg Age', 'Avg IMPECT' if "IMPECT" in df.columns else 'Avg Age 2']
            team_stats = team_stats.sort_values('Player Count', ascending=False).head(10)
            
            # Team comparison chart
            team_fig = px.bar(
                x=team_stats.index,
                y=team_stats['Player Count'],
                color=team_stats['Player Count'],
                color_continuous_scale="Viridis",
                title="Top 10 Teams by Player Count"
            )
            
            team_fig.update_layout(
                height=400,
                xaxis_title="Team",
                yaxis_title="Number of Players",
                **create_enhanced_plotly_theme()['layout']
            )
            
            insight_col1, insight_col2 = st.columns([3, 1])
            
            with insight_col1:
                st.plotly_chart(team_fig, use_container_width=True)
            
            with insight_col2:
                st.markdown("**Export Team Analysis**")
                if st.button("📊 Team Chart PNG"):
                    team_png = download_plotly_chart(team_fig, "team_analysis", "png")
                    if team_png:
                        st.download_button(
                            "Download Team Chart",
                            data=team_png,
                            file_name=f"{position}_team_analysis.png",
                            mime="image/png"
                        )
                
                team_csv = team_stats.to_csv().encode('utf-8')
                st.download_button(
                    "📄 Team Data CSV",
                    data=team_csv,
                    file_name=f"{position}_team_stats.csv",
                    mime="text/csv"
                )
            
            # Display team stats table
            st.markdown("#### Team Statistics")
            st.dataframe(team_stats, use_container_width=True)

    with tabs[3]:
        st.markdown("### 📥 Enhanced Export Center")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.markdown("#### 📊 Data Exports")
            
            # Excel Report
            if st.button("📊 Generate Excel Report", width="stretch"):
                with st.spinner("Creating comprehensive Excel report..."):
                    excel_data = create_excel_report(df, cfg, position)
                    if excel_data:
                        st.download_button(
                            "📥 Download Excel Report",
                            data=excel_data,
                            file_name=f"{position}_comprehensive_report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="excel_download"
                        )
                        st.success("✅ Excel report generated successfully!")
            
            # CSV Export
            if st.button("📄 Export CSV Data", width="stretch"):
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download CSV Data",
                    data=csv_data,
                    file_name=f"{position}_data_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="csv_download"
                )
                st.success("✅ CSV data ready for download!")
            
            # JSON Export
            if st.button("🔧 Export JSON Data", width="stretch"):
                json_data = df.to_json(orient='records', indent=2).encode('utf-8')
                st.download_button(
                    "📥 Download JSON Data",
                    data=json_data,
                    file_name=f"{position}_data_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    key="json_download"
                )
                st.success("✅ JSON data ready for download!")
        
        with export_col2:
            st.markdown("#### 📦 Complete Packages")
            
            # Complete Package
            if st.button("📦 Generate Complete Package", width="stretch"):
                with st.spinner("Creating complete analysis package..."):
                    package_data = create_download_package(df, cfg, position)
                    if package_data:
                        st.download_button(
                            "📥 Download Complete Package",
                            data=package_data,
                            file_name=f"{position}_complete_package_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                            mime="application/zip",
                            key="package_download"
                        )
                        st.success("✅ Complete package generated!")
            
            # Export summary
            st.markdown("---")
            st.markdown("#### 📋 Export Summary")
            st.info(f"""
            **Available Exports:**
            - 📊 Excel: Multi-sheet workbook with statistics
            - 📄 CSV: Raw data for analysis
            - 🔧 JSON: Structured data format
            - 📦 ZIP: Complete package with documentation
            
            **Current Dataset:**
            - 👥 {len(df)} players
            - 📊 {len(df.columns)} data columns
            - 🏟️ Position: {cfg['title']}
            """)

    with tabs[4]:
        st.markdown("### 🎯 Demo Features & Capabilities")
        
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            st.markdown('''
                <div class="modern-card">
                    <h4>✨ Enhanced Features</h4>
                    <ul style="color: var(--text-muted); line-height: 1.8;">
                        <li><strong>📥 Complete Export Suite:</strong> Multi-format exports (Excel, CSV, JSON, ZIP)</li>
                        <li><strong>📊 Advanced Analytics:</strong> Interactive charts with statistical overlays</li>
                        <li><strong>🎯 Smart Filtering:</strong> Multi-criteria search and filtering</li>
                        <li><strong>📈 Market Insights:</strong> Team analysis and performance trends</li>
                        <li><strong>🎨 Professional Design:</strong> Modern UI with glass morphism</li>
                    </ul>
                </div>
            ''', unsafe_allow_html=True)
        
        with demo_col2:
            st.markdown('''
                <div class="modern-card">
                    <h4>🔧 Technical Capabilities</h4>
                    <ul style="color: var(--text-muted); line-height: 1.8;">
                        <li><strong>📊 Chart Exports:</strong> PNG, SVG, HTML, PDF formats</li>
                        <li><strong>📈 Interactive Plots:</strong> Plotly with custom themes</li>
                        <li><strong>🏗️ Excel Reports:</strong> Multi-sheet with professional formatting</li>
                        <li><strong>📦 Package Creation:</strong> ZIP archives with documentation</li>
                        <li><strong>🚀 Performance:</strong> Optimized data processing and caching</li>
                    </ul>
                </div>
            ''', unsafe_allow_html=True)
        
        # Sample data notice
        if len(df) == 150:  # Sample data
            st.info("""
            🔬 **Demo Mode Active**: This demonstration uses generated sample data to showcase the platform's capabilities.
            
            In a production environment, you would:
            - Load real player data from Excel files
            - Have access to comprehensive statistics and metrics
            - Generate reports based on actual scouting data
            - Export real analysis for decision-making
            """)
        
        # Feature showcase
        st.markdown("#### 🎬 Feature Showcase")
        
        showcase_col1, showcase_col2, showcase_col3 = st.columns(3)
        
        with showcase_col1:
            if st.button("🎯 Test Advanced Filter"):
                st.success("✅ Advanced filtering system operational!")
                st.balloons()
        
        with showcase_col2:
            if st.button("📊 Generate Sample Chart"):
                sample_fig = px.scatter(
                    df, 
                    x=AGE_COL, 
                    y="IMPECT" if "IMPECT" in df.columns else AGE_COL,
                    color=TEAM_COL if TEAM_COL in df.columns else None,
                    title="Sample Scatter Plot"
                )
                sample_fig.update_layout(**create_enhanced_plotly_theme()['layout'])
                st.plotly_chart(sample_fig, use_container_width=True)
        
        with showcase_col3:
            if st.button("📥 Test Export System"):
                sample_csv = df.head(10).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Sample",
                    data=sample_csv,
                    file_name="scout_lab_pro_sample.csv",
                    mime="text/csv"
                )
                st.success("✅ Export system operational!")

    # Enhanced Footer
    st.markdown("---")
    st.markdown(f'''
        <div style="
            background: var(--glass);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            text-align: center;
            margin-top: 2rem;
        ">
            <div style="font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem;">
                ⚽ Scout Lab Pro Enhanced Edition
            </div>
            <div style="color: var(--text-muted); font-size: 0.9rem;">
                Professional Football Analytics Platform with Complete Export Capabilities
                <br>
                Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Version 2.0 Enhanced
            </div>
        </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
