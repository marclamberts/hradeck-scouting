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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
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
=======
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Scout Lab", layout="wide", page_icon="⚽")

=======
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Scout Lab", layout="wide", page_icon="⚽")

>>>>>>> Stashed changes
# =====================================================
# COLORS (YOUR PALETTE)
# =====================================================
COLORS = {
    "yellow": "#F4C430",
    "black": "#0B0B0B",
    "white": "#F7F7F7",
    "grey": "#9AA0A6",
    "background": "#0e1117",
}

# =====================================================
# CANONICAL COLUMN NAMES (expected if possible)
# =====================================================
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
NAME_COL = "Name"
TEAM_COL = "Team"
COMP_COL = "Competition"
AGE_COL = "Age"
NAT_COL = "Nationality"
SHARE_COL = "Match Share"

# =====================================================
<<<<<<< Updated upstream
<<<<<<< Updated upstream
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
            ["Report Title", f"FCHK Scout - {cfg.get('title', position_key)} Analysis"],
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
                    "title": f"FCHK - {cfg.get('title', position_key)}",
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
=======
=======
>>>>>>> Stashed changes
# POSITION/DATASET CONFIG  (EDIT THESE TO MATCH YOUR EXCEL COLUMNS)
# =====================================================
POSITION_CONFIG = {
    "RWB": {
        "file": "RWB.xlsx",
        "title": "Right Wingback (RWB)",
        "metrics": [
            "Progressive Receiving",
            "Progressive Receiving - where to? - Wide Right Zone",
            "Distance Covered Dribbles - Dribble",
            "Breaking Opponent Defence",
            "Breaking Opponent Defence - where from? - Wide Right Zone",
            "High Cross",
            "Low Cross",
            "Defensive Ball Control",
            "Number of presses during opponent build-up",
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
        ],
        "role_defs": {
            "Attacking Wingback Score": {
                "High Cross": 0.30,
                "Low Cross": 0.25,
                "Offensive IMPECT": 0.25,
                "Progressive Receiving - where to? - Wide Right Zone": 0.20,
            },
            "Progressor Score": {
                "Progressive Receiving": 0.30,
                "Breaking Opponent Defence": 0.30,
                "Distance Covered Dribbles - Dribble": 0.25,
                "Breaking Opponent Defence - where from? - Wide Right Zone": 0.15,
            },
            "Defensive Wingback Score": {
                "Defensive Ball Control": 0.30,
                "Number of presses during opponent build-up": 0.30,
                "Defensive IMPECT": 0.30,
                "IMPECT": 0.10,
            },
            "Balanced Score": {
                "IMPECT": 0.30,
                "Offensive IMPECT": 0.20,
                "Defensive IMPECT": 0.20,
                "Progressive Receiving": 0.15,
                "High Cross": 0.15,
            },
        },
        "key_metrics": [
            "IMPECT",
            "Offensive IMPECT",
            "Defensive IMPECT",
            "High Cross",
            "Low Cross",
            "Breaking Opponent Defence",
        ],
        "radar_metrics": [
            "Progressive Receiving",
            "Breaking Opponent Defence",
            "Distance Covered Dribbles - Dribble",
            "High Cross",
            "Low Cross",
            "Defensive Ball Control",
            "Number of presses during opponent build-up",
            "IMPECT",
        ],
        "default_sort": "Balanced Score",
    },
    # NOTE: Replace placeholder metric names with EXACT CM.xlsx headers.
    "CM": {
        "file": "CM.xlsx",
        "title": "Central Defender (from CM.xlsx)",
        "metrics": [
            "Progressive Passing",
            "Pass Completion %",
            "Aerial Duels Won %",
            "Defensive Duels Won %",
            "Interceptions",
            "Blocks",
            "Clearances",
            "Pressures",
            "IMPECT",
            "Defensive IMPECT",
            "Offensive IMPECT",
        ],
        "role_defs": {
            "Stopper Score": {
                "Defensive Duels Won %": 0.30,
                "Blocks": 0.20,
                "Clearances": 0.20,
                "Defensive IMPECT": 0.30,
            },
            "Ball-Playing Score": {
                "Progressive Passing": 0.35,
                "Pass Completion %": 0.25,
                "Offensive IMPECT": 0.20,
                "IMPECT": 0.20,
            },
            "Aerial Score": {
                "Aerial Duels Won %": 0.45,
                "Clearances": 0.20,
                "Blocks": 0.15,
                "Defensive IMPECT": 0.20,
            },
            "Balanced Score": {
                "IMPECT": 0.30,
                "Defensive IMPECT": 0.25,
                "Progressive Passing": 0.20,
                "Defensive Duels Won %": 0.15,
                "Aerial Duels Won %": 0.10,
            },
        },
        "key_metrics": [
            "IMPECT",
            "Defensive IMPECT",
            "Progressive Passing",
            "Interceptions",
            "Aerial Duels Won %",
        ],
        "radar_metrics": [
            "Progressive Passing",
            "Pass Completion %",
            "Interceptions",
            "Blocks",
            "Aerial Duels Won %",
            "IMPECT",
        ],
        "default_sort": "Balanced Score",
    },
}
>>>>>>> Stashed changes

DISPLAY_RENAMES = {
    "Balanced Score": "Balanced",
    "Attacking Wingback Score": "Attacking WB",
    "Defensive Wingback Score": "Defensive WB",
    "Progressor Score": "Progressor",
    "Stopper Score": "Stopper",
    "Ball-Playing Score": "Ball-Playing",
    "Aerial Score": "Aerial",
    "Match Share": "Share",
}

DISPLAY_RENAMES = {
    "Balanced Score": "Balanced",
    "Attacking Wingback Score": "Attacking WB",
    "Defensive Wingback Score": "Defensive WB",
    "Progressor Score": "Progressor",
    "Stopper Score": "Stopper",
    "Ball-Playing Score": "Ball-Playing",
    "Aerial Score": "Aerial",
    "Match Share": "Share",
}

# =====================================================
<<<<<<< Updated upstream
<<<<<<< Updated upstream
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
=======
=======
>>>>>>> Stashed changes
# UI THEME (tight / dark / custom)
# =====================================================
st.markdown(
    f"""
<style>
/* Hide streamlit chrome */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

/* App base */
.stApp {{
  background: {COLORS["background"]};
  color: {COLORS["white"]};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(11,11,11,0.92) 0%, rgba(14,17,23,0.96) 100%);
  border-right: 1px solid rgba(244,196,48,0.12);
}}
section[data-testid="stSidebar"] * {{
  color: {COLORS["white"]};
}}

/* Typography */
h1, h2, h3 {{
  letter-spacing: -0.03em;
}}
.small {{
  color: {COLORS["grey"]};
  font-size: 0.92rem;
}}
.muted {{
  color: {COLORS["grey"]};
}}
.kicker {{
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.75rem;
  color: {COLORS["grey"]};
  font-weight: 900;
}}

/* Cards */
.card {{
  background: rgba(11,11,11,0.65);
  border: 1px solid rgba(244,196,48,0.10);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.30);
}}
.card-strong {{
  background: rgba(11,11,11,0.78);
  border: 1px solid rgba(244,196,48,0.18);
}}

/* Header bar */
.headerbar {{
  background: linear-gradient(135deg, rgba(11,11,11,0.85) 0%, rgba(14,17,23,0.85) 100%);
  border: 1px solid rgba(244,196,48,0.16);
  border-radius: 18px;
  padding: 14px 16px;
  display:flex;
  align-items:center;
  justify-content:space-between;
}}
.header-left {{
  display:flex;
  gap:10px;
  align-items:baseline;
  flex-wrap: wrap;
}}
.brand {{
  font-size: 1.35rem;
  font-weight: 950;
}}
.pill {{
  display:inline-flex;
  align-items:center;
  gap:8px;
  border: 1px solid rgba(247,247,247,0.10);
  background: rgba(11,11,11,0.55);
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 0.9rem;
  color: {COLORS["white"]};
}}
.pill-accent {{
  border-color: rgba(244,196,48,0.40);
  background: rgba(244,196,48,0.10);
}}
.pill-solid {{
  border-color: rgba(244,196,48,0.65);
  background: {COLORS["yellow"]};
  color: {COLORS["black"]};
}}

/* Chips */
.chip {{
  display:inline-flex;
  align-items:center;
  gap:6px;
  border: 1px solid rgba(247,247,247,0.10);
  background: rgba(11,11,11,0.55);
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.9rem;
  color: {COLORS["white"]};
  margin-right: 6px;
  margin-bottom: 6px;
}}
.chip strong {{ font-weight: 950; color: {COLORS["yellow"]}; }}

/* Results row */
.player-row {{
  border: 1px solid rgba(247,247,247,0.08);
  border-radius: 14px;
  background: rgba(11,11,11,0.55);
  padding: 12px 12px;
}}
.player-row:hover {{
  border-color: rgba(244,196,48,0.28);
}}
.player-name {{
  font-weight: 950;
  font-size: 1.03rem;
  color: {COLORS["white"]};
}}
.player-meta {{
  color: {COLORS["grey"]};
  font-size: 0.9rem;
  margin-top: 2px;
}}

/* Buttons */
div.stButton > button {{
  border-radius: 12px;
  border: 1px solid rgba(247,247,247,0.12);
  background: rgba(11,11,11,0.55);
  color: {COLORS["white"]};
  font-weight: 900;
}}
div.stButton > button:hover {{
  border-color: rgba(244,196,48,0.45);
  background: rgba(244,196,48,0.10);
}}
/* Primary button */
button[kind="primary"] {{
  border: 1px solid rgba(244,196,48,0.65) !important;
  background: {COLORS["yellow"]} !important;
  color: {COLORS["black"]} !important;
  font-weight: 950 !important;
}}

/* Inputs */
div[data-baseweb="input"] > div {{
  border-radius: 12px !important;
  background: rgba(11,11,11,0.55) !important;
  border: 1px solid rgba(247,247,247,0.10) !important;
}}
div[data-baseweb="select"] > div {{
  border-radius: 12px !important;
  background: rgba(11,11,11,0.55) !important;
  border: 1px solid rgba(247,247,247,0.10) !important;
}}
/* Slider */
div[data-testid="stSlider"] > div {{
  border-radius: 14px;
  background: rgba(11,11,11,0.40);
  border: 1px solid rgba(247,247,247,0.08);
  padding: 8px 10px;
}}
/* Tabs */
button[data-baseweb="tab"] {{
  font-weight: 950;
}}
div[data-baseweb="tab-list"] {{
  gap: 6px;
}}
/* Reduce vertical gaps */
div[data-testid="stVerticalBlock"] > div {{ gap: 0.62rem; }}

/* Dataframe header */
div[data-testid="stDataFrame"] thead tr th {{
  font-weight: 950;
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
}}
</style>
"""

# =====================================================
<<<<<<< Updated upstream
<<<<<<< Updated upstream
# UTILITY FUNCTIONS
# =====================================================
def safe_float(x):
    if x is None: return np.nan
    try: return float(str(x).strip().replace("%", "").replace(",", ""))
    except: return np.nan
=======
=======
>>>>>>> Stashed changes
# SAFE PARSING + SCORING
# =====================================================
def safe_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—"}:
        return np.nan
    s = s.replace("%", "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    if s.count(",") >= 1 and s.count(".") == 1:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan
>>>>>>> Stashed changes

def safe_fmt(x, decimals=2):
    v = safe_float(x)
    return "—" if np.isnan(v) else f"{v:.{decimals}f}"

<<<<<<< Updated upstream
def percentile_rank(s):
=======
def safe_int_fmt(x):
    v = safe_float(x)
    if np.isnan(v):
        return "—"
    return f"{int(round(v))}"

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(safe_float)

def percentile_rank(s: pd.Series) -> pd.Series:
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
def score_from_z(z: pd.Series) -> pd.Series:
    z = z.map(safe_float).fillna(0.0)
    return (50 + 15 * z).clip(0, 100)

def rename_for_display(df_: pd.DataFrame) -> pd.DataFrame:
    return df_.rename(columns=DISPLAY_RENAMES)

# =====================================================
# "PRO TABLE" (NO STYLER)
# =====================================================
def pro_table(df: pd.DataFrame, pct_cols: list[str] | None = None, height: int = 600):
    pct_cols = pct_cols or []
    pct_cols = [c for c in pct_cols if c in df.columns]
    col_config = {}

    for c in pct_cols:
        col_config[c] = st.column_config.ProgressColumn(
            label=c,
            min_value=0,
            max_value=100,
            format="%.0f",
            help="Percentile (0–100)",
        )

    for c in df.columns:
        if c in pct_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            col_config[c] = st.column_config.NumberColumn(label=c, format="%.2f")

    if NAME_COL in df.columns:
        col_config[NAME_COL] = st.column_config.TextColumn(label=NAME_COL, width="large")
    if TEAM_COL in df.columns:
        col_config[TEAM_COL] = st.column_config.TextColumn(label=TEAM_COL, width="medium")
    if COMP_COL in df.columns:
        col_config[COMP_COL] = st.column_config.TextColumn(label=COMP_COL, width="medium")

    st.dataframe(df, width="stretch", height=height, column_config=col_config, hide_index=True)

# =====================================================
# LOAD + PREP DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_and_prepare(position_key: str) -> tuple[pd.DataFrame, dict]:
    cfg = POSITION_CONFIG[position_key]
    fp = Path(cfg["file"])
    if not fp.exists():
        raise FileNotFoundError(f"Missing {cfg['file']}. Put it next to app.py.")

    df = pd.read_excel(fp)
    df.columns = [str(c).strip() for c in df.columns]

    coerce_numeric(df, cfg["metrics"] + [AGE_COL, SHARE_COL])

    for c in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    for m in cfg["metrics"]:
        if m in df.columns:
            df[m + " (pct)"] = percentile_rank(df[m])

    for role, weights in cfg["role_defs"].items():
        z = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            if col in df.columns:
                z = z + zscore(df[col]) * float(w)
        df[role] = score_from_z(z)

    coerce_numeric(df, list(cfg["role_defs"].keys()))
    return df, cfg

# =====================================================
# STATE + SHORTLIST + COMPARE
# =====================================================
def ensure_state():
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "shortlist" not in st.session_state:
        st.session_state.shortlist = {}  # "POS||Name" -> {"tags":"", "notes":""}
    if "pinned" not in st.session_state:
        st.session_state.pinned = {}  # pos -> name
    if "selected_player" not in st.session_state:
        st.session_state.selected_player = None
    if "compare_picks" not in st.session_state:
        st.session_state.compare_picks = {}  # pos -> list[str]

def shortlist_key(position_key: str, player_name: str) -> str:
    return f"{position_key}||{player_name}"

def add_to_shortlist(position_key: str, player_name: str):
    k = shortlist_key(position_key, player_name)
    if k not in st.session_state.shortlist:
        st.session_state.shortlist[k] = {"tags": "", "notes": ""}

def remove_from_shortlist(position_key: str, player_name: str):
    k = shortlist_key(position_key, player_name)
    if k in st.session_state.shortlist:
        del st.session_state.shortlist[k]

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T

def similar_players(df_f: pd.DataFrame, player_name: str, feature_cols: list[str], topk: int = 10) -> pd.DataFrame:
    if df_f.empty or NAME_COL not in df_f.columns:
        return pd.DataFrame()
    if player_name not in df_f[NAME_COL].values:
        return pd.DataFrame()

    cols = [c for c in feature_cols if c in df_f.columns and pd.api.types.is_numeric_dtype(df_f[c])]
    if not cols:
        return pd.DataFrame()

    X = np.column_stack([zscore(df_f[c]).to_numpy() for c in cols])
    sim = cosine_similarity_matrix(X)

    idx = df_f.index[df_f[NAME_COL] == player_name][0]
    base_i = df_f.index.get_loc(idx)
    scores = pd.Series(sim[base_i], index=df_f.index)

    out = df_f.loc[scores.sort_values(ascending=False).index].copy()
    out["Similarity"] = scores.loc[out.index].values
    out = out[out[NAME_COL] != player_name].head(topk)

    show_cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] if c in out.columns] + ["Similarity"]
    return out[show_cols]

def player_meta(row: pd.Series) -> str:
    team = str(row.get(TEAM_COL, "—"))
    comp = str(row.get(COMP_COL, "—"))
    nat = str(row.get(NAT_COL, "—"))
    age = safe_int_fmt(row.get(AGE_COL, np.nan))
    share = safe_fmt(row.get(SHARE_COL, np.nan), 2)
    return f"{team} · {comp} · {nat} · Age {age} · Share {share}"

def strengths_weaknesses(cfg: dict, row: pd.Series, topn: int = 6):
    pairs = []
    for m in cfg["metrics"]:
        pct = safe_float(row.get(m + " (pct)", np.nan))
        if not np.isnan(pct):
            pairs.append((m, pct))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:topn]
    bottom = list(reversed(pairs[-topn:])) if len(pairs) >= topn else list(reversed(pairs))
    return top, bottom

# =====================================================
# FILTERS
# =====================================================
def default_filters_for(df: pd.DataFrame):
    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        if len(vals):
            lo = int(max(15, np.floor(vals.min())))
            hi = int(min(50, np.ceil(vals.max())))
        else:
            lo, hi = 15, 45
    else:
        lo, hi = 15, 45

    return {
        "q": "",
        "min_share": 0.20,
        "competitions": [],
        "teams": [],
        "nats": [],
        "age_range": (lo, hi),
    }

def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df.copy()

    if SHARE_COL in out.columns:
        out = out[out[SHARE_COL].fillna(0) >= float(f.get("min_share", 0.0))]

    if AGE_COL in out.columns and "age_range" in f:
        lo, hi = f["age_range"]
        out = out[(out[AGE_COL].fillna(lo) >= lo) & (out[AGE_COL].fillna(hi) <= hi)]

    if f.get("competitions") and COMP_COL in out.columns:
        out = out[out[COMP_COL].isin(f["competitions"])]

    if f.get("teams") and TEAM_COL in out.columns:
        out = out[out[TEAM_COL].isin(f["teams"])]

    if f.get("nats") and NAT_COL in out.columns:
        out = out[out[NAT_COL].isin(f["nats"])]

    q = str(f.get("q", "")).strip().lower()
    if q:
        mask = pd.Series(False, index=out.index)
        for col in [NAME_COL, TEAM_COL, COMP_COL, NAT_COL]:
            if col in out.columns:
                mask = mask | out[col].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]

    return out

# =====================================================
# APP START
# =====================================================
ensure_state()

# Sidebar: dataset + compact filters
st.sidebar.markdown("### ⚙️ Control Room")
position = st.sidebar.selectbox("Dataset", list(POSITION_CONFIG.keys()), index=0)
df, cfg = load_and_prepare(position)

role_cols_all = [r for r in cfg["role_defs"].keys() if r in df.columns]
default_sort = cfg.get("default_sort", "Balanced Score")
if default_sort not in df.columns and role_cols_all:
    default_sort = role_cols_all[0]

if position not in st.session_state.filters:
    st.session_state.filters[position] = default_filters_for(df)
f = st.session_state.filters[position]

# Filters in expander (tighter)
with st.sidebar.expander("Filters", expanded=True):
    f["q"] = st.text_input("Search", value=f.get("q", ""), placeholder="Name / Team / Comp / Nat…")
    f["min_share"] = st.slider("Min Share", 0.0, 1.0, float(f.get("min_share", 0.20)), 0.05)

    if AGE_COL in df.columns and len(df):
        vals = df[AGE_COL].dropna()
        if len(vals):
            min_age = int(max(15, np.floor(vals.min())))
            max_age = int(min(50, np.ceil(vals.max())))
        else:
            min_age, max_age = 15, 45
        lo, hi = f.get("age_range", (min_age, max_age))
        lo = max(min_age, lo)
        hi = min(max_age, hi)
        f["age_range"] = st.slider("Age", min_age, max_age, (lo, hi), 1)

    if COMP_COL in df.columns:
        comps_all = sorted([c for c in df[COMP_COL].dropna().unique().tolist() if str(c).strip() != ""])
        f["competitions"] = st.multiselect("Competitions", comps_all, default=f.get("competitions", []))

    if TEAM_COL in df.columns:
        teams_all = sorted([t for t in df[TEAM_COL].dropna().unique().tolist() if str(t).strip() != ""])
        f["teams"] = st.multiselect("Teams", teams_all, default=f.get("teams", []))

    if NAT_COL in df.columns:
        nats_all = sorted([n for n in df[NAT_COL].dropna().unique().tolist() if str(n).strip() != ""])
        f["nats"] = st.multiselect("Nationalities", nats_all, default=f.get("nats", []))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset", type="primary", key=f"reset_{position}"):
            st.session_state.filters[position] = default_filters_for(df)
            st.rerun()
    with c2:
        st.caption("Live")

# Apply filters
df_f = apply_filters(df, f)

# Pinned defaults
if position not in st.session_state.pinned:
    if len(df_f) and NAME_COL in df_f.columns:
        st.session_state.pinned[position] = df_f.sort_values(default_sort, ascending=False).iloc[0][NAME_COL]
    else:
        st.session_state.pinned[position] = None

# Compare picks per position
if position not in st.session_state.compare_picks:
    st.session_state.compare_picks[position] = []

# Header bar
shortlist_count = len(st.session_state.shortlist)
teams_n = df_f[TEAM_COL].nunique() if TEAM_COL in df_f.columns else 0
comps_n = df_f[COMP_COL].nunique() if COMP_COL in df_f.columns else 0

st.markdown(
    f"""
<div class="headerbar">
  <div class="header-left">
    <div class="brand">⚽ Scout Lab</div>
    <span class="pill pill-accent">{cfg["title"]}</span>
    <span class="pill">Players <strong>{len(df_f)}</strong></span>
    <span class="pill">Teams <strong>{teams_n}</strong></span>
    <span class="pill">Comps <strong>{comps_n}</strong></span>
  </div>
  <div style="display:flex;gap:10px;align-items:center;">
    <span class="pill pill-solid">⭐ Shortlist {shortlist_count}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Search", "Profile", "Compare", "Leaderboards", "Distributions", "Shortlist"])

# =====================================================
# TAB: SEARCH (tight results + pinned preview)
# =====================================================
with tabs[0]:
    st.markdown('<div class="kicker">Scout</div>', unsafe_allow_html=True)
    st.markdown("## Search")

    if df_f.empty:
        st.info("No players match your filters. Try lowering Min Share or clearing selections.")
        c1, c2, c3 = st.columns(3)
        if c1.button("Share 0.10", key=f"lowshare_{position}", type="primary"):
            f["min_share"] = 0.10
            st.rerun()
        if c2.button("Clear search", key=f"clearq_{position}"):
            f["q"] = ""
            st.rerun()
        if c3.button("Reset all", key=f"resetall_{position}"):
            st.session_state.filters[position] = default_filters_for(df)
            st.rerun()
    else:
        # Sort selector
        sort_options = [c for c in ([default_sort] + role_cols_all) if c in df_f.columns]
        if not sort_options:
            sort_options = [c for c in df_f.columns if pd.api.types.is_numeric_dtype(df_f[c])]
        sort_col = st.selectbox("Sort", options=sort_options, index=0, key=f"sort_{position}")

        # Layout: results + preview
        left, right = st.columns([1.15, 1])

        with left:
            st.markdown("### Results")
            st.caption("Open to pin → preview updates. ★ to shortlist.")

            results = df_f.sort_values(sort_col, ascending=False).head(60).copy()
            for _, r in results.iterrows():
                name = str(r.get(NAME_COL, "—"))
                in_sl = shortlist_key(position, name) in st.session_state.shortlist

                st.markdown('<div class="player-row">', unsafe_allow_html=True)
                a, b, c = st.columns([3.3, 1.1, 1.4])
                with a:
                    st.markdown(f'<div class="player-name">{name}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="player-meta">{player_meta(r)}</div>', unsafe_allow_html=True)
                with b:
                    st.metric(DISPLAY_RENAMES.get(sort_col, sort_col).replace(" Score", ""), safe_fmt(r.get(sort_col, np.nan), 1))
                with c:
                    if st.button("Open", key=f"open_{position}_{name}"):
                        st.session_state.pinned[position] = name
                        st.session_state.selected_player = name
                        st.rerun()
                    if st.button("★" if not in_sl else "✓", key=f"slq_{position}_{name}"):
                        add_to_shortlist(position, name) if not in_sl else remove_from_shortlist(position, name)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("### Pinned")
            pinned = st.session_state.pinned.get(position)

            if not pinned:
                st.caption("Pick a player in Results.")
            else:
                p = df_f[df_f[NAME_COL] == pinned].head(1)
                if p.empty:
                    st.caption("Pinned player not found.")
                else:
                    row = p.iloc[0]
                    st.markdown('<div class="card card-strong">', unsafe_allow_html=True)
                    st.markdown(f"### {pinned}")
                    st.caption(player_meta(row))
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Role tiles (top 4)
                    st.markdown("#### Role snapshot")
                    tiles = st.columns(min(4, max(1, len(role_cols_all))))
                    for i, rc in enumerate(role_cols_all[:4]):
                        tiles[i].metric(DISPLAY_RENAMES.get(rc, rc).replace(" Score", ""), safe_fmt(row.get(rc, np.nan), 1))

                    # Strengths/Weaknesses
                    st.markdown("#### Signals (pct)")
                    top, bottom = strengths_weaknesses(cfg, row, topn=5)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Strengths**")
                        for m, pct in top:
                            st.write(f"↑ {m} — {pct:.0f}")
                    with c2:
                        st.markdown("**Risks**")
                        for m, pct in bottom:
                            st.write(f"↓ {m} — {pct:.0f}")

                    st.markdown("#### Actions")
                    a1, a2, a3 = st.columns(3)
                    with a1:
                        in_sl = shortlist_key(position, pinned) in st.session_state.shortlist
                        if st.button("Shortlist" if not in_sl else "Shortlisted", key=f"sl_pin_{position}", type="primary"):
                            add_to_shortlist(position, pinned) if not in_sl else remove_from_shortlist(position, pinned)
                            st.rerun()
                    with a2:
                        picks = st.session_state.compare_picks[position]
                        if st.button("Add to Compare", key=f"addcmp_{position}"):
                            if pinned not in picks:
                                picks.append(pinned)
                                st.session_state.compare_picks[position] = picks[:6]
                            st.rerun()
                    with a3:
                        st.caption("Profile tab uses pinned")

# =====================================================
# TAB: PROFILE (report-style)
# =====================================================
with tabs[1]:
    st.markdown('<div class="kicker">Report</div>', unsafe_allow_html=True)
    st.markdown("## Player Profile")

    if df_f.empty or NAME_COL not in df_f.columns:
        st.warning("No players available with current filters.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique().tolist())
        default_player = (
            st.session_state.selected_player
            or st.session_state.pinned.get(position)
            or (players[0] if players else None)
        )
        if default_player not in players and players:
            default_player = players[0]

        player = st.selectbox("Player", players, index=players.index(default_player) if default_player in players else 0, key=f"profile_{position}")
        p = df_f[df_f[NAME_COL] == player].head(1)
        row = p.iloc[0]

        # Header card
        st.markdown('<div class="card card-strong">', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1, 1, 1.2])
        c1.markdown(f"### {player}")
        c1.caption(player_meta(row))
        c2.metric("Age", safe_int_fmt(row.get(AGE_COL, np.nan)))
        c3.metric("Share", safe_fmt(row.get(SHARE_COL, np.nan), 2))
        if "Balanced Score" in df_f.columns:
            c4.metric("Balanced", safe_fmt(row.get("Balanced Score", np.nan), 1))
        else:
            c4.metric("Score", "—")

        in_sl = shortlist_key(position, player) in st.session_state.shortlist
        if c5.button("Shortlist" if not in_sl else "Shortlisted", key=f"sl_profile_{position}_{player}", type="primary"):
            add_to_shortlist(position, player) if not in_sl else remove_from_shortlist(position, player)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Notes/tags (only if shortlisted)
        if shortlist_key(position, player) in st.session_state.shortlist:
            meta = st.session_state.shortlist[shortlist_key(position, player)]
            st.markdown("### Notes")
            a, b = st.columns([1, 2])
            meta["tags"] = a.text_input("Tags", value=meta.get("tags", ""), placeholder="e.g. U23, target, left-footed", key=f"tags_{position}_{player}")
            meta["notes"] = b.text_area("Notes", value=meta.get("notes", ""), height=90, key=f"notes_{position}_{player}")

        st.markdown("---")

        # Strengths / Weaknesses
        top, bottom = strengths_weaknesses(cfg, row, topn=6)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Strengths (pct)")
            for m, pct in top:
                st.write(f"↑ **{m}** — {pct:.0f}")
        with c2:
            st.markdown("### Risks (pct)")
            for m, pct in bottom:
                st.write(f"↓ **{m}** — {pct:.0f}")

        st.markdown("---")
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Role scores")
            role_cols = [c for c in role_cols_all if c in df_f.columns]
            role_row = pd.DataFrame(
                [
                    {
                        "Role": DISPLAY_RENAMES.get(c, c).replace(" Score", ""),
                        "Score": safe_float(row.get(c, np.nan)),
                        "Percentile": safe_float(percentile_rank(df_f[c]).loc[p.index[0]]) if c in df_f.columns else np.nan,
                    }
                    for c in role_cols
                ]
            )
            pro_table(role_row, pct_cols=["Percentile"], height=320)

            st.markdown("### Key metrics (pct)")
            key = []
            for m in cfg["metrics"]:
                if m in df_f.columns and (m + " (pct)") in df_f.columns:
                    key.append(
                        {
                            "Metric": m,
                            "Value": safe_float(row.get(m, np.nan)),
                            "Percentile": safe_float(row.get(m + " (pct)", np.nan)),
                        }
                    )
            key_df = pd.DataFrame(key)
            if len(key_df):
                pro_table(key_df, pct_cols=["Percentile"], height=520)
            else:
                st.info("No metric columns found.")

        with right:
            st.markdown("### Radar (roles)")
            radar_cols = [c for c in role_cols_all if c in df_f.columns]
            if radar_cols:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=[safe_float(row.get(c, np.nan)) if not np.isnan(safe_float(row.get(c, np.nan))) else 0 for c in radar_cols],
                        theta=[DISPLAY_RENAMES.get(c, c).replace(" Score", "") for c in radar_cols],
                        fill="toself",
                        name=player,
                    )
                )
                fig.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(247,247,247,0.10)")),
                    height=520,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Similar players")
            st.caption("Cosine similarity on selected features (z-scored).")
            sim_default = [m for m in cfg["radar_metrics"] if m in df_f.columns][:6]
            sim_features = st.multiselect(
                "Features",
                options=[m for m in cfg["metrics"] if m in df_f.columns],
                default=sim_default,
                key=f"simfeat_{position}",
            )
            topk = st.slider("Top K", 5, 25, 10, 1, key=f"simk_{position}")
            sim_df = similar_players(df_f, player, sim_features, topk=topk)
            if len(sim_df):
                pro_table(sim_df, pct_cols=[], height=360)
                if st.button("Add top 3 to Compare", key=f"addsimcmp_{position}_{player}", type="primary"):
                    picks = st.session_state.compare_picks[position]
                    for nm in sim_df[NAME_COL].head(3).tolist():
                        if nm not in picks:
                            picks.append(nm)
                    st.session_state.compare_picks[position] = picks[:6]
                    st.rerun()
            else:
                st.info("Not enough data/features to compute similarity.")

# =====================================================
# TAB: COMPARE
# =====================================================
with tabs[2]:
    st.markdown('<div class="kicker">Decision</div>', unsafe_allow_html=True)
    st.markdown("## Compare")

    if df_f.empty or NAME_COL not in df_f.columns:
        st.warning("No players available with current filters.")
    else:
        players = sorted(df_f[NAME_COL].dropna().unique().tolist())
        picks = [p for p in st.session_state.compare_picks.get(position, []) if p in players]

        default = picks[:]
        if len(default) < 2 and len(players) >= 2:
            default = players[:2]

        chosen = st.multiselect("Players (2–6)", players, default=default, key=f"cmp_{position}")
        st.session_state.compare_picks[position] = chosen

        if len(chosen) < 2:
            st.info("Pick at least 2 players to compare.")
        else:
            comp_df = df_f[df_f[NAME_COL].isin(chosen)].copy()

            st.markdown("### Quick cards")
            cols = st.columns(min(4, len(chosen)))
            for i, nm in enumerate(chosen[:4]):
                r = comp_df[comp_df[NAME_COL] == nm].head(1).iloc[0]
                cols[i].markdown('<div class="card">', unsafe_allow_html=True)
                cols[i].markdown(f"**{nm}**")
                cols[i].caption(player_meta(r))
                if "Balanced Score" in comp_df.columns:
                    cols[i].metric("Balanced", safe_fmt(r.get("Balanced Score", np.nan), 1))
                cols[i].markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Role bars")
            role_cols = [c for c in role_cols_all if c in comp_df.columns]
            if role_cols:
                melt = comp_df.melt(
                    id_vars=[c for c in [NAME_COL, TEAM_COL, COMP_COL] if c in comp_df.columns],
                    value_vars=role_cols,
                    var_name="Role",
                    value_name="Score",
                )
                melt["Role"] = melt["Role"].map(lambda x: DISPLAY_RENAMES.get(x, x).replace(" Score", ""))
                fig = px.bar(melt, x="Score", y=NAME_COL, color="Role", barmode="group")
                fig.update_layout(
                    height=520,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Radar (key metric pct)")
            radar_metrics = [m + " (pct)" for m in cfg["radar_metrics"] if (m + " (pct)") in comp_df.columns]
            if radar_metrics:
                fig2 = go.Figure()
                for nm in chosen:
                    sub = comp_df[comp_df[NAME_COL] == nm].head(1)
                    r = [
                        safe_float(sub.iloc[0].get(m, np.nan)) if not np.isnan(safe_float(sub.iloc[0].get(m, np.nan))) else 0
                        for m in radar_metrics
                    ]
                    theta = [m.replace(" (pct)", "") for m in radar_metrics]
                    fig2.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name=nm))
                fig2.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100], gridcolor="rgba(247,247,247,0.10)")),
                    height=620,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Table")
            show = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL] + role_cols if c in comp_df.columns]
            sort_role = "Balanced Score" if "Balanced Score" in comp_df.columns else (role_cols[0] if role_cols else show[-1])
            st.dataframe(comp_df[show].sort_values(sort_role, ascending=False), use_container_width=True, height=520)

# =====================================================
# TAB: LEADERBOARDS
# =====================================================
with tabs[3]:
    st.markdown('<div class="kicker">Market</div>', unsafe_allow_html=True)
    st.markdown("## Leaderboards")

    if df_f.empty:
        st.info("No players to rank with current filters.")
    else:
        available_roles = [r for r in role_cols_all if r in df_f.columns]
        if not available_roles:
            st.warning("No role score columns found for this dataset.")
        else:
            role = st.selectbox(
                "Role",
                available_roles,
                index=available_roles.index(default_sort) if default_sort in available_roles else 0,
                key=f"lb_role_{position}",
            )
            n = st.slider("Rows", 10, 100, 40, 5, key=f"lb_n_{position}")

            cols = [c for c in [NAME_COL, TEAM_COL, COMP_COL, AGE_COL, SHARE_COL, NAT_COL, role] if c in df_f.columns]
            out = df_f.sort_values(role, ascending=False).head(n)[cols].copy()
            out[role + " (pct)"] = percentile_rank(df_f[role]).reindex(out.index)

            pro_table(rename_for_display(out), pct_cols=[role + " (pct)"], height=740)

            st.markdown("### Top 20 bar")
            fig = px.bar(
                df_f.sort_values(role, ascending=False).head(20),
                x=role,
                y=NAME_COL,
                orientation="h",
                color=role,
                hover_data=[c for c in [TEAM_COL, COMP_COL, AGE_COL, SHARE_COL] if c in df_f.columns],
            )
            fig.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=650,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["white"]),
            )
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB: DISTRIBUTIONS
# =====================================================
with tabs[4]:
    st.markdown('<div class="kicker">Context</div>', unsafe_allow_html=True)
    st.markdown("## Distributions")

    if df_f.empty:
        st.info("No players with current filters.")
    else:
        numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            default_metric = default_sort if default_sort in numeric_cols else numeric_cols[0]
            metric = st.selectbox("Metric", numeric_cols, index=numeric_cols.index(default_metric), key=f"dist_{position}")

            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.histogram(df_f, x=metric, nbins=30)
                fig1.update_layout(
                    height=420,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.box(df_f, y=metric, points="all")
                fig2.update_layout(
                    height=420,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Mean by Competition / Team")
            split_options = [c for c in [COMP_COL, TEAM_COL] if c in df_f.columns]
            if split_options:
                split = st.radio("Split", split_options, horizontal=True, key=f"split_{position}")
                topk = st.slider("Top groups", 5, 30, 12, 1, key=f"topk_{position}")
                g = df_f.groupby(split, dropna=True)[metric].mean().sort_values(ascending=False).head(topk).reset_index()
                fig3 = px.bar(g, x=metric, y=split, orientation="h", color=metric)
                fig3.update_layout(
                    height=600,
                    yaxis=dict(categoryorder="total ascending"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["white"]),
                )
                st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# TAB: SHORTLIST (editable + export)
# =====================================================
with tabs[5]:
    st.markdown('<div class="kicker">Targets</div>', unsafe_allow_html=True)
    st.markdown("## Shortlist")
    st.caption("Saved targets across datasets. Edit tags/notes and export.")

    items = []
    for k, meta in st.session_state.shortlist.items():
        pos, name = k.split("||", 1)
        items.append({"Position": pos, "Name": name, "Tags": meta.get("tags", ""), "Notes": meta.get("notes", "")})

    if not items:
        st.info("Shortlist is empty. Add players from Search or Profile.")
    else:
        sl_df = pd.DataFrame(items)
        edited = st.data_editor(
            sl_df,
            use_container_width=True,
            height=520,
            num_rows="dynamic",
            column_config={
                "Position": st.column_config.TextColumn(width="small"),
                "Name": st.column_config.TextColumn(width="medium"),
                "Tags": st.column_config.TextColumn(width="medium"),
                "Notes": st.column_config.TextColumn(width="large"),
            },
            key="shortlist_editor",
        )

        # Write edits back
        new_shortlist = {}
        for _, r in edited.iterrows():
            pos = str(r.get("Position", "")).strip()
            name = str(r.get("Name", "")).strip()
            if not pos or not name:
                continue
            new_shortlist[shortlist_key(pos, name)] = {
                "tags": str(r.get("Tags", "") or ""),
                "notes": str(r.get("Notes", "") or ""),
            }
        st.session_state.shortlist = new_shortlist

        c1, c2 = st.columns([1, 1])
        with c1:
            st.download_button(
                "Download shortlist (CSV)",
                data=edited.to_csv(index=False).encode("utf-8"),
                file_name="shortlist.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("Clear shortlist", key="clear_shortlist", type="primary"):
                st.session_state.shortlist = {}
                st.rerun()
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
