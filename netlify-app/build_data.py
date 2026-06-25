"""
build_data.py  —  Export Lamberts Index Excel → JSON files for Netlify app
Run:  python netlify-app/build_data.py
Output: netlify-app/public/data/*.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

SRC  = Path(__file__).parent.parent / "data" / "Lamberts_Index_Full_Report.xlsx"
DEST = Path(__file__).parent / "public" / "data"
DEST.mkdir(parents=True, exist_ok=True)

# Columns to keep (compact payload)
KEEP = [
    "Player", "Team", "League", "League Tier", "Birth Country",
    "Pos", "Full Position", "Age", "Age Band", "Height (cm)", "Weight (kg)",
    "Foot", "On Loan", "Contract", "Exp?",
    "Mkt Val (€)", "Model Val (€)", "Val Ratio",
    "Tier", "LI Score", "Lamberts Index",
    "DIS", "ADI", "PADS", "PDS",
    "Status", "vs Hradec", "Minutes", "Matches", "Min/Match",
    # Key stats per position (shared subset)
    "Def Actions/90", "Def Duel %", "Aerial/90", "Aerial Won %",
    "PAdj Intercept", "PAdj Slides", "Blocks/90",
    "Goals/90", "xG/90", "Assists/90", "xA/90", "Key Pass/90",
    "Pass Acc %", "Prog Pass/90", "Dribbles/90", "Dribble %",
    "Save %", "Prev Goals/90", "Exits/90",
    "NP Goals/90", "Shots/90", "Touch Box/90",
    "Cross/90", "Acc Cross %", "Deep Cross/90",
    "Smart Pass/90", "Pass F3rd/90", "Deep Compl/90",
    "Prog Run/90", "Off Duel %",
]

LEAGUE_TIER_LABEL = {
    "T1": "T1 — Elite",
    "T2": "T2 — Strong",
    "T3": "T3 — Mid-Tier",
    "T4": "T4 — Developing",
    "T5": "T5 — Emerging",
}

def clean(v):
    if v is None: return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 2)
    return v

def sheet_to_json(df: pd.DataFrame, name: str):
    cols = [c for c in KEEP if c in df.columns]
    sub  = df[cols].copy()
    records = []
    for row in sub.itertuples(index=False):
        records.append([clean(v) for v in row])
    out = {"cols": cols, "rows": records}
    path = DEST / f"{name}.json"
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    kb = path.stat().st_size / 1024
    print(f"  {name}.json  {len(records):,} rows  {kb:.0f} KB")

print(f"Reading {SRC.name}…")
all_sheets = {}

# Read the main data sheets
for sheet in ["All Players", "GK", "CB", "FB", "DM", "CM", "W", "FW",
              "Priority List", "Elite Picks", "Expiring 2026"]:
    try:
        df = pd.read_excel(SRC, sheet_name=sheet, header=2)
        df.columns = [str(c).strip() for c in df.columns]
        all_sheets[sheet] = df
    except Exception as e:
        print(f"  SKIP {sheet}: {e}")

# Export each sheet
for name, df in all_sheets.items():
    key = name.lower().replace(" ", "_")
    sheet_to_json(df, key)

# Export league summary
try:
    la = pd.read_excel(SRC, sheet_name="League Analysis", header=2)
    la.columns = [str(c).strip() for c in la.columns]
    records = []
    for row in la.itertuples(index=False):
        rec = {}
        for col, val in zip(la.columns, row):
            rec[col] = clean(val)
        records.append(rec)
    with open(DEST / "league_analysis.json", "w") as f:
        json.dump(records, f, separators=(",", ":"))
    print(f"  league_analysis.json  {len(records)} rows")
except Exception as e:
    print(f"  SKIP League Analysis: {e}")

# Export meta (tier counts, position counts)
if "All Players" in all_sheets:
    df = all_sheets["All Players"]
    tier_col  = "Tier"   if "Tier"   in df.columns else None
    pos_col   = "Pos"    if "Pos"    in df.columns else None
    ltier_col = "League Tier" if "League Tier" in df.columns else None

    meta = {
        "total": len(df),
        "tiers": df[tier_col].value_counts().to_dict() if tier_col else {},
        "positions": df[pos_col].value_counts().to_dict() if pos_col else {},
        "league_tiers": df[ltier_col].value_counts().to_dict() if ltier_col else {},
        "league_tier_labels": LEAGUE_TIER_LABEL,
    }
    with open(DEST / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"  meta.json")

print("Done.")
