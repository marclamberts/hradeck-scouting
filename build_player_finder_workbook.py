"""
Standalone 50-metric Player Finder workbook.

Separate deliverable from FCHK_Recruitment_Model.xlsx: a fast, self-contained
scouting tool built entirely on native Excel formulas (dynamic-array FILTER,
XLOOKUP, conditional formatting, data validation) — no macros, per explicit
user choice, since this sandbox can't compile/verify VBA.
"""
import pickle
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, NamedStyle
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule, IconSetRule, FormulaRule
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter
from openpyxl.comments import Comment
from openpyxl.workbook.defined_name import DefinedName


def define_name(wb, name, ref):
    wb.defined_names[name] = DefinedName(name, attr_text=ref)

# ── Palette (consistent with the rest of this project's deliverables) ──────
NAVY = "0D1B2A"
ACCENT = "1DB876"
ACCENT_DARK = "0E7A4C"
CRITICAL = "C0392B"
AMBER = "B7950B"
LIGHT = "F4F6F5"
WHITE = "FFFFFF"
GREY = "6B7480"
BORDER_C = "D8DEE2"

FONT_TITLE = Font(name="Calibri", size=20, bold=True, color=WHITE)
FONT_SUB = Font(name="Calibri", size=11, italic=True, color="D9E2E8")
FONT_H2 = Font(name="Calibri", size=14, bold=True, color=NAVY)
FONT_BODY = Font(name="Calibri", size=11, color=NAVY)
FONT_BODY_MUTED = Font(name="Calibri", size=10.5, color=GREY)
FONT_HDR = Font(name="Calibri", size=10, bold=True, color=WHITE)
FILL_NAVY = PatternFill("solid", fgColor=NAVY)
FILL_ACCENT = PatternFill("solid", fgColor=ACCENT)
FILL_LIGHT = PatternFill("solid", fgColor=LIGHT)
FILL_WHITE = PatternFill("solid", fgColor=WHITE)
THIN = Side(style="thin", color=BORDER_C)
BORDER_ALL = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def title_block(ws, title, subtitle, ncols, height=(1, 2)):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    c1 = ws.cell(row=1, column=1, value=title)
    c1.font = FONT_TITLE
    c1.fill = FILL_NAVY
    c1.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    c2 = ws.cell(row=2, column=1, value=subtitle)
    c2.font = FONT_SUB
    c2.fill = FILL_NAVY
    c2.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[1].height = 34
    ws.row_dimensions[2].height = 20
    for r in (1, 2):
        for c in range(1, ncols + 1):
            ws.cell(row=r, column=c).fill = FILL_NAVY


# ── 1. Load data ─────────────────────────────────────────────────────────────
with open("/home/user/hradeck-scouting/data/dashboard_cache.pkl", "rb") as f:
    cache = pickle.load(f)
master = cache["master"].copy()

# ── 2. Rank/percentile context, computed against the FULL pool (43,213 senior
#    / 2,693 youth) before we cut down to the workable export subset — so
#    "#234 of 43,213" stays meaningful rather than being re-based on a small
#    export sample. ──────────────────────────────────────────────────────────
def add_rank(df, by, within, rank_col, pool_col):
    grp = df.groupby(within)[by]
    df[rank_col] = grp.rank(method="min", ascending=False)
    df[pool_col] = grp.transform("size")

add_rank(master, "Composite Score", ["TeamType"], "Overall Rank", "Overall Pool")
add_rank(master, "Composite Score", ["TeamType", "Pos"], "Position Rank", "Position Pool")
add_rank(master, "Composite Score", ["TeamType", "League"], "League Rank", "League Pool")
add_rank(master, "Model Val (€)", ["TeamType"], "Value Rank", "Value Pool")

PEAK_WINDOWS = {"GK": (27, 33), "CB": (26, 30), "FB": (24, 28), "DM": (25, 29),
                "CM": (24, 28), "AM": (23, 27), "W": (22, 27), "ST": (24, 28)}
master["Peak Window Start"] = master["Pos"].map(lambda p: PEAK_WINDOWS.get(p, (24, 28))[0])
master["Peak Window End"] = master["Pos"].map(lambda p: PEAK_WINDOWS.get(p, (24, 28))[1])
master["Value per 90 Min (€)"] = np.where(
    master["Minutes"] > 0, master["Model Val (€)"] / (master["Minutes"] / 90.0), np.nan
).round(0)

# ── 3. Scope to a fast, workable export subset ──────────────────────────────
senior_pool = master[(master["TeamType"] == "Senior") & (master["In Scope"] == True)].copy()  # noqa: E712
senior_pool = senior_pool.sort_values("Composite Score", ascending=False).head(6000)

youth_pool = master[(master["TeamType"] == "Youth") & (master["Age"].fillna(99) <= 20)].copy()
youth_pool = youth_pool.sort_values("Composite Score", ascending=False).head(500)

export = pd.concat([senior_pool, youth_pool], ignore_index=True)
print(f"Export rows: {len(export)} ({len(senior_pool)} senior in-scope + {len(youth_pool)} youth prospects)")

# ── 4. The 50 metrics — name, category, one-line definition, ↑/↓ direction ──
METRICS = [
    # (column name in export df, category, definition, direction)
    ("Composite Score", "Rating", "Cross-league adjusted quality score (0-100), the model's headline number.", "Higher = better"),
    ("League-Relative Score", "Rating", "Percentile purely within the player's own league/position peers — how dominant they are at home.", "Higher = better"),
    ("Rating Low", "Rating", "Lower bound of the confidence band around Composite Score, widened for low-minute samples.", "Context"),
    ("Rating High", "Rating", "Upper bound of the confidence band around Composite Score.", "Context"),
    ("Rating Band (±)", "Rating", "Half-width of the confidence band — bigger means less certain, usually low minutes played.", "Lower = more certain"),
    ("Role Archetype", "Rating", "The statistical playing-style sub-type this player was classified into (e.g. Poacher, Ball-Playing CB).", "Context"),
    ("Role Fit", "Rating", "Percentile fit to that specific role archetype's metric weights.", "Higher = better"),
    ("Mkt Val (€)", "Valuation", "Listed market value on file (Wyscout/Transfermarkt-sourced), used only to calibrate the model.", "Context"),
    ("Model Val (€)", "Valuation", "What the player's own metrics say they should be worth — the model's price, not a copy of the market.", "Higher = more talent"),
    ("Value Gap (€)", "Valuation", "Model Val minus Market Val — how much the market is underpricing (or overpricing) this player.", "Higher = bigger bargain"),
    ("Value Ratio", "Valuation", "Model Val ÷ Market Val — above 1.0 means undervalued.", "Higher = better value"),
    ("Value Tier", "Valuation", "Elite / High / Value / Fair / Overpriced / No Market Comp, from Value Ratio.", "Context"),
    ("Yrs to Expiry", "Valuation", "Years remaining on the player's contract.", "Lower = more urgency to buy"),
    ("Contract Discount", "Valuation", "Multiplier (0.35-1.0) applied to Model Val for an expiring contract.", "Context"),
    ("Effective Val (€)", "Valuation", "Model Val × Contract Discount — the realistic fee given time left on contract.", "Higher = pricier"),
    ("Peak Val (€)", "Valuation", "Projected Model Val once the player reaches their position's peak age window.", "Higher = more upside"),
    ("Dev. Upside (€)", "Valuation", "Peak Val minus Model Val — the resale/development play, Brighton-style.", "Higher = bigger resale play"),
    ("Trajectory", "Age & Trajectory", "Rising / Peak Window / Past Peak / Unknown, from age vs. the position's peak-age window.", "Context"),
    ("Physical Fit", "Fit & Adaptability", "0-100 adaptability score: this player's duel/aerial/tempo profile vs. the Czech First League's own norms.", "Higher = better fit"),
    ("Physical Fit Label", "Fit & Adaptability", "Excellent / Good / Moderate / Below Profile, from Physical Fit.", "Context"),
    ("Brighton Score", "Recruitment Mechanics", "Buy-low/develop/resell composite — age, upside, and value blended Brighton & Hove Albion-style.", "Higher = better buy-low candidate"),
    ("Brighton Fit", "Recruitment Mechanics", "Prime Target / Strong Fit / Speculative / Long Shot / Not A Fit, from Brighton Score.", "Context"),
    ("Realistic Source", "Recruitment Mechanics", "TRUE if this league tier (3-6) is one the club could plausibly buy from.", "Context"),
    ("Within Budget", "Recruitment Mechanics", "TRUE if Model Val sits under the club's fee ceiling.", "Context"),
    ("In Scope", "Recruitment Mechanics", "TRUE if both Realistic Source and Within Budget hold — the default recruitment pool.", "Context"),
    ("SL: Like-for-Like", "Recruitment Mechanics", "Shortlist score weighted for replacing a starter with a comparable-or-better player.", "Higher = better fit for this need"),
    ("SL: Emergency Depth", "Recruitment Mechanics", "Shortlist score weighted for cheap, available-now cover.", "Higher = better fit for this need"),
    ("SL: Resale Play", "Recruitment Mechanics", "Shortlist score weighted for buy-young-develop-sell trajectory.", "Higher = better fit for this need"),
    ("SL: Balanced", "Recruitment Mechanics", "Shortlist score with an even blend across quality/fit/value/trajectory.", "Higher = better all-round"),
    ("Goals/90", "Per-90 Output", "Goals scored per 90 minutes played.", "Higher = better"),
    ("xG/90", "Per-90 Output", "Expected goals per 90 minutes — shot quality/volume, not just finishing luck.", "Higher = better"),
    ("Assists/90", "Per-90 Output", "Assists per 90 minutes played.", "Higher = better"),
    ("xA/90", "Per-90 Output", "Expected assists per 90 minutes — chance creation quality/volume.", "Higher = better"),
    ("Prog Pass/90", "Per-90 Output", "Progressive passes per 90 — passes that meaningfully advance the ball upfield.", "Higher = better"),
    ("Prog Run/90", "Per-90 Output", "Progressive carries per 90 — ball-carrying that advances play.", "Higher = better"),
    ("Dribbles/90", "Per-90 Output", "Dribble attempts per 90 minutes.", "Higher = better (context-dependent)"),
    ("Def Duel %", "Per-90 Output", "Defensive duel win rate.", "Higher = better"),
    ("Aerial %", "Per-90 Output", "Aerial duel win rate.", "Higher = better"),
    ("Save %", "Per-90 Output", "Save rate (goalkeepers only).", "Higher = better"),
    ("Overall Rank", "Rank & Context", "Rank by Composite Score across the entire senior (or youth) pool.", "Lower = better"),
    ("Overall Pool", "Rank & Context", "Size of that pool — the denominator for Overall Rank.", "Context"),
    ("Position Rank", "Rank & Context", "Rank by Composite Score within this player's own position group.", "Lower = better"),
    ("Position Pool", "Rank & Context", "Size of that position's pool.", "Context"),
    ("League Rank", "Rank & Context", "Rank by Composite Score within this player's own league.", "Lower = better"),
    ("League Pool", "Rank & Context", "Size of that league's pool.", "Context"),
    ("Value Rank", "Rank & Context", "Rank by Model Val across the entire senior (or youth) pool.", "Lower = higher-valued"),
    ("Value Pool", "Rank & Context", "Size of that pool.", "Context"),
    ("Peak Window Start", "Age & Trajectory", "Age this position group typically enters its performance peak.", "Context"),
    ("Peak Window End", "Age & Trajectory", "Age this position group typically exits its performance peak.", "Context"),
    ("Value per 90 Min (€)", "Valuation", "Model Val ÷ (Minutes ÷ 90) — value density; flags high-value players on a thin sample.", "Higher = more value per minute of evidence"),
]
assert len(METRICS) == 50, f"Expected exactly 50 metrics, got {len(METRICS)}"

CONTEXT_COLS = ["Player", "Club", "League", "Country", "Tier Label", "Pos", "Age", "Minutes", "Contract", "TeamType"]
METRIC_COLS = [m[0] for m in METRICS]
ALL_COLS = CONTEXT_COLS + METRIC_COLS

for c in ALL_COLS:
    if c not in export.columns:
        raise SystemExit(f"Missing expected column: {c}")

export = export[ALL_COLS].copy()
for boolcol in ["Realistic Source", "Within Budget", "In Scope"]:
    export[boolcol] = export[boolcol].astype("boolean").fillna(False)

print("Data prepared:", export.shape)

# ── 5. Workbook shell ────────────────────────────────────────────────────────
wb = Workbook()
wb.remove(wb.active)

ws_start = wb.create_sheet("Start Here")
ws_glossary = wb.create_sheet("Metric Glossary")
ws_finder = wb.create_sheet("Player Finder")
ws_leader = wb.create_sheet("Leaderboards")
ws_data = wb.create_sheet("Master Data")
ws_lists = wb.create_sheet("Lists")
ws_lists.sheet_state = "hidden"

# ══════════════════════════════════════════════════════════════════════════
# MASTER DATA
# ══════════════════════════════════════════════════════════════════════════
ws_data.append(ALL_COLS)
for cell in ws_data[1]:
    cell.font = FONT_HDR
    cell.fill = FILL_NAVY
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
ws_data.row_dimensions[1].height = 32

for row in export.itertuples(index=False):
    vals = []
    for v in row:
        if isinstance(v, float) and np.isnan(v):
            vals.append(None)
        elif pd.isna(v):
            vals.append(None)
        else:
            vals.append(v)
    ws_data.append(vals)

n_rows = len(export) + 1
n_cols = len(ALL_COLS)
last_col_letter = get_column_letter(n_cols)
tbl = Table(displayName="PlayerData", ref=f"A1:{last_col_letter}{n_rows}")
tbl.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showFirstColumn=False)
ws_data.add_table(tbl)
ws_data.freeze_panes = "A2"

# column widths + number formats
COL_FMT = {
    "Mkt Val (€)": '#,##0 "€"', "Model Val (€)": '#,##0 "€"', "Value Gap (€)": '#,##0 "€"',
    "Effective Val (€)": '#,##0 "€"', "Peak Val (€)": '#,##0 "€"', "Dev. Upside (€)": '#,##0 "€"',
    "Value per 90 Min (€)": '#,##0 "€"', "Value Ratio": "0.00", "Contract Discount": "0.00",
    "Composite Score": "0.0", "League-Relative Score": "0.0", "Rating Low": "0.0", "Rating High": "0.0",
    "Rating Band (±)": "0.0", "Role Fit": "0.0", "Physical Fit": "0.0", "Brighton Score": "0.0",
    "SL: Like-for-Like": "0.0", "SL: Emergency Depth": "0.0", "SL: Resale Play": "0.0", "SL: Balanced": "0.0",
    "Goals/90": "0.00", "xG/90": "0.00", "Assists/90": "0.00", "xA/90": "0.00", "Prog Pass/90": "0.00",
    "Prog Run/90": "0.00", "Dribbles/90": "0.00", "Def Duel %": "0.0", "Aerial %": "0.0", "Save %": "0.0",
    "Age": "0", "Minutes": "#,##0", "Yrs to Expiry": "0.00",
}
for i, col in enumerate(ALL_COLS, start=1):
    letter = get_column_letter(i)
    width = 12
    if col == "Player": width = 20
    elif col in ("Club", "League"): width = 18
    elif col in ("Role Archetype", "Contract", "Value Tier", "Physical Fit Label", "Brighton Fit"): width = 16
    ws_data.column_dimensions[letter].width = width
    if col in COL_FMT:
        for r in range(2, n_rows + 1):
            ws_data.cell(row=r, column=i).number_format = COL_FMT[col]

col_idx = {c: i + 1 for i, c in enumerate(ALL_COLS)}


def col_range(name, start=2, end=n_rows):
    i = col_idx[name]
    return f"{get_column_letter(i)}{start}:{get_column_letter(i)}{end}"


# conditional formatting on Master Data
ws_data.conditional_formatting.add(
    col_range("Composite Score"),
    ColorScaleRule(start_type="min", start_color="F2A6A0", mid_type="percentile", mid_value=50, mid_color="FCE9A8", end_type="max", end_color="8FD9AE"),
)
ws_data.conditional_formatting.add(
    col_range("Value Ratio"),
    DataBarRule(start_type="min", start_value=0, end_type="max", end_value=8, color=ACCENT_DARK),
)
ws_data.conditional_formatting.add(
    col_range("Physical Fit"),
    ColorScaleRule(start_type="min", start_color="F2A6A0", mid_type="percentile", mid_value=50, mid_color="FCE9A8", end_type="max", end_color="8FD9AE"),
)
_vt_letter = get_column_letter(col_idx["Value Tier"])
ws_data.conditional_formatting.add(
    col_range("Value Tier"),
    FormulaRule(formula=[f'${_vt_letter}2="OVERPRICED"'], fill=PatternFill("solid", fgColor="F8D7D3")),
)

print("Master Data sheet built.")

# ══════════════════════════════════════════════════════════════════════════
# LISTS (hidden helper sheet for data-validation dropdowns)
# ══════════════════════════════════════════════════════════════════════════
pos_list = ["All"] + sorted(export["Pos"].dropna().unique().tolist())
tier_list = ["All"] + sorted(export["Value Tier"].dropna().unique().tolist())
traj_list = ["All"] + sorted(export["Trajectory"].dropna().unique().tolist())
brighton_list = ["All"] + sorted(export["Brighton Fit"].dropna().unique().tolist())

ws_lists["A1"] = "Position"; ws_lists["B1"] = "ValueTier"; ws_lists["C1"] = "Trajectory"; ws_lists["D1"] = "BrightonFit"
for i, v in enumerate(pos_list, start=2): ws_lists.cell(row=i, column=1, value=v)
for i, v in enumerate(tier_list, start=2): ws_lists.cell(row=i, column=2, value=v)
for i, v in enumerate(traj_list, start=2): ws_lists.cell(row=i, column=3, value=v)
for i, v in enumerate(brighton_list, start=2): ws_lists.cell(row=i, column=4, value=v)

define_name(wb, "PosList", f"'Lists'!$A$2:$A${len(pos_list)+1}")
define_name(wb, "ValueTierList", f"'Lists'!$B$2:$B${len(tier_list)+1}")
define_name(wb, "TrajectoryList", f"'Lists'!$C$2:$C${len(traj_list)+1}")
define_name(wb, "BrightonFitList", f"'Lists'!$D$2:$D${len(brighton_list)+1}")

# ══════════════════════════════════════════════════════════════════════════
# PLAYER FINDER
# ══════════════════════════════════════════════════════════════════════════
title_block(ws_finder, "Player Finder", "Type to search, pick filters, results update live — no macros needed (Excel 2021 / Microsoft 365 required for FILTER)", 16)

ws_finder.column_dimensions["A"].width = 2
ws_finder.column_dimensions["B"].width = 26
ws_finder.column_dimensions["C"].width = 16
for col in "DEFGHIJKLMNOP":
    ws_finder.column_dimensions[col].width = 13

criteria = [
    ("Player / club name contains", "SearchName", ""),
    ("Position", "FilterPos", "All"),
    ("Value tier", "FilterValueTier", "All"),
    ("Trajectory", "FilterTrajectory", "All"),
    ("Min age", "MinAge", ""),
    ("Max age", "MaxAge", ""),
    ("Min composite score", "MinComposite", ""),
    ("Min minutes played", "MinMinutes", ""),
]
r = 4
ws_finder.cell(row=r, column=2, value="SEARCH & FILTER").font = FONT_H2
r += 1
name_to_cell = {}
for label, name, default in criteria:
    ws_finder.cell(row=r, column=2, value=label).font = FONT_BODY
    cell = ws_finder.cell(row=r, column=3, value=default)
    cell.fill = FILL_LIGHT
    cell.border = BORDER_ALL
    cell.font = Font(name="Calibri", size=11, bold=True, color=NAVY)
    cell.alignment = Alignment(horizontal="center")
    addr = f"'Player Finder'!$C${r}"
    define_name(wb, name, addr)
    name_to_cell[name] = f"$C${r}"
    r += 1

dv_pos = DataValidation(type="list", formula1="=PosList", allow_blank=False)
dv_tier = DataValidation(type="list", formula1="=ValueTierList", allow_blank=False)
dv_traj = DataValidation(type="list", formula1="=TrajectoryList", allow_blank=False)
for dv in (dv_pos, dv_tier, dv_traj):
    ws_finder.add_data_validation(dv)
dv_pos.add(ws_finder[name_to_cell["FilterPos"]])
dv_tier.add(ws_finder[name_to_cell["FilterValueTier"]])
dv_traj.add(ws_finder[name_to_cell["FilterTrajectory"]])

match_row = r + 1
ws_finder.cell(row=match_row, column=2, value="Matches found").font = Font(name="Calibri", size=11, bold=True, color=ACCENT_DARK)
include_expr = (
    '((SearchName="")+(ISNUMBER(SEARCH(SearchName,PlayerData[Player]))+ISNUMBER(SEARCH(SearchName,PlayerData[Club]))))'
    '*((FilterPos="All")+(PlayerData[Pos]=FilterPos))'
    '*((FilterValueTier="All")+(PlayerData[Value Tier]=FilterValueTier))'
    '*((FilterTrajectory="All")+(PlayerData[Trajectory]=FilterTrajectory))'
    '*((MinAge="")+(PlayerData[Age]>=MinAge))'
    '*((MaxAge="")+(PlayerData[Age]<=MaxAge))'
    '*((MinComposite="")+(PlayerData[Composite Score]>=MinComposite))'
    '*((MinMinutes="")+(PlayerData[Minutes]>=MinMinutes))'
)
ws_finder.cell(row=match_row, column=3, value=f"=SUMPRODUCT({include_expr})").font = Font(name="Calibri", size=11, bold=True, color=ACCENT_DARK)

result_cols = ["Player", "Club", "League", "Pos", "Age", "Trajectory", "Value Tier", "Mkt Val (€)",
               "Model Val (€)", "Value Ratio", "Composite Score", "Role Fit", "Role Archetype", "Physical Fit", "Brighton Score"]
header_row = match_row + 2
for i, name in enumerate(result_cols):
    cell = ws_finder.cell(row=header_row, column=2 + i, value=name)
    cell.font = FONT_HDR
    cell.fill = FILL_NAVY
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
ws_finder.row_dimensions[header_row].height = 30

array_expr = "HSTACK(" + ",".join(f"PlayerData[{c}]" for c in result_cols) + ")"
spill_row = header_row + 1
formula = f'=IFERROR(FILTER({array_expr},{include_expr}),"No matches — widen your filters")'
ws_finder.cell(row=spill_row, column=2, value=formula)

spill_end_row = spill_row + 3000
comp_letter = get_column_letter(2 + result_cols.index("Composite Score"))
ratio_letter = get_column_letter(2 + result_cols.index("Value Ratio"))
rolefit_letter = get_column_letter(2 + result_cols.index("Role Fit"))
ws_finder.conditional_formatting.add(
    f"{comp_letter}{spill_row}:{comp_letter}{spill_end_row}",
    ColorScaleRule(start_type="num", start_value=0, start_color="F2A6A0", mid_type="num", mid_value=50, mid_color="FCE9A8", end_type="num", end_value=100, end_color="8FD9AE"),
)
ws_finder.conditional_formatting.add(
    f"{ratio_letter}{spill_row}:{ratio_letter}{spill_end_row}",
    DataBarRule(start_type="num", start_value=0, end_type="num", end_value=8, color=ACCENT_DARK),
)
ws_finder.conditional_formatting.add(
    f"{rolefit_letter}{spill_row}:{rolefit_letter}{spill_end_row}",
    DataBarRule(start_type="num", start_value=0, end_type="num", end_value=100, color="5B8DEF"),
)
for i in range(len(result_cols)):
    ws_finder.column_dimensions[get_column_letter(2 + i)].width = 14
ws_finder.column_dimensions["B"].width = 20
ws_finder.freeze_panes = ws_finder.cell(row=spill_row, column=2).coordinate

print("Player Finder sheet built.")

# ══════════════════════════════════════════════════════════════════════════
# METRIC GLOSSARY
# ══════════════════════════════════════════════════════════════════════════
title_block(ws_glossary, "Metric Glossary", "All 50 metrics in this workbook — what each one means and which direction is good", 5)
ws_glossary.column_dimensions["A"].width = 3
ws_glossary.column_dimensions["B"].width = 24
ws_glossary.column_dimensions["C"].width = 20
ws_glossary.column_dimensions["D"].width = 78
ws_glossary.column_dimensions["E"].width = 26

hdr = ["#", "Metric", "Category", "Definition", "Direction"]
ws_glossary.append([None])
ws_glossary.append([None])
hr = 4
for i, h in enumerate(hdr, start=1):
    c = ws_glossary.cell(row=hr, column=i, value=h)
    c.font = FONT_HDR
    c.fill = FILL_NAVY
    c.alignment = Alignment(horizontal="center", vertical="center")
ws_glossary.freeze_panes = f"A{hr+1}"

CATEGORY_COLORS = {
    "Rating": "E8F6EE", "Valuation": "FFF6E0", "Age & Trajectory": "EAF1FB",
    "Fit & Adaptability": "F1E9FB", "Recruitment Mechanics": "E0F7F4",
    "Per-90 Output": "F3F3F3", "Rank & Context": "FBEAEA",
}
row = hr + 1
for i, (name, cat, definition, direction) in enumerate(METRICS, start=1):
    ws_glossary.cell(row=row, column=1, value=i).alignment = Alignment(horizontal="center")
    ws_glossary.cell(row=row, column=2, value=name).font = Font(bold=True, size=10.5, color=NAVY)
    ws_glossary.cell(row=row, column=3, value=cat)
    ws_glossary.cell(row=row, column=4, value=definition).alignment = Alignment(wrap_text=True, vertical="center")
    ws_glossary.cell(row=row, column=5, value=direction)
    fill = PatternFill("solid", fgColor=CATEGORY_COLORS.get(cat, "FFFFFF"))
    for c in range(1, 6):
        ws_glossary.cell(row=row, column=c).fill = fill
        ws_glossary.cell(row=row, column=c).border = BORDER_ALL
        if c != 4:
            ws_glossary.cell(row=row, column=c).font = Font(size=10.5, color=NAVY)
        else:
            ws_glossary.cell(row=row, column=c).font = Font(size=10, color=GREY)
    ws_glossary.row_dimensions[row].height = 30
    row += 1

print("Metric Glossary sheet built.")

# ══════════════════════════════════════════════════════════════════════════
# LEADERBOARDS
# ══════════════════════════════════════════════════════════════════════════
title_block(ws_leader, "Leaderboards", "Four pre-built views into the same 6,500-player pool — live formulas, always current", 8)
for col, w in zip("ABCDEFGH", [2, 20, 16, 10, 10, 12, 12, 12]):
    ws_leader.column_dimensions[col].width = w

panels = [
    ("Top 15 — Highest Composite Score", "Composite Score", True, None),
    ("Top 15 — Biggest Value Gap (bargains)", "Value Gap (€)", True, None),
    ("Top 15 — Brighton Buy-Low Targets", "Brighton Score", True, "Brighton Fit"),
    ("Top 15 — Best Physical / Czech-League Fit", "Physical Fit", True, None),
]
sub_cols = ["Player", "Club", "Pos", "Age", "Value"]
panel_positions = [(4, 1), (4, 5), (22, 1), (22, 5)]
for (title, sort_col, desc, extra_filter), (prow, pcol) in zip(panels, panel_positions):
    ws_leader.cell(row=prow, column=pcol, value=title).font = FONT_H2
    hrow = prow + 1
    for i, h in enumerate(sub_cols):
        c = ws_leader.cell(row=hrow, column=pcol + i, value=h)
        c.font = FONT_HDR
        c.fill = FILL_ACCENT if not desc else FILL_NAVY
        c.fill = FILL_NAVY
    sorted_df = export.sort_values(sort_col, ascending=not desc)
    if extra_filter:
        sorted_df = sorted_df[sorted_df[extra_filter].isin(["Prime Target", "Strong Fit"])]
    sorted_df = sorted_df.head(15)
    for j, (_, prow_data) in enumerate(sorted_df.iterrows()):
        rr = hrow + 1 + j
        ws_leader.cell(row=rr, column=pcol, value=prow_data["Player"])
        ws_leader.cell(row=rr, column=pcol + 1, value=prow_data["Club"])
        ws_leader.cell(row=rr, column=pcol + 2, value=prow_data["Pos"])
        ws_leader.cell(row=rr, column=pcol + 3, value=int(prow_data["Age"]) if pd.notna(prow_data["Age"]) else None)
        val = prow_data[sort_col]
        vcell = ws_leader.cell(row=rr, column=pcol + 4, value=round(float(val), 1) if pd.notna(val) else None)
        if "€" in sort_col:
            vcell.number_format = '#,##0 "€"'
        fill = FILL_LIGHT if j % 2 else FILL_WHITE
        for cc in range(pcol, pcol + 5):
            ws_leader.cell(row=rr, column=cc).fill = fill
            ws_leader.cell(row=rr, column=cc).border = BORDER_ALL
            ws_leader.cell(row=rr, column=cc).font = Font(size=10.5, color=NAVY)

print("Leaderboards sheet built.")

# ══════════════════════════════════════════════════════════════════════════
# START HERE
# ══════════════════════════════════════════════════════════════════════════
title_block(ws_start, "FCHK Player Finder", "50-metric scouting workbook · built on the Waltzing Analytics Model", 8)
ws_start.column_dimensions["A"].width = 3
ws_start.column_dimensions["B"].width = 90
for col in "CDEFGH":
    ws_start.column_dimensions[col].width = 12

lines = [
    ("h", "What this is"),
    ("p", "A standalone, fast, formula-only scouting workbook covering 6,500 players (6,000 realistic-source/in-budget "
          "senior players plus 500 top youth prospects), scored on exactly 50 model metrics — everything from cross-league "
          "rating and EUR valuation to Brighton-style buy-low fit and per-90 output."),
    ("p", "It reads from the same recruitment engine as the interactive HTML dashboard and Excel workbook already built for "
          "FC Hradec Králové, cut down to a single lightweight file with no external dependencies."),
    ("s", ""),
    ("h", "How to use it"),
    ("p", "1.  Open the Player Finder sheet. Type a name/club fragment and/or set the dropdown and numeric filters — "
          "results update live below as you type or change any cell."),
    ("p", "2.  Open Leaderboards for four ready-made Top-15 views (best overall, biggest bargains, best Brighton-style "
          "buy-low targets, best Czech-league physical fit)."),
    ("p", "3.  Open Master Data to browse, sort, or filter the full 6,500-row table directly — it's a proper Excel Table "
          "(named PlayerData), so PivotTables, slicers, and your own formulas all work against it immediately."),
    ("p", "4.  Open Metric Glossary any time you're not sure what a column means or which direction is good."),
    ("s", ""),
    ("h", "Requirements"),
    ("p", "Player Finder's live search uses FILTER() and HSTACK() — dynamic-array functions available in Excel 2021, "
          "Microsoft 365, and Excel for the web. In older Excel versions the Player Finder formulas will show a #NAME? "
          "error; Master Data and Leaderboards are static/compatible with any Excel version, so sort/filter there instead."),
    ("s", ""),
    ("h", "A note on the numbers"),
    ("p", "Model Value is not a copy of the transfer market — it's a regression fit once across the whole player pool "
          "(performance + age + league strength + minutes), then applied to each player's own metrics. Composite Score is "
          "cross-league adjusted, so a 70 in a Tier 5 league and a 70 in a Tier 1 league reflect genuinely comparable "
          "quality, not the same raw dominance. Full methodology: WAM_ARCHITECTURE.md and the Metric Glossary sheet."),
]
row = 4
for kind, text in lines:
    if kind == "h":
        c = ws_start.cell(row=row, column=2, value=text)
        c.font = FONT_H2
        row += 1
    elif kind == "p":
        c = ws_start.cell(row=row, column=2, value=text)
        c.font = FONT_BODY
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws_start.row_dimensions[row].height = 48
        row += 1
    else:
        row += 1

nav_row = row + 1
ws_start.cell(row=nav_row, column=2, value="Jump to:").font = FONT_H2
nav_row += 1
for sheet_name in ["Player Finder", "Leaderboards", "Master Data", "Metric Glossary"]:
    c = ws_start.cell(row=nav_row, column=2, value=f"→ {sheet_name}")
    c.font = Font(color="0563C1", underline="single", size=11.5)
    c.hyperlink = f"#'{sheet_name}'!A1"
    nav_row += 1

wb.active = 0
for s in wb.worksheets:
    s.sheet_view.showGridLines = False

OUT = "/home/user/hradeck-scouting/reports/FCHK_Player_Finder.xlsx"
wb.save(OUT)
print("Saved:", OUT)
