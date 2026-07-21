"""
build_recruitment_model.py
───────────────────────────
Build the FC Hradec Králové Recruitment Model workbook from the full
Wyscout league universe ("Wyscout Files/*.xlsx").

Distinct from build_lamberts_total.py:
  - "Model Value (€)" is a metrics-derived valuation (regression fit across
    the whole population), not the player's own market value rescaled.
  - Leagues AND clubs get power-ranking tier lists, senior and youth split
    out separately.
  - A dedicated "Undervalued & Not Past Peak" board — the flagship screen.
  - A Czech-First-League-calibrated "Physical / Quick League Fit" score.

Usage:
  python build_recruitment_model.py
  python build_recruitment_model.py --min-minutes-senior 600 --output data/My_Model.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import recruitment_model as rm

ROOT = Path(__file__).parent

# ── Colors ───────────────────────────────────────────────────────────────────
C = {
    "navy": "0D1B2A", "gold": "C9A84C", "header": "154360",
    "white": "FFFFFF", "light": "EBF5FB",
    "elite": "1A5276", "high": "1E8449", "value": "117A65",
    "fair": "626567", "over": "922B21", "noval": "839192",
    "rising": "1E8449", "peak": "1A5276", "past": "922B21", "unknown": "839192",
    "excellent": "1A5276", "good": "1E8449", "moderate": "B7950B", "below": "922B21",
    "tier1": "1A5276", "tier2": "1E8449", "tier3": "117A65",
    "tier4": "626567", "tier5": "839192", "tier6": "884EA0",
}

VALUE_TIER_COLORS = {
    "ELITE VALUE": C["elite"], "HIGH VALUE": C["high"], "VALUE": C["value"],
    "FAIR VALUE": C["fair"], "OVERPRICED": C["over"], "NO MARKET COMP": C["noval"],
}
TRAJECTORY_COLORS = {
    "Rising": C["rising"], "Peak Window": C["peak"], "Past Peak": C["past"], "Unknown": C["unknown"],
}
FIT_COLORS = {
    "Excellent Fit": C["excellent"], "Good Fit": C["good"],
    "Moderate Fit": C["moderate"], "Below Profile": C["below"],
}
BRIGHTON_COLORS = {
    "Prime Target": C["elite"], "Strong Fit": C["high"], "Speculative": C["fair"],
    "Long Shot": C["noval"], "Not A Fit": C["noval"],
}
TIER_COLORS = {1: C["tier1"], 2: C["tier2"], 3: C["tier3"], 4: C["tier4"], 5: C["tier5"], 6: C["tier6"]}

POS_LABELS = {
    "GK": "GOALKEEPER", "CB": "CENTRE-BACK", "FB": "FULL-BACK", "DM": "DEFENSIVE MID",
    "CM": "CENTRAL MID", "AM": "ATTACKING MID", "W": "WINGER", "ST": "STRIKER",
}
POS_ORDER = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]

STAT_MAP = {
    "Goals/90": "Goals per 90", "xG/90": "xG per 90",
    "Assists/90": "Assists per 90", "xA/90": "xA per 90",
    "Prog Pass/90": "Progressive passes per 90", "Prog Run/90": "Progressive runs per 90",
    "Dribbles/90": "Dribbles per 90", "Def Duel %": "Defensive duels won, %",
    "Aerial %": "Aerial duels won, %", "Save %": "Save rate, %",
}

MASTER_COLS = [
    "Player", "Club", "TeamType", "League", "Country", "Tier", "TierLabel",
    "PositionGroup", "Full Position", "AgeYears", "Contract", "_minutes",
    "_mkt_val", "ModelValueEUR", "ValueGapEUR", "ValueRatio", "ValueTier",
    "ProjectedPeakValueEUR", "DevelopmentUpsideEUR", "BrightonScore", "BrightonLabel",
    "TrajectoryTag", "PhysicalLeagueFitScore", "PhysicalLeagueFitLabel",
    "AdjustedCompositeScore", "CompositeRecruitmentScore",
] + list(STAT_MAP.values())

DISPLAY_RENAME = {
    "TierLabel": "Tier Label",
    "PositionGroup": "Pos", "AgeYears": "Age", "_minutes": "Minutes",
    "_mkt_val": "Mkt Val (€)", "ModelValueEUR": "Model Val (€)",
    "ValueGapEUR": "Value Gap (€)", "ValueRatio": "Value Ratio",
    "ValueTier": "Value Tier", "TrajectoryTag": "Trajectory",
    "PhysicalLeagueFitScore": "Physical Fit", "PhysicalLeagueFitLabel": "Physical Fit Label",
    "AdjustedCompositeScore": "Composite Score", "CompositeRecruitmentScore": "League-Relative Score",
    "ProjectedPeakValueEUR": "Peak Val (€)", "DevelopmentUpsideEUR": "Dev. Upside (€)",
    "BrightonScore": "Brighton Score", "BrightonLabel": "Brighton Fit",
} | {v: k for k, v in STAT_MAP.items()}


# ── Master table ─────────────────────────────────────────────────────────────

def build_master(players: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()
    contract_col = next((c for c in ["Contract expires", "ContractExpires"] if c in df.columns), None)
    if contract_col:
        df["Contract"] = pd.to_datetime(df[contract_col], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["Contract"] = None
    df["Full Position"] = df.get("Position", df.get("Pos", ""))

    for c in MASTER_COLS:
        if c not in df.columns:
            df[c] = np.nan

    out = df[MASTER_COLS].copy()
    out["ValueRatio"] = out["ValueRatio"].round(2)
    out["_mkt_val"] = out["_mkt_val"].fillna(0).astype(int)
    out["ModelValueEUR"] = out["ModelValueEUR"].fillna(0).astype(int)
    out["ValueGapEUR"] = out["ValueGapEUR"].fillna(0).astype(int)
    out["AgeYears"] = pd.to_numeric(out["AgeYears"], errors="coerce")
    out["_minutes"] = out["_minutes"].fillna(0).astype(int)
    out["CompositeRecruitmentScore"] = out["CompositeRecruitmentScore"].round(1)
    out["AdjustedCompositeScore"] = out["AdjustedCompositeScore"].round(1)
    out["ProjectedPeakValueEUR"] = out["ProjectedPeakValueEUR"].fillna(0).astype(int)
    out["DevelopmentUpsideEUR"] = out["DevelopmentUpsideEUR"].fillna(0).astype(int)
    out["BrightonScore"] = out["BrightonScore"].round(1)
    for c in STAT_MAP.values():
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    out = out.rename(columns=DISPLAY_RENAME)
    return out


# ── Excel helpers ────────────────────────────────────────────────────────────

def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _border() -> Border:
    thin = Side(style="thin", color="CCCCCC")
    return Border(left=thin, right=thin, top=thin, bottom=thin)


def _autofit(ws) -> None:
    for col_cells in ws.columns:
        try:
            max_len = max(
                len(str(col_cells[0].value or "")),
                *(len(str(c.value or "")) for c in col_cells[1:10]),
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)
        except Exception:
            pass


def _title_block(ws, title: str, subtitle: str, ncols: int) -> None:
    ws.sheet_view.showGridLines = False
    ws.append([title])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(ncols, 10))
    ws["A1"].font = Font(bold=True, color=C["white"], size=13)
    ws["A1"].fill = _fill(C["navy"])
    ws.row_dimensions[1].height = 22

    ws.append([subtitle])
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(ncols, 10))
    ws["A2"].font = Font(italic=True, color=C["gold"], size=9)
    ws["A2"].fill = _fill(C["navy"])
    ws.row_dimensions[2].height = 16


def write_data_sheet(
    ws, title: str, subtitle: str, df: pd.DataFrame,
    color_cols: dict[str, dict[str, str]] | None = None,
) -> None:
    _title_block(ws, title, subtitle, len(df.columns))
    if df.empty:
        ws.append(["No rows matched this view."])
        return

    ws.append(list(df.columns))
    hdr_row = ws.max_row
    for cell in ws[hdr_row]:
        cell.font = Font(bold=True, color=C["white"], size=9)
        cell.fill = _fill(C["header"])
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _border()
    ws.row_dimensions[hdr_row].height = 26

    color_cols = color_cols or {}
    col_idx = {c: i + 1 for i, c in enumerate(df.columns)}

    for i, row_vals in enumerate(df.itertuples(index=False), start=1):
        ws.append(list(row_vals))
        data_row = ws.max_row
        bg = C["light"] if i % 2 == 0 else C["white"]
        for cell in ws[data_row]:
            cell.font = Font(size=9)
            cell.fill = _fill(bg)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = _border()

        for col_name, cmap in color_cols.items():
            idx = col_idx.get(col_name)
            if not idx:
                continue
            cell = ws.cell(data_row, idx)
            hexc = cmap.get(str(cell.value or ""))
            if hexc:
                cell.fill = _fill(hexc)
                cell.font = Font(bold=True, color=C["white"], size=9)

    ws.freeze_panes = f"A{hdr_row + 1}"
    _autofit(ws)


# ── README / Methodology ────────────────────────────────────────────────────

def build_readme(ws, universe: dict, min_minutes_senior: int, min_minutes_youth: int) -> None:
    ws.title = "README"
    ws.sheet_view.showGridLines = False

    players = universe["players"]
    n_players = len(players)
    n_senior = (players["TeamType"] == "Senior").sum()
    n_youth = (players["TeamType"] == "Youth").sum()
    n_leagues = players["League"].nunique()

    ws.append(["FC HRADEC KRÁLOVÉ — RECRUITMENT MODEL"])
    ws.merge_cells("A1:D1")
    ws["A1"].font = Font(bold=True, color=C["white"], size=15)
    ws["A1"].fill = _fill(C["navy"])
    ws.row_dimensions[1].height = 28

    ws.append([f"Waltzing Analytics  ·  {n_players:,} players  ·  {n_leagues} leagues  ·  "
               f"Senior min {min_minutes_senior}′ / Youth min {min_minutes_youth}′"])
    ws.merge_cells("A2:D2")
    ws["A2"].font = Font(italic=True, color=C["gold"], size=10)
    ws["A2"].fill = _fill(C["navy"])
    ws.row_dimensions[2].height = 18

    ws.append([None])
    ws.append([None, "WORKBOOK STRUCTURE"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])

    sheets = [
        ("README", "This guide + full methodology"),
        ("League Power Rankings — Senior", "Every senior league tiered 1 (Elite) to 5 (Lower) and power-ranked within tier"),
        ("League Power Rankings — Youth", "U17/U19 and other youth competitions, ranked separately from senior football"),
        ("Club Power Rankings — Senior", "Every senior club rated and tier-adjusted for the strength of its league"),
        ("Club Power Rankings — Youth", "Youth teams (youth leagues + reserve/academy sides in senior leagues) ranked separately"),
        ("Undervalued & Not Past Peak", "Flagship screen: ELITE/HIGH/VALUE tier AND still Rising or in their Peak Window"),
        ("All Undervalued", "Every ELITE/HIGH/VALUE player regardless of age trajectory"),
        ("Physical League Fits", "Best fits for a quick, physical league, calibrated against the Czech First League"),
        ("Position Boards", "Top undervalued, not-past-peak targets per position"),
        ("Youth Prospects", "Best performers age ≤ 20 in youth leagues/teams — evaluated on output, not market value"),
        ("Brighton Mechanics", "Buy-low/develop/resell targets — modelled on Brighton & Hove Albion's recruitment approach"),
        ("Full Database", "Every player loaded, all columns, for manual filtering"),
    ]
    ws.append([None, "Sheet", "Contents"])
    hdr = ws.max_row
    for col in "BC":
        cell = ws[f"{col}{hdr}"]
        cell.font = Font(bold=True, color=C["white"])
        cell.fill = _fill(C["header"])
    for name, desc in sheets:
        ws.append([None, name, desc])
        ws[f"B{ws.max_row}"].font = Font(bold=True, color=C["navy"])

    ws.append([None])
    ws.append([None, "HOW MODEL VALUE (€) WORKS — READ THIS FIRST"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    for line in rm.VALUE_MODEL_NOTES.strip("\n").split("\n"):
        ws.append([None, line])
        row = ws.max_row
        if line.isupper() or line.startswith("─"):
            ws[f"B{row}"].font = Font(bold=True)

    ws.append([None])
    ws.append([None, "TWO PERFORMANCE SCORES"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    ws.append([None, "League-Relative Score", "Percentile vs the player's own league/position peers only — best read as 'how dominant is he in his current league'"])
    ws[f"B{ws.max_row}"].font = Font(bold=True)
    ws.append([None, "Composite Score", "League-Relative Score scaled down by league strength — cross-league comparable, "
               "used for Model Value, sorting and Youth Prospects, so a standout in a weak league isn't confused with a standout in a strong one"])
    ws[f"B{ws.max_row}"].font = Font(bold=True)

    ws.append([None])
    ws.append([None, "LEAGUE TIERS"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    tier_notes = [
        ("Tier 1 — Elite", "Big-5 pyramid tops: Premier League, Ligue 1, Bundesliga, Serie A, La Liga"),
        ("Tier 2 — Top", "Strong top flights outside the big 5 (Eredivisie, Primeira Liga, MLS, etc.)"),
        ("Tier 3 — Strong", "Solid top flights + big-5 second tiers (Championship, 2. Bundesliga, Ligue 2…)"),
        ("Tier 4 — Developing", "Smaller top flights + most second/third tiers"),
        ("Tier 5 — Lower", "Lower regional/fourth-tier competitions"),
        ("Tier 6 — Youth/Grassroots", "U17/U19/U21 and other youth competitions"),
        ("Fallback tiers", "Leagues absent from the curated pyramid table get a conservative tier from their division depth so nothing is left unranked"),
    ]
    for term, desc in tier_notes:
        ws.append([None, term, desc])
        ws[f"B{ws.max_row}"].font = Font(bold=True)

    ws.append([None])
    ws.append([None, "PEAK-AGE WINDOWS (by position)"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    for pos, (s, e) in rm.PEAK_WINDOWS.items():
        ws.append([None, POS_LABELS.get(pos, pos), f"Rising below {s}  ·  Peak window {s}–{e}  ·  Past Peak above {e}"])
        ws[f"B{ws.max_row}"].font = Font(bold=True)

    ws.append([None])
    ws.append([None, "YOUTH vs SENIOR"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    ws.append([None, "Youth league", "Whole competition is youth (Tier 6 or filename tagged U17/U19/…) — every team in it is Youth"])
    ws[f"B{ws.max_row}"].font = Font(bold=True)
    ws.append([None, "Youth team", "Inside a senior league, reserve/academy sides are flagged by name pattern "
               "(…II, …B, U15–U23, Youth, Junior, Academy, Reserves) so first-team recruitment boards aren't diluted by feeder-team stats"])
    ws[f"B{ws.max_row}"].font = Font(bold=True)

    ws.append([None])
    ws.append([None, "PHYSICAL / QUICK LEAGUE FIT"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    ws.append([None, "Benchmark", f"Czech First League ({rm.BENCHMARK_LEAGUE_FILE}.xlsx) — a fast, duel-heavy, physically demanding top flight"])
    ws[f"B{ws.max_row}"].font = Font(bold=True)
    ws.append([None, "Method", "Every player's duel/aerial/tempo metrics are z-scored against the Czech First "
               "League's own per-position averages (not their own league's) — a high score means they already "
               "produce Czech-top-flight-level physical/tempo numbers"])
    ws[f"B{ws.max_row}"].font = Font(bold=True)

    ws.append([None])
    ws.append([None, "BRIGHTON MECHANICS — BUY LOW, DEVELOP, RESELL"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    for line in rm.BRIGHTON_NOTES.strip("\n").split("\n"):
        ws.append([None, line])
        row = ws.max_row
        if line.isupper() or line.startswith("─"):
            ws[f"B{row}"].font = Font(bold=True)

    ws.column_dimensions["A"].width = 3
    ws.column_dimensions["B"].width = 26
    ws.column_dimensions["C"].width = 95


# ── Sheet builders ───────────────────────────────────────────────────────────

def league_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["PowerRank", "League", "LeagueDisplayName", "Country", "Division",
            "TierLabel", "Tier", "ClubsRated", "AvgClubScore", "TopClub"]
    return df[[c for c in cols if c in df.columns]].rename(columns={
        "LeagueDisplayName": "Competition", "TierLabel": "Tier Label",
        "ClubsRated": "Clubs Rated", "AvgClubScore": "Avg Club Score", "TopClub": "Top Club",
    })


def club_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["PowerRank", "LeagueRank", "Team", "League", "Country", "TierLabel", "Tier",
            "Players", "TotalMinutes", "CompositeRecruitmentScore", "ScoringThreatScore",
            "CreativeProgressionScore", "DefensiveDisruptionScore", "PressingScore",
            "AerialScore", "TierAdjustedScore"]
    return df[[c for c in cols if c in df.columns]].rename(columns={
        "LeagueRank": "League Rank", "TierLabel": "Tier Label", "TotalMinutes": "Minutes",
        "CompositeRecruitmentScore": "Composite", "ScoringThreatScore": "Attacking",
        "CreativeProgressionScore": "Creation", "DefensiveDisruptionScore": "Defending",
        "PressingScore": "Pressing", "AerialScore": "Aerial", "TierAdjustedScore": "Power Score",
    })


def build_undervalued_board(master: pd.DataFrame, not_past_peak_only: bool) -> pd.DataFrame:
    df = master[master["Value Tier"].isin(["ELITE VALUE", "HIGH VALUE", "VALUE"])].copy()
    if not_past_peak_only:
        df = df[df["Trajectory"].isin(["Rising", "Peak Window"])]
    # Sort by absolute value created (€ gap), not raw ratio — a tiny market
    # value denominator can produce a huge but practically meaningless ratio
    # (e.g. €25k -> €400k reads as "16.8x" even though the gap is modest).
    tier_order = {"ELITE VALUE": 0, "HIGH VALUE": 1, "VALUE": 2}
    df["_o"] = df["Value Tier"].map(tier_order)
    df = df.sort_values(["_o", "Value Gap (€)"], ascending=[True, False]).drop(columns="_o")
    return df.reset_index(drop=True)


def build_physical_fit_board(master: pd.DataFrame, top_n: int = 300) -> pd.DataFrame:
    # Outfield only — duel/aerial rate stats for goalkeepers are low-sample
    # and don't speak to "thrives in a quick, physical league" the way they
    # do for outfield players.
    df = master[master["Pos"] != "GK"].sort_values("Physical Fit", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def build_youth_prospects(master: pd.DataFrame, max_age: int = 20, top_n: int = 300) -> pd.DataFrame:
    df = master[(master["TeamType"] == "Youth") & (master["Age"].fillna(99) <= max_age)]
    df = df.sort_values("Composite Score", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def build_brighton_board(master: pd.DataFrame, top_n: int = 400) -> pd.DataFrame:
    df = master[master["Brighton Fit"].isin(["Prime Target", "Strong Fit", "Speculative"])]
    df = df.sort_values("Brighton Score", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def write_position_boards(ws, master: pd.DataFrame, top_n: int = 15) -> None:
    ws.title = "Position Boards"
    ws.sheet_view.showGridLines = False
    ws.append(["POSITION BOARDS — Undervalued, Not-Past-Peak Targets by Position"])
    ncols = 14
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws["A1"].font = Font(bold=True, color=C["white"], size=13)
    ws["A1"].fill = _fill(C["navy"])
    ws.row_dimensions[1].height = 22
    ws.append(["Ranked by Value Ratio  ·  Value Tier ELITE/HIGH/VALUE  ·  Trajectory Rising or Peak Window"])
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    ws["A2"].font = Font(italic=True, color=C["gold"], size=9)
    ws["A2"].fill = _fill(C["navy"])

    board = build_undervalued_board(master, not_past_peak_only=True)
    cols = ["Player", "Club", "League", "Tier Label", "Age", "Trajectory",
            "Mkt Val (€)", "Model Val (€)", "Value Ratio", "Value Tier",
            "Physical Fit", "Composite Score"]

    for pos in POS_ORDER:
        grp = board[board["Pos"] == pos].head(top_n)
        if grp.empty:
            continue
        ws.append([f"  {pos} — {POS_LABELS.get(pos, pos)}"])
        ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=ncols)
        r = ws.max_row
        ws[f"A{r}"].font = Font(bold=True, color=C["white"], size=10)
        ws[f"A{r}"].fill = _fill(C["navy"])
        ws.row_dimensions[r].height = 18

        ws.append(cols)
        hdr = ws.max_row
        for cell in ws[hdr]:
            cell.font = Font(bold=True, color=C["white"], size=9)
            cell.fill = _fill(C["header"])
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        for i, (_, row) in enumerate(grp[cols].iterrows()):
            ws.append(list(row))
            dr = ws.max_row
            bg = C["light"] if i % 2 == 0 else C["white"]
            for cell in ws[dr]:
                cell.font = Font(size=9)
                cell.fill = _fill(bg)
                cell.alignment = Alignment(horizontal="center")
            vt_cell = ws.cell(dr, cols.index("Value Tier") + 1)
            hexc = VALUE_TIER_COLORS.get(str(vt_cell.value or ""))
            if hexc:
                vt_cell.fill = _fill(hexc)
                vt_cell.font = Font(bold=True, color=C["white"], size=9)
        ws.append([None])

    _autofit(ws)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(min_minutes_senior: int, min_minutes_youth: int, leagues: list[str] | None, output: Path) -> None:
    print(f"\n{'='*60}\n  FC Hradec Králové — Recruitment Model Builder\n{'='*60}\n")

    universe = rm.build_recruitment_universe(
        min_minutes_senior=min_minutes_senior,
        min_minutes_youth=min_minutes_youth,
        leagues=leagues,
    )
    master = build_master(universe["players"])
    print(f"\n  → {len(master)} total players in master table")

    wb = Workbook()
    wb.remove(wb.active)

    print("Writing README / Methodology…")
    build_readme(wb.create_sheet("README"), universe, min_minutes_senior, min_minutes_youth)

    print("Writing League Power Rankings…")
    write_data_sheet(
        wb.create_sheet("League Power Rankings - Senior"),
        "LEAGUE POWER RANKINGS — SENIOR",
        "Tiered 1 (Elite) to 5 (Lower) by football-pyramid position  ·  ranked within tier by rated club strength",
        league_rank_columns(universe["league_rankings_senior"]),
        color_cols={"Tier Label": {}},
    )
    write_data_sheet(
        wb.create_sheet("League Power Rankings - Youth"),
        "LEAGUE POWER RANKINGS — YOUTH",
        "Youth/grassroots competitions only, kept separate from senior football",
        league_rank_columns(universe["league_rankings_youth"]),
    )

    print("Writing Club Power Rankings…")
    write_data_sheet(
        wb.create_sheet("Club Power Rankings - Senior"),
        "CLUB POWER RANKINGS — SENIOR",
        "Power Score = minutes-weighted squad composite, tier-adjusted for league strength",
        club_rank_columns(universe["club_rankings_senior"]),
    )
    write_data_sheet(
        wb.create_sheet("Club Power Rankings - Youth"),
        "CLUB POWER RANKINGS — YOUTH",
        "Youth leagues + reserve/academy sides inside senior leagues, ranked separately from first teams",
        club_rank_columns(universe["club_rankings_youth"]),
    )

    print("Writing Undervalued & Not Past Peak…")
    flagship = build_undervalued_board(master, not_past_peak_only=True)
    write_data_sheet(
        wb.create_sheet("Undervalued & Not Past Peak"),
        f"UNDERVALUED & NOT PAST PEAK — {len(flagship)} PLAYERS",
        "Model Value well above listed Market Value AND still Rising or in Peak Window  ·  Sorted by Value Ratio",
        flagship,
        color_cols={"Value Tier": VALUE_TIER_COLORS, "Trajectory": TRAJECTORY_COLORS},
    )

    print("Writing All Undervalued…")
    all_under = build_undervalued_board(master, not_past_peak_only=False)
    write_data_sheet(
        wb.create_sheet("All Undervalued"),
        f"ALL UNDERVALUED — {len(all_under)} PLAYERS",
        "ELITE / HIGH / VALUE tier regardless of age trajectory  ·  Sorted by Value Ratio",
        all_under,
        color_cols={"Value Tier": VALUE_TIER_COLORS, "Trajectory": TRAJECTORY_COLORS},
    )

    print("Writing Physical League Fits…")
    phys = build_physical_fit_board(master)
    write_data_sheet(
        wb.create_sheet("Physical League Fits"),
        f"PHYSICAL / QUICK LEAGUE FITS — Top {len(phys)}",
        f"Calibrated against {rm.BENCHMARK_LEAGUE_FILE} First League duel/aerial/tempo norms by position  ·  Sorted by Physical Fit",
        phys,
        color_cols={"Physical Fit Label": FIT_COLORS, "Value Tier": VALUE_TIER_COLORS},
    )

    print("Writing Position Boards…")
    write_position_boards(wb.create_sheet("Position Boards"), master)

    print("Writing Youth Prospects…")
    youth = build_youth_prospects(master)
    write_data_sheet(
        wb.create_sheet("Youth Prospects"),
        f"YOUTH PROSPECTS — Age ≤ 20 — Top {len(youth)}",
        "Youth leagues/teams evaluated on performance output (market value is unreliable at this level)  ·  Sorted by Composite Score",
        youth,
        color_cols={"Physical Fit Label": FIT_COLORS},
    )

    print("Writing Brighton Mechanics…")
    brighton = build_brighton_board(master)
    write_data_sheet(
        wb.create_sheet("Brighton Mechanics"),
        f"BRIGHTON MECHANICS — BUY LOW, DEVELOP, RESELL — {len(brighton)} PLAYERS",
        "Age Fit + Composite Score + Undervaluation + Trajectory, modelled on Brighton's recruitment approach  ·  Sorted by Brighton Score",
        brighton,
        color_cols={"Brighton Fit": BRIGHTON_COLORS, "Value Tier": VALUE_TIER_COLORS, "Trajectory": TRAJECTORY_COLORS},
    )

    print("Writing Full Database…")
    write_data_sheet(
        wb.create_sheet("Full Database"),
        f"FULL DATABASE — {len(master)} PLAYERS",
        "Every player loaded across the recruitment universe  ·  use column filters to narrow",
        master.sort_values("Composite Score", ascending=False).reset_index(drop=True),
        color_cols={"Value Tier": VALUE_TIER_COLORS, "Trajectory": TRAJECTORY_COLORS, "Physical Fit Label": FIT_COLORS},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {size_mb:.1f} MB → {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the FCHK Recruitment Model workbook from Wyscout data")
    parser.add_argument("--min-minutes-senior", type=int, default=rm.DEFAULT_MIN_MINUTES_SENIOR)
    parser.add_argument("--min-minutes-youth", type=int, default=rm.DEFAULT_MIN_MINUTES_YOUTH)
    parser.add_argument("--leagues", nargs="+", default=None, help="Subset of league file stems. Omit to load ALL.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "FCHK_Recruitment_Model.xlsx")
    args = parser.parse_args()
    run(args.min_minutes_senior, args.min_minutes_youth, args.leagues, args.output)
