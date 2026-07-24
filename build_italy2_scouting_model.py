"""
build_italy2_scouting_model.py
───────────────────────────────
Scouting model for data/Wyscout DB/Italy II.xlsx (Serie B).

Produces, per player:
  Scouting Score      — 0-100, position-relative composite of underlying
                         per-90 metrics (reuses wyscout_model.py's blueprint
                         system: ScoringThreat, CreativeProgression,
                         DefensiveDisruption, Pressing, BallSecurity,
                         ExpectedThreat, ASA_GoalsAdded, weighted by role).
  Uncertainty Score    — 0-100, how much sample-size noise sits behind the
                         Scouting Score. Built from minutes played using a
                         sqrt scaling (rate-stat standard error shrinks with
                         1/sqrt(n), so uncertainty is modelled the same way).
  Label                — Scouting tier + Confidence tier + article storyline
                         tag (Proven Performer / Breakout Watch / etc.)
  Position Rank        — rank within the player's position group
  Overall Rank         — rank across the whole Italy II file

Methodology is fully documented on the "Methodology" sheet of the output
workbook so the numbers can be defended/explained in an article.

Usage
─────
  python build_italy2_scouting_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from wyscout_model import (
    SCORE_BLUEPRINTS,
    COMPOSITE_WEIGHTS,
    POSITION_RELEVANCE,
    WYSCOUT_POSITION_MAP,
    _z_to_percentile,
)

SRC_FILE = ROOT / "data" / "Wyscout DB" / "Italy II.xlsx"
OUT_FILE = ROOT / "data" / "Italy II - Scouting Model.xlsx"

LEAGUE_LABEL = "Italy II (Serie B)"

# Minutes threshold below which a player is excluded from the group's mean/
# stdev baseline (small samples would distort the baseline for everyone),
# but the player still receives a score computed against that baseline.
STABLE_MINUTES = 630          # ≈ 7 full matches
# Minutes benchmark treated as "full, reliable sample" for the Uncertainty Score.
RELIABLE_MINUTES_BENCHMARK = 2700   # ≈ 30 full matches / a season as starter

SCORE_TIERS = [
    (90, "Elite"),
    (75, "Very Good"),
    (55, "Good"),
    (35, "Average"),
    (15, "Below Average"),
    (-1, "Fringe"),
]

CONFIDENCE_TIERS = [
    (25, "High Confidence"),
    (50, "Medium Confidence"),
    (75, "Low Confidence"),
    (101, "Very Low Confidence"),
]

# ── Striker archetype blueprints ────────────────────────────────────────────────
# Two contrasting axes computed for the ST (CF) role group only. A player isn't
# labelled "dynamic" just because he is good in isolation — he has to score high
# on ball-carrying/movement metrics *relative to* his aerial/target-man profile.
# This makes "dynamic striker" a falsifiable, defensible definition rather than a
# vibe: a Dynamic Runner is a striker whose game is built on Mobility, a Target
# Man is one built on Aerial/hold-up presence, a Complete Forward is strong on
# both, a Limited Profile is below role average on both.
DYNAMIC_MOBILITY_BLUEPRINT: list[tuple[str, float]] = [
    ("Progressive runs per 90",     3.0),   # ball-carrying into dangerous zones
    ("Accelerations per 90",        2.5),   # explosive off-the-ball movement
    ("Dribbles per 90",             2.0),   # take-on frequency
    ("Successful dribbles, %",      1.5),   # take-on quality
    ("Offensive duels per 90",      1.0),   # 1v1 involvement
    ("Offensive duels won, %",      1.0),   # 1v1 quality
    ("Fouls suffered per 90",       0.5),   # dribbling into contact / drawing fouls
    ("xG per 90",                   1.5),   # still needs to convert the movement into threat
    ("Non-penalty goals per 90",    1.0),
]

AERIAL_TARGET_BLUEPRINT: list[tuple[str, float]] = [
    ("Aerial duels won, %",         3.0),
    ("Aerial duels per 90",         2.0),
    ("Head goals per 90",           2.0),
    ("Received long passes per 90", 2.0),   # target for direct/long-ball supply
    ("Touches in box per 90",       1.0),
]

ARCHETYPE_TIER_CUTOFF = 50   # percentile split, within the ST group, for each axis


def _archetype(row: pd.Series) -> str:
    mobile = row["Dynamic Mobility Score"] >= ARCHETYPE_TIER_CUTOFF
    aerial = row["Aerial/Target Score"] >= ARCHETYPE_TIER_CUTOFF
    if mobile and aerial:
        return "Complete Forward"
    if mobile and not aerial:
        return "Dynamic Runner"
    if aerial and not mobile:
        return "Target Man"
    return "Limited Profile"


# ── Load ───────────────────────────────────────────────────────────────────────

def load_italy_ii() -> pd.DataFrame:
    df = pd.read_excel(SRC_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    df["Position"] = df["Position"].astype(str).str.split(",").str[0].str.strip()
    df["PositionGroup"] = df["Position"].map(WYSCOUT_POSITION_MAP).fillna("Other")
    df = df.loc[df["PositionGroup"] != "Other"].reset_index(drop=True)

    skip = {"Player", "Team", "Position", "PositionGroup", "Birth country",
            "Passport country", "Foot", "On loan", "Team within selected timeframe",
            "Contract expires"}
    for col in df.columns:
        if col not in skip:
            c = pd.to_numeric(df[col], errors="coerce")
            if c.notna().any():
                df[col] = c

    return df


# ── Scouting Score (position-relative, small-sample-safe baseline) ────────────

def _weighted_zscore_score_stable(grp: pd.DataFrame, blueprint: list[tuple[str, float]]) -> pd.Series:
    """
    Same weighted z-score logic as wyscout_model._weighted_zscore_score, but
    the mean/stdev baseline is computed only from players with >= STABLE_MINUTES,
    so a handful of 20-minute cameos can't drag the position-group baseline
    around. Every player in the group is still scored against that baseline.
    """
    available = [(m, w) for m, w in blueprint if m in grp.columns]
    if not available:
        return pd.Series(50.0, index=grp.index)

    stable = grp["Minutes played"] >= STABLE_MINUTES
    baseline = grp.loc[stable] if stable.sum() >= 8 else grp

    total_w = sum(w for _, w in available)
    z_composite = pd.Series(0.0, index=grp.index)

    for metric, weight in available:
        col_all  = pd.to_numeric(grp[metric], errors="coerce").fillna(0)
        col_base = pd.to_numeric(baseline[metric], errors="coerce").fillna(0)
        mu  = col_base.mean()
        sig = col_base.std() or 1e-9
        z_composite += (weight / total_w) * (col_all - mu) / sig

    return pd.Series(_z_to_percentile(z_composite.values), index=grp.index)


def compute_striker_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Dynamic Mobility Score, Aerial/Target Score and Archetype for the ST
    role group only (n/a for every other role group).
    """
    df = df.copy()
    for col in ("Dynamic Mobility Score", "Aerial/Target Score"):
        df[col] = np.nan
    df["Striker Archetype"] = ""

    st_mask = df["PositionGroup"] == "ST"
    st = df.loc[st_mask].copy()
    if st.empty:
        return df

    st["Dynamic Mobility Score"] = _weighted_zscore_score_stable(st, DYNAMIC_MOBILITY_BLUEPRINT).round(1)
    st["Aerial/Target Score"] = _weighted_zscore_score_stable(st, AERIAL_TARGET_BLUEPRINT).round(1)
    st["Striker Archetype"] = st.apply(_archetype, axis=1)

    df.loc[st_mask, "Dynamic Mobility Score"] = st["Dynamic Mobility Score"]
    df.loc[st_mask, "Aerial/Target Score"] = st["Aerial/Target Score"]
    df.loc[st_mask, "Striker Archetype"] = st["Striker Archetype"]

    df["Dynamic Rank"] = np.nan
    df.loc[st_mask, "Dynamic Rank"] = (
        df.loc[st_mask, "Dynamic Mobility Score"]
        .rank(ascending=False, method="min")
    )
    return df


def compute_scouting_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    frames = []
    for pos_group, grp in df.groupby("PositionGroup"):
        grp = grp.copy()
        for score_col, blueprint in SCORE_BLUEPRINTS.items():
            grp[score_col] = _weighted_zscore_score_stable(grp, blueprint)

        composite, total_w = pd.Series(0.0, index=grp.index), 0.0
        rel = POSITION_RELEVANCE.get(str(pos_group), {})
        for sc, base_w in COMPOSITE_WEIGHTS.items():
            if sc in grp.columns:
                w = base_w * rel.get(sc, 1.0)
                composite += w * grp[sc]
                total_w += w
        grp["Scouting Score"] = (composite / (total_w or 1)).round(1).clip(0, 100)
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


# ── Uncertainty Score ───────────────────────────────────────────────────────────

def compute_uncertainty_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mins = df["Minutes played"].clip(lower=0)
    sample_fraction = (mins / RELIABLE_MINUTES_BENCHMARK).clip(upper=1.0)
    # sqrt scaling: std. error of a per-90 rate ~ 1/sqrt(minutes), so
    # uncertainty is modelled as (1 - sqrt(sample_fraction)).
    df["Uncertainty Score"] = (100 * (1 - np.sqrt(sample_fraction))).round(1)
    return df


# ── Labels & ranks ──────────────────────────────────────────────────────────────

def _score_tier(pct: float) -> str:
    for cutoff, name in SCORE_TIERS:
        if pct >= cutoff:
            return name
    return "Fringe"


def _confidence_tier(uncertainty: float) -> str:
    for cutoff, name in CONFIDENCE_TIERS:
        if uncertainty <= cutoff:
            return name
    return "Very Low Confidence"


def _storyline(row: pd.Series) -> str:
    tier, conf, age = row["Score Tier"], row["Confidence Tier"], row["Age"]
    high_tier = tier in ("Elite", "Very Good")
    low_tier  = tier in ("Below Average", "Fringe")
    if high_tier and conf in ("Low Confidence", "Very Low Confidence") and pd.notna(age) and age <= 23:
        return "Breakout Watch"
    if high_tier and conf == "High Confidence":
        return "Proven Performer"
    if tier == "Good" and conf == "High Confidence":
        return "Steady Contributor"
    if conf in ("Low Confidence", "Very Low Confidence") and row["Minutes played"] < STABLE_MINUTES:
        return "Small-Sample Flyer"
    if low_tier and conf == "High Confidence":
        return "Reliably Below Bar"
    return "Rotation Piece"


def compute_labels_and_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Percentile of Scouting Score *within* position group -> tier label
    df["Score Percentile (in role)"] = (
        df.groupby("PositionGroup")["Scouting Score"].rank(pct=True) * 100
    ).round(1)
    df["Score Tier"] = df["Score Percentile (in role)"].apply(_score_tier)
    df["Confidence Tier"] = df["Uncertainty Score"].apply(_confidence_tier)
    df["Label"] = df["Score Tier"] + " / " + df["Confidence Tier"]
    df["Storyline"] = df.apply(_storyline, axis=1)

    df["Position Rank"] = (
        df.groupby("PositionGroup")["Scouting Score"]
        .rank(ascending=False, method="min").astype(int)
    )
    df["Overall Rank"] = df["Scouting Score"].rank(ascending=False, method="min").astype(int)

    return df


# ── Excel export ────────────────────────────────────────────────────────────────

FONT_NAME = "Calibri"
NAVY   = "1F3864"
LIGHT  = "D9E2F3"
WHITE_FONT = Font(name=FONT_NAME, color="FFFFFF", bold=True, size=11)
BODY_FONT  = Font(name=FONT_NAME, size=10)
BOLD_FONT  = Font(name=FONT_NAME, size=10, bold=True)
HEADER_FILL = PatternFill("solid", fgColor=NAVY)
STRIPE_FILL = PatternFill("solid", fgColor=LIGHT)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _style_header(ws, ncols: int, row: int = 1) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = WHITE_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def _write_table(ws, df: pd.DataFrame, start_row: int = 1) -> None:
    for j, col in enumerate(df.columns, start=1):
        ws.cell(row=start_row, column=j, value=col)
    _style_header(ws, len(df.columns), row=start_row)

    for i, (_, row) in enumerate(df.iterrows(), start=start_row + 1):
        for j, col in enumerate(df.columns, start=1):
            val = row[col]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            cell = ws.cell(row=i, column=j, value=val)
            cell.font = BODY_FONT
            cell.border = BORDER
            if i % 2 == 0:
                cell.fill = STRIPE_FILL

    for j, col in enumerate(df.columns, start=1):
        width = max(len(str(col)), df[col].astype(str).map(len).max() if len(df) else 0)
        ws.column_dimensions[get_column_letter(j)].width = min(max(width + 2, 10), 40)

    ws.freeze_panes = ws.cell(row=start_row + 1, column=1).coordinate


def build_workbook(df: pd.DataFrame) -> None:
    board_cols = [
        "Overall Rank", "Position Rank", "Player", "Team", "Position", "PositionGroup",
        "Age", "Minutes played", "Matches played",
        "Scouting Score", "Score Percentile (in role)", "Score Tier",
        "Uncertainty Score", "Confidence Tier", "Storyline", "Label",
    ]
    board = df[board_cols].sort_values("Overall Rank").reset_index(drop=True)
    board = board.rename(columns={"PositionGroup": "Role Group"})

    sub_score_cols = ["Player", "Team", "PositionGroup"] + list(SCORE_BLUEPRINTS.keys()) + ["Scouting Score"]
    detail = df[sub_score_cols].rename(columns={"PositionGroup": "Role Group"}) \
                                 .sort_values("Scouting Score", ascending=False).reset_index(drop=True)

    role_cols = ["Position Rank", "Player", "Team", "Age", "Minutes played",
                 "Scouting Score", "Uncertainty Score", "Label"]
    by_role_frames = {}
    for grp_name, grp in df.groupby("PositionGroup"):
        top = grp[role_cols].sort_values("Position Rank").head(15).reset_index(drop=True)
        by_role_frames[grp_name] = top

    watch = df.loc[df["Storyline"] == "Breakout Watch", board_cols] \
              .sort_values("Scouting Score", ascending=False).reset_index(drop=True)
    watch = watch.rename(columns={"PositionGroup": "Role Group"})

    dyn_cols = ["Dynamic Rank", "Player", "Team", "Age", "Minutes played", "Matches played",
                "Dynamic Mobility Score", "Aerial/Target Score", "Striker Archetype",
                "Scouting Score", "Uncertainty Score", "Confidence Tier", "Label"]
    dynamic = df.loc[df["PositionGroup"] == "ST", dyn_cols] \
                .sort_values("Dynamic Rank").reset_index(drop=True)
    dynamic["Dynamic Rank"] = dynamic["Dynamic Rank"].astype(int)

    wb = Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("Scouting Board")
    _write_table(ws, board)
    ws.auto_filter.ref = ws.dimensions
    # colour scale on Scouting Score
    col_idx = board.columns.get_loc("Scouting Score") + 1
    col_letter = get_column_letter(col_idx)
    rng = f"{col_letter}2:{col_letter}{len(board)+1}"
    ws.conditional_formatting.add(
        rng,
        ColorScaleRule(start_type="min", start_color="F8696B",
                        mid_type="percentile", mid_value=50, mid_color="FFEB84",
                        end_type="max", end_color="63BE7B"),
    )
    col_idx_u = board.columns.get_loc("Uncertainty Score") + 1
    col_letter_u = get_column_letter(col_idx_u)
    rng_u = f"{col_letter_u}2:{col_letter_u}{len(board)+1}"
    ws.conditional_formatting.add(
        rng_u,
        ColorScaleRule(start_type="min", start_color="63BE7B",
                        mid_type="percentile", mid_value=50, mid_color="FFEB84",
                        end_type="max", end_color="F8696B"),
    )

    ws2 = wb.create_sheet("Breakout Watch")
    if len(watch):
        _write_table(ws2, watch)
        ws2.auto_filter.ref = ws2.dimensions
    else:
        ws2["A1"] = "No players met the Breakout Watch criteria in this file."
        ws2["A1"].font = BODY_FONT

    ws_dyn = wb.create_sheet("Dynamic Striker Finder")
    _write_table(ws_dyn, dynamic)
    ws_dyn.auto_filter.ref = ws_dyn.dimensions
    dyn_col_idx = dynamic.columns.get_loc("Dynamic Mobility Score") + 1
    dyn_col_letter = get_column_letter(dyn_col_idx)
    ws_dyn.conditional_formatting.add(
        f"{dyn_col_letter}2:{dyn_col_letter}{len(dynamic)+1}",
        ColorScaleRule(start_type="min", start_color="F8696B",
                        mid_type="percentile", mid_value=50, mid_color="FFEB84",
                        end_type="max", end_color="63BE7B"),
    )

    for grp_name, top in by_role_frames.items():
        ws_r = wb.create_sheet(f"Role - {grp_name}"[:31])
        _write_table(ws_r, top)

    ws3 = wb.create_sheet("Sub-Score Detail")
    _write_table(ws3, detail)
    ws3.auto_filter.ref = ws3.dimensions

    ws_m = wb.create_sheet("Methodology")
    _write_methodology(ws_m)

    wb.move_sheet("Methodology", offset=-(len(wb.sheetnames) - 1))
    wb.save(OUT_FILE)


def _write_methodology(ws) -> None:
    ws.column_dimensions["A"].width = 100
    ws.sheet_view.showGridLines = False
    row = 1

    def title(text, size=14):
        nonlocal row
        c = ws.cell(row=row, column=1, value=text)
        c.font = Font(name=FONT_NAME, bold=True, size=size, color=NAVY)
        row += 2

    def para(text):
        nonlocal row
        c = ws.cell(row=row, column=1, value=text)
        c.font = BODY_FONT
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = max(15, 15 * (len(text) // 95 + 1))
        row += 1

    def spacer(n=1):
        nonlocal row
        row += n

    title(f"Scouting Model Methodology — {LEAGUE_LABEL}", 16)
    para("Source: Italy II.xlsx (Wyscout export), 500 players, single competition season snapshot.")
    spacer()

    title("1. Position grouping", 12)
    para("Each player's first listed Wyscout position is mapped to one of eight role groups so that "
         "scores are always computed relative to positional peers, never across positions: "
         "ST (out-and-out forwards), W (wide forwards), AM (attacking midfield), CM (central midfield), "
         "DM (defensive midfield), FB (full-back / wing-back), CB (centre-back), GK (goalkeeper). "
         "Comparing a centre-back's tackle numbers to a striker's would be meaningless, so every "
         "statistic below is standardised inside its own role group.")
    spacer()

    title("2. Scouting Score (0-100)", 12)
    para("Built from the same seven underlying per-90 blueprints used across the club's Wyscout model: "
         "ScoringThreatScore, CreativeProgressionScore, DefensiveDisruptionScore, PressingScore, "
         "BallSecurityScore, ExpectedThreatScore and ASA_GoalsAddedScore. Each blueprint is a weighted "
         "set of raw per-90 / % metrics (e.g. ScoringThreatScore = Goals p90, npGoals p90, xG p90, "
         "Shots p90, Shots on target %, Goal conversion %, Touches in box p90).")
    para("Step 1 — Baseline: for every role group, the mean and standard deviation of each metric are "
         "computed only from players with at least 630 minutes (~7 full matches). This stops a handful "
         "of small-sample cameos from distorting what 'average' looks like for the group.")
    para("Step 2 — Z-score: every player in the group (including sub-630-minute players) is standardised "
         "against that stable baseline: z = (player value − baseline mean) / baseline st. dev.")
    para("Step 3 — Blueprint score: the weighted sum of a blueprint's metric z-scores is converted to a "
         "0-100 value via the normal CDF (i.e. it reads as 'percentile versus role peers').")
    para("Step 4 — Composite: the seven blueprint scores are combined into one Scouting Score using "
         "role-specific relevance weights (e.g. DefensiveDisruptionScore counts far more for a CB than "
         "for a striker; ScoringThreatScore counts far more for a striker than for a CB). This mirrors "
         "how a human scout would weigh categories differently depending on the position being watched.")
    spacer()

    title("3. Uncertainty Score (0-100)", 12)
    para("The Scouting Score is a rate statistic (per 90 minutes). Rate statistics estimated from a small "
         "number of minutes carry more sampling noise — the standard error of a per-90 rate shrinks "
         "roughly with 1/sqrt(minutes played), not linearly. The Uncertainty Score encodes exactly that:")
    para("Uncertainty Score = 100 x (1 − sqrt(min(minutes played, 2700) / 2700))")
    para("2,700 minutes (~30 full matches) is treated as a 'full, reliable sample' for a Serie B season "
         "and is capped at 0 uncertainty; a player with ~675 minutes (a quarter of that) still carries "
         "50 points of uncertainty because sqrt(0.25) = 0.5, not 0.75. A player with only a handful of "
         "cameo appearances will show a high Scouting Score with high Uncertainty far more often than a "
         "low one — that combination is the signal, not noise, and is precisely what the labels below "
         "are built to surface.")
    spacer()

    title("4. Label", 12)
    para("Label = Score Tier / Confidence Tier, where:")
    para("Score Tier (percentile of Scouting Score within the player's own role group): "
         "Elite (>=90th pct) · Very Good (75-89) · Good (55-74) · Average (35-54) · "
         "Below Average (15-34) · Fringe (<15).")
    para("Confidence Tier (from Uncertainty Score): High Confidence (<=25) · Medium Confidence (25-50) · "
         "Low Confidence (50-75) · Very Low Confidence (>75).")
    para("Storyline (article-ready tag derived from the same two axes): "
         "'Proven Performer' = high tier + high confidence; 'Breakout Watch' = high tier + low/very low "
         "confidence + age <=23 (the classic hidden-gem/small-sample-but-promising profile); "
         "'Steady Contributor' = good tier + high confidence; 'Small-Sample Flyer' = any tier with low "
         "confidence and under 630 minutes; 'Reliably Below Bar' = below-average tier + high confidence; "
         "'Rotation Piece' = everything else.")
    spacer()

    title("5. Dynamic Striker Finder (ST role group only)", 12)
    para("'Dynamic' is defined relative to its opposite, not as a synonym for 'good', so the model scores "
         "every CF/ST on two independent axes within the ST baseline (>=630 stable minutes, same z-score "
         "method as the Scouting Score):")
    para("Dynamic Mobility Score = Progressive runs p90 (3.0) + Accelerations p90 (2.5) + Dribbles p90 "
         "(2.0) + Successful dribbles % (1.5) + Offensive duels p90 (1.0) + Offensive duels won % (1.0) + "
         "Fouls suffered p90 (0.5) + xG p90 (1.5) + Non-penalty goals p90 (1.0). This rewards strikers who "
         "create their own chances by carrying the ball and beating defenders, not just finishing service.")
    para("Aerial/Target Score = Aerial duels won % (3.0) + Aerial duels p90 (2.0) + Head goals p90 (2.0) + "
         "Received long passes p90 (2.0) + Touches in box p90 (1.0). This captures the classic target-man "
         "profile: aerial presence and being the outlet for direct/long-ball supply.")
    para("Striker Archetype (median split, i.e. >=50th percentile within the ST group, on each axis): "
         "'Complete Forward' = high on both axes; 'Dynamic Runner' = high mobility, average-or-below "
         "aerial/target play — this is the 'dynamic striker' profile; 'Target Man' = the reverse; "
         "'Limited Profile' = below role average on both.")
    para("The Dynamic Striker Finder sheet ranks every CF/ST purely by Dynamic Mobility Score (Dynamic "
         "Rank), and carries the Aerial/Target Score and Archetype alongside it so a 'Dynamic Runner' can "
         "be told apart from a 'Complete Forward' who merely also happens to rank high on mobility. Cross-"
         "check Uncertainty Score before trusting a small-sample mobility outlier.")
    spacer()

    title("6. Ranking", 12)
    para("Position Rank = rank by Scouting Score within the player's role group (the primary, most "
         "meaningful ranking — always compares like-for-like).")
    para("Overall Rank = rank by Scouting Score across all 500 players in the file. Because Scouting "
         "Score is already normalised to a 0-100 percentile-style scale within each role group, cross-role "
         "comparison is directionally reasonable but should be used for headline/article framing "
         "('best player in Serie B this season') rather than as a scouting decision on its own — always "
         "read Overall Rank alongside Role Group and Position Rank.")
    spacer()

    title("7. How to use this in an article", 12)
    para("- Lead with Position Rank + Scouting Score for a like-for-like 'best CB in the league' framing.")
    para("- Use the 'Breakout Watch' sheet for a 'players to watch' section — young, high scoring, "
         "small sample, i.e. exactly the profile a recruitment article wants to flag before the price rises.")
    para("- Use the 'Dynamic Striker Finder' sheet for a dedicated 'most dynamic forward in Serie B' "
         "angle — sort by Dynamic Rank, quote the Aerial/Target Score as the contrast that proves the "
         "player is a carrier/mover rather than just a name with a high overall Scouting Score.")
    para("- Quote the Uncertainty Score whenever a headline number is built on a partial season, so the "
         "claim is defensible ('elite Scouting Score, but only 8 appearances — treat as provisional').")
    para("- The Sub-Score Detail sheet lets you justify *why* a player scores well (e.g. 'driven by "
         "DefensiveDisruptionScore, not by ball retention') rather than just citing the headline number.")


def main() -> None:
    df = load_italy_ii()
    df = compute_scouting_score(df)
    df = compute_uncertainty_score(df)
    df = compute_labels_and_ranks(df)
    df = compute_striker_archetypes(df)
    build_workbook(df)
    print(f"Wrote {OUT_FILE} ({len(df)} players)")


if __name__ == "__main__":
    main()
