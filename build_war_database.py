"""
build_war_database.py — WAR (Wins Above Replacement) for every wide attacker in the Wyscout DB

Methodology
-----------
Offensive rate  = xG/90 + xA/90×0.85 + ShotAssists/90×0.28
Carry rate      = ProgRuns/90×0.025  + (Dribbles/90 × Drib%/100)×0.015
Total rate      = Offensive + Carry

Replacement     = 15th-percentile total rate across ALL DB wide attackers (≥300 min)
WAR             = (player_rate − replacement_rate) × (minutes / 90) ÷ 3.0 goals/win

Output: reports/WAR_All_Players.xlsx
  Sheet 1 — WAR Rankings  : one row per player, sorted by WAR desc
  Sheet 2 — Methodology   : parameter reference
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                                  GradientFill)
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter
    HAS_OPX = True
except ImportError:
    HAS_OPX = False

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
WIDE_ATK_POS = {"LAMF", "RAMF", "LW", "RW", "LWF", "RWF", "AMF", "LWB", "RWB"}
MIN_MINS         = 300
GOALS_PER_WIN    = 3.0
REPL_PERCENTILE  = 15   # "freely available" player tier

NON_METRIC = {
    "Player", "Team", "Team within selected timeframe", "Position",
    "Age", "Market value", "Contract expires", "Birth country",
    "Passport country", "Foot", "Height", "Weight", "On loan",
}

RATE_COLS = [
    "xG per 90", "xA per 90", "Shot assists per 90",
    "Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in NON_METRIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pool() -> pd.DataFrame:
    print("  Loading Wyscout files …")
    parts = []
    skipped = []
    for f in sorted(Path("data/Wyscout DB").glob("*.xlsx")):
        if f.stem in ("Czech U17", "Czech U19"):
            continue
        try:
            d = _numeric(pd.read_excel(f))
            d["_league"] = f.stem
            d["_pos1"]   = d["Position"].astype(str).str.split(",").str[0].str.strip()
            parts.append(d)
        except Exception as e:
            skipped.append(f.stem)

    if skipped:
        print(f"  Skipped {len(skipped)} files (read errors)")

    df_all = pd.concat(parts, ignore_index=True)
    mask   = (df_all["_pos1"].isin(WIDE_ATK_POS) &
              (df_all["Minutes played"].fillna(0) >= MIN_MINS))
    pool   = df_all[mask].copy().reset_index(drop=True)
    print(f"  {len(pool):,} wide attackers across {df_all['_league'].nunique()} leagues")
    return pool


# ── WAR calculation ────────────────────────────────────────────────────────────

def _get(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return df[col].fillna(default) if col in df.columns else pd.Series(default, index=df.index)


def compute_rates(df: pd.DataFrame):
    xg  = _get(df, "xG per 90")
    xa  = _get(df, "xA per 90")
    sa  = _get(df, "Shot assists per 90")
    pr  = _get(df, "Progressive runs per 90")
    d   = _get(df, "Dribbles per 90")
    dp  = _get(df, "Successful dribbles, %", 50.0)

    off   = xg + xa * 0.85 + sa * 0.28
    carry = pr * 0.025 + (d * dp / 100.0) * 0.015
    return off, carry, off + carry


def build_war(pool: pd.DataFrame) -> pd.DataFrame:
    off, carry, total = compute_rates(pool)

    repl       = float(np.percentile(total.dropna(), REPL_PERCENTILE))
    repl_off   = float(np.percentile(off.dropna(),   REPL_PERCENTILE))
    repl_carry = float(np.percentile(carry.dropna(), REPL_PERCENTILE))
    print(f"  Replacement rate  = {repl:.4f}  "
          f"(off {repl_off:.4f}  carry {repl_carry:.4f})")

    minutes = _get(pool, "Minutes played", 450.0)
    n90     = minutes / 90.0
    var90   = total - repl
    war     = var90 * n90 / GOALS_PER_WIN

    # ── Assemble result frame ─────────────────────────────────────────────────
    res = pd.DataFrame({
        "Player":         pool["Player"],
        "Team":           pool.get("Team within selected timeframe", pool.get("Team", "")),
        "League":         pool["_league"],
        "Position":       pool["Position"],
        "Age":            pool.get("Age", np.nan),
        "Minutes":        minutes.round(0).astype(int),
        "90s":            n90.round(2),
        "WAR":            war.round(3),
        "WAR_per_90":     (var90 / GOALS_PER_WIN).round(4),
        "Offensive_rate": off.round(4),
        "Carry_rate":     carry.round(4),
        "Total_rate":     total.round(4),
        "Rate_vs_repl":   var90.round(4),
    })

    # DB rank & percentile
    war_arr = war.values
    res["DB_rank"]       = res["WAR"].rank(ascending=False, method="min").astype(int)
    res["DB_percentile"] = res["WAR"].apply(
        lambda w: round(float(np.mean(war_arr < w) * 100), 1) if pd.notna(w) else np.nan
    )

    # League rank & percentile (within same league file)
    res["Lg_rank"] = (res.groupby("League")["WAR"]
                        .rank(ascending=False, method="min").astype(int))
    res["Lg_n"]    = res.groupby("League")["WAR"].transform("count").astype(int)
    res["Lg_percentile"] = (res.groupby("League")["WAR"]
                              .transform(lambda x: x.apply(
                                  lambda w: round(float((x < w).mean() * 100), 1)
                                  if pd.notna(w) else np.nan)))

    # Deduplicate: same player+team in multiple league files → keep max-minutes row
    before = len(res)
    res = (res.sort_values("Minutes", ascending=False)
              .drop_duplicates(subset=["Player", "Team"], keep="first"))
    removed = before - len(res)
    if removed:
        print(f"  Removed {removed} duplicate player-team rows (same player in multiple files)")

    # Sort by WAR descending
    res = res.sort_values("WAR", ascending=False).reset_index(drop=True)
    res.insert(0, "Rank", range(1, len(res) + 1))

    print(f"\n  Top 10 by WAR:")
    for _, r in res.head(10).iterrows():
        print(f"    {int(r['Rank']):>4}.  {str(r['Player']):<28}  "
              f"{str(r['League']):<20}  WAR={r['WAR']:+.3f}")

    return res, repl


# ── Excel output ───────────────────────────────────────────────────────────────

def _col_widths(df: pd.DataFrame) -> dict[str, int]:
    widths = {}
    for col in df.columns:
        header_w = len(str(col)) + 2
        try:
            max_data = df[col].astype(str).str.len().max()
        except Exception:
            max_data = 10
        widths[col] = max(header_w, min(int(max_data) + 2, 40))
    return widths


def format_workbook(wb, df: pd.DataFrame, repl: float):
    if not HAS_OPX:
        return

    ws = wb["WAR Rankings"]

    # Styles
    hdr_font   = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
    hdr_fill   = PatternFill("solid", fgColor="1D4ED8")
    hdr_align  = Alignment(horizontal="center", vertical="center", wrap_text=True)
    data_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left",   vertical="center")
    thin       = Side(style="thin", color="D1D9E6")
    border     = Border(bottom=thin)

    # Header row
    for cell in ws[1]:
        cell.font      = hdr_font
        cell.fill      = hdr_fill
        cell.alignment = hdr_align
        cell.border    = border

    # Freeze header + auto-filter
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    # Alternating row fill
    light = PatternFill("solid", fgColor="F7F9FC")
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
        for cell in row:
            cell.alignment = data_align
            if row_idx % 2 == 0:
                cell.fill = light
        # left-align player / team / league columns
        for ci in [2, 3, 4, 5]:
            if ci <= len(row):
                row[ci - 1].alignment = left_align

    # Column widths
    widths = _col_widths(df)
    for ci, col in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(col, 12)

    # WAR colour scale (column index)
    war_col_idx = list(df.columns).index("WAR") + 1
    war_col_ltr = get_column_letter(war_col_idx)
    ws.conditional_formatting.add(
        f"{war_col_ltr}2:{war_col_ltr}{ws.max_row}",
        ColorScaleRule(
            start_type="percentile", start_value=5,  start_color="DC2626",
            mid_type="percentile",   mid_value=50,   mid_color="D97706",
            end_type="percentile",   end_value=95,   end_color="16A34A",
        ),
    )

    # Methodology sheet styling
    ws2 = wb["Methodology"]
    for cell in ws2[1]:
        cell.font      = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
        cell.fill      = PatternFill("solid", fgColor="1D4ED8")
        cell.alignment = hdr_align
    for col in ws2.columns:
        max_w = max(len(str(c.value or "")) for c in col) + 4
        ws2.column_dimensions[col[0].column_letter].width = min(max_w, 80)


def save_excel(res: pd.DataFrame, repl: float, path: Path):
    method_rows = [
        ("Offensive rate / 90",  "xG/90  +  xA/90 × 0.85  +  ShotAssists/90 × 0.28"),
        ("Carry rate / 90",      "ProgRuns/90 × 0.025  +  (Dribbles/90 × Drib%/100) × 0.015"),
        ("Total rate / 90",      "Offensive rate + Carry rate"),
        ("Replacement level",    f"{repl:.4f}  (15th-percentile total rate, all DB wide attackers ≥{MIN_MINS} min)"),
        ("Goals per win",        f"{GOALS_PER_WIN:.1f}  (conservative estimate for converting goal-equivalents to wins)"),
        ("WAR formula",          "(Total_rate − Replacement) × (Minutes / 90) ÷ Goals_per_win"),
        ("WAR_per_90",           "(Total_rate − Replacement) ÷ Goals_per_win"),
        ("Positions included",   ", ".join(sorted(WIDE_ATK_POS))),
        ("Min. minutes filter",  f"≥{MIN_MINS} minutes played"),
        ("League percentile",    "Within same Wyscout file (league)"),
        ("DB percentile",        "Across full database (all positions/leagues after position filter)"),
    ]

    method_df = pd.DataFrame(method_rows, columns=["Parameter", "Definition"])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        res.to_excel(writer, index=False, sheet_name="WAR Rankings")
        method_df.to_excel(writer, index=False, sheet_name="Methodology")

        if HAS_OPX:
            format_workbook(writer.book, res, repl)

    print(f"\n  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Building WAR database …")
    pool      = load_pool()
    res, repl = build_war(pool)

    out = OUT_DIR / "WAR_All_Players.xlsx"
    save_excel(res, repl, out)
    print(f"  {len(res):,} players · {res['League'].nunique()} leagues")


if __name__ == "__main__":
    main()
