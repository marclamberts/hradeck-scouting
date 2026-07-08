"""
build_japan_strikers_scouting.py — Striker archetype scouting model for Japan II/III

Source: "Japan II III.xlsx" (Wyscout export covering J.League 2 + J3, the only
Japan-tier-below-top-flight file in the DB — there is no separate "Japan II.xlsx").

Scope: players whose Position is CF, or whose first-listed position is CF
(e.g. "CF, AMF" counts, "AMF, CF" does not).

Method
------
1. Z-score every metric within the CF pool itself (this screen is relative to
   these 70 strikers, not the whole database).
2. Group metrics into four striker archetypes; average the z-scores per group
   and convert to a 0-100 scale via the normal CDF (norm.cdf(z) * 100) —
   i.e. "0-100 normalised z-scores" per the brief.
3. Primary Archetype = highest-scoring group; "Complete Forward" overrides
   when a player scores strongly (>=65) in all four groups at once.
4. Scouting Score = weighted blend of the four archetype z-scores (goal-
   scoring weighted highest, since finishing is a striker's core job),
   converted to 0-100 the same way.
5. Uncertainty Score = 0-100, higher = less reliable. Weighted blend of five
   components (each 0-100, higher = more uncertain):
     - Minutes gap     (35%): shortfall vs. a 1800-minute "settled" reference
     - Missing data    (15%): fraction of the 24 scoring metrics that were NaN
     - Shot-volume gap (20%): shortfall vs. a 30-shot reference — Goal
       conversion % and Shots on target % are noisy at low shot counts
     - Age volatility  (15%): distance outside a 24-29 "prime" age band —
       younger/older players' underlying rates swing more season to season
     - Metric spread   (15%): how outlier-driven the profile is — pool
       percentile rank of the std-dev across the player's 24 metric z-scores;
       a score propped up by one or two standout metrics is less trustworthy
       than one built on a broadly consistent profile

Output: reports/Japan_Strikers_Scouting.xlsx
  Sheet 1 — Strikers       : full scouting table
  Sheet 2 — Uncertainty    : per-component uncertainty breakdown
  Sheet 3 — Archetype Z    : raw per-metric z-scores used in each archetype
  Sheet 4 — Methodology    : formulas and metric groupings
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter
    HAS_OPX = True
except ImportError:
    HAS_OPX = False

SRC     = Path("data/Wyscout DB/Japan II III.xlsx")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)
OUT     = OUT_DIR / "Japan_Strikers_Scouting.xlsx"

REFERENCE_MINUTES = 1800.0   # ~20 full matches -> "settled" sample
REFERENCE_SHOTS   = 30.0     # shot volume above which conversion-type metrics start to stabilise
PRIME_AGE_LO, PRIME_AGE_HI = 24, 29    # outside this band, underlying rates are treated as more volatile
AGE_VOLATILITY_PER_YEAR    = 10.0      # uncertainty points added per year outside the prime band

UNCERTAINTY_WEIGHTS = {
    "Minutes gap":     0.35,
    "Missing data":    0.15,
    "Shot-volume gap": 0.20,
    "Age volatility":  0.15,
    "Metric spread":   0.15,
}

ARCHETYPES = {
    "Target Striker": [
        "Aerial duels per 90",
        "Aerial duels won, %",
        "Head goals per 90",
        "Height",
        "Offensive duels won, %",
        "Received long passes per 90",
    ],
    "Dynamic Striker": [
        "Progressive runs per 90",
        "Accelerations per 90",
        "Dribbles per 90",
        "Successful dribbles, %",
        "Received passes per 90",
        "Offensive duels per 90",
    ],
    "Goalscoring Striker": [
        "Non-penalty goals per 90",
        "xG per 90",
        "Shots per 90",
        "Shots on target, %",
        "Goal conversion, %",
        "Touches in box per 90",
    ],
    "Second Striker": [
        "xA per 90",
        "Assists per 90",
        "Key passes per 90",
        "Smart passes per 90",
        "Through passes per 90",
        "Progressive passes per 90",
    ],
}

SCOUTING_WEIGHTS = {
    "Goalscoring Striker": 0.40,
    "Target Striker":      0.20,
    "Dynamic Striker":     0.20,
    "Second Striker":      0.20,
}

COMPLETE_FORWARD_THRESHOLD = 65.0

SCORE_TIERS = [
    (80, "Elite"),
    (65, "Very Good"),
    (50, "Solid"),
    (35, "Fringe"),
    (-1, "Development"),
]

NON_METRIC = {
    "Player", "Team", "Team within selected timeframe", "Position", "Age",
    "Market value", "Contract expires", "Birth country", "Passport country",
    "Foot", "Height", "Weight", "On loan",
}


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in NON_METRIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_strikers() -> pd.DataFrame:
    df = _numeric(pd.read_excel(SRC, sheet_name=0))
    pos1 = df["Position"].astype(str).str.split(",").str[0].str.strip()
    strikers = df.loc[pos1 == "CF"].copy().reset_index(drop=True)
    return strikers


def zscore_pool(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Z-score each metric within df. Returns (z-scores df, missing-fraction per row)."""
    missing = df[metrics].isna().mean(axis=1)
    filled  = df[metrics].apply(lambda s: s.fillna(s.median()))
    mu, sig = filled.mean(), filled.std(ddof=0).replace(0, 1e-9)
    z = (filled - mu) / sig
    z.columns = [f"z::{c}" for c in metrics]
    return z, missing


def score_tier(x: float) -> str:
    for cutoff, label in SCORE_TIERS:
        if x >= cutoff:
            return label
    return "Development"


def confidence_label(u: float) -> str:
    if u <= 25:
        return "High confidence"
    if u <= 50:
        return "Medium confidence"
    return "Low confidence"


def compute_uncertainty(strikers: pd.DataFrame, z: pd.DataFrame, missing_frac: pd.Series) -> pd.DataFrame:
    """Five 0-100 components (higher = more uncertain), blended by UNCERTAINTY_WEIGHTS."""
    idx = strikers.index

    minutes      = strikers["Minutes played"].fillna(0)
    minutes_rel  = (minutes / REFERENCE_MINUTES).clip(upper=1.0)
    minutes_gap  = 100 * (1 - minutes_rel)

    missing_data = missing_frac * 100

    shots       = strikers.get("Shots", pd.Series(0, index=idx)).fillna(0)
    shots_rel   = (shots / REFERENCE_SHOTS).clip(upper=1.0)
    shot_gap    = 100 * (1 - shots_rel)

    age      = pd.to_numeric(strikers["Age"], errors="coerce")
    age_gap  = np.where(age <= PRIME_AGE_LO, PRIME_AGE_LO - age,
                 np.where(age >= PRIME_AGE_HI, age - PRIME_AGE_HI, 0.0))
    age_gap  = pd.Series(age_gap, index=idx).fillna(PRIME_AGE_HI - PRIME_AGE_LO)  # unknown age -> mid-range penalty
    age_vol  = (age_gap * AGE_VOLATILITY_PER_YEAR).clip(0, 100)

    z_std         = z.std(axis=1)
    metric_spread = (z_std.rank(pct=True) * 100)

    comp = pd.DataFrame({
        "Minutes gap":     minutes_gap.round(1),
        "Missing data":    missing_data.round(1),
        "Shot-volume gap": shot_gap.round(1),
        "Age volatility":  age_vol.round(1),
        "Metric spread":   metric_spread.round(1),
    }, index=idx)

    blended = sum(comp[name] * w for name, w in UNCERTAINTY_WEIGHTS.items())
    comp["Uncertainty Score"] = blended.clip(0, 100).round(1)
    return comp


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strikers = load_strikers()

    all_metrics = sorted({m for grp in ARCHETYPES.values() for m in grp})
    z, missing_frac = zscore_pool(strikers, all_metrics)

    archetype_z   = {}
    archetype_pct = {}
    for name, metrics in ARCHETYPES.items():
        zcols = [f"z::{m}" for m in metrics]
        gz = z[zcols].mean(axis=1)
        archetype_z[name]   = gz
        archetype_pct[name] = norm.cdf(gz) * 100

    pct_df = pd.DataFrame(archetype_pct).round(1)

    scouting_z = sum(archetype_z[name] * w for name, w in SCOUTING_WEIGHTS.items())
    scouting_score = pd.Series(norm.cdf(scouting_z) * 100, index=scouting_z.index).round(1)

    uncert_df   = compute_uncertainty(strikers, z, missing_frac)
    uncertainty = uncert_df["Uncertainty Score"]

    minutes = strikers["Minutes played"].fillna(0)

    primary = pct_df.idxmax(axis=1)
    is_complete = (pct_df.min(axis=1) >= COMPLETE_FORWARD_THRESHOLD)
    primary = primary.where(~is_complete, "Complete Forward")

    tier  = scouting_score.apply(score_tier)
    label = tier + " " + primary

    out = pd.DataFrame({
        "Player":  strikers["Player"],
        "Team":    strikers.get("Team within selected timeframe", strikers.get("Team", "")),
        "Position": strikers["Position"],
        "Age":     strikers["Age"],
        "Minutes": minutes.round(0).astype(int),
        "90s":     (minutes / 90).round(1),
    })
    out = pd.concat([out, pct_df.add_suffix(" %")], axis=1)
    out["Primary Archetype"]  = primary
    out["Scouting Score"]     = scouting_score
    out["Uncertainty Score"]  = uncertainty
    out["Confidence"]         = uncertainty.apply(confidence_label)
    out["Label"]              = label

    order  = out["Scouting Score"].sort_values(ascending=False).index
    ptteam = out[["Player", "Team"]].loc[order].reset_index(drop=True)
    out    = out.loc[order].reset_index(drop=True)
    zsheet = pd.concat([ptteam, z.loc[order].round(3).reset_index(drop=True)], axis=1)
    usheet = pd.concat([ptteam, uncert_df.loc[order].reset_index(drop=True)], axis=1)

    return out, zsheet, usheet


def build_shortlist(main: pd.DataFrame, max_per_archetype: int = 10) -> pd.DataFrame:
    """Top-N players per Primary Archetype, ranked by Scouting Score."""
    order = (main["Primary Archetype"].value_counts().index.tolist())
    frames = []
    for arch in order:
        grp = (main.loc[main["Primary Archetype"] == arch]
                   .sort_values("Scouting Score", ascending=False)
                   .head(max_per_archetype)
                   .copy())
        grp.insert(0, "Rank", range(1, len(grp) + 1))
        frames.append(grp)
    out = pd.concat(frames, ignore_index=True)
    cols = ["Primary Archetype", "Rank", "Player", "Team", "Age", "Minutes",
            "Scouting Score", "Uncertainty Score", "Confidence", "Label"]
    return out[cols]


METHOD_ROWS = [
    ("Source",              "data/Wyscout DB/Japan II III.xlsx (J.League 2 + J3 combined export)"),
    ("Filter",              "Position == 'CF', or first-listed position == 'CF' (e.g. 'CF, AMF')"),
    ("Pool size",           "z-scores computed within this striker pool only, not the full DB"),
    ("", ""),
    ("Target Striker",      ", ".join(ARCHETYPES["Target Striker"])),
    ("Dynamic Striker",     ", ".join(ARCHETYPES["Dynamic Striker"])),
    ("Goalscoring Striker", ", ".join(ARCHETYPES["Goalscoring Striker"])),
    ("Second Striker",      ", ".join(ARCHETYPES["Second Striker"])),
    ("", ""),
    ("Archetype %",         "mean(z-scores in group) -> norm.cdf(z) * 100"),
    ("Complete Forward",    f"overrides Primary Archetype when all four archetype %'s >= {COMPLETE_FORWARD_THRESHOLD:.0f}"),
    ("Scouting Score",      "norm.cdf(0.40*Goalscoring_z + 0.20*Target_z + 0.20*Dynamic_z + 0.20*Second_z) * 100"),
    ("Score tiers",         "Elite >=80, Very Good >=65, Solid >=50, Fringe >=35, else Development"),
    ("", ""),
    ("Uncertainty Score",     "weighted blend of 5 components (0-100 each, higher = more uncertain) — see 'Uncertainty' sheet for the per-player breakdown"),
    ("  Minutes gap (35%)",   f"100 * (1 - min(minutes/{REFERENCE_MINUTES:.0f}, 1)) — shortfall vs. a settled ~20-match sample"),
    ("  Missing data (15%)",  "100 * fraction of the 24 scoring metrics that were NaN for this player"),
    ("  Shot-volume gap (20%)", f"100 * (1 - min(total shots/{REFERENCE_SHOTS:.0f}, 1)) — Goal conversion %% / Shots on target %% are noisy on few shots"),
    ("  Age volatility (15%)",  f"years outside the {PRIME_AGE_LO}-{PRIME_AGE_HI} prime band * {AGE_VOLATILITY_PER_YEAR:.0f}, capped at 100"),
    ("  Metric spread (15%)",   "pool percentile rank of std-dev across the player's 24 metric z-scores — flags scores propped up by 1-2 standout metrics"),
    ("Confidence label",    "High <=25, Medium <=50, Low >50"),
    ("Label",               "'{Score tier} {Primary Archetype}', e.g. 'Elite Goalscoring Striker'"),
]


def style_workbook(path: Path, main: pd.DataFrame) -> None:
    if not HAS_OPX:
        return
    from openpyxl import load_workbook
    wb = load_workbook(path)
    ws = wb["Strikers"]

    hfill = PatternFill("solid", fgColor="1D4ED8")
    hfont = Font(color="FFFFFF", bold=True)
    for cell in ws[1]:
        cell.fill = hfill
        cell.font = hfont
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    pct_cols = [c for c in main.columns if c.endswith(" %")] + ["Scouting Score", "Uncertainty Score"]
    for col_name in pct_cols:
        ci  = main.columns.get_loc(col_name) + 1
        ltr = get_column_letter(ci)
        rng = f"{ltr}2:{ltr}{ws.max_row}"
        reverse = (col_name == "Uncertainty Score")
        lo, mid, hi = ("F87171", "FEF08A", "4ADE80") if not reverse else ("4ADE80", "FEF08A", "F87171")
        ws.conditional_formatting.add(rng, ColorScaleRule(
            start_type="min", start_color=lo,
            mid_type="percentile", mid_value=50, mid_color=mid,
            end_type="max", end_color=hi,
        ))

    widths = {"Player": 22, "Team": 22, "Position": 16, "Primary Archetype": 20,
              "Confidence": 16, "Label": 30}
    for ci, col in enumerate(main.columns, 1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(col, 13)

    if "Uncertainty" in wb.sheetnames:
        uws = wb["Uncertainty"]
        for cell in uws[1]:
            cell.fill = hfill
            cell.font = hfont
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        uws.freeze_panes = "A2"
        uws.auto_filter.ref = uws.dimensions
        uncert_cols = list(UNCERTAINTY_WEIGHTS.keys()) + ["Uncertainty Score"]
        for col_name in uncert_cols:
            header_vals = [c.value for c in uws[1]]
            if col_name not in header_vals:
                continue
            ci  = header_vals.index(col_name) + 1
            ltr = get_column_letter(ci)
            rng = f"{ltr}2:{ltr}{uws.max_row}"
            uws.conditional_formatting.add(rng, ColorScaleRule(
                start_type="min", start_color="4ADE80",
                mid_type="percentile", mid_value=50, mid_color="FEF08A",
                end_type="max", end_color="F87171",
            ))
        for ci, col in enumerate(header_vals, 1):
            uws.column_dimensions[get_column_letter(ci)].width = widths.get(col, 15)

    if "Shortlist" in wb.sheetnames:
        sws = wb["Shortlist"]
        for cell in sws[1]:
            cell.fill = hfill
            cell.font = hfont
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sws.freeze_panes = "A2"
        sws.auto_filter.ref = sws.dimensions
        header_vals = [c.value for c in sws[1]]
        for col_name in ["Scouting Score", "Uncertainty Score"]:
            ci  = header_vals.index(col_name) + 1
            ltr = get_column_letter(ci)
            rng = f"{ltr}2:{ltr}{sws.max_row}"
            reverse = (col_name == "Uncertainty Score")
            lo, mid, hi = ("F87171", "FEF08A", "4ADE80") if not reverse else ("4ADE80", "FEF08A", "F87171")
            sws.conditional_formatting.add(rng, ColorScaleRule(
                start_type="min", start_color=lo,
                mid_type="percentile", mid_value=50, mid_color=mid,
                end_type="max", end_color=hi,
            ))
        sl_widths = {"Primary Archetype": 20, "Player": 22, "Team": 22, "Confidence": 16, "Label": 30}
        for ci, col in enumerate(header_vals, 1):
            sws.column_dimensions[get_column_letter(ci)].width = sl_widths.get(col, 12)
        # shade alternating archetype groups for readability
        band = PatternFill("solid", fgColor="F1F5FF")
        archetypes_seen: list[str] = []
        for ri in range(2, sws.max_row + 1):
            val = sws.cell(row=ri, column=1).value
            if val not in archetypes_seen:
                archetypes_seen.append(val)
            if archetypes_seen.index(val) % 2 == 1:
                for ci in range(1, sws.max_column + 1):
                    sws.cell(row=ri, column=ci).fill = band

    wb.save(path)


def save(main: pd.DataFrame, zsheet: pd.DataFrame, usheet: pd.DataFrame, shortlist: pd.DataFrame) -> None:
    method_df = pd.DataFrame(METHOD_ROWS, columns=["Item", "Detail"])
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        main.to_excel(w, index=False, sheet_name="Strikers")
        shortlist.to_excel(w, index=False, sheet_name="Shortlist")
        usheet.to_excel(w, index=False, sheet_name="Uncertainty")
        zsheet.to_excel(w, index=False, sheet_name="Archetype Z")
        method_df.to_excel(w, index=False, sheet_name="Methodology")
    style_workbook(OUT, main)


def main() -> None:
    print("  Loading Japan II/III strikers …")
    main_df, z_df, u_df = build()
    shortlist_df = build_shortlist(main_df)
    print(f"  {len(main_df)} CF-listed strikers scored")
    save(main_df, z_df, u_df, shortlist_df)
    print(f"  Saved -> {OUT}")
    print(shortlist_df.to_string(index=False))


if __name__ == "__main__":
    main()
