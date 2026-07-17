"""
build_lamberts_model.py
───────────────────────
Build THE LAMBERTS MODEL — a full-universe (165 league) recruitment dataset
combining three real-world recruitment philosophies into one blended score,
plus an Expected Value regression that is independent of (and directly
compared against) the Transfermarkt-sourced "Market value" column Wyscout
ships with.

Sub-models
──────────
  Bentham      — Brentford / FC Midtjylland "Moneyball" statistical model.
                 Rewards high-volume, set-piece-heavy statistical output,
                 translated across leagues via a league-strength coefficient,
                 tuned to the 22-27 prime trade-value age window.
  Jamestown    — Brighton / Union Saint-Gilloise / Hearts multi-club data
                 network model. Rewards technical/creative ceiling (dribbling,
                 progressive carrying, chance creation) found early in
                 lower-profile leagues, damped league-strength penalty,
                 skewed to age 18-23.
  Red Bull     — RB Salzburg pressing/athleticism pipeline model. Rewards
                 duels, aerial power, transition speed and pressing actions,
                 the steepest youth curve of the three (peaks 17-20).

LAMBERTS SCORE = weighted blend of the three (default 35/35/30), re-ranked to
a 0-100 percentile ("Lamberts Index") within each position group.

EXPECTED VALUE — an OLS regression of log(Market value) on Lamberts Index,
age, age^2, log(minutes), position and league tier, fit across every player
with a listed Transfermarkt value. The fitted model then PREDICTS a value for
every player — an estimate that is independent of, and directly benchmarked
against, the actual Transfermarkt figure (Value Gap = Expected − Market).

Usage:
  python3 build_lamberts_model.py
  python3 build_lamberts_model.py --min-minutes 500
  python3 build_lamberts_model.py --output data/Lamberts_Model_Data.js
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
WYSCOUT_DIR = ROOT / "Wyscout Files"
LEAGUE_TIERS_XLSX = ROOT / "League Analysis" / "League Tiers.xlsx"

DEFAULT_MIN_MINUTES = 500
DEFAULT_OUTPUT = ROOT / "data" / "Lamberts_Model_Data.js"

# ── League strength tiers ───────────────────────────────────────────────────

TIER_LABELS = {
    1: "Elite", 2: "Top", 3: "Strong", 4: "Developing", 5: "Lower", 6: "Youth",
}
TIER_MULTIPLIER = {1: 1.00, 2: 0.90, 3: 0.80, 4: 0.68, 5: 0.55, 6: 0.40}

ROMAN = {"II": 2, "III": 3, "IV": 4, "V": 5}
COUNTRY_ALIAS = {
    "Czech": "Czech Republic", "Turkiye": "Türkiye", "Korea": "South Korea",
    "Moldovia": "Moldova", "Kyrgystan": "Kyrgyzstan", "Saudi": "Saudi Arabia",
}
SPECIAL_STEMS = {
    "Czech U17": (6, "Youth", "Czech U17 League"),
    "Czech U19": (6, "Youth", "Czech U19 League"),
    "Japan II III": (4, "Developing", "J2/J3 League"),
}
# Leagues absent from the League Tiers workbook — assigned by comparable size/quality
MANUAL_TIER = {
    "Estonia": 4,     # Meistriliiga — Baltic peer of Latvia/Lithuania (both tier 4)
    "Faroe Islands": 5,
    "USA III": 5,     # USL League One, below USL Championship (tier 4)
}


def _parse_stem(stem: str) -> tuple[str, int]:
    m = re.match(r"^(.*?)(?:\s+-\s+Part\s+[IVX]+)?$", stem)
    base = m.group(1) if m else stem
    tokens = base.split()
    division = 1
    country_tokens = tokens[:]
    if tokens and tokens[-1] in ROMAN:
        division = ROMAN[tokens[-1]]
        country_tokens = tokens[:-1]
    elif tokens and tokens[-1].isdigit():
        division = int(tokens[-1])
        country_tokens = tokens[:-1]
    country = " ".join(country_tokens)
    return COUNTRY_ALIAS.get(country, country), division


def load_league_tier_map() -> dict[str, tuple[int, str, str]]:
    """Return {wyscout_filename_stem: (tier, tier_label, official_league_name)}."""
    tiers = pd.read_excel(LEAGUE_TIERS_XLSX, sheet_name="All Leagues")
    lookup: dict[tuple[str, int], tuple[int, str, str]] = {}
    for _, r in tiers.iterrows():
        try:
            div = int(r["Division"])
        except (ValueError, TypeError):
            continue
        lookup[(str(r["Country"]).strip(), div)] = (
            int(r["Tier"]), str(r["Tier Label"]), str(r["League Name"])
        )

    result: dict[str, tuple[int, str, str]] = {}
    for path in sorted(WYSCOUT_DIR.glob("*.xlsx")):
        stem = path.stem
        if stem in SPECIAL_STEMS:
            result[stem] = SPECIAL_STEMS[stem]
            continue
        country, division = _parse_stem(stem)
        key = (country, division)
        if key in lookup:
            result[stem] = lookup[key]
        elif country in MANUAL_TIER:
            t = MANUAL_TIER[country]
            result[stem] = (t, TIER_LABELS[t], f"{country} (div {division})")
        else:
            # Stem-level manual overrides (e.g. "USA III")
            base_country = stem.split(" II")[0].split(" III")[0].split(" IV")[0].split(" V")[0]
            if stem in MANUAL_TIER:
                t = MANUAL_TIER[stem]
            elif base_country in MANUAL_TIER:
                t = MANUAL_TIER[base_country]
            else:
                t = 4  # conservative default
            result[stem] = (t, TIER_LABELS[t], f"{country} (div {division})")
    return result


# ── Position normalisation ──────────────────────────────────────────────────

POS_MAP: dict[str, str] = {
    "CF": "FW", "SS": "FW",
    "LW": "W", "RW": "W", "LWF": "W", "RWF": "W", "WF": "W",
    "LAMF": "W", "RAMF": "W",
    "AMF": "CM",
    "CMF": "CM", "LCM": "CM", "RCM": "CM", "LCMF": "CM", "RCMF": "CM",
    "DMF": "DM", "LDM": "DM", "RDM": "DM", "LDMF": "DM", "RDMF": "DM",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "GK": "GK",
}
POS_LABELS = {
    "GK": "Goalkeeper", "CB": "Centre-Back", "FB": "Full-Back",
    "DM": "Defensive Mid", "CM": "Central Mid", "W": "Winger", "FW": "Forward",
}


def map_position(pos_str) -> str:
    if not isinstance(pos_str, str):
        return "Other"
    first = pos_str.split(",")[0].strip()
    return POS_MAP.get(first, "Other")


# ── Sub-model metric blueprints (metric, weight) per position group ────────

BENTHAM_BLUEPRINTS: dict[str, list[tuple[str, float]]] = {
    "GK": [("Save rate, %", 3.0), ("Prevented goals per 90", 3.0), ("Aerial duels per 90", 1.5),
           ("Accurate long passes, %", 1.0), ("Exits per 90", 1.5)],
    "CB": [("Successful defensive actions per 90", 2.5), ("Aerial duels won, %", 2.5),
           ("PAdj Interceptions", 2.0), ("Progressive passes per 90", 1.5),
           ("Accurate long passes, %", 1.5), ("Head goals per 90", 1.0)],
    "FB": [("Crosses per 90", 2.0), ("Accurate crosses, %", 1.5), ("Deep completed crosses per 90", 1.5),
           ("xA per 90", 1.5), ("Progressive passes per 90", 1.5), ("Successful defensive actions per 90", 2.0)],
    "DM": [("Successful defensive actions per 90", 2.5), ("PAdj Interceptions", 2.0),
           ("Progressive passes per 90", 2.0), ("Accurate passes, %", 1.5), ("Aerial duels won, %", 1.5)],
    "CM": [("Progressive passes per 90", 2.5), ("Key passes per 90", 2.0), ("xA per 90", 2.0),
           ("Accurate passes, %", 1.5), ("Goals per 90", 1.5), ("Corners per 90", 1.0)],
    "W":  [("xG per 90", 2.0), ("xA per 90", 2.0), ("Dribbles per 90", 1.5),
           ("Crosses per 90", 1.5), ("Accurate crosses, %", 1.0), ("Touches in box per 90", 1.5)],
    "FW": [("Goals per 90", 3.0), ("xG per 90", 2.5), ("Head goals per 90", 2.0),
           ("Aerial duels won, %", 1.5), ("Touches in box per 90", 1.5), ("Shots on target, %", 1.5)],
}

JAMESTOWN_BLUEPRINTS: dict[str, list[tuple[str, float]]] = {
    "GK": [("Accurate passes, %", 2.0), ("Accurate long passes, %", 1.5),
           ("Save rate, %", 2.0), ("Exits per 90", 1.5)],
    "CB": [("Progressive passes per 90", 2.5), ("Accurate passes, %", 2.0), ("Dribbles per 90", 1.0),
           ("Defensive duels won, %", 2.0), ("Interceptions per 90", 1.5)],
    "FB": [("Progressive runs per 90", 2.5), ("Dribbles per 90", 2.0), ("Successful dribbles, %", 1.5),
           ("xA per 90", 2.0), ("Accurate passes, %", 1.5)],
    "DM": [("Progressive passes per 90", 2.5), ("Accurate passes, %", 2.0), ("Dribbles per 90", 1.5),
           ("Key passes per 90", 1.5), ("Successful defensive actions per 90", 1.5)],
    "CM": [("Key passes per 90", 2.5), ("xA per 90", 2.5), ("Progressive runs per 90", 2.0),
           ("Dribbles per 90", 2.0), ("Successful dribbles, %", 1.5)],
    "W":  [("Dribbles per 90", 3.0), ("Successful dribbles, %", 2.0), ("Progressive runs per 90", 2.0),
           ("xA per 90", 2.0), ("Offensive duels won, %", 1.5)],
    "FW": [("xG per 90", 2.5), ("Dribbles per 90", 1.5), ("Touches in box per 90", 2.0),
           ("Successful dribbles, %", 1.5), ("xA per 90", 1.5), ("Goals per 90", 2.0)],
}

REDBULL_BLUEPRINTS: dict[str, list[tuple[str, float]]] = {
    "GK": [("Aerial duels per 90", 2.0), ("Exits per 90", 2.5), ("Save rate, %", 2.0),
           ("Accurate long passes, %", 1.0)],
    "CB": [("Aerial duels won, %", 2.5), ("Defensive duels won, %", 2.5), ("Interceptions per 90", 2.0),
           ("Duels per 90", 1.5), ("Progressive runs per 90", 1.0)],
    "FB": [("Accelerations per 90", 2.5), ("Progressive runs per 90", 2.5), ("Defensive duels won, %", 2.0),
           ("Duels per 90", 1.5), ("Successful defensive actions per 90", 1.5)],
    "DM": [("Defensive duels won, %", 3.0), ("Duels per 90", 2.0), ("Interceptions per 90", 2.5),
           ("Aerial duels won, %", 1.5), ("Accelerations per 90", 1.0)],
    "CM": [("Duels per 90", 2.0), ("Progressive runs per 90", 2.0), ("Accelerations per 90", 2.0),
           ("Fouls suffered per 90", 1.5), ("Successful defensive actions per 90", 2.0)],
    "W":  [("Accelerations per 90", 2.5), ("Progressive runs per 90", 2.5), ("Dribbles per 90", 2.0),
           ("Fouls suffered per 90", 1.5), ("Offensive duels won, %", 1.5)],
    "FW": [("Aerial duels won, %", 2.0), ("Duels per 90", 2.0), ("Accelerations per 90", 2.0),
           ("Offensive duels won, %", 1.5), ("Goals per 90", 2.0)],
}

SUB_MODELS = {
    "bentham":   dict(blueprint=BENTHAM_BLUEPRINTS,   tier_damping=1.00, perf_weight=0.85),
    "jamestown": dict(blueprint=JAMESTOWN_BLUEPRINTS, tier_damping=0.40, perf_weight=0.75),
    "redbull":   dict(blueprint=REDBULL_BLUEPRINTS,   tier_damping=0.80, perf_weight=0.70),
}

LAMBERTS_WEIGHTS = {"bentham": 0.35, "jamestown": 0.35, "redbull": 0.30}


# ── Age curves (0-100), one philosophy each ─────────────────────────────────

def bentham_age(age: float) -> float:
    """Brentford/Midtjylland: peak prime trade-value window 22-27."""
    if pd.isna(age):
        return 50.0
    if 22 <= age <= 27:
        return 100.0
    if age < 22:
        return max(100.0 - (22 - age) * 6.0, 40.0)
    return max(100.0 - (age - 27) * 8.0, 15.0)


def jamestown_age(age: float) -> float:
    """Brighton/USG/Hearts: skewed to early technical ceiling 18-22."""
    if pd.isna(age):
        return 50.0
    if 18 <= age <= 22:
        return 100.0
    if age < 18:
        return 90.0
    if age <= 25:
        return max(100.0 - (age - 22) * 8.0, 76.0)
    return max(100.0 - (age - 22) * 12.0, 10.0)


def redbull_age(age: float) -> float:
    """RB Salzburg: steepest youth curve, peaks 17-20, falls off hard by mid-20s."""
    if pd.isna(age):
        return 50.0
    if age <= 20:
        return 100.0
    if age <= 23:
        return max(100.0 - (age - 20) * 10.0, 70.0)
    return max(100.0 - (age - 20) * 15.0, 5.0)


AGE_CURVES = {"bentham": bentham_age, "jamestown": jamestown_age, "redbull": redbull_age}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_leagues(min_minutes: int) -> pd.DataFrame:
    tier_map = load_league_tier_map()
    paths = sorted(WYSCOUT_DIR.glob("*.xlsx"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        stem = path.stem
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"  [warn] Could not read {path.name}: {e}")
            continue
        tier, tier_label, league_name = tier_map.get(stem, (4, "Developing", stem))
        df = df.copy()
        df["_League"] = stem
        df["_LeagueName"] = league_name
        df["_Tier"] = tier
        df["_TierLabel"] = tier_label
        df["_TierMult"] = TIER_MULTIPLIER[tier]
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True, sort=False)
    print(f"  Loaded {len(paths)} league files → {len(raw)} player rows")

    mins_col = next((c for c in ["Minutes played", "MinutesPlayed", "Minutes"] if c in raw.columns), None)
    raw["_minutes"] = pd.to_numeric(raw[mins_col], errors="coerce").fillna(0) if mins_col else 0
    raw = raw[raw["_minutes"] >= min_minutes].copy()
    print(f"  → {len(raw)} players after {min_minutes}+ minute filter")

    pos_col = next((c for c in ["Position", "Pos"] if c in raw.columns), None)
    raw["_pos_group"] = raw[pos_col].apply(map_position) if pos_col else "Other"
    raw["_full_position"] = raw[pos_col].fillna("Unknown") if pos_col else "Unknown"
    raw = raw[raw["_pos_group"] != "Other"].copy()
    print(f"  → {len(raw)} players with a recognised outfield/GK position")

    raw["_age"] = pd.to_numeric(raw.get("Age"), errors="coerce")
    raw["_mkt_val"] = pd.to_numeric(raw.get("Market value"), errors="coerce").fillna(0)

    return raw.reset_index(drop=True)


# ── Sub-model scoring ────────────────────────────────────────────────────────

def compute_output_percentile(df: pd.DataFrame, blueprint: dict[str, list[tuple[str, float]]],
                               tier_damping: float) -> pd.Series:
    """Percentile (0-100) of tier-adjusted per-90 output, within position group,
    across the ENTIRE combined (all-league) pool — this is what makes the score
    cross-league comparable ('translated' talent, à la Brentford's model)."""
    out = pd.Series(np.nan, index=df.index)
    tier_mult_damped = df["_TierMult"].astype(float) ** tier_damping
    for pos, blueprint_metrics in blueprint.items():
        mask = df["_pos_group"] == pos
        if mask.sum() == 0:
            continue
        idx = df.index[mask]
        score = pd.Series(0.0, index=idx)
        total_w = 0.0
        for metric, w in blueprint_metrics:
            if metric not in df.columns:
                continue
            vals = pd.to_numeric(df.loc[idx, metric], errors="coerce").fillna(0)
            adjusted = vals * tier_mult_damped.loc[idx]
            pct = adjusted.rank(pct=True) * 100
            score += w * pct
            total_w += w
        if total_w > 0:
            score /= total_w
        out.loc[idx] = score
    return out


def compute_submodels(df: pd.DataFrame) -> pd.DataFrame:
    for name, cfg in SUB_MODELS.items():
        perf = compute_output_percentile(df, cfg["blueprint"], cfg["tier_damping"])
        age_curve = df["_age"].apply(AGE_CURVES[name])
        blended = cfg["perf_weight"] * perf.fillna(50.0) + (1 - cfg["perf_weight"]) * age_curve
        df[f"_{name}_score"] = blended.round(2)
    return df


def compute_lamberts(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    total_w = sum(weights.values())
    lamberts_raw = sum(df[f"_{k}_score"] * (w / total_w) for k, w in weights.items())
    df["_lamberts_score"] = lamberts_raw.round(2)

    idx_rank = pd.Series(np.nan, index=df.index)
    for pos in POS_MAP.values():
        mask = df["_pos_group"] == pos
        if mask.sum() == 0:
            continue
        idx_rank.loc[mask] = df.loc[mask, "_lamberts_score"].rank(pct=True) * 100
    df["_lamberts_index"] = idx_rank.round(2)

    def tier(li: float) -> str:
        if pd.isna(li):
            return "UNRANKED"
        if li >= 90:
            return "ELITE"
        if li >= 75:
            return "HIGH POTENTIAL"
        if li >= 55:
            return "SOLID TARGET"
        if li >= 35:
            return "ROTATION"
        return "LONGSHOT"

    df["_lamberts_tier"] = df["_lamberts_index"].apply(tier)
    return df


# ── Expected Value regression (independent of, benchmarked against, TM value) ─

POSITION_ORDER = ["GK", "CB", "FB", "DM", "CM", "W", "FW"]


def fit_expected_value(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """OLS: log1p(MarketValue) ~ Bentham + Jamestown + RedBull + Age + Age^2 + log1p(Minutes)
    + position dummies + league tier. Fit on players with a listed market
    value; predict for everyone. Deliberately uses the three RAW sub-model
    scores rather than the blended Lamberts Index, so Expected Value stays
    an objective, blend-independent estimate — it does not move when a
    user drags the Bentham/Jamestown/Red Bull weight sliders in the
    dashboard, only the Lamberts Index and ranking do. This produces a
    valuation that is DERIVED from performance/age/context data rather
    than copied from Transfermarkt, so it can (and does) diverge from the
    actual listed figure."""
    work = df.copy()
    work["_log_minutes"] = np.log1p(work["_minutes"])
    work["_age_f"] = work["_age"].fillna(work["_age"].median())
    work["_age_sq"] = work["_age_f"] ** 2
    work["_bentham_f"] = work["_bentham_score"].fillna(50.0)
    work["_jamestown_f"] = work["_jamestown_score"].fillna(50.0)
    work["_redbull_f"] = work["_redbull_score"].fillna(50.0)

    feature_cols = ["_bentham_f", "_jamestown_f", "_redbull_f", "_age_f", "_age_sq", "_log_minutes", "_Tier"]
    pos_dummy_cols = [f"_pos_{p}" for p in POSITION_ORDER[1:]]  # GK = baseline
    for p in POSITION_ORDER[1:]:
        work[f"_pos_{p}"] = (work["_pos_group"] == p).astype(float)

    X_cols = feature_cols + pos_dummy_cols
    X_all = work[X_cols].to_numpy(dtype=float)
    X_all = np.column_stack([np.ones(len(work)), X_all])

    train_mask = work["_mkt_val"] > 0
    y_train = np.log1p(work.loc[train_mask, "_mkt_val"].to_numpy(dtype=float))
    X_train = X_all[train_mask.to_numpy()]

    coef, residuals, rank, sv = np.linalg.lstsq(X_train, y_train, rcond=None)

    y_pred_train = X_train @ coef
    ss_res = float(np.sum((y_train - y_pred_train) ** 2))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    y_pred_all = X_all @ coef
    expected_value = np.expm1(y_pred_all)
    expected_value = np.clip(expected_value, 0, None)

    diagnostics = {
        "n_train": int(train_mask.sum()),
        "n_total": int(len(work)),
        "r_squared": round(r2, 4),
        "features": ["intercept"] + X_cols,
        "coefficients": [round(float(c), 5) for c in coef],
    }
    return pd.Series(expected_value, index=df.index), diagnostics


def apply_value_gap(df: pd.DataFrame, expected_value: pd.Series) -> pd.DataFrame:
    df["_expected_value"] = expected_value.round(0)
    has_mv = df["_mkt_val"] > 0
    df["_value_gap"] = np.where(has_mv, df["_expected_value"] - df["_mkt_val"], np.nan)
    df["_value_ratio"] = np.where(has_mv, df["_expected_value"] / df["_mkt_val"], np.nan)

    def tag(row) -> str:
        if row["_mkt_val"] <= 0:
            return "UNLISTED"
        if row["_value_ratio"] >= 1.3:
            return "UNDERVALUED"
        if row["_value_ratio"] <= 0.7:
            return "OVERVALUED"
        return "FAIR"

    df["_value_tag"] = df.apply(tag, axis=1)
    return df


# ── Output ───────────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "player", "team", "league", "leagueName", "country", "tier", "tierLabel",
    "pos", "fullPos", "age", "foot", "height", "contract", "minutes",
    "marketValue", "expectedValue", "valueGap", "valueRatio", "valueTag",
    "benthamScore", "jamestownScore", "redbullScore", "lambertsScore", "lambertsIndex", "lambertsTier",
    "goals90", "xg90", "assists90", "xa90", "progPasses90", "progRuns90",
    "dribbles90", "duelsWon", "aerialWon", "defDuelsWon", "keyPasses90", "saveRate",
]


def build_rows(df: pd.DataFrame) -> list[list]:
    def g(row, col, default=0):
        v = row.get(col, default)
        try:
            if pd.isna(v):
                return default
        except TypeError:
            pass
        return v

    def num(v, nd=2):
        try:
            if pd.isna(v):
                return None
            return round(float(v), nd)
        except (TypeError, ValueError):
            return None

    rows: list[list] = []
    for _, r in df.iterrows():
        contract = r.get("Contract expires")
        if pd.notna(contract):
            try:
                contract = pd.to_datetime(contract).strftime("%Y-%m-%d")
            except Exception:
                contract = str(contract)
        else:
            contract = None

        rows.append([
            g(r, "Player", ""), g(r, "Team", ""), r["_League"], r["_LeagueName"],
            r["_League"].split(" ")[0], int(r["_Tier"]), r["_TierLabel"],
            r["_pos_group"], r["_full_position"],
            int(r["_age"]) if pd.notna(r["_age"]) else None,
            g(r, "Foot", "") or "", num(r.get("Height"), 0), contract, int(r["_minutes"]),
            int(r["_mkt_val"]) if r["_mkt_val"] > 0 else 0,
            int(r["_expected_value"]) if pd.notna(r["_expected_value"]) else 0,
            num(r["_value_gap"], 0), num(r["_value_ratio"], 2), r["_value_tag"],
            num(r["_bentham_score"]), num(r["_jamestown_score"]), num(r["_redbull_score"]),
            num(r["_lamberts_score"]), num(r["_lamberts_index"]), r["_lamberts_tier"],
            num(r.get("Goals per 90")), num(r.get("xG per 90")), num(r.get("Assists per 90")),
            num(r.get("xA per 90")), num(r.get("Progressive passes per 90")),
            num(r.get("Progressive runs per 90")), num(r.get("Dribbles per 90")),
            num(r.get("Duels won, %")), num(r.get("Aerial duels won, %")),
            num(r.get("Defensive duels won, %")), num(r.get("Key passes per 90")),
            num(r.get("Save rate, %")),
        ])
    return rows


def run(min_minutes: int, output: Path) -> None:
    print(f"\n{'='*70}\n  THE LAMBERTS MODEL — building recruitment dataset\n"
          f"  {WYSCOUT_DIR}  ·  min minutes = {min_minutes}\n{'='*70}\n")

    df = load_all_leagues(min_minutes)

    print("Computing Bentham / Jamestown / Red Bull sub-model scores…")
    df = compute_submodels(df)

    print("Blending THE LAMBERTS MODEL composite…")
    df = compute_lamberts(df, LAMBERTS_WEIGHTS)

    print("Fitting Expected Value regression (independent of Transfermarkt)…")
    ev, diagnostics = fit_expected_value(df)
    df = apply_value_gap(df, ev)
    print(f"  R² = {diagnostics['r_squared']}  ·  trained on {diagnostics['n_train']} "
          f"of {diagnostics['n_total']} players with a listed market value")

    print("Serialising rows…")
    rows = build_rows(df)

    league_summary = (
        df.groupby(["_League", "_LeagueName", "_Tier", "_TierLabel"])
        .size().reset_index(name="players")
        .sort_values(["_Tier", "_LeagueName"])
    )
    leagues_out = [
        {"league": r["_League"], "leagueName": r["_LeagueName"], "tier": int(r["_Tier"]),
         "tierLabel": r["_TierLabel"], "players": int(r["players"])}
        for _, r in league_summary.iterrows()
    ]

    tier_counts = df["_Tier"].value_counts().to_dict()

    payload = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "minMinutes": min_minutes,
        "totalPlayers": len(df),
        "leagueCount": df["_League"].nunique(),
        "positionLabels": POS_LABELS,
        "tierLabels": TIER_LABELS,
        "tierMultiplier": TIER_MULTIPLIER,
        "tierCounts": {str(k): int(v) for k, v in tier_counts.items()},
        "lambertsWeights": LAMBERTS_WEIGHTS,
        "regression": diagnostics,
        "leagues": leagues_out,
        "columns": OUTPUT_COLUMNS,
        "rows": rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("const LAMBERTS_DATA = ")
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
        f.write(";\n")

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {len(rows)} players · {payload['leagueCount']} leagues · "
          f"{size_mb:.1f} MB → {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build The Lamberts Model recruitment dataset")
    parser.add_argument("--min-minutes", type=int, default=DEFAULT_MIN_MINUTES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(min_minutes=args.min_minutes, output=args.output)
