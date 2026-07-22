"""
recruitment_model.py
─────────────────────
Complete recruitment model built on the full Wyscout league universe
("Wyscout Files/*.xlsx").

Unlike the earlier Lamberts Index model — where "Model Value" is just the
player's own Transfermarkt market value rescaled by their SQS rank — this
model derives a euro valuation *from performance metrics*: a regression is
fit once, across the whole player population, mapping performance +
age + league strength + minutes + position onto Transfermarkt market value.
Transfermarkt values are used only to calibrate the monetary scale of that
regression; the value assigned to any individual player is the model's
output for their own metrics, not a lookup of their own listed price. See
`VALUE_MODEL_NOTES` for the full methodology writeup.

Five building blocks, each independently callable:
  1. League tiering & youth/senior classification  (parse_league_filename,
     build_league_index)
  2. Player loading + IMPECT-style composite scores (reuses wyscout_model)
  3. Peak-age trajectory tagging                    (compute_peak_trajectory)
  4. Metrics-derived EUR valuation model             (compute_value_model)
  5. Czech-First-League-calibrated physical/tempo fit (compute_physical_fit)

Plus club- and league-level power rankings, split senior vs youth.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from wyscout_model import (
    WYSCOUT_POSITION_MAP,
    compute_wyscout_scores,
)
from scouting_model import AnomalyEngine, SimilarityEngine, SetPieceAnalyzer

ROOT = Path(__file__).parent
WYSCOUT_DIR = ROOT / "Wyscout Files"
LEAGUE_TIERS_PATH = ROOT / "League Analysis" / "League Tiers.xlsx"

SKIP_FILES = {
    "FCHK Model V3 - Loaded Leagues", "FCHK Model V3 - Model Input",
    "FCHK Model V3 - Player Scores", "FCHK Model V3 - Player Styles",
    "FCHK Model V3 - Recruitment Scores", "FCHK Model V3 - Smart Club Closeness",
    "FCHK Model V3 - Summary", "FCHK Model V3 Scores", "FCHK Scouting Report",
    "Leagues Overview", "Wyscout Anomaly Report", "Wyscout Full Scouting Report",
}

BENCHMARK_LEAGUE_FILE = "Czech"   # Czech First League — quick, physical benchmark
DEFAULT_MIN_MINUTES_SENIOR = 500
DEFAULT_MIN_MINUTES_YOUTH = 250

# ── 1. League filename parsing ──────────────────────────────────────────────

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}

_COUNTRY_ALIAS = {
    "Czech": "Czech Republic",
    "Korea": "South Korea",
    "Kyrgystan": "Kyrgyzstan",
    "Moldovia": "Moldova",
    "Saudi": "Saudi Arabia",
    "Turkiye": "Türkiye",
}

_YOUTH_NAME_RE = re.compile(r"\bU1[5-9]\b|\bU2[0-3]\b|Youth|Junior", re.IGNORECASE)


def parse_league_filename(stem: str) -> tuple[str, str, bool]:
    """
    Parse a Wyscout filename stem into (country, division_token, is_youth_by_name).

    "Czech II"            -> ("Czech Republic", "2", False)
    "Czech U19"            -> ("Czech Republic", "U19", True)
    "Germany 4 - Part II"  -> ("Germany", "4", False)
    "Japan II III"         -> ("Japan", "2", False)   # combined J2/J3 export
    """
    s = re.sub(r"\s*-\s*Part\s+[IVXLC]+$", "", stem).strip()
    is_youth = bool(_YOUTH_NAME_RE.search(s))
    tokens = s.split()
    division_token = "1"
    country_tokens = tokens

    if len(tokens) >= 2:
        last = tokens[-1]
        if re.fullmatch(r"U\d{1,2}", last, re.IGNORECASE):
            division_token = last.upper()
            country_tokens = tokens[:-1]
        elif last.isdigit():
            division_token = last
            country_tokens = tokens[:-1]
        elif last.upper() in _ROMAN:
            if len(tokens) >= 3 and tokens[-2].upper() in _ROMAN:
                division_token = str(_ROMAN[tokens[-2].upper()])
                country_tokens = tokens[:-2]
            else:
                division_token = str(_ROMAN[last.upper()])
                country_tokens = tokens[:-1]

    country = " ".join(country_tokens)
    country = _COUNTRY_ALIAS.get(country, country)
    return country, division_token, is_youth


# Numeric strength (0-100) used as a regression feature and for tier-adjusting
# club/league power scores. Tier 1 = strongest.
TIER_STRENGTH = {1: 100.0, 2: 85.0, 3: 70.0, 4: 55.0, 5: 40.0, 6: 22.0}
TIER_LABELS = {
    1: "Elite", 2: "Top", 3: "Strong", 4: "Developing", 5: "Lower", 6: "Youth/Grassroots",
}


def _load_league_reference() -> pd.DataFrame:
    ref = pd.read_excel(LEAGUE_TIERS_PATH, sheet_name="All Leagues")
    ref["Division"] = ref["Division"].astype(str)
    return ref


def build_league_index() -> pd.DataFrame:
    """
    One row per Wyscout league file: File, Country, Division, IsYouthLeague,
    Tier (1-6), TierLabel, LeagueDisplayName, LeagueStrength (0-100).

    Tiers are inherited from the curated League Tiers.xlsx pyramid (division
    position in each country's football pyramid — not derived from market
    value). Files with no match get a conservative fallback tier from their
    division number so every league still ranks somewhere.
    """
    if not WYSCOUT_DIR.exists():
        return pd.DataFrame()

    files = sorted(p.stem for p in WYSCOUT_DIR.glob("*.xlsx") if p.stem not in SKIP_FILES)
    ref = _load_league_reference()

    rows = []
    for stem in files:
        country, division, is_youth_name = parse_league_filename(stem)
        match = ref[(ref["Country"] == country) & (ref["Division"] == division)]
        if not match.empty:
            r = match.iloc[0]
            tier = int(r["Tier"])
            tier_label = str(r["Tier Label"])
            display = str(r["League Name"])
        else:
            # Fallback: conservative tier from division depth / youth flag
            if is_youth_name or division.upper().startswith("U"):
                tier = 6
            else:
                try:
                    div_num = int(division)
                except ValueError:
                    div_num = 2
                tier = min(5, 3 + max(div_num - 1, 0))
            tier_label = TIER_LABELS[tier]
            display = f"{stem} (unranked — fallback tier)"

        is_youth_league = tier == 6 or is_youth_name or division.upper().startswith("U")
        rows.append({
            "League": stem,
            "Country": country,
            "Division": division,
            "IsYouthLeague": is_youth_league,
            "Tier": tier,
            "TierLabel": tier_label,
            "LeagueDisplayName": display,
            "LeagueStrength": TIER_STRENGTH[tier],
        })

    return pd.DataFrame(rows)


# ── Youth-team detection within senior leagues (reserve/academy squads) ────

_YOUTH_TEAM_RE = re.compile(
    r"\bU1[4-9]\b|\bU2[0-3]\b|\bYouth\b|\bJunior(?:es)?\b|\bAcademy\b|"
    r"\bReserves?\b|\bSub-?23\b|\bJuvenil\b|\bCadete\b|\b(?:II|B)$",
    re.IGNORECASE,
)


def classify_team_type(team: str) -> str:
    """Heuristic Senior / Youth classification for a team name."""
    if not isinstance(team, str) or not team.strip():
        return "Senior"
    return "Youth" if _YOUTH_TEAM_RE.search(team.strip()) else "Senior"


# ── 2. Player loading ───────────────────────────────────────────────────────

_SKIP_NUMERIC_COLS = {
    "Player", "Team", "Team within selected timeframe", "Position", "PositionGroup",
    "League", "Country", "Division", "TeamType", "LeagueDisplayName", "TierLabel",
    "Birth country", "Passport country", "Foot", "On loan", "Contract expires",
}


def load_raw_players(
    league_index: pd.DataFrame,
    min_minutes_senior: int = DEFAULT_MIN_MINUTES_SENIOR,
    min_minutes_youth: int = DEFAULT_MIN_MINUTES_YOUTH,
    leagues: list[str] | None = None,
) -> pd.DataFrame:
    """Load every player row across the Wyscout league universe with league metadata attached."""
    idx = league_index if leagues is None else league_index[league_index["League"].isin(leagues)]

    frames: list[pd.DataFrame] = []
    for _, meta in idx.iterrows():
        path = WYSCOUT_DIR / f"{meta['League']}.xlsx"
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"  [warn] could not read {path.name}: {e}")
            continue
        df.columns = [str(c).strip() for c in df.columns]
        meta_df = pd.DataFrame([meta.to_dict()] * len(df), index=df.index)
        df = pd.concat([df, meta_df], axis=1)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True).copy()

    # First listed position only
    pos_col = next((c for c in ["Position", "Pos"] if c in raw.columns), None)
    if pos_col:
        raw[pos_col] = raw[pos_col].astype(str).str.split(",").str[0].str.strip()
        raw["PositionGroup"] = raw[pos_col].map(WYSCOUT_POSITION_MAP).fillna("Other")
    else:
        raw["PositionGroup"] = "Other"

    # Numeric coercion for everything except identifier/label columns
    for col in raw.columns:
        if col not in _SKIP_NUMERIC_COLS:
            c = pd.to_numeric(raw[col], errors="coerce")
            if c.notna().any():
                raw[col] = c

    mins_col = next((c for c in ["Minutes played", "MinutesPlayed"] if c in raw.columns), None)
    raw["_minutes"] = pd.to_numeric(raw[mins_col], errors="coerce").fillna(0) if mins_col else 0

    # "Team" is the player's current/most recent club overall, which can be a
    # different club (even a different league) to where these particular stats
    # were recorded. "Team within selected timeframe" is the club the stats
    # actually belong to — use that for club rosters/rankings whenever present.
    tf_col = "Team within selected timeframe"
    if tf_col in raw.columns and "Team" in raw.columns:
        raw["Club"] = raw[tf_col].where(raw[tf_col].notna() & (raw[tf_col].astype(str).str.strip() != ""), raw["Team"])
    elif "Team" in raw.columns:
        raw["Club"] = raw["Team"]
    else:
        raw["Club"] = "Unknown"

    raw["TeamType"] = np.where(
        raw["IsYouthLeague"],
        "Youth",
        raw["Club"].apply(classify_team_type),
    )

    min_needed = np.where(raw["TeamType"] == "Youth", min_minutes_youth, min_minutes_senior)
    raw = raw.loc[raw["_minutes"] >= min_needed].copy()
    raw = raw.loc[raw["PositionGroup"] != "Other"].reset_index(drop=True)

    if "Player" in raw.columns:
        raw["PlayerName"] = raw["Player"]
    if "Age" in raw.columns:
        raw["AgeYears"] = pd.to_numeric(raw["Age"], errors="coerce")

    return raw


# ── 3. Peak-age trajectory ──────────────────────────────────────────────────

# (peak_start, peak_end) in years, by WYSCOUT_POSITION_MAP group
PEAK_WINDOWS: dict[str, tuple[int, int]] = {
    "GK": (27, 33),
    "CB": (26, 30),
    "FB": (24, 28),
    "DM": (25, 29),
    "CM": (24, 28),
    "AM": (23, 27),
    "W":  (22, 27),
    "ST": (24, 28),
}


def compute_peak_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age = pd.to_numeric(df.get("AgeYears"), errors="coerce")

    starts = df["PositionGroup"].map(lambda p: PEAK_WINDOWS.get(p, (24, 28))[0])
    ends = df["PositionGroup"].map(lambda p: PEAK_WINDOWS.get(p, (24, 28))[1])

    df["PeakWindowStart"] = starts
    df["PeakWindowEnd"] = ends
    df["YearsFromPeakStart"] = age - starts

    def tag(a, s, e):
        if pd.isna(a):
            return "Unknown"
        if a < s:
            return "Rising"
        if a <= e:
            return "Peak Window"
        return "Past Peak"

    df["TrajectoryTag"] = [
        tag(a, s, e) for a, s, e in zip(age, starts, ends)
    ]
    df["PastPeak"] = df["TrajectoryTag"] == "Past Peak"
    return df


# ── 4. Metrics-derived EUR valuation model ──────────────────────────────────

VALUE_MODEL_NOTES = """
HOW "MODEL VALUE (€)" IS BUILT
────────────────────────────────
Model Value is NOT a copy or rescale of a player's own Transfermarkt market
value. It is the output of a single regression fit once across the entire
player pool:

    ln(Market Value + 1)  ~  Composite performance score (percentile within the
                              player's own league, then scaled down for how
                              weak/strong that league is — so a striker
                              running riot in a weak division doesn't get
                              mistaken for one doing it in a strong one)
                            + Age + Age²
                            + ln(Minutes played)
                            + League Strength (pyramid-position based, 0-100)
                            + Position

Market values are used only in aggregate, to teach the model what euro
amount the transfer market typically attaches to a given level of output,
age, league context and position. Once fit, the coefficients are applied to
EVERY player's own metrics — including players with no listed market value —
to produce "Model Value (€)": what the market SHOULD pay for a player with
those underlying numbers, independent of what any single player happens to
be currently priced at.

Undervalued Ratio = Model Value ÷ Market Value. A ratio well above 1.0 means
the player's output/age/league profile is worth more than the market
currently prices them at — the definition of undervalued used throughout
this workbook. Players with no market value on file still get a Model Value
(useful for lower-league / youth scouting) but no ratio ("No Market Comp").
"""

_POSITION_ORDER = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]


def _standardize(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(mat, axis=0)
    sig = np.nanstd(mat, axis=0)
    sig = np.where(sig == 0, 1e-9, sig)
    return (mat - mu) / sig, mu, sig


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 5.0) -> np.ndarray:
    n, p = X.shape
    Xc = np.hstack([np.ones((n, 1)), X])
    reg = np.eye(p + 1) * lam
    reg[0, 0] = 0.0
    beta = np.linalg.solve(Xc.T @ Xc + reg, Xc.T @ y)
    return beta


def _ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    Xc = np.hstack([np.ones((n, 1)), X])
    return Xc @ beta


def compute_value_model(df: pd.DataFrame, ridge_lambda: float = 5.0) -> pd.DataFrame:
    """Fit the metrics -> EUR regression and attach ModelValueEUR / ValueRatio / ValueTier."""
    df = df.copy()

    comp = pd.to_numeric(df.get("AdjustedCompositeScore"), errors="coerce").fillna(50.0)
    comp_z = pd.Series(norm.ppf(((comp.rank(pct=True) * 0.998) + 0.001).clip(0.001, 0.999)), index=comp.index)
    age = pd.to_numeric(df.get("AgeYears"), errors="coerce").fillna(25.0)
    minutes = pd.to_numeric(df.get("_minutes"), errors="coerce").fillna(0.0)
    strength = pd.to_numeric(df.get("LeagueStrength"), errors="coerce").fillna(50.0)

    pos_dummies = pd.get_dummies(
        df["PositionGroup"].where(df["PositionGroup"].isin(_POSITION_ORDER), "CM"),
    )
    for p in _POSITION_ORDER:
        if p not in pos_dummies.columns:
            pos_dummies[p] = 0
    pos_dummies = pos_dummies[_POSITION_ORDER].drop(columns=["CM"])  # CM = baseline

    feat = pd.DataFrame({
        "comp": comp,
        "comp_z": comp_z,
        "age": age,
        "age2": age ** 2,
        "ln_minutes": np.log1p(minutes),
        "strength": strength,
    })
    feat = pd.concat([feat, pos_dummies.astype(float)], axis=1)

    mv = pd.to_numeric(df.get("Market value"), errors="coerce").fillna(0.0)
    df["_mkt_val"] = mv

    train_mask = (mv > 1000) & (minutes >= 300)
    X_all = feat.values.astype(float)
    X_train, mu, sig = _standardize(X_all[train_mask.values])
    X_all_std = (X_all - mu) / sig

    if train_mask.sum() < 30:
        # Not enough signal to fit — fall back to composite-only scaling
        df["ModelValueEUR"] = (comp / 100.0 * 2_000_000).round(-3)
        df["_value_r2"] = np.nan
        df["ProjectedPeakValueEUR"] = df["ModelValueEUR"]
    else:
        y_train = np.log1p(mv[train_mask].values)
        beta = _ridge_fit(X_train, y_train, lam=ridge_lambda)

        y_pred_train = _ridge_predict(X_train, beta)
        ss_res = np.sum((y_train - y_pred_train) ** 2)
        ss_tot = np.sum((y_train - y_train.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        y_pred_all = _ridge_predict(X_all_std, beta)
        df["ModelValueEUR"] = np.expm1(y_pred_all).clip(min=0).round(-3)
        df["_value_r2"] = r2

        # "Aging into peak" projection: same player (same quality, league,
        # minutes, position) but at the age their position typically peaks —
        # isolates the pure age-value effect the regression already learned,
        # without speculating about further skill development. This is the
        # basis for the Brighton-style "development upside" read: what does
        # this player's current output become worth once they reach the age
        # window the market usually pays most for at their position.
        peak_start = df["PositionGroup"].map(lambda p: PEAK_WINDOWS.get(p, (24, 28))[0]).astype(float)
        feat_peak = feat.copy()
        feat_peak["age"] = peak_start
        feat_peak["age2"] = peak_start ** 2
        X_peak_std = (feat_peak.values.astype(float) - mu) / sig
        y_pred_peak = _ridge_predict(X_peak_std, beta)
        projected = np.expm1(y_pred_peak).clip(min=0).round(-3)
        # No further aging upside once already at/past that window
        projected = np.where(age.values >= peak_start.values, df["ModelValueEUR"].values, projected)
        df["ProjectedPeakValueEUR"] = np.maximum(projected, df["ModelValueEUR"].values)

    df["DevelopmentUpsideEUR"] = (df["ProjectedPeakValueEUR"] - df["ModelValueEUR"]).clip(lower=0)
    df["ValueGapEUR"] = df["ModelValueEUR"] - df["_mkt_val"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(df["_mkt_val"] > 1000, df["ModelValueEUR"] / df["_mkt_val"], np.nan)
    df["ValueRatio"] = ratio

    def tier(r: float) -> str:
        if pd.isna(r):
            return "NO MARKET COMP"
        if r >= 2.0:
            return "ELITE VALUE"
        if r >= 1.5:
            return "HIGH VALUE"
        if r >= 1.15:
            return "VALUE"
        if r >= 0.85:
            return "FAIR VALUE"
        return "OVERPRICED"

    df["ValueTier"] = df["ValueRatio"].apply(tier)
    return df


# ── 4b. Brighton mechanics — buy low, develop, resell ───────────────────────

BRIGHTON_NOTES = """
BRIGHTON MECHANICS — BUY LOW, DEVELOP, RESELL
────────────────────────────────────────────
Modelled on Brighton & Hove Albion's own recruitment approach: a wide global
scouting net into mid-tier and unfashionable leagues, a strong preference for
players still short of their prime, and a willingness to pay for underlying
data over reputation — buying early, developing on the pitch, and banking the
resale profit once the market catches up.

Brighton Score blends four signals per player:
  30%  Age Fit         — peaks at 19-21, decays either side, roughly zero by 28
  30%  Quality         — Composite Score (cross-league adjusted performance)
  25%  Undervaluation  — percentile rank of Value Ratio (Model Value / Market Value)
  15%  Trajectory      — Rising scores highest, Peak Window partial, Past Peak excluded

Development Upside (€) reuses the valuation regression with one change: the
player's age is swapped for their position's typical peak-age, holding
quality, league and minutes fixed. That isolates the pure "aging into peak"
effect the market already pays for — the closest honest read on resale
upside without speculating about further skill growth.
"""

_AGE_SCORE_IDEAL = (19.0, 21.0)


def _age_fit_score(age: float) -> float:
    if pd.isna(age):
        return 40.0
    lo, hi = _AGE_SCORE_IDEAL
    if lo <= age <= hi:
        return 100.0
    if age < lo:
        return max(0.0, 100.0 - (lo - age) * 16.0)
    return max(0.0, 100.0 - (age - hi) * 13.0)


_BRIGHTON_TRAJ_SCORE = {"Rising": 100.0, "Peak Window": 55.0, "Past Peak": 0.0, "Unknown": 40.0}


def compute_brighton_mechanics(df: pd.DataFrame) -> pd.DataFrame:
    """Score every player on fit to Brighton's data-driven, buy-low/develop/resell recruitment model."""
    df = df.copy()
    age = pd.to_numeric(df.get("AgeYears"), errors="coerce")

    age_score = age.apply(_age_fit_score)

    ratio = pd.to_numeric(df.get("ValueRatio"), errors="coerce")
    log_ratio = np.log(ratio.clip(lower=0.05))
    value_score = (log_ratio.rank(pct=True) * 100).fillna(45.0)

    quality_score = pd.to_numeric(df.get("AdjustedCompositeScore"), errors="coerce").fillna(40.0)
    traj_score = df["TrajectoryTag"].map(_BRIGHTON_TRAJ_SCORE).fillna(40.0)

    df["BrightonScore"] = (
        0.30 * age_score + 0.30 * quality_score + 0.25 * value_score + 0.15 * traj_score
    ).round(1)

    df["BrightonEligible"] = (age.fillna(99) <= 25) & (df["TrajectoryTag"] != "Past Peak")

    def label(score: float, eligible: bool) -> str:
        if not eligible:
            return "Not A Fit"
        if score >= 75:
            return "Prime Target"
        if score >= 62:
            return "Strong Fit"
        if score >= 48:
            return "Speculative"
        return "Long Shot"

    df["BrightonLabel"] = [label(s, e) for s, e in zip(df["BrightonScore"], df["BrightonEligible"])]
    return df


# ── 5. Czech-First-League-calibrated physical / quick-league fit ───────────

PHYSICAL_DUEL_METRICS = [
    "Duels per 90", "Duels won, %", "Defensive duels per 90", "Defensive duels won, %",
    "Aerial duels per 90", "Aerial duels won, %", "Sliding tackles per 90",
]
TEMPO_METRICS = [
    "Accelerations per 90", "Progressive runs per 90", "Dribbles per 90",
    "Received long passes per 90", "Fouls suffered per 90",
]
PHYSICAL_FIT_WEIGHTS: dict[str, float] = {
    "Duels per 90": 1.5, "Duels won, %": 2.0,
    "Defensive duels per 90": 1.0, "Defensive duels won, %": 2.0,
    "Aerial duels per 90": 1.5, "Aerial duels won, %": 2.0,
    "Sliding tackles per 90": 1.0,
    "Accelerations per 90": 2.0, "Progressive runs per 90": 1.5,
    "Dribbles per 90": 1.0, "Received long passes per 90": 1.0,
    "Fouls suffered per 90": 1.0,
}


def compute_physical_fit(
    df: pd.DataFrame,
    benchmark_league: str = BENCHMARK_LEAGUE_FILE,
    min_benchmark_n: int = 5,
) -> pd.DataFrame:
    """
    Score every player on fit to a quick, physical league, calibrated against
    the Czech First League's own per-position metric distributions (not the
    player's own league). A high score means the player's duel, aerial and
    tempo output already matches — or exceeds — what starters in that league
    produce, at the same position.
    """
    df = df.copy()
    metrics = [m for m in (PHYSICAL_DUEL_METRICS + TEMPO_METRICS) if m in df.columns]
    if not metrics:
        df["PhysicalLeagueFitScore"] = 50.0
        df["PhysicalLeagueFitLabel"] = "Unavailable"
        return df

    bench = df[df["League"] == benchmark_league]
    global_stats = {m: (df[m].mean(), df[m].std() or 1e-9) for m in metrics}

    z_total = pd.Series(0.0, index=df.index)
    total_w = 0.0

    for pos, grp_idx in df.groupby("PositionGroup").groups.items():
        bench_pos = bench[bench["PositionGroup"] == pos]
        rows = df.loc[grp_idx]
        for m in metrics:
            w = PHYSICAL_FIT_WEIGHTS.get(m, 1.0)
            if bench_pos[m].notna().sum() >= min_benchmark_n:
                mu, sig = bench_pos[m].mean(), (bench_pos[m].std() or 1e-9)
            else:
                mu, sig = global_stats[m]
            vals = pd.to_numeric(rows[m], errors="coerce").fillna(mu)
            z_total.loc[grp_idx] += w * (vals - mu) / sig

    for m in metrics:
        total_w += PHYSICAL_FIT_WEIGHTS.get(m, 1.0)

    z_avg = z_total / (total_w or 1.0)
    df["PhysicalLeagueFitScore"] = (norm.cdf(z_avg.values) * 100).round(1)

    def label(v: float) -> str:
        if v >= 80:
            return "Excellent Fit"
        if v >= 65:
            return "Good Fit"
        if v >= 45:
            return "Moderate Fit"
        return "Below Profile"

    df["PhysicalLeagueFitLabel"] = df["PhysicalLeagueFitScore"].apply(label)
    return df


# ── Club & league power rankings ────────────────────────────────────────────

_SUBSCORE_COLS = [
    "CompositeRecruitmentScore", "ScoringThreatScore", "CreativeProgressionScore",
    "DefensiveDisruptionScore", "PressingScore", "BallSecurityScore",
    "ExpectedThreatScore", "AerialScore",
]


def build_club_rankings(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """
    scope: 'Senior' or 'Youth'. Minutes-weighted club rating, tier-adjusted
    for the strength of the league the club plays in.
    """
    pool = df[df["TeamType"] == scope].copy()
    if pool.empty:
        return pd.DataFrame()

    def wavg(g: pd.DataFrame, col: str) -> float:
        w = g["_minutes"].clip(lower=1)
        vals = pd.to_numeric(g[col], errors="coerce")
        mask = vals.notna()
        if not mask.any():
            return np.nan
        return float(np.average(vals[mask], weights=w[mask]))

    rows = []
    for (team, league), g in pool.groupby(["Club", "League"]):
        row = {
            "Team": team,
            "League": league,
            "Country": g["Country"].iloc[0],
            "TierLabel": g["TierLabel"].iloc[0],
            "Tier": int(g["Tier"].iloc[0]),
            "LeagueStrength": float(g["LeagueStrength"].iloc[0]),
            "Players": g["Player"].nunique() if "Player" in g.columns else len(g),
            "TotalMinutes": int(g["_minutes"].sum()),
        }
        for c in _SUBSCORE_COLS:
            if c in g.columns:
                row[c] = round(wavg(g, c), 1)
        rows.append(row)

    club = pd.DataFrame(rows)
    club["RawClubScore"] = club["CompositeRecruitmentScore"]
    club["TierAdjustedScore"] = (
        club["RawClubScore"] * (0.55 + 0.45 * club["LeagueStrength"] / 100.0)
    ).round(1)

    club = club.sort_values(["Tier", "TierAdjustedScore"], ascending=[True, False]).reset_index(drop=True)
    club.insert(0, "LeagueRank", club.groupby("League")["TierAdjustedScore"].rank(ascending=False, method="first").astype(int))
    club = club.sort_values("TierAdjustedScore", ascending=False).reset_index(drop=True)
    club.insert(0, "PowerRank", range(1, len(club) + 1))
    return club


def build_league_rankings(club_rankings: pd.DataFrame, league_index: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Aggregate club power scores up to league level, Tier ordered."""
    idx = league_index[league_index["IsYouthLeague"] == (scope == "Youth")]
    if club_rankings.empty:
        base = idx.copy()
        base["ClubsRated"] = 0
        base["AvgClubScore"] = np.nan
        base["TopClub"] = ""
    else:
        agg = club_rankings.groupby("League").agg(
            ClubsRated=("Team", "nunique"),
            AvgClubScore=("TierAdjustedScore", "mean"),
        ).reset_index()
        top_club = (
            club_rankings.sort_values("TierAdjustedScore", ascending=False)
            .drop_duplicates("League")[["League", "Team"]]
            .rename(columns={"Team": "TopClub"})
        )
        agg = agg.merge(top_club, on="League", how="left")
        base = idx.merge(agg, on="League", how="left")
        base["ClubsRated"] = base["ClubsRated"].fillna(0).astype(int)
        base["TopClub"] = base["TopClub"].fillna("")

    base["AvgClubScore"] = base["AvgClubScore"].round(1)
    base = base.sort_values(["Tier", "AvgClubScore"], ascending=[True, False]).reset_index(drop=True)
    base.insert(0, "PowerRank", range(1, len(base) + 1))
    return base


# ── 6. Role archetypes — statistical playing-style sub-types ───────────────

ROLE_ARCHETYPES: dict[str, dict[str, dict[str, float]]] = {
    "GK": {
        "Sweeper Keeper": {"Exits per 90": 2.5, "Accurate passes, %": 1.5, "Accurate long passes, %": 1.0, "Aerial duels per 90": 1.0},
        "Shot-Stopper": {"Save rate, %": 3.0, "Prevented goals per 90": 2.5},
    },
    "CB": {
        "Ball-Playing CB": {"Accurate passes, %": 2.0, "Progressive passes per 90": 2.5, "Passes to final third per 90": 1.5, "Long passes per 90": 1.0},
        "Aggressive Stopper": {"Defensive duels per 90": 2.0, "Defensive duels won, %": 2.5, "Interceptions per 90": 2.0, "Aerial duels won, %": 1.5, "Sliding tackles per 90": 1.0},
    },
    "FB": {
        "Attacking FB": {"Crosses per 90": 2.0, "xA per 90": 2.0, "Progressive runs per 90": 2.0, "Accurate crosses, %": 1.5, "Assists per 90": 1.0},
        "Defensive FB": {"Defensive duels won, %": 2.5, "Interceptions per 90": 2.0, "Aerial duels won, %": 1.5, "Successful defensive actions per 90": 1.5},
    },
    "DM": {
        "Deep Playmaker": {"Passes per 90": 2.0, "Accurate passes, %": 2.0, "Progressive passes per 90": 2.5, "Passes to final third per 90": 1.5},
        "Ball-Winner": {"Defensive duels won, %": 2.5, "Interceptions per 90": 2.5, "PAdj Interceptions": 2.0, "Aerial duels won, %": 1.0},
    },
    "CM": {
        "Progressor": {"Progressive passes per 90": 2.5, "Progressive runs per 90": 2.0, "Accurate passes, %": 1.5},
        "Creator": {"Key passes per 90": 2.5, "xA per 90": 2.5, "Smart passes per 90": 1.5, "Through passes per 90": 1.0},
        "Box-to-Box": {"Successful defensive actions per 90": 1.5, "Goals per 90": 1.5, "Progressive runs per 90": 1.5, "Duels per 90": 1.5},
    },
    "AM": {
        "Creative 10": {"Key passes per 90": 2.5, "xA per 90": 2.5, "Through passes per 90": 1.5, "Smart passes per 90": 1.5},
        "Second Striker": {"Goals per 90": 2.5, "xG per 90": 2.0, "Touches in box per 90": 2.0, "Shots per 90": 1.0},
    },
    "W": {
        "Direct Dribbler": {"Dribbles per 90": 2.5, "Successful dribbles, %": 1.5, "Progressive runs per 90": 2.0, "Accelerations per 90": 1.5},
        "Creative Winger": {"xA per 90": 2.5, "Key passes per 90": 2.0, "Crosses per 90": 1.5, "Accurate crosses, %": 1.0},
    },
    "ST": {
        "Poacher": {"Goals per 90": 3.0, "xG per 90": 2.0, "Touches in box per 90": 2.0, "Goal conversion, %": 1.5},
        "Target Man": {"Aerial duels won, %": 2.5, "Head goals per 90": 2.0, "Received long passes per 90": 1.5},
        "Complete Forward": {"Goals per 90": 1.5, "xA per 90": 1.5, "Dribbles per 90": 1.0, "Progressive runs per 90": 1.0, "Aerial duels won, %": 1.0},
    },
}


def compute_role_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify every player into a statistical playing-style archetype within
    their position group (e.g. Ball-Playing CB vs Aggressive Stopper) — the
    specific, recruitable profile a data-driven scouting department targets,
    rather than a bare position label.
    """
    df = df.copy()
    df["RoleArchetype"] = "Unclassified"
    df["RoleArchetypeScore"] = np.nan

    for pos, archetypes in ROLE_ARCHETYPES.items():
        mask = df["PositionGroup"] == pos
        if mask.sum() == 0:
            continue
        grp = df.loc[mask]
        scores: dict[str, pd.Series] = {}
        for name, weights in archetypes.items():
            total = pd.Series(0.0, index=grp.index)
            total_w = 0.0
            for metric, w in weights.items():
                if metric not in grp.columns:
                    continue
                vals = pd.to_numeric(grp[metric], errors="coerce")
                mu = vals.mean()
                sig = vals.std() or 1e-9
                total = total + w * (vals.fillna(mu) - mu) / sig
                total_w += w
            scores[name] = total / (total_w or 1.0)

        score_df = pd.DataFrame(scores)
        best_name = score_df.idxmax(axis=1)
        best_z = score_df.max(axis=1)
        df.loc[mask, "RoleArchetype"] = best_name
        df.loc[mask, "RoleArchetypeScore"] = (norm.cdf(best_z.values) * 100).round(1)

    return df


# ── 7. Squad needs analysis ─────────────────────────────────────────────────

HRADEC_CLUB = "Hradec Králové"
HRADEC_LEAGUE = "Czech"


def compute_squad_needs(
    df: pd.DataFrame,
    club: str = HRADEC_CLUB,
    league: str = HRADEC_LEAGUE,
    top_n_targets: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Benchmark a club's actual current senior squad, position by position,
    against the full cross-league player pool (AdjustedCompositeScore is
    already cross-league comparable). Positions where the club's own best
    player ranks low get flagged as priority needs, each paired with
    recommended upgrade targets pulled from the wider database.

    Returns (needs_summary, priority_targets) — one row per position, and
    one row per recommended target respectively.
    """
    squad = df[(df["Club"] == club) & (df["League"] == league) & (df["TeamType"] == "Senior")]
    pool = df[df["TeamType"] == "Senior"]
    if squad.empty or pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    target_rows = []

    for pos in _POSITION_ORDER:
        pos_pool = pool[pool["PositionGroup"] == pos]
        if pos_pool.empty:
            continue
        pos_squad = squad[squad["PositionGroup"] == pos].sort_values("AdjustedCompositeScore", ascending=False)

        if pos_squad.empty:
            starter_name, starter_age, starter_score, starter_pct = "— none registered —", None, 0.0, 0.0
        else:
            starter = pos_squad.iloc[0]
            starter_name = starter["Player"]
            starter_age = starter.get("AgeYears")
            starter_score = float(starter["AdjustedCompositeScore"])
            starter_pct = float((pos_pool["AdjustedCompositeScore"] < starter_score).mean() * 100)

        if starter_pct >= 70:
            priority = "Low"
        elif starter_pct >= 45:
            priority = "Medium"
        else:
            priority = "High"

        summary_rows.append({
            "PositionGroup": pos,
            "CurrentStarter": starter_name,
            "StarterAge": starter_age,
            "StarterScore": round(starter_score, 1),
            "StarterPercentile": round(starter_pct, 1),
            "SquadDepth": int(len(pos_squad)),
            "Priority": priority,
        })

        targets = pos_pool[
            (pos_pool["Club"] != club)
            & (pos_pool["AdjustedCompositeScore"] > starter_score)
            & (pos_pool["TrajectoryTag"] != "Past Peak")
        ].sort_values("ValueRatio", ascending=False, na_position="last").head(top_n_targets)

        for rank, (_, r) in enumerate(targets.iterrows(), 1):
            target_rows.append({
                "PositionGroup": pos,
                "Priority": priority,
                "Rank": rank,
                "Player": r.get("Player"),
                "Club": r.get("Club"),
                "League": r.get("League"),
                "AgeYears": r.get("AgeYears"),
                "TrajectoryTag": r.get("TrajectoryTag"),
                "_mkt_val": r.get("_mkt_val"),
                "ModelValueEUR": r.get("ModelValueEUR"),
                "ValueRatio": r.get("ValueRatio"),
                "AdjustedCompositeScore": r.get("AdjustedCompositeScore"),
            })

    summary = pd.DataFrame(summary_rows)
    prio_order = {"High": 0, "Medium": 1, "Low": 2}
    summary["_o"] = summary["Priority"].map(prio_order)
    summary = summary.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    targets_df = pd.DataFrame(target_rows)
    if not targets_df.empty:
        targets_df["_o"] = targets_df["Priority"].map(prio_order)
        targets_df = targets_df.sort_values(["_o", "PositionGroup", "Rank"]).drop(columns="_o").reset_index(drop=True)

    return summary, targets_df


# ── 8. Similarity search — replacements for the current squad ──────────────

SIMILARITY_FEATURES: dict[str, list[str]] = {
    "GK": ["Save rate, %", "Prevented goals per 90", "Exits per 90", "Accurate passes, %", "Accurate long passes, %", "Aerial duels per 90"],
    "CB": ["Successful defensive actions per 90", "Defensive duels won, %", "Aerial duels won, %", "Interceptions per 90", "Accurate passes, %", "Progressive passes per 90"],
    "FB": ["Crosses per 90", "Accurate crosses, %", "xA per 90", "Progressive runs per 90", "Defensive duels won, %", "Aerial duels won, %"],
    "DM": ["Successful defensive actions per 90", "Defensive duels won, %", "Interceptions per 90", "Passes per 90", "Accurate passes, %", "Progressive passes per 90"],
    "CM": ["Passes per 90", "Progressive passes per 90", "Key passes per 90", "xA per 90", "Progressive runs per 90", "Successful defensive actions per 90", "Goals per 90"],
    "AM": ["Key passes per 90", "xA per 90", "Goals per 90", "xG per 90", "Dribbles per 90", "Touches in box per 90"],
    "W": ["Dribbles per 90", "Successful dribbles, %", "xA per 90", "Key passes per 90", "Progressive runs per 90", "Goals per 90"],
    "ST": ["Goals per 90", "xG per 90", "Touches in box per 90", "Aerial duels won, %", "xA per 90", "Dribbles per 90"],
}


def compute_squad_similar_players(
    df: pd.DataFrame,
    club: str = HRADEC_CLUB,
    league: str = HRADEC_LEAGUE,
    n: int = 8,
) -> pd.DataFrame:
    """For every current senior squad player, find the most statistically similar players in the full universe — potential replacements, backups, or like-for-like upgrades."""
    squad = df[(df["Club"] == club) & (df["League"] == league) & (df["TeamType"] == "Senior")]
    if squad.empty:
        return pd.DataFrame()

    rows = []
    for pos, feats in SIMILARITY_FEATURES.items():
        pos_squad = squad[squad["PositionGroup"] == pos]
        if pos_squad.empty:
            continue
        pool = df[(df["PositionGroup"] == pos) & (df["TeamType"] == "Senior")]
        engine = SimilarityEngine([f for f in feats if f in df.columns])
        for _, target_row in pos_squad.iterrows():
            sims = engine.find_similar(pool, target_row, method="cosine", n=n + 1, same_position=False)
            if sims.empty:
                continue
            sims = sims[sims["Club"] != club].head(n)
            for rank, (_, r) in enumerate(sims.iterrows(), 1):
                rows.append({
                    "OurPlayer": target_row["Player"],
                    "PositionGroup": pos,
                    "Rank": rank,
                    "Player": r.get("Player"),
                    "Club": r.get("Club"),
                    "League": r.get("League"),
                    "AgeYears": r.get("AgeYears"),
                    "Similarity": round(float(r["_similarity"]), 3),
                    "TrajectoryTag": r.get("TrajectoryTag"),
                    "_mkt_val": r.get("_mkt_val"),
                    "ModelValueEUR": r.get("ModelValueEUR"),
                    "ValueRatio": r.get("ValueRatio"),
                })
    return pd.DataFrame(rows)


# ── 9. Club style profiles ──────────────────────────────────────────────────

STYLE_SUBSCORES = ["ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore", "PressingScore", "AerialScore"]
STYLE_LABELS = {
    "ScoringThreatScore": "Attacking", "CreativeProgressionScore": "Creation",
    "DefensiveDisruptionScore": "Defending", "PressingScore": "Pressing", "AerialScore": "Aerial",
}


def compute_club_style_profiles(club_rankings: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank each club's subscores across the whole rated pool — puts every club's tactical identity on the same 0-100 scale for radar comparison."""
    df = club_rankings.copy()
    if df.empty:
        return df
    for col in STYLE_SUBSCORES:
        if col in df.columns:
            df[f"{col}Pctl"] = (pd.to_numeric(df[col], errors="coerce").rank(pct=True) * 100).round(1)
    return df


def style_similar_clubs(
    style_df: pd.DataFrame, reference_club: str = HRADEC_CLUB, reference_league: str = HRADEC_LEAGUE, n: int = 10,
) -> pd.DataFrame:
    """Which clubs play most similarly to the reference club, by style-profile cosine similarity."""
    cols = [f"{c}Pctl" for c in STYLE_SUBSCORES if f"{c}Pctl" in style_df.columns]
    ref = style_df[(style_df["Team"] == reference_club) & (style_df["League"] == reference_league)]
    if ref.empty or not cols:
        return pd.DataFrame()
    ref_vec = ref.iloc[0][cols].values.astype(float)
    mat = style_df[cols].values.astype(float)
    ref_norm = np.linalg.norm(ref_vec) or 1e-9
    mat_norm = np.linalg.norm(mat, axis=1)
    mat_norm = np.where(mat_norm == 0, 1e-9, mat_norm)
    sims = (mat @ ref_vec) / (mat_norm * ref_norm)
    out = style_df.copy()
    out["StyleSimilarity"] = sims.round(3)
    out = out[(out["Team"] != reference_club) | (out["League"] != reference_league)]
    return out.sort_values("StyleSimilarity", ascending=False).head(n)


# ── 10. Set-piece specialists ────────────────────────────────────────────────

def compute_set_piece_specialists(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Reuse the set-piece engine to surface corner takers, dead-ball specialists, aerial threats, etc. across the whole universe."""
    analyzer = SetPieceAnalyzer(threshold=1.5)
    enriched = analyzer.fit_transform(df)
    boards = analyzer.top_players_by_role(enriched, top_n=top_n)
    rows = []
    for role, board in boards.items():
        for _, r in board.iterrows():
            rows.append({
                "Role": role,
                "Player": r.get("Player"),
                "Team": r.get("Team"),
                "Position": r.get("Position"),
                "Age": r.get("Age"),
                "RoleScore": round(float(r.get(f"_sp_role_{role}", 0) or 0), 2),
                "Composite": round(float(r.get("_sp_composite", 0) or 0), 2),
            })
    return pd.DataFrame(rows)


# ── 11. Hidden gems — pure statistical anomalies ────────────────────────────

ANOMALY_METRICS = [
    "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
    "Progressive passes per 90", "Progressive runs per 90", "Dribbles per 90",
    "Key passes per 90", "Successful defensive actions per 90",
    "Interceptions per 90", "Aerial duels won, %", "Duels won, %",
]


def compute_hidden_gems(df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    """
    Pure statistical outlier detection, independent of market value — a
    different lens to the Undervalued board: players whose underlying
    output is exceptional relative to position peers regardless of what
    anyone currently pays for them.
    """
    pool = df[df["PositionGroup"] != "GK"].copy()
    engine = AnomalyEngine(threshold=1.8, method="z-score", groupby="PositionGroup")
    scored = engine.fit_transform(pool, ANOMALY_METRICS)
    gems = engine.filter_anomalies(scored, top_n=top_n)
    keep = [
        "Player", "Club", "League", "PositionGroup", "AgeYears", "TrajectoryTag",
        "_mkt_val", "ModelValueEUR", "ValueTier", "_anomaly_type", "_anomaly_score", "_peak_z", "_anomaly_breadth",
    ]
    keep = [c for c in keep if c in gems.columns]
    return gems[keep].rename(columns={
        "_anomaly_type": "AnomalyType", "_anomaly_score": "AnomalyScore",
        "_peak_z": "PeakZ", "_anomaly_breadth": "Breadth",
    }).reset_index(drop=True)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def build_recruitment_universe(
    min_minutes_senior: int = DEFAULT_MIN_MINUTES_SENIOR,
    min_minutes_youth: int = DEFAULT_MIN_MINUTES_YOUTH,
    leagues: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    if verbose:
        print("Building league index…")
    league_index = build_league_index()

    if verbose:
        print(f"Loading players from {len(league_index) if leagues is None else len(leagues)} league files…")
    raw = load_raw_players(league_index, min_minutes_senior, min_minutes_youth, leagues)
    if raw.empty:
        raise RuntimeError("No player data loaded — check Wyscout Files directory.")
    if verbose:
        print(f"  → {len(raw)} player-rows loaded")

    if verbose:
        print("Computing composite performance scores (within own league, to avoid weak-league stat inflation)…")
    scored = pd.concat(
        [compute_wyscout_scores(g) for _, g in raw.groupby("League")],
        ignore_index=True,
    )
    # CompositeRecruitmentScore above is a percentile *within the player's own
    # league* — fair (compares him to his actual peers) but not cross-league
    # comparable: a striker running riot in a weak league would otherwise look
    # identical to one doing it in a strong one. AdjustedCompositeScore scales
    # that league-relative percentile by how strong the league itself is, the
    # same way club power scores are tier-adjusted, so it's safe to compare
    # and sort across the whole universe (used for the valuation model and
    # for cross-league boards). Club/league rankings keep using the
    # unadjusted, within-league CompositeRecruitmentScore and apply their own
    # tier adjustment at the club level, so the two adjustments don't stack.
    scored["AdjustedCompositeScore"] = (
        scored["CompositeRecruitmentScore"] * (0.5 + 0.5 * scored["LeagueStrength"] / 100.0)
    ).round(1)

    if verbose:
        print("Computing peak-age trajectory…")
    scored = compute_peak_trajectory(scored)

    if verbose:
        print("Fitting metrics-derived EUR valuation model…")
    scored = compute_value_model(scored)
    if verbose and "_value_r2" in scored.columns:
        r2 = scored["_value_r2"].dropna()
        if not r2.empty:
            print(f"  → valuation model R² (log market value, train fold): {r2.iloc[0]:.3f}")

    if verbose:
        print(f"Scoring physical/quick-league fit vs {BENCHMARK_LEAGUE_FILE} First League…")
    scored = compute_physical_fit(scored)

    if verbose:
        print("Scoring Brighton mechanics (buy low, develop, resell)…")
    scored = compute_brighton_mechanics(scored)

    if verbose:
        print("Classifying role archetypes…")
    scored = compute_role_archetypes(scored)

    if verbose:
        print("Building club power rankings…")
    club_senior = build_club_rankings(scored, "Senior")
    club_youth = build_club_rankings(scored, "Youth")

    if verbose:
        print("Building league power rankings…")
    league_senior = build_league_rankings(club_senior, league_index, "Senior")
    league_youth = build_league_rankings(club_youth, league_index, "Youth")

    if verbose:
        print(f"Analysing {HRADEC_CLUB} squad needs…")
    squad_needs, squad_targets = compute_squad_needs(scored)

    if verbose:
        print("Finding statistical comparables for the current squad…")
    squad_similar = compute_squad_similar_players(scored)

    if verbose:
        print("Building club style profiles…")
    club_style_senior = compute_club_style_profiles(club_senior)
    style_similar = style_similar_clubs(club_style_senior)

    if verbose:
        print("Scoring set-piece specialists…")
    set_piece = compute_set_piece_specialists(scored)

    if verbose:
        print("Detecting hidden gems (statistical anomalies)…")
    hidden_gems = compute_hidden_gems(scored)

    return {
        "players": scored,
        "league_index": league_index,
        "club_rankings_senior": club_senior,
        "club_rankings_youth": club_youth,
        "league_rankings_senior": league_senior,
        "league_rankings_youth": league_youth,
        "squad_needs_summary": squad_needs,
        "squad_priority_targets": squad_targets,
        "squad_similar_players": squad_similar,
        "club_style_senior": club_style_senior,
        "style_similar_to_hradec": style_similar,
        "set_piece_specialists": set_piece,
        "hidden_gems": hidden_gems,
    }
