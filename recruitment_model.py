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

    ln(Market Value + 1)  ~  Composite performance score (percentile)
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

    comp = pd.to_numeric(df.get("CompositeRecruitmentScore"), errors="coerce").fillna(50.0)
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
        print("Computing composite performance scores…")
    scored = compute_wyscout_scores(raw)

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
        print("Building club power rankings…")
    club_senior = build_club_rankings(scored, "Senior")
    club_youth = build_club_rankings(scored, "Youth")

    if verbose:
        print("Building league power rankings…")
    league_senior = build_league_rankings(club_senior, league_index, "Senior")
    league_youth = build_league_rankings(club_youth, league_index, "Youth")

    return {
        "players": scored,
        "league_index": league_index,
        "club_rankings_senior": club_senior,
        "club_rankings_youth": club_youth,
        "league_rankings_senior": league_senior,
        "league_rankings_youth": league_youth,
    }
