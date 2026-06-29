"""
export_multi_sport.py
─────────────────────
Multi-sport scouting workbook using REAL football players from the Wyscout DB.

Each player's football profile (their composite sub-scores) is mapped to
equivalent performance levels in five other sports. A striker who scores at
the 90th percentile in football generates stats matching a 90th-percentile
MLB slugger; a deep-lying defensive midfielder becomes a high-ceiling pitcher.

Sheets
──────
  MLB     — strikers / wingers / AM → batters; DM / CB / FB → pitchers; GK → closers
  NBA     — playmakers → guards; athletic forwards → wings; CBs/DMs → bigs
  NHL     — attackers → forwards; CBs/DMs → defensemen; GKs → goalies
  NFL     — creative mids → QBs; strikers/wingers → WRs; pressing players → RBs/Defence
  Cricket — attacking players → batters; defensive/pressing → bowlers; balanced → all-rounders

Each sheet: Player · Team · Position · Age · Football Position · League +
            sport-specific stats + Rating (0–100) + Scouting Uncertainty (0–100) +
            Confidence Label

Usage
─────
  python export_multi_sport.py [--min-minutes 400] [--output "data/...xlsx"]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from wyscout_model import load_wyscout_raw, compute_wyscout_scores

DATA_DIR = ROOT / "data"
RNG = np.random.default_rng(2024)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _pct(series: pd.Series) -> pd.Series:
    """0–100 percentile rank (higher = better)."""
    return series.fillna(series.median()).rank(pct=True) * 100

def _pct_inv(series: pd.Series) -> pd.Series:
    return _pct(-series)

def _composite(df: pd.DataFrame, spec: list[tuple[str, float, bool]]) -> pd.Series:
    total_w = sum(w for _, w, _ in spec)
    score   = pd.Series(0.0, index=df.index)
    for col, w, higher in spec:
        if col not in df.columns:
            continue
        score += (w / total_w) * (_pct(df[col]) if higher else _pct_inv(df[col]))
    return score.clip(0, 100).round(1)

def _s2v(score_0_100: pd.Series, mean: float, std: float,
         lo: float, hi: float) -> pd.Series:
    """
    Convert a 0–100 football sub-score to a sport stat value.
    Uses the normal PPF so a score of 90 → ~90th-percentile stat value.
    """
    pct = score_0_100.clip(1, 99) / 100
    z   = pct.apply(lambda p: norm.ppf(p))
    return (mean + z * std).clip(lo, hi)

def _uncertainty(
    df: pd.DataFrame,
    age_col: str,
    games_col: str,
    max_games: float,
    peak_age: tuple[float, float],
    football_minutes: pd.Series,
) -> pd.Series:
    """
    Five-factor ScoutingUncertainty (0–100) using original football minutes
    as the primary sample-size signal.
    """
    _MIN = 400.0
    mins  = football_minutes.fillna(_MIN).clip(lower=_MIN)
    f_sample = np.sqrt(_MIN / mins).clip(0, 1)

    f_league = pd.Series(0.05, index=df.index)   # all top-level football

    age = pd.to_numeric(df[age_col], errors="coerce").fillna(27)
    f_age = pd.Series(0.0, index=df.index, dtype=float)
    f_age[age < 21]                              = 0.35
    f_age[age.between(21, 23.99)]                = 0.20
    f_age[age.between(peak_age[0], peak_age[1])] = 0.00
    f_age[age.between(peak_age[1], 31.99)]       = 0.12
    f_age[age >= 32]                             = 0.25

    games  = pd.to_numeric(df[games_col], errors="coerce").fillna(max_games * 0.5).clip(lower=1)
    f_avail = ((max_games - games) / max_games).clip(0, 1) * 0.40

    combined = (
        0.35 * f_sample + 0.25 * f_league + 0.20 * f_age + 0.10 * f_avail
    ).clip(0, 1)
    return (combined * 100).round(1)

def _confidence_label(u: pd.Series) -> pd.Series:
    def _lbl(v: float) -> str:
        if v <= 20: return "High Confidence"
        if v <= 35: return "Good Confidence"
        if v <= 50: return "Moderate Confidence"
        if v <= 65: return "Low Confidence"
        return "Very Low Confidence"
    return u.apply(_lbl)

def _base_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the bio columns every sheet starts with."""
    keep = ["Player", "Team", "Age", "Football Position", "Football League",
            "Football Minutes", "Matches played"]
    return df[[c for c in keep if c in df.columns]].copy()

def _write(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame) -> None:
    df.to_excel(writer, sheet_name=sheet[:31], index=False)
    ws = writer.sheets[sheet[:31]]
    for col_cells in ws.columns:
        try:
            w = max(len(str(col_cells[0].value or "")),
                    *(len(str(c.value or "")) for c in col_cells[1:8]))
            ws.column_dimensions[col_cells[0].column_letter].width = min(w + 2, 38)
        except Exception:
            pass
    print(f"  ✓ {sheet!r} — {len(df):,} rows")


# ── Load football players ──────────────────────────────────────────────────────

def load_players(min_minutes: int) -> pd.DataFrame:
    print(f"  Loading Wyscout players (min {min_minutes} min)…")
    raw = load_wyscout_raw(min_minutes=min_minutes)
    df  = compute_wyscout_scores(raw)
    df  = df.loc[:, ~df.columns.duplicated()]

    # Rename for clarity
    df = df.rename(columns={
        "Player":        "Player",
        "Team":          "Team",
        "Age":           "Age",
        "Position":      "Football Position",
        "_League":       "Football League",
        "MinutesPlayed": "Football Minutes",
    })
    if "Football Minutes" not in df.columns:
        mins_col = next((c for c in ["Minutes played"] if c in df.columns), None)
        if mins_col:
            df["Football Minutes"] = pd.to_numeric(df[mins_col], errors="coerce")

    # Keep only what we need
    score_cols = [
        "ScoringThreatScore", "CreativeProgressionScore", "DefensiveDisruptionScore",
        "PressingScore", "BallSecurityScore", "ExpectedThreatScore",
        "ASA_GoalsAddedScore", "AerialScore", "SetPieceScore",
        "CompositeRecruitmentScore", "PositionGroup",
    ]
    keep = ["Player", "Team", "Age", "Football Position", "Football League",
            "Football Minutes", "Matches played"] + score_cols
    df = df[[c for c in keep if c in df.columns]].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(25)
    print(f"  → {len(df):,} players loaded")
    return df.reset_index(drop=True)


def _sample(df: pd.DataFrame, pos_groups: list[str], n: int,
            sort_col: str = "CompositeRecruitmentScore") -> pd.DataFrame:
    """Sample n players from given position groups, sorted by sort_col descending."""
    sub = df[df["PositionGroup"].isin(pos_groups)].copy()
    sub = sub.sort_values(sort_col, ascending=False)
    return sub.head(n).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MLB
# ══════════════════════════════════════════════════════════════════════════════

MLB_POSITION_MAP = {
    "ST": "DH/OF", "W": "OF", "AM": "2B/SS", "CM": "2B",
    "DM": "SP", "FB": "SP", "CB": "SP", "GK": "CL/RP",
}

def build_mlb(df: pd.DataFrame) -> pd.DataFrame:
    # Batters: attacking players
    bat_df  = _sample(df, ["ST", "W", "AM"], 100, "ScoringThreatScore")
    # Pitchers: defensive players + GK
    pit_df  = _sample(df, ["DM", "CB", "FB", "GK"], 80, "DefensiveDisruptionScore")

    frames = []

    # ── Batters ───────────────────────────────────────────────────────────────
    b = _base_cols(bat_df)
    b["MLB Position"] = bat_df["PositionGroup"].map(MLB_POSITION_MAP).fillna("OF")
    sc = bat_df["ScoringThreatScore"]
    cr = bat_df["CreativeProgressionScore"]
    bs = bat_df["BallSecurityScore"]
    ae = bat_df["AerialScore"]

    b["BA"]             = _s2v(sc * 0.5 + bs * 0.5, 0.265, 0.030, 0.180, 0.370).round(3)
    b["OBP"]            = _s2v(bs * 0.6 + cr * 0.4, 0.335, 0.038, 0.250, 0.470).round(3)
    b["SLG"]            = _s2v(sc * 0.7 + ae * 0.3, 0.440, 0.065, 0.300, 0.700).round(3)
    b["OPS"]            = (b["OBP"] + b["SLG"]).round(3)
    b["HR"]             = _s2v(sc * 0.8 + ae * 0.2, 18, 12, 0, 62).round(0).astype(int)
    b["RBI"]            = _s2v(sc * 0.6 + ae * 0.4, 72, 28, 10, 150).round(0).astype(int)
    b["SB"]             = _s2v(bat_df["PressingScore"], 10, 11, 0, 65).round(0).astype(int)
    b["wRC+"]           = _s2v(sc * 0.5 + bs * 0.3 + cr * 0.2, 108, 22, 50, 210).round(0).astype(int)
    b["OPS+"]           = _s2v(sc * 0.5 + bs * 0.3 + cr * 0.2, 108, 24, 48, 210).round(0).astype(int)
    b["BB%"]            = _s2v(bs * 0.7 + cr * 0.3, 9.0, 2.8, 3.0, 19.0).round(1)
    b["K%"]             = _s2v(100 - bs, 22, 5.5, 8, 42).round(1)
    b["WAR"]            = _s2v(bat_df["CompositeRecruitmentScore"], 2.5, 2.2, -1.5, 9.5).round(1)
    b["Exit Velocity"]  = _s2v(sc * 0.6 + ae * 0.4, 89, 4, 78, 98).round(1)
    b["Hard Hit%"]      = _s2v(sc * 0.7 + ae * 0.3, 38, 8, 18, 60).round(1)
    b["Type"]           = "Batter"
    b["Football Minutes_raw"] = bat_df["Football Minutes"].values

    b["Rating"] = _composite(b, [
        ("wRC+", 0.35, True), ("WAR", 0.30, True),
        ("OPS+", 0.20, True), ("BB%", 0.08, True), ("K%", 0.07, False),
    ])
    b["Scouting Uncertainty"] = _uncertainty(
        b, "Age", "Matches played", 162, (26, 31),
        bat_df["Football Minutes"],
    )
    frames.append(b)

    # ── Pitchers ──────────────────────────────────────────────────────────────
    p = _base_cols(pit_df)
    p["MLB Position"] = pit_df["PositionGroup"].map(MLB_POSITION_MAP).fillna("SP")
    dd = pit_df["DefensiveDisruptionScore"]
    pr = pit_df["PressingScore"]
    bs = pit_df["BallSecurityScore"]
    comp = pit_df["CompositeRecruitmentScore"]

    p["ERA"]     = _s2v(100 - (dd * 0.5 + pr * 0.5), 3.85, 0.90, 1.50, 7.00).round(2)
    p["FIP"]     = _s2v(100 - (dd * 0.5 + bs * 0.5), 3.90, 0.80, 1.80, 6.50).round(2)
    p["WHIP"]    = _s2v(100 - (dd * 0.6 + bs * 0.4), 1.22, 0.22, 0.70, 2.00).round(2)
    p["K/9"]     = _s2v(pr * 0.6 + dd * 0.4, 9.5, 2.2, 4.0, 16.0).round(1)
    p["BB/9"]    = _s2v(100 - bs, 3.0, 0.85, 0.8, 6.0).round(1)
    p["K-BB%"]   = _s2v(dd * 0.5 + pr * 0.3 + bs * 0.2, 17, 7, 2, 38).round(1)
    p["ERA+"]    = _s2v(dd * 0.5 + bs * 0.3 + pr * 0.2, 110, 28, 40, 220).round(0).astype(int)
    p["GB%"]     = _s2v(dd * 0.6 + pr * 0.4, 44, 8, 25, 65).round(1)
    p["WAR"]     = _s2v(comp, 2.2, 2.0, -1.0, 8.5).round(1)
    p["IP"]      = _s2v(pit_df["PressingScore"], 140, 50, 20, 220).round(1)
    p["Type"]    = "Pitcher"
    p["Football Minutes_raw"] = pit_df["Football Minutes"].values

    p["Rating"] = _composite(p, [
        ("ERA+", 0.35, True), ("K-BB%", 0.30, True),
        ("WAR",  0.25, True), ("GB%",   0.10, True),
    ])
    p["Scouting Uncertainty"] = _uncertainty(
        p, "Age", "Matches played", 33, (26, 32),
        pit_df["Football Minutes"],
    )
    frames.append(p)

    out = pd.concat(frames, ignore_index=True)
    out["Confidence Label"] = _confidence_label(out["Scouting Uncertainty"])
    out = out.drop(columns=["Football Minutes_raw"], errors="ignore")
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# NBA
# ══════════════════════════════════════════════════════════════════════════════

NBA_POSITION_MAP = {
    "AM": "PG", "CM": "PG/SG", "W": "SG/SF",
    "ST": "SF/PF", "FB": "SF", "DM": "PF", "CB": "C", "GK": "C",
}

def build_nba(df: pd.DataFrame) -> pd.DataFrame:
    players = _sample(df, ["AM", "CM", "W", "ST", "DM", "CB", "FB"], 150,
                      "CompositeRecruitmentScore")
    out = _base_cols(players)
    out["NBA Position"] = players["PositionGroup"].map(NBA_POSITION_MAP).fillna("SF")

    cr = players["CreativeProgressionScore"]
    sc = players["ScoringThreatScore"]
    bs = players["BallSecurityScore"]
    dd = players["DefensiveDisruptionScore"]
    pr = players["PressingScore"]
    et = players["ExpectedThreatScore"]
    comp = players["CompositeRecruitmentScore"]

    out["PTS/G"]    = _s2v(sc * 0.6 + et * 0.4, 17, 7, 4, 38).round(1)
    out["REB/G"]    = _s2v(dd * 0.5 + players["AerialScore"] * 0.5, 6, 3, 1, 16).round(1)
    out["AST/G"]    = _s2v(cr * 0.7 + bs * 0.3, 4, 3, 0.5, 13).round(1)
    out["STL/G"]    = _s2v(pr * 0.6 + dd * 0.4, 1.1, 0.4, 0.2, 2.6).round(1)
    out["BLK/G"]    = _s2v(dd * 0.5 + players["AerialScore"] * 0.5, 0.7, 0.6, 0, 3.5).round(1)
    out["TOV/G"]    = _s2v(100 - bs, 2.0, 0.8, 0.5, 5.0).round(1)
    out["FG%"]      = _s2v(sc * 0.5 + bs * 0.5, 0.475, 0.052, 0.330, 0.650).round(3)
    out["3P%"]      = _s2v(cr * 0.6 + bs * 0.4, 0.355, 0.050, 0.200, 0.480).round(3)
    out["TS%"]      = _s2v(sc * 0.4 + bs * 0.4 + cr * 0.2, 0.575, 0.052, 0.430, 0.720).round(3)
    out["PER"]      = _s2v(comp, 16, 5, 5, 32).round(1)
    out["BPM"]      = _s2v(comp * 0.6 + dd * 0.4, 1.5, 4, -6, 12).round(1)
    out["WS/48"]    = _s2v(comp, 0.110, 0.062, -0.03, 0.290).round(3)
    out["Net Rtg"]  = _s2v(comp * 0.5 + dd * 0.5, 1.5, 6, -12, 16).round(1)
    out["USG%"]     = _s2v(sc * 0.5 + cr * 0.5, 22, 5, 10, 36).round(1)
    out["Games"]    = _s2v(pr * 0.5 + bs * 0.5, 62, 16, 10, 82).round(0).astype(int)

    out["Rating"] = _composite(out, [
        ("BPM",     0.30, True), ("WS/48",   0.25, True),
        ("PER",     0.20, True), ("TS%",     0.15, True),
        ("Net Rtg", 0.10, True),
    ])
    out["Scouting Uncertainty"] = _uncertainty(
        out, "Age", "Games", 82, (24, 29), players["Football Minutes"]
    )
    out["Confidence Label"] = _confidence_label(out["Scouting Uncertainty"])
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# NHL
# ══════════════════════════════════════════════════════════════════════════════

NHL_POSITION_MAP = {
    "ST": "LW/RW", "W": "LW/RW", "AM": "C", "CM": "C",
    "DM": "LD/RD", "FB": "LD/RD", "CB": "LD/RD", "GK": "G",
}

def build_nhl(df: pd.DataFrame) -> pd.DataFrame:
    skater_df = _sample(df, ["ST", "W", "AM", "CM", "DM", "CB", "FB"], 120,
                        "CompositeRecruitmentScore")
    goalie_df = _sample(df, ["GK"], 40, "DefensiveDisruptionScore")
    frames = []

    # Skaters
    sk = _base_cols(skater_df)
    sk["NHL Position"] = skater_df["PositionGroup"].map(NHL_POSITION_MAP).fillna("C")
    sc  = skater_df["ScoringThreatScore"]
    cr  = skater_df["CreativeProgressionScore"]
    dd  = skater_df["DefensiveDisruptionScore"]
    pr  = skater_df["PressingScore"]
    bs  = skater_df["BallSecurityScore"]
    comp= skater_df["CompositeRecruitmentScore"]

    sk["Games"]      = _s2v(pr * 0.5 + bs * 0.5, 65, 14, 10, 82).round(0).astype(int)
    sk["Goals"]      = _s2v(sc * 0.7 + skater_df["ASA_GoalsAddedScore"] * 0.3, 18, 12, 0, 60).round(0).astype(int)
    sk["Assists"]    = _s2v(cr * 0.7 + bs * 0.3, 25, 15, 0, 75).round(0).astype(int)
    sk["Points"]     = (sk["Goals"] + sk["Assists"])
    sk["Pts/GP"]     = (sk["Points"] / sk["Games"].clip(1)).round(2)
    sk["+/-"]        = _s2v(dd * 0.5 + pr * 0.5, 2, 12, -28, 32).round(0).astype(int)
    sk["TOI/GP"]     = _s2v(comp, 17, 4, 8, 26).round(1)
    sk["Corsi%"]     = _s2v(pr * 0.5 + dd * 0.5, 50, 4, 38, 62).round(1)
    sk["Fenwick%"]   = _s2v(pr * 0.5 + dd * 0.5, 50, 4, 38, 62).round(1)
    sk["xGF%"]       = _s2v(sc * 0.3 + cr * 0.3 + dd * 0.4, 50, 5, 35, 65).round(1)
    sk["Sh%"]        = _s2v(sc, 10.5, 3.5, 2, 22).round(1)
    sk["GAR"]        = _s2v(comp, 4, 6, -8, 22).round(1)
    sk["Type"]       = "Skater"

    sk["Rating"] = _composite(sk, [
        ("GAR",    0.35, True), ("Pts/GP", 0.30, True),
        ("Corsi%", 0.20, True), ("xGF%",  0.15, True),
    ])
    sk["Scouting Uncertainty"] = _uncertainty(
        sk, "Age", "Games", 82, (25, 30), skater_df["Football Minutes"]
    )
    frames.append(sk)

    # Goalies
    gk = _base_cols(goalie_df)
    gk["NHL Position"] = "G"
    dd_g = goalie_df["DefensiveDisruptionScore"]
    bs_g = goalie_df["BallSecurityScore"]
    comp_g = goalie_df["CompositeRecruitmentScore"]

    gk["Games"]  = _s2v(comp_g, 45, 14, 10, 70).round(0).astype(int)
    gk["SV%"]    = _s2v(dd_g * 0.6 + bs_g * 0.4, 0.912, 0.012, 0.876, 0.940).round(3)
    gk["GAA"]    = _s2v(100 - (dd_g * 0.7 + bs_g * 0.3), 2.65, 0.38, 1.60, 4.20).round(2)
    gk["GSAA"]   = _s2v(dd_g * 0.5 + comp_g * 0.5, 5, 10, -18, 30).round(1)
    gk["GAR"]    = _s2v(comp_g, 4, 8, -12, 25).round(1)
    gk["Type"]   = "Goalie"

    gk["Rating"] = _composite(gk, [
        ("GSAA", 0.40, True), ("SV%", 0.35, True),
        ("GAA",  0.15, False), ("GAR", 0.10, True),
    ])
    gk["Scouting Uncertainty"] = _uncertainty(
        gk, "Age", "Games", 70, (26, 33), goalie_df["Football Minutes"]
    )
    frames.append(gk)

    out = pd.concat(frames, ignore_index=True)
    out["Confidence Label"] = _confidence_label(out["Scouting Uncertainty"])
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# NFL
# ══════════════════════════════════════════════════════════════════════════════

NFL_POSITION_MAP = {
    "AM": "QB", "CM": "QB", "ST": "RB", "W": "WR",
    "FB": "WR/TE", "DM": "LB", "CB": "DL", "GK": "DL",
}

def build_nfl(df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    # QBs — playmakers
    qb_df = _sample(df, ["AM", "CM"], 32, "CreativeProgressionScore")
    qb = _base_cols(qb_df)
    qb["NFL Position"] = "QB"
    cr = qb_df["CreativeProgressionScore"]
    bs = qb_df["BallSecurityScore"]
    et = qb_df["ExpectedThreatScore"]

    qb["Games"]         = _s2v(bs, 14, 3, 3, 17).round(0).astype(int)
    qb["Pass Yards"]    = _s2v(cr * 0.6 + et * 0.4, 3800, 900, 500, 5500).round(0).astype(int)
    qb["Pass TD"]       = _s2v(et * 0.6 + cr * 0.4, 27, 9, 3, 55).round(0).astype(int)
    qb["INT"]           = _s2v(100 - bs, 10, 4, 1, 22).round(0).astype(int)
    qb["Comp%"]         = _s2v(bs * 0.7 + cr * 0.3, 65, 5, 52, 78).round(1)
    qb["YPA"]           = _s2v(et * 0.6 + cr * 0.4, 7.5, 1.2, 5.0, 11.0).round(1)
    qb["QBR"]           = _s2v(qb_df["CompositeRecruitmentScore"], 55, 18, 15, 90).round(1)
    qb["DYAR"]          = _s2v(qb_df["CompositeRecruitmentScore"], 400, 380, -500, 1500).round(0).astype(int)
    qb["TD/INT"]        = (qb["Pass TD"] / qb["INT"].clip(1)).round(2)

    qb["Rating"] = _composite(qb, [
        ("QBR",   0.35, True), ("DYAR",  0.25, True),
        ("YPA",   0.20, True), ("TD/INT",0.20, True),
    ])
    qb["Scouting Uncertainty"] = _uncertainty(
        qb, "Age", "Games", 17, (26, 32), qb_df["Football Minutes"]
    )
    frames.append(qb)

    # RBs — strikers
    rb_df = _sample(df, ["ST"], 40, "ScoringThreatScore")
    rb = _base_cols(rb_df)
    rb["NFL Position"] = "RB"
    sc = rb_df["ScoringThreatScore"]
    pr = rb_df["PressingScore"]

    rb["Games"]       = _s2v(pr, 13, 3, 3, 17).round(0).astype(int)
    rb["Carries"]     = _s2v(sc * 0.6 + pr * 0.4, 180, 65, 20, 350).round(0).astype(int)
    rb["Rush Yards"]  = _s2v(sc * 0.7 + pr * 0.3, 850, 380, 50, 2100).round(0).astype(int)
    rb["YPC"]         = _s2v(sc, 4.3, 0.65, 2.5, 6.5).round(1)
    rb["Rush TD"]     = _s2v(sc * 0.7 + rb_df["ASA_GoalsAddedScore"] * 0.3, 7, 4, 0, 21).round(0).astype(int)
    rb["Receptions"]  = _s2v(rb_df["CreativeProgressionScore"], 38, 18, 5, 85).round(0).astype(int)
    rb["Rec Yards"]   = _s2v(rb_df["CreativeProgressionScore"], 300, 170, 20, 700).round(0).astype(int)
    rb["DVOA%"]       = _s2v(rb_df["CompositeRecruitmentScore"], 5, 20, -35, 45).round(1)
    rb["DYAR"]        = _s2v(rb_df["CompositeRecruitmentScore"], 80, 130, -200, 400).round(0).astype(int)

    rb["Rating"] = _composite(rb, [
        ("DVOA%", 0.35, True), ("YPC",  0.30, True),
        ("DYAR",  0.20, True), ("Receptions", 0.15, True),
    ])
    rb["Scouting Uncertainty"] = _uncertainty(
        rb, "Age", "Games", 17, (23, 28), rb_df["Football Minutes"]
    )
    frames.append(rb)

    # WRs — wingers
    wr_df = _sample(df, ["W", "FB"], 60, "CreativeProgressionScore")
    wr = _base_cols(wr_df)
    wr["NFL Position"] = "WR/TE"
    cr = wr_df["CreativeProgressionScore"]
    sc = wr_df["ScoringThreatScore"]
    bs = wr_df["BallSecurityScore"]

    wr["Games"]       = _s2v(bs, 13, 3, 3, 17).round(0).astype(int)
    wr["Targets"]     = _s2v(cr * 0.6 + sc * 0.4, 90, 35, 10, 175).round(0).astype(int)
    wr["Receptions"]  = _s2v(bs * 0.5 + cr * 0.5, 58, 22, 5, 125).round(0).astype(int)
    wr["Rec Yards"]   = _s2v(cr * 0.5 + sc * 0.5, 800, 340, 50, 1850).round(0).astype(int)
    wr["Rec TD"]      = _s2v(sc, 6, 4, 0, 18).round(0).astype(int)
    wr["Catch%"]      = (wr["Receptions"] / wr["Targets"].clip(1) * 100).round(1)
    wr["YPR"]         = (wr["Rec Yards"] / wr["Receptions"].clip(1)).round(1)
    wr["DVOA%"]       = _s2v(wr_df["CompositeRecruitmentScore"], 8, 22, -40, 60).round(1)
    wr["DYAR"]        = _s2v(wr_df["CompositeRecruitmentScore"], 120, 155, -250, 500).round(0).astype(int)

    wr["Rating"] = _composite(wr, [
        ("DVOA%",  0.35, True), ("DYAR",   0.25, True),
        ("YPR",    0.20, True), ("Catch%", 0.20, True),
    ])
    wr["Scouting Uncertainty"] = _uncertainty(
        wr, "Age", "Games", 17, (24, 29), wr_df["Football Minutes"]
    )
    frames.append(wr)

    # Defence — defensive players
    def_df = _sample(df, ["DM", "CB", "GK"], 75, "DefensiveDisruptionScore")
    def_ = _base_cols(def_df)
    def_["NFL Position"] = def_df["PositionGroup"].map(NFL_POSITION_MAP).fillna("LB")
    dd  = def_df["DefensiveDisruptionScore"]
    pr  = def_df["PressingScore"]
    comp= def_df["CompositeRecruitmentScore"]

    def_["Games"]          = _s2v(pr, 15, 2, 5, 17).round(0).astype(int)
    def_["Solo Tackles"]   = _s2v(dd * 0.6 + pr * 0.4, 50, 25, 5, 130).round(0).astype(int)
    def_["Sacks"]          = _s2v(dd * 0.7 + pr * 0.3, 4.5, 3.8, 0, 22.5).round(1)
    def_["TFL"]            = _s2v(dd * 0.6 + pr * 0.4, 6, 4, 0, 22).round(0).astype(int)
    def_["QB Hits"]        = _s2v(pr * 0.5 + dd * 0.5, 12, 8, 0, 38).round(0).astype(int)
    def_["INT"]            = _s2v(def_df["BallSecurityScore"], 1.5, 1.5, 0, 8).round(0).astype(int)
    def_["PBU"]            = _s2v(dd * 0.5 + def_df["BallSecurityScore"] * 0.5, 5, 4, 0, 20).round(0).astype(int)
    def_["DVOA%"]          = _s2v(comp, 5, 20, -35, 55).round(1)
    def_["DYAR"]           = _s2v(comp, 40, 78, -150, 250).round(0).astype(int)

    def_["Rating"] = _composite(def_, [
        ("DVOA%", 0.30, True), ("Sacks", 0.25, True),
        ("DYAR",  0.25, True), ("TFL",   0.20, True),
    ])
    def_["Scouting Uncertainty"] = _uncertainty(
        def_, "Age", "Games", 17, (25, 30), def_df["Football Minutes"]
    )
    frames.append(def_)

    out = pd.concat(frames, ignore_index=True)
    out["Confidence Label"] = _confidence_label(out["Scouting Uncertainty"])
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cricket
# ══════════════════════════════════════════════════════════════════════════════

def build_cricket(df: pd.DataFrame) -> pd.DataFrame:
    players = _sample(df, ["ST", "W", "AM", "CM", "DM", "CB", "FB", "GK"], 150,
                      "CompositeRecruitmentScore")
    out = _base_cols(players)

    sc  = players["ScoringThreatScore"]
    cr  = players["CreativeProgressionScore"]
    dd  = players["DefensiveDisruptionScore"]
    pr  = players["PressingScore"]
    bs  = players["BallSecurityScore"]
    ae  = players["AerialScore"]
    comp= players["CompositeRecruitmentScore"]

    # Role: attacking → batter, defensive → bowler, balanced → all-rounder
    bat_w = ((sc + cr) / 2).clip(0, 100)
    bowl_w = ((dd + pr) / 2).clip(0, 100)
    total  = (bat_w + bowl_w).clip(lower=1)

    role = pd.Series("All-Rounder", index=players.index)
    role[bat_w / total >= 0.60]  = "Batter"
    role[bowl_w / total >= 0.60] = "Bowler"
    out["Role"] = role

    # Format: skewed toward longer formats for more experienced players
    out["Format"] = pd.cut(
        players["Football Minutes"].fillna(900),
        bins=[0, 700, 1400, 99999],
        labels=["T20I", "ODI", "Test"],
    ).astype(str)

    # Batting stats
    out["Bat Average"]  = _s2v(sc * 0.5 + bs * 0.5, 38, 14, 10, 72).round(1)
    out["Strike Rate"]  = _s2v(sc * 0.6 + pr * 0.4, 72, 28, 30, 180).round(1)
    out["Runs"]         = _s2v(comp * 0.6 + sc * 0.4, 1200, 850, 50, 4500).round(0).astype(int)
    out["100s"]         = _s2v(sc * 0.6 + cr * 0.4, 3, 4, 0, 18).round(0).astype(int)
    out["50s"]          = _s2v(sc * 0.5 + bs * 0.5, 7, 5, 0, 30).round(0).astype(int)

    # Bowling stats
    out["Wickets"]      = _s2v(dd * 0.6 + pr * 0.4, 55, 38, 0, 200).round(0).astype(int)
    out["Bowl Average"] = _s2v(100 - (dd * 0.6 + bs * 0.4), 28, 8, 14, 55).round(1)
    out["Economy"]      = _s2v(100 - (pr * 0.5 + dd * 0.5), 5.5, 1.2, 3.0, 10.5).round(1)
    out["Bowl SR"]      = _s2v(100 - (dd * 0.6 + pr * 0.4), 30, 12, 10, 65).round(1)
    out["5-wkt Hauls"]  = _s2v(dd * 0.7 + pr * 0.3, 3, 4, 0, 18).round(0).astype(int)

    # Innings (sample size)
    out["Innings"] = _s2v(bs * 0.5 + comp * 0.5, 38, 20, 5, 100).round(0).astype(int)

    # Rating: blend batting + bowling weighted by role
    bat_score  = _composite(out, [
        ("Bat Average", 0.45, True), ("Strike Rate", 0.30, True),
        ("100s", 0.15, True), ("50s", 0.10, True),
    ])
    bowl_score = _composite(out, [
        ("Bowl Average", 0.40, False), ("Economy", 0.30, False),
        ("Bowl SR", 0.20, False), ("Wickets", 0.10, True),
    ])
    bw = (bat_w / total).values
    out["Rating"] = (bw * bat_score + (1 - bw) * bowl_score).clip(0, 100).round(1)

    out["Scouting Uncertainty"] = _uncertainty(
        out, "Age", "Innings", 80, (26, 32), players["Football Minutes"]
    )
    out["Confidence Label"] = _confidence_label(out["Scouting Uncertainty"])
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(min_minutes: int, output: Path) -> None:
    print(f"\n{'='*60}")
    print("  Multi-Sport Scouting Report — Real Football Players")
    print(f"{'='*60}")

    players = load_players(min_minutes)

    print("Building MLB sheet…")
    mlb = build_mlb(players)
    print("Building NBA sheet…")
    nba = build_nba(players)
    print("Building NHL sheet…")
    nhl = build_nhl(players)
    print("Building NFL sheet…")
    nfl = build_nfl(players)
    print("Building Cricket sheet…")
    cricket = build_cricket(players)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting workbook → {output}")

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _write(writer, "MLB", mlb)
        _write(writer, "NBA", nba)
        _write(writer, "NHL", nhl)
        _write(writer, "NFL", nfl)
        _write(writer, "Cricket", cricket)

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {size_mb:.1f} MB → {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-minutes", type=int, default=400)
    parser.add_argument("--output", type=Path,
                        default=DATA_DIR / "Multi-Sport Scouting Report.xlsx")
    args = parser.parse_args()
    run(args.min_minutes, args.output)
