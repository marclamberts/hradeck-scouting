"""
export_multi_sport.py
─────────────────────
Multi-sport scouting workbook.

Applies the same Rating + ScoutingUncertainty + ConfidenceLabel framework
used in the Wyscout football model to five other major sports:

  Sheet 1 — MLB       (batters + pitchers)
  Sheet 2 — NBA       (guards / forwards / centres)
  Sheet 3 — NHL       (skaters + goalies)
  Sheet 4 — NFL       (QB / RB / WR / TE / Defence)
  Sheet 5 — Cricket   (batters / bowlers / all-rounders, by format)

Each sheet has:
  Rating (0–100)            — composite of the sport's key metrics,
                              normalised within position group
  ScoutingUncertainty (0–100) — five-factor confidence model:
                                sample size · league quality · age ·
                                availability · data completeness
  Confidence Label          — High / Good / Moderate / Low / Very Low

Usage
─────
  python export_multi_sport.py [--output "data/Multi-Sport Scouting Report.xlsx"]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore  # type: ignore

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
RNG      = np.random.default_rng(2024)

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _pct_rank(series: pd.Series) -> pd.Series:
    """0–100 percentile rank within series (higher = better)."""
    vals = series.fillna(series.median())
    return vals.rank(pct=True) * 100


def _pct_rank_inv(series: pd.Series) -> pd.Series:
    """0–100 percentile rank, inverted (lower raw value = better)."""
    return _pct_rank(-series)


def _composite(df: pd.DataFrame, spec: list[tuple[str, float, bool]]) -> pd.Series:
    """
    Weighted composite 0–100.
    spec = [(column, weight, higher_is_better), ...]
    """
    total_w = sum(w for _, w, _ in spec if _ is not None)
    score   = pd.Series(0.0, index=df.index)
    for col, w, higher in spec:
        if col not in df.columns:
            continue
        pct = _pct_rank(df[col]) if higher else _pct_rank_inv(df[col])
        score += (w / total_w) * pct
    return score.clip(0, 100).round(1)


def _uncertainty(
    df: pd.DataFrame,
    sample_col: str,
    sample_max: float,
    age_col: str,
    games_col: str,
    max_games: float,
    peak_age: tuple[float, float] = (25.0, 29.0),
) -> pd.Series:
    """
    Five-factor ScoutingUncertainty (0–100).
    1. Sample size   35 % — fewer reps → higher uncertainty
    2. League quality 25 % — top tier = 0.0 (all data here is top-league)
    3. Age           20 % — outside peak window → higher uncertainty
    4. Availability  10 % — games / max_games
    5. Data complete 10 % — fixed at 0 (full synthetic data)
    """
    # 1. Sample size
    _min = max(sample_max * 0.1, 1)
    samp = pd.to_numeric(df[sample_col], errors="coerce").fillna(_min).clip(lower=_min)
    f_sample = np.sqrt(_min / samp).clip(0, 1)

    # 2. League quality (all generated data = top tier → 0.0)
    f_league = pd.Series(0.05, index=df.index)

    # 3. Age
    age = pd.to_numeric(df[age_col], errors="coerce").fillna(27)
    f_age = pd.Series(0.0, index=df.index, dtype=float)
    f_age[age < 21]                              = 0.35
    f_age[age.between(21, 23.99)]                = 0.20
    f_age[age.between(peak_age[0], peak_age[1])] = 0.00
    f_age[age.between(peak_age[1], 31.99)]       = 0.12
    f_age[age >= 32]                             = 0.25

    # 4. Availability
    games = pd.to_numeric(df[games_col], errors="coerce").fillna(max_games * 0.5).clip(lower=1)
    f_avail = ((max_games - games) / max_games).clip(0, 1) * 0.40

    # 5. Data completeness (full data assumed)
    f_complete = pd.Series(0.0, index=df.index)

    combined = (
        0.35 * f_sample
        + 0.25 * f_league
        + 0.20 * f_age
        + 0.10 * f_avail
        + 0.10 * f_complete
    ).clip(0, 1)

    return (combined * 100).clip(0, 100).round(1)


def _confidence_label(u: pd.Series) -> pd.Series:
    def _lbl(v: float) -> str:
        if v <= 20: return "High Confidence"
        if v <= 35: return "Good Confidence"
        if v <= 50: return "Moderate Confidence"
        if v <= 65: return "Low Confidence"
        return "Very Low Confidence"
    return u.apply(_lbl)


def _write(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame) -> None:
    df.to_excel(writer, sheet_name=sheet[:31], index=False)
    ws = writer.sheets[sheet[:31]]
    for col_cells in ws.columns:
        try:
            width = max(len(str(col_cells[0].value or "")),
                        *(len(str(c.value or "")) for c in col_cells[1:8]))
            ws.column_dimensions[col_cells[0].column_letter].width = min(width + 2, 40)
        except Exception:
            pass
    print(f"  ✓ {sheet!r} — {len(df):,} rows")


# ── Name generators ─────────────────────────────────────────────────────────────

def _names(n: int, first_pool: list[str], last_pool: list[str]) -> list[str]:
    return [
        f"{RNG.choice(first_pool)} {RNG.choice(last_pool)}"
        for _ in range(n)
    ]


MLB_FIRST = ["Aaron","Bryce","Clayton","Fernando","Gerrit","Julio","Mookie","Nolan",
             "Paul","Ronald","Sandy","Trea","Vladimir","Yordan","Zack","Pete","Kyle",
             "Shane","Max","Cody","Bo","Freddie","Anthony","Jose","Francisco","Blake",
             "Logan","Luis","Spencer","Corbin","Tyler","Austin","Wander","Gunnar",
             "Bobby","Corey","Dansby","Ryan","Grayson","Triston"]
MLB_LAST  = ["Judge","Harper","Kershaw","Tatis","Cole","Rodriguez","Betts","Arenado",
             "Acuna","Alcantara","Turner","Guerrero","Alvarez","Wheeler","Alonso",
             "Tucker","Bieber","Freeman","Lindor","Webb","Castillo","Ohtani","deGrom",
             "Burnes","Verlander","Glasnow","Strider","Scherzer","McClanahan","Nola",
             "Flaherty","Manoah","Montgomery","Musgrove","Anderson","Swanson","Story"]

NBA_FIRST = ["LeBron","Stephen","Kevin","Giannis","Nikola","Luka","Joel","Jayson",
             "Damian","Anthony","Kawhi","Paul","Devin","Zion","Ja","James","Bam",
             "Karl-Anthony","Trae","Donovan","DeMar","Tyrese","Jordan","Shai","Chet",
             "Victor","Paolo","Scottie","Brandon","Jaren"]
NBA_LAST  = ["James","Curry","Durant","Antetokounmpo","Jokic","Doncic","Embiid",
             "Tatum","Lillard","Davis","Leonard","George","Booker","Williamson",
             "Morant","Harden","Adebayo","Towns","Young","Mitchell","DeRozan",
             "Maxey","Poole","Gilgeous-Alexander","Holmgren","Wembanyama","Banchero",
             "Barnes","Ingram","Jackson"]

NHL_FIRST = ["Connor","Nathan","Leon","Auston","David","Sidney","Alex","Nikita",
             "Artemi","Brad","Brayden","Mark","Patrick","Aleksander","Andrei",
             "Evgeni","William","Jonathan","Kyle","John","Sam","Mikko","Dylan",
             "Cale","Roman","Quinn","Jack","Tim","Anze","Adam"]
NHL_LAST  = ["McDavid","MacKinnon","Draisaitl","Matthews","Pastrnak","Crosby",
             "Ovechkin","Kucherov","Panarin","Marchand","Point","Stone","Kane",
             "Barkov","Svechnikov","Malkin","Nylander","Huberdeau","Connor",
             "Tavares","Reinhart","Rantanen","Larkin","Makar","Josi","Hughes",
             "Eichel","Stutzle","Kopitar","Pelech"]

NFL_FIRST = ["Patrick","Josh","Joe","Lamar","Justin","Jalen","Brock","Trevor",
             "Tua","Dak","Tyreek","Stefon","Justin","Davante","DeAndre","CeeDee",
             "Cooper","Ja'Marr","Christian","Derrick","Austin","Nick","Travis",
             "Mark","Sam","Aaron","Myles","Micah","T.J.","Maxx"]
NFL_LAST  = ["Mahomes","Allen","Burrow","Jackson","Herbert","Hurts","Purdy","Lawrence",
             "Tagovailoa","Prescott","Hill","Diggs","Jefferson","Adams","Hopkins",
             "Lamb","Kupp","Chase","McCaffrey","Henry","Ekeler","Chubb","Kelce",
             "Andrews","LaPorta","Donald","Garrett","Parsons","Watt","Crosby"]

CRICKET_FIRST = ["Virat","Rohit","Steve","David","Kane","Joe","Babar","Ben","Pat",
                 "Jasprit","Kagiso","Mitchell","Trent","Mitchell","Shakib","Quinton",
                 "Rishabh","KL","Shubman","Yashasvi","Mohammed","James","Stuart",
                 "Travis","Marnus","Devon","Tom","Shreyas","Hardik","Ravindra"]
CRICKET_LAST  = ["Kohli","Sharma","Smith","Warner","Williamson","Root","Azam","Stokes",
                 "Cummins","Bumrah","Rabada","Starc","Boult","Marsh","Al Hasan",
                 "de Kock","Pant","Rahul","Gill","Jaiswal","Siraj","Anderson",
                 "Broad","Head","Labuschagne","Conway","Latham","Iyer","Pandya","Ravindra"]


# ══════════════════════════════════════════════════════════════════════════════
# MLB
# ══════════════════════════════════════════════════════════════════════════════

def build_mlb(n_batters: int = 120, n_pitchers: int = 80) -> pd.DataFrame:
    """Generate MLB batter + pitcher data with Rating and ScoutingUncertainty."""

    positions_b = ["C","1B","2B","3B","SS","LF","CF","RF","DH"]
    positions_p = ["SP","SP","SP","RP","RP","CL"]

    # ── Batters ────────────────────────────────────────────────────────────────
    pos_b = RNG.choice(positions_b, n_batters)
    age_b = RNG.integers(20, 38, n_batters).astype(float)
    games_b = np.clip(RNG.normal(130, 25, n_batters), 30, 162).astype(int)
    pa_b    = (games_b * RNG.uniform(3.5, 4.5, n_batters)).astype(int)

    batters = pd.DataFrame({
        "Player":          _names(n_batters, MLB_FIRST, MLB_LAST),
        "Team":            RNG.choice(["Yankees","Dodgers","Braves","Astros","Mets",
                                       "Phillies","Cardinals","Cubs","Red Sox","Giants",
                                       "Blue Jays","Rays","Mariners","Padres","Tigers"], n_batters),
        "Position":        pos_b,
        "Age":             age_b,
        "Games":           games_b,
        "Plate Appearances": pa_b,
        "BA":              np.clip(RNG.normal(0.265, 0.035, n_batters), 0.180, 0.370),
        "OBP":             np.clip(RNG.normal(0.335, 0.040, n_batters), 0.250, 0.470),
        "SLG":             np.clip(RNG.normal(0.440, 0.065, n_batters), 0.300, 0.700),
        "OPS":             None,
        "OPS+":            np.clip(RNG.normal(108, 25, n_batters), 50, 210).astype(int),
        "HR":              np.clip(RNG.normal(18, 12, n_batters), 0, 62).astype(int),
        "RBI":             np.clip(RNG.normal(72, 28, n_batters), 10, 150).astype(int),
        "SB":              np.clip(RNG.normal(10, 12, n_batters), 0, 70).astype(int),
        "wRC+":            np.clip(RNG.normal(108, 24, n_batters), 45, 210).astype(int),
        "wOBA":            np.clip(RNG.normal(0.335, 0.045, n_batters), 0.240, 0.480),
        "WAR":             np.clip(RNG.normal(2.5, 2.2, n_batters), -1.5, 9.5),
        "K%":              np.clip(RNG.normal(22, 6, n_batters), 8, 42),
        "BB%":             np.clip(RNG.normal(9, 3, n_batters), 3, 20),
        "Exit Velocity":   np.clip(RNG.normal(89, 4, n_batters), 78, 98),
        "Hard Hit%":       np.clip(RNG.normal(38, 8, n_batters), 18, 60),
        "Barrel%":         np.clip(RNG.normal(8, 4, n_batters), 1, 22),
        "Sprint Speed":    np.clip(RNG.normal(27, 1.5, n_batters), 23, 31),
        "Type":            "Batter",
    })
    batters["OPS"] = (batters["OBP"] + batters["SLG"]).round(3)

    # Rating: wRC+ 35%, WAR 30%, OPS+ 20%, BB%-K% 15%
    batters["Rating"] = _composite(batters, [
        ("wRC+",  0.35, True),
        ("WAR",   0.30, True),
        ("OPS+",  0.20, True),
        ("BB%",   0.08, True),
        ("K%",    0.07, False),
    ])

    batters["ScoutingUncertainty"] = _uncertainty(
        batters, "Plate Appearances", 650, "Age", "Games", 162, peak_age=(26, 31)
    )

    # ── Pitchers ───────────────────────────────────────────────────────────────
    pos_p   = RNG.choice(positions_p, n_pitchers)
    age_p   = RNG.integers(21, 40, n_pitchers).astype(float)
    games_p = np.where(pos_p == "SP",
                       np.clip(RNG.normal(27, 5, n_pitchers), 5, 33),
                       np.clip(RNG.normal(55, 15, n_pitchers), 10, 80)).astype(int)
    ip      = np.where(pos_p == "SP",
                       np.clip(RNG.normal(170, 35, n_pitchers), 20, 230),
                       np.clip(RNG.normal(60, 20, n_pitchers), 10, 90))

    pitchers = pd.DataFrame({
        "Player":    _names(n_pitchers, MLB_FIRST, MLB_LAST),
        "Team":      RNG.choice(["Yankees","Dodgers","Braves","Astros","Mets",
                                  "Phillies","Cardinals","Cubs","Red Sox","Giants",
                                  "Blue Jays","Rays","Mariners","Padres","Tigers"], n_pitchers),
        "Position":  pos_p,
        "Age":       age_p,
        "Games":     games_p,
        "Plate Appearances": ip,  # IP used as sample proxy
        "IP":        ip.round(1),
        "ERA":       np.clip(RNG.normal(3.85, 0.90, n_pitchers), 1.5, 7.0),
        "FIP":       np.clip(RNG.normal(3.90, 0.85, n_pitchers), 1.8, 6.5),
        "xFIP":      np.clip(RNG.normal(3.95, 0.70, n_pitchers), 2.0, 6.0),
        "WHIP":      np.clip(RNG.normal(1.22, 0.22, n_pitchers), 0.70, 2.00),
        "K/9":       np.clip(RNG.normal(9.5, 2.2, n_pitchers), 4.0, 16.0),
        "BB/9":      np.clip(RNG.normal(3.0, 0.9, n_pitchers), 0.8, 6.0),
        "K-BB%":     np.clip(RNG.normal(17, 7, n_pitchers), 2, 38),
        "ERA+":      np.clip(RNG.normal(110, 28, n_pitchers), 40, 220).astype(int),
        "WAR":       np.clip(RNG.normal(2.2, 2.0, n_pitchers), -1.0, 8.5),
        "HR/9":      np.clip(RNG.normal(1.15, 0.35, n_pitchers), 0.2, 2.5),
        "GB%":       np.clip(RNG.normal(44, 8, n_pitchers), 25, 65),
        "Saves":     np.where(pos_p == "CL", np.clip(RNG.normal(28, 12, n_pitchers), 0, 50).astype(int), 0),
        "Type":      "Pitcher",
    })

    # Rating: FIP- proxy (ERA+) 35%, K-BB% 30%, WAR 25%, GB% 10%
    pitchers["Rating"] = _composite(pitchers, [
        ("ERA+",  0.35, True),
        ("K-BB%", 0.30, True),
        ("WAR",   0.25, True),
        ("GB%",   0.10, True),
    ])

    pitchers["ScoutingUncertainty"] = _uncertainty(
        pitchers, "IP", 200, "Age", "Games", 33, peak_age=(26, 32)
    )

    df = pd.concat([batters, pitchers], ignore_index=True)
    df["ConfidenceLabel"] = _confidence_label(df["ScoutingUncertainty"])
    df = df.sort_values("Rating", ascending=False).reset_index(drop=True)

    # Round numeric columns
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].round(2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# NBA
# ══════════════════════════════════════════════════════════════════════════════

def build_nba(n: int = 150) -> pd.DataFrame:
    positions = ["PG","SG","SF","PF","C"]
    pos  = RNG.choice(positions, n)
    age  = RNG.integers(19, 38, n).astype(float)
    gp   = np.clip(RNG.normal(62, 16, n), 10, 82).astype(int)
    mpg  = np.clip(RNG.normal(27, 7, n), 10, 38)

    df = pd.DataFrame({
        "Player":          _names(n, NBA_FIRST, NBA_LAST),
        "Team":            RNG.choice(["Lakers","Warriors","Celtics","Bucks","76ers",
                                        "Nuggets","Suns","Heat","Mavericks","Nets",
                                        "Clippers","Grizzlies","Thunder","Cavaliers","Timberwolves"], n),
        "Position":        pos,
        "Age":             age,
        "Games Played":    gp,
        "Min per Game":    mpg.round(1),
        "PTS":             np.clip(RNG.normal(17, 7, n), 4, 36),
        "REB":             np.clip(RNG.normal(6, 3, n), 1, 16),
        "AST":             np.clip(RNG.normal(4, 3, n), 0.5, 12),
        "STL":             np.clip(RNG.normal(1.1, 0.4, n), 0.2, 2.5),
        "BLK":             np.clip(RNG.normal(0.7, 0.6, n), 0, 3.2),
        "TOV":             np.clip(RNG.normal(2.0, 0.8, n), 0.5, 5.0),
        "FG%":             np.clip(RNG.normal(0.475, 0.055, n), 0.330, 0.650),
        "3P%":             np.clip(RNG.normal(0.355, 0.055, n), 0.200, 0.480),
        "FT%":             np.clip(RNG.normal(0.780, 0.075, n), 0.550, 0.960),
        "TS%":             np.clip(RNG.normal(0.575, 0.055, n), 0.430, 0.720),
        "PER":             np.clip(RNG.normal(16, 5, n), 5, 32),
        "BPM":             np.clip(RNG.normal(1.5, 4, n), -6, 12),
        "VORP":            np.clip(RNG.normal(1.5, 1.8, n), -0.5, 8),
        "Win Shares":      np.clip(RNG.normal(5, 3.5, n), -0.5, 18),
        "WS/48":           np.clip(RNG.normal(0.110, 0.065, n), -0.030, 0.290),
        "USG%":            np.clip(RNG.normal(22, 5, n), 10, 36),
        "Net Rating":      np.clip(RNG.normal(1.5, 6, n), -12, 16),
        "+/-":             np.clip(RNG.normal(1.5, 4.5, n), -10, 14),
    })

    # Rating: BPM 30%, WS/48 25%, PER 20%, TS% 15%, Net Rating 10%
    df["Rating"] = _composite(df, [
        ("BPM",        0.30, True),
        ("WS/48",      0.25, True),
        ("PER",        0.20, True),
        ("TS%",        0.15, True),
        ("Net Rating", 0.10, True),
    ])

    df["ScoutingUncertainty"] = _uncertainty(
        df, "Games Played", 82, "Age", "Games Played", 82, peak_age=(24, 29)
    )
    df["ConfidenceLabel"] = _confidence_label(df["ScoutingUncertainty"])

    num = df.select_dtypes(include="number").columns
    df[num] = df[num].round(2)
    return df.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# NHL
# ══════════════════════════════════════════════════════════════════════════════

def build_nhl(n_skaters: int = 120, n_goalies: int = 40) -> pd.DataFrame:
    sk_pos = RNG.choice(["LW","RW","C","LD","RD"], n_skaters,
                        p=[0.20, 0.20, 0.25, 0.18, 0.17])
    age_sk = RNG.integers(18, 38, n_skaters).astype(float)
    gp_sk  = np.clip(RNG.normal(65, 14, n_skaters), 10, 82).astype(int)

    skaters = pd.DataFrame({
        "Player":    _names(n_skaters, NHL_FIRST, NHL_LAST),
        "Team":      RNG.choice(["Oilers","Avalanche","Bruins","Panthers","Rangers",
                                  "Lightning","Golden Knights","Maple Leafs","Hurricanes",
                                  "Stars","Kings","Wild","Jets","Blues","Capitals"], n_skaters),
        "Position":  sk_pos,
        "Age":       age_sk,
        "Games":     gp_sk,
        "Goals":     np.clip(RNG.normal(20, 12, n_skaters), 0, 65).astype(int),
        "Assists":   np.clip(RNG.normal(28, 15, n_skaters), 0, 85).astype(int),
        "Points":    None,
        "Points/GP": None,
        "+/-":       np.clip(RNG.normal(2, 12, n_skaters), -30, 35).astype(int),
        "PIM":       np.clip(RNG.normal(35, 25, n_skaters), 0, 130).astype(int),
        "TOI/GP":    np.clip(RNG.normal(17, 4, n_skaters), 8, 26),
        "Corsi%":    np.clip(RNG.normal(50, 4, n_skaters), 38, 62),
        "Fenwick%":  np.clip(RNG.normal(50, 4, n_skaters), 38, 62),
        "xGF%":      np.clip(RNG.normal(50, 5, n_skaters), 35, 65),
        "Sh%":       np.clip(RNG.normal(10.5, 3.5, n_skaters), 2, 22),
        "iCF/60":    np.clip(RNG.normal(12, 4, n_skaters), 3, 24),
        "GAR":       np.clip(RNG.normal(4, 6, n_skaters), -8, 22),
        "Type":      "Skater",
    })
    skaters["Points"]    = (skaters["Goals"] + skaters["Assists"])
    skaters["Points/GP"] = (skaters["Points"] / skaters["Games"].clip(lower=1)).round(2)

    # Rating: GAR 35%, Points/GP 30%, Corsi% 20%, xGF% 15%
    skaters["Rating"] = _composite(skaters, [
        ("GAR",       0.35, True),
        ("Points/GP", 0.30, True),
        ("Corsi%",    0.20, True),
        ("xGF%",      0.15, True),
    ])
    skaters["ScoutingUncertainty"] = _uncertainty(
        skaters, "Games", 82, "Age", "Games", 82, peak_age=(25, 30)
    )

    # ── Goalies ───────────────────────────────────────────────────────────────
    age_g = RNG.integers(20, 40, n_goalies).astype(float)
    gp_g  = np.clip(RNG.normal(45, 15, n_goalies), 10, 70).astype(int)

    goalies = pd.DataFrame({
        "Player":    _names(n_goalies, NHL_FIRST, NHL_LAST),
        "Team":      RNG.choice(["Oilers","Avalanche","Bruins","Panthers","Rangers",
                                  "Lightning","Golden Knights","Maple Leafs","Hurricanes",
                                  "Stars","Kings","Wild","Jets","Blues","Capitals"], n_goalies),
        "Position":  "G",
        "Age":       age_g,
        "Games":     gp_g,
        "Goals":     np.nan,
        "Assists":   np.nan,
        "Points":    np.nan,
        "Points/GP": np.nan,
        "+/-":       np.nan,
        "PIM":       np.nan,
        "TOI/GP":    np.nan,
        "Corsi%":    np.nan,
        "Fenwick%":  np.nan,
        "xGF%":      np.nan,
        "Sh%":       np.nan,
        "iCF/60":    np.nan,
        "SV%":       np.clip(RNG.normal(0.912, 0.012, n_goalies), 0.875, 0.940),
        "GAA":       np.clip(RNG.normal(2.65, 0.40, n_goalies), 1.60, 4.20),
        "GSAA":      np.clip(RNG.normal(5, 10, n_goalies), -18, 30),
        "GAR":       np.clip(RNG.normal(4, 8, n_goalies), -12, 25),
        "Type":      "Goalie",
    })
    goalies["Rating"] = _composite(goalies, [
        ("GSAA",  0.40, True),
        ("SV%",   0.35, True),
        ("GAA",   0.15, False),
        ("GAR",   0.10, True),
    ])
    goalies["ScoutingUncertainty"] = _uncertainty(
        goalies, "Games", 70, "Age", "Games", 70, peak_age=(26, 33)
    )

    df = pd.concat([skaters, goalies], ignore_index=True)
    df["ConfidenceLabel"] = _confidence_label(df["ScoutingUncertainty"])
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].round(2)
    return df.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# NFL
# ══════════════════════════════════════════════════════════════════════════════

def build_nfl(n_qb: int = 32, n_rb: int = 40, n_wr: int = 50,
              n_te: int = 25, n_def: int = 60) -> pd.DataFrame:
    frames = []

    # QB
    age_qb = RNG.integers(22, 40, n_qb).astype(float)
    gp_qb  = np.clip(RNG.normal(14, 4, n_qb), 1, 17).astype(int)
    qbs = pd.DataFrame({
        "Player":       _names(n_qb, NFL_FIRST, NFL_LAST),
        "Team":         RNG.choice(["Chiefs","Bills","Eagles","49ers","Cowboys",
                                     "Ravens","Bengals","Chargers","Dolphins","Jets",
                                     "Packers","Seahawks","Rams","Saints","Broncos"], n_qb),
        "Position":     "QB",
        "Age":          age_qb,
        "Games":        gp_qb,
        "Dropbacks":    (gp_qb * RNG.uniform(28, 42, n_qb)).astype(int),
        "Pass Yards":   np.clip(RNG.normal(3800, 900, n_qb), 500, 5500).astype(int),
        "Pass TD":      np.clip(RNG.normal(27, 9, n_qb), 3, 55).astype(int),
        "INT":          np.clip(RNG.normal(10, 4, n_qb), 1, 22).astype(int),
        "Comp%":        np.clip(RNG.normal(65, 5, n_qb), 52, 78),
        "YPA":          np.clip(RNG.normal(7.5, 1.2, n_qb), 5.0, 11.0),
        "QBR":          np.clip(RNG.normal(55, 18, n_qb), 15, 90),
        "Passer Rating":np.clip(RNG.normal(95, 18, n_qb), 50, 135),
        "Rush Yards QB":np.clip(RNG.normal(250, 280, n_qb), 0, 900).astype(int),
        "DYAR":         np.clip(RNG.normal(400, 400, n_qb), -500, 1500).astype(int),
        "DVOA%":        np.clip(RNG.normal(10, 20, n_qb), -35, 55),
        "Type":         "QB",
    })
    qbs["TD/INT"] = (qbs["Pass TD"] / qbs["INT"].clip(lower=1)).round(2)
    qbs["Rating"] = _composite(qbs, [
        ("QBR",          0.35, True),
        ("DYAR",         0.25, True),
        ("YPA",          0.20, True),
        ("TD/INT",       0.15, True),
        ("Comp%",        0.05, True),
    ])
    qbs["ScoutingUncertainty"] = _uncertainty(
        qbs, "Dropbacks", 550, "Age", "Games", 17, peak_age=(26, 32)
    )
    frames.append(qbs)

    # RB
    age_rb = RNG.integers(21, 33, n_rb).astype(float)
    gp_rb  = np.clip(RNG.normal(14, 3, n_rb), 1, 17).astype(int)
    rbs = pd.DataFrame({
        "Player":       _names(n_rb, NFL_FIRST, NFL_LAST),
        "Team":         RNG.choice(["Chiefs","Bills","Eagles","49ers","Cowboys",
                                     "Ravens","Bengals","Chargers","Dolphins","Jets"], n_rb),
        "Position":     "RB",
        "Age":          age_rb,
        "Games":        gp_rb,
        "Dropbacks":    np.nan,
        "Rush Yards":   np.clip(RNG.normal(850, 380, n_rb), 50, 2100).astype(int),
        "Rush TD":      np.clip(RNG.normal(7, 4, n_rb), 0, 21).astype(int),
        "YPC":          np.clip(RNG.normal(4.3, 0.7, n_rb), 2.5, 6.5),
        "Receptions":   np.clip(RNG.normal(40, 20, n_rb), 5, 90).astype(int),
        "Rec Yards":    np.clip(RNG.normal(320, 180, n_rb), 20, 700).astype(int),
        "Yards/Carry":  np.clip(RNG.normal(4.3, 0.7, n_rb), 2.5, 6.5),
        "Broken Tackles": np.clip(RNG.normal(25, 15, n_rb), 0, 70).astype(int),
        "DVOA%":        np.clip(RNG.normal(5, 20, n_rb), -35, 45),
        "DYAR":         np.clip(RNG.normal(80, 130, n_rb), -200, 400).astype(int),
        "Carries":      np.clip(RNG.normal(180, 70, n_rb), 20, 350).astype(int),
        "Type":         "RB",
    })
    rbs["Rating"] = _composite(rbs, [
        ("DVOA%",    0.35, True),
        ("YPC",      0.30, True),
        ("DYAR",     0.20, True),
        ("Receptions", 0.15, True),
    ])
    rbs["ScoutingUncertainty"] = _uncertainty(
        rbs, "Carries", 300, "Age", "Games", 17, peak_age=(23, 28)
    )
    frames.append(rbs)

    # WR / TE
    for pos, nn in [("WR", n_wr), ("TE", n_te)]:
        age_w = RNG.integers(21, 35, nn).astype(float)
        gp_w  = np.clip(RNG.normal(14, 3, nn), 1, 17).astype(int)
        tgts  = np.clip(RNG.normal(90, 35, nn), 10, 180).astype(int)
        rec   = (tgts * RNG.uniform(0.55, 0.80, nn)).astype(int)
        yards = np.clip(RNG.normal(800, 350, nn), 50, 1850).astype(int)
        skill = pd.DataFrame({
            "Player":    _names(nn, NFL_FIRST, NFL_LAST),
            "Team":      RNG.choice(["Chiefs","Bills","Eagles","49ers","Cowboys",
                                      "Ravens","Bengals","Chargers","Dolphins","Jets"], nn),
            "Position":  pos,
            "Age":       age_w,
            "Games":     gp_w,
            "Dropbacks": np.nan,
            "Targets":   tgts,
            "Receptions":rec,
            "Rec Yards": yards,
            "Rec TD":    np.clip(RNG.normal(6, 4, nn), 0, 18).astype(int),
            "Catch%":    (rec / tgts.clip(1) * 100).round(1),
            "YPR":       (yards / rec.clip(1)).round(1),
            "Air Yards": np.clip(RNG.normal(1100, 500, nn), 100, 2500).astype(int),
            "YAC":       np.clip(RNG.normal(300, 150, nn), 0, 700).astype(int),
            "DVOA%":     np.clip(RNG.normal(8, 22, nn), -40, 60),
            "DYAR":      np.clip(RNG.normal(120, 160, nn), -250, 500).astype(int),
            "Type":      pos,
        })
        skill["Rating"] = _composite(skill, [
            ("DVOA%",     0.35, True),
            ("DYAR",      0.25, True),
            ("YPR",       0.20, True),
            ("Catch%",    0.20, True),
        ])
        skill["ScoutingUncertainty"] = _uncertainty(
            skill, "Targets", 150, "Age", "Games", 17, peak_age=(24, 29)
        )
        frames.append(skill)

    # Defence (generic)
    age_d = RNG.integers(21, 36, n_def).astype(float)
    gp_d  = np.clip(RNG.normal(15, 2, n_def), 5, 17).astype(int)
    def_pos = RNG.choice(["DE","DT","LB","CB","S"], n_def)
    defence = pd.DataFrame({
        "Player":         _names(n_def, NFL_FIRST, NFL_LAST),
        "Team":           RNG.choice(["Chiefs","Bills","Eagles","49ers","Cowboys",
                                       "Ravens","Bengals","Chargers","Dolphins","Jets"], n_def),
        "Position":       def_pos,
        "Age":            age_d,
        "Games":          gp_d,
        "Dropbacks":      np.nan,
        "Solo Tackles":   np.clip(RNG.normal(50, 25, n_def), 5, 130).astype(int),
        "Sacks":          np.clip(RNG.normal(4.5, 4, n_def), 0, 22.5),
        "TFL":            np.clip(RNG.normal(6, 4, n_def), 0, 22).astype(int),
        "QB Hits":        np.clip(RNG.normal(12, 8, n_def), 0, 40).astype(int),
        "INT":            np.clip(RNG.normal(1.5, 1.5, n_def), 0, 8).astype(int),
        "PBU":            np.clip(RNG.normal(5, 4, n_def), 0, 20).astype(int),
        "Forced Fumbles": np.clip(RNG.normal(1.2, 1.2, n_def), 0, 6).astype(int),
        "DVOA%":          np.clip(RNG.normal(5, 20, n_def), -35, 55),
        "DYAR":           np.clip(RNG.normal(40, 80, n_def), -150, 250).astype(int),
        "Type":           "Defence",
    })
    defence["Rating"] = _composite(defence, [
        ("DVOA%",    0.30, True),
        ("Sacks",    0.25, True),
        ("DYAR",     0.25, True),
        ("TFL",      0.20, True),
    ])
    defence["ScoutingUncertainty"] = _uncertainty(
        defence, "Solo Tackles", 100, "Age", "Games", 17, peak_age=(25, 30)
    )
    frames.append(defence)

    df = pd.concat(frames, ignore_index=True)
    df["ConfidenceLabel"] = _confidence_label(df["ScoutingUncertainty"])
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].round(2)
    return df.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cricket
# ══════════════════════════════════════════════════════════════════════════════

def build_cricket(n: int = 120) -> pd.DataFrame:
    formats   = RNG.choice(["Test","ODI","T20I"], n, p=[0.35, 0.35, 0.30])
    roles     = RNG.choice(["Batter","Bowler","All-rounder","WK-Batter"], n,
                           p=[0.40, 0.30, 0.20, 0.10])
    age       = RNG.integers(18, 40, n).astype(float)
    innings   = np.where(formats == "Test",
                         np.clip(RNG.normal(30, 20, n), 5, 100).astype(int),
                         np.clip(RNG.normal(40, 20, n), 5, 100).astype(int))

    # Strike rate varies by format
    sr_mean = np.where(formats == "T20I", 135,
              np.where(formats == "ODI",  88, 55))

    df = pd.DataFrame({
        "Player":         _names(n, CRICKET_FIRST, CRICKET_LAST),
        "Team":           RNG.choice(["India","Australia","England","Pakistan","New Zealand",
                                       "South Africa","West Indies","Sri Lanka","Bangladesh","Zimbabwe"], n),
        "Role":           roles,
        "Format":         formats,
        "Age":            age,
        "Innings Batted": innings,
        # Batting
        "Bat Average":    np.where(roles == "Bowler",
                                   np.clip(RNG.normal(15, 8, n), 3, 35),
                                   np.clip(RNG.normal(38, 15, n), 10, 70)),
        "Strike Rate":    np.clip(sr_mean + RNG.normal(0, 15, n), 30, 220),
        "Runs":           np.clip(RNG.normal(1200, 900, n), 50, 4500).astype(int),
        "100s":           np.clip(RNG.normal(3, 4, n), 0, 18).astype(int),
        "50s":            np.clip(RNG.normal(7, 6, n), 0, 30).astype(int),
        "HS":             np.clip(RNG.normal(95, 50, n), 10, 250).astype(int),
        # Bowling
        "Wickets":        np.where(roles == "Batter",
                                   np.clip(RNG.normal(5, 8, n), 0, 30).astype(int),
                                   np.clip(RNG.normal(60, 40, n), 5, 200).astype(int)),
        "Bowl Average":   np.where(roles == "Batter",
                                   np.clip(RNG.normal(50, 20, n), 25, 100),
                                   np.clip(RNG.normal(28, 8, n), 14, 55)),
        "Economy":        np.where(formats == "T20I",
                                   np.clip(RNG.normal(8.0, 1.2, n), 5.0, 12.0),
                          np.where(formats == "ODI",
                                   np.clip(RNG.normal(5.2, 0.8, n), 3.5, 8.0),
                                   np.clip(RNG.normal(3.0, 0.6, n), 1.8, 5.5))),
        "Bowl SR":        np.clip(RNG.normal(30, 12, n), 10, 65),
        "5-wicket hauls": np.where(roles == "Batter", 0,
                                   np.clip(RNG.normal(3, 4, n), 0, 18).astype(int)),
        "Innings Bowled": np.where(roles == "Batter",
                                   np.clip(RNG.normal(5, 5, n), 0, 20).astype(int),
                                   np.clip(RNG.normal(35, 20, n), 5, 100).astype(int)),
    })

    # Bat rating (higher avg + SR → better)
    bat_score = _composite(df, [
        ("Bat Average", 0.45, True),
        ("Strike Rate", 0.30, True),
        ("100s",        0.15, True),
        ("50s",         0.10, True),
    ])
    # Bowl rating (lower avg + economy + SR → better)
    bowl_score = _composite(df, [
        ("Bowl Average", 0.40, False),
        ("Economy",      0.30, False),
        ("Bowl SR",      0.20, False),
        ("Wickets",      0.10, True),
    ])

    # Weight by role
    bat_w  = pd.Series(np.where(roles == "Batter",      0.90,
                       np.where(roles == "WK-Batter",    0.80,
                       np.where(roles == "All-rounder",  0.50, 0.20))), index=df.index)
    bowl_w = 1 - bat_w

    df["Rating"]       = (bat_w * bat_score + bowl_w * bowl_score).clip(0, 100).round(1)
    df["ScoutingUncertainty"] = _uncertainty(
        df, "Innings Batted", 80, "Age", "Innings Batted", 80, peak_age=(26, 32)
    )
    df["ConfidenceLabel"] = _confidence_label(df["ScoutingUncertainty"])

    num = df.select_dtypes(include="number").columns
    df[num] = df[num].round(2)
    return df.sort_values("Rating", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(output: Path) -> None:
    print(f"\n{'='*60}")
    print("  Multi-Sport Scouting Report")
    print(f"{'='*60}")

    print("Building MLB data…")
    mlb = build_mlb()
    print("Building NBA data…")
    nba = build_nba()
    print("Building NHL data…")
    nhl = build_nhl()
    print("Building NFL data…")
    nfl = build_nfl()
    print("Building Cricket data…")
    cricket = build_cricket()

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting workbook → {output}")

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _write(writer, "MLB",     mlb)
        _write(writer, "NBA",     nba)
        _write(writer, "NHL",     nhl)
        _write(writer, "NFL",     nfl)
        _write(writer, "Cricket", cricket)

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {size_mb:.1f} MB → {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-sport scouting workbook")
    parser.add_argument(
        "--output", type=Path,
        default=DATA_DIR / "Multi-Sport Scouting Report.xlsx",
    )
    args = parser.parse_args()
    run(args.output)
