"""
Build IMPECT ↔ Wyscout player linking database.

Matches players across both databases for the 17 leagues that appear in both,
using normalized last name + fuzzy team name (+ first-initial tie-break).

Output: data/IMPECT_Wyscout_Link.csv
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

DATA_DIR = Path("data")
WYSCOUT_DIR = DATA_DIR / "Wyscout DB"
OUT_FILE = DATA_DIR / "IMPECT_Wyscout_Link.csv"

# IMPECT competition_name + country → Wyscout filename
# Note: "Fortuna Liga" covers both Czech and Slovak leagues
LEAGUE_MAP: list[tuple[str, str, str]] = [
    # (impect_competition_name, country_nation,       wyscout_file)
    ("2. Liga",                    "Austria",         "Austria II.xlsx"),
    ("Challenger Pro League",      "Belgium",         "Belgium II.xlsx"),
    ("Fortuna Liga",               "Czechia",         "Czech.xlsx"),
    ("Chance Narodni Liga",        "Czechia",         "Czech II.xlsx"),
    ("Fortuna Liga",               "Slovakia",        "Slovakia.xlsx"),
    ("1.Division",                 "Denmark",         "Denmark II.xlsx"),
    ("Veikkausliiga",              "Finland",         "Finland.xlsx"),
    ("Nemzeti Bajnokság",          "Hungary",         "Hungary.xlsx"),
    ("Virsliga",                   "Latvia",          "Latvia.xlsx"),
    ("Keuken Kampioen Divisie",    "Netherlands",     "Netherlands II.xlsx"),
    ("OBOS-ligaen",                "Norway",          "Norway II.xlsx"),
    ("PKO BP Ekstraklasa",         "Poland",          "Poland.xlsx"),
    ("Betclic 1 Liga",             "Poland",          "Poland II.xlsx"),
    ("Prva Liga",                  "Serbia",          "Slovenia.xlsx"),   # IMPECT country_nation mislabelled; teams are Slovenian
    ("Challenge League",           "Switzerland",     "Switzerland II.xlsx"),
    ("Superettan",                 "Sweden",          "Sweden II.xlsx"),
]

# ── Text normalisation ────────────────────────────────────────────────────────

def norm(s: object) -> str:
    """Lowercase ASCII, strip punctuation — for fuzzy comparison."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def extract_last(name: object) -> str:
    """Return last whitespace-delimited token of a name string."""
    n = norm(str(name)) if name else ""
    return n.split()[-1] if n else ""


def impect_last(row: pd.Series) -> str:
    """Get normalised last name from an IMPECT row."""
    ln = row.get("lastname")
    if isinstance(ln, str) and ln.strip():
        return norm(ln.split()[-1])
    cn = row.get("commonname", "")
    return extract_last(cn)


def impect_first_initial(row: pd.Series) -> str:
    """Get first initial from an IMPECT row."""
    fn = row.get("firstname")
    if isinstance(fn, str) and fn.strip():
        return norm(fn)[0] if norm(fn) else ""
    cn = row.get("commonname", "")
    parts = norm(str(cn)).split()
    return parts[0][0] if len(parts) > 1 else ""


def wyscout_last(ws_name: str) -> str:
    """Extract normalised last name from Wyscout 'F. Lastname' format."""
    n = norm(ws_name)
    # Drop leading initial+dot patterns like "a. " or "ab. "
    n = re.sub(r"^[a-z]{1,3}\. ?", "", n)
    return n.split()[-1] if n else ""


def wyscout_first_initial(ws_name: str) -> str:
    """Extract first initial from Wyscout 'F. Lastname' format."""
    n = norm(ws_name)
    m = re.match(r"^([a-z])", n)
    return m.group(1) if m else ""


# ── Matching ─────────────────────────────────────────────────────────────────

TEAM_THRESHOLD_HIGH   = 75
TEAM_THRESHOLD_MEDIUM = 50


def best_team_match(imp_team: str, ws_teams: list[str]) -> tuple[str, int]:
    """Return (best_ws_team, score) using token-set fuzzy ratio.

    token_set_ratio handles common prefix/suffix patterns well:
    'FC Admira Wacker Mödling' vs 'Admira' → 100
    'SK Austria Klagenfurt' vs 'Austria Klagenfurt' → 100
    'SKN St. Pölten' vs 'Blau-Weiß Linz' → 32 (correctly low)
    """
    if not ws_teams:
        return "", 0
    imp_n = norm(imp_team)
    ws_norms = [norm(t) for t in ws_teams]
    result = process.extractOne(imp_n, ws_norms, scorer=fuzz.token_set_ratio)
    if result is None:
        return "", 0
    _, score, idx = result
    return ws_teams[idx], score


def match_league(
    imp_df: pd.DataFrame,
    ws_df: pd.DataFrame,
    ws_file: str,
) -> list[dict]:
    """Match players between one IMPECT competition slice and its Wyscout file."""

    # Pre-index Wyscout players by normalised last name
    ws_df = ws_df.copy()
    ws_df["_ws_last"] = ws_df["Player"].astype(str).apply(wyscout_last)
    ws_df["_ws_init"] = ws_df["Player"].astype(str).apply(wyscout_first_initial)
    ws_by_last: dict[str, list[int]] = {}
    for idx, row in ws_df.iterrows():
        k = row["_ws_last"]
        if k:
            ws_by_last.setdefault(k, []).append(idx)

    all_ws_teams = ws_df["Team"].dropna().astype(str).unique().tolist()

    records: list[dict] = []

    for _, imp_row in imp_df.iterrows():
        last = impect_last(imp_row)
        init = impect_first_initial(imp_row)
        imp_team = str(imp_row.get("squadName", "") or "")
        _, team_score = best_team_match(imp_team, all_ws_teams)

        candidates = ws_by_last.get(last, [])

        rec: dict = {
            "IMPECT_PlayerID":   imp_row.get("playerId"),
            "IMPECT_Name":       imp_row.get("commonname"),
            "IMPECT_FirstName":  imp_row.get("firstname"),
            "IMPECT_LastName":   imp_row.get("lastname"),
            "IMPECT_Team":       imp_team,
            "IMPECT_Competition": imp_row.get("impect_competition_name"),
            "IMPECT_Country":    imp_row.get("country_nation"),
            "IMPECT_Season":     imp_row.get("impect_season"),
            "Wyscout_Name":      None,
            "Wyscout_Team":      None,
            "Wyscout_Position":  None,
            "Wyscout_Age":       None,
            "Wyscout_File":      ws_file,
            "MatchConfidence":   "NONE",
            "LastNameMatchScore": 100 if candidates else 0,
            "TeamMatchScore":    0,
            "InitialMatch":      None,
        }

        if not candidates:
            records.append(rec)
            continue

        # Score each candidate by team match
        scored: list[tuple[int, int, str]] = []  # (idx, team_score, ws_team)
        for idx in candidates:
            ws_row = ws_df.loc[idx]
            ws_team = str(ws_row.get("Team", "") or "")
            ts = fuzz.token_set_ratio(norm(imp_team), norm(ws_team))
            init_ok = (not init or not ws_df.loc[idx, "_ws_init"] or
                       init == ws_df.loc[idx, "_ws_init"])
            scored.append((idx, ts, ws_team, init_ok))

        # Sort by team_score desc, init_ok desc
        scored.sort(key=lambda x: (x[1], x[3]), reverse=True)
        best_idx, best_ts, best_team, best_init_ok = scored[0]
        best_ws_row = ws_df.loc[best_idx]

        # Confidence
        if best_ts >= TEAM_THRESHOLD_HIGH:
            conf = "HIGH"
        elif best_ts >= TEAM_THRESHOLD_MEDIUM:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        # Downgrade if initial mismatch on a clear match
        if conf == "HIGH" and not best_init_ok:
            conf = "MEDIUM"

        # Downgrade if multiple candidates with equally close team
        top_scores = [s for _, s, _, _ in scored]
        if len(top_scores) > 1 and top_scores[0] - top_scores[1] < 10 and conf == "HIGH":
            conf = "MEDIUM"

        rec.update({
            "Wyscout_Name":       best_ws_row.get("Player"),
            "Wyscout_Team":       best_ws_row.get("Team"),
            "Wyscout_Position":   best_ws_row.get("Position"),
            "Wyscout_Age":        best_ws_row.get("Age"),
            "MatchConfidence":    conf,
            "TeamMatchScore":     best_ts,
            "InitialMatch":       best_init_ok,
        })
        records.append(rec)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading IMPECT model input…")
    mi = pd.read_excel(DATA_DIR / "FCHK Model V3 - Model Input.xlsx")

    all_records: list[dict] = []
    seen_pairs: set[tuple] = set()  # (playerId, ws_name, ws_team) — avoid dups

    for comp, country, ws_file in LEAGUE_MAP:
        ws_path = WYSCOUT_DIR / ws_file
        if not ws_path.exists():
            print(f"  SKIP (file missing): {ws_file}")
            continue

        mask = mi["impect_competition_name"].eq(comp)
        if country:
            mask &= mi["country_nation"].eq(country)
        imp_slice = mi.loc[mask].copy()

        if imp_slice.empty:
            print(f"  SKIP (no IMPECT data): {comp} / {country}")
            continue

        print(f"  Matching {len(imp_slice):>4} IMPECT players → {ws_file}  [{comp} / {country}]")
        ws_df = pd.read_excel(ws_path)

        records = match_league(imp_slice, ws_df, ws_file)

        for r in records:
            key = (r["IMPECT_PlayerID"], r["Wyscout_Name"], r["Wyscout_Team"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                all_records.append(r)

    link_df = pd.DataFrame(all_records)

    # Summary
    total = len(link_df)
    high   = (link_df["MatchConfidence"] == "HIGH").sum()
    medium = (link_df["MatchConfidence"] == "MEDIUM").sum()
    low    = (link_df["MatchConfidence"] == "LOW").sum()
    none   = (link_df["MatchConfidence"] == "NONE").sum()

    print(f"\nTotal IMPECT entries : {total:,}")
    print(f"  HIGH confidence    : {high:,}  ({100*high/total:.1f}%)")
    print(f"  MEDIUM confidence  : {medium:,}  ({100*medium/total:.1f}%)")
    print(f"  LOW confidence     : {low:,}  ({100*low/total:.1f}%)")
    print(f"  No match           : {none:,}  ({100*none/total:.1f}%)")

    link_df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}  ({OUT_FILE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
