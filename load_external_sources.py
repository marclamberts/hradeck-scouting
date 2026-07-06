"""
Extend data/wyscout.db with the other scouting data sources living in data/,
so the database covers more than raw Wyscout counting stats:

  skillcorner_physical      GPS/tracking physical output   (SkillCorner.csv)
  impect_wyscout_link       precomputed IMPECT <-> Wyscout player identity link
  fchk_loaded_leagues       leagues covered by the FCHK Model V3 run
  fchk_player_scores        headline per-player model scores
  fchk_player_styles        player style/archetype classification
  fchk_recruitment_scores   full recruitment scoring model output (127 factors)
  fchk_smart_club_closeness club-fit similarity scores (player x candidate club)
  fchk_run_summary          key/value metadata about the FCHK model run

None of these sources ship a Wyscout player_id, so each player-level table
gets a best-effort `wyscout_player_id` / `wyscout_team_id`, resolved against
the `players` / `teams` tables from build_wyscout_database.py by matching
(first-initial, last name) plus fuzzy team-name similarity — the same
approach build_link_db.py uses for the IMPECT link. `match_confidence`
records how much to trust the resolved id; NONE means no id was assigned
and the original name/team text columns are the only source of truth.

Run build_wyscout_database.py first — this script requires data/wyscout.db
to already contain the players/teams tables.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "wyscout.db"

TEAM_THRESHOLD_HIGH = 75
TEAM_THRESHOLD_MEDIUM = 50


# ── Name normalisation / matching (shared with build_link_db.py's approach) ──

def norm(s: object) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def wyscout_key(name: object) -> tuple[str, str]:
    """Key for Wyscout 'F. Lastname' style names -> (initial, last_word)."""
    n = norm(name)
    n = re.sub(r"^[a-z]{1,3}\. ?", "", n)
    parts = n.split()
    if not parts:
        return ("", "")
    initial = norm(name)[0] if norm(name) else ""
    return (initial, parts[-1])


def full_name_key(name: object) -> tuple[str, str]:
    """Key for 'Firstname ... Lastname' style names -> (initial, last_word)."""
    n = norm(name)
    parts = n.split()
    if not parts:
        return ("", "")
    return (parts[0][0], parts[-1])


class PlayerResolver:
    """Resolves external (name, team) pairs to wyscout player_id/team_id."""

    def __init__(self, conn: sqlite3.Connection):
        df = pd.read_sql(
            """
            SELECT p.player_id, p.name, s.team_id, t.team_name
            FROM players p
            LEFT JOIN player_league_stints s ON s.player_id = p.player_id
            LEFT JOIN teams t ON t.team_id = s.team_id
            """,
            conn,
        )
        self._by_key: dict[tuple, list[tuple]] = {}
        for row in df.itertuples(index=False):
            key = wyscout_key(row.name)
            self._by_key.setdefault(key, []).append((row.player_id, row.team_id, row.team_name))

    def resolve(self, name: object, team: object, key_fn=full_name_key) -> tuple[int | None, int | None, str]:
        key = key_fn(name)
        candidates = self._by_key.get(key)
        if not candidates:
            return None, None, "NONE"

        team_n = norm(team)
        if not team_n:
            if len({c[0] for c in candidates}) == 1:
                return candidates[0][0], candidates[0][1], "MEDIUM"
            return None, None, "LOW"

        cand_teams = [norm(c[2]) for c in candidates]
        result = process.extractOne(team_n, cand_teams, scorer=fuzz.token_set_ratio)
        if result is None:
            return None, None, "LOW"
        _, score, idx = result
        player_id, team_id, _ = candidates[idx]

        if score >= TEAM_THRESHOLD_HIGH:
            return player_id, team_id, "HIGH"
        if score >= TEAM_THRESHOLD_MEDIUM:
            return player_id, team_id, "MEDIUM"
        return None, None, "LOW"


def quote_cols(cols: list[str]) -> list[str]:
    return [f'"{c}"' for c in cols]


RESOLVED_COL_TYPES = {
    "wyscout_player_id": "INTEGER REFERENCES players(player_id)",
    "wyscout_team_id": "INTEGER REFERENCES teams(team_id)",
    "match_confidence": "TEXT",
}


def load_dataframe_table(conn, table: str, df: pd.DataFrame) -> None:
    conn.execute(f'DROP TABLE IF EXISTS {table}')
    col_defs = ", ".join(
        f'"{c}" {RESOLVED_COL_TYPES.get(c, "TEXT")}' for c in df.columns
    )
    conn.execute(f'CREATE TABLE {table} ({col_defs})')
    df.to_sql(table, conn, if_exists="append", index=False)


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} not found — run build_wyscout_database.py first")

    conn = sqlite3.connect(DB_PATH)
    resolver = PlayerResolver(conn)

    # ── SkillCorner physical/tracking data ──
    sc = pd.read_csv(DATA_DIR / "SkillCorner.csv", sep=";")
    sc.columns = [c.strip('"') for c in sc.columns]
    for col in sc.columns:
        sc[col] = sc[col].astype(str).str.strip('"')
        sc[col] = sc[col].replace({"nan": None, "": None})
    res = sc.apply(lambda r: resolver.resolve(r["Player"], r["Team"]), axis=1)
    sc["wyscout_player_id"] = [r[0] for r in res]
    sc["wyscout_team_id"] = [r[1] for r in res]
    sc["match_confidence"] = [r[2] for r in res]
    load_dataframe_table(conn, "skillcorner_physical", sc)
    print(f"skillcorner_physical: {len(sc):,} rows "
          f"({(sc['match_confidence'] != 'NONE').sum():,} linked to a Wyscout player)")

    # ── IMPECT <-> Wyscout link (already computed by build_link_db.py) ──
    link = pd.read_csv(DATA_DIR / "IMPECT_Wyscout_Link.csv")
    res = link.apply(lambda r: resolver.resolve(r["Wyscout_Name"], r["Wyscout_Team"], key_fn=wyscout_key)
                      if pd.notna(r["Wyscout_Name"]) else (None, None, "NONE"), axis=1)
    link["wyscout_player_id"] = [r[0] for r in res]
    link["wyscout_team_id"] = [r[1] for r in res]
    load_dataframe_table(conn, "impect_wyscout_link", link)
    print(f"impect_wyscout_link: {len(link):,} rows")

    # ── FCHK Model V3 outputs ──
    fchk_sources = {
        "fchk_loaded_leagues": ("FCHK Model V3 - Loaded Leagues.xlsx", None, None),
        "fchk_player_scores": ("FCHK Model V3 - Player Scores.xlsx", "PlayerName", "TeamName"),
        "fchk_player_styles": ("FCHK Model V3 - Player Styles.xlsx", "PlayerName", "TeamName"),
        "fchk_recruitment_scores": ("FCHK Model V3 - Recruitment Scores.xlsx", "PlayerName", "TeamName"),
        "fchk_smart_club_closeness": ("FCHK Model V3 - Smart Club Closeness.xlsx", "PlayerName", "TeamName"),
        "fchk_run_summary": ("FCHK Model V3 - Summary.xlsx", None, None),
    }

    for table, (fname, name_col, team_col) in fchk_sources.items():
        df = pd.read_excel(DATA_DIR / fname)
        if name_col:
            res = df.apply(lambda r: resolver.resolve(r[name_col], r[team_col]), axis=1)
            df["wyscout_player_id"] = [r[0] for r in res]
            df["wyscout_team_id"] = [r[1] for r in res]
            df["match_confidence"] = [r[2] for r in res]
        load_dataframe_table(conn, table, df)
        linked = f", {(df['match_confidence'] != 'NONE').sum():,} linked" if name_col else ""
        print(f"{table}: {len(df):,} rows{linked}")

    conn.execute('CREATE INDEX IF NOT EXISTS idx_skillcorner_player ON skillcorner_physical(wyscout_player_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_implink_player ON impect_wyscout_link(wyscout_player_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_fchk_scores_player ON fchk_player_scores(wyscout_player_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_fchk_recruit_player ON fchk_recruitment_scores(wyscout_player_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_fchk_styles_player ON fchk_player_styles(wyscout_player_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_fchk_closeness_player ON fchk_smart_club_closeness(wyscout_player_id)')
    conn.commit()
    conn.close()

    print(f"\nUpdated {DB_PATH}  ({DB_PATH.stat().st_size // 1024:,} KB)")


if __name__ == "__main__":
    main()
