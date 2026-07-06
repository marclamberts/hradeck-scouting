"""
Build a normalized SQLite database from the raw Wyscout league export files.

Reads every .xlsx under data/Wyscout DB/ (one wide export per league, or per
"Part" slice of a league) and splits the 115 raw columns into a relational
schema instead of one flat sheet per competition:

  leagues                 one row per source file (a league, or a Part-slice
                           of a league that Wyscout split across files)
  teams                    deduplicated club dimension
  players                  deduplicated player dimension (name + bio key)
  player_league_stints     one row per player-per-file: club, position(s),
                           age, market value, contract, minutes/matches played
  player_positions         stint x position bridge (Wyscout lists multiple
                           positions per player, e.g. "RCMF, RDMF, LCMF")
  stats_general            goals/xG/assists/xA/duels headline numbers
  stats_defensive          tackles, interceptions, aerials, blocks
  stats_discipline         fouls and cards
  stats_attacking          shooting and attacking-action output
  stats_crossing           crossing volume/accuracy by flank
  stats_carrying           dribbling, progressive runs, ball receipt
  stats_passing            passing volume/accuracy/direction/length
  stats_creation           chance creation (xA, key passes, zone entries)
  stats_goalkeeping        GK-specific shot-stopping and distribution
  stats_set_pieces         free kicks, corners, penalties

Every stats_* table is keyed 1:1 on stint_id (FK -> player_league_stints).
Splitting by category keeps each table narrow and lets you join only the
slice of data a given analysis needs, instead of loading all 115 columns.

Output: data/wyscout.db (SQLite)
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

WYSCOUT_DIR = Path("data/Wyscout DB")
OUT_DB = Path("data/wyscout.db")

# ── Column groupings ─────────────────────────────────────────────────────────
# Every one of the 115 raw columns is assigned to exactly one group below.

BIO_COLS = ["Player", "Birth country", "Passport country", "Foot", "Height", "Weight"]

STINT_COLS = [
    "Team", "Team within selected timeframe", "Position", "Age",
    "Market value", "Contract expires", "On loan",
    "Matches played", "Minutes played",
]

STAT_GROUPS: dict[str, list[str]] = {
    "stats_general": [
        "Goals", "xG", "Assists", "xA", "Duels per 90", "Duels won, %",
    ],
    "stats_defensive": [
        "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
        "Sliding tackles per 90", "PAdj Sliding tackles", "Shots blocked per 90",
        "Interceptions per 90", "PAdj Interceptions",
    ],
    "stats_discipline": [
        "Fouls per 90", "Yellow cards", "Yellow cards per 90",
        "Red cards", "Red cards per 90", "Fouls suffered per 90",
    ],
    "stats_attacking": [
        "Successful attacking actions per 90", "Goals per 90",
        "Non-penalty goals", "Non-penalty goals per 90", "xG per 90",
        "Head goals", "Head goals per 90", "Shots", "Shots per 90",
        "Shots on target, %", "Goal conversion, %", "Assists per 90",
    ],
    "stats_crossing": [
        "Crosses per 90", "Accurate crosses, %",
        "Crosses from left flank per 90", "Accurate crosses from left flank, %",
        "Crosses from right flank per 90", "Accurate crosses from right flank, %",
        "Crosses to goalie box per 90",
    ],
    "stats_carrying": [
        "Dribbles per 90", "Successful dribbles, %", "Offensive duels per 90",
        "Offensive duels won, %", "Touches in box per 90",
        "Progressive runs per 90", "Accelerations per 90",
        "Received passes per 90", "Received long passes per 90",
    ],
    "stats_passing": [
        "Passes per 90", "Accurate passes, %", "Forward passes per 90",
        "Accurate forward passes, %", "Back passes per 90",
        "Accurate back passes, %", "Lateral passes per 90",
        "Accurate lateral passes, %", "Short / medium passes per 90",
        "Accurate short / medium passes, %", "Long passes per 90",
        "Accurate long passes, %", "Average pass length, m",
        "Average long pass length, m",
    ],
    "stats_creation": [
        "xA per 90", "Shot assists per 90", "Second assists per 90",
        "Third assists per 90", "Smart passes per 90",
        "Accurate smart passes, %", "Key passes per 90",
        "Passes to final third per 90", "Accurate passes to final third, %",
        "Passes to penalty area per 90", "Accurate passes to penalty area, %",
        "Through passes per 90", "Accurate through passes, %",
        "Deep completions per 90", "Deep completed crosses per 90",
        "Progressive passes per 90", "Accurate progressive passes, %",
    ],
    "stats_goalkeeping": [
        "Conceded goals", "Conceded goals per 90", "Shots against",
        "Shots against per 90", "Clean sheets", "Save rate, %",
        "xG against", "xG against per 90", "Prevented goals",
        "Prevented goals per 90", "Back passes received as GK per 90",
        "Exits per 90", "Aerial duels per 90.1",
    ],
    "stats_set_pieces": [
        "Free kicks per 90", "Direct free kicks per 90",
        "Direct free kicks on target, %", "Corners per 90",
        "Penalties taken", "Penalty conversion, %",
    ],
}

# Part-suffix pattern, e.g. "Australia II - Part I.xlsx" -> ("Australia II", "Part I")
PART_RE = re.compile(r"^(.*?)\s*-\s*(Part [IVX]+)$")


def parse_league_name(stem: str) -> tuple[str, str | None]:
    m = PART_RE.match(stem)
    if m:
        return m.group(1).strip(), m.group(2)
    return stem, None


def _clean(v):
    if pd.isna(v):
        return None
    return v


def main() -> None:
    files = sorted(WYSCOUT_DIR.glob("*.xlsx"))
    print(f"Found {len(files)} Wyscout league files in {WYSCOUT_DIR}/")

    leagues_rows: list[dict] = []
    teams_by_name: dict[str, int] = {}
    players_by_key: dict[tuple, int] = {}
    stints_rows: list[dict] = []
    positions_rows: list[dict] = []
    stat_rows: dict[str, list[dict]] = {g: [] for g in STAT_GROUPS}

    next_team_id = 1
    next_player_id = 1
    next_stint_id = 1

    for i, xlsx in enumerate(files, start=1):
        league_id = i
        league_name, part_label = parse_league_name(xlsx.stem)
        leagues_rows.append({
            "league_id": league_id,
            "source_file": xlsx.name,
            "league_name": league_name,
            "part_label": part_label,
        })

        try:
            df = pd.read_excel(xlsx)
        except Exception as exc:
            print(f"  SKIP {xlsx.name}: {exc}")
            continue

        if "Player" not in df.columns:
            print(f"  SKIP {xlsx.name}: no 'Player' column")
            continue

        for _, row in df.iterrows():
            name = row.get("Player")
            if pd.isna(name) or not str(name).strip():
                continue

            # ── players dimension ──
            bio_key = (
                str(name).strip(),
                _clean(row.get("Birth country")),
                _clean(row.get("Foot")),
                _clean(row.get("Height")),
                _clean(row.get("Weight")),
            )
            player_id = players_by_key.get(bio_key)
            if player_id is None:
                player_id = next_player_id
                next_player_id += 1
                players_by_key[bio_key] = player_id

            # ── teams dimension ──
            team_name = row.get("Team")
            team_id = None
            if pd.notna(team_name) and str(team_name).strip():
                team_name = str(team_name).strip()
                team_id = teams_by_name.get(team_name)
                if team_id is None:
                    team_id = next_team_id
                    next_team_id += 1
                    teams_by_name[team_name] = team_id

            stint_id = next_stint_id
            next_stint_id += 1

            stints_rows.append({
                "stint_id": stint_id,
                "player_id": player_id,
                "team_id": team_id,
                "league_id": league_id,
                "team_within_timeframe": _clean(row.get("Team within selected timeframe")),
                "position": _clean(row.get("Position")),
                "age": _clean(row.get("Age")),
                "market_value": _clean(row.get("Market value")),
                "contract_expires": _clean(row.get("Contract expires")),
                "on_loan": _clean(row.get("On loan")),
                "matches_played": _clean(row.get("Matches played")),
                "minutes_played": _clean(row.get("Minutes played")),
            })

            # ── player <-> position bridge ──
            pos_raw = row.get("Position")
            if pd.notna(pos_raw) and str(pos_raw).strip():
                for rank, code in enumerate(str(pos_raw).split(","), start=1):
                    code = code.strip()
                    if code:
                        positions_rows.append({
                            "stint_id": stint_id,
                            "position_rank": rank,
                            "position_code": code,
                        })

            # ── stat category tables ──
            for group, cols in STAT_GROUPS.items():
                rec = {"stint_id": stint_id}
                for col in cols:
                    rec[col] = _clean(row.get(col))
                stat_rows[group].append(rec)

        print(f"  [{i:>3}/{len(files)}] {xlsx.name}: {len(df):,} rows")

    # ── players table (one row per unique bio key) ──
    players_rows = [
        {
            "player_id": pid,
            "name": key[0],
            "birth_country": key[1],
            "foot": key[2],
            "height": key[3],
            "weight": key[4],
        }
        for key, pid in players_by_key.items()
    ]
    teams_rows = [{"team_id": tid, "team_name": name} for name, tid in teams_by_name.items()]

    print(
        f"\nBuilt {len(players_rows):,} unique players, {len(teams_rows):,} unique teams, "
        f"{len(stints_rows):,} player-league stints"
    )

    OUT_DB.parent.mkdir(exist_ok=True)
    OUT_DB.unlink(missing_ok=True)
    conn = sqlite3.connect(OUT_DB)

    def col_defs(cols: list[str]) -> str:
        return ", ".join(f'"{c}" REAL' for c in cols)

    conn.executescript(f"""
    CREATE TABLE leagues (
        league_id INTEGER PRIMARY KEY,
        source_file TEXT NOT NULL,
        league_name TEXT NOT NULL,
        part_label TEXT
    );

    CREATE TABLE teams (
        team_id INTEGER PRIMARY KEY,
        team_name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE players (
        player_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        birth_country TEXT,
        foot TEXT,
        height REAL,
        weight REAL
    );

    CREATE TABLE player_league_stints (
        stint_id INTEGER PRIMARY KEY,
        player_id INTEGER NOT NULL REFERENCES players(player_id),
        team_id INTEGER REFERENCES teams(team_id),
        league_id INTEGER NOT NULL REFERENCES leagues(league_id),
        team_within_timeframe TEXT,
        position TEXT,
        age REAL,
        market_value REAL,
        contract_expires TEXT,
        on_loan TEXT,
        matches_played REAL,
        minutes_played REAL
    );

    CREATE TABLE player_positions (
        stint_id INTEGER NOT NULL REFERENCES player_league_stints(stint_id),
        position_rank INTEGER NOT NULL,
        position_code TEXT NOT NULL,
        PRIMARY KEY (stint_id, position_rank)
    );

    CREATE TABLE stats_general        (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_general'])});
    CREATE TABLE stats_defensive      (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_defensive'])});
    CREATE TABLE stats_discipline     (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_discipline'])});
    CREATE TABLE stats_attacking      (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_attacking'])});
    CREATE TABLE stats_crossing       (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_crossing'])});
    CREATE TABLE stats_carrying       (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_carrying'])});
    CREATE TABLE stats_passing        (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_passing'])});
    CREATE TABLE stats_creation       (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_creation'])});
    CREATE TABLE stats_goalkeeping    (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_goalkeeping'])});
    CREATE TABLE stats_set_pieces     (stint_id INTEGER PRIMARY KEY REFERENCES player_league_stints(stint_id), {col_defs(STAT_GROUPS['stats_set_pieces'])});

    CREATE INDEX idx_stints_player ON player_league_stints(player_id);
    CREATE INDEX idx_stints_team   ON player_league_stints(team_id);
    CREATE INDEX idx_stints_league ON player_league_stints(league_id);
    CREATE INDEX idx_positions_code ON player_positions(position_code);
    CREATE INDEX idx_players_name ON players(name);
    """)

    pd.DataFrame(leagues_rows).to_sql("leagues", conn, if_exists="append", index=False)
    pd.DataFrame(teams_rows).to_sql("teams", conn, if_exists="append", index=False)
    pd.DataFrame(players_rows).to_sql("players", conn, if_exists="append", index=False)
    pd.DataFrame(stints_rows).to_sql("player_league_stints", conn, if_exists="append", index=False)
    if positions_rows:
        pd.DataFrame(positions_rows).to_sql("player_positions", conn, if_exists="append", index=False)

    for group in STAT_GROUPS:
        pd.DataFrame(stat_rows[group]).to_sql(group, conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    print(f"\nDatabase written -> {OUT_DB}  ({OUT_DB.stat().st_size // 1024:,} KB)")
    print(f"Tables: leagues, teams, players, player_league_stints, player_positions, "
          f"{', '.join(STAT_GROUPS)}")


if __name__ == "__main__":
    main()
