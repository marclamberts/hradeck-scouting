# app.py — IMPECT Live Scouting (API) + Profile Radars (like your CSV script)
# ==========================================================================
# ✅ One-page Streamlit app
# ✅ Pulls from IMPECT API (no CSV needed)
# ✅ Computes the SAME derived metrics + same profile templates you shared
# ✅ Auto-builds profiles ONLY when KPIs exist in the loaded iteration
# ✅ Benchmarks by position tokens (fallback to full pool if sample too small)
# ✅ Shows radar + KPI table in-app (no PNG export)
# ✅ Handles duplicate KPI rows (groupby + pivot_table)
#
# Recommended secrets:
# .streamlit/secrets.toml
#   IMPECT_USERNAME="you@example.com"
#   IMPECT_PASSWORD="your-rotated-password"

import time
import random
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="IMPECT Scout", page_icon="⚽", layout="wide")

# -------------------------
# IMPECT API config
# -------------------------
TOKEN_URL = "https://login.impect.com/auth/realms/production/protocol/openid-connect/token"
BASE_API_URL = "https://api.impect.com"

TIMEOUT_SECONDS = 30
MAX_RETRIES = 8
BACKOFF_BASE = 0.75
BACKOFF_CAP = 30.0

# -------------------------
# Profile/radar config
# -------------------------
BENCHMARK_BY_POSITION = True
MIN_POSITION_SAMPLE = 10
MIN_METRICS_PER_RADAR = 6
MAX_METRICS_PER_RADAR = 12

DEFAULT_PROFILES = ["Winger", "Striker", "Attacking Mid", "Midfielder", "Fullback", "Centerback", "Goalkeeper"]

POSITION_TO_PROFILES = {
    "CENTER_FORWARD": ["Striker"],
    "LEFT_WINGER": ["Winger"],
    "RIGHT_WINGER": ["Winger"],
    "LEFT_WINGBACK_DEFENDER": ["Fullback"],
    "RIGHT_WINGBACK_DEFENDER": ["Fullback"],
    "CENTER_BACK": ["Centerback"],
    "CENTRE_BACK": ["Centerback"],
    "CENTRAL_DEFENDER": ["Centerback"],
    "GOALKEEPER": ["Goalkeeper"],
    "ATTACKING_MIDFIELD": ["Attacking Mid"],
    "CENTRAL_MIDFIELD": ["Midfielder"],
    "DEFENSE_MIDFIELD": ["Midfielder"],
    "MIDFIELD": ["Midfielder"],
}

# Invert: profile -> [position tokens]
PROFILE_TO_POSITIONS = {}
for pos_key, prof_list in POSITION_TO_PROFILES.items():
    for p in prof_list:
        PROFILE_TO_POSITIONS.setdefault(p, []).append(pos_key)

# -------------------------
# Derived metrics (same as your script)
# -------------------------
DERIVED = {
    "Aerial Win %": {
        "num": ["Won Aerial Duels (96)"],
        "denom": ["Won Aerial Duels (96)", "Lost Aerial Duels (97)"],
        "scale": 100,
    },
    "Ground Duel Win %": {
        "num": ["Won Ground Duels (94)"],
        "denom": ["Won Ground Duels (94)", "Lost Ground Duels (95)"],
        "scale": 100,
    },
    "Pass Accuracy %": {
        "num": ["Successful Passes (90)"],
        "denom": ["Successful Passes (90)", "Unsuccessful Passes (91)"],
        "scale": 100,
    },
    # NOTE: Your CSV script had a placeholder here; we keep it but make it safe.
    "Shots on Target %": {
        "num": ["Total Shots (100)"],
        "denom": ["Total Shots (100)"],
        "scale": 100,
    },
    "Crosses": {
        "num": [
            "SUCCESSFUL_PASSES_BY_ACTION_LOW_CROSS (293)",
            "SUCCESSFUL_PASSES_BY_ACTION_HIGH_CROSS (294)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Cross Bypasses": {
        "num": [
            "BYPASSED_OPPONENTS_BY_ACTION_LOW_CROSS (110)",
            "BYPASSED_OPPONENTS_BY_ACTION_HIGH_CROSS (111)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Pass Bypasses": {
        "num": [
            "BYPASSED_OPPONENTS_BY_ACTION_LOW_PASS (106)",
            "BYPASSED_OPPONENTS_BY_ACTION_DIAGONAL_PASS (107)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Pass Def Bypasses": {
        "num": [
            "BYPASSED_DEFENDERS_BY_ACTION_LOW_PASS (167)",
            "BYPASSED_DEFENDERS_BY_ACTION_DIAGONAL_PASS (168)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Advanced Touches": {
        "num": [
            "OFFENSIVE_TOUCHES_IN_PITCH_POSITION_FINAL_THIRD (598)",
            "OFFENSIVE_TOUCHES_IN_PITCH_POSITION_OPPONENT_BOX (599)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Defensive Actions": {
        "num": [
            "DEFENSIVE_TOUCHES_BY_ACTION_INTERCEPTION (611)",
            "DEFENSIVE_TOUCHES_BY_ACTION_DUEL (612)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Final Third Bypasses": {
        "num": [
            "BYPASSED_OPPONENTS_FROM_PITCH_POSITION_FINAL_THIRD (145)",
            "BYPASSED_OPPONENTS_FROM_PITCH_POSITION_OPPONENT_BOX (146)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Final Third Def Bypasses": {
        "num": [
            "BYPASSED_DEFENDERS_FROM_PITCH_POSITION_FINAL_THIRD (206)",
            "BYPASSED_DEFENDERS_FROM_PITCH_POSITION_OPPONENT_BOX (207)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Deep Passes": {
        "num": [
            "SUCCESSFUL_PASSES_FROM_PITCH_POSITION_FINAL_THIRD (327)",
            "SUCCESSFUL_PASSES_FROM_PITCH_POSITION_OPPONENT_BOX (328)",
        ],
        "denom": None,
        "scale": 1,
    },
    "Deep Ball Losses": {
        "num": [
            "BALL_LOSS_REMOVED_TEAMMATES_FROM_PITCH_POSITION_FINAL_THIRD (682)",
            "BALL_LOSS_REMOVED_TEAMMATES_FROM_PITCH_POSITION_OPPONENT_BOX (683)",
        ],
        "denom": None,
        "scale": 1,
    },
}

# -------------------------
# Profile templates (same as your script)
# Format: (display_label, [resolver candidates...], invert_flag)
# candidates can be:
#   - int KPI id
#   - str exact column
#   - derived metric name (must match DERIVED key)
# -------------------------
PROFILE_TEMPLATES = {
    "Striker": [
        ("Goals", [28], False),
        ("Assists", [77], False),
        ("xG", [82], False),
        ("Non-Shot xG", [83], False),
        ("Total Shots", [100], False),
        ("Shots Off Target", [101], True),
        ("Touches in Box", [599], False),
        ("Advanced Touches", ["Advanced Touches"], False),
        ("Bypassed Opponents", [0], False),
        ("Aerial Win %", ["Aerial Win %"], False),
        ("Critical Ball Losses", [49], True),
        ("Deep Ball Losses", ["Deep Ball Losses"], True),
    ],
    "Winger": [
        ("Goals", [28], False),
        ("Assists", [77], False),
        ("xG", [82], False),
        ("Non-Shot xG", [83], False),
        ("Crosses", ["Crosses"], False),
        ("Cross Bypasses", ["Cross Bypasses"], False),
        ("Dribble Bypasses", [87], False),
        ("Defenders Beaten", [88], False),
        ("Touches in Box", [599], False),
        ("Final Third Bypasses", ["Final Third Bypasses"], False),
        ("Ball Losses", [22], True),
        ("Critical Ball Losses", [49], True),
    ],
    "Attacking Mid": [
        ("Assists", [77], False),
        ("xG", [82], False),
        ("Non-Shot xG", [83], False),
        ("Pass Bypasses", ["Pass Bypasses"], False),
        ("Pass Def Bypasses", ["Pass Def Bypasses"], False),
        ("Bypassed Opponents", [0], False),
        ("Bypassed Defenders", [2], False),
        ("Deep Passes", ["Deep Passes"], False),
        ("Advanced Touches", ["Advanced Touches"], False),
        ("Final Third Bypasses", ["Final Third Bypasses"], False),
        ("Ball Losses", [22], True),
        ("Critical Ball Losses", [49], True),
    ],
    "Midfielder": [
        ("Bypassed Opponents", [0], False),
        ("Bypassed Defenders", [2], False),
        ("Pass Bypasses", ["Pass Bypasses"], False),
        ("Pass Def Bypasses", ["Pass Def Bypasses"], False),
        ("Bypassed Midfielders", [29], False),
        ("Pass Accuracy %", ["Pass Accuracy %"], False),
        ("Aerial Win %", ["Aerial Win %"], False),
        ("Ground Duel Win %", ["Ground Duel Win %"], False),
        ("Defensive Actions", ["Defensive Actions"], False),
        ("Ball Losses", [22], True),
        ("Critical Ball Losses", [49], True),
    ],
    "Centerback": [
        ("Aerial Win %", ["Aerial Win %"], False),
        ("Ground Duel Win %", ["Ground Duel Win %"], False),
        ("Defensive Actions", ["Defensive Actions"], False),
        ("Bypassed Opponents", [0], False),
        ("Bypassed Defenders", [2], False),
        ("Pass Accuracy %", ["Pass Accuracy %"], False),
        ("Pass Def Bypasses", ["Pass Def Bypasses"], False),
        ("Clearance Bypasses", [112], False),
        ("Header Bypasses", [113], False),
        ("Ball Losses", [22], True),
        ("Critical Ball Losses", [49], True),
    ],
    "Fullback": [
        ("Crosses", ["Crosses"], False),
        ("Cross Bypasses", ["Cross Bypasses"], False),
        ("Bypassed Opponents", [0], False),
        ("Bypassed Defenders", [2], False),
        ("Aerial Win %", ["Aerial Win %"], False),
        ("Ground Duel Win %", ["Ground Duel Win %"], False),
        ("Defensive Actions", ["Defensive Actions"], False),
        ("Pass Accuracy %", ["Pass Accuracy %"], False),
        ("Pass Def Bypasses", ["Pass Def Bypasses"], False),
        ("Final Third Bypasses", ["Final Third Bypasses"], False),
        ("Ball Losses", [22], True),
    ],
    "Goalkeeper": [
        ("Pass Accuracy %", ["Pass Accuracy %"], False),
        ("Bypassed Opponents", [0], False),
        ("Bypassed Defenders", [2], False),
        ("Pass Bypasses", ["Pass Bypasses"], False),
        ("Pass Def Bypasses", ["Pass Def Bypasses"], False),
        ("Offensive Touches", [92], False),
        ("Clearance Bypasses", [112], False),
        ("Ball Losses", [22], True),
        ("Critical Ball Losses", [49], True),
        ("Own Goals", [38], True),
    ],
}


# -------------------------
# Name + position helpers
# -------------------------
def positions_column(df: pd.DataFrame):
    if "positions" in df.columns:
        return "positions"
    if "position" in df.columns:
        return "position"
    return None


def normalize_positions_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def profiles_for_player(pos_str: str, available_profiles: set[str]):
    if not pos_str:
        return [p for p in DEFAULT_PROFILES if p in available_profiles][:3]

    tokens = [t.strip().upper() for t in pos_str.replace(";", ",").split(",") if t.strip()]
    chosen = []
    for t in tokens:
        for key, plist in POSITION_TO_PROFILES.items():
            if key in t:
                for p in plist:
                    if p in available_profiles and p not in chosen:
                        chosen.append(p)

    if not chosen:
        chosen = [p for p in DEFAULT_PROFILES if p in available_profiles]
    return chosen[:3]


def filter_for_profile(df: pd.DataFrame, pos_col: str | None, profile_name: str):
    if not pos_col or profile_name not in PROFILE_TO_POSITIONS:
        return df
    tokens = PROFILE_TO_POSITIONS[profile_name]
    mask = pd.Series(False, index=df.index)
    for t in tokens:
        mask |= df[pos_col].astype(str).str.contains(t, case=False, na=False)
    out = df[mask]
    return out if len(out) >= MIN_POSITION_SAMPLE else df


# -------------------------
# KPI resolver helpers (same logic as your script)
# -------------------------
def kpi_id_from_col(colname: str):
    if not isinstance(colname, str):
        return None
    m = re.search(r"\((\d+)\)\s*$", colname.strip())
    return int(m.group(1)) if m else None


def resolve_by_kpi_id(kpi_id: int, columns):
    suffix = f"({kpi_id})"
    hits = [c for c in columns if isinstance(c, str) and c.strip().endswith(suffix)]
    return sorted(hits, key=len)[0] if hits else None


def resolve_any(candidates, columns):
    for cand in candidates:
        if isinstance(cand, int):
            col = resolve_by_kpi_id(cand, columns)
            if col:
                return col
        elif isinstance(cand, str):
            # derived metric exact name OR exact column name
            if cand in columns:
                return cand
            # if "Name (123)" parse the id
            cid = kpi_id_from_col(cand)
            if cid is not None:
                col = resolve_by_kpi_id(cid, columns)
                if col:
                    return col
    return None


def build_profiles_from_df(df: pd.DataFrame):
    profiles = {}
    cols = list(df.columns)
    for prof_name, metrics in PROFILE_TEMPLATES.items():
        built = []
        for label, candidates, invert in metrics:
            real = resolve_any(candidates, cols)
            if real is None:
                continue
            built.append((label, real, invert))
        if len(built) >= MIN_METRICS_PER_RADAR:
            profiles[prof_name] = built[:MAX_METRICS_PER_RADAR]
    return profiles


# -------------------------
# Percentile + normalization (same logic as your script)
# -------------------------
def safe_divide(num, denom, scale=1.0):
    num_f = pd.to_numeric(num, errors="coerce").astype(float)
    den_f = pd.to_numeric(denom, errors="coerce").astype(float)
    out = np.full(len(num_f), np.nan, dtype=float)
    np.divide(num_f.to_numpy(), den_f.to_numpy(), out=out, where=(den_f.to_numpy() > 0))
    return out * float(scale)


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ensure numeric for any base KPI columns used by derived
    needed = []
    for spec in DERIVED.values():
        needed += spec["num"]
        if spec["denom"] is not None:
            needed += spec["denom"]
    for c in set(needed):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for name, spec in DERIVED.items():
        if any(c not in out.columns for c in spec["num"]):
            out[name] = np.nan
            continue
        num = out[spec["num"]].sum(axis=1)
        if spec["denom"] is None:
            out[name] = pd.to_numeric(num, errors="coerce").astype(float) * float(spec["scale"])
            continue
        if any(c not in out.columns for c in spec["denom"]):
            out[name] = np.nan
            continue
        denom = out[spec["denom"]].sum(axis=1)
        out[name] = safe_divide(num, denom, scale=spec["scale"])
    return out


def percentile_rank(series: pd.Series, value, invert: bool):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return np.nan
    rank = (s < value).sum() + 0.5 * (s == value).sum()
    pct = rank / len(s) * 100
    return 100 - pct if invert else pct


def axis_range(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (0.0, 1.0)
    lo, hi = float(s.quantile(0.05)), float(s.quantile(0.95))
    if hi == lo:
        lo, hi = float(s.min()), float(s.max())
    if hi == lo:
        hi = lo + 1.0
    return lo, hi


def norm01(value, lo, hi, invert=False):
    if pd.isna(value) or hi == lo:
        return np.nan
    x = np.clip((float(value) - lo) / (hi - lo), 0, 1)
    return 1 - x if invert else float(x)


# -------------------------
# API core
# -------------------------
def parse_decimal_comma(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def get_with_retry(session: requests.Session, url: str, token: str):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    for attempt in range(MAX_RETRIES + 1):
        r = session.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
        if r.status_code < 400:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt))
            wait += random.uniform(0, 0.35 * wait)
            if attempt >= MAX_RETRIES:
                r.raise_for_status()
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError("Retry loop exited unexpectedly")


def get_access_token(session: requests.Session, username: str, password: str) -> str:
    payload = {"client_id": "api", "grant_type": "password", "username": username, "password": password}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = session.post(TOKEN_URL, data=payload, headers=headers, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()["access_token"]


def unwrap_data(obj):
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj


def fetch_kpi_defs(session: requests.Session, token: str, language: str):
    url = f"{BASE_API_URL}/v5/customerapi/kpis?language={language}"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_iteration_players(session: requests.Session, iteration_id: int, token: str):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/players"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_iteration_squads(session: requests.Session, iteration_id: int, token: str):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/squads"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_player_kpis_for_squad(session: requests.Session, iteration_id: int, squad_id: int, token: str):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/squads/{squad_id}/player-kpis"
    return unwrap_data(get_with_retry(session, url, token))


def extract_long_rows(iteration_id: int, squad_id: int, squad_name: str, items: list):
    out = []
    for item in items or []:
        matches = item.get("matches") or 0
        player_obj = item.get("player") if isinstance(item, dict) else None
        p = player_obj if isinstance(player_obj, dict) else item
        player_id = (p or {}).get("id") or item.get("playerId")

        for kv in (item.get("kpis") or []):
            out.append(
                dict(
                    iterationId=iteration_id,
                    squadId=squad_id,
                    squadName=squad_name,
                    playerId=player_id,
                    matches=matches,
                    kpiId=kv.get("kpiId"),
                    value_raw=kv.get("value"),
                )
            )
    return out


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_impect_iteration(username: str, password: str, iteration_id: int, language: str, max_workers: int, base_sleep: float):
    session = requests.Session()
    token = get_access_token(session, username, password)

    # KPI defs -> rename KPI columns
    kpi_defs = fetch_kpi_defs(session, token, language)
    kpi_df = pd.json_normalize(kpi_defs if isinstance(kpi_defs, list) else [])
    label_col = "details.label" if "details.label" in kpi_df.columns else "name"

    kpi_lookup = (
        kpi_df.rename(columns={"id": "kpiId", label_col: "kpiLabel", "name": "kpiTechnicalName"})[
            ["kpiId", "kpiLabel", "kpiTechnicalName"]
        ]
        .drop_duplicates()
    )
    kpi_lookup["kpiId"] = pd.to_numeric(kpi_lookup["kpiId"], errors="coerce")

    rename_map = dict(
        zip(
            kpi_lookup["kpiId"],
            (
                kpi_lookup["kpiLabel"]
                .fillna(kpi_lookup["kpiTechnicalName"])
                .fillna("kpi")
                + " (" + kpi_lookup["kpiId"].astype("Int64").astype(str) + ")"
            ),
        )
    )

    # Players
    players = fetch_iteration_players(session, iteration_id, token)
    players_df = pd.json_normalize(players if isinstance(players, list) else [])
    if "id" in players_df.columns:
        players_df = players_df.rename(columns={"id": "playerId"})
    players_df["playerId"] = pd.to_numeric(players_df["playerId"], errors="coerce")

    for c in ["commonname", "firstname", "lastname", "positions"]:
        if c not in players_df.columns:
            players_df[c] = ""

    # Squads
    squads = fetch_iteration_squads(session, iteration_id, token)
    squads_df = pd.DataFrame([{"squadId": s.get("id"), "squadName": s.get("name")} for s in (squads or [])])
    squads_df["squadId"] = pd.to_numeric(squads_df["squadId"], errors="coerce")
    squads_list = squads_df.dropna(subset=["squadId"]).to_dict("records")

    # KPIs per squad
    long_rows = []

    def worker(squad):
        sid = int(squad["squadId"])
        sname = str(squad.get("squadName", ""))
        if base_sleep:
            time.sleep(base_sleep + random.uniform(0, base_sleep))
        items = fetch_player_kpis_for_squad(session, iteration_id, sid, token)
        if not isinstance(items, list):
            return []
        return extract_long_rows(iteration_id, sid, sname, items)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, s) for s in squads_list]
        for fut in as_completed(futures):
            long_rows.extend(fut.result())

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        raise RuntimeError("No KPI rows returned from IMPECT API.")

    long_df["playerId"] = pd.to_numeric(long_df["playerId"], errors="coerce")
    long_df["kpiId"] = pd.to_numeric(long_df["kpiId"], errors="coerce")
    long_df["matches"] = pd.to_numeric(long_df["matches"], errors="coerce").fillna(0)
    long_df["value"] = parse_decimal_comma(long_df["value_raw"])
    long_df = long_df.dropna(subset=["playerId", "kpiId"])

    # ✅ Fix duplicates before pivot
    key_cols = ["iterationId", "squadId", "squadName", "playerId", "matches", "kpiId"]
    long_df = long_df.groupby(key_cols, as_index=False)["value"].mean()

    wide_df = (
        long_df.pivot_table(
            index=["iterationId", "squadId", "squadName", "playerId", "matches"],
            columns="kpiId",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide_df.columns.name = None
    wide_df = wide_df.rename(columns={k: v for k, v in rename_map.items() if k in wide_df.columns})

    # Join player identity onto wide
    wide_df = wide_df.merge(players_df, how="left", on="playerId")

    # Friendly display name
    cn = wide_df["commonname"].fillna("").astype(str).str.strip()
    fallback = (
        wide_df["firstname"].fillna("").astype(str).str.strip()
        + " "
        + wide_df["lastname"].fillna("").astype(str).str.strip()
    ).str.strip()
    wide_df["displayName"] = np.where(cn == "", fallback, cn)

    # Ensure KPI columns are numeric
    meta = {"iterationId", "squadId", "squadName", "playerId", "matches", "commonname", "firstname", "lastname", "displayName", "positions"}
    for c in wide_df.columns:
        if c in meta:
            continue
        wide_df[c] = pd.to_numeric(wide_df[c], errors="coerce")

    # Add derived metrics
    wide_df = compute_derived(wide_df)

    return wide_df, players_df, squads_df, kpi_lookup


# -------------------------
# Radar UI builders (Plotly)
# -------------------------
def make_radar_plot(df_comp: pd.DataFrame, row: pd.Series, profile_metrics: list[tuple[str, str, bool]]):
    labels = [m[0] for m in profile_metrics]
    cols = [m[1] for m in profile_metrics]
    invert_flags = [m[2] for m in profile_metrics]
    n = len(labels)

    player_vals = np.array([pd.to_numeric(row.get(c, np.nan), errors="coerce") for c in cols], dtype=float)
    league_vals = np.array([pd.to_numeric(df_comp[c], errors="coerce").mean() for c in cols], dtype=float)

    pcts = np.array([percentile_rank(df_comp[c], player_vals[i], invert_flags[i]) for i, c in enumerate(cols)], dtype=float)

    ranges = [axis_range(df_comp[c]) for c in cols]
    player_norm = np.array([norm01(player_vals[i], *ranges[i], invert_flags[i]) for i in range(n)], dtype=float)
    league_norm = np.array([norm01(league_vals[i], *ranges[i], invert_flags[i]) for i in range(n)], dtype=float)

    player_norm = np.nan_to_num(player_norm, nan=0.0)
    league_norm = np.nan_to_num(league_norm, nan=0.0)
    pcts_clean = np.nan_to_num(pcts, nan=0.0)

    # Close polygons
    theta = labels + [labels[0]]
    r_player = player_norm.tolist() + [player_norm[0]]
    r_league = league_norm.tolist() + [league_norm[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=r_league,
            theta=theta,
            name="Benchmark avg",
            fill="toself",
            opacity=0.25,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=r_player,
            theta=theta,
            name="Player",
            fill="toself",
            opacity=0.65,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], showticklabels=False, ticks=""),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        height=520,
        margin=dict(t=50, b=20, l=30, r=30),
    )

    # Table
    table = pd.DataFrame(
        {
            "Metric": labels,
            "Column": cols,
            "Value": player_vals,
            "Percentile": pcts_clean,
            "Lower is better": invert_flags,
        }
    ).sort_values("Percentile", ascending=False)

    return fig, table


# -------------------------
# UI
# -------------------------
st.title("⚽ IMPECT Scout (API) — Profile Radars")
st.caption("This mirrors your CSV radar logic, but uses the live IMPECT API.")

with st.expander("🔐 Connect & Load", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([2.2, 2.2, 1.1, 1.2, 1.3])

    default_user = st.secrets.get("IMPECT_USERNAME", "")
    default_pass = st.secrets.get("IMPECT_PASSWORD", "")

    with c1:
        username = st.text_input("Username", value=default_user)
    with c2:
        password = st.text_input("Password", value=default_pass, type="password")
    with c3:
        iteration_id = st.number_input("Iteration", min_value=1, value=1421, step=1)
    with c4:
        language = st.selectbox("KPI language", ["en", "nl", "de", "fr"], index=0)
    with c5:
        load_btn = st.button("🚀 Load", width="stretch")

    max_workers = st.slider("Workers", 1, 6, 3)
    base_sleep = st.slider("Throttle", 0.0, 1.0, 0.2, 0.05)

    clear = st.button("Clear cache", width="stretch")
    if clear:
        load_impect_iteration.clear()
        for k in ["wide_df", "kpi_lookup"]:
            st.session_state.pop(k, None)
        st.success("Cache cleared.")

if load_btn:
    if not username or not password:
        st.error("Please provide username/password (or set Streamlit secrets).")
    else:
        try:
            with st.spinner("Loading iteration from IMPECT..."):
                wide_df, players_df, squads_df, kpi_lookup = load_impect_iteration(
                    username=username,
                    password=password,
                    iteration_id=int(iteration_id),
                    language=language,
                    max_workers=int(max_workers),
                    base_sleep=float(base_sleep),
                )
            st.session_state["wide_df"] = wide_df
            st.session_state["kpi_lookup"] = kpi_lookup
            st.success(f"Loaded ✅ {wide_df.shape[0]:,} rows × {wide_df.shape[1]:,} columns")
        except Exception as e:
            st.error(f"IMPECT load failed: {e}")

if "wide_df" not in st.session_state:
    st.info("Load an iteration to start.")
    st.stop()

wide_df: pd.DataFrame = st.session_state["wide_df"]

# Build profiles that exist in this iteration
profiles = build_profiles_from_df(wide_df)
if not profiles:
    st.error("No profiles could be built from this iteration (KPIs missing).")
    st.stop()

pos_col = positions_column(wide_df)

# Sidebar-like left column
left, right = st.columns([1.15, 2.25])

with left:
    st.subheader("🔎 Player picker")

    q = st.text_input("Search name", placeholder="Type player…")
    squads = sorted([s for s in wide_df["squadName"].dropna().unique().tolist()]) if "squadName" in wide_df.columns else []
    squad_sel = st.multiselect("Squad", squads, default=[])

    min_matches = st.number_input("Min matches", min_value=0, value=0, step=1)

    df_pool = wide_df.copy()
    if q:
        df_pool = df_pool[df_pool["displayName"].astype(str).str.lower().str.contains(q.strip().lower(), na=False)]
    if squad_sel and "squadName" in df_pool.columns:
        df_pool = df_pool[df_pool["squadName"].isin(squad_sel)]
    if "matches" in df_pool.columns:
        df_pool = df_pool[pd.to_numeric(df_pool["matches"], errors="coerce").fillna(0) >= float(min_matches)]

    st.caption(f"{df_pool.shape[0]:,} rows in selection pool")

    if df_pool.empty:
        st.warning("No rows match your filters.")
        st.stop()

    df_pick = df_pool.sort_values(["displayName", "matches"], ascending=[True, False]).copy()
    df_pick["__opt__"] = (
        df_pick["displayName"].astype(str)
        + "  •  "
        + df_pick.get("squadName", pd.Series([""] * len(df_pick))).astype(str)
        + "  •  matches="
        + df_pick.get("matches", pd.Series([0] * len(df_pick))).astype(str)
    )

    opt = st.selectbox("Select player row", df_pick["__opt__"].tolist(), index=0)
    player_row = df_pick[df_pick["__opt__"] == opt].iloc[0]

    st.markdown("#### Player")
    info_cols = [c for c in ["displayName", "squadName", "positions", "matches", "playerId"] if c in player_row.index]
    st.dataframe(pd.DataFrame([player_row[info_cols].to_dict()]), width="stretch")

    # Profile selection: auto suggestion like your script
    suggested = profiles_for_player(normalize_positions_str(player_row.get(pos_col, "")) if pos_col else "", set(profiles.keys()))
    prof_default = suggested[0] if suggested else sorted(list(profiles.keys()))[0]

    profile_choice = st.selectbox("Profile", options=sorted(list(profiles.keys())), index=sorted(list(profiles.keys())).index(prof_default))
    benchmark_by_pos = st.checkbox("Benchmark by position tokens", value=BENCHMARK_BY_POSITION)

with right:
    st.subheader("📊 Radar (like your PNGs, but live)")

    prof_metrics = profiles.get(profile_choice, [])
    if len(prof_metrics) < MIN_METRICS_PER_RADAR:
        st.warning("Not enough metrics available for this profile in this iteration.")
        st.stop()

    # Benchmark pool selection (like your filter_for_profile)
    if benchmark_by_pos:
        df_comp = filter_for_profile(df_pool, pos_col, profile_choice)
    else:
        df_comp = df_pool

    fig, table = make_radar_plot(df_comp, player_row, prof_metrics)
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### KPI Table")
    st.dataframe(
        table[["Metric", "Value", "Percentile", "Lower is better"]].sort_values("Percentile", ascending=False),
        width="stretch",
        height=420,
    )

    # quick downloads
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download radar table (CSV)",
        data=csv,
        file_name=f"radar_{player_row.get('displayName','player')}_{profile_choice}.csv".replace(" ", "_"),
        mime="text/csv",
        width="stretch",
    )

st.divider()

tabs = st.tabs(["📦 Data (filtered pool)", "📦 Data (full wide)", "🧾 KPI definitions", "🧰 Profiles built"])
with tabs[0]:
    st.dataframe(df_pool, width="stretch", height=650)
with tabs[1]:
    st.dataframe(wide_df, width="stretch", height=650)
with tabs[2]:
    kpi_lookup = st.session_state.get("kpi_lookup")
    if isinstance(kpi_lookup, pd.DataFrame):
        st.dataframe(kpi_lookup, width="stretch", height=650)
    else:
        st.info("KPI definitions not available.")
with tabs[3]:
    prof_info = []
    for p, mets in profiles.items():
        prof_info.append({"Profile": p, "Metrics": len(mets), "Metric names": ", ".join([m[0] for m in mets])})
    st.dataframe(pd.DataFrame(prof_info).sort_values("Profile"), width="stretch", height=450)
