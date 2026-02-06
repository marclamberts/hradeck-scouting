import time
import random
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

TOKEN_URL = "https://login.impect.com/auth/realms/production/protocol/openid-connect/token"
BASE_API_URL = "https://api.impect.com"

TIMEOUT_SECONDS = 30
MAX_RETRIES = 8
BACKOFF_BASE = 0.75
BACKOFF_CAP = 30.0


def unwrap_data(obj):
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj


def get_access_token(session: requests.Session, username: str, password: str) -> str:
    payload = {
        "client_id": "api",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = session.post(TOKEN_URL, data=payload, headers=headers, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()["access_token"]


def get_with_retry(session: requests.Session, url: str, token: str):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    for attempt in range(MAX_RETRIES + 1):
        r = session.get(url, headers=headers, timeout=TIMEOUT_SECONDS)

        if r.status_code < 400:
            return r.json()

        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt))
            wait += random.uniform(0, 0.35 * wait)
            if attempt >= MAX_RETRIES:
                r.raise_for_status()
            time.sleep(wait)
            continue

        r.raise_for_status()

    raise RuntimeError("Retry loop exited unexpectedly")


def fetch_kpi_defs(session, token, language="en"):
    url = f"{BASE_API_URL}/v5/customerapi/kpis?language={language}"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_iteration_players(session, iteration_id, token):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/players"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_iteration_squads(session, iteration_id, token):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/squads"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_player_kpis_for_squad(session, iteration_id, squad_id, token):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/squads/{squad_id}/player-kpis"
    return unwrap_data(get_with_retry(session, url, token))


def extract_long_rows(iteration_id, squad_id, squad_name, items):
    out = []
    for item in items or []:
        matches = item.get("matches") or 0
        player_obj = item.get("player") if isinstance(item, dict) else None
        p = player_obj if isinstance(player_obj, dict) else item
        player_id = (p or {}).get("id") or item.get("playerId")

        for kv in (item.get("kpis") or []):
            out.append({
                "iterationId": iteration_id,
                "squadId": squad_id,
                "squadName": squad_name,
                "playerId": player_id,
                "matches": matches,
                "kpiId": kv.get("kpiId"),
                "value_raw": kv.get("value"),
            })
    return out


def build_iteration_wide_df(
    username: str,
    password: str,
    iteration_id: int,
    language: str = "en",
    max_workers: int = 3,
    base_sleep: float = 0.2,
):
    """
    Returns (wide_df, players_df, kpi_lookup_df)
    wide_df contains KPI columns renamed with labels + (id)
    """
    session = requests.Session()
    token = get_access_token(session, username, password)

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

    players = fetch_iteration_players(session, iteration_id, token)
    players_df = pd.json_normalize(players if isinstance(players, list) else [])
    if "id" in players_df.columns:
        players_df = players_df.rename(columns={"id": "playerId"})
    players_df["playerId"] = pd.to_numeric(players_df["playerId"], errors="coerce")

    squads = fetch_iteration_squads(session, iteration_id, token)
    squads_lookup = [{"squadId": s.get("id"), "squadName": s.get("name")} for s in (squads or [])]

    long_rows = []

    def worker(squad):
        sid = squad["squadId"]
        sname = squad["squadName"]
        if sid is None:
            return []
        if base_sleep:
            time.sleep(base_sleep + random.uniform(0, base_sleep))
        items = fetch_player_kpis_for_squad(session, iteration_id, int(sid), token)
        if not isinstance(items, list):
            return []
        return extract_long_rows(iteration_id, sid, sname, items)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, s) for s in squads_lookup]
        for fut in as_completed(futures):
            long_rows.extend(fut.result())

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        raise RuntimeError("No KPI rows returned")

    # decimal comma -> dot
    s = (
        long_df["value_raw"]
        .astype("string")
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    long_df["value"] = pd.to_numeric(s, errors="coerce")

    long_df["playerId"] = pd.to_numeric(long_df["playerId"], errors="coerce")
    long_df["kpiId"] = pd.to_numeric(long_df["kpiId"], errors="coerce")
    long_df["matches"] = pd.to_numeric(long_df["matches"], errors="coerce").fillna(0)

    long_df = long_df.dropna(subset=["playerId", "kpiId"])

    wide = (
        long_df.pivot(
            index=["iterationId", "squadId", "squadName", "playerId", "matches"],
            columns="kpiId",
            values="value"
        )
        .reset_index()
    )
    wide.columns.name = None

    # rename kpis after pivot
    wide = wide.rename(columns={k: v for k, v in rename_map.items() if k in wide.columns})

    # join player identity (names/positions/etc)
    wide = wide.merge(players_df, how="left", on="playerId")

    return wide, players_df, kpi_lookup
