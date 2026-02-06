import time
import random
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

TOKEN_URL = "https://login.impect.com/auth/realms/production/protocol/openid-connect/token"
BASE_API_URL = "https://api.impect.com"

TIMEOUT_SECONDS = 30
MAX_WORKERS = 3
BASE_SLEEP_SECONDS = 0.2
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
            retry_after = r.headers.get("Retry-After")
            wait = None
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = None

            if wait is None:
                wait = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt))
                wait += random.uniform(0, 0.35 * wait)

            if attempt >= MAX_RETRIES:
                r.raise_for_status()

            time.sleep(wait)
            continue

        r.raise_for_status()

    raise RuntimeError("Retry loop exited unexpectedly")


def fetch_kpi_defs(session: requests.Session, token: str, language: str):
    url = f"{BASE_API_URL}/v5/customerapi/kpis?language={language}"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_iteration_squads(session: requests.Session, iteration_id: int, token: str):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/squads"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_iteration_players(session: requests.Session, iteration_id: int, token: str):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/players"
    return unwrap_data(get_with_retry(session, url, token))


def fetch_player_kpis_for_squad(session: requests.Session, iteration_id: int, squad_id: int, token: str):
    url = f"{BASE_API_URL}/v5/customerapi/iterations/{iteration_id}/squads/{squad_id}/player-kpis"
    return unwrap_data(get_with_retry(session, url, token))


def extract_long_rows(iteration_id: int, squad_id: int, squad_name: str, items: list):
    out = []
    for item in items:
        matches = item.get("matches")
        if matches is None or (isinstance(matches, str) and matches.strip() == ""):
            matches = 0

        player_obj = item.get("player") if isinstance(item, dict) else None
        p = player_obj if isinstance(player_obj, dict) else (item if isinstance(item, dict) else {})
        player_id = p.get("id") or item.get("playerId")

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


def export_iteration_to_csv(
    username: str,
    password: str,
    iteration_id: int,
    output_csv: str,
    kpi_language: str = "en",
) -> dict:
    """
    Exports iteration player KPIs to a wide CSV (like your script).
    Returns a small summary dict you can show in Streamlit.
    """
    session = requests.Session()
    t0 = time.time()

    token = get_access_token(session, username, password)

    # KPI defs
    kpi_defs = fetch_kpi_defs(session, token, kpi_language)
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
    expected_player_cols = [
        "id", "firstname", "lastname", "commonname", "birthdate", "birthplace",
        "leg", "countryIds", "gender", "idMappings", "positions"
    ]
    for c in expected_player_cols:
        if c not in players_df.columns:
            players_df[c] = None

    players_df = players_df[expected_player_cols].rename(columns={"id": "playerId"})
    players_df["playerId"] = pd.to_numeric(players_df["playerId"], errors="coerce")

    # Squads
    squads = fetch_iteration_squads(session, iteration_id, token)
    squads_lookup = [{"squadId": s.get("id"), "squadName": s.get("name")} for s in (squads or [])]

    # Fetch player-kpis per squad
    long_rows = []

    def worker(squad):
        sid = squad["squadId"]
        sname = squad["squadName"]
        if sid is None:
            return []
        if BASE_SLEEP_SECONDS:
            time.sleep(BASE_SLEEP_SECONDS + random.uniform(0, BASE_SLEEP_SECONDS))
        items = fetch_player_kpis_for_squad(session, iteration_id, int(sid), token)
        if not isinstance(items, list):
            return []
        return extract_long_rows(iteration_id, sid, sname, items)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(worker, s) for s in squads_lookup]
        for fut in as_completed(futures):
            long_rows.extend(fut.result())

    player_long = pd.DataFrame(long_rows)
    if player_long.empty:
        raise RuntimeError("No player KPI rows returned.")

    # Convert decimal comma
    player_long["playerId"] = pd.to_numeric(player_long["playerId"], errors="coerce")
    player_long["kpiId"] = pd.to_numeric(player_long["kpiId"], errors="coerce")
    player_long["matches"] = pd.to_numeric(player_long["matches"], errors="coerce").fillna(0)

    s = (
        player_long["value_raw"]
        .astype("string")
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    player_long["value"] = pd.to_numeric(s, errors="coerce")
    player_long = player_long.dropna(subset=["playerId", "kpiId"])

    player_long = player_long.drop_duplicates(
        subset=["iterationId", "squadId", "playerId", "matches", "kpiId"],
        keep="first"
    )

    wide = (
        player_long.pivot(
            index=["iterationId", "squadId", "squadName", "playerId", "matches"],
            columns="kpiId",
            values="value"
        )
        .reset_index()
    )
    wide.columns.name = None

    # Rename KPIs after pivot
    wide = wide.rename(columns={k: v for k, v in rename_map.items() if k in wide.columns})

    # Join identities (including positions)
    wide = wide.merge(players_df, how="left", on="playerId")

    wide.to_csv(output_csv, index=False, encoding="utf-8", float_format="%.6f")

    return {
        "players_rows": int(wide.shape[0]),
        "cols": int(wide.shape[1]),
        "seconds": round(time.time() - t0, 1),
        "output_csv": output_csv,
    }
