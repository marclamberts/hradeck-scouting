# app.py — IMPECT Live (simple one-page)
# =====================================
# ✅ One page
# ✅ Load IMPECT API (iteration) into memory
# ✅ Show data tables (players wide, KPI lookup, optional filters)
# ✅ Fixes duplicate KPI rows (groupby + pivot_table)
#
# SECRETS (recommended):
# .streamlit/secrets.toml
#   IMPECT_USERNAME="you@example.com"
#   IMPECT_PASSWORD="your-rotated-password"
#
# Run:
#   streamlit run app.py

import time
import random
import requests
import pandas as pd
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="IMPECT Live Loader", page_icon="⚽", layout="wide")

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
# Helpers
# -------------------------
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
                {
                    "iterationId": iteration_id,
                    "squadId": squad_id,
                    "squadName": squad_name,
                    "playerId": player_id,
                    "matches": matches,
                    "kpiId": kv.get("kpiId"),
                    "value_raw": kv.get("value"),
                }
            )
    return out


def parse_decimal_comma(series: pd.Series) -> pd.Series:
    # Handles "27,513527" -> 27.513527
    s = series.astype("string").str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_impect_iteration(
    username: str,
    password: str,
    iteration_id: int,
    language: str = "en",
    max_workers: int = 3,
    base_sleep: float = 0.2,
):
    """
    Returns:
      wide_df: 1 row per (iteration,squad,player,matches) with KPI columns
      players_df: player master data
      squads_df: squad list
      kpi_lookup: KPI definitions
      long_df: long-format KPI rows (optional debugging)
    """
    session = requests.Session()
    token = get_access_token(session, username, password)

    # KPI defs (rename KPI ids -> human labels + (id))
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

    # Squads
    squads = fetch_iteration_squads(session, iteration_id, token)
    squads_df = pd.DataFrame([{"squadId": s.get("id"), "squadName": s.get("name")} for s in (squads or [])])
    squads_df["squadId"] = pd.to_numeric(squads_df["squadId"], errors="coerce")

    # Player KPIs (per squad, threaded)
    squads_list = squads_df.dropna(subset=["squadId"]).to_dict("records")
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

    # Types + parse value
    long_df["playerId"] = pd.to_numeric(long_df["playerId"], errors="coerce")
    long_df["kpiId"] = pd.to_numeric(long_df["kpiId"], errors="coerce")
    long_df["matches"] = pd.to_numeric(long_df["matches"], errors="coerce").fillna(0)
    long_df["value"] = parse_decimal_comma(long_df["value_raw"])

    long_df = long_df.dropna(subset=["playerId", "kpiId"])

    # ✅ Fix duplicates BEFORE pivot
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

    # Rename KPI numeric columns into human labels (Label (id))
    wide_df = wide_df.rename(columns={k: v for k, v in rename_map.items() if k in wide_df.columns})

    # Join player identity
    wide_df = wide_df.merge(players_df, how="left", on="playerId")

    return wide_df, players_df, squads_df, kpi_lookup, long_df


def safe_contains(s: pd.Series, q: str) -> pd.Series:
    q = (q or "").strip().lower()
    if not q:
        return pd.Series([True] * len(s), index=s.index)
    return s.astype(str).str.lower().str.contains(q, na=False)


# -------------------------
# UI
# -------------------------
st.title("⚽ IMPECT Live Loader (API → Tables)")
st.caption("One-page app: load an iteration from IMPECT API, then explore tables.")

with st.expander("🔐 Connection", expanded=True):
    # Prefer secrets; allow overrides for debugging
    default_user = st.secrets.get("IMPECT_USERNAME", "")
    default_pass = st.secrets.get("IMPECT_PASSWORD", "")

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1:
        username = st.text_input("Username", value=default_user, placeholder="IMPECT username/email")
    with c2:
        password = st.text_input("Password", value=default_pass, type="password", placeholder="IMPECT password")
    with c3:
        iteration_id = st.number_input("Iteration ID", min_value=1, value=1421, step=1)
    with c4:
        language = st.selectbox("Language", ["en", "nl", "de", "fr"], index=0)

    c5, c6, c7 = st.columns([1, 1, 2])
    with c5:
        max_workers = st.slider("Workers", 1, 6, 3)
    with c6:
        base_sleep = st.slider("Throttle", 0.0, 1.0, 0.2, 0.05)
    with c7:
        st.write("")

    load_btn = st.button("🚀 Load from IMPECT API", width="stretch")

if load_btn:
    if not username or not password:
        st.error("Please provide username & password (or set them in secrets.toml).")
    else:
        try:
            with st.spinner("Loading IMPECT iteration..."):
                wide_df, players_df, squads_df, kpi_lookup, long_df = load_impect_iteration(
                    username=username,
                    password=password,
                    iteration_id=int(iteration_id),
                    language=language,
                    max_workers=int(max_workers),
                    base_sleep=float(base_sleep),
                )
            st.session_state["wide_df"] = wide_df
            st.session_state["players_df"] = players_df
            st.session_state["squads_df"] = squads_df
            st.session_state["kpi_lookup"] = kpi_lookup
            st.session_state["long_df"] = long_df
            st.success(f"Loaded ✅  wide: {wide_df.shape[0]} rows × {wide_df.shape[1]} cols")
        except Exception as e:
            st.error(f"IMPECT load failed: {e}")

# If loaded, show tables
if "wide_df" not in st.session_state:
    st.info("Load an iteration to view tables.")
    st.stop()

wide_df: pd.DataFrame = st.session_state["wide_df"]
players_df: pd.DataFrame = st.session_state["players_df"]
squads_df: pd.DataFrame = st.session_state["squads_df"]
kpi_lookup: pd.DataFrame = st.session_state["kpi_lookup"]
long_df: pd.DataFrame = st.session_state["long_df"]

# Summary cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Wide rows", f"{wide_df.shape[0]:,}")
with c2:
    st.metric("Wide columns", f"{wide_df.shape[1]:,}")
with c3:
    st.metric("Players", f"{players_df.shape[0]:,}")
with c4:
    st.metric("KPI defs", f"{kpi_lookup.shape[0]:,}")

st.divider()

# Filters
with st.expander("🔎 Filters (optional)", expanded=True):
    q_name = st.text_input("Search player name (commonname/firstname/lastname)", "")
    squad_filter = st.multiselect(
        "Squad",
        options=sorted([s for s in wide_df.get("squadName", pd.Series(dtype=str)).dropna().unique().tolist()]),
        default=[],
    )
    min_matches = st.number_input("Min matches", min_value=0, value=0, step=1)

filtered = wide_df.copy()

# Name search
name_cols = [c for c in ["commonname", "firstname", "lastname"] if c in filtered.columns]
if q_name and name_cols:
    mask = pd.Series(False, index=filtered.index)
    for c in name_cols:
        mask |= safe_contains(filtered[c], q_name)
    filtered = filtered[mask]

# Squad filter
if squad_filter and "squadName" in filtered.columns:
    filtered = filtered[filtered["squadName"].isin(squad_filter)]

# Matches filter
if "matches" in filtered.columns:
    filtered = filtered[pd.to_numeric(filtered["matches"], errors="coerce").fillna(0) >= float(min_matches)]

st.caption(f"Showing {filtered.shape[0]:,} rows after filters.")

# Tabs with tables
tab1, tab2, tab3, tab4 = st.tabs(["📦 Wide KPI Table", "🧑 Players", "🏟️ Squads", "🧾 KPI Definitions"])

with tab1:
    st.subheader("Wide KPI Table (1 row per squad+player+matches)")
    st.dataframe(filtered, width="stretch", height=600)

    # Quick download (optional)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered wide CSV", data=csv, file_name="impect_wide_filtered.csv", mime="text/csv", width="stretch")

with tab2:
    st.subheader("Players (from /iterations/{id}/players)")
    st.dataframe(players_df, width="stretch", height=600)

with tab3:
    st.subheader("Squads (from /iterations/{id}/squads)")
    st.dataframe(squads_df, width="stretch", height=400)

with tab4:
    st.subheader("KPI Definitions (from /kpis)")
    st.dataframe(kpi_lookup, width="stretch", height=600)

    # Quick KPI search
    q_kpi = st.text_input("Search KPI label/technical name", "")
    if q_kpi:
        k = kpi_lookup.copy()
        cols = [c for c in ["kpiLabel", "kpiTechnicalName"] if c in k.columns]
        if cols:
            m = pd.Series(False, index=k.index)
            for c in cols:
                m |= safe_contains(k[c], q_kpi)
            st.dataframe(k[m], width="stretch", height=400)

with st.expander("🛠️ Debug (long format rows)", expanded=False):
    st.caption("Long-format KPI rows (after de-dup aggregation happens before pivot).")
    st.dataframe(long_df.head(5000), width="stretch", height=500)
