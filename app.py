# app.py — IMPECT Live (friendly one-page) + Radars
# ================================================
# ✅ One page
# ✅ Load IMPECT API iteration (no CSV)
# ✅ User-friendly filters + player picker
# ✅ Radar charts (percentile-based) + KPI table
# ✅ Handles duplicate KPI rows safely (groupby + pivot_table)
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
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="IMPECT Live Scouting", page_icon="⚽", layout="wide")

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
# API helpers
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


def parse_decimal_comma(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


# -------------------------
# Load iteration (cached)
# -------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_impect_iteration(
    username: str,
    password: str,
    iteration_id: int,
    language: str = "en",
    max_workers: int = 3,
    base_sleep: float = 0.2,
):
    session = requests.Session()
    token = get_access_token(session, username, password)

    # KPI defs (id → label)
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

    # KPIs per squad (threaded)
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

    # Types + parse values
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

    # Rename KPI columns after pivot
    wide_df = wide_df.rename(columns={k: v for k, v in rename_map.items() if k in wide_df.columns})

    # Join player identity
    wide_df = wide_df.merge(players_df, how="left", on="playerId")

    # Add a friendly display name
    for c in ["commonname", "firstname", "lastname"]:
        if c not in wide_df.columns:
            wide_df[c] = ""
    wide_df["displayName"] = wide_df["commonname"].fillna("").astype(str).str.strip()
    fallback = (wide_df["firstname"].fillna("").astype(str).str.strip() + " " + wide_df["lastname"].fillna("").astype(str).str.strip()).str.strip()
    wide_df["displayName"] = np.where(wide_df["displayName"] == "", fallback, wide_df["displayName"])
    wide_df["displayName"] = wide_df["displayName"].fillna("").astype(str)

    return wide_df, players_df, squads_df, kpi_lookup


# -------------------------
# Radar helpers
# -------------------------
def compute_percentile(series: pd.Series, value: float, invert: bool = False) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return np.nan
    rank = (s < value).sum() + 0.5 * (s == value).sum()
    pct = rank / len(s) * 100
    return 100 - pct if invert else pct


def guess_invert(colname: str) -> bool:
    # heuristics: "lower is better"
    s = (colname or "").lower()
    bad = ["loss", "lost", "conced", "error", "own goal", "foul", "yellow", "red", "unsuccess", "unsuccessful", "turnover"]
    return any(k in s for k in bad)


def build_radar(player_row: pd.Series, pool_df: pd.DataFrame, kpi_cols: list[str]):
    labels = []
    pcts = []
    raw = []

    for c in kpi_cols:
        v = pd.to_numeric(player_row.get(c, np.nan), errors="coerce")
        inv = guess_invert(c)
        pct = compute_percentile(pool_df[c], v, invert=inv)
        labels.append(c[:26] + ("…" if len(c) > 26 else ""))
        pcts.append(0 if pd.isna(pct) else float(pct))
        raw.append(v)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=pcts,
            theta=labels,
            fill="toself",
            name="Percentile",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], showgrid=True)),
        showlegend=False,
        height=520,
        margin=dict(t=30, b=20, l=40, r=40),
    )

    table = pd.DataFrame(
        {
            "KPI": kpi_cols,
            "Value": raw,
            "Percentile": pcts,
        }
    ).sort_values("Percentile", ascending=False)

    return fig, table


# -------------------------
# UI
# -------------------------
st.title("⚽ IMPECT Live Scouting")
st.caption("Load an iteration from IMPECT API, pick a player, and generate radar charts + tables.")

# Top controls
with st.container():
    c1, c2, c3, c4, c5 = st.columns([2.2, 2.2, 1.1, 1.2, 1.3])

    default_user = st.secrets.get("IMPECT_USERNAME", "")
    default_pass = st.secrets.get("IMPECT_PASSWORD", "")

    with c1:
        username = st.text_input("Username", value=default_user, placeholder="IMPECT username/email")
    with c2:
        password = st.text_input("Password", value=default_pass, type="password", placeholder="IMPECT password")
    with c3:
        iteration_id = st.number_input("Iteration", min_value=1, value=1421, step=1)
    with c4:
        language = st.selectbox("KPI lang", ["en", "nl", "de", "fr"], index=0)
    with c5:
        load_btn = st.button("🚀 Load iteration", width="stretch")

st.divider()

# Advanced throttling
with st.expander("⚙️ Advanced (throttle / 429 control)", expanded=False):
    max_workers = st.slider("Workers", 1, 6, 3)
    base_sleep = st.slider("Throttle", 0.0, 1.0, 0.2, 0.05)
    clear = st.button("Clear cached load", width="stretch")
    if clear:
        load_impect_iteration.clear()
        for k in ["wide_df", "players_df", "squads_df", "kpi_lookup"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Cleared cached data.")

# Load
if load_btn:
    if not username or not password:
        st.error("Please provide username & password (or set them in secrets.toml).")
    else:
        try:
            with st.spinner("Loading IMPECT iteration..."):
                wide_df, players_df, squads_df, kpi_lookup = load_impect_iteration(
                    username=username,
                    password=password,
                    iteration_id=int(iteration_id),
                    language=language,
                    max_workers=int(max_workers) if "max_workers" in locals() else 3,
                    base_sleep=float(base_sleep) if "base_sleep" in locals() else 0.2,
                )
            st.session_state["wide_df"] = wide_df
            st.session_state["players_df"] = players_df
            st.session_state["squads_df"] = squads_df
            st.session_state["kpi_lookup"] = kpi_lookup
            st.success(f"Loaded ✅  {wide_df.shape[0]:,} rows × {wide_df.shape[1]:,} columns")
        except Exception as e:
            st.error(f"IMPECT load failed: {e}")

if "wide_df" not in st.session_state:
    st.info("Load an iteration to start.")
    st.stop()

wide_df: pd.DataFrame = st.session_state["wide_df"]
kpi_lookup: pd.DataFrame = st.session_state["kpi_lookup"]

# Summary
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows (player+squad+matches)", f"{wide_df.shape[0]:,}")
m2.metric("Columns", f"{wide_df.shape[1]:,}")
m3.metric("Players (unique)", f"{wide_df['playerId'].nunique():,}" if "playerId" in wide_df.columns else "—")
m4.metric("Squads (unique)", f"{wide_df['squadId'].nunique():,}" if "squadId" in wide_df.columns else "—")

st.divider()

# Friendly filters + player picker
left, right = st.columns([1.1, 2.2])

with left:
    st.subheader("🔎 Find player")

    q = st.text_input("Search", placeholder="Type name…")

    squads = sorted([s for s in wide_df.get("squadName", pd.Series(dtype=str)).dropna().unique().tolist()])
    squad_sel = st.multiselect("Squad", squads, default=[])

    min_matches = st.number_input("Min matches", min_value=0, value=0, step=1)

    # Filter dataset for selection pool
    pool = wide_df.copy()

    if q:
        pool = pool[pool["displayName"].astype(str).str.lower().str.contains(q.strip().lower(), na=False)]

    if squad_sel and "squadName" in pool.columns:
        pool = pool[pool["squadName"].isin(squad_sel)]

    if "matches" in pool.columns:
        pool = pool[pd.to_numeric(pool["matches"], errors="coerce").fillna(0) >= float(min_matches)]

    st.caption(f"Matches: {pool.shape[0]:,} rows")

    # Player selection list (from filtered pool)
    if pool.empty:
        st.warning("No players match your filters.")
        st.stop()

    # Build selection options as "Name • Squad (matches)"
    tmp = pool.copy()
    tmp["__opt__"] = (
        tmp["displayName"].astype(str)
        + "  •  "
        + tmp.get("squadName", pd.Series([""] * len(tmp))).astype(str)
        + "  •  matches="
        + tmp.get("matches", pd.Series([0] * len(tmp))).astype(str)
    )

    # Prefer highest matches rows first for each player
    tmp = tmp.sort_values(["displayName", "matches"], ascending=[True, False])

    opt = st.selectbox("Select player row", options=tmp["__opt__"].tolist(), index=0)
    player_row = tmp[tmp["__opt__"] == opt].iloc[0]

    st.markdown("#### Player info")
    info_cols = ["displayName", "squadName", "positions", "matches", "playerId"]
    show_cols = [c for c in info_cols if c in tmp.columns]
    st.dataframe(pd.DataFrame([player_row[show_cols].to_dict()]), width="stretch")

with right:
    st.subheader("📊 Radar + KPI table")

    meta_cols = {
        "iterationId",
        "squadId",
        "squadName",
        "playerId",
        "matches",
        "commonname",
        "firstname",
        "lastname",
        "displayName",
        "positions",
    }
    numeric_cols = [c for c in wide_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(wide_df[c])]

    # Default pick: take first 12 numeric columns (you can customize by keyword in future)
    default = numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols[:]

    st.caption("Choose KPIs (6–12 recommended). Percentiles are computed vs the current filtered pool on the left.")
    chosen_kpis = st.multiselect("KPIs for radar", options=numeric_cols, default=default)

    if len(chosen_kpis) < 3:
        st.info("Pick at least 3 KPIs to render a radar.")
    else:
        # Radar pool = filtered pool from left (for benchmarking)
        radar_pool = pool
        # Ensure all chosen columns exist in pool
        missing = [c for c in chosen_kpis if c not in radar_pool.columns]
        if missing:
            st.error(f"Some KPIs are missing in the pool: {missing[:5]}")
        else:
            fig, table = build_radar(player_row, radar_pool, chosen_kpis)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(table, width="stretch", height=450)

            csv = table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download this radar KPI table (CSV)",
                data=csv,
                file_name=f"radar_{player_row.get('displayName','player')}.csv",
                mime="text/csv",
                width="stretch",
            )

st.divider()

# Data tables (friendly tabs)
tab1, tab2, tab3 = st.tabs(["📦 Wide table (filtered)", "📦 Wide table (full)", "🧾 KPI definitions"])

with tab1:
    st.subheader("Wide KPI table (filtered)")
    st.dataframe(pool, width="stretch", height=650)

with tab2:
    st.subheader("Wide KPI table (full)")
    st.dataframe(wide_df, width="stretch", height=650)

with tab3:
    st.subheader("KPI definitions")
    st.dataframe(kpi_lookup, width="stretch", height=650)
    qk = st.text_input("Search KPI", "")
    if qk:
        kk = kpi_lookup.copy()
        for c in ["kpiLabel", "kpiTechnicalName"]:
            if c not in kk.columns:
                kk[c] = ""
        mask = (
            kk["kpiLabel"].astype(str).str.lower().str.contains(qk.lower(), na=False)
            | kk["kpiTechnicalName"].astype(str).str.lower().str.contains(qk.lower(), na=False)
        )
        st.dataframe(kk[mask], width="stretch", height=450)
