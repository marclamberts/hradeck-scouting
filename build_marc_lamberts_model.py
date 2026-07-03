"""
build_marc_lamberts_model.py
──────────────────────────────
The Marc Lamberts Model — full consolidation of every scouting model in
this repo into one per-player dashboard.

This is deliberately NOT a new blended score. It imports the actual
compute functions from each existing model and runs them on ONE shared
loaded dataset, so every column is a direct, unmodified output of the
model that produced it — nothing here is re-derived or re-weighted.

Models combined (source script → column group)
-----------------------------------------------
  build_lamberts_total.py   → Lamberts Index block   (SQS Rank, Market Value
                               Rank, Lamberts Index, Tier, vs Hradec, Status)
  wyscout_model.py          → Wyscout Composite block (10 documented composite
                               scores: ScoringThreat … CompositeRecruitment)
  build_trajectory_model.py → Trajectory block        (Current Level, age-curve
                               projections, Trajectory Score/Rank, Dev Tier)
  build_style_clusters.py   → Style Cluster block      (unsupervised playing-
                               style archetype, Style Fit)
  build_player_profiles.py → Player Profile block     (Primary Role from The
                               Athletic's 18-role system, Anomaly Type/Score)
  scouting_model.py         → Set-Piece block          (SetPieceAnalyzer role +
                               composite score, works on any position)
  build_war_database.py    → WAR + Grades block       (WIDE ATTACKERS ONLY —
                               20-80 tool grades, OVR/FV/Scout, plus-stats)
  build_hockey_scouting.py → Hockey-Style block        (WIDE ATTACKERS ONLY —
                               WAR/90, Line, Corsi/Fenwick, zones, HERO chart)

WAR is computed once. build_hockey_scouting.py's compute_war() uses the
identical formula to build_war_database.py's compute_war_rates() by
construction (documented in the hockey script's own header) — showing it
twice would be noise, not two different models' opinions.

Position scope caveat: WAR + Grades and Hockey-Style were both built for
wide attackers only (WIDE_ATK_POS). Every other block covers all outfield
positions + GK. Those two blocks are blank outside that subset — this is
a scope limit of the original models, not a bug in this consolidation.

Output: reports/Marc_Lamberts_Model.xlsx
  README · All Players · GK/CB/FB/DM/CM/W/FW

Usage:
  python build_marc_lamberts_model.py
  python build_marc_lamberts_model.py --leagues "Czech II" Slovakia --min-minutes 700
  python build_marc_lamberts_model.py --output reports/My_Marc_Lamberts_Model.xlsx
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

warnings.filterwarnings("ignore")

import build_lamberts_total as lam
import wyscout_model as wsm
import build_trajectory_model as traj
import build_style_clusters as style
import build_player_profiles as prof
import build_war_database as war
import build_hockey_scouting as nhl
import scouting_model as sm

# ── Config ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
WYSCOUT_DIR = ROOT / "Wyscout Files"
OUT_DIR = ROOT / "reports"

SKIP_FILES = {"FCHK Model V3 - Loaded Leagues", "FCHK Model V3 - Model Input",
              "FCHK Model V3 - Player Scores", "FCHK Model V3 - Player Styles",
              "FCHK Model V3 - Recruitment Scores", "FCHK Model V3 - Smart Club Closeness",
              "FCHK Model V3 - Summary", "FCHK Model V3 Scores", "FCHK Scouting Report",
              "Leagues Overview", "Wyscout Anomaly Report", "Wyscout Full Scouting Report"}

DEFAULT_LEAGUES = None
DEFAULT_MIN_MINUTES = 500

C = {
    "navy":   "0D1B2A",
    "gold":   "C9A84C",
    "white":  "FFFFFF",
    "light":  "EBF5FB",
    # section header colors, one per source model
    "sec_id":     "424949",
    "sec_lam":    "154360",
    "sec_wyscout":"117A65",
    "sec_traj":   "6C3483",
    "sec_style":  "B7950B",
    "sec_prof":   "1E8449",
    "sec_setp":   "935116",
    "sec_war":    "922B21",
    "sec_hockey": "2E4053",
}


# ── Shared loading ───────────────────────────────────────────────────────────────

def load_leagues(leagues: list[str] | None, min_minutes: int) -> pd.DataFrame:
    if leagues is None:
        paths = sorted(p for p in WYSCOUT_DIR.glob("*.xlsx") if p.stem not in SKIP_FILES)
    else:
        paths = [WYSCOUT_DIR / f"{lg}.xlsx" for lg in leagues]

    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            print(f"  [warn] {path} not found — skipping")
            continue
        lg = path.stem
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"  [warn] Could not read {path.name}: {e}")
            continue
        df = df.copy()
        df["_League"] = lg
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No Wyscout files found in {WYSCOUT_DIR}")

    raw = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(frames)} league files, {len(raw)} player rows")

    mins_col = next((c for c in ["Minutes played", "MinutesPlayed", "Minutes"] if c in raw.columns), None)
    raw["_minutes"] = pd.to_numeric(raw[mins_col], errors="coerce").fillna(0) if mins_col else 0
    raw = raw[raw["_minutes"] >= min_minutes].copy()
    print(f"  → {len(raw)} players after {min_minutes}+ minute filter")
    return raw.reset_index(drop=True)


def add_all_position_groupings(df: pd.DataFrame) -> pd.DataFrame:
    """Every source model groups positions its own way — compute them all once."""
    df = df.copy()
    pos_col = next((c for c in ["Position", "Pos"] if c in df.columns), None)
    df["_pos1"] = df[pos_col].astype(str).str.split(",").str[0].str.strip() if pos_col else ""

    # Lamberts / Trajectory / Style Clusters grouping (GK/CB/FB/DM/CM/W/FW)
    df["_pos_group"] = df["_pos1"].map(lam.POS_MAP).fillna("Other")
    df["Full Position"] = df[pos_col].fillna("Unknown") if pos_col else "Unknown"

    # Wyscout composite model grouping (GK/CB/FB/DM/CM/AM/W/ST)
    df["PositionGroup"] = df["_pos1"].map(wsm.WYSCOUT_POSITION_MAP).fillna("Other")

    # Player Profiles grouping (Central attacker / Wide attacker / ... / Goalkeeper)
    df["PositionFamily"] = df["_pos1"].map(prof.POSITION_FAMILY).fillna("Other")
    df["PositionFamilyGroup"] = df["PositionFamily"].map(prof.FAMILY_TO_GROUP).fillna("Other")

    return df


# ── Full-population anomaly (Player Profiles engine, unfiltered) ───────────────

def compute_full_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Anomaly Type"] = ""
    df["Anomaly Score"] = np.nan

    for group, metrics in prof.GROUP_KEY_METRICS.items():
        mask = df["PositionFamilyGroup"] == group
        metrics = [m for m in metrics if m in df.columns]
        if not metrics or mask.sum() < 5:
            continue
        grp = df.loc[mask]
        X = grp[metrics].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
        mu, sig = X.mean(axis=0), X.std(axis=0)
        sig = np.where(sig == 0, 1e-9, sig)
        Z = (X - mu) / sig

        peak_z = Z.max(axis=1)
        mean_z = Z.mean(axis=1)
        breadth = (Z >= prof.Z_GROUP_THRESH).sum(axis=1)
        anomaly_score = 0.45 * np.clip(peak_z, 0, None) + 0.35 * breadth + 0.20 * np.clip(mean_z, 0, None)

        ages = pd.to_numeric(grp.get("Age"), errors="coerce").fillna(99).to_numpy()
        types = []
        for pz, br, age in zip(peak_z, breadth, ages):
            if br >= 5:
                types.append("Multi-dimensional")
            elif pz >= prof.Z_GROUP_THRESH * 1.6 and br <= 2:
                types.append("Specialist Elite")
            elif pz >= prof.Z_GROUP_THRESH and age <= 22:
                types.append("Age-adjusted Gem")
            elif br >= 3:
                types.append("Consistent Overperformer")
            else:
                types.append("Emerging Talent")

        df.loc[mask, "Anomaly Type"] = types
        df.loc[mask, "Anomaly Score"] = anomaly_score

    return df


# ── Wide-attacker-only models (WAR + Grades, Hockey-Style) ──────────────────────

def attach_wide_attacker_models(master: pd.DataFrame) -> pd.DataFrame:
    wide_mask = master["_pos1"].isin(war.WIDE_ATK_POS)
    wide_subset = master.loc[wide_mask].copy()
    if wide_subset.empty:
        return master

    orig_idx = wide_subset.index.to_numpy()
    team_col = "Team within selected timeframe" if "Team within selected timeframe" in wide_subset.columns else "Team"
    pool = wide_subset.reset_index(drop=True)
    pool["_orig_pos"] = np.arange(len(pool))
    pool = (pool.sort_values("_minutes", ascending=False)
                .drop_duplicates(subset=["Player", team_col], keep="first"))
    orig_idx = orig_idx[pool["_orig_pos"].to_numpy()]
    pool = pool.reset_index(drop=True)

    print(f"    Wide-attacker pool: {len(pool)} players (WAR + Grades, Hockey-Style)")

    grades = war.compute_scouting_grades(pool)

    war_s, war90_s, total_rate, _ = nhl.compute_war(pool)
    cf = nhl.compute_corsi_fenwick(pool)
    pts = nhl.compute_points(pool)
    zones = nhl.compute_zones(pool)
    hero = nhl.compute_hero(pool)
    gs = nhl.compute_game_score(pool)
    line = nhl.classify_line(war90_s)
    age_traj = pool.get("Age", pd.Series(np.nan, index=pool.index)).apply(nhl.age_trajectory)

    combined = pd.concat([
        grades[["G_Finishing", "G_Dribbling", "G_Athleticism", "G_Crossing", "G_Creativity",
                "G_Defending", "OVR", "FV", "Scout", "FV_Label",
                "xG_plus", "xA_plus", "xP_plus", "GCONV", "WAR_plus"]],
        cf[["CF%", "FF%", "xGF%", "PDO"]],
        pts[["P/G", "Primary%"]],
        zones[["OZone", "NZone", "DZone", "DomZone"]],
        hero,
    ], axis=1)
    combined["WAR"] = war_s.round(3)
    combined["WAR/90"] = war90_s.round(4)
    combined["GS/90"] = gs
    combined["Line"] = line
    combined["Age Trajectory (Hockey)"] = age_traj.values

    combined.index = orig_idx
    return master.join(combined, how="left")


# ── Style clusters (per-position, keeps master's original index) ──────────────

def attach_style_clusters(master: pd.DataFrame) -> pd.DataFrame:
    master["Style Cluster"] = np.nan
    master["Style Label"] = ""
    master["Style Fit"] = np.nan

    for pos, metrics in style.STYLE_METRICS.items():
        df_pos = master[master["_pos_group"] == pos]
        if df_pos.empty:
            continue
        fit = style.fit_style_clusters(df_pos, metrics)
        if fit is None:
            continue
        fit_pct = style.style_fit_percentile(fit)
        labels_txt = [style.cluster_label(fit["centers_z"][c], fit["metrics"]) for c in range(fit["k"])]

        master.loc[df_pos.index, "Style Cluster"] = fit["labels"]
        master.loc[df_pos.index, "Style Label"] = [labels_txt[c] for c in fit["labels"]]
        master.loc[df_pos.index, "Style Fit"] = np.round(fit_pct, 1)
        print(f"    {pos}: k={fit['k']}  silhouette={fit['silhouette']:.3f}")

    return master


# ── Set-piece model (global, any position) ──────────────────────────────────────

def attach_set_piece_model(master: pd.DataFrame) -> pd.DataFrame:
    analyzer = sm.SetPieceAnalyzer(threshold=1.5)
    enriched = analyzer.fit_transform(master)
    if "_sp_composite" not in enriched.columns:
        master["Set-Piece Role"] = ""
        master["Set-Piece Score"] = np.nan
        return master
    master["Set-Piece Role"] = enriched["_sp_primary_role"]
    master["Set-Piece Score"] = (enriched["_sp_composite"].rank(pct=True) * 100).round(1)
    return master


# ── Excel helpers ────────────────────────────────────────────────────────────────

def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _border() -> Border:
    thin = Side(style="thin", color="CCCCCC")
    return Border(left=thin, right=thin, top=thin, bottom=thin)


def _autofit(ws, header_row: int) -> None:
    for col_cells in ws.columns:
        try:
            max_len = max(
                len(str(col_cells[header_row - 1].value or "")),
                *(len(str(c.value or "")) for c in col_cells[header_row:header_row + 10]),
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 30)
        except Exception:
            pass


# Column → (section label, section color) so the header shows provenance
COLUMN_SECTIONS: list[tuple[str, str, list[str]]] = [
    ("IDENTITY", C["sec_id"], [
        "Player", "Team", "League", "Pos", "Full Position", "Age", "Contract",
        "Mkt Val (€)", "Minutes",
    ]),
    ("LAMBERTS INDEX  ·  build_lamberts_total.py", C["sec_lam"], [
        "SQS Rank", "Mkt Val Rank", "Lamberts Index", "Tier", "vs Hradec", "Status", "Model Val (€)",
    ]),
    ("WYSCOUT COMPOSITE  ·  wyscout_model.py", C["sec_wyscout"], [
        "Scoring Threat", "Creative Progression", "Defensive Disruption", "Pressing",
        "Ball Security", "Expected Threat", "ASA Goals Added", "Aerial (Wyscout)",
        "Set Piece (Wyscout)", "Composite Recruitment",
    ]),
    ("TRAJECTORY MODEL  ·  build_trajectory_model.py", C["sec_traj"], [
        "Current Level", "Predicted @ Age", "Vs Age Curve", "Peak Age", "Years to Peak",
        "Trend", "Proj +1Y", "Proj +3Y", "Proj +5Y", "Trajectory Score", "Trajectory Rank", "Dev Tier",
    ]),
    ("STYLE CLUSTER MODEL  ·  build_style_clusters.py", C["sec_style"], [
        "Style Cluster", "Style Label", "Style Fit",
    ]),
    ("PLAYER PROFILES  ·  build_player_profiles.py", C["sec_prof"], [
        "Primary Role", "Primary Role Score", "Anomaly Type", "Anomaly Score",
    ]),
    ("SET-PIECE MODEL  ·  scouting_model.py", C["sec_setp"], [
        "Set-Piece Role", "Set-Piece Score",
    ]),
    ("WAR + SCOUTING GRADES (wide attackers only)  ·  build_war_database.py", C["sec_war"], [
        "WAR", "OVR", "FV", "Scout", "FV_Label",
        "G_Finishing", "G_Dribbling", "G_Athleticism", "G_Crossing", "G_Creativity", "G_Defending",
        "xG_plus", "xA_plus", "xP_plus", "GCONV", "WAR_plus",
    ]),
    ("HOCKEY-STYLE SCOUTING (wide attackers only)  ·  build_hockey_scouting.py", C["sec_hockey"], [
        "WAR/90", "Line", "Age Trajectory (Hockey)", "GS/90",
        "CF%", "FF%", "xGF%", "PDO", "P/G", "Primary%",
        "OZone", "NZone", "DZone", "DomZone",
        "H_Shooting", "H_Playmaking", "H_Skating", "H_Physicality", "H_Defence",
    ]),
]

ALL_COLUMNS = [c for _, _, cols in COLUMN_SECTIONS for c in cols]


def write_master_sheet(ws, title: str, subtitle: str, df: pd.DataFrame) -> None:
    cols = [c for c in ALL_COLUMNS if c in df.columns]
    n = len(cols)

    ws.append([title])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(n, 10))
    ws["A1"].font = Font(bold=True, color=C["white"], size=13)
    ws["A1"].fill = _fill(C["navy"])
    ws.row_dimensions[1].height = 22

    ws.append([subtitle])
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(n, 10))
    ws["A2"].font = Font(italic=True, color=C["gold"], size=9)
    ws["A2"].fill = _fill(C["navy"])
    ws.row_dimensions[2].height = 16

    if df.empty:
        return

    # Section header row
    section_row = ws.max_row + 1
    col_ptr = 1
    for label, color, sec_cols in COLUMN_SECTIONS:
        present = [c for c in sec_cols if c in df.columns]
        if not present:
            continue
        start = col_ptr
        end = col_ptr + len(present) - 1
        ws.cell(section_row, start, label)
        if end > start:
            ws.merge_cells(start_row=section_row, start_column=start, end_row=section_row, end_column=end)
        for c in range(start, end + 1):
            cell = ws.cell(section_row, c)
            cell.font = Font(bold=True, color=C["white"], size=8)
            cell.fill = _fill(color)
            cell.alignment = Alignment(horizontal="center", vertical="center")
        col_ptr = end + 1
    ws.row_dimensions[section_row].height = 16

    # Column name row
    ws.append(cols)
    hdr_row = ws.max_row
    col_ptr = 1
    for label, color, sec_cols in COLUMN_SECTIONS:
        present = [c for c in sec_cols if c in df.columns]
        for _ in present:
            cell = ws.cell(hdr_row, col_ptr)
            cell.font = Font(bold=True, color=C["white"], size=9)
            cell.fill = _fill(color)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = _border()
            col_ptr += 1
    ws.row_dimensions[hdr_row].height = 30

    df_out = df[cols]
    for i, row_vals in enumerate(df_out.itertuples(index=False), start=1):
        ws.append(list(row_vals))
        data_row = ws.max_row
        bg = C["light"] if i % 2 == 0 else C["white"]
        for cell in ws[data_row]:
            cell.font = Font(size=8)
            cell.fill = _fill(bg)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = _border()

    ws.freeze_panes = f"A{hdr_row + 1}"
    _autofit(ws, hdr_row)


def build_readme(ws, leagues: list[str], total: int, min_minutes: int) -> None:
    ws.title = "README"
    ws.sheet_view.showGridLines = False

    ws.append(["THE MARC LAMBERTS MODEL"])
    ws.merge_cells("A1:D1")
    ws["A1"].font = Font(bold=True, color=C["white"], size=15)
    ws["A1"].fill = _fill(C["navy"])
    ws.row_dimensions[1].height = 28

    league_label = f"{len(leagues)} leagues" if len(leagues) > 5 else " + ".join(leagues)
    ws.append([f"Waltzing Analytics  ·  Every scouting model in this repo, one row per player  ·  "
               f"{league_label}  ·  {min_minutes}+ min  ·  {total:,} players"])
    ws.merge_cells("A2:D2")
    ws["A2"].font = Font(italic=True, color=C["gold"], size=10)
    ws["A2"].fill = _fill(C["navy"])
    ws.row_dimensions[2].height = 18

    ws.append([None])
    ws.append([None, "WHAT THIS IS"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    intro = [
        "This is NOT a new blended score. It runs the actual compute functions from every",
        "model already built in this repo on one shared loaded dataset, then lays every",
        "output side by side. Nothing here is re-derived or re-weighted — each column is a",
        "direct, unmodified readout of the model named in its section header.",
        "",
        "Column sections are color-coded by source model. Two things worth knowing:",
        "  • SQS Rank (Lamberts) and Current Level (Trajectory) start from the identical",
        "    weighted-metric blueprint, but SQS Rank applies an extra percentile-of-percentile",
        "    pass that Current Level doesn't — so they're highly correlated (r≈0.98) but not",
        "    numerically identical. Kept as separate columns since they feed two different",
        "    downstream models (value vs. age curve).",
        "  • WAR is computed once. Hockey-Style's WAR formula is identical to WAR Database's",
        "    by design (documented in the hockey script's own header) — showing it twice",
        "    would be noise, not two independent opinions.",
    ]
    for line in intro:
        ws.append([None, line])

    ws.append([None])
    ws.append([None, "MODELS COMBINED"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    ws.append([None, "Section", "Source script", "Scope"])
    for cell in ws[ws.max_row]:
        if cell.value:
            cell.font = Font(bold=True, color=C["white"])
            cell.fill = _fill(C["header"] if "header" in C else C["navy"])
            cell.alignment = Alignment(horizontal="left")

    rows = [
        ("Lamberts Index",       "build_lamberts_total.py",   "All positions"),
        ("Wyscout Composite",    "wyscout_model.py",          "All positions"),
        ("Trajectory Model",     "build_trajectory_model.py", "All positions"),
        ("Style Cluster Model",  "build_style_clusters.py",   "All positions"),
        ("Player Profiles",      "build_player_profiles.py",  "All outfield positions (18-role system, no GK roles)"),
        ("Set-Piece Model",      "scouting_model.py",         "All positions"),
        ("WAR + Scouting Grades","build_war_database.py",     "Wide attackers only (LW/RW/LWF/RWF/AMF/LAMF/RAMF/LWB/RWB)"),
        ("Hockey-Style Scouting","build_hockey_scouting.py",  "Wide attackers only (same subset as above)"),
    ]
    for section, script, scope in rows:
        ws.append([None, section, script, scope])
        ws[f"B{ws.max_row}"].font = Font(bold=True)

    ws.append([None])
    ws.append([None, "WORKBOOK STRUCTURE"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    ws.append([None, "All Players", f"Full {total:,}-player database, every column, every position"])
    ws.append([None, "GK / CB / FB / DM / CM / W / FW", "Same full column set, filtered to that position, sorted by Lamberts Index"])
    for _ in range(2):
        ws[f"B{ws.max_row - 1 if _ else ws.max_row}"].font = Font(bold=True, color=C["navy"])

    ws.append([None])
    ws.append([None, "WHERE TO GO DEEPER"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    deeper = [
        "This dashboard intentionally skips each model's own reference/methodology sheets to",
        "stay usable at this width. For the full picture on any one model:",
        "  • Trajectory Model  → reports/Trajectory_Model.xlsx (age curves, Breakout/Decline sheets)",
        "  • Style Cluster Model → reports/Style_Cluster_Model.xlsx (cluster overview, prototypes)",
        "  • WAR + Grades / Hockey-Style → reports/WAR_All_Players.xlsx, reports/Hockey_Style_Scouting.xlsx",
        "  • Lamberts Index → data/Lamberts_Index_Model_All_Leagues.xlsx",
    ]
    for line in deeper:
        ws.append([None, line])

    ws.column_dimensions["A"].width = 3
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 28
    ws.column_dimensions["D"].width = 60


# ── Main ─────────────────────────────────────────────────────────────────────────

def run(leagues: list[str] | None, min_minutes: int, output: Path) -> None:
    print(f"\n{'='*60}")
    print("  The Marc Lamberts Model — Full Consolidation Builder")
    print(f"  Leagues: {'ALL' if leagues is None else leagues}")
    print(f"  Min minutes: {min_minutes}")
    print(f"{'='*60}\n")

    print("Loading Wyscout files…")
    raw = load_leagues(leagues, min_minutes)
    raw = add_all_position_groupings(raw)
    raw = raw[raw["_pos_group"] != "Other"].copy()
    print(f"  → {len(raw)} players with known position")

    print("\n[1/8] Lamberts Index (build_lamberts_total.py)…")
    raw = lam.compute_sqs(raw)
    raw = lam.compute_mv_rank(raw)
    raw = lam.compute_lamberts(raw)
    raw = lam.compute_vs_hradec(raw)

    print("[2/8] Wyscout Composite (wyscout_model.py)…")
    raw = wsm.compute_wyscout_scores(raw)

    print("[3/8] Trajectory Model (build_trajectory_model.py)…")
    raw = traj.compute_current_level(raw)
    curves = traj.fit_all_curves(raw)
    raw = traj.apply_trajectory(raw, curves)

    print("[4/8] Style Cluster Model (build_style_clusters.py)…")
    raw = attach_style_clusters(raw)

    print("[5/8] Player Profiles (build_player_profiles.py)…")
    raw = prof.compute_role_scores(raw)
    raw = compute_full_anomaly(raw)

    print("[6/8] Set-Piece Model (scouting_model.py)…")
    raw = attach_set_piece_model(raw)

    print("[7-8/8] WAR + Grades and Hockey-Style (wide attackers only)…")
    raw = attach_wide_attacker_models(raw)

    print("\nAssembling master table…")
    contract_col = next((c for c in ["Contract expires", "ContractExpires"] if c in raw.columns), None)
    mv_col = next((c for c in ["Market value", "MarketValue"] if c in raw.columns), None)

    master = pd.DataFrame({
        "Player": raw.get("Player", ""),
        "Team": raw.get("Team", ""),
        "League": raw.get("_League", ""),
        "Pos": raw.get("_pos_group", ""),
        "Full Position": raw.get("Full Position", ""),
        "Age": pd.to_numeric(raw.get("Age"), errors="coerce"),
        "Contract": pd.to_datetime(raw.get(contract_col), errors="coerce").dt.strftime("%Y-%m-%d") if contract_col else "",
        "Mkt Val (€)": pd.to_numeric(raw.get(mv_col, 0), errors="coerce").fillna(0).astype(int) if mv_col else 0,
        "Minutes": raw.get("_minutes", 0).astype(int),
        "SQS Rank": raw["_sqs_rank"].round(2),
        "Mkt Val Rank": raw["_mv_rank"].round(2),
        "Lamberts Index": raw["_lamberts"].round(2),
        "Tier": raw["_tier"],
        "vs Hradec": raw["_vs_hradec"],
        "Status": raw["_status"],
        "Model Val (€)": raw.apply(lambda r: lam.model_value(float(r.get(mv_col, 0) or 0), float(r.get("_sqs_rank", 0) or 0)), axis=1) if mv_col else 0,
        "Scoring Threat": raw.get("ScoringThreatScore", np.nan).round(2),
        "Creative Progression": raw.get("CreativeProgressionScore", np.nan).round(2),
        "Defensive Disruption": raw.get("DefensiveDisruptionScore", np.nan).round(2),
        "Pressing": raw.get("PressingScore", np.nan).round(2),
        "Ball Security": raw.get("BallSecurityScore", np.nan).round(2),
        "Expected Threat": raw.get("ExpectedThreatScore", np.nan).round(2),
        "ASA Goals Added": raw.get("ASA_GoalsAddedScore", np.nan).round(2),
        "Aerial (Wyscout)": raw.get("AerialScore", np.nan).round(2),
        "Set Piece (Wyscout)": raw.get("SetPieceScore", np.nan).round(2),
        "Composite Recruitment": raw.get("CompositeRecruitmentScore", np.nan).round(2),
        "Current Level": raw["_level"].round(2),
        "Predicted @ Age": raw["_predicted"].round(2),
        "Vs Age Curve": raw["_residual"],
        "Peak Age": raw["_peak_age"],
        "Years to Peak": raw["_years_to_peak"],
        "Trend": raw["_trend"],
        "Proj +1Y": raw["_proj1"],
        "Proj +3Y": raw["_proj3"],
        "Proj +5Y": raw["_proj5"],
        "Trajectory Score": raw["_trajectory"],
        "Trajectory Rank": raw["_traj_rank"],
        "Dev Tier": raw["_dev_tier"],
        "Style Cluster": raw["Style Cluster"],
        "Style Label": raw["Style Label"],
        "Style Fit": raw["Style Fit"],
        "Primary Role": raw.get("PrimaryRole", ""),
        "Primary Role Score": raw.get("PrimaryRoleScore", np.nan),
        "Anomaly Type": raw.get("Anomaly Type", ""),
        "Anomaly Score": raw.get("Anomaly Score", np.nan).round(2) if "Anomaly Score" in raw else np.nan,
        "Set-Piece Role": raw.get("Set-Piece Role", ""),
        "Set-Piece Score": raw.get("Set-Piece Score", np.nan),
    })

    war_cols = ["WAR", "OVR", "FV", "Scout", "FV_Label", "G_Finishing", "G_Dribbling",
                "G_Athleticism", "G_Crossing", "G_Creativity", "G_Defending",
                "xG_plus", "xA_plus", "xP_plus", "GCONV", "WAR_plus",
                "WAR/90", "Line", "Age Trajectory (Hockey)", "GS/90",
                "CF%", "FF%", "xGF%", "PDO", "P/G", "Primary%",
                "OZone", "NZone", "DZone", "DomZone",
                "H_Shooting", "H_Playmaking", "H_Skating", "H_Physicality", "H_Defence"]
    for c in war_cols:
        if c in raw.columns:
            master[c] = raw[c].values

    master = master.sort_values("Lamberts Index", ascending=False).reset_index(drop=True)
    print(f"  → {len(master)} total players, {len(master.columns)} columns")

    print(f"\nWriting workbook → {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)

    print("  Writing README…")
    ws_readme = wb.create_sheet("README")
    league_list = leagues if leagues else sorted(master["League"].unique().tolist())
    build_readme(ws_readme, league_list, len(master), min_minutes)

    print("  Writing All Players…")
    ws_all = wb.create_sheet("All Players")
    write_master_sheet(ws_all, f"ALL PLAYERS — {len(master)} · {len(master.columns)} columns across 8 models",
                        "Sorted by Lamberts Index  ·  Color-coded by source model", master)

    pos_labels = {"GK": "GK — GOALKEEPER", "CB": "CB — CENTRE-BACK", "FB": "FB — FULL-BACK",
                  "DM": "DM — DEFENSIVE MID", "CM": "CM — CENTRAL MID", "W": "W — WINGER",
                  "FW": "FW — FORWARD"}
    for pos, label in pos_labels.items():
        grp = master[master["Pos"] == pos]
        print(f"  Writing {pos} ({len(grp)})…")
        ws_pos = wb.create_sheet(pos)
        write_master_sheet(ws_pos, f"{label} — {len(grp)} players", "Sorted by Lamberts Index", grp)

    wb.save(output)
    print(f"\n✓ Saved {output}\n")


def main():
    parser = argparse.ArgumentParser(description="Build the Marc Lamberts Model workbook")
    parser.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES,
                        help="League names (Wyscout Files stems). Omit for all leagues.")
    parser.add_argument("--min-minutes", type=int, default=DEFAULT_MIN_MINUTES)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "Marc_Lamberts_Model.xlsx")
    args = parser.parse_args()

    run(args.leagues, args.min_minutes, args.output)


if __name__ == "__main__":
    main()
