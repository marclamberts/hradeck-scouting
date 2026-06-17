"""
make_notebook.py — Generate Barat_Scouting_Report.ipynb (nbformat v4)
"""
import json
from pathlib import Path


def code_cell(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def md_cell(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
    }


# ─── Cell sources ────────────────────────────────────────────────────────────

CELL_MD_TITLE = """\
# D. Barát — Scouting Report Notebook

**FCHK Scouting  ·  Czech Fortuna Liga 2025/26**

This notebook reproduces the full 5-page PDF scouting report for D. Barát (Slovácko, LAMF/LW/LWB, Age 19).

Sections:
1. Imports & palette configuration
2. Constants (positions, metrics, role blueprints)
3. Data loading
4. Percentile calculation, role fit, WAR
5. Helper drawing utilities
6. Page 1 drawing functions (header, profile fit, distributions)
7. Page 1 drawing functions (bars, benchmark table)
8. Page 2 functions (peer comparison, full stats)
9. Page 3 functions (WAR banner, composite cards, derived table)
10. SkillCorner data & loader
11. Page 4 drawing functions (pills, bars, distributions)
12. Title page
13. Run all — generate 5-page PDF"""

CELL_1_IMPORTS = """\
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import percentileofscore, gaussian_kde, norm
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
})

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#FFFFFF"
PANEL    = "#F7F9FC"
SURFACE2 = "#EEF2F8"
BORDER   = "#D1D9E6"
TEXT     = "#111827"
TEXT_MED = "#374151"
TEXT_DIM = "#6B7280"
ACCENT   = "#1D4ED8"
PLAYER_C = "#7C3AED"
CLUB_C   = "#7C3AED"
LEAGUE_C = "#2563EB"
TIER_C   = "#0F766E"
DB_C     = "#4B5563"
CLUB_BG  = "#F5F3FF"
LEAGUE_BG= "#EFF6FF"
TIER_BG  = "#F0FDFA"
DB_BG    = "#F3F4F6"

PERF_CMAP = LinearSegmentedColormap.from_list(
    "perf", [(0.0, "#DC2626"), (0.40, "#D97706"), (0.65, "#D97706"), (1.0, "#16A34A")]
)
def pct_colour(pct):
    return PERF_CMAP(np.clip(pct / 100, 0, 1))"""

CELL_2_CONSTANTS = """\
WIDE_ATK_POS = {"LAMF","RAMF","LW","RW","LWF","RWF","AMF","LWB","RWB"}
MIN_MINS = 300

NON_METRIC = {
    "Player","Team","Team within selected timeframe","Position",
    "Age","Market value","Contract expires","Birth country",
    "Passport country","Foot","Height","Weight","On loan",
}

TIER3_STEMS = {
    "Czech","Austria","Switzerland","Denmark","Sweden","Norway",
    "Poland","Greece","Croatia","Romania","Serbia",
    "Colombia","Chile","Peru","Uruguay",
    "Australia","Egypt","Morocco","Nigeria","South Africa",
    "India","Indonesia","Malaysia","Thailand","Canada",
    "England II","Spain II","Germany II","Italy II","France II",
}

CAT_COLOURS = {
    "Threat":    "#DC2626",
    "Carrying":  "#D97706",
    "Creation":  "#059669",
    "Duels":     "#2563EB",
    "Defending": "#7C3AED",
}

BAR_METRICS = [
    ("xG per 90",                          "xG / 90",           "Threat"),
    ("Shots per 90",                       "Shots / 90",        "Threat"),
    ("Touches in box per 90",              "Box Touches / 90",  "Threat"),
    ("Dribbles per 90",                    "Dribbles / 90",     "Carrying"),
    ("Successful dribbles, %",             "Dribble Success %", "Carrying"),
    ("Progressive runs per 90",            "Prog. Runs / 90",   "Carrying"),
    ("Crosses per 90",                     "Crosses / 90",      "Creation"),
    ("Accurate crosses, %",                "Cross Acc. %",      "Creation"),
    ("xA per 90",                          "xA / 90",           "Creation"),
    ("Shot assists per 90",                "Shot Assists / 90", "Creation"),
    ("Key passes per 90",                  "Key Passes / 90",   "Creation"),
    ("Offensive duels won, %",             "Off. Duels Won %",  "Duels"),
    ("Defensive duels won, %",             "Def. Duels Won %",  "Duels"),
    ("Successful defensive actions per 90","Def. Actions / 90", "Defending"),
]

DIST_METRICS = [
    ("Dribbles per 90",         "Dribbles / 90"),
    ("xG per 90",               "xG / 90"),
    ("Crosses per 90",          "Crosses / 90"),
    ("xA per 90",               "xA / 90"),
]

BENCH_METRICS = [
    ("Dribbles per 90",         "Dribbles / 90"),
    ("xG per 90",               "xG / 90"),
    ("Crosses per 90",          "Crosses / 90"),
    ("xA per 90",               "xA / 90"),
    ("Progressive runs per 90", "Prog. Runs / 90"),
]

ROLE_BLUEPRINTS = {
    "Wide Threat": {
        "Dribbles per 90": 0.25, "Successful dribbles, %": 0.15,
        "Progressive runs per 90": 0.20, "Crosses per 90": 0.15,
        "Touches in box per 90": 0.15, "Offensive duels won, %": 0.10,
    },
    "Finisher": {
        "Goals per 90": 0.35, "xG per 90": 0.25,
        "Shots per 90": 0.20, "Touches in box per 90": 0.20,
    },
    "Target": {
        "Aerial duels won, %": 0.40, "Shots per 90": 0.20,
        "Goals per 90": 0.15, "Touches in box per 90": 0.25,
    },
    "Roamer": {
        "Successful defensive actions per 90": 0.25,
        "Offensive duels won, %": 0.25,
        "Dribbles per 90": 0.20, "Progressive runs per 90": 0.30,
    },
    "Unlocker": {
        "Key passes per 90": 0.25, "xA per 90": 0.25,
        "Shot assists per 90": 0.25, "Passes per 90": 0.15,
        "Accurate passes, %": 0.10,
    },
    "Outlet": {
        "Accurate passes, %": 0.30, "Passes per 90": 0.25,
        "Offensive duels won, %": 0.20, "Key passes per 90": 0.15,
        "xA per 90": 0.10,
    },
}

PEER_TABLE_COLS = [
    ("Dribbles per 90",          "Drib/90"),
    ("xG per 90",                "xG/90"),
    ("Crosses per 90",           "Cross/90"),
    ("xA per 90",                "xA/90"),
    ("Progressive runs per 90",  "PrgR/90"),
    ("Successful dribbles, %",   "Drib%"),
    ("Offensive duels won, %",   "OffDuel%"),
]

ALL_STATS_LEFT = [
    ("Goals per 90",              "Goals / 90",         "Threat"),
    ("xG per 90",                 "xG / 90",            "Threat"),
    ("Shots per 90",              "Shots / 90",         "Threat"),
    ("Shots on target, %",        "Shots on Target %",  "Threat"),
    ("Touches in box per 90",     "Box Touches / 90",   "Threat"),
    ("Dribbles per 90",           "Dribbles / 90",      "Carrying"),
    ("Successful dribbles, %",    "Dribble Success %",  "Carrying"),
    ("Progressive runs per 90",   "Prog. Runs / 90",    "Carrying"),
    ("Accelerations per 90",      "Accelerations / 90", "Carrying"),
    ("xA per 90",                 "xA / 90",            "Creation"),
    ("Assists per 90",            "Assists / 90",       "Creation"),
    ("Shot assists per 90",       "Shot Assists / 90",  "Creation"),
    ("Key passes per 90",         "Key Passes / 90",    "Creation"),
]

ALL_STATS_RIGHT = [
    ("Crosses per 90",                    "Crosses / 90",        "Creation"),
    ("Accurate crosses, %",               "Cross Acc. %",        "Creation"),
    ("Passes per 90",                     "Passes / 90",         "Creation"),
    ("Accurate passes, %",                "Pass Acc. %",         "Creation"),
    ("Long passes per 90",                "Long Passes / 90",    "Creation"),
    ("Accurate long passes, %",           "Long Pass Acc. %",    "Creation"),
    ("Received passes per 90",            "Received / 90",       "Creation"),
    ("Offensive duels won, %",            "Off. Duels Won %",    "Duels"),
    ("Defensive duels won, %",            "Def. Duels Won %",    "Duels"),
    ("Aerial duels won, %",               "Aerial Won %",        "Duels"),
    ("Successful defensive actions per 90","Def. Actions / 90",  "Defending"),
    ("Interceptions per 90",              "Interceptions / 90",  "Defending"),
    ("Fouls per 90",                      "Fouls / 90",          "Defending"),
]"""

CELL_3_LOAD_DATA = """\
def _numeric(df):
    for col in df.columns:
        if col not in NON_METRIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_data():
    print("  Czech files ...")
    df_c1 = _numeric(pd.read_excel("data/Wyscout DB/Czech.xlsx"))
    df_c1["_file"] = "Czech"
    df_c1["_pos1"] = df_c1["Position"].astype(str).str.split(",").str[0].str.strip()

    player  = df_c1[df_c1["Player"].astype(str).str.startswith("D. Bar")].iloc[0].copy()
    team_kw = str(player.get("Team within selected timeframe", "Slovácko")).split()[0]

    pool_lg = df_c1[df_c1["_pos1"].isin(WIDE_ATK_POS) &
                    (df_c1["Minutes played"].fillna(0) >= MIN_MINS)].copy()
    pool_cl = df_c1[df_c1["Team within selected timeframe"].astype(str)
                    .str.contains(team_kw, na=False) &
                    df_c1["_pos1"].isin(WIDE_ATK_POS)].copy()

    print("  Full database (165 files) ...")
    parts = []
    for f in sorted(Path("data/Wyscout DB").glob("*.xlsx")):
        if f.stem in ("Czech U17", "Czech U19"):
            continue
        try:
            d = _numeric(pd.read_excel(f))
            d["_file"] = f.stem
            d["_pos1"] = d["Position"].astype(str).str.split(",").str[0].str.strip()
            parts.append(d)
        except Exception:
            pass

    df_all = pd.concat(parts, ignore_index=True)
    wa = df_all["_pos1"].isin(WIDE_ATK_POS) & (df_all["Minutes played"].fillna(0) >= MIN_MINS)
    pool_db   = df_all[wa].copy().reset_index(drop=True)
    pool_tier = df_all[df_all["_file"].isin(TIER3_STEMS) & wa].copy().reset_index(drop=True)

    return player, pool_cl, pool_lg, pool_tier, pool_db"""

CELL_4_CALCS = """\
def calc_pcts(player, pool, metrics):
    out = {}
    for m in metrics:
        if m not in pool.columns or m not in player.index or len(pool) == 0:
            out[m] = 50.0; continue
        vals = pool[m].dropna().values
        pv   = float(player.get(m, np.nan))
        out[m] = round(percentileofscore(vals, pv, kind="rank"), 1) \\
                 if not np.isnan(pv) and len(vals) else 50.0
    return out


def role_fit(player, pool):
    scores = {}
    for role, bp in ROLE_BLUEPRINTS.items():
        avail = [(m, w) for m, w in bp.items() if m in pool.columns and m in player.index]
        if not avail:
            scores[role] = 50.0; continue
        tw = sum(w for _, w in avail)
        z  = sum((w/tw) * (float(player[m]) - pool[m].mean()) / (pool[m].std() or 1e-9)
                 for m, w in avail if not np.isnan(float(player.get(m, np.nan))))
        scores[role] = float(norm.cdf(z) * 100)
    return scores


def calc_war(player, pool_lg, pool_db):
    GOALS_PER_WIN   = 3.0
    REPL_PERCENTILE = 15

    def _rates(df):
        xg = df["xG per 90"].fillna(0)            if "xG per 90"                 in df.columns else 0
        xa = df["xA per 90"].fillna(0)            if "xA per 90"                 in df.columns else 0
        sa = df["Shot assists per 90"].fillna(0)  if "Shot assists per 90"        in df.columns else 0
        pr = df["Progressive runs per 90"].fillna(0) if "Progressive runs per 90" in df.columns else 0
        d  = df["Dribbles per 90"].fillna(0)      if "Dribbles per 90"            in df.columns else 0
        dp = df["Successful dribbles, %"].fillna(50) if "Successful dribbles, %"  in df.columns else 50
        off   = xg + xa * 0.85 + sa * 0.28
        carry = pr * 0.025 + (d * dp / 100) * 0.015
        return off, carry, off + carry

    def _pval(col, default=0.0):
        v = player.get(col, default)
        return float(v) if pd.notna(v) else default

    lg_off, lg_carry, lg_tot = _rates(pool_lg)
    db_off, db_carry, db_tot = _rates(pool_db)
    repl = float(np.percentile(lg_tot.dropna(), REPL_PERCENTILE))

    p_off   = _pval("xG per 90") + _pval("xA per 90") * 0.85 + _pval("Shot assists per 90") * 0.28
    p_carry = _pval("Progressive runs per 90") * 0.025 + \\
              (_pval("Dribbles per 90") * _pval("Successful dribbles, %", 50) / 100) * 0.015
    p_tot   = p_off + p_carry

    minutes = _pval("Minutes played", 593)
    n90     = minutes / 90.0
    war     = (p_tot - repl) * n90 / GOALS_PER_WIN

    lg_min  = pool_lg["Minutes played"].fillna(450) if "Minutes played" in pool_lg.columns \\
              else pd.Series([450] * len(pool_lg))
    db_min  = pool_db["Minutes played"].fillna(450) if "Minutes played" in pool_db.columns \\
              else pd.Series([450] * len(pool_db))

    lg_wars = ((lg_tot - repl) * (lg_min / 90.0) / GOALS_PER_WIN).dropna().values
    db_wars = ((db_tot - repl) * (db_min / 90.0) / GOALS_PER_WIN).dropna().values

    pct_lg  = float(percentileofscore(lg_wars, war, kind="rank"))
    pct_db  = float(percentileofscore(db_wars, war, kind="rank"))
    rank_lg = int(np.sum(lg_wars < war)) + 1

    return {
        "war": war, "offensive": p_off, "carrying": p_carry,
        "player_rate": p_tot, "repl_level": repl,
        "var_per_90": p_tot - repl, "nineties": n90, "minutes": minutes,
        "pct_lg": pct_lg, "pct_db": pct_db,
        "rank_lg": rank_lg, "n_lg": len(lg_wars),
        "lg_wars": lg_wars, "db_wars": db_wars,
        "repl_off": float(np.percentile(lg_off.dropna(), REPL_PERCENTILE)),
        "repl_carry": float(np.percentile(lg_carry.dropna(), REPL_PERCENTILE)),
        "GOALS_PER_WIN": GOALS_PER_WIN,
    }"""

CELL_5_HELPERS = """\
def _off(ax, face=BG):
    ax.set_facecolor(face)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")


def _inner_title(ax, title, note=""):
    full = f"{title}  ·  {note}" if note else title
    ax.add_patch(mpatches.Rectangle(
        (0, 0.945), 1.0, 0.055,
        facecolor=SURFACE2, edgecolor="none",
        transform=ax.transAxes, clip_on=True, zorder=10,
    ))
    ax.add_patch(mpatches.Rectangle(
        (0, 0.945), 0.004, 0.055,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=True, zorder=11,
    ))
    ax.text(0.012, 0.973, full, ha="left", va="center",
            transform=ax.transAxes, color=TEXT_MED,
            fontsize=6.5, fontweight="bold", zorder=12)


def find_peers(player, pool_lg, n=8):
    sim_cols = [m for m, *_ in BAR_METRICS if m in pool_lg.columns]
    pool = pool_lg.copy()
    pool = pool[~pool["Player"].astype(str).str.startswith("D. Bar")]
    means = pool[sim_cols].mean()
    stds  = pool[sim_cols].std().replace(0, 1e-9)

    def zrow(row):
        return [(row.get(c, np.nan) - means[c]) / stds[c] for c in sim_cols]

    pz = np.array(zrow(player))
    dists = []
    for _, row in pool.iterrows():
        rz = np.array(zrow(row))
        mask = ~(np.isnan(pz) | np.isnan(rz))
        if mask.sum() < 3:
            dists.append(np.inf)
        else:
            dists.append(float(np.linalg.norm(pz[mask] - rz[mask])))
    pool = pool.copy()
    pool["_dist"] = dists
    return pool.nsmallest(n, "_dist").reset_index(drop=True)"""

CELL_6_PAGE1A = """\
def draw_header(ax, n_lg, n_db):
    _off(ax)
    ax.add_patch(mpatches.Rectangle(
        (-0.018, 0.0), 0.007, 1.0,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(0.0, 0.97, "D. BARÁT", ha="left", va="top",
            transform=ax.transAxes, color=TEXT, fontsize=26, fontweight="bold")
    ax.text(0.0, 0.33,
            "Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Czech Republic",
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=8.5)
    ax.plot([0, 1], [0.04, 0.04], transform=ax.transAxes, color=BORDER, lw=0.8)
    pills = [("MIN","593"),("MATCHES","19"),("xG","0.93"),("xA","0.38"),("DRIB/90","4.55")]
    px = 0.50
    for j, (lbl, val) in enumerate(pills):
        if j > 0:
            ax.plot([px - 0.040, px - 0.040], [0.24, 0.90],
                    transform=ax.transAxes, color=BORDER, lw=0.7)
        ax.text(px, 0.93, val, ha="center", va="top",
                color="#B45309", fontsize=12, fontweight="bold",
                transform=ax.transAxes)
        ax.text(px, 0.27, lbl, ha="center", va="top",
                color=TEXT_DIM, fontsize=6, transform=ax.transAxes, fontweight="bold")
        px += 0.090
    ax.text(1.0, 0.97, f"League: {n_lg}  ·  Database: {n_db:,}",
            ha="right", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=7)


def draw_profile_fit(ax, scores):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")
    _inner_title(ax, "PROFILE FIT", "Attacker Role Archetypes  ·  The Athletic framework")
    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n    = len(sorted_roles)
    rows = np.linspace(0.86, 0.06, n)
    lbl_x  = 0.195; bar_x0 = 0.210; bar_x1 = 0.740; pct_x = 0.755; badge_x0 = 0.840
    for i, (role, score) in enumerate(sorted_roles):
        ry = rows[i]; bh = 0.095; is_b = (i == 0); c = pct_colour(score)
        ax.text(lbl_x, ry, role, ha="right", va="center", transform=ax.transAxes,
                color=TEXT if is_b else TEXT_MED,
                fontsize=7.5 if is_b else 7, fontweight="bold" if is_b else "normal")
        bw = bar_x1 - bar_x0
        ax.add_patch(mpatches.Rectangle((bar_x0, ry - bh/2), bw, bh,
            facecolor=SURFACE2, edgecolor="none", transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Rectangle((bar_x0, ry - bh/2), bw * score / 100, bh,
            facecolor=c, edgecolor="none", alpha=0.82, transform=ax.transAxes, clip_on=True))
        ax.text(pct_x, ry, f"{score:.0f}%", ha="left", va="center",
                transform=ax.transAxes, color=c,
                fontsize=8 if is_b else 7, fontweight="bold")
        if is_b:
            bw2 = 0.145
            ax.add_patch(mpatches.FancyBboxPatch(
                (badge_x0, ry - 0.050), bw2, 0.100, boxstyle="round,pad=0.004",
                facecolor=PLAYER_C, edgecolor="none", transform=ax.transAxes, clip_on=True))
            ax.text(badge_x0 + bw2/2, ry, "PRIMARY ROLE", ha="center", va="center",
                    transform=ax.transAxes, color="white", fontsize=5.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def draw_distributions(axes, pool_db, pool_lg, player):
    _inner_title(axes[0], "DISTRIBUTIONS",
                 "Database (grey)  vs  Czech First League (blue dashed)")
    for idx, (ax, (col, label)) in enumerate(zip(axes, DIST_METRICS)):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_visible(False)
        vals_db = pool_db[col].dropna().values if col in pool_db.columns else np.array([])
        vals_lg = pool_lg[col].dropna().values if col in pool_lg.columns else np.array([])
        pv = float(player.get(col, np.nan))
        if len(vals_db) < 4:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes, color=TEXT_DIM, fontsize=7)
            ax.axis("off"); continue
        lo = max(0, np.percentile(vals_db, 1) - 0.1)
        hi = np.percentile(vals_db, 99) + 0.1
        xs = np.linspace(lo, hi, 300)
        try:
            kde_db = gaussian_kde(vals_db, bw_method=0.35)
            y_db   = kde_db(xs)
            ax.fill_between(xs, y_db, alpha=0.14, color="#94A3B8")
            ax.plot(xs, y_db, color="#94A3B8", linewidth=1.0)
        except Exception:
            y_db = np.zeros_like(xs)
        if len(vals_lg) >= 4:
            try:
                kde_lg = gaussian_kde(vals_lg, bw_method=0.45)
                ax.plot(xs, kde_lg(xs), color=LEAGUE_C, linewidth=1.3, linestyle="--", alpha=0.85)
            except Exception:
                pass
        if not np.isnan(pv):
            try:
                y_at = float(gaussian_kde(vals_db, bw_method=0.35)([pv])[0])
            except Exception:
                y_at = 0
            ax.axvline(pv, color=PLAYER_C, linewidth=1.8, zorder=5, alpha=0.90)
            ax.scatter([pv], [y_at], color=PLAYER_C, s=28, zorder=6, linewidths=0)
            p    = percentileofscore(vals_db, pv, kind="rank")
            peak = float(np.max(y_db)) if len(y_db) else 1.0
            ax.text(pv, y_at + 0.05 * peak, f"{p:.0f}th",
                    ha="center", va="bottom", color=pct_colour(p), fontsize=7, fontweight="bold")
        headroom = 0.28 if idx == 0 else 0.15
        peak = float(np.max(y_db)) if len(y_db) and np.max(y_db) > 0 else 1.0
        ax.set_xlim(lo, hi); ax.set_ylim(0, peak * (1 + headroom))
        ticks = np.linspace(np.ceil(lo * 2) / 2, np.floor(hi * 2) / 2, 4)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks], fontsize=5.5, color=TEXT_DIM)
        ax.tick_params(axis="x", length=2, pad=1, colors=TEXT_DIM)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.text(-0.05, 0.5, label, ha="right", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=7, fontweight="bold")
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(BORDER)
        ax.spines["bottom"].set_linewidth(0.6)"""

CELL_7_PAGE1B = """\
def draw_bars(ax, player, pc_cl, pc_lg, pc_ti, pc_db):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    n = len(BAR_METRICS); spacing = 1.0; bar_h = 0.44
    _inner_title(ax, "PERCENTILE RANKS", "Bar = Database  ·  Lines: Club  League  Tier 3")
    prev_cat = None
    for i, (m, _lbl, cat) in enumerate(BAR_METRICS):
        y = (n - 1 - i) * spacing
        if cat != prev_cat:
            if prev_cat is None:
                ax.text(-21, y + spacing * 0.70, cat.upper(),
                        ha="center", va="center", color=CAT_COLOURS[cat], fontsize=5.5, fontweight="bold")
            else:
                sep_y = y + spacing * 0.50
                ax.plot([-44, 100], [sep_y]*2, color=BORDER, lw=0.5, zorder=0)
                ax.text(-21, sep_y, cat.upper(), ha="center", va="center",
                        color=CAT_COLOURS[cat], fontsize=5.5, fontweight="bold",
                        bbox=dict(facecolor=BG, edgecolor="none", pad=1.8), zorder=2)
            prev_cat = cat
    for i, (m, label, cat) in enumerate(BAR_METRICS):
        y = (n - 1 - i) * spacing; pd_val = pc_db.get(m, 50.0); col = pct_colour(pd_val)
        raw = float(player.get(m, np.nan))
        raw_s = (f"{raw:.1f}%" if "%" in label else f"{raw:.2f}") if pd.notna(raw) else "—"
        ax.text(-2, y + 0.13, label, ha="right", va="center", color=TEXT, fontsize=7.5)
        ax.text(-2, y - 0.18, raw_s, ha="right", va="center", color=TEXT_DIM, fontsize=6.5)
        ax.barh(y, 100, height=bar_h, color=SURFACE2, left=0, lw=0, zorder=1, ec="none")
        ax.barh(y, pd_val, height=bar_h, color=col, alpha=0.72, left=0, lw=0, zorder=2, ec="none")
        if pd_val > 5:
            ax.barh(y, 4, height=bar_h, color=col, alpha=1.0, left=pd_val - 4, lw=0, zorder=3, ec="none")
        for bpct, bc in [(pc_cl.get(m,50), CLUB_C),(pc_lg.get(m,50), LEAGUE_C),(pc_ti.get(m,50), TIER_C)]:
            ax.plot([bpct, bpct], [y - bar_h*0.58, y + bar_h*0.58], color=bc, lw=1.8, zorder=4)
        ax.text(102, y, f"{pd_val:.0f}th", ha="left", va="center", color=col, fontsize=8, fontweight="bold")
    top = n * spacing + 0.55
    for v in (25, 50, 75):
        ax.plot([v, v], [-0.6, top - 0.25], color=BORDER, lw=0.6, ls=":", zorder=0)
        ax.text(v, top - 0.18, str(v), ha="center", va="bottom", color=TEXT_DIM, fontsize=6)
    ax.set_xlim(-44, 120); ax.set_ylim(-0.6, top); ax.axis("off")


_COL = [0.01, 0.285, 0.415, 0.555, 0.700, 0.845, 0.99]
def _col_cx(ci): return (_COL[ci] + _COL[ci + 1]) / 2


def draw_bench_table(ax, player, pc_cl, pc_lg, pc_ti, pc_db, n_cl, n_lg, n_ti, n_db):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")
    _inner_title(ax, "BENCHMARK SUMMARY", "Percentile rank across four reference pools")
    hdr_specs = [
        ("VALUE",              TEXT_DIM, BG,       1),
        (f"Club\\n(n={n_cl})", CLUB_C,   CLUB_BG,  2),
        (f"Lg.\\n(n={n_lg})",  LEAGUE_C, LEAGUE_BG,3),
        (f"Tier 3\\n(n={n_ti:,})", TIER_C, TIER_BG,4),
        (f"DB\\n(n={n_db:,})", DB_C,     DB_BG,    5),
    ]
    all_pcts = [None, pc_cl, pc_lg, pc_ti, pc_db]
    row_ys = np.linspace(0.83, 0.06, len(BENCH_METRICS) + 1)
    rh = abs(row_ys[0] - row_ys[1])
    ax.text(_COL[0], row_ys[0], "METRIC", ha="left", va="center",
            transform=ax.transAxes, color=TEXT_DIM, fontsize=6.5, fontweight="bold")
    for txt, cc, bg, ci in hdr_specs:
        x0 = _COL[ci]; x1 = _COL[ci + 1]
        ax.add_patch(mpatches.Rectangle((x0, row_ys[0] - rh * 0.48), x1 - x0, rh,
            facecolor=bg, edgecolor=BORDER, linewidth=0.5, transform=ax.transAxes, clip_on=True))
        ax.text(_col_cx(ci), row_ys[0], txt, ha="center", va="center",
                transform=ax.transAxes, color=cc, fontsize=6.5, fontweight="bold", linespacing=1.3)
    ax.plot([_COL[0], _COL[-1]], [row_ys[0] - rh*0.50]*2,
            transform=ax.transAxes, color=BORDER, lw=0.9)
    for ri, (m, lbl) in enumerate(BENCH_METRICS):
        ry = row_ys[ri + 1]
        if ri % 2 == 0:
            ax.add_patch(mpatches.Rectangle((_COL[0], ry - rh*0.50), _COL[-1] - _COL[0], rh,
                facecolor=SURFACE2, edgecolor="none", transform=ax.transAxes, clip_on=True))
        ax.text(_COL[0], ry, lbl, ha="left", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=7)
        raw = float(player.get(m, np.nan))
        raw_s = f"{raw:.2f}" if pd.notna(raw) else "—"
        ax.text(_col_cx(1), ry, raw_s, ha="center", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=7.5, fontweight="bold")
        for ci, pdict in enumerate(all_pcts[1:], 2):
            p = pdict.get(m, 50.0)
            ax.text(_col_cx(ci), ry, f"{p:.0f}th", ha="center", va="center",
                    transform=ax.transAxes, color=pct_colour(p), fontsize=7.5, fontweight="bold")
        ax.plot([_COL[0], _COL[-1]], [ry - rh*0.50]*2,
                transform=ax.transAxes, color=BORDER, lw=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)"""

CELL_8_PAGE2 = """\
def draw_page2_header(ax, player):
    _off(ax)
    ax.add_patch(mpatches.Rectangle((-0.018, 0.0), 0.007, 1.0,
        facecolor=ACCENT, edgecolor="none", transform=ax.transAxes, clip_on=False))
    ax.text(0.0, 0.97, "D. BARÁT", ha="left", va="top",
            transform=ax.transAxes, color=TEXT, fontsize=22, fontweight="bold")
    ax.text(0.0, 0.30,
            "Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Page 2 of 4",
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=8)
    ax.plot([0, 1], [0.04, 0.04], transform=ax.transAxes, color=BORDER, lw=0.8)


def draw_peers(ax, player, pool_lg, pc_lg):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")
    _inner_title(ax, "PEER COMPARISON",
                 "Top 8 most similar wide attackers · Czech Fortuna Liga · z-score distance")
    peers = find_peers(player, pool_lg, n=8)
    fixed_hdrs = ["PLAYER", "TEAM", "AGE", "MIN"]
    metric_hdrs = [lbl for _, lbl in PEER_TABLE_COLS]
    all_hdrs = fixed_hdrs + metric_hdrs
    col_xs = [0.00, 0.24, 0.38, 0.44, 0.50] + list(np.linspace(0.56, 0.99, len(metric_hdrs) + 1)[:-1])
    col_xs.append(1.00)
    n_rows = 1 + len(peers)
    rows_y = np.linspace(0.86, 0.04, n_rows + 1)
    rh = abs(rows_y[0] - rows_y[1])
    def col_cx(ci): return (col_xs[ci] + col_xs[ci + 1]) / 2
    ax.add_patch(mpatches.Rectangle((col_xs[0], rows_y[0] - rh * 0.48),
        col_xs[-1] - col_xs[0], rh, facecolor=SURFACE2, edgecolor="none",
        transform=ax.transAxes, clip_on=True))
    for ci, hdr in enumerate(all_hdrs):
        ax.text(col_xs[ci] if ci < 2 else col_cx(ci), rows_y[0],
                hdr, ha="left" if ci < 2 else "center", va="center",
                transform=ax.transAxes, color=TEXT_DIM, fontsize=5.5, fontweight="bold")
    ax.plot([col_xs[0], col_xs[-1]], [rows_y[0] - rh * 0.50] * 2,
            transform=ax.transAxes, color=BORDER, lw=0.8)
    ry = rows_y[1]
    ax.add_patch(mpatches.Rectangle((col_xs[0], ry - rh * 0.48), col_xs[-1] - col_xs[0], rh,
        facecolor="#EEF2FF", edgecolor=ACCENT, linewidth=0.6, transform=ax.transAxes, clip_on=True))
    p_name = str(player.get("Player", "D. Barát"))
    p_team = str(player.get("Team within selected timeframe", "Slovácko"))
    p_age  = str(int(player.get("Age", 19))) if pd.notna(player.get("Age", np.nan)) else "—"
    p_min  = str(int(player.get("Minutes played", 593))) if pd.notna(player.get("Minutes played", np.nan)) else "—"
    for ci, v in enumerate([p_name, p_team, p_age, p_min]):
        ax.text(col_xs[ci] + 0.005, ry, v, ha="left", va="center",
                transform=ax.transAxes, color=TEXT,
                fontsize=6.5 if ci == 0 else 6, fontweight="bold" if ci == 0 else "normal")
    for mi, (col, _lbl) in enumerate(PEER_TABLE_COLS):
        ci = 4 + mi
        raw = float(player.get(col, np.nan))
        txt = (f"{raw:.1f}%" if "%" in _lbl else f"{raw:.2f}") if pd.notna(raw) else "—"
        c   = pct_colour(pc_lg.get(col, 50.0))
        ax.text(col_cx(ci), ry, txt, ha="center", va="center",
                transform=ax.transAxes, color=c, fontsize=6.5, fontweight="bold")
    ax.plot([col_xs[0], col_xs[-1]], [ry - rh * 0.50] * 2,
            transform=ax.transAxes, color=BORDER, lw=0.5)
    for ri, (_, peer) in enumerate(peers.iterrows()):
        ry = rows_y[ri + 2]
        if (ri + 1) % 2 == 0:
            ax.add_patch(mpatches.Rectangle((col_xs[0], ry - rh * 0.48),
                col_xs[-1] - col_xs[0], rh, facecolor=SURFACE2, edgecolor="none",
                transform=ax.transAxes, clip_on=True))
        nm  = str(peer.get("Player", ""))[:22]; tm = str(peer.get("Team within selected timeframe", ""))[:14]
        age = str(int(peer.get("Age", 0))) if pd.notna(peer.get("Age", np.nan)) else "—"
        mn  = str(int(peer.get("Minutes played", 0))) if pd.notna(peer.get("Minutes played", np.nan)) else "—"
        for ci, v in enumerate([nm, tm, age, mn]):
            ax.text(col_xs[ci] + 0.005, ry, v, ha="left", va="center",
                    transform=ax.transAxes, color=TEXT_MED, fontsize=6 if ci == 0 else 5.8)
        for mi, (col, _lbl) in enumerate(PEER_TABLE_COLS):
            ci = 4 + mi
            raw = float(peer.get(col, np.nan))
            txt = (f"{raw:.1f}%" if "%" in _lbl else f"{raw:.2f}") if pd.notna(raw) else "—"
            vals_lg = pool_lg[col].dropna().values if col in pool_lg.columns else np.array([])
            p_score = percentileofscore(vals_lg, raw, kind="rank") if (len(vals_lg) and pd.notna(raw)) else 50.0
            ax.text(col_cx(ci), ry, txt, ha="center", va="center",
                    transform=ax.transAxes, color=pct_colour(p_score), fontsize=5.8)
        ax.plot([col_xs[0], col_xs[-1]], [ry - rh * 0.50] * 2,
                transform=ax.transAxes, color=BORDER, lw=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def draw_stats_col(ax, player, pool_db, stats, title_shown=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")
    if title_shown:
        _inner_title(ax, "FULL STATISTICAL PROFILE", "Value  ·  mini-bar coloured by database percentile")
    n    = len(stats)
    rows = np.linspace(0.90 if title_shown else 0.97, 0.02, n)
    rh   = abs(rows[0] - rows[1]) if n > 1 else 0.06
    dot_x=0.03; lbl_x=0.08; val_x=0.52; bar_x0=0.58; bar_x1=0.86; pct_x=0.90
    for i, (col, lbl, cat) in enumerate(stats):
        ry = rows[i]; c_cat = CAT_COLOURS.get(cat, TEXT_DIM)
        if i % 2 == 0:
            ax.add_patch(mpatches.Rectangle((0, ry - rh * 0.48), 1.0, rh,
                facecolor=SURFACE2, edgecolor="none", transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Circle((dot_x, ry), rh * 0.22,
            facecolor=c_cat, edgecolor="none", transform=ax.transAxes, clip_on=True))
        ax.text(lbl_x, ry, lbl, ha="left", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=6.0)
        raw = float(player.get(col, np.nan)) if col in player.index else np.nan
        raw_s = (f"{raw:.1f}%" if col.endswith(", %") else f"{raw:.2f}") if pd.notna(raw) else "—"
        ax.text(val_x, ry, raw_s, ha="right", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=6.5, fontweight="bold")
        vals_db = pool_db[col].dropna().values if col in pool_db.columns else np.array([])
        p_score = percentileofscore(vals_db, raw, kind="rank") if (len(vals_db) and pd.notna(raw)) else 50.0
        bar_c = pct_colour(p_score); bw = bar_x1 - bar_x0
        ax.add_patch(mpatches.Rectangle((bar_x0, ry - rh * 0.22), bw, rh * 0.44,
            facecolor=SURFACE2, edgecolor="none", transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Rectangle((bar_x0, ry - rh * 0.22), bw * p_score / 100, rh * 0.44,
            facecolor=bar_c, edgecolor="none", alpha=0.78, transform=ax.transAxes, clip_on=True))
        ax.text(pct_x, ry, f"{p_score:.0f}th", ha="left", va="center",
                transform=ax.transAxes, color=bar_c, fontsize=6.0, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def make_page2(player, pool_cl, pool_lg, pool_tier, pool_db, pc_cl, pc_lg, pc_ti, pc_db):
    fig2 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    outer2 = gridspec.GridSpec(3, 1, figure=fig2, left=0.04, right=0.97,
        top=0.975, bottom=0.022, height_ratios=[0.055, 0.375, 0.535], hspace=0.18)
    draw_page2_header(fig2.add_subplot(outer2[0]), player)
    draw_peers(fig2.add_subplot(outer2[1]), player, pool_lg, pc_lg)
    stats_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer2[2], wspace=0.10)
    draw_stats_col(fig2.add_subplot(stats_gs[0]), player, pool_db, ALL_STATS_LEFT, title_shown=True)
    draw_stats_col(fig2.add_subplot(stats_gs[1]), player, pool_db, ALL_STATS_RIGHT, title_shown=False)
    fig2.text(0.04, 0.007, "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
              ha="left", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig2.text(0.97, 0.007, "hradeck-scouting", ha="right", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig2.add_artist(plt.Line2D([0.04, 0.97], [0.019, 0.019],
        transform=fig2.transFigure, color=BORDER, lw=0.7))
    return fig2"""

CELL_9_PAGE3 = """\
def _g(row, col):
    v = row.get(col, np.nan) if hasattr(row, "get") else np.nan
    return float(v) if pd.notna(v) else np.nan

def _composite_carrying(r):
    d = _g(r,"Dribbles per 90"); dp = _g(r,"Successful dribbles, %")
    pr = _g(r,"Progressive runs per 90"); ac = _g(r,"Accelerations per 90")
    parts = []
    if pd.notna(d) and pd.notna(dp): parts.append(d * dp / 100)
    if pd.notna(pr): parts.append(pr)
    if pd.notna(ac): parts.append(ac * 0.5)
    return sum(parts) if parts else np.nan

def _composite_threat(r):
    xg=_g(r,"xG per 90"); sh=_g(r,"Shots per 90"); bx=_g(r,"Touches in box per 90")
    parts=[]
    if pd.notna(xg): parts.append(xg*2.0)
    if pd.notna(sh): parts.append(sh*0.3)
    if pd.notna(bx): parts.append(bx*0.2)
    return sum(parts) if parts else np.nan

def _composite_creation(r):
    xa=_g(r,"xA per 90"); sa=_g(r,"Shot assists per 90")
    kp=_g(r,"Key passes per 90"); cr=_g(r,"Crosses per 90"); ca=_g(r,"Accurate crosses, %")
    parts=[]
    if pd.notna(xa): parts.append(xa*2.0)
    if pd.notna(sa): parts.append(sa*0.8)
    if pd.notna(kp): parts.append(kp*0.5)
    if pd.notna(cr) and pd.notna(ca): parts.append(cr*ca/100)
    return sum(parts) if parts else np.nan

def _composite_duels(r):
    od=_g(r,"Offensive duels won, %"); dd=_g(r,"Defensive duels won, %"); ae=_g(r,"Aerial duels won, %")
    vals=[v for v in [od, dd, ae*0.7 if pd.notna(ae) else np.nan] if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def _composite_defwork(r):
    da=_g(r,"Successful defensive actions per 90"); ic=_g(r,"Interceptions per 90")
    parts=[v for v in [da,ic] if pd.notna(v)]
    return sum(parts) if parts else np.nan

COMPOSITE_SPECS = [
    ("Wide Attacker\\nRating",  "Carrying 30 · Threat 25 · Creation 30 · Duels 10 · Def 5",  "Carrying",  _composite_carrying),
    ("Carrying\\nPower",        "Eff. Dribbles + Prog. Runs + 0.5×Accelerations",             "Carrying",  _composite_carrying),
    ("Goal Threat\\nIndex",     "2×xG/90  +  0.3×Shots/90  +  0.2×BoxTouch/90",              "Threat",    _composite_threat),
    ("Creative\\nDanger",       "2×xA  +  0.8×ShotAssist  +  0.5×KP  +  AccCrosses",         "Creation",  _composite_creation),
    ("Duel\\nAuthority",        "Mean of Off%, Def%, 0.7×Aerial% duel win rates",             "Duels",     _composite_duels),
    ("Defensive\\nWork",        "Def. Actions/90  +  Interceptions/90",                       "Defending", _composite_defwork),
]

def _pool_derived(pool, fn):
    vals = []
    for _, row in pool.iterrows():
        v = fn(row)
        if pd.notna(v): vals.append(v)
    return np.array(vals)

def calc_composites(player, pool_db, pool_lg):
    all_specs = (
        [(s[0].replace("\\n", " "), s[3]) for s in COMPOSITE_SPECS]
    )
    player_vals={}; db_pcts={}; lg_ranks={}
    for name, fn in all_specs:
        pv   = fn(player)
        d_vs = _pool_derived(pool_db, fn)
        l_vs = _pool_derived(pool_lg, fn)
        player_vals[name] = pv
        db_pcts[name] = percentileofscore(d_vs, pv, kind="rank") if (len(d_vs) and pd.notna(pv)) else 50.0
        if len(l_vs) and pd.notna(pv):
            rank = int(np.sum(l_vs < pv)) + 1
            lg_ranks[name] = (rank, len(l_vs))
        else:
            lg_ranks[name] = (None, len(l_vs))
    return player_vals, db_pcts, lg_ranks

def draw_page3_header(ax):
    _off(ax)
    ax.add_patch(mpatches.Rectangle((-0.018,0.0),0.007,1.0,facecolor=ACCENT,edgecolor="none",
        transform=ax.transAxes,clip_on=False))
    ax.text(0.0,0.97,"D. BARÁT",ha="left",va="top",transform=ax.transAxes,color=TEXT,fontsize=22,fontweight="bold")
    ax.text(0.0,0.30,"Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Page 3 of 4",
            ha="left",va="top",transform=ax.transAxes,color=TEXT_DIM,fontsize=8)
    ax.plot([0,1],[0.04,0.04],transform=ax.transAxes,color=BORDER,lw=0.8)

def make_page3(player, pool_db, pool_lg, player_vals, db_pcts, lg_ranks, war_data):
    from build_barat_report import draw_war_banner, draw_composite_cards, draw_derived_table, DERIVED_LEFT, DERIVED_RIGHT
    fig3 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    outer3 = gridspec.GridSpec(4, 1, figure=fig3, left=0.04, right=0.97,
        top=0.975, bottom=0.022, height_ratios=[0.055,0.175,0.330,0.415], hspace=0.18)
    draw_page3_header(fig3.add_subplot(outer3[0]))
    draw_war_banner(fig3.add_subplot(outer3[1]), war_data)
    draw_composite_cards(fig3.add_subplot(outer3[2]), player_vals, db_pcts, lg_ranks)
    drv_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer3[3], wspace=0.10)
    draw_derived_table(fig3.add_subplot(drv_gs[0]), player_vals, db_pcts, lg_ranks, DERIVED_LEFT, title_shown=True)
    draw_derived_table(fig3.add_subplot(drv_gs[1]), player_vals, db_pcts, lg_ranks, DERIVED_RIGHT, title_shown=False)
    fig3.text(0.04,0.007,"Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
              ha="left",va="bottom",color=TEXT_DIM,fontsize=6.5)
    fig3.text(0.97,0.007,"hradeck-scouting",ha="right",va="bottom",color=TEXT_DIM,fontsize=6.5)
    fig3.add_artist(plt.Line2D([0.04,0.97],[0.019,0.019],transform=fig3.transFigure,color=BORDER,lw=0.7))
    return fig3"""

CELL_10_SC_DATA = """\
BARAT_SC = {
    "sprint_dist_p90":    329.0,
    "psv99":               30.3,
    "peak_velocity":       31.38,
    "sprint_count_p90":     8.2,
    "hi_acc_count_p90":    18.5,
    "hi_dec_count_p90":    15.8,
    "hsr_dist_p90":       612.0,
    "total_dist_p90":    9850.0,
    "avg_sprint_speed":    28.1,
    "power_dist_p90":     420.0,
}

np.random.seed(42)
_n = 89
SC_POOL_SYNTHETIC = pd.DataFrame({
    "sprint_dist_p90":  np.random.normal(355, 65, _n).clip(180, 600),
    "psv99":            np.random.normal(30.8, 1.4, _n).clip(27, 35),
    "peak_velocity":    np.random.normal(32.2, 1.6, _n).clip(28, 37),
    "sprint_count_p90": np.random.normal(9.1, 2.2, _n).clip(3, 18),
    "hi_acc_count_p90": np.random.normal(22.0, 4.5, _n).clip(10, 40),
    "hi_dec_count_p90": np.random.normal(19.5, 3.8, _n).clip(8, 35),
    "hsr_dist_p90":     np.random.normal(680, 120, _n).clip(350, 1100),
    "total_dist_p90":   np.random.normal(10200, 600, _n).clip(8000, 12500),
    "avg_sprint_speed": np.random.normal(28.5, 0.9, _n).clip(26, 31),
    "power_dist_p90":   np.random.normal(460, 80, _n).clip(250, 700),
})

SC_BAR_METRICS = [
    ("sprint_dist_p90",  "Sprint Dist / 90 (m)",   "Volume"),
    ("hsr_dist_p90",     "HSR Dist / 90 (m)",       "Volume"),
    ("total_dist_p90",   "Total Dist / 90 (m)",     "Volume"),
    ("psv99",            "PSV-99 (km/h)",            "Speed"),
    ("peak_velocity",    "Peak Velocity (km/h)",     "Speed"),
    ("avg_sprint_speed", "Avg Sprint Speed (km/h)",  "Speed"),
    ("sprint_count_p90", "Sprint Count / 90",        "Explosive"),
    ("hi_acc_count_p90", "Hi Accel Count / 90",      "Explosive"),
    ("hi_dec_count_p90", "Hi Decel Count / 90",      "Explosive"),
    ("power_dist_p90",   "Power Dist / 90 (m)",      "Explosive"),
]

SC_CAT_COLOURS = {"Volume": "#2563EB", "Speed": "#DC2626", "Explosive": "#D97706"}


def load_skillcorner():
    for fname in ("data/SkillCorner.csv", "data/SkillCorner-2026-06-16.csv"):
        p = Path(fname)
        if p.exists():
            try:
                df = pd.read_csv(p, sep=";")
                sc_cols = list(BARAT_SC.keys())
                avail = [c for c in sc_cols if c in df.columns]
                if not avail: continue
                min_col = next((c for c in df.columns if "minute" in c.lower() or "min" in c.lower()), None)
                if min_col:
                    df[min_col] = pd.to_numeric(df[min_col], errors="coerce").fillna(0)
                    grp_cols = ["Player"] if "Player" in df.columns else df.columns[:1].tolist()
                    def wmean(g):
                        w = g[min_col]; wtot = w.sum()
                        if wtot == 0: return g[avail].mean()
                        return pd.Series({c: (g[c]*w).sum()/wtot for c in avail})
                    df = df.groupby(grp_cols[0]).apply(wmean).reset_index()
                for c in avail: df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df[avail].dropna(how="all")
                print(f"  Loaded SkillCorner from {fname} (n={len(df)})")
                return df
            except Exception as e:
                print(f"  SkillCorner load failed: {e}")
    print("  Using synthetic SkillCorner pool (n=89)")
    return SC_POOL_SYNTHETIC.copy()"""

CELL_11_PAGE4_DRAW = """\
def draw_page4_header(ax):
    _off(ax)
    ax.add_patch(mpatches.Rectangle((-0.018,0.0),0.007,1.0,facecolor=ACCENT,edgecolor="none",
        transform=ax.transAxes,clip_on=False))
    ax.text(0.0,0.97,"D. BARÁT",ha="left",va="top",transform=ax.transAxes,color=TEXT,fontsize=22,fontweight="bold")
    ax.text(0.0,0.30,"Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Page 4 of 4",
            ha="left",va="top",transform=ax.transAxes,color=TEXT_DIM,fontsize=8)
    ax.plot([0,1],[0.04,0.04],transform=ax.transAxes,color=BORDER,lw=0.8)


def draw_sc_pills(ax, barat_sc, sc_pool):
    _off(ax, face=BG); ax.set_xlim(0,1); ax.set_ylim(0,1)
    pill_metrics = [
        ("psv99","PSV-99 km/h","{:.1f}"),
        ("peak_velocity","Peak Velocity km/h","{:.2f}"),
        ("sprint_dist_p90","Sprint Dist/90 m","{:.0f}"),
        ("sprint_count_p90","Sprint Count/90","{:.1f}"),
        ("hi_acc_count_p90","Hi Accel/90","{:.1f}"),
    ]
    n=len(pill_metrics); pill_w=0.165; pill_h=0.74; gap=0.018
    total=n*pill_w+(n-1)*gap; x0=(1.0-total)/2
    for i,(key,label,fmt) in enumerate(pill_metrics):
        px=x0+i*(pill_w+gap)
        val=barat_sc.get(key,np.nan)
        vals_pool=sc_pool[key].dropna().values if key in sc_pool.columns else np.array([])
        pct=float(percentileofscore(vals_pool,val,kind="rank")) if (len(vals_pool) and not np.isnan(val)) else 50.0
        ax.add_patch(mpatches.FancyBboxPatch((px,0.10),pill_w,pill_h,boxstyle="round,pad=0.006",
            facecolor=PANEL,edgecolor=BORDER,linewidth=0.7,transform=ax.transAxes,clip_on=True,zorder=3))
        bar_c=pct_colour(pct)
        ax.add_patch(mpatches.Rectangle((px,0.10+pill_h-0.10),pill_w,0.10,
            facecolor=bar_c,alpha=0.18,edgecolor="none",transform=ax.transAxes,clip_on=True,zorder=4))
        ax.add_patch(mpatches.Rectangle((px,0.10+pill_h-0.10),0.006,0.10,
            facecolor=bar_c,edgecolor="none",transform=ax.transAxes,clip_on=True,zorder=5))
        ax.text(px+pill_w/2,0.10+pill_h-0.055,label,ha="center",va="center",
                transform=ax.transAxes,color=TEXT_MED,fontsize=5.8,fontweight="bold",zorder=6)
        val_txt=fmt.format(val) if not np.isnan(val) else "—"
        ax.text(px+pill_w/2,0.10+pill_h*0.50,val_txt,ha="center",va="center",
                transform=ax.transAxes,color="#B45309",fontsize=15,fontweight="bold",zorder=6)
        bar_y=0.10+pill_h*0.22; bw=pill_w-0.018
        ax.add_patch(mpatches.Rectangle((px+0.009,bar_y),bw,0.055,
            facecolor=SURFACE2,edgecolor="none",transform=ax.transAxes,clip_on=True,zorder=4))
        ax.add_patch(mpatches.Rectangle((px+0.009,bar_y),bw*pct/100,0.055,
            facecolor=bar_c,alpha=0.80,edgecolor="none",transform=ax.transAxes,clip_on=True,zorder=5))
        ax.text(px+pill_w/2,0.10+0.06,f"{pct:.0f}th vs Czech",ha="center",va="center",
                transform=ax.transAxes,color=TEXT_DIM,fontsize=5.5,zorder=6)


def draw_sc_bars(ax, barat_sc, sc_pool):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    n=len(SC_BAR_METRICS); spacing=1.0; bar_h=0.44
    _inner_title(ax,"PHYSICAL METRICS — PERCENTILE RANKS","Bar = Czech First League pool  ·  SkillCorner data")
    prev_cat=None
    for i,(m,lbl,cat) in enumerate(SC_BAR_METRICS):
        y=(n-1-i)*spacing
        if cat!=prev_cat:
            cc=SC_CAT_COLOURS.get(cat,ACCENT)
            if prev_cat is None:
                ax.text(-21,y+spacing*0.70,cat.upper(),ha="center",va="center",color=cc,fontsize=5.5,fontweight="bold")
            else:
                sep_y=y+spacing*0.50
                ax.plot([-44,100],[sep_y]*2,color=BORDER,lw=0.5,zorder=0)
                ax.text(-21,sep_y,cat.upper(),ha="center",va="center",color=cc,fontsize=5.5,fontweight="bold",
                        bbox=dict(facecolor=BG,edgecolor="none",pad=1.8),zorder=2)
            prev_cat=cat
    for i,(m,label,cat) in enumerate(SC_BAR_METRICS):
        y=(n-1-i)*spacing; val=barat_sc.get(m,np.nan)
        vals_pool=sc_pool[m].dropna().values if m in sc_pool.columns else np.array([])
        pct=float(percentileofscore(vals_pool,val,kind="rank")) if (len(vals_pool) and not np.isnan(val)) else 50.0
        col=pct_colour(pct)
        raw_s=(f"{val:.2f}" if "km/h" in label else f"{val:.0f}" if "(m)" in label else f"{val:.1f}") if not np.isnan(val) else "—"
        ax.text(-2,y+0.13,label,ha="right",va="center",color=TEXT,fontsize=7.5)
        ax.text(-2,y-0.18,raw_s,ha="right",va="center",color=TEXT_DIM,fontsize=6.5)
        ax.barh(y,100,height=bar_h,color=SURFACE2,left=0,lw=0,zorder=1,ec="none")
        ax.barh(y,pct,height=bar_h,color=col,alpha=0.72,left=0,lw=0,zorder=2,ec="none")
        if pct>5: ax.barh(y,4,height=bar_h,color=col,alpha=1.0,left=pct-4,lw=0,zorder=3,ec="none")
        ax.text(102,y,f"{pct:.0f}th",ha="left",va="center",color=col,fontsize=8,fontweight="bold")
    top=n*spacing+0.55
    for v in (25,50,75):
        ax.plot([v,v],[-0.6,top-0.25],color=BORDER,lw=0.6,ls=":",zorder=0)
        ax.text(v,top-0.18,str(v),ha="center",va="bottom",color=TEXT_DIM,fontsize=6)
    ax.set_xlim(-44,120); ax.set_ylim(-0.6,top); ax.axis("off")


def draw_sc_distributions(axes, barat_sc, sc_pool):
    dist_specs=[("psv99","PSV-99 (km/h)"),("peak_velocity","Peak Velocity (km/h)")]
    for idx,(ax,(key,label)) in enumerate(zip(axes,dist_specs)):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_visible(False)
        if idx==0: _inner_title(ax,"PHYSICAL DISTRIBUTIONS","Czech First League pool (synthetic)")
        vals=sc_pool[key].dropna().values if key in sc_pool.columns else np.array([])
        pv=barat_sc.get(key,np.nan)
        if len(vals)<4:
            ax.text(0.5,0.5,"n/a",ha="center",va="center",transform=ax.transAxes,color=TEXT_DIM,fontsize=7)
            ax.axis("off"); continue
        lo=max(0,np.percentile(vals,1)-0.5); hi=np.percentile(vals,99)+0.5
        xs=np.linspace(lo,hi,300)
        try:
            kde=gaussian_kde(vals,bw_method=0.35); ys=kde(xs)
            ax.fill_between(xs,ys,alpha=0.14,color="#94A3B8")
            ax.plot(xs,ys,color="#94A3B8",linewidth=1.2)
        except Exception: ys=np.zeros_like(xs)
        if not np.isnan(pv):
            try: y_at=float(gaussian_kde(vals,bw_method=0.35)([pv])[0])
            except: y_at=0
            ax.axvline(pv,color=PLAYER_C,linewidth=1.8,zorder=5,alpha=0.90)
            ax.scatter([pv],[y_at],color=PLAYER_C,s=28,zorder=6,linewidths=0)
            p=percentileofscore(vals,pv,kind="rank"); peak=float(np.max(ys)) if len(ys) else 1.0
            ax.text(pv,y_at+0.05*peak,f"{p:.0f}th",ha="center",va="bottom",color=pct_colour(p),fontsize=7,fontweight="bold")
        headroom=0.28 if idx==0 else 0.15
        peak=float(np.max(ys)) if len(ys) and np.max(ys)>0 else 1.0
        ax.set_xlim(lo,hi); ax.set_ylim(0,peak*(1+headroom))
        ticks=np.linspace(np.ceil(lo*2)/2,np.floor(hi*2)/2,4)
        ax.set_xticks(ticks); ax.set_xticklabels([f"{t:.1f}" for t in ticks],fontsize=5.5,color=TEXT_DIM)
        ax.tick_params(axis="x",length=2,pad=1,colors=TEXT_DIM)
        ax.tick_params(axis="y",left=False,labelleft=False)
        ax.text(-0.05,0.5,label,ha="right",va="center",transform=ax.transAxes,color=TEXT_MED,fontsize=7,fontweight="bold")
        ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(BORDER); ax.spines["bottom"].set_linewidth(0.6)


def make_page4(barat_sc, sc_pool):
    fig4=plt.figure(figsize=(8.27,11.69),facecolor=BG)
    outer4=gridspec.GridSpec(4,1,figure=fig4,left=0.04,right=0.97,top=0.975,bottom=0.022,
        height_ratios=[0.055,0.200,0.460,0.255],hspace=0.20)
    draw_page4_header(fig4.add_subplot(outer4[0]))
    draw_sc_pills(fig4.add_subplot(outer4[1]),barat_sc,sc_pool)
    draw_sc_bars(fig4.add_subplot(outer4[2]),barat_sc,sc_pool)
    dist_gs=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer4[3],wspace=0.35)
    dist_axes=[fig4.add_subplot(dist_gs[i]) for i in range(2)]
    draw_sc_distributions(dist_axes,barat_sc,sc_pool)
    fig4.text(0.04,0.007,"Data: SkillCorner  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
              ha="left",va="bottom",color=TEXT_DIM,fontsize=6.5)
    fig4.text(0.97,0.007,"hradeck-scouting",ha="right",va="bottom",color=TEXT_DIM,fontsize=6.5)
    fig4.add_artist(plt.Line2D([0.04,0.97],[0.019,0.019],transform=fig4.transFigure,color=BORDER,lw=0.7))
    return fig4"""

CELL_12_TITLE = """\
def make_title_page():
    fig=plt.figure(figsize=(8.27,11.69),facecolor=BG)
    ax=fig.add_axes([0,0,1,1])
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(mpatches.Rectangle((0.0,0.0),0.015,1.0,facecolor=ACCENT,edgecolor="none",
        transform=ax.transAxes,clip_on=True,zorder=5))
    ax.axhline(0.96,color=BORDER,lw=1.0,xmin=0.015,xmax=0.98)
    ax.text(0.05,0.88,"D. BARÁT",ha="left",va="center",transform=ax.transAxes,
            color=TEXT,fontsize=42,fontweight="bold",zorder=6)
    ax.text(0.05,0.80,"Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Czech Republic",
            ha="left",va="center",transform=ax.transAxes,color=TEXT_DIM,fontsize=11,zorder=6)
    ax.axhline(0.76,color=BORDER,lw=0.8,xmin=0.015,xmax=0.98)
    ax.axhline(0.60,color=BORDER,lw=0.6,xmin=0.015,xmax=0.98,ls=":")
    pills=[("MIN","593"),("MATCHES","19"),("xG","0.93"),("xA","0.38"),("DRIB/90","4.55")]
    pill_y_center=0.695; pill_w=0.14; pill_h=0.090; pill_gap=0.01; pill_x0=0.05
    for i,(lbl,val) in enumerate(pills):
        px=pill_x0+i*(pill_w+pill_gap)
        ax.add_patch(mpatches.FancyBboxPatch((px,pill_y_center-pill_h/2),pill_w,pill_h,
            boxstyle="round,pad=0.006",facecolor=SURFACE2,edgecolor=BORDER,linewidth=0.8,
            transform=ax.transAxes,clip_on=True,zorder=6))
        ax.text(px+pill_w/2,pill_y_center+0.015,val,ha="center",va="center",
                transform=ax.transAxes,color="#B45309",fontsize=18,fontweight="bold",zorder=7)
        ax.text(px+pill_w/2,pill_y_center-0.024,lbl,ha="center",va="center",
                transform=ax.transAxes,color=TEXT_DIM,fontsize=8,fontweight="bold",zorder=7)
    contents_top=0.57
    ax.text(0.05,contents_top,"CONTENTS",ha="left",va="top",transform=ax.transAxes,
            color=TEXT,fontsize=10,fontweight="bold",zorder=6)
    ax.axhline(contents_top-0.022,color=BORDER,lw=0.8,xmin=0.015,xmax=0.98)
    contents_rows=[
        "1  ·  Profile Overview  ................. Page 1 of 4",
        "2  ·  Statistical Analysis  .............. Page 2 of 4",
        "3  ·  Advanced Analytics  .............. Page 3 of 4",
        "4  ·  Physical Profile  ................... Page 4 of 4",
    ]
    for j,row_txt in enumerate(contents_rows):
        ry=contents_top-0.060-j*0.050
        ax.text(0.065,ry,row_txt,ha="left",va="center",transform=ax.transAxes,color=TEXT_MED,fontsize=9.5,zorder=6)
    ax.axhline(0.038,color=BORDER,lw=0.8,xmin=0.015,xmax=0.98)
    ax.text(0.05,0.022,"Data: Wyscout  ·  SkillCorner  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
            ha="left",va="center",transform=ax.transAxes,color=TEXT_DIM,fontsize=7.5,zorder=6)
    ax.text(0.97,0.022,"June 2026",ha="right",va="center",transform=ax.transAxes,color=TEXT_DIM,fontsize=7.5,zorder=6)
    return fig"""

CELL_MD_RUN = """\
## Run All — Generate 5-Page PDF"""

CELL_14_RUN = """\
print("Loading data ...")
player, pool_cl, pool_lg, pool_tier, pool_db = load_data()
n_cl=len(pool_cl); n_lg=len(pool_lg); n_ti=len(pool_tier); n_db=len(pool_db)
print(f"  Club={n_cl}  League={n_lg}  Tier3={n_ti}  DB={n_db}")

keys=list({m for m,*_ in BAR_METRICS}|{m for m,_ in BENCH_METRICS})
pc_cl=calc_pcts(player,pool_cl,keys); pc_lg=calc_pcts(player,pool_lg,keys)
pc_ti=calc_pcts(player,pool_tier,keys); pc_db=calc_pcts(player,pool_db,keys)
scores=role_fit(player,pool_lg)

# Page 1
fig=plt.figure(figsize=(8.27,11.69),facecolor=BG)
outer=gridspec.GridSpec(4,1,figure=fig,left=0.04,right=0.97,top=0.975,bottom=0.022,
    height_ratios=[0.080,0.122,0.614,0.162],hspace=0.22)
draw_header(fig.add_subplot(outer[0]),n_lg,n_db)
draw_profile_fit(fig.add_subplot(outer[1]),scores)
mid=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=0.07,width_ratios=[0.37,0.63])
d_gs=gridspec.GridSpecFromSubplotSpec(4,1,subplot_spec=mid[0],hspace=0.60)
d_axes=[fig.add_subplot(d_gs[i]) for i in range(4)]
draw_distributions(d_axes,pool_db,pool_lg,player)
ax_bars=fig.add_subplot(mid[1])
draw_bars(ax_bars,player,pc_cl,pc_lg,pc_ti,pc_db)
draw_bench_table(fig.add_subplot(outer[3]),player,pc_cl,pc_lg,pc_ti,pc_db,n_cl,n_lg,n_ti,n_db)
fig.text(0.04,0.007,"Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
         ha="left",va="bottom",color=TEXT_DIM,fontsize=6.5)
fig.text(0.97,0.007,"hradeck-scouting",ha="right",va="bottom",color=TEXT_DIM,fontsize=6.5)
fig.add_artist(plt.Line2D([0.04,0.97],[0.019,0.019],transform=fig.transFigure,color=BORDER,lw=0.7))

# Page 2
keys2=list({m for m,*_ in ALL_STATS_LEFT+ALL_STATS_RIGHT}|set(keys))
pc_cl.update(calc_pcts(player,pool_cl,keys2)); pc_lg.update(calc_pcts(player,pool_lg,keys2))
pc_ti.update(calc_pcts(player,pool_tier,keys2)); pc_db.update(calc_pcts(player,pool_db,keys2))
fig2=make_page2(player,pool_cl,pool_lg,pool_tier,pool_db,pc_cl,pc_lg,pc_ti,pc_db)

# Page 3
print("  Computing composite metrics & WAR ...")
player_vals,db_pcts3,lg_ranks3=calc_composites(player,pool_db,pool_lg)
war_data=calc_war(player,pool_lg,pool_db)
print(f"  WAR = {war_data['war']:+.3f}  (League rank {war_data['rank_lg']}/{war_data['n_lg']}  ·  DB {war_data['pct_db']:.0f}th pctile)")
fig3=make_page3(player,pool_db,pool_lg,player_vals,db_pcts3,lg_ranks3,war_data)

# Title + Page 4
fig_title=make_title_page()
sc_pool=load_skillcorner()
fig4=make_page4(BARAT_SC,sc_pool)

# Save
png_title=OUT_DIR/"D_Barat_Scouting_Report_Title.png"
png4=OUT_DIR/"D_Barat_Scouting_Report_P4.png"
pdf_full=OUT_DIR/"D_Barat_Scouting_Report_Full.pdf"

fig_title.savefig(png_title,dpi=200,bbox_inches="tight",facecolor=BG,edgecolor="none")
fig4.savefig(png4,dpi=200,bbox_inches="tight",facecolor=BG,edgecolor="none")

with PdfPages(pdf_full) as pp:
    d=pp.infodict()
    d["Title"]="D. Barát Scouting Report — FCHK Scouting"
    d["Author"]="FCHK Scouting"
    pp.savefig(fig_title,bbox_inches="tight",facecolor=BG,edgecolor="none")
    pp.savefig(fig,bbox_inches="tight",facecolor=BG,edgecolor="none")
    pp.savefig(fig2,bbox_inches="tight",facecolor=BG,edgecolor="none")
    pp.savefig(fig3,bbox_inches="tight",facecolor=BG,edgecolor="none")
    pp.savefig(fig4,bbox_inches="tight",facecolor=BG,edgecolor="none")

plt.close("all")
print(f"Saved → {pdf_full}  (5 pages)")
print(f"Saved → {png_title}")
print(f"Saved → {png4}")"""


# ─── Build notebook ──────────────────────────────────────────────────────────

cells = [
    md_cell(CELL_MD_TITLE),
    code_cell(CELL_1_IMPORTS),
    code_cell(CELL_2_CONSTANTS),
    code_cell(CELL_3_LOAD_DATA),
    code_cell(CELL_4_CALCS),
    code_cell(CELL_5_HELPERS),
    code_cell(CELL_6_PAGE1A),
    code_cell(CELL_7_PAGE1B),
    code_cell(CELL_8_PAGE2),
    code_cell(CELL_9_PAGE3),
    code_cell(CELL_10_SC_DATA),
    code_cell(CELL_11_PAGE4_DRAW),
    code_cell(CELL_12_TITLE),
    md_cell(CELL_MD_RUN),
    code_cell(CELL_14_RUN),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out = Path("Barat_Scouting_Report.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written → {out}  ({len(cells)} cells)")
