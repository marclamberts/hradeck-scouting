"""
build_barat_report.py  —  A4 portrait, professional edition
4 benchmark levels · profile fit · KDE distributions · percentile bars
"""
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
from scipy.stats import percentileofscore, gaussian_kde, norm
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"],
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

ACCENT   = "#1D4ED8"    # deep blue (section titles / accent bar)
PLAYER_C = "#7C3AED"    # violet — player marker in distributions

CLUB_C   = "#7C3AED"    # violet
LEAGUE_C = "#2563EB"    # blue
TIER_C   = "#0F766E"    # teal
DB_C     = "#4B5563"    # dark grey

# Pool colour light variants (header backgrounds in table)
CLUB_BG   = "#F5F3FF"
LEAGUE_BG = "#EFF6FF"
TIER_BG   = "#F0FDFA"
DB_BG     = "#F3F4F6"

PERF_CMAP = LinearSegmentedColormap.from_list(
    "perf", [(0.0, "#DC2626"), (0.40, "#D97706"), (0.65, "#D97706"), (1.0, "#16A34A")]
)

def pct_colour(pct: float):
    return PERF_CMAP(np.clip(pct / 100, 0, 1))

# ── Config ─────────────────────────────────────────────────────────────────────
WIDE_ATK_POS = {"LAMF","RAMF","LW","RW","LWF","RWF","AMF","LWB","RWB"}
MIN_MINS     = 300

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
        "Successful defensive actions per 90": 0.25, "Offensive duels won, %": 0.25,
        "Dribbles per 90": 0.20, "Progressive runs per 90": 0.30,
    },
    "Unlocker": {
        "Key passes per 90": 0.25, "xA per 90": 0.25,
        "Shot assists per 90": 0.25, "Passes per 90": 0.15, "Accurate passes, %": 0.10,
    },
    "Outlet": {
        "Accurate passes, %": 0.30, "Passes per 90": 0.25,
        "Offensive duels won, %": 0.20, "Key passes per 90": 0.15, "xA per 90": 0.10,
    },
}

# ── Data ───────────────────────────────────────────────────────────────────────

def _numeric(df):
    for col in df.columns:
        if col not in NON_METRIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_data():
    print("  Czech files …")
    df_c1 = _numeric(pd.read_excel("data/Wyscout DB/Czech.xlsx"))
    df_c1["_file"] = "Czech"
    df_c1["_pos1"] = df_c1["Position"].astype(str).str.split(",").str[0].str.strip()

    player   = df_c1[df_c1["Player"].astype(str).str.startswith("D. Bar")].iloc[0].copy()
    team_kw  = str(player.get("Team within selected timeframe", "Slovácko")).split()[0]

    pool_lg  = df_c1[df_c1["_pos1"].isin(WIDE_ATK_POS) &
                     (df_c1["Minutes played"].fillna(0) >= MIN_MINS)].copy()
    pool_cl  = df_c1[df_c1["Team within selected timeframe"].astype(str)
                     .str.contains(team_kw, na=False) &
                     df_c1["_pos1"].isin(WIDE_ATK_POS)].copy()

    print("  Full database (165 files) …")
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
    mask_wa = df_all["_pos1"].isin(WIDE_ATK_POS) & (df_all["Minutes played"].fillna(0) >= MIN_MINS)

    pool_db   = df_all[mask_wa].copy().reset_index(drop=True)
    pool_tier = df_all[df_all["_file"].isin(TIER3_STEMS) & mask_wa].copy().reset_index(drop=True)

    return player, pool_cl, pool_lg, pool_tier, pool_db


def pct(player, pool, metrics):
    out = {}
    for m in metrics:
        if m not in pool.columns or m not in player.index or len(pool) == 0:
            out[m] = 50.0; continue
        vals = pool[m].dropna().values
        pv   = float(player.get(m, np.nan))
        out[m] = round(percentileofscore(vals, pv, kind="rank"), 1) \
                 if not np.isnan(pv) and len(vals) else 50.0
    return out


def role_fit(player, pool):
    scores = {}
    for role, bp in ROLE_BLUEPRINTS.items():
        avail = [(m, w) for m, w in bp.items() if m in pool.columns and m in player.index]
        if not avail:
            scores[role] = 50.0; continue
        tw = sum(w for _, w in avail)
        z  = sum((w / tw) * (float(player[m]) - pool[m].mean()) / (pool[m].std() or 1e-9)
                 for m, w in avail if not np.isnan(float(player.get(m, np.nan))))
        scores[role] = float(norm.cdf(z) * 100)
    return scores


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _off(ax, face=BG):
    ax.set_facecolor(face)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axis("off")


def _section_label(ax, title, note=""):
    """Draws ■ TITLE  ·  note  above the axes."""
    # Accent square
    ax.add_patch(mpatches.Rectangle(
        (0.0, 1.018), 0.014, 0.014,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))
    txt = f"  {title}"
    if note:
        txt += f"  ·  {note}"
    ax.text(0.016, 1.025, txt, ha="left", va="center",
            transform=ax.transAxes,
            color=TEXT_MED, fontsize=7, fontweight="bold")


# ── Header ─────────────────────────────────────────────────────────────────────

def draw_header(ax, n_lg, n_db):
    _off(ax)

    # Blue left accent bar
    ax.add_patch(mpatches.Rectangle(
        (-0.012, 0.0), 0.006, 1.0,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))

    # Player name
    ax.text(0.0, 0.97, "D. BARÁT", ha="left", va="top",
            transform=ax.transAxes, color=TEXT, fontsize=26, fontweight="bold")
    ax.text(0.0, 0.34,
            "Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Czech Republic",
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=8.5)

    # Bottom line
    ax.plot([0, 1], [0.03, 0.03], transform=ax.transAxes,
            color=BORDER, lw=0.8, clip_on=False)

    # Stat pills — thin vertical dividers between each
    pills = [("MIN","593"),("MATCHES","19"),("xG","0.93"),("xA","0.38"),("DRIB/90","4.55")]
    px = 0.50
    for j, (lbl, val) in enumerate(pills):
        if j > 0:  # divider
            ax.plot([px - 0.042, px - 0.042], [0.25, 0.90],
                    transform=ax.transAxes, color=BORDER, lw=0.7)
        ax.text(px, 0.93, val, ha="center", va="top",
                color="#B45309", fontsize=12, fontweight="bold",
                transform=ax.transAxes)
        ax.text(px, 0.28, lbl, ha="center", va="top",
                color=TEXT_DIM, fontsize=6, transform=ax.transAxes,
                fontweight="bold")
        px += 0.092

    # Pool note top-right
    ax.text(1.0, 0.97, f"League: {n_lg} players  ·  Database: {n_db:,}",
            ha="right", va="top", transform=ax.transAxes,
            color=TEXT_DIM, fontsize=7)


# ── Profile Fit ────────────────────────────────────────────────────────────────

def draw_profile_fit(ax, scores):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axis("off")

    _section_label(ax, "PROFILE FIT", "Attacker Role Archetypes  ·  The Athletic framework")

    # Sort descending
    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n    = len(sorted_roles)
    rows = np.linspace(0.88, 0.06, n)

    lbl_x  = 0.20     # right edge of role label
    bar_x0 = 0.22     # bar starts
    bar_x1 = 0.78     # bar ends
    pct_x  = 0.80     # score text left edge

    for i, (role, score) in enumerate(sorted_roles):
        ry  = rows[i]
        bh  = 0.09
        is_best = (i == 0)
        c   = pct_colour(score)
        bw  = bar_x1 - bar_x0

        # Role label
        ax.text(lbl_x, ry, role, ha="right", va="center",
                transform=ax.transAxes,
                color=TEXT if is_best else TEXT_MED,
                fontsize=7.5 if is_best else 7,
                fontweight="bold" if is_best else "normal")

        # Background track
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, ry - bh/2), bw, bh,
            facecolor=SURFACE2, edgecolor="none",
            transform=ax.transAxes, clip_on=False,
        ))
        # Fill
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, ry - bh/2), bw * score / 100, bh,
            facecolor=c, edgecolor="none", alpha=0.82,
            transform=ax.transAxes, clip_on=False,
        ))

        # Score text
        ax.text(pct_x, ry, f"{score:.0f}%", ha="left", va="center",
                transform=ax.transAxes, color=c,
                fontsize=8 if is_best else 7, fontweight="bold")

        # "PRIMARY ROLE" pill
        if is_best:
            px0 = pct_x + 0.08
            ax.add_patch(mpatches.FancyBboxPatch(
                (px0, ry - 0.052), 0.145, 0.104,
                boxstyle="round,pad=0.005",
                facecolor=PLAYER_C, edgecolor="none",
                transform=ax.transAxes, clip_on=False,
            ))
            ax.text(px0 + 0.073, ry, "PRIMARY ROLE", ha="center", va="center",
                    transform=ax.transAxes, color="white",
                    fontsize=5.5, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ── Distribution plots ─────────────────────────────────────────────────────────

def draw_distributions(axes, pool_db, pool_lg, player):
    _section_label(axes[0], "DISTRIBUTIONS",
                   "Database (grey)  vs  Czech First League (dashed blue)")

    for ax, (col, label) in zip(axes, DIST_METRICS):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_visible(False)

        vals_db = pool_db[col].dropna().values  if col in pool_db.columns  else np.array([])
        vals_lg = pool_lg[col].dropna().values  if col in pool_lg.columns  else np.array([])
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
                ax.plot(xs, kde_lg(xs), color=LEAGUE_C, linewidth=1.3,
                        linestyle="--", alpha=0.85)
            except Exception:
                pass

        if not np.isnan(pv):
            try:
                y_at = float(gaussian_kde(vals_db, bw_method=0.35)([pv])[0])
            except Exception:
                y_at = 0
            ax.axvline(pv, color=PLAYER_C, linewidth=1.8, zorder=5, alpha=0.9)
            ax.scatter([pv], [y_at], color=PLAYER_C, s=28, zorder=6, linewidths=0)
            p    = percentileofscore(vals_db, pv, kind="rank")
            peak = max(y_db) if len(y_db) else 1
            ax.text(pv, y_at + 0.05 * peak, f"{p:.0f}th",
                    ha="center", va="bottom",
                    color=pct_colour(p), fontsize=7, fontweight="bold")

        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)

        # Minimal x-ticks, no y-ticks
        n_ticks = 4
        ticks = np.linspace(np.ceil(lo * 2) / 2, np.floor(hi * 2) / 2, n_ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks],
                           fontsize=5.5, color=TEXT_DIM)
        ax.tick_params(axis="x", length=2, pad=1, colors=TEXT_DIM)
        ax.tick_params(axis="y", left=False, labelleft=False)

        # Metric label left
        ax.text(-0.04, 0.5, label, ha="right", va="center",
                transform=ax.transAxes, color=TEXT_MED,
                fontsize=7, fontweight="bold")

        # Baseline
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(BORDER)
        ax.spines["bottom"].set_linewidth(0.6)


# ── Percentile bars ─────────────────────────────────────────────────────────────

def draw_bars(ax, player, pc_cl, pc_lg, pc_ti, pc_db):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)

    _section_label(ax, "PERCENTILE RANKS",
                   "Bar = Database  ·  Lines: Club  League  Tier 3")

    n       = len(BAR_METRICS)
    spacing = 1.0
    bar_h   = 0.50
    prev_cat = None

    for i, (m, label, cat) in enumerate(BAR_METRICS):
        y      = (n - 1 - i) * spacing
        pd_val = pc_db.get(m, 50.0)
        col    = pct_colour(pd_val)

        raw = float(player.get(m, np.nan))
        raw_s = (f"{raw:.1f}%" if "%" in label else f"{raw:.2f}") if pd.notna(raw) else "—"

        if cat != prev_cat:
            if prev_cat is not None:
                sep_y = y + spacing * 0.72
                # Coloured short line in left margin + category name
                ax.plot([-38, -2], [sep_y]*2, color=CAT_COLOURS[cat], lw=1.0, alpha=0.8)
                ax.text(-20, sep_y + 0.08, cat.upper(), ha="center", va="bottom",
                        color=CAT_COLOURS[cat], fontsize=5.5, fontweight="bold")
                ax.plot([0, 100], [sep_y]*2, color=BORDER, lw=0.5)
            else:
                # Label the first category at the top
                ax.text(-20, y + spacing * 0.55, cat.upper(), ha="center", va="bottom",
                        color=CAT_COLOURS[cat], fontsize=5.5, fontweight="bold")
            prev_cat = cat

        # Metric label + raw value
        ax.text(-2, y + 0.15, label, ha="right", va="center",
                color=TEXT, fontsize=7.5)
        ax.text(-2, y - 0.20, raw_s, ha="right", va="center",
                color=TEXT_DIM, fontsize=6.5)

        # Grey track
        ax.barh(y, 100, height=bar_h, color=SURFACE2, left=0, lw=0, zorder=1, ec="none")
        # Gradient fill
        ax.barh(y, pd_val, height=bar_h, color=col, alpha=0.70,
                left=0, lw=0, zorder=2, ec="none")
        # Bright leading tip
        if pd_val > 5:
            ax.barh(y, 4, height=bar_h, color=col, alpha=1.0,
                    left=pd_val - 4, lw=0, zorder=3, ec="none")

        # Benchmark tick lines (club/league/tier)
        for bpct, bc in [(pc_cl.get(m,50), CLUB_C),
                         (pc_lg.get(m,50), LEAGUE_C),
                         (pc_ti.get(m,50), TIER_C)]:
            ax.plot([bpct, bpct], [y - bar_h*0.55, y + bar_h*0.55],
                    color=bc, lw=1.8, zorder=4)

        # Percentile number
        ax.text(102, y, f"{pd_val:.0f}th", ha="left", va="center",
                color=col, fontsize=8, fontweight="bold")

    # Reference gridlines + labels
    for v in (25, 50, 75):
        ax.plot([v, v], [-0.8, n * spacing - 0.2], color=BORDER, lw=0.6, ls=":", zorder=0)
        ax.text(v, n * spacing, str(v), ha="center", va="bottom",
                color=TEXT_DIM, fontsize=6)

    ax.set_xlim(-42, 120)
    ax.set_ylim(-0.8, n * spacing + 0.2)
    ax.axis("off")


# ── Benchmark table ─────────────────────────────────────────────────────────────

def draw_bench_table(ax, player, pc_cl, pc_lg, pc_ti, pc_db,
                     n_cl, n_lg, n_ti, n_db):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axis("off")

    _section_label(ax, "BENCHMARK SUMMARY",
                   "Percentile rank across all four reference pools")

    # Column config
    col_xs    = [0.01, 0.30, 0.45, 0.59, 0.73, 0.87]
    data_xs   = col_xs[1:]           # VALUE, CLUB, LEAGUE, TIER3, DB
    hdr_text  = [
        "VALUE",
        f"Club\n(n={n_cl})",
        f"League\n(n={n_lg})",
        f"Tier 3\n(n={n_ti:,})",
        f"DB\n(n={n_db:,})",
    ]
    hdr_cols  = [TEXT_DIM, CLUB_C, LEAGUE_C, TIER_C, DB_C]
    hdr_bgs   = [BG, CLUB_BG, LEAGUE_BG, TIER_BG, DB_BG]
    all_pcts  = [None, pc_cl, pc_lg, pc_ti, pc_db]

    row_ys    = np.linspace(0.82, 0.10, len(BENCH_METRICS) + 1)
    row_h     = abs(row_ys[1] - row_ys[0])

    # ── Header row ─────────────────────────────────────────────────────────────
    ax.text(col_xs[0], row_ys[0], "METRIC", ha="left", va="center",
            transform=ax.transAxes, color=TEXT_DIM, fontsize=6.5, fontweight="bold")

    for cx, txt, cc, bg in zip(data_xs, hdr_text, hdr_cols, hdr_bgs):
        w  = col_xs[data_xs.index(cx) + 2] - cx if cx != data_xs[-1] else 0.99 - cx
        ax.add_patch(mpatches.Rectangle(
            (cx - 0.005, row_ys[0] - 0.055), w, 0.115,
            facecolor=bg, edgecolor=BORDER, linewidth=0.5,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(cx + w/2 - 0.005, row_ys[0], txt, ha="center", va="center",
                transform=ax.transAxes, color=cc,
                fontsize=6.5, fontweight="bold", linespacing=1.3)

    # Header bottom border
    ax.plot([col_xs[0], 0.99], [row_ys[0] - 0.065]*2,
            transform=ax.transAxes, color=BORDER, lw=0.9)

    # ── Data rows ──────────────────────────────────────────────────────────────
    for ri, (m, lbl) in enumerate(BENCH_METRICS):
        ry = row_ys[ri + 1]

        # Alternating row background
        if ri % 2 == 0:
            ax.add_patch(mpatches.Rectangle(
                (col_xs[0] - 0.005, ry - row_h*0.52), 0.99, row_h,
                facecolor=SURFACE2, edgecolor="none",
                transform=ax.transAxes, clip_on=False,
            ))

        # Metric name
        ax.text(col_xs[0], ry, lbl, ha="left", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=7)

        # Value
        raw = float(player.get(m, np.nan))
        raw_s = f"{raw:.2f}" if pd.notna(raw) else "—"
        ax.text(data_xs[0] + 0.025, ry, raw_s, ha="center", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=7.5, fontweight="bold")

        # Percentile cells
        for ci, pdict in enumerate(all_pcts[1:], 1):
            cx  = data_xs[ci]
            w   = (data_xs[ci] - data_xs[ci-1]) if ci > 0 else 0.1
            p   = pdict.get(m, 50.0)
            c   = pct_colour(p)
            ax.text(cx + w/2 - 0.005, ry, f"{p:.0f}th", ha="center", va="center",
                    transform=ax.transAxes, color=c, fontsize=7.5, fontweight="bold")

        # Row bottom border
        ax.plot([col_xs[0] - 0.005, 0.99], [ry - row_h*0.52]*2,
                transform=ax.transAxes, color=BORDER, lw=0.4)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading …")
    player, pool_cl, pool_lg, pool_tier, pool_db = load_data()
    n_cl = len(pool_cl); n_lg = len(pool_lg)
    n_ti = len(pool_tier); n_db = len(pool_db)
    print(f"  Club={n_cl}  League={n_lg}  Tier3={n_ti}  DB={n_db}")

    keys = list({m for m, *_ in BAR_METRICS} | {m for m, _ in BENCH_METRICS})
    pc_cl = pct(player, pool_cl,   keys)
    pc_lg = pct(player, pool_lg,   keys)
    pc_ti = pct(player, pool_tier, keys)
    pc_db = pct(player, pool_db,   keys)

    scores = role_fit(player, pool_lg)
    print(f"  Roles: { {r: f'{v:.0f}%' for r,v in scores.items()} }")

    # ── Figure (A4 portrait) ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)

    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        left=0.04, right=0.97,
        top=0.975, bottom=0.020,
        height_ratios=[0.082, 0.125, 0.608, 0.168],
        hspace=0.14,
    )

    draw_header(fig.add_subplot(outer[0]), n_lg, n_db)
    draw_profile_fit(fig.add_subplot(outer[1]), scores)

    # Main: distributions (left) | bars (right)
    mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        wspace=0.06, width_ratios=[0.37, 0.63],
    )

    # 4 stacked distribution plots
    d_gs   = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=mid[0], hspace=0.65)
    d_axes = [fig.add_subplot(d_gs[i]) for i in range(4)]
    draw_distributions(d_axes, pool_db, pool_lg, player)

    # Distribution legend
    d_axes[-1].legend(
        handles=[
            Line2D([0],[0], color="#94A3B8", lw=1.3, label="Full database"),
            Line2D([0],[0], color=LEAGUE_C,  lw=1.3, ls="--",
                   label="Czech First League"),
            Line2D([0],[0], color=PLAYER_C,  lw=1.8, label="D. Barát"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.92),
        ncol=3, frameon=False, fontsize=5.8,
        labelcolor=TEXT_DIM, handlelength=1.2,
    )

    # Percentile bars
    ax_bars = fig.add_subplot(mid[1])
    draw_bars(ax_bars, player, pc_cl, pc_lg, pc_ti, pc_db)

    # Two legends for bars
    leg_cat = ax_bars.legend(
        handles=[mpatches.Patch(facecolor=c, label=cat, alpha=0.85)
                 for cat, c in CAT_COLOURS.items()],
        loc="lower right", bbox_to_anchor=(1.01, -0.01),
        frameon=True, facecolor=PANEL, edgecolor=BORDER,
        fontsize=5.5, labelcolor=TEXT, handlelength=0.8, ncol=1,
        title="Category", title_fontsize=5.5,
    )
    ax_bars.add_artist(leg_cat)
    ax_bars.legend(
        handles=[
            Line2D([0],[0], color=CLUB_C,   lw=1.6, label=f"Club  (n={n_cl})"),
            Line2D([0],[0], color=LEAGUE_C, lw=1.6, label=f"League  (n={n_lg})"),
            Line2D([0],[0], color=TIER_C,   lw=1.6, label=f"Tier 3  (n={n_ti:,})"),
        ],
        loc="upper right", bbox_to_anchor=(1.01, 1.05),
        frameon=True, facecolor=PANEL, edgecolor=BORDER,
        fontsize=5.5, labelcolor=TEXT, handlelength=1.2, ncol=1,
        title="Benchmark lines", title_fontsize=5.5,
    )

    # Benchmark table
    draw_bench_table(
        fig.add_subplot(outer[3]),
        player, pc_cl, pc_lg, pc_ti, pc_db,
        n_cl, n_lg, n_ti, n_db,
    )

    # Footer
    ax_foot = fig.add_axes([0.04, 0.005, 0.93, 0.012])
    _off(ax_foot)
    ax_foot.plot([0, 1], [0.9, 0.9], transform=ax_foot.transAxes, color=BORDER, lw=0.7)
    ax_foot.text(0.0, 0.0, "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
                 ha="left", va="bottom", transform=ax_foot.transAxes,
                 color=TEXT_DIM, fontsize=6.5)
    ax_foot.text(1.0, 0.0, "hradeck-scouting",
                 ha="right", va="bottom", transform=ax_foot.transAxes,
                 color=TEXT_DIM, fontsize=6.5)

    # Save
    png = OUT_DIR / "D_Barat_Scouting_Report.png"
    pdf = OUT_DIR / "D_Barat_Scouting_Report.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig.savefig(pdf, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {png}")
    print(f"  Saved → {pdf}")


if __name__ == "__main__":
    main()
