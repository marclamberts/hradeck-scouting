"""
build_barat_report.py  —  A4 portrait, professional edition v3
All section labels drawn INSIDE axes · two-pass bars · fixed table columns
Page 2: peer comparison + full statistical profile
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

CLUB_BG   = "#F5F3FF"
LEAGUE_BG = "#EFF6FF"
TIER_BG   = "#F0FDFA"
DB_BG     = "#F3F4F6"

PERF_CMAP = LinearSegmentedColormap.from_list(
    "perf", [(0.0, "#DC2626"), (0.40, "#D97706"), (0.65, "#D97706"), (1.0, "#16A34A")]
)
def pct_colour(pct):
    return PERF_CMAP(np.clip(pct / 100, 0, 1))

# ── Config ─────────────────────────────────────────────────────────────────────
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

# ── Player config (monkey-patchable for multi-player reports) ──────────────────
P_HEADER_NAME     = "D. BARÁT"
P_HEADER_SUBTITLE = ("Slovácko  ·  Czech Fortuna Liga 2025/26 (Final)  ·  "
                     "LAMF / LW / LWB  ·  Age 19  ·  Czech Republic")
P_PAGE2_SUBTITLE  = ("Slovácko  ·  Czech Fortuna Liga  ·  "
                     "LAMF / LW / LWB  ·  Age 19  ·  Page 2 of 2")
P_PAGE3_SUBTITLE  = ("Slovácko  ·  Czech Fortuna Liga  ·  "
                     "LAMF / LW / LWB  ·  Age 19  ·  Page 3 of 3")
P_PILLS           = [("MIN","593"),("MATCHES","19"),("xG","0.93"),("xA","0.38"),("DRIB/90","4.55")]
P_WYSCOUT_FILTER  = "D. Bar"
P_LEGEND_LABEL    = "D. Barát"

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

    player  = df_c1[df_c1["Player"].astype(str).str.startswith(P_WYSCOUT_FILTER)].iloc[0].copy()
    team_kw = str(player.get("Team within selected timeframe", "Slovácko")).split()[0]

    pool_lg = df_c1[df_c1["_pos1"].isin(WIDE_ATK_POS) &
                    (df_c1["Minutes played"].fillna(0) >= MIN_MINS)].copy()
    pool_cl = df_c1[df_c1["Team within selected timeframe"].astype(str)
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
    wa = df_all["_pos1"].isin(WIDE_ATK_POS) & (df_all["Minutes played"].fillna(0) >= MIN_MINS)
    pool_db   = df_all[wa].copy().reset_index(drop=True)
    pool_tier = df_all[df_all["_file"].isin(TIER3_STEMS) & wa].copy().reset_index(drop=True)

    return player, pool_cl, pool_lg, pool_tier, pool_db


def calc_pcts(player, pool, metrics):
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
        z  = sum((w/tw) * (float(player[m]) - pool[m].mean()) / (pool[m].std() or 1e-9)
                 for m, w in avail if not np.isnan(float(player.get(m, np.nan))))
        scores[role] = float(norm.cdf(z) * 100)
    return scores


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _off(ax, face=BG):
    ax.set_facecolor(face)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")


def _inner_title(ax, title, note=""):
    """Section title drawn INSIDE the top of the axes — no overlap risk."""
    full = f"{title}  ·  {note}" if note else title
    # Thin coloured strip across top
    ax.add_patch(mpatches.Rectangle(
        (0, 0.945), 1.0, 0.055,
        facecolor=SURFACE2, edgecolor="none",
        transform=ax.transAxes, clip_on=True, zorder=10,
    ))
    # Left accent bar
    ax.add_patch(mpatches.Rectangle(
        (0, 0.945), 0.004, 0.055,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=True, zorder=11,
    ))
    ax.text(0.012, 0.973, full, ha="left", va="center",
            transform=ax.transAxes, color=TEXT_MED,
            fontsize=6.5, fontweight="bold", zorder=12)


# ── Header ─────────────────────────────────────────────────────────────────────

def draw_header(ax, n_lg, n_db):
    _off(ax)
    # Left accent bar
    ax.add_patch(mpatches.Rectangle(
        (-0.018, 0.0), 0.007, 1.0,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(0.0, 0.97, P_HEADER_NAME, ha="left", va="top",
            transform=ax.transAxes, color=TEXT, fontsize=26, fontweight="bold")
    ax.text(0.0, 0.33, P_HEADER_SUBTITLE,
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=8.5)
    ax.plot([0, 1], [0.04, 0.04], transform=ax.transAxes, color=BORDER, lw=0.8)

    pills = P_PILLS
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


# ── Profile Fit ────────────────────────────────────────────────────────────────

def draw_profile_fit(ax, scores):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")

    _inner_title(ax, "PROFILE FIT", "Attacker Role Archetypes  ·  The Athletic framework")

    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n    = len(sorted_roles)
    rows = np.linspace(0.86, 0.06, n)  # top=0.86 leaves room for inner title

    # Layout columns (all in axes fraction, sum ≤ 1.0)
    lbl_x  = 0.195   # right edge of role name
    bar_x0 = 0.210   # bar start
    bar_x1 = 0.740   # bar end
    pct_x  = 0.755   # pct text left
    badge_x0 = 0.840  # badge left

    for i, (role, score) in enumerate(sorted_roles):
        ry    = rows[i]
        bh    = 0.095
        is_b  = (i == 0)
        c     = pct_colour(score)

        ax.text(lbl_x, ry, role, ha="right", va="center",
                transform=ax.transAxes,
                color=TEXT if is_b else TEXT_MED,
                fontsize=7.5 if is_b else 7,
                fontweight="bold" if is_b else "normal")

        bw = bar_x1 - bar_x0
        # Track
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, ry - bh/2), bw, bh,
            facecolor=SURFACE2, edgecolor="none",
            transform=ax.transAxes, clip_on=True))
        # Fill
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, ry - bh/2), bw * score / 100, bh,
            facecolor=c, edgecolor="none", alpha=0.82,
            transform=ax.transAxes, clip_on=True))

        ax.text(pct_x, ry, f"{score:.0f}%", ha="left", va="center",
                transform=ax.transAxes, color=c,
                fontsize=8 if is_b else 7, fontweight="bold")

        if is_b:  # PRIMARY ROLE badge
            bw2 = 0.145
            ax.add_patch(mpatches.FancyBboxPatch(
                (badge_x0, ry - 0.050), bw2, 0.100,
                boxstyle="round,pad=0.004",
                facecolor=PLAYER_C, edgecolor="none",
                transform=ax.transAxes, clip_on=True))
            ax.text(badge_x0 + bw2/2, ry, "PRIMARY ROLE",
                    ha="center", va="center",
                    transform=ax.transAxes, color="white",
                    fontsize=5.5, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ── Distribution plots ─────────────────────────────────────────────────────────

def draw_distributions(axes, pool_db, pool_lg, player):
    # Title only on the first (top) axes
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
                ax.plot(xs, kde_lg(xs), color=LEAGUE_C, linewidth=1.3,
                        linestyle="--", alpha=0.85)
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
                    ha="center", va="bottom",
                    color=pct_colour(p), fontsize=7, fontweight="bold")

        # Leave extra headroom at top for the section title on axes[0]
        headroom = 0.28 if idx == 0 else 0.15
        peak = float(np.max(y_db)) if len(y_db) and np.max(y_db) > 0 else 1.0
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, peak * (1 + headroom))

        ticks = np.linspace(
            np.ceil(lo * 2) / 2, np.floor(hi * 2) / 2, 4
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks], fontsize=5.5, color=TEXT_DIM)
        ax.tick_params(axis="x", length=2, pad=1, colors=TEXT_DIM)
        ax.tick_params(axis="y", left=False, labelleft=False)

        ax.text(-0.05, 0.5, label, ha="right", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=7, fontweight="bold")

        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(BORDER)
        ax.spines["bottom"].set_linewidth(0.6)


# ── Percentile bars ─────────────────────────────────────────────────────────────

def draw_bars(ax, player, pc_cl, pc_lg, pc_ti, pc_db):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)

    n       = len(BAR_METRICS)
    spacing = 1.0
    bar_h   = 0.44   # slightly thinner → more gap between categories

    _inner_title(ax, "PERCENTILE RANKS",
                 "Bar = Database  ·  Lines: Club  League  Tier 3")

    # ── Pass 1: category dividers (drawn first so bars sit on top) ──────────
    prev_cat = None
    for i, (m, _lbl, cat) in enumerate(BAR_METRICS):
        y = (n - 1 - i) * spacing
        if cat != prev_cat:
            if prev_cat is None:
                # Label above first bar (top of chart)
                ax.text(-21, y + spacing * 0.70, cat.upper(),
                        ha="center", va="center",
                        color=CAT_COLOURS[cat], fontsize=5.5, fontweight="bold")
            else:
                # Separator at midpoint of inter-group gap
                sep_y = y + spacing * 0.50
                ax.plot([-44, 100], [sep_y]*2, color=BORDER, lw=0.5, zorder=0)
                # Category label sits ON the line with white cutout
                ax.text(-21, sep_y, cat.upper(), ha="center", va="center",
                        color=CAT_COLOURS[cat], fontsize=5.5, fontweight="bold",
                        bbox=dict(facecolor=BG, edgecolor="none", pad=1.8),
                        zorder=2)
            prev_cat = cat

    # ── Pass 2: bars ────────────────────────────────────────────────────────
    for i, (m, label, cat) in enumerate(BAR_METRICS):
        y      = (n - 1 - i) * spacing
        pd_val = pc_db.get(m, 50.0)
        col    = pct_colour(pd_val)

        raw   = float(player.get(m, np.nan))
        raw_s = (f"{raw:.1f}%" if "%" in label else f"{raw:.2f}") \
                if pd.notna(raw) else "—"

        # Label + raw value
        ax.text(-2, y + 0.13, label, ha="right", va="center",
                color=TEXT, fontsize=7.5)
        ax.text(-2, y - 0.18, raw_s, ha="right", va="center",
                color=TEXT_DIM, fontsize=6.5)

        # Grey track
        ax.barh(y, 100, height=bar_h, color=SURFACE2, left=0, lw=0, zorder=1, ec="none")
        # Gradient fill
        ax.barh(y, pd_val, height=bar_h, color=col, alpha=0.72,
                left=0, lw=0, zorder=2, ec="none")
        # Bright leading tip
        if pd_val > 5:
            ax.barh(y, 4, height=bar_h, color=col, alpha=1.0,
                    left=pd_val - 4, lw=0, zorder=3, ec="none")

        # Benchmark tick lines (Club / League / Tier)
        for bpct, bc in [(pc_cl.get(m,50), CLUB_C),
                         (pc_lg.get(m,50), LEAGUE_C),
                         (pc_ti.get(m,50), TIER_C)]:
            ax.plot([bpct, bpct], [y - bar_h*0.58, y + bar_h*0.58],
                    color=bc, lw=1.8, zorder=4)

        # Percentile number
        ax.text(102, y, f"{pd_val:.0f}th", ha="left", va="center",
                color=col, fontsize=8, fontweight="bold")

    # Reference gridlines
    top = n * spacing + 0.55
    for v in (25, 50, 75):
        ax.plot([v, v], [-0.6, top - 0.25], color=BORDER, lw=0.6, ls=":", zorder=0)
        ax.text(v, top - 0.18, str(v), ha="center", va="bottom",
                color=TEXT_DIM, fontsize=6)

    ax.set_xlim(-44, 120)
    ax.set_ylim(-0.6, top)
    ax.axis("off")


# ── Benchmark table ─────────────────────────────────────────────────────────────

# Column boundaries (left edge of each column, right edge = next column's left)
_COL = [0.01, 0.285, 0.415, 0.555, 0.700, 0.845, 0.99]
#        metric  value   club  league  tier   db    end

def _col_cx(ci):
    """Centre x of column ci (1-indexed into _COL)."""
    return (_COL[ci] + _COL[ci + 1]) / 2

def draw_bench_table(ax, player, pc_cl, pc_lg, pc_ti, pc_db,
                     n_cl, n_lg, n_ti, n_db):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")

    _inner_title(ax, "BENCHMARK SUMMARY",
                 "Percentile rank across four reference pools")

    # Column specs: (label, colour, bg_colour, col_index_in_COL)
    hdr_specs = [
        ("VALUE",              TEXT_DIM, BG,        1),
        (f"Club\n(n={n_cl})", CLUB_C,   CLUB_BG,   2),
        (f"Lg.\n(n={n_lg})",  LEAGUE_C, LEAGUE_BG, 3),
        (f"Tier 3\n(n={n_ti:,})", TIER_C, TIER_BG, 4),
        (f"DB\n(n={n_db:,})", DB_C,     DB_BG,     5),
    ]
    all_pcts = [None, pc_cl, pc_lg, pc_ti, pc_db]

    # Row y-positions: leave top 8% for inner title, spread rest
    row_ys = np.linspace(0.83, 0.06, len(BENCH_METRICS) + 1)
    rh = abs(row_ys[0] - row_ys[1])

    # ── Header row ──────────────────────────────────────────────────────────
    ax.text(_COL[0], row_ys[0], "METRIC", ha="left", va="center",
            transform=ax.transAxes, color=TEXT_DIM, fontsize=6.5, fontweight="bold")

    for txt, cc, bg, ci in hdr_specs:
        x0 = _COL[ci]; x1 = _COL[ci + 1]
        ax.add_patch(mpatches.Rectangle(
            (x0, row_ys[0] - rh * 0.48), x1 - x0, rh,
            facecolor=bg, edgecolor=BORDER, linewidth=0.5,
            transform=ax.transAxes, clip_on=True,
        ))
        ax.text(_col_cx(ci), row_ys[0], txt, ha="center", va="center",
                transform=ax.transAxes, color=cc,
                fontsize=6.5, fontweight="bold", linespacing=1.3)

    ax.plot([_COL[0], _COL[-1]], [row_ys[0] - rh*0.50]*2,
            transform=ax.transAxes, color=BORDER, lw=0.9)

    # ── Data rows ──────────────────────────────────────────────────────────
    for ri, (m, lbl) in enumerate(BENCH_METRICS):
        ry = row_ys[ri + 1]

        if ri % 2 == 0:
            ax.add_patch(mpatches.Rectangle(
                (_COL[0], ry - rh*0.50), _COL[-1] - _COL[0], rh,
                facecolor=SURFACE2, edgecolor="none",
                transform=ax.transAxes, clip_on=True))

        ax.text(_COL[0], ry, lbl, ha="left", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=7)

        raw   = float(player.get(m, np.nan))
        raw_s = f"{raw:.2f}" if pd.notna(raw) else "—"
        ax.text(_col_cx(1), ry, raw_s, ha="center", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=7.5, fontweight="bold")

        for ci, pdict in enumerate(all_pcts[1:], 2):
            p = pdict.get(m, 50.0)
            ax.text(_col_cx(ci), ry, f"{p:.0f}th", ha="center", va="center",
                    transform=ax.transAxes, color=pct_colour(p),
                    fontsize=7.5, fontweight="bold")

        ax.plot([_COL[0], _COL[-1]], [ry - rh*0.50]*2,
                transform=ax.transAxes, color=BORDER, lw=0.4)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ── Page 2 constants ───────────────────────────────────────────────────────────

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
]


# ── Page 2 helpers ─────────────────────────────────────────────────────────────

def find_peers(player, pool_lg, n=8):
    """Return n most similar players in pool_lg (excl. the player) by z-score dist."""
    sim_cols = [m for m, *_ in BAR_METRICS if m in pool_lg.columns]
    pool = pool_lg.copy()
    # exclude the player themselves
    player_name = str(player.get("Player", ""))
    pool = pool[~pool["Player"].astype(str).str.startswith(P_WYSCOUT_FILTER)]

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
    return pool.nsmallest(n, "_dist").reset_index(drop=True)


def draw_page2_header(ax, player):
    _off(ax)
    ax.add_patch(mpatches.Rectangle(
        (-0.018, 0.0), 0.007, 1.0,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(0.0, 0.97, P_HEADER_NAME, ha="left", va="top",
            transform=ax.transAxes, color=TEXT, fontsize=22, fontweight="bold")
    ax.text(0.0, 0.30, P_PAGE2_SUBTITLE,
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=8)
    ax.plot([0, 1], [0.04, 0.04], transform=ax.transAxes, color=BORDER, lw=0.8)


def draw_peers(ax, player, pool_lg, pc_lg):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")

    _inner_title(ax, "PEER COMPARISON",
                 "Top 8 most similar wide attackers · Czech Fortuna Liga · z-score distance")

    peers = find_peers(player, pool_lg, n=8)

    # Build display header
    fixed_hdrs = ["PLAYER", "TEAM", "AGE", "MIN"]
    metric_hdrs = [lbl for _, lbl in PEER_TABLE_COLS]
    all_hdrs = fixed_hdrs + metric_hdrs

    # Column x-positions (fractions)
    col_xs = [0.00, 0.24, 0.38, 0.44, 0.50] + \
             list(np.linspace(0.56, 0.99, len(metric_hdrs) + 1)[:-1])
    col_xs.append(1.00)  # right edge sentinel

    n_rows = 1 + len(peers)   # header + peers
    # add player row at top
    rows_y = np.linspace(0.86, 0.04, n_rows + 1)
    rh = abs(rows_y[0] - rows_y[1])

    def col_cx(ci):
        return (col_xs[ci] + col_xs[ci + 1]) / 2

    # ── Header row ──────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (col_xs[0], rows_y[0] - rh * 0.48),
        col_xs[-1] - col_xs[0], rh,
        facecolor=SURFACE2, edgecolor="none",
        transform=ax.transAxes, clip_on=True))

    for ci, hdr in enumerate(all_hdrs):
        ax.text(col_xs[ci] if ci < 2 else col_cx(ci), rows_y[0],
                hdr, ha="left" if ci < 2 else "center", va="center",
                transform=ax.transAxes, color=TEXT_DIM,
                fontsize=5.5, fontweight="bold")

    ax.plot([col_xs[0], col_xs[-1]], [rows_y[0] - rh * 0.50] * 2,
            transform=ax.transAxes, color=BORDER, lw=0.8)

    # ── Barát row (highlighted) ─────────────────────────────────────────────
    ry = rows_y[1]
    ax.add_patch(mpatches.Rectangle(
        (col_xs[0], ry - rh * 0.48),
        col_xs[-1] - col_xs[0], rh,
        facecolor="#EEF2FF", edgecolor=ACCENT, linewidth=0.6,
        transform=ax.transAxes, clip_on=True))

    p_name = str(player.get("Player", "D. Barát"))
    p_team = str(player.get("Team within selected timeframe", "Slovácko"))
    p_age  = str(int(player.get("Age", 19))) if pd.notna(player.get("Age", np.nan)) else "—"
    p_min  = str(int(player.get("Minutes played", 593))) \
             if pd.notna(player.get("Minutes played", np.nan)) else "—"

    fixed_vals = [p_name, p_team, p_age, p_min]
    for ci, v in enumerate(fixed_vals):
        ax.text(col_xs[ci] + 0.005, ry, v,
                ha="left", va="center",
                transform=ax.transAxes, color=TEXT,
                fontsize=6.5 if ci == 0 else 6, fontweight="bold" if ci == 0 else "normal")

    for mi, (col, _lbl) in enumerate(PEER_TABLE_COLS):
        ci = 4 + mi
        raw = float(player.get(col, np.nan))
        txt = (f"{raw:.1f}%" if "%" in _lbl else f"{raw:.2f}") if pd.notna(raw) else "—"
        c   = pct_colour(pc_lg.get(col, 50.0))
        ax.text(col_cx(ci), ry, txt,
                ha="center", va="center",
                transform=ax.transAxes, color=c,
                fontsize=6.5, fontweight="bold")

    ax.plot([col_xs[0], col_xs[-1]], [ry - rh * 0.50] * 2,
            transform=ax.transAxes, color=BORDER, lw=0.5)

    # ── Peer rows ───────────────────────────────────────────────────────────
    for ri, (_, peer) in enumerate(peers.iterrows()):
        ry = rows_y[ri + 2]
        if (ri + 1) % 2 == 0:
            ax.add_patch(mpatches.Rectangle(
                (col_xs[0], ry - rh * 0.48),
                col_xs[-1] - col_xs[0], rh,
                facecolor=SURFACE2, edgecolor="none",
                transform=ax.transAxes, clip_on=True))

        nm  = str(peer.get("Player", ""))[:22]
        tm  = str(peer.get("Team within selected timeframe", ""))[:14]
        age = str(int(peer.get("Age", 0))) if pd.notna(peer.get("Age", np.nan)) else "—"
        mn  = str(int(peer.get("Minutes played", 0))) \
              if pd.notna(peer.get("Minutes played", np.nan)) else "—"

        for ci, v in enumerate([nm, tm, age, mn]):
            ax.text(col_xs[ci] + 0.005, ry, v,
                    ha="left", va="center",
                    transform=ax.transAxes, color=TEXT_MED,
                    fontsize=6 if ci == 0 else 5.8)

        for mi, (col, _lbl) in enumerate(PEER_TABLE_COLS):
            ci = 4 + mi
            raw = float(peer.get(col, np.nan))
            txt = (f"{raw:.1f}%" if "%" in _lbl else f"{raw:.2f}") if pd.notna(raw) else "—"
            # colour relative to league pool
            vals_lg = pool_lg[col].dropna().values if col in pool_lg.columns else np.array([])
            p_score = percentileofscore(vals_lg, raw, kind="rank") if (len(vals_lg) and pd.notna(raw)) else 50.0
            ax.text(col_cx(ci), ry, txt,
                    ha="center", va="center",
                    transform=ax.transAxes, color=pct_colour(p_score),
                    fontsize=5.8)

        ax.plot([col_xs[0], col_xs[-1]], [ry - rh * 0.50] * 2,
                transform=ax.transAxes, color=BORDER, lw=0.3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def draw_stats_col(ax, player, pool_db, stats, title_shown=False):
    """Draw one column of the full statistical profile."""
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")

    if title_shown:
        _inner_title(ax, "FULL STATISTICAL PROFILE",
                     "Value  ·  mini-bar coloured by database percentile")

    n    = len(stats)
    rows = np.linspace(0.90 if title_shown else 0.97, 0.02, n)
    rh   = abs(rows[0] - rows[1]) if n > 1 else 0.06

    dot_x   = 0.03
    lbl_x   = 0.08
    val_x   = 0.52
    bar_x0  = 0.58
    bar_x1  = 0.86
    pct_x   = 0.90

    for i, (col, lbl, cat) in enumerate(stats):
        ry = rows[i]
        c_cat = CAT_COLOURS.get(cat, TEXT_DIM)

        if i % 2 == 0:
            ax.add_patch(mpatches.Rectangle(
                (0, ry - rh * 0.48), 1.0, rh,
                facecolor=SURFACE2, edgecolor="none",
                transform=ax.transAxes, clip_on=True))

        # Category dot
        ax.add_patch(mpatches.Circle(
            (dot_x, ry), rh * 0.22,
            facecolor=c_cat, edgecolor="none",
            transform=ax.transAxes, clip_on=True))

        # Label
        ax.text(lbl_x, ry, lbl, ha="left", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=6.0)

        # Value
        raw = float(player.get(col, np.nan)) if col in player.index else np.nan
        raw_s = (f"{raw:.1f}%" if col.endswith(", %") or col.endswith(", %")
                 else f"{raw:.2f}") if pd.notna(raw) else "—"
        ax.text(val_x, ry, raw_s, ha="right", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=6.5, fontweight="bold")

        # Mini gradient bar
        vals_db = pool_db[col].dropna().values if col in pool_db.columns else np.array([])
        p_score = percentileofscore(vals_db, raw, kind="rank") \
                  if (len(vals_db) and pd.notna(raw)) else 50.0
        bar_c = pct_colour(p_score)
        bw = bar_x1 - bar_x0
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, ry - rh * 0.22), bw, rh * 0.44,
            facecolor=SURFACE2, edgecolor="none",
            transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, ry - rh * 0.22), bw * p_score / 100, rh * 0.44,
            facecolor=bar_c, edgecolor="none", alpha=0.78,
            transform=ax.transAxes, clip_on=True))

        # Percentile text
        ax.text(pct_x, ry, f"{p_score:.0f}th", ha="left", va="center",
                transform=ax.transAxes, color=bar_c, fontsize=6.0, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def make_page2(player, pool_cl, pool_lg, pool_tier, pool_db,
               pc_cl, pc_lg, pc_ti, pc_db):
    fig2 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)

    outer2 = gridspec.GridSpec(
        3, 1, figure=fig2,
        left=0.04, right=0.97,
        top=0.975, bottom=0.022,
        height_ratios=[0.055, 0.375, 0.535],
        hspace=0.18,
    )

    draw_page2_header(fig2.add_subplot(outer2[0]), player)
    draw_peers(fig2.add_subplot(outer2[1]), player, pool_lg, pc_lg)

    # Full stats — 2 columns
    stats_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer2[2], wspace=0.10,
    )
    draw_stats_col(fig2.add_subplot(stats_gs[0]), player, pool_db,
                   ALL_STATS_LEFT, title_shown=True)
    draw_stats_col(fig2.add_subplot(stats_gs[1]), player, pool_db,
                   ALL_STATS_RIGHT, title_shown=False)

    # Footer
    fig2.text(0.04, 0.007,
              "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
              ha="left", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig2.text(0.97, 0.007, "hradeck-scouting",
              ha="right", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig2.add_artist(plt.Line2D(
        [0.04, 0.97], [0.019, 0.019],
        transform=fig2.transFigure, color=BORDER, lw=0.7,
    ))

    return fig2


# ── Page 3: composite & derived metrics ────────────────────────────────────────

def _g(row, col):
    """Safe float getter from Series or dict."""
    v = row.get(col, np.nan) if hasattr(row, "get") else np.nan
    return float(v) if pd.notna(v) else np.nan


def _composite_carrying(r):
    d = _g(r, "Dribbles per 90"); dp = _g(r, "Successful dribbles, %")
    pr = _g(r, "Progressive runs per 90"); ac = _g(r, "Accelerations per 90")
    parts = []
    if pd.notna(d) and pd.notna(dp): parts.append(d * dp / 100)
    if pd.notna(pr): parts.append(pr)
    if pd.notna(ac): parts.append(ac * 0.5)
    return sum(parts) if parts else np.nan

def _composite_threat(r):
    xg = _g(r, "xG per 90"); sh = _g(r, "Shots per 90"); bx = _g(r, "Touches in box per 90")
    parts = []
    if pd.notna(xg): parts.append(xg * 2.0)
    if pd.notna(sh): parts.append(sh * 0.3)
    if pd.notna(bx): parts.append(bx * 0.2)
    return sum(parts) if parts else np.nan

def _composite_creation(r):
    xa = _g(r, "xA per 90"); sa = _g(r, "Shot assists per 90")
    kp = _g(r, "Key passes per 90"); cr = _g(r, "Crosses per 90")
    ca = _g(r, "Accurate crosses, %")
    parts = []
    if pd.notna(xa): parts.append(xa * 2.0)
    if pd.notna(sa): parts.append(sa * 0.8)
    if pd.notna(kp): parts.append(kp * 0.5)
    if pd.notna(cr) and pd.notna(ca): parts.append(cr * ca / 100)
    return sum(parts) if parts else np.nan

def _composite_duels(r):
    od = _g(r, "Offensive duels won, %"); dd = _g(r, "Defensive duels won, %")
    ae = _g(r, "Aerial duels won, %")
    vals = [v for v in [od, dd, ae * 0.7 if pd.notna(ae) else np.nan] if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def _composite_defwork(r):
    da = _g(r, "Successful defensive actions per 90"); ic = _g(r, "Interceptions per 90")
    parts = [v for v in [da, ic] if pd.notna(v)]
    return sum(parts) if parts else np.nan

def _composite_war(r):
    ca = _composite_carrying(r); th = _composite_threat(r)
    cr = _composite_creation(r); du = _composite_duels(r); dw = _composite_defwork(r)
    # normalise duels (already in %-space ~50) by /100 to bring to ~0.5 range
    du_n = du / 100 if pd.notna(du) else np.nan
    parts = [(ca, 0.30), (th, 0.25), (cr, 0.30), (du_n, 0.10), (dw, 0.05)]
    valid = [(v, w) for v, w in parts if pd.notna(v)]
    if not valid: return np.nan
    tw = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / tw


# Each entry: (display_name, subtitle/formula, cat_colour_key, fn)
COMPOSITE_SPECS = [
    ("Wide Attacker\nRating",  "Carrying 30 · Threat 25 · Creation 30 · Duels 10 · Def 5",
     "Carrying",  _composite_war),
    ("Carrying\nPower",        "Eff. Dribbles + Prog. Runs + 0.5×Accelerations",
     "Carrying",  _composite_carrying),
    ("Goal Threat\nIndex",     "2×xG/90  +  0.3×Shots/90  +  0.2×BoxTouch/90",
     "Threat",    _composite_threat),
    ("Creative\nDanger",       "2×xA  +  0.8×ShotAssist  +  0.5×KP  +  AccCrosses",
     "Creation",  _composite_creation),
    ("Duel\nAuthority",        "Mean of Off%, Def%, 0.7×Aerial% duel win rates",
     "Duels",     _composite_duels),
    ("Defensive\nWork",        "Def. Actions/90  +  Interceptions/90",
     "Defending", _composite_defwork),
]


def _derived_xg_per_shot(r):
    xg = _g(r, "xG per 90"); sh = _g(r, "Shots per 90")
    return xg / sh if (pd.notna(xg) and pd.notna(sh) and sh > 0) else np.nan

def _derived_g_minus_xg(r):
    g = _g(r, "Goals per 90"); xg = _g(r, "xG per 90")
    return g - xg if (pd.notna(g) and pd.notna(xg)) else np.nan

def _derived_eff_dribbles(r):
    d = _g(r, "Dribbles per 90"); dp = _g(r, "Successful dribbles, %")
    return d * dp / 100 if (pd.notna(d) and pd.notna(dp)) else np.nan

def _derived_acc_crosses(r):
    c = _g(r, "Crosses per 90"); ca = _g(r, "Accurate crosses, %")
    return c * ca / 100 if (pd.notna(c) and pd.notna(ca)) else np.nan

def _derived_xa_per_kp(r):
    xa = _g(r, "xA per 90"); kp = _g(r, "Key passes per 90")
    return xa / kp if (pd.notna(xa) and pd.notna(kp) and kp > 0) else np.nan

def _derived_box_to_shot(r):
    sh = _g(r, "Shots per 90"); bx = _g(r, "Touches in box per 90")
    return sh / bx if (pd.notna(sh) and pd.notna(bx) and bx > 0) else np.nan

def _derived_prog_carry(r):
    pr = _g(r, "Progressive runs per 90"); ac = _g(r, "Accelerations per 90")
    parts = [v for v in [pr, ac] if pd.notna(v)]
    return sum(parts) if parts else np.nan

def _derived_involvement(r):
    ps = _g(r, "Passes per 90"); rp = _g(r, "Received passes per 90")
    d  = _g(r, "Dribbles per 90"); du_raw = _g(r, "Offensive duels per 90") if "Offensive duels per 90" in (r.index if hasattr(r, "index") else {}) else np.nan
    parts = [v for v in [ps, rp, d] if pd.notna(v)]
    return sum(parts) if parts else np.nan

def _derived_duel_overall(r):
    return _composite_duels(r)

def _derived_shot_conv(r):
    g = _g(r, "Goals per 90"); sh = _g(r, "Shots per 90")
    return g / sh if (pd.notna(g) and pd.notna(sh) and sh > 0) else np.nan

def _derived_def_intensity(r):
    da = _g(r, "Successful defensive actions per 90")
    ic = _g(r, "Interceptions per 90"); fo = _g(r, "Fouls per 90")
    parts = [v for v in [da, ic, fo] if pd.notna(v)]
    return sum(parts) if parts else np.nan

def _derived_press_resist(r):
    dp = _g(r, "Successful dribbles, %"); od = _g(r, "Offensive duels won, %")
    vals = [v for v in [dp, od] if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


# (display_name, short_formula, cat, fn, fmt_str)
DERIVED_LEFT = [
    ("xG per Shot",           "xG/90 ÷ Shots/90",              "Threat",    _derived_xg_per_shot,   "{:.3f}"),
    ("G − xG (per 90)",       "Goals/90 minus xG/90",           "Threat",    _derived_g_minus_xg,    "{:+.3f}"),
    ("Effective Dribbles",    "Dribbles/90 × Success%",         "Carrying",  _derived_eff_dribbles,  "{:.2f}"),
    ("Accurate Crosses/90",   "Crosses/90 × Accuracy%",         "Creation",  _derived_acc_crosses,   "{:.2f}"),
    ("xA per Key Pass",       "xA/90 ÷ KeyPass/90",             "Creation",  _derived_xa_per_kp,     "{:.3f}"),
    ("Box→Shot Rate",         "Shots/90 ÷ BoxTouches/90",       "Threat",    _derived_box_to_shot,   "{:.2f}"),
    ("Ball Progression/90",   "Prog. Runs + Accelerations",     "Carrying",  _derived_prog_carry,    "{:.2f}"),
    ("Involvement Index",     "Passes + Received + Dribbles",   "Creation",  _derived_involvement,   "{:.1f}"),
]

DERIVED_RIGHT = [
    ("Overall Duel Win%",     "Mean(Off%, Def%, 0.7×Aerial%)",  "Duels",     _derived_duel_overall,  "{:.1f}%"),
    ("Shot Conversion",       "Goals/90 ÷ Shots/90",            "Threat",    _derived_shot_conv,     "{:.3f}"),
    ("Def. Intensity/90",     "DefActions + Intercepts + Fouls","Defending", _derived_def_intensity, "{:.2f}"),
    ("Press Resistance",      "Mean(Drib%, OffDuel%)",          "Carrying",  _derived_press_resist,  "{:.1f}%"),
    ("Carrying Power",        "Eff.Drib + ProgRuns + 0.5×Accel","Carrying", _composite_carrying,    "{:.2f}"),
    ("Creative Danger",       "2×xA + 0.8×SA + 0.5×KP + AccCr","Creation", _composite_creation,    "{:.2f}"),
    ("Goal Threat Index",     "2×xG + 0.3×Shots + 0.2×BxTch",  "Threat",    _composite_threat,      "{:.2f}"),
    ("Duel Authority",        "Weighted duel win rate index",    "Duels",     _composite_duels,       "{:.1f}%"),
]


def _pool_derived(pool, fn):
    """Apply a derived metric function to every row in a pool, return array of floats."""
    vals = []
    for _, row in pool.iterrows():
        v = fn(row)
        if pd.notna(v):
            vals.append(v)
    return np.array(vals)


def calc_composites(player, pool_db, pool_lg):
    """Return (player_vals, db_pcts, lg_ranks) dicts for composite + derived specs."""
    all_specs = (
        [(s[0].replace("\n", " "), s[3]) for s in COMPOSITE_SPECS] +
        [(s[0], s[3]) for s in DERIVED_LEFT + DERIVED_RIGHT]
    )
    player_vals = {}
    db_pcts     = {}
    lg_ranks    = {}

    for name, fn in all_specs:
        pv   = fn(player)
        d_vs = _pool_derived(pool_db, fn)
        l_vs = _pool_derived(pool_lg, fn)

        player_vals[name] = pv
        db_pcts[name] = percentileofscore(d_vs, pv, kind="rank") \
                        if (len(d_vs) and pd.notna(pv)) else 50.0

        # League rank: how many are strictly worse + 1
        if len(l_vs) and pd.notna(pv):
            rank = int(np.sum(l_vs < pv)) + 1
            lg_ranks[name] = (rank, len(l_vs))
        else:
            lg_ranks[name] = (None, len(l_vs))

    return player_vals, db_pcts, lg_ranks


def draw_composite_cards(ax, player_vals, db_pcts, lg_ranks):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")
    _inner_title(ax, "COMPOSITE METRIC SCORES",
                 "Custom indices derived from raw Wyscout stats · benchmarked vs full database")

    n_cols, n_rows = 3, 2
    # card layout in axes fraction
    pad_x, pad_y = 0.02, 0.06
    cw = (1.0 - pad_x * (n_cols + 1)) / n_cols
    ch = (0.88 - pad_y * (n_rows + 1)) / n_rows  # 0.88 leaves room for inner title

    for idx, (name, subtitle, cat_key, fn) in enumerate(COMPOSITE_SPECS):
        col = idx % n_cols
        row = idx // n_cols
        x0 = pad_x + col * (cw + pad_x)
        y0 = (0.88 - pad_y) - row * (ch + pad_y) - ch

        c_cat = CAT_COLOURS.get(cat_key, ACCENT)
        key   = name.replace("\n", " ")
        pv    = player_vals.get(key, np.nan)
        pct   = db_pcts.get(key, 50.0)
        rank, n_lg = lg_ranks.get(key, (None, 0))
        bar_c = pct_colour(pct)

        # Card background
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), cw, ch, boxstyle="round,pad=0.005",
            facecolor=PANEL, edgecolor=BORDER, linewidth=0.6,
            transform=ax.transAxes, clip_on=True))

        # Top colour stripe
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0 + ch - 0.028), cw, 0.028, boxstyle="round,pad=0.002",
            facecolor=c_cat, edgecolor="none", alpha=0.15,
            transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Rectangle(
            (x0, y0 + ch - 0.028), 0.005, 0.028,
            facecolor=c_cat, edgecolor="none",
            transform=ax.transAxes, clip_on=True))

        # Metric name
        ax.text(x0 + 0.008, y0 + ch - 0.014, name,
                ha="left", va="center", transform=ax.transAxes,
                color=TEXT, fontsize=6.2, fontweight="bold", linespacing=1.2)

        # Score value (large)
        if pd.notna(pv):
            val_txt = f"{pv:.2f}"
        else:
            val_txt = "—"
        ax.text(x0 + cw / 2, y0 + ch * 0.54, val_txt,
                ha="center", va="center", transform=ax.transAxes,
                color=bar_c, fontsize=17, fontweight="bold")

        # Subtitle / formula
        ax.text(x0 + cw / 2, y0 + ch * 0.30, subtitle,
                ha="center", va="center", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=4.8, style="italic", wrap=True)

        # Mini percentile bar
        bar_y  = y0 + 0.020
        bar_h2 = 0.014
        bw = cw - 0.014
        ax.add_patch(mpatches.Rectangle(
            (x0 + 0.007, bar_y), bw, bar_h2,
            facecolor=SURFACE2, edgecolor="none",
            transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Rectangle(
            (x0 + 0.007, bar_y), bw * pct / 100, bar_h2,
            facecolor=bar_c, edgecolor="none", alpha=0.80,
            transform=ax.transAxes, clip_on=True))

        # Percentile label
        rank_txt = f"Rank {rank}/{n_lg} · {pct:.0f}th %ile DB" \
                   if rank else f"{pct:.0f}th %ile vs DB"
        ax.text(x0 + cw / 2, y0 + 0.005, rank_txt,
                ha="center", va="bottom", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=4.8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def draw_derived_table(ax, player_vals, db_pcts, lg_ranks, specs, title_shown=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")

    if title_shown:
        _inner_title(ax, "DERIVED EFFICIENCY METRICS",
                     "Computed ratios & indices not directly available in Wyscout")

    top_y = 0.92 if title_shown else 0.97
    rows  = np.linspace(top_y, 0.03, len(specs))
    rh    = abs(rows[0] - rows[1]) if len(rows) > 1 else 0.08

    dot_x  = 0.03
    lbl_x  = 0.08
    fml_x  = 0.08
    val_x  = 0.56
    bar_x0 = 0.62
    bar_x1 = 0.84
    pct_x  = 0.86
    rk_x   = 0.99

    for i, (name, formula, cat_key, fn, fmt) in enumerate(specs):
        ry  = rows[i]
        c_c = CAT_COLOURS.get(cat_key, TEXT_DIM)

        if i % 2 == 0:
            ax.add_patch(mpatches.Rectangle(
                (0, ry - rh * 0.48), 1.0, rh,
                facecolor=SURFACE2, edgecolor="none",
                transform=ax.transAxes, clip_on=True))

        ax.add_patch(mpatches.Circle(
            (dot_x, ry), rh * 0.20,
            facecolor=c_c, edgecolor="none",
            transform=ax.transAxes, clip_on=True))

        ax.text(lbl_x, ry + rh * 0.14, name,
                ha="left", va="center", transform=ax.transAxes,
                color=TEXT, fontsize=6.2, fontweight="bold")
        ax.text(fml_x, ry - rh * 0.16, formula,
                ha="left", va="center", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=4.8, style="italic")

        pv  = player_vals.get(name, np.nan)
        raw_s = fmt.format(pv) if pd.notna(pv) else "—"
        pct = db_pcts.get(name, 50.0)
        bar_c = pct_colour(pct)

        ax.text(val_x, ry, raw_s, ha="right", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=6.5, fontweight="bold")

        bw = bar_x1 - bar_x0
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, ry - rh * 0.20), bw, rh * 0.40,
            facecolor=SURFACE2, edgecolor="none",
            transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, ry - rh * 0.20), bw * pct / 100, rh * 0.40,
            facecolor=bar_c, edgecolor="none", alpha=0.78,
            transform=ax.transAxes, clip_on=True))

        ax.text(pct_x, ry, f"{pct:.0f}th", ha="left", va="center",
                transform=ax.transAxes, color=bar_c, fontsize=5.8, fontweight="bold")

        rank, n_pool = lg_ranks.get(name, (None, 0))
        rk_txt = f"{rank}/{n_pool}" if rank else "—"
        ax.text(rk_x, ry, rk_txt, ha="right", va="center",
                transform=ax.transAxes, color=TEXT_DIM, fontsize=5.5)

    # Column headers at top
    hdr_y = top_y + rh * 0.65
    for x, lbl, ha_ in [(val_x, "VALUE", "right"), (pct_x, "DB %ILE", "left"),
                         (rk_x, "LG RANK", "right")]:
        ax.text(x, hdr_y, lbl, ha=ha_, va="center", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=5.0, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def calc_war(player, pool_lg, pool_db):
    """
    Simplified football WAR for wide attackers.
    Replacement = 15th-percentile Czech First League wide attacker.
    3 goal-equivalents ≈ 1 win (conservative Czech league estimate).
    """
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

    # Pool rates (vectorized)
    lg_off, lg_carry, lg_tot = _rates(pool_lg)
    db_off, db_carry, db_tot = _rates(pool_db)

    repl = float(np.percentile(lg_tot.dropna(), REPL_PERCENTILE))

    # Player rates
    p_off   = _pval("xG per 90") + _pval("xA per 90") * 0.85 + _pval("Shot assists per 90") * 0.28
    p_carry = _pval("Progressive runs per 90") * 0.025 + \
              (_pval("Dribbles per 90") * _pval("Successful dribbles, %", 50) / 100) * 0.015
    p_tot   = p_off + p_carry

    minutes = _pval("Minutes played", 593)
    n90     = minutes / 90.0
    war     = (p_tot - repl) * n90 / GOALS_PER_WIN

    # Pool WARs for distribution
    lg_min  = pool_lg["Minutes played"].fillna(450) if "Minutes played" in pool_lg.columns \
              else pd.Series([450] * len(pool_lg))
    db_min  = pool_db["Minutes played"].fillna(450) if "Minutes played" in pool_db.columns \
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
    }


def draw_war_banner(ax, wd):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")
    _inner_title(ax, "WAR — WINS ABOVE REPLACEMENT",
                 "vs 15th-percentile Czech First League wide attacker  ·  3 goals ≈ 1 win")

    war   = wd["war"]
    c_war = pct_colour(wd["pct_db"])

    # ── Left: big WAR number ─────────────────────────────────────────────────
    # Background circle
    ax.add_patch(mpatches.Circle(
        (0.115, 0.46), 0.100,
        facecolor=SURFACE2, edgecolor=c_war, linewidth=1.5,
        transform=ax.transAxes, clip_on=True))
    ax.text(0.115, 0.50, f"{war:+.2f}",
            ha="center", va="center", transform=ax.transAxes,
            color=c_war, fontsize=19, fontweight="bold")
    ax.text(0.115, 0.14, "Wins Above\nReplacement",
            ha="center", va="center", transform=ax.transAxes,
            color=TEXT_MED, fontsize=6.5, linespacing=1.3)

    # ── Middle: component breakdown ───────────────────────────────────────────
    mid_x0, mid_x1 = 0.26, 0.62
    bw = mid_x1 - mid_x0

    components = [
        ("Offensive",          wd["offensive"],  wd["repl_off"],  "#DC2626"),
        ("Carrying",           wd["carrying"],   wd["repl_carry"],"#D97706"),
    ]
    comp_ys = [0.68, 0.44]

    for (label, pv, rv, cc), cy in zip(components, comp_ys):
        ax.text(mid_x0, cy + 0.08, label, ha="left", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=7, fontweight="bold")
        # Track
        ax.add_patch(mpatches.Rectangle(
            (mid_x0, cy - 0.04), bw, 0.090,
            facecolor=SURFACE2, edgecolor="none",
            transform=ax.transAxes, clip_on=True))
        # Scale so max ≈ full bar (use max of player + repl as scale)
        scale = max(pv, rv, 0.01) * 1.5
        # Replacement marker
        rx = mid_x0 + bw * min(rv / scale, 1.0)
        ax.plot([rx, rx], [cy - 0.04, cy + 0.05], color=TEXT_DIM, lw=1.0, ls="--", zorder=3)
        ax.text(rx, cy + 0.07, "repl.", ha="center", va="bottom",
                transform=ax.transAxes, color=TEXT_DIM, fontsize=4.8)
        # Player bar
        px = mid_x0 + bw * min(pv / scale, 1.0)
        ax.add_patch(mpatches.Rectangle(
            (mid_x0, cy - 0.04), px - mid_x0, 0.090,
            facecolor=cc, edgecolor="none", alpha=0.72,
            transform=ax.transAxes, clip_on=True))
        # value labels
        sign = "+" if pv >= rv else ""
        ax.text(mid_x1 + 0.015, cy + 0.02, f"{sign}{pv - rv:+.3f}",
                ha="left", va="center", transform=ax.transAxes,
                color=cc, fontsize=7, fontweight="bold")
        ax.text(mid_x1 + 0.015, cy - 0.04, "above repl.",
                ha="left", va="center", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=5.2)

    # Methodology note
    ax.text((mid_x0 + mid_x1) / 2, 0.11,
            f"Rate/90: {wd['player_rate']:.3f}  ·  Repl. level: {wd['repl_level']:.3f}  "
            f"·  {wd['nineties']:.1f} × 90s  ·  ÷ {wd['GOALS_PER_WIN']:.0f} goals/win",
            ha="center", va="center", transform=ax.transAxes,
            color=TEXT_DIM, fontsize=5.5)

    # ── Right: mini KDE distribution ─────────────────────────────────────────
    kde_ax_x0, kde_ax_x1 = 0.69, 0.99
    kde_w = kde_ax_x1 - kde_ax_x0
    lg_w  = wd["lg_wars"]
    war_v = wd["war"]

    if len(lg_w) > 4:
        lo = np.percentile(lg_w, 2); hi = np.percentile(lg_w, 98)
        xs = np.linspace(lo, hi, 200)
        try:
            kde = gaussian_kde(lg_w, bw_method=0.45)
            ys  = kde(xs)
            peak = max(ys.max(), 1e-9)
            # normalise y to [0, 0.78] axes fraction, baseline at 0.18
            base_y, top_y = 0.18, 0.88
            def ty(y_raw): return base_y + (y_raw / peak) * (top_y - base_y)

            # Map x to axes fraction
            def tx(x_raw): return kde_ax_x0 + (x_raw - lo) / max(hi - lo, 1e-9) * kde_w

            x_ax  = [tx(x) for x in xs]
            y_ax  = [ty(y) for y in ys]
            x_fill = x_ax + [x_ax[-1], x_ax[0]]
            y_fill = y_ax + [base_y, base_y]

            ax.fill(x_fill, y_fill, transform=ax.transAxes, color=LEAGUE_C, alpha=0.15, zorder=1)
            ax.plot(x_ax, y_ax, transform=ax.transAxes, color=LEAGUE_C, lw=1.2, zorder=2)

            # Player line
            if lo <= war_v <= hi:
                vx = tx(war_v)
                ax.plot([vx, vx], [base_y, top_y * 0.92], transform=ax.transAxes,
                        color=PLAYER_C, lw=2.0, zorder=4)
                y_at = ty(float(kde([war_v])[0]))
                ax.scatter([vx], [y_at], transform=ax.transAxes,
                           color=PLAYER_C, s=30, zorder=5, linewidths=0)
                ax.text(vx, top_y * 0.94, f"{war_v:+.2f}",
                        ha="center", va="bottom", transform=ax.transAxes,
                        color=PLAYER_C, fontsize=6.5, fontweight="bold")

            # Axis label
            ax.text((kde_ax_x0 + kde_ax_x1) / 2, base_y - 0.06,
                    "WAR distribution · Czech First League",
                    ha="center", va="top", transform=ax.transAxes,
                    color=TEXT_DIM, fontsize=5.2)
        except Exception:
            pass

    # Right-side rank badge
    ax.text(kde_ax_x1 - 0.01, 0.15,
            f"Rank  {wd['rank_lg']} / {wd['n_lg']}\n"
            f"{wd['pct_lg']:.0f}th  League\n"
            f"{wd['pct_db']:.0f}th  Database",
            ha="right", va="bottom", transform=ax.transAxes,
            color=TEXT_MED, fontsize=6.5, linespacing=1.5, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def draw_page3_header(ax):
    _off(ax)
    ax.add_patch(mpatches.Rectangle(
        (-0.018, 0.0), 0.007, 1.0,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(0.0, 0.97, P_HEADER_NAME, ha="left", va="top",
            transform=ax.transAxes, color=TEXT, fontsize=22, fontweight="bold")
    ax.text(0.0, 0.30, P_PAGE3_SUBTITLE,
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=8)
    ax.plot([0, 1], [0.04, 0.04], transform=ax.transAxes, color=BORDER, lw=0.8)


def make_page3(player, pool_db, pool_lg, player_vals, db_pcts, lg_ranks, war_data):
    fig3 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)

    outer3 = gridspec.GridSpec(
        4, 1, figure=fig3,
        left=0.04, right=0.97,
        top=0.975, bottom=0.022,
        height_ratios=[0.055, 0.175, 0.330, 0.415],
        hspace=0.18,
    )

    draw_page3_header(fig3.add_subplot(outer3[0]))
    draw_war_banner(fig3.add_subplot(outer3[1]), war_data)
    draw_composite_cards(fig3.add_subplot(outer3[2]), player_vals, db_pcts, lg_ranks)

    # Derived metrics — 2 columns
    drv_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer3[3], wspace=0.10,
    )
    draw_derived_table(fig3.add_subplot(drv_gs[0]), player_vals, db_pcts, lg_ranks,
                       DERIVED_LEFT, title_shown=True)
    draw_derived_table(fig3.add_subplot(drv_gs[1]), player_vals, db_pcts, lg_ranks,
                       DERIVED_RIGHT, title_shown=False)

    # Footer
    fig3.text(0.04, 0.007,
              "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
              ha="left", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig3.text(0.97, 0.007, "hradeck-scouting",
              ha="right", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig3.add_artist(plt.Line2D(
        [0.04, 0.97], [0.019, 0.019],
        transform=fig3.transFigure, color=BORDER, lw=0.7,
    ))

    return fig3


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading …")
    player, pool_cl, pool_lg, pool_tier, pool_db = load_data()
    n_cl = len(pool_cl); n_lg = len(pool_lg)
    n_ti = len(pool_tier); n_db = len(pool_db)
    print(f"  Club={n_cl}  League={n_lg}  Tier3={n_ti}  DB={n_db}")

    keys = list({m for m, *_ in BAR_METRICS} | {m for m, _ in BENCH_METRICS})
    pc_cl = calc_pcts(player, pool_cl,   keys)
    pc_lg = calc_pcts(player, pool_lg,   keys)
    pc_ti = calc_pcts(player, pool_tier, keys)
    pc_db = calc_pcts(player, pool_db,   keys)

    scores = role_fit(player, pool_lg)
    print(f"  Roles: { {r: f'{v:.0f}%' for r,v in scores.items()} }")

    # ── A4 portrait figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)

    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        left=0.04, right=0.97,
        top=0.975, bottom=0.022,
        height_ratios=[0.080, 0.122, 0.614, 0.162],
        hspace=0.22,          # generous gap between sections
    )

    draw_header(fig.add_subplot(outer[0]), n_lg, n_db)
    draw_profile_fit(fig.add_subplot(outer[1]), scores)

    # Distributions | Bars
    mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        wspace=0.07, width_ratios=[0.37, 0.63],
    )

    d_gs   = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=mid[0], hspace=0.60)
    d_axes = [fig.add_subplot(d_gs[i]) for i in range(4)]
    draw_distributions(d_axes, pool_db, pool_lg, player)

    d_axes[-1].legend(
        handles=[
            Line2D([0],[0], color="#94A3B8", lw=1.3, label="Full database"),
            Line2D([0],[0], color=LEAGUE_C,  lw=1.3, ls="--",
                   label="Czech First League"),
            Line2D([0],[0], color=PLAYER_C,  lw=1.8, label=P_LEGEND_LABEL),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.94),
        ncol=3, frameon=False, fontsize=5.8, labelcolor=TEXT_DIM, handlelength=1.2,
    )

    ax_bars = fig.add_subplot(mid[1])
    draw_bars(ax_bars, player, pc_cl, pc_lg, pc_ti, pc_db)

    # Legends — both INSIDE the axes, stacked at bottom-right and top-right
    leg_bm = ax_bars.legend(
        handles=[
            Line2D([0],[0], color=CLUB_C,   lw=1.6, label=f"Club (n={n_cl})"),
            Line2D([0],[0], color=LEAGUE_C, lw=1.6, label=f"League (n={n_lg})"),
            Line2D([0],[0], color=TIER_C,   lw=1.6, label=f"Tier 3 (n={n_ti:,})"),
        ],
        loc="upper right",
        frameon=True, facecolor=PANEL, edgecolor=BORDER,
        fontsize=5.5, labelcolor=TEXT, handlelength=1.2, ncol=1,
        title="Benchmark lines", title_fontsize=5.0,
        borderpad=0.5,
    )
    ax_bars.add_artist(leg_bm)
    ax_bars.legend(
        handles=[mpatches.Patch(facecolor=c, label=cat, alpha=0.85)
                 for cat, c in CAT_COLOURS.items()],
        loc="lower right",
        frameon=True, facecolor=PANEL, edgecolor=BORDER,
        fontsize=5.5, labelcolor=TEXT, handlelength=0.8, ncol=1,
        title="Category", title_fontsize=5.0,
        borderpad=0.5,
    )

    draw_bench_table(
        fig.add_subplot(outer[3]),
        player, pc_cl, pc_lg, pc_ti, pc_db,
        n_cl, n_lg, n_ti, n_db,
    )

    # Footer
    fig.text(0.04, 0.007,
             "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
             ha="left", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig.text(0.97, 0.007, "hradeck-scouting",
             ha="right", va="bottom", color=TEXT_DIM, fontsize=6.5)
    # Thin footer rule
    fig.add_artist(plt.Line2D(
        [0.04, 0.97], [0.019, 0.019],
        transform=fig.transFigure, color=BORDER, lw=0.7,
    ))

    # Build page 2
    keys2 = list({m for m, *_ in ALL_STATS_LEFT + ALL_STATS_RIGHT} | set(keys))
    pc_cl2 = calc_pcts(player, pool_cl,   keys2)
    pc_lg2 = calc_pcts(player, pool_lg,   keys2)
    pc_ti2 = calc_pcts(player, pool_tier, keys2)
    pc_db2 = calc_pcts(player, pool_db,   keys2)
    # merge into existing dicts
    pc_cl.update(pc_cl2); pc_lg.update(pc_lg2)
    pc_ti.update(pc_ti2); pc_db.update(pc_db2)

    fig2 = make_page2(player, pool_cl, pool_lg, pool_tier, pool_db,
                      pc_cl, pc_lg, pc_ti, pc_db)

    # Build page 3 composites + WAR
    print("  Computing composite metrics …")
    player_vals, db_pcts3, lg_ranks3 = calc_composites(player, pool_db, pool_lg)
    print("  Computing WAR …")
    war_data = calc_war(player, pool_lg, pool_db)
    print(f"  WAR = {war_data['war']:+.3f}  "
          f"(League rank {war_data['rank_lg']}/{war_data['n_lg']}  ·  "
          f"DB {war_data['pct_db']:.0f}th pctile)")
    fig3 = make_page3(player, pool_db, pool_lg, player_vals, db_pcts3, lg_ranks3, war_data)

    png  = OUT_DIR / "D_Barat_Scouting_Report.png"
    png2 = OUT_DIR / "D_Barat_Scouting_Report_P2.png"
    png3 = OUT_DIR / "D_Barat_Scouting_Report_P3.png"
    pdf  = OUT_DIR / "D_Barat_Scouting_Report.pdf"

    fig.savefig(png,   dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig2.savefig(png2, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig3.savefig(png3, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")

    with PdfPages(pdf) as pp:
        pp.savefig(fig,  bbox_inches="tight", facecolor=BG, edgecolor="none")
        pp.savefig(fig2, bbox_inches="tight", facecolor=BG, edgecolor="none")
        pp.savefig(fig3, bbox_inches="tight", facecolor=BG, edgecolor="none")

    plt.close(fig); plt.close(fig2); plt.close(fig3)
    print(f"  Saved → {png}")
    print(f"  Saved → {png2}")
    print(f"  Saved → {png3}")
    print(f"  Saved → {pdf}  (3 pages)")


if __name__ == "__main__":
    main()
