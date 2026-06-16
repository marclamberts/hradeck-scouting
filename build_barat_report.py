"""
build_barat_report.py  —  A4 portrait, professional edition v3
All section labels drawn INSIDE axes · two-pass bars · fixed table columns
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

    player  = df_c1[df_c1["Player"].astype(str).str.startswith("D. Bar")].iloc[0].copy()
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
            Line2D([0],[0], color=PLAYER_C,  lw=1.8, label="D. Barát"),
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

    png = OUT_DIR / "D_Barat_Scouting_Report.png"
    pdf = OUT_DIR / "D_Barat_Scouting_Report.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig.savefig(pdf, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {png}")
    print(f"  Saved → {pdf}")


if __name__ == "__main__":
    main()
