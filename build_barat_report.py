"""
build_barat_report.py  —  A4 portrait, white theme
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

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

# ── White theme ────────────────────────────────────────────────────────────────
BG       = "#FFFFFF"
SURFACE  = "#F8FAFC"
SURFACE2 = "#EDF1F7"
BORDER   = "#D8E0EC"
TEXT     = "#0F172A"
TEXT_MED = "#334155"
TEXT_DIM = "#64748B"
PLAYER_C = "#6366F1"    # indigo — player marker

CLUB_C   = "#8B5CF6"    # purple
LEAGUE_C = "#3B82F6"    # blue
TIER_C   = "#0D9488"    # teal
DB_C     = "#6B7280"    # grey

PERF_CMAP = LinearSegmentedColormap.from_list(
    "perf", [(0.0, "#EF4444"), (0.5, "#F59E0B"), (1.0, "#22C55E")]
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

# ── Metrics ────────────────────────────────────────────────────────────────────
CAT_COLOURS = {
    "Threat":    "#EF4444",
    "Carrying":  "#F59E0B",
    "Creation":  "#10B981",
    "Duels":     "#3B82F6",
    "Defending": "#8B5CF6",
}

BAR_METRICS = [
    ("xG per 90",                          "xG / 90",           "Threat"),
    ("Shots per 90",                       "Shots / 90",        "Threat"),
    ("Touches in box per 90",              "Box Touches / 90",  "Threat"),
    ("Dribbles per 90",                    "Dribbles / 90",     "Carrying"),
    ("Successful dribbles, %",             "Dribble %",         "Carrying"),
    ("Progressive runs per 90",            "Prog. Runs / 90",   "Carrying"),
    ("Crosses per 90",                     "Crosses / 90",      "Creation"),
    ("Accurate crosses, %",                "Cross Acc. %",      "Creation"),
    ("xA per 90",                          "xA / 90",           "Creation"),
    ("Shot assists per 90",                "Shot Assists / 90", "Creation"),
    ("Key passes per 90",                  "Key Passes / 90",   "Creation"),
    ("Offensive duels won, %",             "Off. Duels %",      "Duels"),
    ("Defensive duels won, %",             "Def. Duels %",      "Duels"),
    ("Successful defensive actions per 90","Def. Actions / 90", "Defending"),
]

DIST_METRICS = [
    ("Dribbles per 90",         "Dribbles\n/ 90"),
    ("xG per 90",               "xG\n/ 90"),
    ("Crosses per 90",          "Crosses\n/ 90"),
    ("xA per 90",               "xA\n/ 90"),
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
        "Dribbles per 90": 0.25,
        "Successful dribbles, %": 0.15,
        "Progressive runs per 90": 0.20,
        "Crosses per 90": 0.15,
        "Touches in box per 90": 0.15,
        "Offensive duels won, %": 0.10,
    },
    "Finisher": {
        "Goals per 90": 0.35,
        "xG per 90": 0.25,
        "Shots per 90": 0.20,
        "Touches in box per 90": 0.20,
    },
    "Target": {
        "Aerial duels won, %": 0.40,
        "Shots per 90": 0.20,
        "Goals per 90": 0.15,
        "Touches in box per 90": 0.25,
    },
    "Roamer": {
        "Successful defensive actions per 90": 0.25,
        "Offensive duels won, %": 0.25,
        "Dribbles per 90": 0.20,
        "Progressive runs per 90": 0.30,
    },
    "Unlocker": {
        "Key passes per 90": 0.25,
        "xA per 90": 0.25,
        "Shot assists per 90": 0.25,
        "Passes per 90": 0.15,
        "Accurate passes, %": 0.10,
    },
    "Outlet": {
        "Accurate passes, %": 0.30,
        "Passes per 90": 0.25,
        "Offensive duels won, %": 0.20,
        "Key passes per 90": 0.15,
        "xA per 90": 0.10,
    },
}

# ── Data ───────────────────────────────────────────────────────────────────────

def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in NON_METRIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_data():
    print("  Czech files …")
    df_c1 = _numeric(pd.read_excel("data/Wyscout DB/Czech.xlsx"))
    df_c1["_file"] = "Czech"
    df_c1["_pos1"] = df_c1["Position"].astype(str).str.split(",").str[0].str.strip()

    player = df_c1[df_c1["Player"].astype(str).str.startswith("D. Bar")].iloc[0].copy()
    team   = str(player.get("Team within selected timeframe", "Slovácko"))
    team_kw = team.split()[0]   # "Slovácko"

    pool_league = df_c1[
        df_c1["_pos1"].isin(WIDE_ATK_POS) &
        (df_c1["Minutes played"].fillna(0) >= MIN_MINS)
    ].copy()

    pool_club = df_c1[
        df_c1["Team within selected timeframe"].astype(str).str.contains(team_kw, na=False) &
        df_c1["_pos1"].isin(WIDE_ATK_POS)
    ].copy()

    print("  Full database (165 files) …")
    parts = []
    for f in sorted(Path("data/Wyscout DB").glob("*.xlsx")):
        stem = f.stem
        if stem in ("Czech U17", "Czech U19"):
            continue
        try:
            d = _numeric(pd.read_excel(f))
            d["_file"] = stem
            d["_pos1"] = d["Position"].astype(str).str.split(",").str[0].str.strip()
            parts.append(d)
        except Exception:
            pass

    df_all = pd.concat(parts, ignore_index=True)

    pool_db = df_all[
        df_all["_pos1"].isin(WIDE_ATK_POS) &
        (df_all["Minutes played"].fillna(0) >= MIN_MINS)
    ].copy().reset_index(drop=True)

    pool_tier = df_all[
        df_all["_file"].isin(TIER3_STEMS) &
        df_all["_pos1"].isin(WIDE_ATK_POS) &
        (df_all["Minutes played"].fillna(0) >= MIN_MINS)
    ].copy().reset_index(drop=True)

    return player, pool_club, pool_league, pool_tier, pool_db


def pct(player, pool, metrics):
    out = {}
    for m in metrics:
        if m not in pool.columns or m not in player.index or len(pool) == 0:
            out[m] = 50.0; continue
        vals = pool[m].dropna().values
        pv = float(player.get(m, np.nan))
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
        z = sum((w / tw) * (float(player[m]) - pool[m].mean()) / (pool[m].std() or 1e-9)
                for m, w in avail if not np.isnan(float(player.get(m, np.nan))))
        scores[role] = float(norm.cdf(z) * 100)
    return scores


# ── Drawing ────────────────────────────────────────────────────────────────────

def _clean_ax(ax, face=SURFACE):
    ax.set_facecolor(face)
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_header(ax, n_lg, n_db):
    _clean_ax(ax, BG)
    ax.axis("off")
    ax.text(0, 1.0, "D. BARÁT", ha="left", va="top",
            transform=ax.transAxes, color=TEXT, fontsize=28, fontweight="bold")
    ax.text(0, 0.36,
            "Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Czech Republic",
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=8)
    ax.plot([0, 1], [0.04, 0.04], transform=ax.transAxes,
            color=LEAGUE_C, linewidth=2.0, clip_on=False)

    pills = [("MIN","593"),("M","19"),("xG","0.93"),("xA","0.38"),("DRIB/90","4.55")]
    px = 0.50
    for lbl, val in pills:
        ax.text(px, 0.97, val, ha="center", va="top",
                color="#D97706", fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.text(px, 0.34, lbl, ha="center", va="top",
                color=TEXT_DIM, fontsize=6.5, transform=ax.transAxes)
        px += 0.094

    ax.text(1.0, 0.97, f"League pool: {n_lg}  ·  Database: {n_db}",
            ha="right", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=7)


def draw_profile_fit(ax, scores):
    _clean_ax(ax, SURFACE)
    ax.axis("off")

    ax.text(0.5, 1.04, "ATTACKER PROFILE FIT  ·  The Athletic role archetypes",
            ha="center", va="bottom", transform=ax.transAxes,
            color=TEXT_DIM, fontsize=7, fontweight="bold", style="italic")

    roles  = list(scores.keys())
    vals   = [scores[r] for r in roles]
    best_i = int(np.argmax(vals))
    n      = len(roles)

    slot_w = 1.0 / n
    pad    = 0.008

    for i, (role, score) in enumerate(zip(roles, vals)):
        x0 = i * slot_w + pad
        x1 = (i + 1) * slot_w - pad
        mx = (x0 + x1) / 2
        is_best = (i == best_i)
        c = pct_colour(score)
        edge = PLAYER_C if is_best else BORDER
        lw   = 1.8 if is_best else 0.7
        face = "#EEECFF" if is_best else SURFACE

        box = mpatches.FancyBboxPatch(
            (x0, 0.04), x1 - x0, 0.92,
            boxstyle="round,pad=0.005",
            linewidth=lw, edgecolor=edge, facecolor=face,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)

        ax.text(mx, 0.88, role, ha="center", va="top",
                transform=ax.transAxes, color=TEXT if is_best else TEXT_MED,
                fontsize=7 if is_best else 6.5,
                fontweight="bold" if is_best else "normal")

        bx0 = x0 + 0.008; bw = x1 - x0 - 0.016
        # track
        ax.add_patch(mpatches.Rectangle(
            (bx0, 0.46), bw, 0.10, facecolor=SURFACE2, edgecolor="none",
            transform=ax.transAxes, clip_on=False))
        # fill
        ax.add_patch(mpatches.Rectangle(
            (bx0, 0.46), bw * score / 100, 0.10,
            facecolor=c, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, clip_on=False))

        ax.text(mx, 0.40, f"{score:.0f}%", ha="center", va="top",
                transform=ax.transAxes, color=c,
                fontsize=8 if is_best else 7, fontweight="bold")

        if is_best:
            ax.text(mx, 0.15, "▲ Best fit", ha="center", va="top",
                    transform=ax.transAxes, color=PLAYER_C, fontsize=5.5)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def draw_distributions(axes, pool_db, pool_league, player):
    for ax, (col, label) in zip(axes, DIST_METRICS):
        _clean_ax(ax)
        vals_db = pool_db[col].dropna().values  if col in pool_db.columns  else np.array([])
        vals_lg = pool_league[col].dropna().values if col in pool_league.columns else np.array([])
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
            y_db = kde_db(xs)
            ax.fill_between(xs, y_db, alpha=0.13, color="#94A3B8")
            ax.plot(xs, y_db, color="#94A3B8", linewidth=0.9)
        except Exception:
            y_db = np.zeros_like(xs)

        if len(vals_lg) >= 4:
            try:
                kde_lg = gaussian_kde(vals_lg, bw_method=0.45)
                ax.plot(xs, kde_lg(xs), color=LEAGUE_C, linewidth=1.2,
                        linestyle="--", alpha=0.82)
            except Exception:
                pass

        if not np.isnan(pv):
            y_at_pv = float(gaussian_kde(vals_db, bw_method=0.35)([pv])[0]) \
                      if len(vals_db) >= 4 else 0
            ax.axvline(pv, color=PLAYER_C, linewidth=1.6, zorder=5, alpha=0.85)
            ax.scatter([pv], [y_at_pv], color=PLAYER_C, s=25, zorder=6)
            p = percentileofscore(vals_db, pv, kind="rank")
            peak = max(y_db) if len(y_db) else 1
            ax.text(pv, y_at_pv + 0.04 * peak, f"{p:.0f}th",
                    ha="center", va="bottom",
                    color=pct_colour(p), fontsize=6.5, fontweight="bold")

        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelsize=5.5, colors=TEXT_DIM, length=2, pad=1)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.text(-0.05, 0.5, label, ha="right", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=6.5,
                fontweight="bold", linespacing=1.3)
        ax.axhline(0, color=BORDER, linewidth=0.5)


def draw_bars(ax, player, pc_club, pc_lg, pc_tier, pc_db):
    _clean_ax(ax, BG)
    n = len(BAR_METRICS)
    spacing = 1.0
    bar_h = 0.50
    prev_cat = None

    for i, (m, label, cat) in enumerate(BAR_METRICS):
        y  = (n - 1 - i) * spacing
        pd_val = pc_db.get(m, 50.0)
        col = pct_colour(pd_val)

        raw = float(player.get(m, np.nan))
        raw_s = (f"{raw:.1f}%" if "%" in label else f"{raw:.2f}") if pd.notna(raw) else "—"

        if cat != prev_cat and prev_cat is not None:
            ax.plot([-36, 101], [y + spacing * 0.74]*2, color=BORDER, lw=0.5)
        prev_cat = cat

        ax.scatter(-0.8, y, s=20, color=CAT_COLOURS[cat], marker="s", linewidths=0, zorder=5)

        ax.text(-1.5, y + 0.14, label, ha="right", va="center", color=TEXT, fontsize=7.5)
        ax.text(-1.5, y - 0.19, raw_s, ha="right", va="center", color=TEXT_DIM, fontsize=6.5)

        # Track
        ax.barh(y, 100, height=bar_h, color=SURFACE2, left=0, lw=0, zorder=1, ec="none")
        # Fill (db pct, gradient)
        ax.barh(y, pd_val, height=bar_h, color=col, alpha=0.68, left=0, lw=0, zorder=2, ec="none")
        # Bright tip
        if pd_val > 5:
            ax.barh(y, 4, height=bar_h, color=col, alpha=1.0, left=pd_val - 4,
                    lw=0, zorder=3, ec="none")

        # Benchmark tick lines
        for bpct, bc in [(pc_club.get(m,50), CLUB_C),
                         (pc_lg.get(m,50),   LEAGUE_C),
                         (pc_tier.get(m,50), TIER_C)]:
            ax.plot([bpct, bpct], [y - bar_h*0.5, y + bar_h*0.5],
                    color=bc, lw=1.6, zorder=4, alpha=0.9)

        ax.text(102, y, f"{pd_val:.0f}th", ha="left", va="center",
                color=col, fontsize=7.5, fontweight="bold")

    for v in (25, 50, 75):
        ax.plot([v, v], [-0.7, n * spacing - 0.3], color=BORDER, lw=0.6, ls=":", zorder=0)
        ax.text(v, n * spacing - 0.05, str(v), ha="center", va="bottom",
                color=TEXT_DIM, fontsize=6)

    ax.set_xlim(-40, 120)
    ax.set_ylim(-0.7, n * spacing - 0.0)
    ax.axis("off")


def draw_bench_table(ax, player, pc_club, pc_lg, pc_tier, pc_db,
                     n_club, n_lg, n_tier, n_db):
    _clean_ax(ax)
    ax.axis("off")
    ax.text(0.5, 1.03,
            "BENCHMARK SUMMARY  ·  Percentile rank vs each pool",
            ha="center", va="bottom", transform=ax.transAxes,
            color=TEXT_DIM, fontsize=7, fontweight="bold", style="italic")

    col_xs   = [0.32, 0.48, 0.63, 0.78, 0.93]
    hdrs     = [
        f"Club\n(n={n_club})", f"League\n(n={n_lg})",
        f"Tier 3\n(n={n_tier})", f"Database\n(n={n_db})",
    ]
    hdr_cs   = [CLUB_C, LEAGUE_C, TIER_C, DB_C]
    all_pcts = [pc_club, pc_lg, pc_tier, pc_db]
    row_ys   = np.linspace(0.85, 0.10, len(BENCH_METRICS) + 1)

    # Header
    ax.text(0.03, row_ys[0], "METRIC", ha="left", va="center",
            transform=ax.transAxes, color=TEXT_DIM, fontsize=6.5, fontweight="bold")
    ax.text(0.32, row_ys[0], "VALUE", ha="left", va="center",
            transform=ax.transAxes, color=TEXT_DIM, fontsize=6.5, fontweight="bold")
    for cx, hdr, cc in zip(col_xs[1:], hdrs, hdr_cs):
        ax.text(cx, row_ys[0], hdr, ha="center", va="center",
                transform=ax.transAxes, color=cc, fontsize=6.5,
                fontweight="bold", linespacing=1.2)
    ax.plot([0.01, 0.99], [row_ys[0] - 0.055]*2,
            transform=ax.transAxes, color=BORDER, lw=0.8)

    for ri, (m, lbl) in enumerate(BENCH_METRICS):
        ry = row_ys[ri + 1]
        if ri % 2 == 1:
            ax.add_patch(mpatches.Rectangle(
                (0.01, ry - 0.07), 0.98, 0.13,
                facecolor=SURFACE2, edgecolor="none",
                transform=ax.transAxes, clip_on=False))

        ax.text(0.03, ry, lbl, ha="left", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=6.5)
        raw = float(player.get(m, np.nan))
        ax.text(0.32, ry, f"{raw:.2f}" if pd.notna(raw) else "—",
                ha="left", va="center", transform=ax.transAxes,
                color=TEXT, fontsize=7, fontweight="bold")

        for cx, pdict in zip(col_xs[1:], all_pcts):
            p = pdict.get(m, 50.0)
            ax.text(cx, ry, f"{p:.0f}th", ha="center", va="center",
                    transform=ax.transAxes, color=pct_colour(p),
                    fontsize=7, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading …")
    player, pool_club, pool_lg, pool_tier, pool_db = load_data()
    n_cl = len(pool_club); n_lg = len(pool_lg)
    n_ti = len(pool_tier); n_db = len(pool_db)
    print(f"  Club={n_cl}  League={n_lg}  Tier3={n_ti}  DB={n_db}")

    keys = list({m for m, *_ in BAR_METRICS} | {m for m, _ in BENCH_METRICS})
    pc_cl  = pct(player, pool_club,  keys)
    pc_lg  = pct(player, pool_lg,    keys)
    pc_ti  = pct(player, pool_tier,  keys)
    pc_db  = pct(player, pool_db,    keys)

    scores = role_fit(player, pool_lg)
    print(f"  Roles: {scores}")

    # ── Figure: A4 portrait ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)

    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        left=0.02, right=0.98,
        top=0.975, bottom=0.018,
        height_ratios=[0.085, 0.095, 0.615, 0.175],
        hspace=0.12,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    draw_header(fig.add_subplot(outer[0]), n_lg, n_db)

    # ── Profile Fit ───────────────────────────────────────────────────────────
    draw_profile_fit(fig.add_subplot(outer[1]), scores)

    # ── Main: distributions | bars ────────────────────────────────────────────
    mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        wspace=0.07, width_ratios=[0.37, 0.63],
    )

    # Distributions — 4 stacked
    d_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=mid[0], hspace=0.60)
    d_axes = [fig.add_subplot(d_gs[i]) for i in range(4)]
    d_axes[0].set_title(
        "DISTRIBUTIONS  ·  Database (grey) vs League (blue dashed)",
        fontsize=6.5, color=TEXT_DIM, pad=5,
        fontweight="bold", style="italic", loc="left",
    )
    draw_distributions(d_axes, pool_db, pool_lg, player)

    d_axes[-1].legend(
        handles=[
            Line2D([0],[0], color="#94A3B8", lw=1.2, label="Database"),
            Line2D([0],[0], color=LEAGUE_C, lw=1.2, ls="--", label="Czech First League"),
            Line2D([0],[0], color=PLAYER_C, lw=1.6, label="D. Barát"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.88),
        ncol=3, frameon=False, fontsize=6, labelcolor=TEXT_DIM,
    )

    # Bars
    ax_bars = fig.add_subplot(mid[1])
    ax_bars.text(0.5, 1.008,
                 "PERCENTILE RANKS  ·  Database (bar)  ▏Club  ▏League  ▏Tier",
                 ha="center", va="bottom", transform=ax_bars.transAxes,
                 color=TEXT_DIM, fontsize=6.5, fontweight="bold", style="italic")

    draw_bars(ax_bars, player, pc_cl, pc_lg, pc_ti, pc_db)

    # Two legends: categories + benchmark lines
    leg_cat = ax_bars.legend(
        handles=[mpatches.Patch(facecolor=c, label=cat, alpha=0.8)
                 for cat, c in CAT_COLOURS.items()],
        loc="lower right", bbox_to_anchor=(1.01, -0.01),
        frameon=True, facecolor=SURFACE, edgecolor=BORDER,
        fontsize=5.5, labelcolor=TEXT, handlelength=0.8, ncol=1,
    )
    ax_bars.add_artist(leg_cat)

    ax_bars.legend(
        handles=[
            Line2D([0],[0], color=CLUB_C,   lw=1.5, label=f"Club (n={n_cl})"),
            Line2D([0],[0], color=LEAGUE_C, lw=1.5, label=f"League (n={n_lg})"),
            Line2D([0],[0], color=TIER_C,   lw=1.5, label=f"Tier 3 (n={n_ti})"),
        ],
        loc="upper right", bbox_to_anchor=(1.01, 1.02),
        frameon=True, facecolor=SURFACE, edgecolor=BORDER,
        fontsize=5.5, labelcolor=TEXT, handlelength=1.0, ncol=1,
    )

    # ── Benchmark table ───────────────────────────────────────────────────────
    draw_bench_table(
        fig.add_subplot(outer[3]),
        player, pc_cl, pc_lg, pc_ti, pc_db,
        n_cl, n_lg, n_ti, n_db,
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.02, 0.006,
             "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
             ha="left", va="bottom", color=TEXT_DIM, fontsize=6)
    fig.text(0.98, 0.006, "hradeck-scouting",
             ha="right", va="bottom", color=TEXT_DIM, fontsize=6)

    # ── Save ──────────────────────────────────────────────────────────────────
    png = OUT_DIR / "D_Barat_Scouting_Report.png"
    pdf = OUT_DIR / "D_Barat_Scouting_Report.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig.savefig(pdf, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {png}")
    print(f"  Saved → {pdf}")


if __name__ == "__main__":
    main()
