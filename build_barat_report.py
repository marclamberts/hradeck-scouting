"""
build_barat_report.py
─────────────────────
One-pager scouting report for D. Barát — white theme.
Left: KDE distribution plots (Czech I vs Czech I+II).
Right: horizontal percentile rank bars, coloured by category + performance.
Output: reports/D_Barat_Scouting_Report.png / .pdf
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import percentileofscore, gaussian_kde
from pathlib import Path

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

# ── White theme ───────────────────────────────────────────────────────────────
BG       = "#FFFFFF"
SURFACE  = "#F8FAFC"
SURFACE2 = "#EEF2F7"
BORDER   = "#DDE3EC"
TEXT     = "#0F172A"
TEXT_DIM = "#64748B"
PLAYER_C = "#6366F1"   # indigo — player marker in distributions
POOL_C   = "#94A3B8"   # slate  — combined pool KDE
C1_C     = "#3B82F6"   # blue   — Czech I KDE (dashed)
GOLD     = "#D97706"
ACCENT2  = "#059669"
ORANGE   = "#EA580C"
DANGER   = "#DC2626"

# ── Pool config ───────────────────────────────────────────────────────────────
WIDE_ATK_POS = {"LAMF","RAMF","LW","RW","LWF","RWF","AMF","LWB","RWB"}
MIN_MINS     = 300

# ── Category colours ──────────────────────────────────────────────────────────
CAT_COLOURS = {
    "Threat":    "#EF4444",
    "Carrying":  "#F59E0B",
    "Creation":  "#10B981",
    "Duels":     "#3B82F6",
    "Defending": "#8B5CF6",
}

# ── 14 bar metrics ────────────────────────────────────────────────────────────
BAR_METRICS = [
    ("xG per 90",                          "xG / 90",               "Threat"),
    ("Shots per 90",                       "Shots / 90",            "Threat"),
    ("Touches in box per 90",              "Box Touches / 90",      "Threat"),
    ("Dribbles per 90",                    "Dribbles / 90",         "Carrying"),
    ("Successful dribbles, %",             "Dribble Success %",     "Carrying"),
    ("Progressive runs per 90",            "Prog. Runs / 90",       "Carrying"),
    ("Crosses per 90",                     "Crosses / 90",          "Creation"),
    ("Accurate crosses, %",                "Cross Accuracy %",      "Creation"),
    ("xA per 90",                          "xA / 90",               "Creation"),
    ("Shot assists per 90",                "Shot Assists / 90",     "Creation"),
    ("Key passes per 90",                  "Key Passes / 90",       "Creation"),
    ("Offensive duels won, %",             "Off. Duels Won %",      "Duels"),
    ("Defensive duels won, %",             "Def. Duels Won %",      "Duels"),
    ("Successful defensive actions per 90","Def. Actions / 90",     "Defending"),
]

# ── 6 distribution metrics ────────────────────────────────────────────────────
DIST_METRICS = [
    ("Dribbles per 90",           "Dribbles / 90"),
    ("xG per 90",                 "xG / 90"),
    ("Progressive runs per 90",   "Prog. Runs / 90"),
    ("Crosses per 90",            "Crosses / 90"),
    ("Shot assists per 90",       "Shot Assists / 90"),
    ("xA per 90",                 "xA / 90"),
]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_pool():
    df1 = pd.read_excel("data/Wyscout DB/Czech.xlsx")
    df2 = pd.read_excel("data/Wyscout DB/Czech II.xlsx")
    df1["_league"] = "Czech First League"
    df2["_league"] = "Czech II"
    df = pd.concat([df1, df2], ignore_index=True)

    skip = {"Player","Team","Team within selected timeframe","Position",
            "Age","Market value","Contract expires","Birth country",
            "Passport country","Foot","On loan","_league"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["_pos1"] = df["Position"].astype(str).str.split(",").str[0].str.strip()
    player = df[df["Player"].astype(str).str.startswith("D. Bar")].iloc[0].copy()

    pool_all = df[
        df["_pos1"].isin(WIDE_ATK_POS) &
        (df["Minutes played"].fillna(0) >= MIN_MINS)
    ].copy().reset_index(drop=True)

    pool_c1 = pool_all[pool_all["_league"] == "Czech First League"].copy()

    if not pool_all["Player"].astype(str).str.startswith("D. Bar").any():
        pool_all = pd.concat([pool_all, player.to_frame().T], ignore_index=True)

    return player, pool_all, pool_c1


def compute_percentiles(player, pool, metrics):
    pcts = {}
    for m in metrics:
        if m not in pool.columns or m not in player.index:
            pcts[m] = 50.0; continue
        vals = pool[m].dropna().values
        pv = float(player.get(m, np.nan))
        if np.isnan(pv) or len(vals) == 0:
            pcts[m] = 50.0
        else:
            pcts[m] = round(percentileofscore(vals, pv, kind="rank"), 1)
    return pcts


def pct_colour(pct: float) -> str:
    if pct >= 85: return ACCENT2
    if pct >= 70: return C1_C
    if pct >= 50: return GOLD
    if pct >= 30: return ORANGE
    return DANGER


# ── Distribution panel ────────────────────────────────────────────────────────

def draw_distributions(axes, pool_all, pool_c1, player):
    for ax, (col, label) in zip(axes, DIST_METRICS):
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_visible(False)

        vals_all = pool_all[col].dropna().values if col in pool_all.columns else np.array([])
        vals_c1  = pool_c1[col].dropna().values  if col in pool_c1.columns  else np.array([])
        pv       = float(player.get(col, np.nan))

        if len(vals_all) < 4:
            ax.text(0.5, 0.5, label + "\n(n/a)", ha="center", va="center",
                    transform=ax.transAxes, color=TEXT_DIM, fontsize=6.5)
            ax.axis("off"); continue

        lo = max(0, np.percentile(vals_all, 1) - 0.2)
        hi = np.percentile(vals_all, 99) + 0.2
        xs = np.linspace(lo, hi, 300)

        # Combined pool — shaded grey
        try:
            kde_all = gaussian_kde(vals_all, bw_method=0.4)
            y_all   = kde_all(xs)
            ax.fill_between(xs, y_all, alpha=0.18, color=POOL_C)
            ax.plot(xs, y_all, color=POOL_C, linewidth=1.2)
        except Exception:
            y_all = np.zeros_like(xs)

        # Czech I — dashed blue
        if len(vals_c1) >= 4:
            try:
                kde_c1 = gaussian_kde(vals_c1, bw_method=0.5)
                y_c1   = kde_c1(xs)
                ax.plot(xs, y_c1, color=C1_C, linewidth=1.2,
                        linestyle="--", alpha=0.85)
            except Exception:
                pass

        # Player marker
        if not np.isnan(pv) and len(vals_all) > 0:
            y_peak = float(gaussian_kde(vals_all, bw_method=0.4)(np.array([pv]))[0]) \
                     if len(vals_all) >= 4 else 0
            ax.axvline(pv, color=PLAYER_C, linewidth=1.8, zorder=5, alpha=0.9)
            ax.scatter([pv], [y_peak], color=PLAYER_C, s=35, zorder=6)
            pct = percentileofscore(vals_all, pv, kind="rank")
            offset = 0.04 * (max(y_all) if len(y_all) else 1)
            ax.text(pv, y_peak + offset,
                    f"{pct:.0f}th",
                    ha="center", va="bottom",
                    color=pct_colour(pct), fontsize=7, fontweight="bold")

        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)

        # X ticks only (no y ticks)
        ax.tick_params(axis="x", labelsize=6, colors=TEXT_DIM, length=2, pad=1)
        ax.tick_params(axis="y", left=False, labelleft=False)

        # Metric label as left-side annotation
        ax.text(-0.02, 0.5, label, ha="right", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=7,
                fontweight="bold")

        # Faint horizontal gridline at y=0
        ax.axhline(0, color=BORDER, linewidth=0.6)


# ── Percentile bar panel ──────────────────────────────────────────────────────

def draw_percentile_bars(ax, player, pcts):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)

    n       = len(BAR_METRICS)
    bar_h   = 0.52
    spacing = 1.0
    prev_cat = None

    for i, (m, label, cat) in enumerate(BAR_METRICS):
        y       = (n - 1 - i) * spacing
        pct     = pcts.get(m, 50.0)
        col     = CAT_COLOURS[cat]
        pct_col = pct_colour(pct)

        raw_v   = float(player.get(m, np.nan))
        if pd.isna(raw_v):
            raw_str = "—"
        elif "%" in label:
            raw_str = f"{raw_v:.1f}%"
        else:
            raw_str = f"{raw_v:.2f}"

        # Category separator line
        if cat != prev_cat and prev_cat is not None:
            sep_y = y + spacing * 0.72
            ax.plot([-56, 117], [sep_y, sep_y], color=BORDER,
                    linewidth=0.6, zorder=0)
        prev_cat = cat

        # Metric label + raw value
        ax.text(-3.5, y + 0.13, label, ha="right", va="center",
                color=TEXT, fontsize=7.5, fontweight="normal")
        ax.text(-3.5, y - 0.22, raw_str, ha="right", va="center",
                color=TEXT_DIM, fontsize=6.5)

        # Grey track
        ax.barh(y, 100, height=bar_h, color=SURFACE2, left=0,
                linewidth=0, zorder=1, ec="none")

        # Filled bar (category colour)
        ax.barh(y, pct, height=bar_h, color=col, alpha=0.65, left=0,
                linewidth=0, zorder=2, ec="none")

        # Brighter leading tip (last 4 pp)
        if pct > 5:
            ax.barh(y, 4, height=bar_h, color=col, alpha=1.0,
                    left=pct - 4, linewidth=0, zorder=3, ec="none")

        # Thin end marker
        ax.plot([pct, pct], [y - bar_h/2, y + bar_h/2],
                color=col, linewidth=1.5, zorder=4)

        # Percentile number (right, coloured by performance)
        ax.text(102, y, f"{pct:.0f}th", ha="left", va="center",
                color=pct_col, fontsize=8.5, fontweight="bold")

        # Category square dot
        ax.scatter(-2, y, s=26, color=col, marker="s", linewidths=0, zorder=5)

    # Reference gridlines
    for v in (25, 50, 75):
        ax.plot([v, v], [-0.7, n*spacing - 0.3],
                color=BORDER, linewidth=0.8, linestyle=":", zorder=0)
        ax.text(v, n*spacing - 0.05, str(v),
                ha="center", va="bottom", color=TEXT_DIM, fontsize=6.5)

    ax.set_xlim(-62, 120)
    ax.set_ylim(-0.7, n*spacing - 0.1)
    ax.axis("off")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    player, pool_all, pool_c1 = load_pool()
    n_pool = len(pool_all)

    print(f"  Pool: {n_pool} total ({len(pool_c1)} Czech I)  |  min {MIN_MINS} min")
    print(f"  D. Barát — {int(player['Minutes played'])} min, "
          f"Pos: {player['Position']}, Age: {int(player['Age'])}")

    bar_metric_keys = [m for m, *_ in BAR_METRICS]
    pcts = compute_percentiles(player, pool_all, bar_metric_keys)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 10.5), facecolor=BG)

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        left=0.01, right=0.99,
        top=0.97, bottom=0.025,
        height_ratios=[0.115, 0.885],
        hspace=0.22,
    )

    # ── HEADER ────────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    ax_hdr.set_facecolor(BG)
    ax_hdr.axis("off")

    ax_hdr.text(0.005, 0.94, "D. BARÁT", ha="left", va="top",
                color=TEXT, fontsize=30, fontweight="bold",
                transform=ax_hdr.transAxes)
    ax_hdr.text(0.005, 0.22,
                "Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Czech Republic",
                ha="left", va="top", color=TEXT_DIM, fontsize=9.5,
                transform=ax_hdr.transAxes)

    # Accent line
    ax_hdr.plot([0.005, 0.995], [0.06, 0.06],
                transform=ax_hdr.transAxes, color=C1_C,
                linewidth=2.0, clip_on=False)

    # Stats pills
    pills = [
        ("MINUTES",    "593"),
        ("MATCHES",    "19"),
        ("xG",         "0.93"),
        ("xA",         "0.38"),
        ("DRIBBLES/90","4.55"),
    ]
    px = 0.49
    for lbl, val in pills:
        ax_hdr.text(px, 0.88, val, ha="center", va="top",
                    color=GOLD, fontsize=14, fontweight="bold",
                    transform=ax_hdr.transAxes)
        ax_hdr.text(px, 0.18, lbl, ha="center", va="top",
                    color=TEXT_DIM, fontsize=7,
                    transform=ax_hdr.transAxes)
        px += 0.096

    # Pool note
    ax_hdr.text(0.995, 0.88, f"vs {n_pool} wide attackers",
                ha="right", va="top", color=TEXT_DIM, fontsize=8,
                transform=ax_hdr.transAxes)
    ax_hdr.text(0.995, 0.30, "Czech Fortuna Liga + Czech II  ·  ≥300 min",
                ha="right", va="top", color=TEXT_DIM, fontsize=7,
                transform=ax_hdr.transAxes)

    # ── CONTENT: 2 columns ────────────────────────────────────────────────────
    content = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[1],
        wspace=0.09,
        width_ratios=[0.40, 0.60],
    )

    # LEFT — 6 distribution subplots
    left_gs = gridspec.GridSpecFromSubplotSpec(
        6, 1,
        subplot_spec=content[0],
        hspace=0.75,
    )
    dist_axes = [fig.add_subplot(left_gs[i]) for i in range(6)]

    # Section title above first subplot
    dist_axes[0].set_title(
        "DISTRIBUTION  ·  Czech I (blue dashed)  vs  Czech I+II (grey)",
        fontsize=7.5, color=TEXT_DIM, pad=7,
        fontweight="bold", style="italic", loc="left",
    )

    draw_distributions(dist_axes, pool_all, pool_c1, player)

    # Legend below last subplot
    leg_handles = [
        Line2D([0],[0], color=POOL_C, linewidth=1.5, label="Czech I + II pool"),
        Line2D([0],[0], color=C1_C, linewidth=1.2, linestyle="--",
               label="Czech First League"),
        Line2D([0],[0], color=PLAYER_C, linewidth=1.8, label="D. Barát"),
    ]
    dist_axes[-1].legend(
        handles=leg_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.85), ncol=3,
        frameon=False, fontsize=6.5,
        labelcolor=TEXT_DIM,
    )

    # RIGHT — percentile bars
    ax_bars = fig.add_subplot(content[1])
    ax_bars.text(0.5, 1.005,
                 "PERCENTILE RANKS  ·  Czech Wide Attackers (≥300 min)",
                 ha="center", va="bottom", transform=ax_bars.transAxes,
                 color=TEXT_DIM, fontsize=8, fontweight="bold", style="italic")

    draw_percentile_bars(ax_bars, player, pcts)

    # Category legend
    cat_handles = [
        mpatches.Patch(facecolor=c, label=cat, alpha=0.8)
        for cat, c in CAT_COLOURS.items()
    ]
    ax_bars.legend(
        handles=cat_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        frameon=True, facecolor=SURFACE, edgecolor=BORDER,
        fontsize=6.5, labelcolor=TEXT,
        handlelength=0.9, framealpha=1.0, ncol=1,
    )

    # Percentile colour legend (bottom of bars)
    perf_handles = [
        mpatches.Patch(facecolor=ACCENT2,  label="≥ 85th"),
        mpatches.Patch(facecolor=C1_C,     label="70–84th"),
        mpatches.Patch(facecolor=GOLD,     label="50–69th"),
        mpatches.Patch(facecolor=ORANGE,   label="30–49th"),
        mpatches.Patch(facecolor=DANGER,   label="< 30th"),
    ]
    ax_bars.legend(
        handles=perf_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=True, facecolor=SURFACE, edgecolor=BORDER,
        fontsize=6.5, labelcolor=TEXT,
        handlelength=0.9, framealpha=1.0, ncol=5,
        title="Percentile colour", title_fontsize=6,
    )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    fig.text(0.005, 0.010, "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
             ha="left", va="bottom", color=TEXT_DIM, fontsize=7)
    fig.text(0.995, 0.010, "hradeck-scouting",
             ha="right", va="bottom", color=TEXT_DIM, fontsize=7)

    # ── SAVE ──────────────────────────────────────────────────────────────────
    png = OUT_DIR / "D_Barat_Scouting_Report.png"
    pdf = OUT_DIR / "D_Barat_Scouting_Report.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig.savefig(pdf, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {png}")
    print(f"  Saved → {pdf}")


if __name__ == "__main__":
    main()
