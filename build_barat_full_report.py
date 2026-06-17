"""
build_barat_full_report.py — 6-page scouting report for D. Barát
Pages: Title · 1 Profile · 2 Stats · 3 Advanced · 4 Physical · 5 FCHK Model V3
Saves: reports/D_Barat_Scouting_Report_Full.pdf  (+ individual PNGs)
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
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import percentileofscore, gaussian_kde

# ── import everything from the existing report ─────────────────────────────────
from build_barat_report import (
    BG, PANEL, SURFACE2, BORDER, TEXT, TEXT_MED, TEXT_DIM,
    ACCENT, PLAYER_C, LEAGUE_C, PERF_CMAP,
    pct_colour, _off, _inner_title,
    load_data, calc_pcts, role_fit,
    draw_header, draw_profile_fit, draw_distributions, draw_bars, draw_bench_table,
    make_page2, make_page3, calc_composites, calc_war,
    BAR_METRICS, DIST_METRICS, BENCH_METRICS,
    ALL_STATS_LEFT, ALL_STATS_RIGHT,
    CLUB_C, TIER_C, DB_C,
    OUT_DIR,
)

# ── SkillCorner config ─────────────────────────────────────────────────────────

SC_PATH = "data/SkillCorner.csv"

SC_PILLS = [                              # (col, label, unit)
    ("PSV-99",                    "PSV-99",          "km/h"),
    ("Peak Velocity",             "Peak Velocity",   "km/h"),
    ("Sprint Distance P90",       "Sprint Dist/90",  "m"),
    ("Sprint Count P90",          "Sprint Count/90", ""),
    ("High Acceleration Count P90","Hi Accel/90",    ""),
]

SC_BAR_METRICS = [
    ("Distance P90",                  "Distance / 90 (m)",     "Volume"),
    ("HSR Distance P90",              "HSR Dist / 90 (m)",     "Volume"),
    ("Sprint Distance P90",           "Sprint Dist / 90 (m)",  "Volume"),
    ("PSV-99",                        "PSV-99 (km/h)",         "Speed"),
    ("Peak Velocity",                 "Peak Velocity (km/h)",  "Speed"),
    ("M/min P90",                     "Metres / min",          "Speed"),
    ("Sprint Count P90",              "Sprint Count / 90",     "Explosive"),
    ("High Acceleration Count P90",   "Hi Accel / 90",         "Explosive"),
    ("High Deceleration Count P90",   "Hi Decel / 90",         "Explosive"),
    ("Change of Direction Count P90", "CoD Count / 90",        "Explosive"),
]

SC_CAT_COLOURS = {
    "Volume":    "#2563EB",
    "Speed":     "#7C3AED",
    "Explosive": "#DC2626",
}


# ── SkillCorner loader ─────────────────────────────────────────────────────────

def load_skillcorner():
    df = pd.read_csv(SC_PATH, sep=";")
    df = df.replace("null", np.nan)
    for col in df.columns[6:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    barat_rows = df[df["Short Name"].str.strip() == "D. Barát"]
    if barat_rows.empty:
        barat_rows = df[df["Player"].str.contains("Barát", na=False)]
    if barat_rows.empty:
        raise ValueError("D. Barát not found in SkillCorner CSV")

    # If multiple stints, keep all (single row in this CSV)
    barat_sc = barat_rows.iloc[0]
    pool_sc  = df.copy()
    print(f"  SkillCorner: {len(pool_sc)} players  ·  "
          f"PSV-99={float(barat_sc['PSV-99']):.2f}  "
          f"PeakV={float(barat_sc['Peak Velocity']):.2f}  "
          f"SprintDist/90={float(barat_sc['Sprint Distance P90']):.0f}m")
    return barat_sc, pool_sc


# ── Page 4 drawing ─────────────────────────────────────────────────────────────

def draw_page4_header(ax):
    _off(ax)
    ax.add_patch(mpatches.Rectangle(
        (-0.018, 0.0), 0.007, 1.0,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(0.0, 0.97, "D. BARÁT", ha="left", va="top",
            transform=ax.transAxes, color=TEXT, fontsize=22, fontweight="bold")
    ax.text(0.0, 0.30,
            "Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  "
            "Physical Profile  ·  Page 4 of 5",
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=8)
    ax.plot([0, 1], [0.04, 0.04], transform=ax.transAxes, color=BORDER, lw=0.8)


def draw_sc_pills(ax, barat_sc, pool_sc):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axis("off")
    _inner_title(ax, "PHYSICAL HEADLINE METRICS",
                 "vs Czech First League (all positions)  ·  SkillCorner")

    n     = len(SC_PILLS)
    pad   = 0.02
    pw    = (1.0 - pad * (n + 1)) / n

    for i, (col, lbl, unit) in enumerate(SC_PILLS):
        x0 = pad + i * (pw + pad)
        pv   = float(barat_sc[col]) if pd.notna(barat_sc.get(col)) else np.nan
        vals = pool_sc[col].dropna().values
        pct  = float(np.mean(vals < pv) * 100) if (len(vals) and pd.notna(pv)) else 50.0
        c    = pct_colour(pct)

        # Card background
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.08), pw, 0.78, boxstyle="round,pad=0.006",
            facecolor=BG, edgecolor=BORDER, linewidth=0.8,
            transform=ax.transAxes, clip_on=True))

        # Colour top stripe
        ax.add_patch(mpatches.Rectangle(
            (x0, 0.78), pw, 0.08,
            facecolor=c, edgecolor="none", alpha=0.20,
            transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Rectangle(
            (x0, 0.78), 0.005, 0.08,
            facecolor=c, edgecolor="none",
            transform=ax.transAxes, clip_on=True))

        # Label
        ax.text(x0 + pw / 2, 0.82, lbl,
                ha="center", va="center", transform=ax.transAxes,
                color=TEXT_MED, fontsize=6.5, fontweight="bold")

        # Big value
        val_txt = f"{pv:.2f}" if unit == "km/h" else (f"{pv:.0f}" if unit == "m" else f"{pv:.1f}")
        if unit:
            val_txt += f" {unit}"
        ax.text(x0 + pw / 2, 0.54, val_txt,
                ha="center", va="center", transform=ax.transAxes,
                color=c, fontsize=14, fontweight="bold")

        # Mini bar
        bx0, bx1 = x0 + 0.010, x0 + pw - 0.010
        bw = bx1 - bx0
        ax.add_patch(mpatches.Rectangle(
            (bx0, 0.32), bw, 0.10,
            facecolor=SURFACE2, edgecolor="none",
            transform=ax.transAxes, clip_on=True))
        ax.add_patch(mpatches.Rectangle(
            (bx0, 0.32), bw * pct / 100, 0.10,
            facecolor=c, edgecolor="none", alpha=0.80,
            transform=ax.transAxes, clip_on=True))

        # Percentile label
        ax.text(x0 + pw / 2, 0.19, f"{pct:.0f}th percentile",
                ha="center", va="center", transform=ax.transAxes,
                color=c, fontsize=6.5, fontweight="bold")
        ax.text(x0 + pw / 2, 0.11, "vs Czech First League",
                ha="center", va="center", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=5.5)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def draw_sc_bars(ax, barat_sc, pool_sc):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)

    n       = len(SC_BAR_METRICS)
    spacing = 1.0
    bar_h   = 0.44

    _inner_title(ax, "PHYSICAL PROFILE BARS",
                 "Percentile vs Czech First League (all positions)  ·  SkillCorner")

    # Pass 1: category dividers
    prev_cat = None
    for i, (col, lbl, cat) in enumerate(SC_BAR_METRICS):
        y = (n - 1 - i) * spacing
        if cat != prev_cat:
            cc = SC_CAT_COLOURS.get(cat, TEXT_DIM)
            if prev_cat is None:
                ax.text(-21, y + spacing * 0.70, cat.upper(),
                        ha="center", va="center",
                        color=cc, fontsize=5.5, fontweight="bold")
            else:
                sep_y = y + spacing * 0.50
                ax.plot([-44, 100], [sep_y]*2, color=BORDER, lw=0.5, zorder=0)
                ax.text(-21, sep_y, cat.upper(), ha="center", va="center",
                        color=cc, fontsize=5.5, fontweight="bold",
                        bbox=dict(facecolor=BG, edgecolor="none", pad=1.8), zorder=2)
            prev_cat = cat

    # Pass 2: bars
    for i, (col, lbl, cat) in enumerate(SC_BAR_METRICS):
        y    = (n - 1 - i) * spacing
        pv   = float(barat_sc[col]) if (col in barat_sc.index and pd.notna(barat_sc[col])) else np.nan
        vals = pool_sc[col].dropna().values
        pct  = float(np.mean(vals < pv) * 100) if (len(vals) and pd.notna(pv)) else 50.0
        c    = pct_colour(pct)

        fmt  = ".2f" if col in ("PSV-99", "Peak Velocity", "M/min P90") else ".0f"
        raw_s = f"{pv:{fmt}}" if pd.notna(pv) else "—"

        ax.text(-2, y + 0.13, lbl, ha="right", va="center", color=TEXT, fontsize=7.5)
        ax.text(-2, y - 0.18, raw_s, ha="right", va="center", color=TEXT_DIM, fontsize=6.5)

        ax.barh(y, 100, height=bar_h, color=SURFACE2, left=0, lw=0, zorder=1, ec="none")
        ax.barh(y, pct, height=bar_h, color=c, alpha=0.72, left=0, lw=0, zorder=2, ec="none")
        if pct > 5:
            ax.barh(y, 4, height=bar_h, color=c, alpha=1.0,
                    left=pct - 4, lw=0, zorder=3, ec="none")
        ax.text(102, y, f"{pct:.0f}th", ha="left", va="center",
                color=c, fontsize=8, fontweight="bold")

    top = n * spacing + 0.55
    for v in (25, 50, 75):
        ax.plot([v, v], [-0.6, top - 0.25], color=BORDER, lw=0.6, ls=":", zorder=0)
        ax.text(v, top - 0.18, str(v), ha="center", va="bottom",
                color=TEXT_DIM, fontsize=6)

    ax.set_xlim(-44, 120)
    ax.set_ylim(-0.6, top)
    ax.axis("off")


def draw_sc_distributions(axes, barat_sc, pool_sc):
    dist_pairs = [
        ("PSV-99",            "PSV-99 (km/h)"),
        ("Sprint Distance P90","Sprint Dist / 90 (m)"),
    ]
    _inner_title(axes[0], "PHYSICAL DISTRIBUTIONS",
                 "Czech First League (grey)  ·  D. Barát (purple)")

    for idx, (ax, (col, label)) in enumerate(zip(axes, dist_pairs)):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_visible(False)

        vals = pool_sc[col].dropna().values
        pv   = float(barat_sc[col]) if pd.notna(barat_sc.get(col)) else np.nan

        if len(vals) < 4:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes, color=TEXT_DIM, fontsize=7)
            ax.axis("off"); continue

        lo = max(0, np.percentile(vals, 1) - 0.5)
        hi = np.percentile(vals, 99) + 0.5
        xs = np.linspace(lo, hi, 300)

        try:
            kde = gaussian_kde(vals, bw_method=0.35)
            ys  = kde(xs)
            ax.fill_between(xs, ys, alpha=0.14, color="#94A3B8")
            ax.plot(xs, ys, color="#94A3B8", linewidth=1.0)
        except Exception:
            ys = np.zeros_like(xs)

        if pd.notna(pv):
            try:
                y_at = float(gaussian_kde(vals, bw_method=0.35)([pv])[0])
            except Exception:
                y_at = 0
            ax.axvline(pv, color=PLAYER_C, linewidth=1.8, zorder=5, alpha=0.90)
            ax.scatter([pv], [y_at], color=PLAYER_C, s=28, zorder=6, linewidths=0)
            p    = float(np.mean(vals < pv) * 100)
            peak = float(np.max(ys)) if len(ys) else 1.0
            ax.text(pv, y_at + 0.05 * peak, f"{p:.0f}th",
                    ha="center", va="bottom",
                    color=pct_colour(p), fontsize=7, fontweight="bold")

        headroom = 0.28 if idx == 0 else 0.15
        peak = float(np.max(ys)) if len(ys) and np.max(ys) > 0 else 1.0
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, peak * (1 + headroom))

        ticks = np.linspace(np.ceil(lo * 2) / 2, np.floor(hi * 2) / 2, 4)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks], fontsize=5.5, color=TEXT_DIM)
        ax.tick_params(axis="x", length=2, pad=1, colors=TEXT_DIM)
        ax.tick_params(axis="y", left=False, labelleft=False)

        ax.text(-0.05, 0.5, label, ha="right", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=7, fontweight="bold")

        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(BORDER)
        ax.spines["bottom"].set_linewidth(0.6)


def make_page4(barat_sc, pool_sc):
    fig4 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)

    outer4 = gridspec.GridSpec(
        4, 1, figure=fig4,
        left=0.04, right=0.97,
        top=0.975, bottom=0.022,
        height_ratios=[0.055, 0.200, 0.460, 0.255],
        hspace=0.20,
    )

    draw_page4_header(fig4.add_subplot(outer4[0]))
    draw_sc_pills(fig4.add_subplot(outer4[1]), barat_sc, pool_sc)
    draw_sc_bars(fig4.add_subplot(outer4[2]), barat_sc, pool_sc)

    dist_gs   = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer4[3], wspace=0.40,
    )
    dist_axes = [fig4.add_subplot(dist_gs[i]) for i in range(2)]
    draw_sc_distributions(dist_axes, barat_sc, pool_sc)

    dist_axes[0].legend(
        handles=[
            Line2D([0],[0], color="#94A3B8", lw=1.3, label="Czech First League"),
            Line2D([0],[0], color=PLAYER_C,  lw=1.8, label="D. Barát"),
        ],
        loc="lower center", bbox_to_anchor=(1.2, -0.94),
        ncol=2, frameon=False, fontsize=5.8, labelcolor=TEXT_DIM, handlelength=1.2,
    )

    # Footer
    fig4.text(0.04, 0.007,
              "Data: SkillCorner  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
              ha="left", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig4.text(0.97, 0.007, "hradeck-scouting",
              ha="right", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig4.add_artist(plt.Line2D(
        [0.04, 0.97], [0.019, 0.019],
        transform=fig4.transFigure, color=BORDER, lw=0.7,
    ))

    return fig4


# ── Title / cover page ─────────────────────────────────────────────────────────

def make_title_page(n_lg, n_db):
    fig0 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    ax   = fig0.add_axes([0, 0, 1, 1])
    _off(ax)

    # Full-height left accent bar
    ax.add_patch(mpatches.Rectangle(
        (0.038, 0.0), 0.012, 1.0,
        facecolor=ACCENT, edgecolor="none",
        transform=ax.transAxes, clip_on=True, zorder=5))

    # Top section: player name + info
    ax.text(0.065, 0.90, "D. BARÁT",
            ha="left", va="top", transform=ax.transAxes,
            color=TEXT, fontsize=46, fontweight="bold")
    ax.text(0.065, 0.795,
            "SCOUTING REPORT  ·  2025/26",
            ha="left", va="top", transform=ax.transAxes,
            color=ACCENT, fontsize=13, fontweight="bold")
    ax.text(0.065, 0.745,
            "Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Czech Republic",
            ha="left", va="top", transform=ax.transAxes,
            color=TEXT_DIM, fontsize=10)

    # Thin rule under name block
    ax.plot([0.065, 0.97], [0.725, 0.725],
            transform=ax.transAxes, color=BORDER, lw=1.0)

    # Key stats pills
    pills = [("MIN","593"), ("MATCHES","19"), ("xG","0.93"),
             ("xA","0.38"), ("DRIB/90","4.55")]
    pill_y = 0.670
    px = 0.065
    pill_w, pill_h = 0.148, 0.055
    pad_between = 0.018

    for j, (lbl, val) in enumerate(pills):
        ax.add_patch(mpatches.FancyBboxPatch(
            (px, pill_y - pill_h), pill_w, pill_h,
            boxstyle="round,pad=0.008",
            facecolor=SURFACE2, edgecolor=BORDER, linewidth=0.8,
            transform=ax.transAxes, clip_on=True))
        ax.text(px + pill_w / 2, pill_y - pill_h * 0.35, val,
                ha="center", va="center", transform=ax.transAxes,
                color="#B45309", fontsize=14, fontweight="bold")
        ax.text(px + pill_w / 2, pill_y - pill_h * 0.80, lbl,
                ha="center", va="center", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=7, fontweight="bold")
        px += pill_w + pad_between

    # Contents section
    contents_y = 0.545
    ax.plot([0.065, 0.97], [contents_y, contents_y],
            transform=ax.transAxes, color=BORDER, lw=0.8)
    ax.text(0.065, contents_y - 0.016, "CONTENTS",
            ha="left", va="top", transform=ax.transAxes,
            color=ACCENT, fontsize=9, fontweight="bold")

    contents = [
        ("1", "Profile Overview",
         "Profile fit · KDE distributions · Percentile bars · Benchmark table",
         "Page 1 of 5"),
        ("2", "Statistical Analysis",
         "Peer comparison · Full statistical profile (26 Wyscout metrics)",
         "Page 2 of 5"),
        ("3", "Advanced Analytics",
         "WAR · Composite scores · Derived efficiency metrics",
         "Page 3 of 5"),
        ("4", "Physical Profile",
         "SkillCorner: sprint mechanics · speed grades · physical distributions",
         "Page 4 of 5"),
        ("5", "FCHK Model V3",
         "Component scores · Smart club closeness · Recruitment signals",
         "Page 5 of 5"),
    ]

    entry_h = 0.068
    for k, (num, title, desc, page) in enumerate(contents):
        ey = contents_y - 0.055 - k * (entry_h + 0.012)

        # Number badge
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.065, ey - 0.042), 0.038, 0.046,
            boxstyle="round,pad=0.004",
            facecolor=ACCENT, edgecolor="none",
            transform=ax.transAxes, clip_on=True))
        ax.text(0.084, ey - 0.018, num,
                ha="center", va="center", transform=ax.transAxes,
                color="white", fontsize=11, fontweight="bold")

        ax.text(0.118, ey - 0.007, title,
                ha="left", va="center", transform=ax.transAxes,
                color=TEXT, fontsize=9, fontweight="bold")
        ax.text(0.118, ey - 0.028, desc,
                ha="left", va="center", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=7.5)

        dots_x = 0.720
        ax.plot([dots_x, 0.88], [ey - 0.017, ey - 0.017],
                transform=ax.transAxes, color=BORDER, lw=0.6, ls=":")
        ax.text(0.97, ey - 0.017, page,
                ha="right", va="center", transform=ax.transAxes,
                color=TEXT_MED, fontsize=8)

        # Bottom rule for each entry
        ax.plot([0.065, 0.97], [ey - 0.048, ey - 0.048],
                transform=ax.transAxes, color=BORDER, lw=0.3)

    # Bottom block: analysis context
    info_y = 0.165
    ax.plot([0.065, 0.97], [info_y, info_y],
            transform=ax.transAxes, color=BORDER, lw=0.8)
    context_lines = [
        f"Benchmarked against:  Club pool (n=10)  ·  Czech First League (n={n_lg})  ·  "
        f"Tier 3 Global (n=2,385)  ·  Full Database (n={n_db:,})",
        "Wyscout data: Czech Fortuna Liga 2025/26  ·  SkillCorner physical data: Czech Fortuna Liga 2025/26",
        "Profile fit scores against The Athletic attacker archetypes  ·  WAR formula: 3 goals/win, 15th-pctile replacement",
    ]
    for ki, line in enumerate(context_lines):
        ax.text(0.065, info_y - 0.018 - ki * 0.022, line,
                ha="left", va="top", transform=ax.transAxes,
                color=TEXT_DIM, fontsize=7.5)

    # Footer
    ax.plot([0.038, 0.97], [0.035, 0.035],
            transform=ax.transAxes, color=BORDER, lw=0.7)
    ax.text(0.065, 0.022, "Data: Wyscout · SkillCorner · Czech Fortuna Liga 2025/26 · FCHK Scouting",
            ha="left", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=7)
    ax.text(0.97, 0.022, "June 2026",
            ha="right", va="top", transform=ax.transAxes, color=TEXT_DIM, fontsize=7)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig0


# ── Page 1 builder ─────────────────────────────────────────────────────────────

def make_page1(player, pool_cl, pool_lg, pool_tier, pool_db,
               pc_cl, pc_lg, pc_ti, pc_db, scores, n_cl, n_lg, n_ti, n_db):
    from build_barat_report import CAT_COLOURS, CLUB_C, LEAGUE_C, TIER_C
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        left=0.04, right=0.97,
        top=0.975, bottom=0.022,
        height_ratios=[0.080, 0.122, 0.614, 0.162],
        hspace=0.22,
    )

    draw_header(fig.add_subplot(outer[0]), n_lg, n_db)
    draw_profile_fit(fig.add_subplot(outer[1]), scores)

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
            Line2D([0],[0], color=LEAGUE_C,  lw=1.3, ls="--", label="Czech First League"),
            Line2D([0],[0], color=PLAYER_C,  lw=1.8, label="D. Barát"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.94),
        ncol=3, frameon=False, fontsize=5.8, labelcolor=TEXT_DIM, handlelength=1.2,
    )

    from matplotlib.colors import LinearSegmentedColormap

    ax_bars = fig.add_subplot(mid[1])
    draw_bars(ax_bars, player, pc_cl, pc_lg, pc_ti, pc_db)

    leg_bm = ax_bars.legend(
        handles=[
            Line2D([0],[0], color=CLUB_C,   lw=1.6, label=f"Club (n={n_cl})"),
            Line2D([0],[0], color=LEAGUE_C, lw=1.6, label=f"League (n={n_lg})"),
            Line2D([0],[0], color=TIER_C,   lw=1.6, label=f"Tier 3 (n={n_ti:,})"),
        ],
        loc="upper right", frameon=True, facecolor=PANEL, edgecolor=BORDER,
        fontsize=5.5, labelcolor=TEXT, handlelength=1.2, ncol=1,
        title="Benchmark lines", title_fontsize=5.0, borderpad=0.5,
    )
    ax_bars.add_artist(leg_bm)
    ax_bars.legend(
        handles=[mpatches.Patch(facecolor=c, label=cat, alpha=0.85)
                 for cat, c in CAT_COLOURS.items()],
        loc="lower right", frameon=True, facecolor=PANEL, edgecolor=BORDER,
        fontsize=5.5, labelcolor=TEXT, handlelength=0.8, ncol=1,
        title="Category", title_fontsize=5.0, borderpad=0.5,
    )

    draw_bench_table(
        fig.add_subplot(outer[3]),
        player, pc_cl, pc_lg, pc_ti, pc_db,
        n_cl, n_lg, n_ti, n_db,
    )

    fig.text(0.04, 0.007,
             "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
             ha="left", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig.text(0.97, 0.007, "hradeck-scouting",
             ha="right", va="bottom", color=TEXT_DIM, fontsize=6.5)
    fig.add_artist(plt.Line2D(
        [0.04, 0.97], [0.019, 0.019],
        transform=fig.transFigure, color=BORDER, lw=0.7,
    ))
    return fig


# ── FCHK Model V3 data ────────────────────────────────────────────────────────

FCHK_V3_PATH = "data/FCHK Model V3 - Recruitment Scores.xlsx"

MODEL_SCORES = [
    ("AttackingScore",       "Attacking"),
    ("CreationScore",        "Creation"),
    ("DefendingScore",       "Defending"),
    ("PhysicalScore",        "Physical"),
    ("BallSecurityScore",    "Ball Security"),
    ("GoalsAddedScore",      "Goals Added"),
    ("xThreatScore",         "xThreat"),
    ("MultiPhaseImpactScore","Multi-Phase"),
    ("DevelopmentValueScore","Dev. Value"),
]

SMART_CLUBS = [
    ("PozzoTradingScore",          "Pozzo Trading"),
    ("BenficaAcademyUpsideScore",  "Benfica Academy"),
    ("DortmundOpportunityScore",   "Dortmund"),
    ("PortugueseTradingScore",     "Portuguese Trading"),
    ("RedBullScore",               "Red Bull"),
    ("AZScore",                    "AZ"),
    ("ToulouseDataValueScore",     "Toulouse"),
    ("BenhamScore",                "Brentford / Benham"),
    ("CopenhagenIntegrationScore", "Copenhagen"),
    ("BrightonBrentfordScore",     "Brighton"),
]

KEY_PILLS = [
    ("CompositeRecruitmentScore", "Composite Score",    ""),
    ("ValueRecruitmentScore",     "Value Score",        ""),
    ("AgeResaleScore",            "Age Resale",         ""),
    ("DevelopmentValueScore",     "Dev. Value",         ""),
    ("SuccessProbability",        "Success Prob.",      "%"),
    ("SampleConfidence",          "Confidence",         "%"),
]


def load_fchk_model() -> dict:
    df = pd.read_excel(FCHK_V3_PATH, sheet_name="Recruitment Scores")
    row = df[df["PlayerName"] == "Daniel Barat"].iloc[0]
    return row.to_dict()


def make_page5(model: dict) -> plt.Figure:
    """FCHK Model V3 page — component scores, smart club closeness, key signals."""
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)

    outer = gridspec.GridSpec(
        5, 1, figure=fig,
        left=0.06, right=0.97,
        top=0.975, bottom=0.022,
        height_ratios=[0.070, 0.115, 0.360, 0.320, 0.115],
        hspace=0.28,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    _off(ax_hdr)
    ax_hdr.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax_hdr.transAxes,
        boxstyle="round,pad=0", fc=ACCENT, ec="none", zorder=0,
    ))
    ax_hdr.text(0.012, 0.54, "FCHK MODEL V3", transform=ax_hdr.transAxes,
                fontsize=10, fontweight="bold", color="white", va="center")
    version = str(model.get("ModelVersion", "V3")).split(" ")[0]
    run_date = str(model.get("RunDate", "2026"))[:10]
    ax_hdr.text(0.98, 0.54, f"{version}  ·  Run {run_date}",
                transform=ax_hdr.transAxes, fontsize=6.5, color="white",
                va="center", ha="right", alpha=0.85)

    # ── Score band banner ─────────────────────────────────────────────────────
    ax_band = fig.add_subplot(outer[1])
    _off(ax_band, face=SURFACE2)
    ax_band.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax_band.transAxes,
        boxstyle="round,pad=0", fc=SURFACE2, ec=BORDER, lw=0.6,
    ))

    composite = float(model.get("CompositeRecruitmentScore", 50))
    band      = str(model.get("ScoreBand", ""))
    style1    = str(model.get("PrimaryPlayerStyle", ""))
    style2    = str(model.get("SecondaryPlayerStyle", ""))
    archetype = str(model.get("ClosestArchetype", ""))
    smart_top = str(model.get("SmartClubTop3", ""))
    conf_band = str(model.get("ConfidenceBand", ""))

    band_colour = "#DC2626" if "Concern" in band else (
        "#16A34A" if "Strong" in band else "#D97706"
    )

    # Big composite number
    ax_band.text(0.03, 0.75, f"{composite:.1f}", transform=ax_band.transAxes,
                 fontsize=28, fontweight="bold", color=ACCENT, va="top")
    ax_band.text(0.03, 0.28, "Composite Recruitment Score",
                 transform=ax_band.transAxes, fontsize=6.5, color=TEXT_DIM, va="center")

    # Score band badge
    ax_band.add_patch(mpatches.FancyBboxPatch(
        (0.22, 0.52), 0.22, 0.30, transform=ax_band.transAxes,
        boxstyle="round,pad=0.02", fc=band_colour, ec="none",
    ))
    ax_band.text(0.33, 0.67, band, transform=ax_band.transAxes,
                 fontsize=6, fontweight="bold", color="white", va="center", ha="center")

    # Style and archetype
    ax_band.text(0.48, 0.75, f"{style1}  ·  {style2}",
                 transform=ax_band.transAxes, fontsize=7.5, fontweight="bold",
                 color=TEXT, va="top")
    ax_band.text(0.48, 0.50, f"Closest Archetype: {archetype}",
                 transform=ax_band.transAxes, fontsize=6.5, color=TEXT_MED, va="top")
    ax_band.text(0.48, 0.28, f"Smart Club: {smart_top}",
                 transform=ax_band.transAxes, fontsize=6, color=TEXT_DIM, va="top")

    conf_colour = "#16A34A" if "Good" in conf_band else "#D97706"
    ax_band.text(0.98, 0.28, f"● {conf_band}", transform=ax_band.transAxes,
                 fontsize=6, color=conf_colour, va="top", ha="right")

    # 6 key pills across bottom row
    pill_cols = [
        "#1D4ED8", "#1D4ED8", "#7C3AED", "#7C3AED", "#059669", "#059669",
    ]
    pill_w, pill_h = 0.145, 0.28
    gap = (1.0 - 6 * pill_w) / 7
    for i, (col_key, label, unit) in enumerate(KEY_PILLS):
        x = gap + i * (pill_w + gap)
        val = model.get(col_key, 0)
        val_str = f"{float(val):.1f}{unit}" if pd.notna(val) else "—"
        ax_band.add_patch(mpatches.FancyBboxPatch(
            (x, -0.05), pill_w, pill_h, transform=ax_band.transAxes,
            boxstyle="round,pad=0.01", fc=pill_cols[i], ec="none", clip_on=False,
        ))
        ax_band.text(x + pill_w / 2, 0.105, val_str, transform=ax_band.transAxes,
                     fontsize=8, fontweight="bold", color="white",
                     va="center", ha="center", clip_on=False)
        ax_band.text(x + pill_w / 2, -0.02, label, transform=ax_band.transAxes,
                     fontsize=5.2, color="white", va="center", ha="center",
                     alpha=0.88, clip_on=False)

    # ── Component scores (horizontal bar chart) ───────────────────────────────
    ax_comp = fig.add_subplot(outer[2])
    _off(ax_comp, face=BG)
    _inner_title(ax_comp, "COMPONENT SCORES  (0–100 percentile scale)")

    labels   = [lbl for _, lbl in MODEL_SCORES]
    vals     = [float(model.get(key, 50)) for key, _ in MODEL_SCORES]
    n_bars   = len(labels)
    bar_h    = 0.55
    y_pos    = list(range(n_bars - 1, -1, -1))

    # Background tracks
    for y in y_pos:
        ax_comp.barh(y, 100, left=0, height=bar_h, color=SURFACE2, zorder=0)

    COMP_CMAP = LinearSegmentedColormap.from_list("comp", ["#DC2626", "#F59E0B", "#16A34A"])
    bar_colours = [COMP_CMAP(v / 100) for v in vals]
    for i, (y, v, c) in enumerate(zip(y_pos, vals, bar_colours)):
        ax_comp.barh(y, v, height=bar_h, color=c, zorder=2)
        ax_comp.text(v + 1.5, y, f"{v:.1f}", va="center", fontsize=6.5, color=TEXT)

    ax_comp.set_yticks(y_pos)
    ax_comp.set_yticklabels(labels, fontsize=7, color=TEXT)
    ax_comp.set_xlim(0, 108)
    ax_comp.set_ylim(-0.6, n_bars - 0.4)
    ax_comp.axvline(50, color=BORDER, lw=0.8, ls="--", zorder=1)
    ax_comp.tick_params(axis="x", labelsize=6, colors=TEXT_DIM)
    ax_comp.set_xlabel("Score (0–100)", fontsize=6.5, color=TEXT_DIM)
    for sp in ax_comp.spines.values():
        sp.set_visible(False)
    ax_comp.tick_params(axis="y", length=0)

    # ── Smart Club Closeness (bar chart) ──────────────────────────────────────
    ax_clubs = fig.add_subplot(outer[3])
    _off(ax_clubs, face=BG)
    _inner_title(ax_clubs, "SMART CLUB CLOSENESS")

    sc_labels = [lbl for _, lbl in SMART_CLUBS]
    sc_vals   = [float(model.get(key, 0)) for key, _ in SMART_CLUBS]
    sc_n      = len(sc_labels)
    sc_y      = list(range(sc_n - 1, -1, -1))

    # Top3 highlighted
    top3_keys = {k for k, _ in SMART_CLUBS[:3]}
    closeness_tier = str(model.get("SmartClubClosenessTier", ""))
    tier_colour = "#16A34A" if "Strong" in closeness_tier else (
        "#D97706" if "Medium" in closeness_tier else TEXT_DIM
    )

    for y in sc_y:
        ax_clubs.barh(y, 100, left=0, height=bar_h, color=SURFACE2, zorder=0)
    for i, (y, v, (key, lbl)) in enumerate(zip(sc_y, sc_vals, SMART_CLUBS)):
        c = ACCENT if i < 3 else "#94A3B8"
        ax_clubs.barh(y, v, height=bar_h, color=c, zorder=2)
        ax_clubs.text(v + 1.5, y, f"{v:.1f}", va="center", fontsize=6.5, color=TEXT)

    ax_clubs.set_yticks(sc_y)
    ax_clubs.set_yticklabels(sc_labels, fontsize=7, color=TEXT)
    ax_clubs.set_xlim(0, 108)
    ax_clubs.set_ylim(-0.6, sc_n - 0.4)
    ax_clubs.axvline(50, color=BORDER, lw=0.8, ls="--", zorder=1)
    ax_clubs.tick_params(axis="x", labelsize=6, colors=TEXT_DIM)
    ax_clubs.set_xlabel("Closeness Score (0–100)", fontsize=6.5, color=TEXT_DIM)
    for sp in ax_clubs.spines.values():
        sp.set_visible(False)
    ax_clubs.tick_params(axis="y", length=0)

    tier_txt = f"Tier: {closeness_tier}" if closeness_tier else ""
    ax_clubs.text(0.98, 1.04, tier_txt, transform=ax_clubs.transAxes,
                  fontsize=6.5, color=tier_colour, ha="right", va="bottom")

    # ── Risk / flags ──────────────────────────────────────────────────────────
    ax_risk = fig.add_subplot(outer[4])
    _off(ax_risk, face=PANEL)
    ax_risk.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax_risk.transAxes,
        boxstyle="round,pad=0", fc=PANEL, ec=BORDER, lw=0.6,
    ))

    flags = [
        ("Wage Risk",         model.get("WageRisk", "—")),
        ("Fee Risk",          model.get("FeeRisk", "—")),
        ("Availability",      model.get("AvailabilityFlag", "—")),
        ("Minutes Flag",      model.get("MinutesRiskFlag", "—")),
        ("Data Coverage",     model.get("DataCoverageFlag", "—")),
        ("Deal Realism",      f"{float(model.get('DealRealismScore', 0)):.1f}"),
    ]
    col_w = 1.0 / len(flags)
    for i, (flag_lbl, flag_val) in enumerate(flags):
        x = (i + 0.5) * col_w
        ax_risk.text(x, 0.68, str(flag_val), transform=ax_risk.transAxes,
                     fontsize=6.8, fontweight="bold", color=TEXT, va="center", ha="center")
        ax_risk.text(x, 0.28, flag_lbl, transform=ax_risk.transAxes,
                     fontsize=5.5, color=TEXT_DIM, va="center", ha="center")

    # Footer note
    ax_risk.text(0.5, -0.18,
                 "FCHK Model V3  ·  Data: Wyscout / IMPECT 2025/26  ·  FCHK Scouting",
                 transform=ax_risk.transAxes, fontsize=5.5, color=TEXT_DIM,
                 va="top", ha="center")

    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading Wyscout data …")
    player, pool_cl, pool_lg, pool_tier, pool_db = load_data()
    n_cl = len(pool_cl); n_lg = len(pool_lg)
    n_ti = len(pool_tier); n_db = len(pool_db)
    print(f"  Club={n_cl}  League={n_lg}  Tier3={n_ti}  DB={n_db}")

    keys = list({m for m, *_ in BAR_METRICS} | {m for m, _ in BENCH_METRICS})
    keys2 = list({m for m, *_ in ALL_STATS_LEFT + ALL_STATS_RIGHT} | set(keys))
    all_keys = list(set(keys + keys2))

    pc_cl = calc_pcts(player, pool_cl,   all_keys)
    pc_lg = calc_pcts(player, pool_lg,   all_keys)
    pc_ti = calc_pcts(player, pool_tier, all_keys)
    pc_db = calc_pcts(player, pool_db,   all_keys)
    scores = role_fit(player, pool_lg)
    print(f"  Roles: { {r: f'{v:.0f}%' for r,v in scores.items()} }")

    print("  Computing composites & WAR …")
    player_vals, db_pcts3, lg_ranks3 = calc_composites(player, pool_db, pool_lg)
    war_data = calc_war(player, pool_lg, pool_db)
    print(f"  WAR = {war_data['war']:+.3f}  "
          f"(League {war_data['rank_lg']}/{war_data['n_lg']}  ·  "
          f"DB {war_data['pct_db']:.0f}th pctile)")

    print("Loading SkillCorner data …")
    barat_sc, pool_sc = load_skillcorner()

    print("Loading FCHK Model V3 …")
    model_data = load_fchk_model()
    print(f"  Composite={model_data['CompositeRecruitmentScore']:.1f}  "
          f"Band={model_data['ScoreBand']}  "
          f"Style={model_data['PrimaryPlayerStyle']}")

    # ── Build pages ────────────────────────────────────────────────────────────
    print("Building pages …")
    fig0 = make_title_page(n_lg, n_db)
    fig1 = make_page1(player, pool_cl, pool_lg, pool_tier, pool_db,
                      pc_cl, pc_lg, pc_ti, pc_db, scores,
                      n_cl, n_lg, n_ti, n_db)
    fig2 = make_page2(player, pool_cl, pool_lg, pool_tier, pool_db,
                      pc_cl, pc_lg, pc_ti, pc_db)
    fig3 = make_page3(player, pool_db, pool_lg, player_vals, db_pcts3, lg_ranks3, war_data)
    fig4 = make_page4(barat_sc, pool_sc)
    fig5 = make_page5(model_data)

    # ── Save ───────────────────────────────────────────────────────────────────
    png_title = OUT_DIR / "D_Barat_Scouting_Report_Title.png"
    png_p4    = OUT_DIR / "D_Barat_Scouting_Report_P4.png"
    png_p5    = OUT_DIR / "D_Barat_Scouting_Report_P5.png"
    pdf_full  = OUT_DIR / "D_Barat_Scouting_Report_Full.pdf"

    fig0.savefig(png_title, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig4.savefig(png_p4,    dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig5.savefig(png_p5,    dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")

    with PdfPages(pdf_full) as pp:
        for fig in [fig0, fig1, fig2, fig3, fig4, fig5]:
            pp.savefig(fig, bbox_inches="tight", facecolor=BG, edgecolor="none")

    for fig in [fig0, fig1, fig2, fig3, fig4, fig5]:
        plt.close(fig)

    print(f"  Saved → {png_title}")
    print(f"  Saved → {png_p4}")
    print(f"  Saved → {png_p5}")
    print(f"  Saved → {pdf_full}  (6 pages: Title + Pages 1–5)")


if __name__ == "__main__":
    main()
