"""
build_barat_full_report.py — 7-page FC Hradec Králové scouting report for D. Barát
Pages: Cover · 1 Profile · 2 Stats · 3 Advanced · 4 Physical · 5 FCHK Model V3 · 6 Verdict
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
from scipy.stats import percentileofscore, gaussian_kde, norm

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

# ── FC Hradec Králové brand identity ──────────────────────────────────────────
import build_barat_report as _bbr

FCHK_BLUE   = "#003EA6"   # FCHK royal blue
FCHK_DARK   = "#002068"   # deep FCHK navy
FCHK_GOLD   = "#F5C400"   # FCHK gold/yellow
FCHK_GOLD_D = "#C49B00"   # darker gold for text on light bg

# Monkey-patch the base module so all draw_* functions inherit FCHK colors
ACCENT   = FCHK_BLUE
PLAYER_C = FCHK_GOLD
_bbr.ACCENT   = FCHK_BLUE
_bbr.PLAYER_C = FCHK_GOLD
_bbr.PANEL    = "#EEF3FF"
_bbr.SURFACE2 = "#DDE7FF"
PANEL    = "#EEF3FF"
SURFACE2 = "#DDE7FF"


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
    "Volume":    FCHK_BLUE,
    "Speed":     "#7C3AED",
    "Explosive": "#DC2626",
}

TOTAL_PAGES = 6   # numbered content pages (excludes cover)
REPORT_REF  = "SCR-2026-047"


# ── FCHK footer + gold top stripe ─────────────────────────────────────────────

def _fchk_footer(fig, page_num: int | str):
    """Overlay FCHK branded footer gold rule + branding text on any figure."""
    fig.add_artist(plt.Line2D(
        [0.038, 0.965], [0.022, 0.022],
        transform=fig.transFigure, color=FCHK_GOLD, lw=1.0, zorder=20,
    ))
    fig.text(0.038, 0.011, "FC HRADEC KRÁLOVÉ  ·  SCOUTING INTELLIGENCE",
             ha="left", va="center", fontsize=5.5, color=FCHK_DARK,
             fontweight="bold", transform=fig.transFigure, zorder=20)
    fig.text(0.502, 0.011, "CONFIDENTIAL — INTERNAL USE ONLY",
             ha="center", va="center", fontsize=5.2, color="#6B7280",
             transform=fig.transFigure, zorder=20)
    if isinstance(page_num, int):
        pg_str = f"PAGE {page_num} OF {TOTAL_PAGES}  ·  JUNE 2026"
    else:
        pg_str = f"{page_num}  ·  JUNE 2026"
    fig.text(0.965, 0.011, pg_str,
             ha="right", va="center", fontsize=5.5, color=FCHK_DARK,
             fontweight="bold", transform=fig.transFigure, zorder=20)
    # thin gold top stripe
    fig.add_artist(plt.Line2D(
        [0, 1], [0.978, 0.978],
        transform=fig.transFigure, color=FCHK_GOLD, lw=2.5, zorder=20,
    ))


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
            "Physical Profile  ·  Page 4 of 6",
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


# ── Cover page ────────────────────────────────────────────────────────────────

def make_title_page(n_lg, n_db):
    fig0 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    ax   = fig0.add_axes([0, 0, 1, 1])
    _off(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ── Left panel: FC Hradec Králové brand block ──────────────────────────────
    PANEL_W = 0.28
    ax.add_patch(mpatches.Rectangle(
        (0, 0), PANEL_W, 1.0,
        facecolor=FCHK_DARK, edgecolor="none",
        transform=ax.transAxes, clip_on=True, zorder=5))
    # gold top stripe on left panel
    ax.add_patch(mpatches.Rectangle(
        (0, 0.978), PANEL_W, 0.022,
        facecolor=FCHK_GOLD, edgecolor="none",
        transform=ax.transAxes, clip_on=True, zorder=6))

    # Club name
    ax.text(PANEL_W / 2, 0.86, "FC", ha="center", va="center",
            transform=ax.transAxes, color=FCHK_GOLD,
            fontsize=22, fontweight="bold", zorder=10)
    ax.text(PANEL_W / 2, 0.80, "HRADEC", ha="center", va="center",
            transform=ax.transAxes, color="white",
            fontsize=19, fontweight="bold", zorder=10)
    ax.text(PANEL_W / 2, 0.745, "KRÁLOVÉ", ha="center", va="center",
            transform=ax.transAxes, color="white",
            fontsize=19, fontweight="bold", zorder=10)

    # Gold separator
    ax.plot([0.03, PANEL_W - 0.03], [0.720, 0.720],
            transform=ax.transAxes, color=FCHK_GOLD, lw=1.5, zorder=10)

    ax.text(PANEL_W / 2, 0.685, "SCOUTING", ha="center", va="center",
            transform=ax.transAxes, color=FCHK_GOLD,
            fontsize=10.5, fontweight="bold", zorder=10)
    ax.text(PANEL_W / 2, 0.650, "INTELLIGENCE", ha="center", va="center",
            transform=ax.transAxes, color="#BFCCE8",
            fontsize=8, fontweight="bold", zorder=10)

    # Report details on left panel
    ax.text(PANEL_W / 2, 0.560, "PLAYER ASSESSMENT", ha="center", va="center",
            transform=ax.transAxes, color="#BFCCE8", fontsize=7, zorder=10)
    ax.text(PANEL_W / 2, 0.530, "REPORT", ha="center", va="center",
            transform=ax.transAxes, color="#BFCCE8", fontsize=7, zorder=10)
    ax.plot([0.03, PANEL_W - 0.03], [0.510, 0.510],
            transform=ax.transAxes, color="#3A5A9F", lw=0.6, zorder=10)

    # Reference & season
    for i, txt in enumerate(["REF: " + REPORT_REF, "SEASON: 2025/26 (Final)", "DATE: June 2026",
                              "WINDOW: Summer 2026 → 2026/27"]):
        ax.text(PANEL_W / 2, 0.475 - i * 0.040, txt, ha="center", va="center",
                transform=ax.transAxes, color="#BFCCE8", fontsize=6.5, zorder=10)

    # Vertical position label
    ax.text(PANEL_W / 2, 0.155, "WIDE ATTACKER", ha="center", va="center",
            transform=ax.transAxes, color=FCHK_GOLD, fontsize=8,
            fontweight="bold", zorder=10)
    ax.text(PANEL_W / 2, 0.118, "LAMF  ·  LW  ·  LWB", ha="center", va="center",
            transform=ax.transAxes, color="#BFCCE8", fontsize=7, zorder=10)
    ax.plot([0.03, PANEL_W - 0.03], [0.095, 0.095],
            transform=ax.transAxes, color="#3A5A9F", lw=0.6, zorder=10)
    ax.text(PANEL_W / 2, 0.065, "1. FC SLOVÁCKO", ha="center", va="center",
            transform=ax.transAxes, color="white", fontsize=7,
            fontweight="bold", zorder=10)
    ax.text(PANEL_W / 2, 0.040, "Czech Fortuna Liga", ha="center", va="center",
            transform=ax.transAxes, color="#BFCCE8", fontsize=6.5, zorder=10)

    # ── Right section: player name + stats + contents ──────────────────────────
    RX = PANEL_W + 0.04   # right section left margin

    # CONFIDENTIAL stamp top-right
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.86, 0.945), 0.105, 0.030,
        boxstyle="round,pad=0.005", facecolor="#FEF3C7",
        edgecolor=FCHK_GOLD_D, linewidth=0.8,
        transform=ax.transAxes, clip_on=True, zorder=10))
    ax.text(0.9125, 0.960, "CONFIDENTIAL", ha="center", va="center",
            transform=ax.transAxes, color=FCHK_GOLD_D,
            fontsize=5.5, fontweight="bold", zorder=11)

    # Player name block
    ax.text(RX, 0.900, "D. BARÁT", ha="left", va="top",
            transform=ax.transAxes, color=FCHK_DARK,
            fontsize=46, fontweight="bold")
    ax.text(RX, 0.802, "PLAYER ASSESSMENT REPORT  ·  SUMMER 2026  ·  2026/27 WINDOW",
            ha="left", va="top", transform=ax.transAxes,
            color=FCHK_BLUE, fontsize=11.5, fontweight="bold")
    ax.text(RX, 0.758,
            "Slovácko  ·  Czech Fortuna Liga  ·  Age 19  ·  Czech Republic",
            ha="left", va="top", transform=ax.transAxes,
            color=TEXT_DIM, fontsize=9.5)

    # Gold rule under name
    ax.plot([RX, 0.97], [0.734, 0.734],
            transform=ax.transAxes, color=FCHK_GOLD, lw=1.8)

    # Key stat pills (5 pills in FCHK blue)
    pills = [("MIN", "593"), ("MATCHES", "19"), ("xG", "0.93"),
             ("xA", "0.38"), ("DRIB/90", "4.55")]
    pill_w = (0.97 - RX - 0.016 * 4) / 5
    for j, (lbl, val) in enumerate(pills):
        px = RX + j * (pill_w + 0.016)
        ax.add_patch(mpatches.FancyBboxPatch(
            (px, 0.672), pill_w, 0.052,
            boxstyle="round,pad=0.006", facecolor=FCHK_BLUE,
            edgecolor="none", transform=ax.transAxes, clip_on=True, zorder=8))
        ax.text(px + pill_w / 2, 0.706, val, ha="center", va="center",
                transform=ax.transAxes, color=FCHK_GOLD,
                fontsize=13, fontweight="bold", zorder=9)
        ax.text(px + pill_w / 2, 0.683, lbl, ha="center", va="center",
                transform=ax.transAxes, color="white",
                fontsize=6.2, fontweight="bold", zorder=9)

    # ── Contents table ─────────────────────────────────────────────────────────
    cy = 0.636
    ax.plot([RX, 0.97], [cy, cy], transform=ax.transAxes, color=BORDER, lw=0.8)
    ax.text(RX, cy - 0.014, "CONTENTS", ha="left", va="top",
            transform=ax.transAxes, color=FCHK_BLUE,
            fontsize=8.5, fontweight="bold")

    contents = [
        ("1", "Profile Overview",
         "Profile fit · KDE distributions · Percentile bars · Benchmark table",
         "Page 1 of 6"),
        ("2", "Statistical Analysis",
         "Peer comparison · Full statistical profile · 26 Wyscout metrics",
         "Page 2 of 6"),
        ("3", "Advanced Analytics",
         "WAR · Composite scores · Derived efficiency metrics",
         "Page 3 of 6"),
        ("4", "Physical Profile",
         "SkillCorner: sprint mechanics · speed grades · distributions",
         "Page 4 of 6"),
        ("5", "FCHK Model V3",
         "Component scores · Smart club closeness · Recruitment signals",
         "Page 5 of 6"),
        ("6", "Analyst Verdict",
         "Recruitment recommendation · Strengths & concerns · Next steps",
         "Page 6 of 6"),
    ]
    entry_h = 0.060
    for k, (num, title, desc, page) in enumerate(contents):
        ey = cy - 0.050 - k * (entry_h + 0.010)

        num_c = FCHK_BLUE if k < 5 else FCHK_DARK
        ax.add_patch(mpatches.FancyBboxPatch(
            (RX, ey - 0.038), 0.034, 0.040,
            boxstyle="round,pad=0.004", facecolor=num_c, edgecolor="none",
            transform=ax.transAxes, clip_on=True))
        ax.text(RX + 0.017, ey - 0.016, num, ha="center", va="center",
                transform=ax.transAxes, color="white",
                fontsize=10, fontweight="bold")
        ax.text(RX + 0.048, ey - 0.006, title, ha="left", va="center",
                transform=ax.transAxes, color=TEXT,
                fontsize=8.5, fontweight="bold")
        ax.text(RX + 0.048, ey - 0.025, desc, ha="left", va="center",
                transform=ax.transAxes, color=TEXT_DIM, fontsize=7)
        ax.plot([0.70, 0.88], [ey - 0.015, ey - 0.015],
                transform=ax.transAxes, color=BORDER, lw=0.5, ls=":")
        ax.text(0.97, ey - 0.015, page, ha="right", va="center",
                transform=ax.transAxes, color=TEXT_MED, fontsize=7.5)
        ax.plot([RX, 0.97], [ey - 0.043, ey - 0.043],
                transform=ax.transAxes, color=BORDER, lw=0.25)

    # ── Methodology context ────────────────────────────────────────────────────
    my = 0.116
    ax.plot([RX, 0.97], [my, my], transform=ax.transAxes, color=BORDER, lw=0.7)
    context_lines = [
        f"Benchmarks: Club (n=10)  ·  Czech First League (n={n_lg})  ·  Tier-3 Global (n=2,385)  ·  Full DB (n={n_db:,})",
        "Data: Wyscout (Czech Fortuna Liga 2025/26, season complete)  ·  SkillCorner physical tracking (Czech FL 2025/26 final)",
        "Profile fit: The Athletic attacker archetypes  ·  WAR: 3 goals/win, 15th-pctile replacement  ·  Prepared: Summer 2026",
    ]
    for ki, line in enumerate(context_lines):
        ax.text(RX, my - 0.017 - ki * 0.022, line, ha="left", va="top",
                transform=ax.transAxes, color=TEXT_DIM, fontsize=7)

    _fchk_footer(fig0, "COVER")
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

    # 6 rows: header | band | pills | comp scores | smart clubs | risk flags
    outer = gridspec.GridSpec(
        6, 1, figure=fig,
        left=0.06, right=0.97,
        top=0.975, bottom=0.022,
        height_ratios=[0.060, 0.110, 0.068, 0.365, 0.322, 0.055],
        hspace=0.25,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    _off(ax_hdr)
    ax_hdr.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax_hdr.transAxes,
        boxstyle="round,pad=0", fc=FCHK_DARK, ec="none", zorder=0,
    ))
    ax_hdr.text(0.012, 0.54, "FCHK MODEL V3  —  RECRUITMENT SCORING",
                transform=ax_hdr.transAxes, fontsize=9.5,
                fontweight="bold", color="white", va="center")
    version = str(model.get("ModelVersion", "V3")).split(" ")[0]
    run_date = str(model.get("RunDate", "2026"))[:10]
    ax_hdr.text(0.98, 0.54, f"{version}  ·  Run {run_date}  ·  2025/26 Final",
                transform=ax_hdr.transAxes, fontsize=6.5, color=FCHK_GOLD,
                va="center", ha="right")

    # ── Score band banner (composite + band badge + style + archetype) ────────
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

    ax_band.text(0.03, 0.80, f"{composite:.1f}", transform=ax_band.transAxes,
                 fontsize=26, fontweight="bold", color=ACCENT, va="top")
    ax_band.text(0.03, 0.20, "Composite Recruitment Score",
                 transform=ax_band.transAxes, fontsize=6.5, color=TEXT_DIM, va="center")

    ax_band.add_patch(mpatches.FancyBboxPatch(
        (0.22, 0.38), 0.24, 0.46, transform=ax_band.transAxes,
        boxstyle="round,pad=0.02", fc=band_colour, ec="none",
    ))
    ax_band.text(0.34, 0.62, band, transform=ax_band.transAxes,
                 fontsize=5.8, fontweight="bold", color="white",
                 va="center", ha="center")

    ax_band.text(0.50, 0.82, f"{style1}  ·  {style2}",
                 transform=ax_band.transAxes, fontsize=7.2, fontweight="bold",
                 color=TEXT, va="top")
    ax_band.text(0.50, 0.52, f"Archetype: {archetype}",
                 transform=ax_band.transAxes, fontsize=6.5, color=TEXT_MED, va="top")
    ax_band.text(0.50, 0.24, f"Smart clubs: {smart_top}",
                 transform=ax_band.transAxes, fontsize=5.8, color=TEXT_DIM, va="top")
    conf_colour = "#16A34A" if "Good" in conf_band else "#D97706"
    ax_band.text(0.98, 0.24, f"● {conf_band}", transform=ax_band.transAxes,
                 fontsize=6, color=conf_colour, va="top", ha="right")

    # ── Key pills (own dedicated row — no overflow) ───────────────────────────
    ax_pills = fig.add_subplot(outer[2])
    _off(ax_pills, face=BG)
    pill_cols = [FCHK_BLUE, FCHK_BLUE, "#7C3AED", "#7C3AED", "#059669", "#059669"]
    pill_w = 0.145
    gap    = (1.0 - 6 * pill_w) / 7
    for i, (col_key, label, unit) in enumerate(KEY_PILLS):
        x   = gap + i * (pill_w + gap)
        val = model.get(col_key, 0)
        val_str = f"{float(val):.1f}{unit}" if pd.notna(val) else "—"
        ax_pills.add_patch(mpatches.FancyBboxPatch(
            (x, 0.06), pill_w, 0.88,
            boxstyle="round,pad=0.010", fc=pill_cols[i], ec="none",
            transform=ax_pills.transAxes, clip_on=True,
        ))
        ax_pills.text(x + pill_w / 2, 0.66, val_str,
                      transform=ax_pills.transAxes, fontsize=9,
                      fontweight="bold", color="white", va="center", ha="center")
        ax_pills.text(x + pill_w / 2, 0.24, label,
                      transform=ax_pills.transAxes, fontsize=5.5,
                      color="white", va="center", ha="center", alpha=0.90)

    # ── Component scores (horizontal bar chart) ───────────────────────────────
    ax_comp = fig.add_subplot(outer[3])
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
    ax_clubs = fig.add_subplot(outer[4])
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
    ax_risk = fig.add_subplot(outer[5])
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


# ── Analyst Verdict page ───────────────────────────────────────────────────────

VERDICT        = "MONITOR"
VERDICT_COLOUR = "#D97706"   # amber = watch/monitor
VERDICT_DESC   = (
    "The 2025/26 season is complete. Barát's elite athletic profile (Top 7% Czech "
    "league sprint distance) and exceptional development ceiling (Dev. Value 76.9) "
    "make him a priority monitor entering the summer 2026/27 window. The model "
    "'wrong role' flag suggests Slovácko underutilises him — a club offering "
    "wider positional freedom could unlock meaningfully higher output. Act now."
)
STRENGTHS = [
    "Elite athleticism  —  93rd pct sprint distance, 92nd pct sprint count (2025/26 final)",
    "Age trajectory  —  Dev. Value 76.9, Age Resale 92.1; maximum upside at 19.7",
    "Dribble volume  —  4.55 dribbles/90, direct carrying threat in transition",
    "Smart Club fit  —  Strong alignment with Pozzo / Benfica academy trading model",
    "Accessible deal  —  Czech Tier 1, lower fee risk, realistic 2026/27 window",
]
CONCERNS = [
    "Role misalignment  —  FCHK Model V3 flags 'Concern / wrong role' for 2025/26",
    "Creative output  —  xA 0.38, xG 0.93 over 593 min; below wide-attacker norms",
    "WAR rank 37/99  —  Slightly below Czech league median for wide attackers",
    "Sample size  —  593 min over season; would benefit from ≥900 min for projection",
    "No video review  —  Live / video assessment not yet completed",
]
NEXT_STEPS = [
    "Request 2025/26 full-season video package (min. 8 matches, focus transitions)",
    "Season complete — re-run FCHK Model V3 with full 2025/26 dataset now available",
    "Submit formal interest to agent by 31 July 2026 for 2026/27 registration",
    "Schedule pre-season assessment visit (Slovácko training camp, July–Aug 2026)",
    "Priority: Summer 2026 window — confirm contract status and transfer fee",
]
TRANSFER_FLAGS = [
    ("Fee Range",     "€100–300k est."),
    ("Wage Risk",     "Lower"),
    ("Fee Risk",      "Potentially realistic"),
    ("Availability",  "Needs check"),
    ("Deal Realism",  "62 / 100"),
    ("Video Review",  "Pending"),
]


def make_verdict_page() -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    outer = gridspec.GridSpec(
        5, 1, figure=fig,
        left=0.06, right=0.97,
        top=0.970, bottom=0.030,
        height_ratios=[0.060, 0.185, 0.420, 0.155, 0.115],
        hspace=0.24,
    )

    # ── Page header ────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    _off(ax_hdr)
    ax_hdr.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax_hdr.transAxes,
        boxstyle="round,pad=0", fc=FCHK_DARK, ec="none"))
    ax_hdr.text(0.012, 0.54, "ANALYST VERDICT  &  RECRUITMENT RECOMMENDATION",
                transform=ax_hdr.transAxes, fontsize=9,
                fontweight="bold", color="white", va="center")
    ax_hdr.text(0.98, 0.54, f"D. BARÁT  ·  {REPORT_REF}",
                transform=ax_hdr.transAxes, fontsize=6.5,
                color=FCHK_GOLD, va="center", ha="right")

    # ── Verdict badge + executive summary ──────────────────────────────────────
    ax_top = fig.add_subplot(outer[1])
    _off(ax_top, face=PANEL)
    ax_top.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax_top.transAxes,
        boxstyle="round,pad=0", fc=PANEL, ec=BORDER, lw=0.6))

    # Big verdict badge
    ax_top.add_patch(mpatches.FancyBboxPatch(
        (0.012, 0.08), 0.185, 0.84,
        boxstyle="round,pad=0.012", fc=VERDICT_COLOUR, ec="none",
        transform=ax_top.transAxes))
    ax_top.text(0.1045, 0.82, "VERDICT", ha="center", va="center",
                transform=ax_top.transAxes, fontsize=7,
                fontweight="bold", color="white", alpha=0.85)
    ax_top.text(0.1045, 0.50, VERDICT, ha="center", va="center",
                transform=ax_top.transAxes, fontsize=22,
                fontweight="bold", color="white")
    ax_top.text(0.1045, 0.22, "WATCH & DEVELOP", ha="center", va="center",
                transform=ax_top.transAxes, fontsize=6.5,
                fontweight="bold", color="white", alpha=0.85)

    # Executive summary text
    ax_top.text(0.21, 0.88, "EXECUTIVE SUMMARY", ha="left", va="top",
                transform=ax_top.transAxes, fontsize=7.5,
                fontweight="bold", color=FCHK_DARK)
    ax_top.plot([0.21, 0.98], [0.84, 0.84],
                transform=ax_top.transAxes, color=FCHK_GOLD, lw=1.0)

    # Word-wrap the summary manually across lines
    words = VERDICT_DESC.split()
    lines_out, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        if len(test) > 85:
            lines_out.append(line)
            line = w
        else:
            line = test
    lines_out.append(line)
    for li, ln in enumerate(lines_out):
        ax_top.text(0.21, 0.80 - li * 0.095, ln, ha="left", va="top",
                    transform=ax_top.transAxes, fontsize=7.5, color=TEXT)

    # ── Three-column panel: Strengths | Concerns | Next Steps ─────────────────
    ax_cols = fig.add_subplot(outer[2])
    _off(ax_cols)
    col_titles = ["STRENGTHS", "CONCERNS", "NEXT STEPS"]
    col_items  = [STRENGTHS, CONCERNS, NEXT_STEPS]
    col_colours= [FCHK_BLUE, "#DC2626", FCHK_DARK]
    col_icons  = ["✓", "✗", "→"]
    col_w      = 1 / 3

    for ci, (ct, items, cc, icon) in enumerate(
            zip(col_titles, col_items, col_colours, col_icons)):
        cx = ci * col_w

        # Column background
        ax_cols.add_patch(mpatches.FancyBboxPatch(
            (cx + 0.005, 0.0), col_w - 0.010, 1.0,
            boxstyle="round,pad=0.008", fc=PANEL if ci % 2 == 0 else BG,
            ec=BORDER, lw=0.5, transform=ax_cols.transAxes))

        # Column header bar
        ax_cols.add_patch(mpatches.Rectangle(
            (cx + 0.005, 0.875), col_w - 0.010, 0.125,
            fc=cc, ec="none", transform=ax_cols.transAxes))
        ax_cols.text(cx + col_w / 2, 0.940, ct, ha="center", va="center",
                     transform=ax_cols.transAxes,
                     fontsize=7.5, fontweight="bold", color="white")

        for ii, item in enumerate(items[:5]):
            # Split item into bold lead and plain body at "  —  "
            parts = item.split("  —  ", 1)
            lead  = parts[0].strip()
            body  = parts[1].strip() if len(parts) > 1 else ""
            iy    = 0.840 - ii * 0.165

            ax_cols.text(cx + 0.018, iy, icon, ha="left", va="top",
                         transform=ax_cols.transAxes,
                         fontsize=8, color=cc, fontweight="bold")
            ax_cols.text(cx + 0.040, iy, lead, ha="left", va="top",
                         transform=ax_cols.transAxes,
                         fontsize=6.8, fontweight="bold", color=TEXT)
            if body:
                # wrap body text
                words2 = body.split()
                blines, bl = [], ""
                for w in words2:
                    t = (bl + " " + w).strip()
                    if len(t) > 42:
                        blines.append(bl)
                        bl = w
                    else:
                        bl = t
                blines.append(bl)
                for bi, bln in enumerate(blines[:2]):
                    ax_cols.text(cx + 0.040, iy - 0.055 - bi * 0.048,
                                 bln, ha="left", va="top",
                                 transform=ax_cols.transAxes,
                                 fontsize=6.2, color=TEXT_DIM)

    # ── Transfer / risk assessment ─────────────────────────────────────────────
    ax_risk = fig.add_subplot(outer[3])
    _off(ax_risk, face=PANEL)
    ax_risk.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax_risk.transAxes,
        boxstyle="round,pad=0", fc=PANEL, ec=BORDER, lw=0.6))
    _inner_title(ax_risk, "TRANSFER & RISK ASSESSMENT")

    col_w2 = 1.0 / len(TRANSFER_FLAGS)
    for ti, (flag_lbl, flag_val) in enumerate(TRANSFER_FLAGS):
        tx = (ti + 0.5) * col_w2
        ax_risk.text(tx, 0.62, flag_val, ha="center", va="center",
                     transform=ax_risk.transAxes, fontsize=9,
                     fontweight="bold", color=FCHK_DARK)
        ax_risk.text(tx, 0.28, flag_lbl, ha="center", va="center",
                     transform=ax_risk.transAxes, fontsize=6,
                     color=TEXT_DIM)
        if ti < len(TRANSFER_FLAGS) - 1:
            ax_risk.plot([col_w2 * (ti + 1)] * 2, [0.12, 0.88],
                         transform=ax_risk.transAxes,
                         color=BORDER, lw=0.5)

    # ── Sign-off ───────────────────────────────────────────────────────────────
    ax_sign = fig.add_subplot(outer[4])
    _off(ax_sign)

    ax_sign.plot([0, 1], [0.80, 0.80],
                 transform=ax_sign.transAxes, color=FCHK_GOLD, lw=1.0)
    ax_sign.text(0.0, 0.62, "This report is produced by the FC Hradec Králové "
                 "Scouting Intelligence department for the Summer 2026 / 2026-27 "
                 "recruitment window. For internal use only. All scores are "
                 "model-derived and must be combined with live observation before "
                 "any recruitment action is taken.",
                 transform=ax_sign.transAxes, fontsize=6.5, color=TEXT_DIM,
                 va="top", wrap=True)

    # Signature line
    ax_sign.plot([0.0, 0.28], [0.10, 0.10],
                 transform=ax_sign.transAxes, color=BORDER, lw=0.7)
    ax_sign.text(0.0, 0.04, "Analyst  ·  FC Hradec Králové Scouting Dept.",
                 transform=ax_sign.transAxes, fontsize=6, color=TEXT_DIM)
    ax_sign.plot([0.72, 1.0], [0.10, 0.10],
                 transform=ax_sign.transAxes, color=BORDER, lw=0.7)
    ax_sign.text(1.0, 0.04, "Date  ·  June 2026",
                 transform=ax_sign.transAxes, fontsize=6, color=TEXT_DIM,
                 ha="right")

    _fchk_footer(fig, TOTAL_PAGES)
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
    fig6 = make_verdict_page()

    # Apply FCHK footer to all content pages (pages built by build_barat_report)
    for pg_num, fig_pg in enumerate([fig1, fig2, fig3, fig4, fig5, fig6], start=1):
        _fchk_footer(fig_pg, pg_num)

    # ── Save ───────────────────────────────────────────────────────────────────
    _sv = lambda f, n: f.savefig(
        OUT_DIR / n, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    png_title = OUT_DIR / "D_Barat_Scouting_Report_Title.png"
    pdf_full  = OUT_DIR / "D_Barat_Scouting_Report_Full.pdf"

    _sv(fig0, "D_Barat_Scouting_Report_Title.png")
    _sv(fig1, "D_Barat_Scouting_Report.png")
    _sv(fig2, "D_Barat_Scouting_Report_P2.png")
    _sv(fig3, "D_Barat_Scouting_Report_P3.png")
    _sv(fig4, "D_Barat_Scouting_Report_P4.png")
    _sv(fig5, "D_Barat_Scouting_Report_P5.png")
    _sv(fig6, "D_Barat_Scouting_Report_P6.png")

    with PdfPages(pdf_full) as pp:
        for fig in [fig0, fig1, fig2, fig3, fig4, fig5, fig6]:
            pp.savefig(fig, bbox_inches="tight", facecolor=BG, edgecolor="none")

    for fig in [fig0, fig1, fig2, fig3, fig4, fig5, fig6]:
        plt.close(fig)

    print(f"  Saved → {png_title} + P1–P6 PNGs")
    print(f"  Saved → {pdf_full}  (7 pages: Cover + Pages 1–6)")


if __name__ == "__main__":
    main()
