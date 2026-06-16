"""
build_barat_report.py
─────────────────────
One-pager scouting data report for D. Barát (Slovácko, Czech Fortuna Liga)
Outputs: reports/D_Barat_Scouting_Report.png  (and .pdf)
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
from matplotlib.patches import FancyBboxPatch, Wedge, Arc
from matplotlib.lines import Line2D
from scipy.stats import percentileofscore
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────────
BG        = "#090C18"
SURFACE   = "#111827"
SURFACE2  = "#1E293B"
BORDER    = "#2D3748"
ACCENT    = "#3B82F6"   # blue
ACCENT2   = "#10B981"   # green
GOLD      = "#F59E0B"
DANGER    = "#EF4444"
MUTED     = "#6B7280"
WHITE     = "#F9FAFB"
TEXT_DIM  = "#9CA3AF"

# ── Player position comparison pool ───────────────────────────────────────────
WIDE_ATK_POS = {"LAMF","RAMF","LW","RW","LWF","RWF","AMF","LWB","RWB"}
MIN_MINS     = 300    # min minutes to enter benchmark pool

# ── Metrics for the report ─────────────────────────────────────────────────────
RADAR_METRICS = [
    ("Dribbles per 90",                "Dribbles\n/90"),
    ("Successful dribbles, %",         "Dribble\nSuccess%"),
    ("Crosses per 90",                 "Crosses\n/90"),
    ("Accurate crosses, %",            "Cross\nAcc%"),
    ("Progressive runs per 90",        "Prog.\nRuns/90"),
    ("Touches in box per 90",          "Box\nTouches"),
    ("Shot assists per 90",            "Shot\nAssists"),
    ("xA per 90",                      "xA/90"),
    ("Offensive duels won, %",         "Off.Duel\nWon%"),
    ("Successful defensive actions per 90", "Def.Actions\n/90"),
]

BAR_METRICS = [
    # (col, label, category)
    ("xG per 90",                       "xG / 90",                  "Threat"),
    ("Shots per 90",                    "Shots / 90",               "Threat"),
    ("Touches in box per 90",           "Touches in Box / 90",      "Threat"),
    ("Dribbles per 90",                 "Dribbles / 90",            "Carrying"),
    ("Successful dribbles, %",          "Dribble Success %",        "Carrying"),
    ("Progressive runs per 90",         "Progressive Runs / 90",    "Carrying"),
    ("Crosses per 90",                  "Crosses / 90",             "Creation"),
    ("Accurate crosses, %",             "Cross Accuracy %",         "Creation"),
    ("xA per 90",                       "xA / 90",                  "Creation"),
    ("Shot assists per 90",             "Shot Assists / 90",        "Creation"),
    ("Key passes per 90",               "Key Passes / 90",          "Creation"),
    ("Offensive duels won, %",          "Off. Duels Won %",         "Duels"),
    ("Defensive duels won, %",          "Def. Duels Won %",         "Duels"),
    ("Successful defensive actions per 90","Def. Actions / 90",     "Defending"),
]

CAT_COLOURS = {
    "Threat":    "#EF4444",
    "Carrying":  "#F59E0B",
    "Creation":  "#10B981",
    "Duels":     "#3B82F6",
    "Defending": "#8B5CF6",
}

# ── Load data ──────────────────────────────────────────────────────────────────

def load_pool() -> tuple[pd.Series, pd.DataFrame]:
    df_czech = pd.read_excel("data/Wyscout DB/Czech.xlsx")
    df_czech2 = pd.read_excel("data/Wyscout DB/Czech II.xlsx")
    df_czech["_league"] = "Czech First League"
    df_czech2["_league"] = "Czech II"
    df = pd.concat([df_czech, df_czech2], ignore_index=True)

    # Coerce
    for col in df.columns:
        if col not in {"Player","Team","Team within selected timeframe","Position",
                       "Age","Market value","Contract expires","Birth country",
                       "Passport country","Foot","On loan","_league"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Primary position
    df["_pos1"] = df["Position"].astype(str).str.split(",").str[0].str.strip()

    # Get player
    player = df[df["Player"].astype(str).str.startswith("D. Bar")].iloc[0].copy()

    # Build pool: same position family, min minutes
    pool = df[
        df["_pos1"].isin(WIDE_ATK_POS) &
        (df["Minutes played"].fillna(0) >= MIN_MINS)
    ].copy().reset_index(drop=True)

    # Make sure player is in pool even if < MIN_MINS
    if not pool["Player"].astype(str).str.startswith("D. Bar").any():
        pool = pd.concat([pool, player.to_frame().T], ignore_index=True)

    return player, pool


def compute_percentiles(player: pd.Series, pool: pd.DataFrame,
                        metrics: list[str]) -> dict[str, float]:
    pcts = {}
    for m in metrics:
        if m not in pool.columns or m not in player.index:
            pcts[m] = 50.0
            continue
        vals = pool[m].dropna().values
        pv   = float(player.get(m, np.nan))
        if np.isnan(pv) or len(vals) == 0:
            pcts[m] = 50.0
        else:
            pcts[m] = round(percentileofscore(vals, pv, kind="rank"), 1)
    return pcts


# ── Drawing helpers ────────────────────────────────────────────────────────────

def pct_colour(pct: float) -> str:
    if pct >= 85:  return ACCENT2
    if pct >= 70:  return ACCENT
    if pct >= 50:  return GOLD
    if pct >= 30:  return "#FB923C"
    return DANGER


def draw_radar(ax, pcts: list[float], labels: list[str]):
    N = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    values = [p / 100 for p in pcts]
    values += values[:1]

    # Grid rings
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ring_x = [r * np.cos(a) for a in np.linspace(0, 2*np.pi, 200)]
        ring_y = [r * np.sin(a) for a in np.linspace(0, 2*np.pi, 200)]
        ax.plot(ring_x, ring_y, color=BORDER, linewidth=0.6, zorder=1)
        if r < 1.0:
            ax.text(0, r + 0.02, f"{int(r*100)}", ha="center", va="bottom",
                    color=MUTED, fontsize=5.5)

    # Spokes
    for a in angles[:-1]:
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color=BORDER,
                linewidth=0.6, zorder=1)

    # Fill
    xs = [v * np.cos(a) for v, a in zip(values, angles)]
    ys = [v * np.sin(a) for v, a in zip(values, angles)]
    ax.fill(xs, ys, alpha=0.25, color=ACCENT, zorder=2)
    ax.plot(xs, ys, color=ACCENT, linewidth=1.6, zorder=3)

    # Dots
    for x, y, p in zip(xs[:-1], ys[:-1], pcts):
        ax.scatter(x, y, s=28, color=pct_colour(p), zorder=4, linewidths=0)

    # Labels
    for a, lbl in zip(angles[:-1], labels):
        x = 1.19 * np.cos(a)
        y = 1.19 * np.sin(a)
        ha = "center"
        if np.cos(a) > 0.15:  ha = "left"
        elif np.cos(a) < -0.15: ha = "right"
        ax.text(x, y, lbl, ha=ha, va="center", color=TEXT_DIM,
                fontsize=5.8, linespacing=1.2)

    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.38, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(SURFACE)


def draw_percentile_bars(ax, bar_data: list[tuple[str, str, float, str, float]]):
    """bar_data = [(label, category, percentile, raw_str, raw_val)]"""
    ax.set_facecolor(SURFACE)
    n = len(bar_data)
    bar_h   = 0.52
    spacing = 1.0

    for i, (label, cat, pct, raw, _) in enumerate(bar_data):
        y = (n - 1 - i) * spacing
        colour = CAT_COLOURS.get(cat, ACCENT)

        # Background track
        ax.barh(y, 100, height=bar_h, color=SURFACE2, zorder=1,
                left=0, linewidth=0)
        # Value bar
        ax.barh(y, pct, height=bar_h, color=colour, alpha=0.85, zorder=2,
                left=0, linewidth=0)
        # Percentile marker line
        ax.plot([pct, pct], [y - bar_h/2, y + bar_h/2],
                color=WHITE, linewidth=1.2, zorder=3)

        # Labels
        ax.text(-1.5, y, label, ha="right", va="center",
                color=WHITE, fontsize=7, fontweight="normal")
        ax.text(101.5, y, f"{pct:.0f}th", ha="left", va="center",
                color=pct_colour(pct), fontsize=7, fontweight="bold")
        ax.text(102 + 12, y, raw, ha="left", va="center",
                color=TEXT_DIM, fontsize=6.5)

        # Category dot
        ax.scatter(-0.5, y, s=22, color=colour, zorder=5,
                   marker="s", linewidths=0)

    ax.set_xlim(-55, 130)
    ax.set_ylim(-0.6, n * spacing - 0.4)
    ax.axis("off")


def draw_benchmark(ax, pool: pd.DataFrame, player: pd.Series,
                   metrics: list[tuple[str, str]]):
    """Top-5 players in pool by combined score, show in table."""
    ax.set_facecolor(SURFACE)
    ax.axis("off")

    score_metrics = [m for m, _ in metrics if m in pool.columns]
    pool2 = pool.copy()
    for m in score_metrics:
        mu  = pool2[m].mean()
        sig = pool2[m].std() or 1e-9
        pool2[f"_z_{m}"] = (pool2[m] - mu) / sig
    z_cols = [f"_z_{m}" for m in score_metrics]
    pool2["_score"] = pool2[z_cols].mean(axis=1)

    top5 = pool2.nlargest(6, "_score")
    # remove player themselves if present
    top5 = top5[~top5["Player"].astype(str).str.startswith("D. Bar")].head(5)

    col_labels = ["Player", "Team"] + [lbl for _, lbl in metrics[:5]]
    col_data   = []
    for _, row in top5.iterrows():
        r = [str(row.get("Player",""))[:18], str(row.get("Team",""))[:14]]
        for m, _ in metrics[:5]:
            v = row.get(m, np.nan)
            r.append(f"{v:.2f}" if pd.notna(v) else "—")
        col_data.append(r)

    # Player's own row
    p_row = [str(player.get("Player",""))[:18], str(player.get("Team within selected timeframe",""))[:14]]
    for m, _ in metrics[:5]:
        v = player.get(m, np.nan)
        p_row.append(f"{v:.2f}" if pd.notna(v) else "—")

    ax.text(0.5, 0.98, "BENCHMARK — Top peers (Czech leagues, similar pos.)",
            ha="center", va="top", transform=ax.transAxes,
            color=TEXT_DIM, fontsize=7, style="italic")

    xs = [0.0, 0.26, 0.44, 0.55, 0.66, 0.77, 0.88]
    headers = col_labels

    # header row
    for x, h in zip(xs, headers):
        ax.text(x, 0.88, h, ha="left", va="top", transform=ax.transAxes,
                color=GOLD, fontsize=6.5, fontweight="bold")
    ax.plot([0, 1], [0.86, 0.86], transform=ax.transAxes, color=BORDER, linewidth=0.5, clip_on=False)

    # data rows
    row_ys = [0.76, 0.64, 0.52, 0.40, 0.28]
    for row_data, ry in zip(col_data, row_ys):
        for x, cell in zip(xs, row_data):
            ax.text(x, ry, cell, ha="left", va="top", transform=ax.transAxes,
                    color=WHITE, fontsize=6.5)

    # Separator
    ax.plot([0, 1], [0.14, 0.14], transform=ax.transAxes, color=ACCENT, linewidth=0.5, linestyle="--", clip_on=False)
    # Player row
    for x, cell in zip(xs, p_row):
        ax.text(x, 0.07, cell, ha="left", va="top", transform=ax.transAxes,
                color=ACCENT, fontsize=6.5, fontweight="bold")
    ax.text(0.0, 0.01, "▶ D. Barát", ha="left", va="top",
            transform=ax.transAxes, color=ACCENT, fontsize=6, style="italic")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    player, pool = load_pool()
    n_pool = len(pool)

    print(f"  Pool size: {n_pool} players  |  Min {MIN_MINS} min")
    print(f"  D. Barát — {int(player['Minutes played'])} min, "
          f"Pos: {player['Position']}, Age: {int(player['Age'])}")

    # Percentiles
    all_metrics = list({m for m, *_ in BAR_METRICS} |
                       {m for m, _ in RADAR_METRICS})
    pcts = compute_percentiles(player, pool, all_metrics)

    bar_data = []
    for m, lbl, cat in BAR_METRICS:
        raw_val = float(player.get(m, np.nan))
        raw_str = f"{raw_val:.2f}" if pd.notna(raw_val) else "—"
        if "%" in lbl:
            raw_str = f"{raw_val:.1f}%"
        bar_data.append((lbl, cat, pcts.get(m, 50.0), raw_str, raw_val))

    radar_pcts  = [pcts.get(m, 50.0) for m, _ in RADAR_METRICS]
    radar_labels = [lbl for _, lbl in RADAR_METRICS]

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    outer = gridspec.GridSpec(1, 1, figure=fig,
                              left=0.01, right=0.99,
                              top=0.96, bottom=0.02)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=outer[0],
        hspace=0.28, wspace=0.08,
        height_ratios=[0.18, 0.82],
        width_ratios=[0.36, 0.30, 0.34],
    )

    # ── Header row ─────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(inner[0, :])
    ax_hdr.set_facecolor(BG)
    ax_hdr.axis("off")

    # Divider line
    ax_hdr.axhline(y=0.08, xmin=0, xmax=1, color=ACCENT, linewidth=1.5)

    # Player name
    ax_hdr.text(0.01, 0.95, "D. BARÁT", ha="left", va="top",
                color=WHITE, fontsize=28, fontweight="bold",
                transform=ax_hdr.transAxes)
    ax_hdr.text(0.01, 0.35, "Slovácko  ·  Czech Fortuna Liga  ·  LAMF / LW / LWB  ·  Age 19  ·  Czech Republic",
                ha="left", va="top", color=TEXT_DIM, fontsize=9,
                transform=ax_hdr.transAxes)

    # Stats pills right side
    pills = [
        ("MINUTES", "593"),
        ("MATCHES", "19"),
        ("xG", "0.93"),
        ("xA", "0.38"),
        ("DRIBBLES/90", "4.55"),
    ]
    pill_x = 0.52
    for label, val in pills:
        ax_hdr.text(pill_x, 0.85, val, ha="center", va="center",
                    color=GOLD, fontsize=13, fontweight="bold",
                    transform=ax_hdr.transAxes)
        ax_hdr.text(pill_x, 0.30, label, ha="center", va="center",
                    color=MUTED, fontsize=6.5, transform=ax_hdr.transAxes)
        pill_x += 0.096

    # Pool context
    ax_hdr.text(0.99, 0.75, f"Benchmarked vs {n_pool} wide attackers",
                ha="right", va="top", color=MUTED, fontsize=8,
                transform=ax_hdr.transAxes)
    ax_hdr.text(0.99, 0.38, "Czech Fortuna Liga + Czech II  ·  min 300 min",
                ha="right", va="top", color=MUTED, fontsize=7,
                transform=ax_hdr.transAxes)

    # ── Percentile bars ────────────────────────────────────────────────────────
    ax_bars = fig.add_subplot(inner[1, 0])
    ax_bars.set_facecolor(SURFACE)

    # Section title
    ax_bars.text(0.5, 1.02, "PERCENTILE RANKS", ha="center", va="bottom",
                 transform=ax_bars.transAxes, color=TEXT_DIM,
                 fontsize=8, fontweight="bold", style="italic")

    draw_percentile_bars(ax_bars, bar_data)

    # Legend for categories
    legend_elements = [
        mpatches.Patch(facecolor=c, label=cat)
        for cat, c in CAT_COLOURS.items()
    ]
    ax_bars.legend(handles=legend_elements, loc="lower right",
                   facecolor=SURFACE2, edgecolor=BORDER,
                   fontsize=5.5, labelcolor=WHITE,
                   handlelength=0.8, framealpha=0.8,
                   ncol=1)

    # ── Radar chart ───────────────────────────────────────────────────────────
    ax_radar = fig.add_subplot(inner[1, 1])
    ax_radar.set_facecolor(SURFACE)

    ax_radar.text(0.5, 1.02, "ROLE PROFILE RADAR", ha="center", va="bottom",
                  transform=ax_radar.transAxes, color=TEXT_DIM,
                  fontsize=8, fontweight="bold", style="italic")

    draw_radar(ax_radar, radar_pcts, radar_labels)

    # Percentile rings legend
    for pct, colour, lbl in [(85, ACCENT2, "≥85"), (70, ACCENT, "70–84"),
                              (50, GOLD, "50–69"), (30, "#FB923C", "30–49"),
                              (0, DANGER, "<30")]:
        pass  # embedded in dots

    ring_legend = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=ACCENT2,
               markersize=5, label='≥85th'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=ACCENT,
               markersize=5, label='70–84th'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=GOLD,
               markersize=5, label='50–69th'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor="#FB923C",
               markersize=5, label='30–49th'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=DANGER,
               markersize=5, label='<30th'),
    ]
    ax_radar.legend(handles=ring_legend, loc="lower center",
                    bbox_to_anchor=(0.5, -0.14), ncol=5,
                    facecolor=SURFACE2, edgecolor=BORDER,
                    fontsize=5.5, labelcolor=WHITE,
                    handlelength=0.5, framealpha=0.8)

    # ── Benchmark panel ───────────────────────────────────────────────────────
    ax_bench = fig.add_subplot(inner[1, 2])
    ax_bench.set_facecolor(SURFACE)

    bench_metrics = [
        ("Dribbles per 90",               "Drib/90"),
        ("Crosses per 90",                "Cross/90"),
        ("Shot assists per 90",           "SA/90"),
        ("xA per 90",                     "xA/90"),
        ("Progressive runs per 90",       "ProgR/90"),
    ]
    draw_benchmark(ax_bench, pool, player, bench_metrics)

    # ── Key insight box ───────────────────────────────────────────────────────
    # Bottom of benchmark panel
    top_drib = (pcts.get("Dribbles per 90", 50))
    top_cross = (pcts.get("Crosses per 90", 50))
    top_def  = (pcts.get("Successful defensive actions per 90", 50))

    insights = []
    if top_drib >= 75:
        insights.append(f"Top {100-top_drib:.0f}% dribbler in peer group")
    if top_cross >= 75:
        insights.append(f"Top {100-top_cross:.0f}% crosser in peer group")
    if top_def >= 75:
        insights.append(f"Top {100-top_def:.0f}% for defensive actions")
    if player.get("Age", 99) <= 20:
        insights.append("Age 19 — outstanding sample for development profile")

    ax_bench.text(0.5, -0.04, "  ·  ".join(insights) if insights else "",
                  ha="center", va="top", transform=ax_bench.transAxes,
                  color=ACCENT2, fontsize=6.5, style="italic", wrap=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.01, 0.005, "Data: Wyscout  ·  Czech Fortuna Liga 2025/26  ·  FCHK Scouting",
             ha="left", va="bottom", color=MUTED, fontsize=7)
    fig.text(0.99, 0.005, "hradeck-scouting",
             ha="right", va="bottom", color=MUTED, fontsize=7)

    # ── Save ──────────────────────────────────────────────────────────────────
    png_path = OUT_DIR / "D_Barat_Scouting_Report.png"
    pdf_path = OUT_DIR / "D_Barat_Scouting_Report.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    fig.savefig(pdf_path, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {png_path}")
    print(f"  Saved → {pdf_path}")


if __name__ == "__main__":
    main()
