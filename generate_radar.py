"""
Stacked percentile radar chart generator.
Usage: python generate_radar.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
import matplotlib.patheffects as pe
from scipy.stats import percentileofscore
import os

# ── Styling ────────────────────────────────────────────────────────────────────
BG       = "#F5F0E8"
C_SHOOT  = "#C0334D"
C_ATTACK = "#D4863A"
C_PASS   = "#2A9D8F"
C_DEF    = "#264E8A"
C_TEXT   = "#1A1A2E"

GROUP_COLORS = {
    "Shooting":  C_SHOOT,
    "Attacking": C_ATTACK,
    "Passing":   C_PASS,
    "Defensive": C_DEF,
}

# ── Metrics ────────────────────────────────────────────────────────────────────
METRIC_GROUPS = {
    "Shooting": [
        ("Sh/90",  "Shots per 90"),
        ("xG/90",  "xG per 90"),
        ("G/90",   "Goals per 90"),
        ("NPG/90", "Non-penalty goals per 90"),
        ("SOT%",   "Shots on target, %"),
    ],
    "Attacking": [
        ("xA/90",    "xA per 90"),
        ("KP/90",    "Key passes per 90"),
        ("SA/90",    "Shot assists per 90"),
        ("Box P/90", "Touches in box per 90"),
        ("A/90",     "Assists per 90"),
    ],
    "Passing": [
        ("P/90",      "Passes per 90"),
        ("Fwd P/90",  "Forward passes per 90"),
        ("Prog P/90", "Progressive passes per 90"),
        ("Pass %",    "Accurate passes, %"),
        ("Fwd P%",    "Accurate forward passes, %"),
    ],
    "Defensive": [
        ("Def A/90", "Successful defensive actions per 90"),
        ("Def D/90", "Defensive duels per 90"),
        ("Def D%",   "Defensive duels won, %"),
        ("PAdj Int", "PAdj Interceptions"),
        ("PAdj Tkl", "PAdj Sliding tackles"),
    ],
}

WYSCOUT_POSITION_MAP = {
    "CF": "ST", "SS": "ST",
    "LW": "W",  "RW": "W",  "LWF": "W",  "RWF": "W",  "WF": "W",
    "AMF": "AM", "LAMF": "AM", "RAMF": "AM",
    "CMF": "CM", "LCM": "CM", "RCM": "CM", "LCMF": "CM", "RCMF": "CM",
    "DMF": "DM", "LDM": "DM", "RDM": "DM", "LDMF": "DM", "RDMF": "DM",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "GK": "GK",
}

# Comparison groups: similar positions grouped together
COMPARISON_GROUPS = {
    "ST": (["ST"], "forwards"),
    "W":  (["W", "AM"], "wingers / attacking midfielders"),
    "AM": (["W", "AM"], "wingers / attacking midfielders"),
    "CM": (["CM", "DM"], "midfielders"),
    "DM": (["CM", "DM"], "midfielders"),
    "FB": (["FB"], "fullbacks / wingbacks"),
    "CB": (["CB"], "centre-backs"),
    "GK": (["GK"], "goalkeepers"),
}

LEAGUE_LABEL = {
    "Portugal II.xlsx": "Liga Portugal 2",
    "Portugal III.xlsx": "Liga Portugal 3",
}


def get_primary_position(pos_str: str) -> str:
    """Return position group for the first listed position."""
    if pd.isna(pos_str):
        return "Other"
    first = pos_str.split(",")[0].strip()
    return WYSCOUT_POSITION_MAP.get(first, "Other")


def get_display_role(pos_str: str) -> str:
    """Return short display string of positions."""
    if pd.isna(pos_str):
        return ""
    parts = [p.strip() for p in pos_str.split(",")]
    return " / ".join(parts[:3])


def load_comparison_pool(df: pd.DataFrame, pos_group: str, min_minutes: int = 300) -> pd.DataFrame:
    """Filter dataframe to comparison group with minimum minutes."""
    groups, _ = COMPARISON_GROUPS.get(pos_group, ([pos_group], pos_group))
    df = df.copy()
    df["_pg"] = df["Position"].apply(get_primary_position)
    pool = df[(df["_pg"].isin(groups)) & (df["Minutes played"] >= min_minutes)].copy()
    return pool


def calc_percentiles(player_row: pd.Series, pool: pd.DataFrame) -> dict:
    """Calculate percentile for each metric."""
    results = {}
    for group, metrics in METRIC_GROUPS.items():
        for label, col in metrics:
            if col not in pool.columns:
                results[label] = (np.nan, np.nan)
                continue
            vals = pool[col].dropna().values
            pval = player_row[col] if col in player_row.index else np.nan
            if pd.isna(pval) or len(vals) == 0:
                results[label] = (pval, np.nan)
            else:
                pct = percentileofscore(vals, pval, kind="rank")
                results[label] = (pval, round(pct))
    return results


def format_val(v, col):
    """Format a metric value for display."""
    if pd.isna(v):
        return "—"
    if "%" in col or col in ("SOT%", "Pass %", "Fwd P%", "Def D%"):
        return f"{v:.1f}%"
    return f"{v:.2f}"


def generate_radar(player_name: str, team: str, df: pd.DataFrame,
                   league_file: str, output_dir: str = "output",
                   min_minutes: int = 300):
    os.makedirs(output_dir, exist_ok=True)

    # ── Find player ───────────────────────────────────────────────────────────
    row = df[df["Player"] == player_name]
    if len(row) == 0:
        raise ValueError(f"Player '{player_name}' not found")
    row = row.iloc[0]

    pos_group = get_primary_position(row["Position"])
    comp_groups, pos_label = COMPARISON_GROUPS.get(pos_group, ([pos_group], pos_group))
    pool = load_comparison_pool(df, pos_group, min_minutes)
    n = len(pool)

    league_label = LEAGUE_LABEL.get(league_file, league_file.replace(".xlsx", ""))
    role_display = get_display_role(row["Position"])
    age = int(row["Age"]) if not pd.isna(row.get("Age", np.nan)) else "?"
    minutes = int(row["Minutes played"])

    pcts = calc_percentiles(row, pool)

    # ── Metric order (clockwise from top) ─────────────────────────────────────
    all_metrics = []
    for group in ["Shooting", "Attacking", "Passing", "Defensive"]:
        for label, col in METRIC_GROUPS[group]:
            all_metrics.append((label, group))

    n_metrics = len(all_metrics)   # 20
    n_groups  = 4
    gap_deg   = 14.0               # degrees gap between groups
    total_bar_deg = 360.0 - n_groups * gap_deg
    bar_deg   = total_bar_deg / n_metrics  # ~13.2°

    # Build angle for each bar (center of bar, clockwise, 0 = top)
    angles_deg = []
    cursor = 0.0
    group_start_angles = {}
    group_end_angles   = {}
    prev_group = None
    for i, (label, group) in enumerate(all_metrics):
        if group != prev_group:
            if prev_group is not None:
                cursor += gap_deg
            group_start_angles[group] = cursor
            prev_group = group
        angles_deg.append(cursor + bar_deg / 2.0)
        cursor += bar_deg
        group_end_angles[group] = cursor

    # Convert to matplotlib polar (radians, 0=right, CCW positive) from
    # our (degrees, 0=top, CW positive)
    def to_polar(deg):
        return np.radians(90.0 - deg)

    angles_rad = [to_polar(d) for d in angles_deg]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 15), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Title area at top
    ax_title = fig.add_axes([0.0, 0.82, 1.0, 0.18])
    ax_title.set_facecolor(BG)
    ax_title.axis("off")

    # Radar in center
    ax = fig.add_axes([0.05, 0.28, 0.90, 0.58], polar=True)
    ax.set_facecolor(BG)

    # Legend at bottom
    ax_leg = fig.add_axes([0.02, 0.01, 0.96, 0.27])
    ax_leg.set_facecolor(BG)
    ax_leg.axis("off")

    # ── Draw radar bars ───────────────────────────────────────────────────────
    bar_width = np.radians(bar_deg * 0.82)  # slight padding within bar slot
    max_r = 100.0
    inner_r = 5.0  # small inner circle radius

    for i, (label, group) in enumerate(all_metrics):
        pct = pcts[label][1]
        color = GROUP_COLORS[group]
        r = pct if not np.isnan(pct) else 0
        theta = angles_rad[i]

        # Draw bar
        ax.bar(theta, r - inner_r, width=bar_width, bottom=inner_r,
               color=color, alpha=0.92, linewidth=0.3, edgecolor=BG, zorder=3)

        # Percentile label on bar
        if not np.isnan(pct) and r > 12:
            label_r = max(inner_r + 3, r - 8)
            ax.text(theta, label_r, str(int(pct)),
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color="white", zorder=5)

    # ── Outer arc decorations per group ───────────────────────────────────────
    arc_r = max_r + 8
    for group in ["Shooting", "Attacking", "Passing", "Defensive"]:
        color = GROUP_COLORS[group]
        start_a = group_start_angles[group]
        end_a   = group_end_angles[group]
        mid_a   = (start_a + end_a) / 2.0
        # Draw arc as thick line
        theta_range = np.linspace(to_polar(start_a + 1), to_polar(end_a - 1), 100)
        ax.plot(theta_range, [arc_r] * 100, color=color, linewidth=3.5,
                solid_capstyle="round", zorder=4)

    # ── Outer reference circles ───────────────────────────────────────────────
    for ref in [25, 50, 75]:
        theta_r = np.linspace(0, 2 * np.pi, 360)
        ax.plot(theta_r, [ref] * 360, color="#CCBFA8", linewidth=0.4,
                linestyle="--", alpha=0.5, zorder=1)

    # ── Metric labels ─────────────────────────────────────────────────────────
    label_r = max_r + 14
    for i, (label, group) in enumerate(all_metrics):
        theta = angles_rad[i]
        color = GROUP_COLORS[group]
        deg = angles_deg[i]
        # Alignment based on position
        if deg < 10 or deg > 350:
            ha, va = "center", "bottom"
        elif 10 <= deg <= 180:
            ha = "left"
            va = "center"
        elif deg == 180:
            ha, va = "center", "top"
        else:
            ha = "right"
            va = "center"

        ax.text(theta, label_r, label,
                ha=ha, va=va, fontsize=8.5,
                fontweight="bold", color=color, zorder=6)

    # ── Polar axis settings ───────────────────────────────────────────────────
    ax.set_ylim(0, max_r + 20)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor(BG)

    # Inner circle
    theta_r = np.linspace(0, 2 * np.pi, 360)
    ax.fill(theta_r, [inner_r] * 360, color=BG, zorder=2)
    ax.plot(theta_r, [inner_r] * 360, color="#CCBFA8", linewidth=1, zorder=2)

    # ── Title block ───────────────────────────────────────────────────────────
    # Player name | team
    display_name = f"{player_name.split()[0][0]}. {' '.join(player_name.split()[1:])}" \
        if len(player_name.split()) > 1 else player_name
    # Use actual player abbreviation as given
    ax_title.text(0.5, 0.90,
                  f"{player_name}  |  {team}",
                  ha="center", va="top", fontsize=22, fontweight="black",
                  color=C_TEXT, transform=ax_title.transAxes)

    ax_title.text(0.5, 0.65,
                  f"Stacked percentile radar  |  {league_label} {pos_label}, "
                  f"{min_minutes}+ minutes (n={n})",
                  ha="center", va="top", fontsize=10.5, color="#555566",
                  transform=ax_title.transAxes)

    year_line = f"2025/26  |  Role: {role_display}  |  Age {age}  |  {minutes} minutes"
    ax_title.text(0.5, 0.44, year_line,
                  ha="center", va="top", fontsize=9.5, color="#555566",
                  transform=ax_title.transAxes)

    # Profile summary
    group_avgs = {}
    for group in METRIC_GROUPS:
        vals = [pcts[lbl][1] for lbl, _ in METRIC_GROUPS[group]
                if not np.isnan(pcts[lbl][1])]
        group_avgs[group] = round(np.mean(vals)) if vals else 0

    sorted_groups = sorted(group_avgs, key=lambda g: group_avgs[g], reverse=True)
    best1, best2 = sorted_groups[0], sorted_groups[1]

    standout = sorted(
        [(lbl, pcts[lbl][1]) for g in METRIC_GROUPS for lbl, _ in METRIC_GROUPS[g]
         if not np.isnan(pcts[lbl][1])],
        key=lambda x: x[1], reverse=True
    )[:2]
    worst = sorted(
        [(lbl, pcts[lbl][1]) for g in METRIC_GROUPS for lbl, _ in METRIC_GROUPS[g]
         if not np.isnan(pcts[lbl][1])],
        key=lambda x: x[1]
    )[:1]

    profile = (
        f"Profile: strongest in {best1} ({group_avgs[best1]}p avg) and "
        f"{best2} ({group_avgs[best2]}p avg); "
        f"standout metrics are {standout[0][0]} and {standout[1][0]}, "
        f"with lower relative output in {worst[0][0]}."
    )
    ax_title.text(0.5, 0.20, profile,
                  ha="center", va="top", fontsize=8.5,
                  color=C_TEXT, fontstyle="italic",
                  transform=ax_title.transAxes,
                  wrap=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax_leg.text(0.01, 0.97,
                "METRIC GROUPS  |  actual value  ·  percentile  ·  distribution marker",
                va="top", fontsize=7, color="#888899",
                transform=ax_leg.transAxes)

    group_names = list(METRIC_GROUPS.keys())
    n_cols = 4
    col_w = 0.25

    for gi, group in enumerate(group_names):
        color = GROUP_COLORS[group]
        x0 = gi * col_w + 0.005

        # Group header with avg
        avg_p = group_avgs[group]
        ax_leg.text(x0, 0.88,
                    f"■  {group}  ·  {avg_p}p avg",
                    va="top", fontsize=9, fontweight="bold",
                    color=color, transform=ax_leg.transAxes)

        for mi, (label, col) in enumerate(METRIC_GROUPS[group]):
            y = 0.73 - mi * 0.155
            val, pct = pcts[label]
            val_str = format_val(val, label)

            # Metric name
            ax_leg.text(x0 + 0.005, y,
                        f"{col[:30]}",
                        va="center", fontsize=7.5, color=C_TEXT,
                        transform=ax_leg.transAxes)

            # Value · percentile
            ax_leg.text(x0 + 0.005, y - 0.06,
                        f"{val_str}  ·  {int(pct) if not np.isnan(pct) else '—'}p",
                        va="center", fontsize=7.5, color="#444455",
                        transform=ax_leg.transAxes)

            # Dot (percentile color)
            if not np.isnan(pct):
                dot_color = color if pct >= 50 else "#BBBBCC"
                ax_leg.scatter([x0 + col_w - 0.025], [y - 0.03],
                               s=20, color=dot_color, zorder=5,
                               transform=ax_leg.transAxes)

                # ELITE / LOW tag
                if pct >= 90:
                    ax_leg.text(x0 + col_w - 0.015, y - 0.03, "ELITE",
                                va="center", fontsize=6.5, fontweight="bold",
                                color="white",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          facecolor=color, edgecolor="none"),
                                transform=ax_leg.transAxes)
                elif pct <= 15:
                    ax_leg.text(x0 + col_w - 0.015, y - 0.03, "LOW",
                                va="center", fontsize=6.5, fontweight="bold",
                                color="white",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          facecolor="#999999", edgecolor="none"),
                                transform=ax_leg.transAxes)

    # Source
    fig.text(0.98, 0.005, f"Source: {league_file}",
             ha="right", va="bottom", fontsize=7.5, color="#888899")

    # ── Save ──────────────────────────────────────────────────────────────────
    safe_name = player_name.replace(" ", "_").replace(".", "")
    out_path = os.path.join(output_dir, f"radar_{safe_name}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    db = "/home/user/hradeck-scouting/data/Wyscout DB"
    out = "/home/user/hradeck-scouting/output/radars"

    df2 = pd.read_excel(f"{db}/Portugal II.xlsx")
    df3 = pd.read_excel(f"{db}/Portugal III.xlsx")

    players = [
        ("Kevin Pinto",  "Benfica II",  df2, "Portugal II.xlsx"),
        ("João Cruz",    "Benfica II",  df2, "Portugal II.xlsx"),
        ("J. Trevisan",  "Benfica II",  df2, "Portugal II.xlsx"),
        ("B. Souza",     "Académica",   df3, "Portugal III.xlsx"),
    ]

    for name, team, df, league_file in players:
        generate_radar(name, team, df, league_file, output_dir=out)
