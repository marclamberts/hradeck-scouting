"""
build_marcolino_report.py  —  FC Hradec Králové scouting report for A. Do Marcolino
Ústí nad Labem  ·  Czech National Football League 2025/26  ·  CF  ·  Age 24  ·  Gabon

Monkey-patches build_barat_report's player/config constants (position pool,
role archetypes, data source, labels) and calls main() to produce a 4-page
"Full" PDF: cover/contents + profile fit + peer comparison + WAR/composites.

Output: reports/A_Marcolino_Scouting_Report_Full.pdf
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import build_barat_report as _bbr

# ── 1. Position pool + CF role archetypes ──────────────────────────────────────
_bbr.WIDE_ATK_POS = {"CF"}

_bbr.ROLE_BLUEPRINTS = {
    "Finisher": {
        "Goals per 90": 0.35, "xG per 90": 0.25,
        "Shots per 90": 0.20, "Touches in box per 90": 0.20,
    },
    "Target Man": {
        "Aerial duels won, %": 0.40, "Shots per 90": 0.20,
        "Goals per 90": 0.15, "Touches in box per 90": 0.25,
    },
    "Carrier": {
        "Dribbles per 90": 0.30, "Successful dribbles, %": 0.20,
        "Progressive runs per 90": 0.30, "Offensive duels won, %": 0.20,
    },
    "Creative CF": {
        "Key passes per 90": 0.30, "xA per 90": 0.30,
        "Shot assists per 90": 0.25, "Passes per 90": 0.15,
    },
    "Second Striker": {
        "xG per 90": 0.25, "xA per 90": 0.25,
        "Dribbles per 90": 0.25, "Progressive runs per 90": 0.25,
    },
    "Pressing CF": {
        "Successful defensive actions per 90": 0.40,
        "Interceptions per 90": 0.30, "Offensive duels won, %": 0.30,
    },
}

# ── 2. Data source / league labels (Czech II — 2nd tier) ───────────────────────
_bbr.P_DATA_FILE      = "Czech II.xlsx"
_bbr.P_TEAM_DEFAULT   = "Ústí"
_bbr.P_LEAGUE_LONG    = "Czech National Football League"
_bbr.P_LEAGUE_SHORT   = "Czech II"
_bbr.P_SEASON         = "2025/26"
_bbr.P_POS_PLURAL     = "strikers"
_bbr.P_POS_SINGULAR   = "striker"
_bbr.P_ARCHETYPE_NOTE = "Role Archetypes"

# ── 3. Player config ────────────────────────────────────────────────────────────
_bbr.P_HEADER_NAME     = "A. DO MARCOLINO"
_bbr.P_HEADER_SUBTITLE = ("Ústí nad Labem  ·  Czech National Football League 2025/26  ·  "
                          "CF  ·  Age 24  ·  Gabon")
_bbr.P_PAGE2_SUBTITLE  = ("Ústí nad Labem  ·  Czech National Football League  ·  "
                          "CF  ·  Age 24  ·  Page 2 of 3")
_bbr.P_PAGE3_SUBTITLE  = ("Ústí nad Labem  ·  Czech National Football League  ·  "
                          "CF  ·  Age 24  ·  Page 3 of 3")
_bbr.P_PILLS           = [("MIN","872"),("MATCHES","13"),("xG","4.97"),("xA","0.45"),("DRIB/90","2.27")]
_bbr.P_WYSCOUT_FILTER  = "A. Do Marcolino"
_bbr.P_LEGEND_LABEL    = "A. Do Marcolino"

# ── 4. Cover page ────────────────────────────────────────────────────────────────
_bbr.P_COVER_ENABLE   = True
_bbr.P_COVER_SUBTITLE = _bbr.P_HEADER_SUBTITLE
_bbr.P_COVER_MONTH    = "July 2026"

_bbr.P_OUT_PREFIX = "A_Marcolino_Scouting_Report"


if __name__ == "__main__":
    _bbr.main()
