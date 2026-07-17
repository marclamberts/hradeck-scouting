"""
build_vinicius_report.py  —  FC Hradec Králové scouting report for Kahuan Vinicius
Karviná  ·  Czech Fortuna Liga 2025/26  ·  CF  ·  Age 22  ·  Brazil

Monkey-patches build_barat_report's player/config constants (position pool,
role archetypes, data source, labels) and calls main() to produce a 4-page
"Full" PDF: cover/contents + profile fit + peer comparison + WAR/composites.

Output: reports/K_Vinicius_Scouting_Report_Full.pdf
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

# ── 2. Data source / league labels (Czech top flight — same file as D. Barát) ──
_bbr.P_DATA_FILE      = "Czech.xlsx"
_bbr.P_TEAM_DEFAULT   = "Karviná"
_bbr.P_LEAGUE_LONG    = "Czech Fortuna Liga"
_bbr.P_LEAGUE_SHORT   = "Czech First League"
_bbr.P_SEASON         = "2025/26"
_bbr.P_POS_PLURAL     = "strikers"
_bbr.P_POS_SINGULAR   = "striker"
_bbr.P_ARCHETYPE_NOTE = "Role Archetypes"

# ── 3. Player config ────────────────────────────────────────────────────────────
_bbr.P_HEADER_NAME     = "K. VINICIUS"
_bbr.P_HEADER_SUBTITLE = ("Karviná  ·  Czech Fortuna Liga 2025/26  ·  "
                          "CF  ·  Age 22  ·  Brazil")
_bbr.P_PAGE2_SUBTITLE  = ("Karviná  ·  Czech Fortuna Liga  ·  "
                          "CF  ·  Age 22  ·  Page 2 of 3")
_bbr.P_PAGE3_SUBTITLE  = ("Karviná  ·  Czech Fortuna Liga  ·  "
                          "CF  ·  Age 22  ·  Page 3 of 3")
_bbr.P_PILLS           = [("MIN","619"),("MATCHES","11"),("xG","3.52"),("xA","0.12"),("DRIB/90","0.87")]
_bbr.P_WYSCOUT_FILTER  = "Kahuan Vinicius"
_bbr.P_LEGEND_LABEL    = "K. Vinicius"

# ── 4. Cover page ────────────────────────────────────────────────────────────────
_bbr.P_COVER_ENABLE   = True
_bbr.P_COVER_SUBTITLE = _bbr.P_HEADER_SUBTITLE
_bbr.P_COVER_MONTH    = "July 2026"

_bbr.P_OUT_PREFIX = "K_Vinicius_Scouting_Report"


if __name__ == "__main__":
    _bbr.main()
