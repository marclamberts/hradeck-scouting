"""
build_milic_report.py  —  FC Hradec Králové scouting report for N. Milić
Karviná  ·  Czech Fortuna Liga 2025/26  ·  LB / LCB  ·  Age 22  ·  Slovenia

Monkey-patches build_barat_report's player/config constants (position pool,
role archetypes, data source, labels) and calls main() to produce a 4-page
"Full" PDF: cover/contents + profile fit + peer comparison + WAR/composites.

Output: reports/N_Milic_Scouting_Report_Full.pdf
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import build_barat_report as _bbr

# ── 1. Position pool + fullback/CB role archetypes ─────────────────────────────
_bbr.WIDE_ATK_POS = {"LB", "LCB"}

_bbr.ROLE_BLUEPRINTS = {
    "Ball-Playing FB": {
        "Accurate passes, %": 0.25, "Progressive passes per 90": 0.35,
        "Passes per 90": 0.15, "Progressive runs per 90": 0.25,
    },
    "Overlapper": {
        "Progressive runs per 90": 0.30, "Dribbles per 90": 0.20,
        "Crosses per 90": 0.25, "Accurate crosses, %": 0.25,
    },
    "Stopper": {
        "Defensive duels won, %": 0.40, "Successful defensive actions per 90": 0.35,
        "Interceptions per 90": 0.25,
    },
    "Aerial Presence": {
        "Aerial duels won, %": 0.55, "Aerial duels per 90": 0.25,
        "Defensive duels won, %": 0.20,
    },
    "Duel Winner": {
        "Duels won, %": 0.40, "Offensive duels won, %": 0.30,
        "Defensive duels won, %": 0.30,
    },
    "Creative Outlet": {
        "Key passes per 90": 0.30, "xA per 90": 0.30,
        "Crosses per 90": 0.20, "Accurate crosses, %": 0.20,
    },
}

# ── 2. Data source / league labels (Czech top flight — same file as D. Barát) ──
_bbr.P_DATA_FILE      = "Czech.xlsx"
_bbr.P_TEAM_DEFAULT   = "Karviná"
_bbr.P_LEAGUE_LONG    = "Czech Fortuna Liga"
_bbr.P_LEAGUE_SHORT   = "Czech First League"
_bbr.P_SEASON         = "2025/26"
_bbr.P_POS_PLURAL     = "full-backs"
_bbr.P_POS_SINGULAR   = "full-back"
_bbr.P_ARCHETYPE_NOTE = "Role Archetypes"

# ── 3. Player config ────────────────────────────────────────────────────────────
_bbr.P_HEADER_NAME     = "N. MILIĆ"
_bbr.P_HEADER_SUBTITLE = ("Karviná  ·  Czech Fortuna Liga 2025/26  ·  "
                          "LB / LCB  ·  Age 22  ·  Slovenia")
_bbr.P_PAGE2_SUBTITLE  = ("Karviná  ·  Czech Fortuna Liga  ·  "
                          "LB / LCB  ·  Age 22  ·  Page 2 of 3")
_bbr.P_PAGE3_SUBTITLE  = ("Karviná  ·  Czech Fortuna Liga  ·  "
                          "LB / LCB  ·  Age 22  ·  Page 3 of 3")
_bbr.P_PILLS           = [("MIN","540"),("MATCHES","10"),("xG","0.00"),("xA","0.29"),("DRIB/90","1.33")]
_bbr.P_WYSCOUT_FILTER  = "N. Milić"
_bbr.P_LEGEND_LABEL    = "N. Milić"

# ── 4. Cover page ────────────────────────────────────────────────────────────────
_bbr.P_COVER_ENABLE   = True
_bbr.P_COVER_SUBTITLE = _bbr.P_HEADER_SUBTITLE
_bbr.P_COVER_MONTH    = "July 2026"

_bbr.P_OUT_PREFIX = "N_Milic_Scouting_Report"


if __name__ == "__main__":
    _bbr.main()
