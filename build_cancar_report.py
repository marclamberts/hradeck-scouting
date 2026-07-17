"""
build_cancar_report.py  —  FC Hradec Králové scouting report for P. Cancar
Western Sydney Wanderers  ·  A-League Men 2025/26  ·  RB / RCB  ·  Age 24  ·  New Zealand

Output: reports/P_Cancar_Scouting_Report_Full.pdf
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import build_barat_report as _bbr

# ── 1. Position pool + fullback/wide-CB role archetypes ────────────────────────
_bbr.WIDE_ATK_POS = {"RB", "RCB"}

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

# ── 2. Data source / league labels ──────────────────────────────────────────────
_bbr.P_DATA_FILE      = "Australia.xlsx"
_bbr.P_TEAM_KEYWORD   = "Western Sydney"
_bbr.P_LEAGUE_LONG    = "Isuzu UTE A-League Men"
_bbr.P_LEAGUE_SHORT   = "A-League"
_bbr.P_SEASON         = "2025/26"
_bbr.P_POS_PLURAL     = "full-backs"
_bbr.P_POS_SINGULAR   = "full-back"
_bbr.P_ARCHETYPE_NOTE = "Role Archetypes"

# ── 3. Player config ────────────────────────────────────────────────────────────
_bbr.P_HEADER_NAME     = "P. CANCAR"
_bbr.P_HEADER_SUBTITLE = ("Western Sydney Wanderers  ·  A-League Men 2025/26  ·  "
                          "RB / RCB  ·  Age 24  ·  New Zealand")
_bbr.P_PAGE2_SUBTITLE  = ("Western Sydney Wanderers  ·  A-League Men  ·  "
                          "RB / RCB  ·  Age 24  ·  Page 2 of 3")
_bbr.P_PAGE3_SUBTITLE  = ("Western Sydney Wanderers  ·  A-League Men  ·  "
                          "RB / RCB  ·  Age 24  ·  Page 3 of 3")
_bbr.P_PILLS           = [("MIN","738"),("MATCHES","11"),("xG","0.88"),("xA","0.10"),("DRIB/90","0.37")]
_bbr.P_WYSCOUT_FILTER  = "P. Cancar"
_bbr.P_LEGEND_LABEL    = "P. Cancar"

# ── 4. Cover page ────────────────────────────────────────────────────────────────
_bbr.P_COVER_ENABLE   = True
_bbr.P_COVER_SUBTITLE = _bbr.P_HEADER_SUBTITLE
_bbr.P_COVER_MONTH    = "July 2026"

_bbr.P_OUT_PREFIX = "P_Cancar_Scouting_Report"


if __name__ == "__main__":
    _bbr.main()
