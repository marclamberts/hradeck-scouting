"""
build_hughes_report.py  —  FC Hradec Králové scouting report for I. Hughes
Wellington Phoenix  ·  A-League Men 2025/26  ·  LCB / RCB  ·  Age 22  ·  England

Output: reports/I_Hughes_Scouting_Report_Full.pdf
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import build_barat_report as _bbr

# ── 1. Position pool + centre-back role archetypes ─────────────────────────────
_bbr.WIDE_ATK_POS = {"LCB", "RCB"}

_bbr.ROLE_BLUEPRINTS = {
    "Ball-Playing CB": {
        "Accurate passes, %": 0.30, "Progressive passes per 90": 0.40,
        "Passes per 90": 0.30,
    },
    "Progressor": {
        "Progressive runs per 90": 0.40, "Dribbles per 90": 0.30,
        "Successful dribbles, %": 0.30,
    },
    "Aerial Dominator": {
        "Aerial duels won, %": 0.55, "Aerial duels per 90": 0.25,
        "Defensive duels won, %": 0.20,
    },
    "Stopper": {
        "Defensive duels won, %": 0.45, "Successful defensive actions per 90": 0.35,
        "Interceptions per 90": 0.20,
    },
    "Cover / Reader": {
        "Interceptions per 90": 0.60, "Defensive duels won, %": 0.40,
    },
    "Duel Winner": {
        "Duels won, %": 0.40, "Offensive duels won, %": 0.30,
        "Defensive duels won, %": 0.30,
    },
}

# ── 2. Data source / league labels ──────────────────────────────────────────────
_bbr.P_DATA_FILE      = "Australia.xlsx"
_bbr.P_TEAM_DEFAULT   = "Wellington"
_bbr.P_LEAGUE_LONG    = "Isuzu UTE A-League Men"
_bbr.P_LEAGUE_SHORT   = "A-League"
_bbr.P_SEASON         = "2025/26"
_bbr.P_POS_PLURAL     = "centre-backs"
_bbr.P_POS_SINGULAR   = "centre-back"
_bbr.P_ARCHETYPE_NOTE = "Role Archetypes"

# ── 3. Player config ────────────────────────────────────────────────────────────
_bbr.P_HEADER_NAME     = "I. HUGHES"
_bbr.P_HEADER_SUBTITLE = ("Wellington Phoenix  ·  A-League Men 2025/26  ·  "
                          "LCB / RCB  ·  Age 22  ·  England")
_bbr.P_PAGE2_SUBTITLE  = ("Wellington Phoenix  ·  A-League Men  ·  "
                          "LCB / RCB  ·  Age 22  ·  Page 2 of 3")
_bbr.P_PAGE3_SUBTITLE  = ("Wellington Phoenix  ·  A-League Men  ·  "
                          "LCB / RCB  ·  Age 22  ·  Page 3 of 3")
_bbr.P_PILLS           = [("MIN","2194"),("MATCHES","23"),("xG","1.57"),("xA","0.68"),("DRIB/90","0.33")]
_bbr.P_WYSCOUT_FILTER  = "I. Hughes"
_bbr.P_LEGEND_LABEL    = "I. Hughes"

# ── 4. Cover page ────────────────────────────────────────────────────────────────
_bbr.P_COVER_ENABLE   = True
_bbr.P_COVER_SUBTITLE = _bbr.P_HEADER_SUBTITLE
_bbr.P_COVER_MONTH    = "July 2026"

_bbr.P_OUT_PREFIX = "I_Hughes_Scouting_Report"


if __name__ == "__main__":
    _bbr.main()
