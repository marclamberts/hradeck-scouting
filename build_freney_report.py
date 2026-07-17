"""
build_freney_report.py  —  FC Hradec Králové scouting report for W. Freney
Perth Glory  ·  A-League Men 2025/26  ·  LCMF / LDMF  ·  Age 20  ·  Australia

Output: reports/W_Freney_Scouting_Report_Full.pdf
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import build_barat_report as _bbr

# ── 1. Position pool + CM/DM role archetypes ────────────────────────────────────
_bbr.WIDE_ATK_POS = {"LCMF", "LDMF", "RCMF", "RDMF"}

_bbr.ROLE_BLUEPRINTS = {
    "Deep Playmaker": {
        "Accurate passes, %": 0.30, "Progressive passes per 90": 0.35,
        "Passes per 90": 0.20, "Key passes per 90": 0.15,
    },
    "Ball-Winner": {
        "Defensive duels won, %": 0.40, "Successful defensive actions per 90": 0.35,
        "Interceptions per 90": 0.25,
    },
    "Progressor": {
        "Progressive runs per 90": 0.40, "Dribbles per 90": 0.30,
        "Successful dribbles, %": 0.30,
    },
    "Box-to-Box": {
        "Duels per 90": 0.30, "Progressive runs per 90": 0.30,
        "Successful defensive actions per 90": 0.40,
    },
    "Creative Eight": {
        "Key passes per 90": 0.30, "xA per 90": 0.30,
        "Shot assists per 90": 0.20, "Crosses per 90": 0.20,
    },
    "Duel Winner": {
        "Duels won, %": 0.40, "Offensive duels won, %": 0.30,
        "Defensive duels won, %": 0.30,
    },
}

# ── 2. Data source / league labels ──────────────────────────────────────────────
_bbr.P_DATA_FILE      = "Australia.xlsx"
_bbr.P_TEAM_DEFAULT   = "Perth"
_bbr.P_LEAGUE_LONG    = "Isuzu UTE A-League Men"
_bbr.P_LEAGUE_SHORT   = "A-League"
_bbr.P_SEASON         = "2025/26"
_bbr.P_POS_PLURAL     = "central midfielders"
_bbr.P_POS_SINGULAR   = "central midfielder"
_bbr.P_ARCHETYPE_NOTE = "Role Archetypes"

# ── 3. Player config ────────────────────────────────────────────────────────────
_bbr.P_HEADER_NAME     = "W. FRENEY"
_bbr.P_HEADER_SUBTITLE = ("Perth Glory  ·  A-League Men 2025/26  ·  "
                          "LCMF / LDMF  ·  Age 20  ·  Australia")
_bbr.P_PAGE2_SUBTITLE  = ("Perth Glory  ·  A-League Men  ·  "
                          "LCMF / LDMF  ·  Age 20  ·  Page 2 of 3")
_bbr.P_PAGE3_SUBTITLE  = ("Perth Glory  ·  A-League Men  ·  "
                          "LCMF / LDMF  ·  Age 20  ·  Page 3 of 3")
_bbr.P_PILLS           = [("MIN","1855"),("MATCHES","23"),("xG","0.42"),("xA","0.75"),("DRIB/90","0.97")]
_bbr.P_WYSCOUT_FILTER  = "W. Freney"
_bbr.P_LEGEND_LABEL    = "W. Freney"

# ── 4. Cover page ────────────────────────────────────────────────────────────────
_bbr.P_COVER_ENABLE   = True
_bbr.P_COVER_SUBTITLE = _bbr.P_HEADER_SUBTITLE
_bbr.P_COVER_MONTH    = "July 2026"

_bbr.P_OUT_PREFIX = "W_Freney_Scouting_Report"


if __name__ == "__main__":
    _bbr.main()
