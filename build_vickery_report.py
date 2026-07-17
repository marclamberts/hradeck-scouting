"""
build_vickery_report.py  —  FC Hradec Králové scouting report for L. Vickery
Macarthur FC  ·  A-League Men 2025/26  ·  RW / RAMF  ·  Age 20  ·  Australia

Uses the template's default wide-attacker role archetypes (Wide Threat /
Finisher / Target / Roamer / Unlocker / Outlet) and position pool unchanged —
Vickery's positions already fall inside that default set.

Output: reports/L_Vickery_Scouting_Report_Full.pdf
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import build_barat_report as _bbr

# ── 1. Data source / league labels ──────────────────────────────────────────────
_bbr.P_DATA_FILE      = "Australia.xlsx"
_bbr.P_TEAM_KEYWORD   = "Macarthur"
_bbr.P_LEAGUE_LONG    = "Isuzu UTE A-League Men"
_bbr.P_LEAGUE_SHORT   = "A-League"
_bbr.P_SEASON         = "2025/26"
_bbr.P_POS_PLURAL     = "wide attackers"
_bbr.P_POS_SINGULAR   = "wide attacker"
_bbr.P_ARCHETYPE_NOTE = "Role Archetypes"

# ── 2. Player config ────────────────────────────────────────────────────────────
_bbr.P_HEADER_NAME     = "L. VICKERY"
_bbr.P_HEADER_SUBTITLE = ("Macarthur FC  ·  A-League Men 2025/26  ·  "
                          "RW / RAMF  ·  Age 20  ·  Australia")
_bbr.P_PAGE2_SUBTITLE  = ("Macarthur FC  ·  A-League Men  ·  "
                          "RW / RAMF  ·  Age 20  ·  Page 2 of 3")
_bbr.P_PAGE3_SUBTITLE  = ("Macarthur FC  ·  A-League Men  ·  "
                          "RW / RAMF  ·  Age 20  ·  Page 3 of 3")
_bbr.P_PILLS           = [("MIN","1474"),("MATCHES","24"),("xG","3.47"),("xA","1.84"),("DRIB/90","4.82")]
_bbr.P_WYSCOUT_FILTER  = "L. Vickery"
_bbr.P_LEGEND_LABEL    = "L. Vickery"

# ── 3. Cover page ────────────────────────────────────────────────────────────────
_bbr.P_COVER_ENABLE   = True
_bbr.P_COVER_SUBTITLE = _bbr.P_HEADER_SUBTITLE
_bbr.P_COVER_MONTH    = "July 2026"

_bbr.P_OUT_PREFIX = "L_Vickery_Scouting_Report"


if __name__ == "__main__":
    _bbr.main()
