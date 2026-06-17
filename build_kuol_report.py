"""
build_kuol_report.py  —  FC Hradec Králové scouting report for Garang Kuol
Sparta Praha  ·  Czech Fortuna Liga 2025/26  ·  RWF / RW / LW  ·  Age 21  ·  Australia

Monkey-patches all player config constants in build_barat_report and
build_barat_full_report, then calls main() to produce a 7-page PDF.

Output: reports/G_Kuol_Scouting_Report_Full.pdf
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

# ── 1. Import modules (triggers module-level execution) ───────────────────────
import build_barat_report as _bbr
import build_barat_full_report as _full

# ── 2. Patch build_barat_report player config ─────────────────────────────────
_bbr.P_HEADER_NAME     = "G. KUOL"
_bbr.P_HEADER_SUBTITLE = ("Sparta Praha  ·  Czech Fortuna Liga 2025/26 (Final)  ·  "
                          "RWF / RW / LW  ·  Age 21  ·  Australia")
_bbr.P_PAGE2_SUBTITLE  = ("Sparta Praha  ·  Czech Fortuna Liga  ·  "
                          "RWF / RW / LW  ·  Age 21  ·  Page 2 of 2")
_bbr.P_PAGE3_SUBTITLE  = ("Sparta Praha  ·  Czech Fortuna Liga  ·  "
                          "RWF / RW / LW  ·  Age 21  ·  Page 3 of 3")
_bbr.P_PILLS           = [("MIN","756"),("MATCHES","19"),("xG","2.13"),("xA","0.91"),("DRIB/90","5.12")]
_bbr.P_WYSCOUT_FILTER  = "G. Kuol"
_bbr.P_LEGEND_LABEL    = "G. Kuol"

# ── 3. Patch build_barat_full_report player config ────────────────────────────
_full.P_SC_SHORT        = "G. Kuol"
_full.P_SC_FULL         = "Kuol"
_full.P_MODEL_NAME      = "Garang Kuol"
_full.P_COVER_NAME      = "G. KUOL"
_full.P_COVER_SUBTITLE  = "PLAYER ASSESSMENT REPORT  ·  SUMMER 2026  ·  2026/27 WINDOW"
_full.P_COVER_INFO      = "Sparta Praha  ·  Czech Fortuna Liga  ·  Age 21  ·  Australia"
_full.P_COVER_POS_LABEL = "WIDE ATTACKER"
_full.P_COVER_POS_SHORT = "RWF  ·  RW  ·  LW"
_full.P_COVER_TEAM      = "AC SPARTA PRAHA"
_full.P_COVER_LEAGUE    = "Czech Fortuna Liga"
_full.P_COVER_PILLS     = [("MIN","756"),("MATCHES","19"),("xG","2.13"),("xA","0.91"),("DRIB/90","5.12")]
_full.P_P4_NAME         = "G. KUOL"
_full.P_P4_SUBTITLE     = ("Sparta Praha  ·  Czech Fortuna Liga  ·  RWF / RW / LW  ·  Age 21  ·  "
                           "Physical Profile  ·  Page 4 of 6")
_full.P_P4_LEGEND_LABEL = "G. Kuol"
_full.P_VERDICT_HEADER  = "G. KUOL"
_full.REPORT_REF        = "SCR-2026-048"
_full.REPORT_OUT_PREFIX = "G_Kuol_Scouting_Report"

# ── 4. Analyst verdict ────────────────────────────────────────────────────────
_full.VERDICT        = "LOAN WATCH"
_full.VERDICT_COLOUR = "#1D4ED8"   # FCHK blue = confident interest / loan target
_full.VERDICT_DESC   = (
    "The 2025/26 season is complete. Kuol's 2.13 xG in 756 minutes and elite "
    "dribble volume (5.12/90, top-5 Czech league) mark him as a high-upside loan "
    "target. Zero goals on 2.13 xG signals finishing variance, not a lack of "
    "attacking intent. His age-resale ceiling (98.3) and development value (78.0) "
    "are exceptional. The 2029 Sparta Praha contract means loan-only for 2026/27 "
    "is the realistic pathway. Move before the summer window closes."
)
_full.STRENGTHS = [
    "xG volume  —  2.13 xG in 756 min; major goal-scoring upside yet to convert",
    "Dribble elite  —  5.12 dribbles/90, 62.8% success; direct vertical carrier",
    "Age ceiling  —  Age Resale 98.3, Dev. Value 78.0; maximum upside at 21",
    "Creation threat  —  0.91 xA / 1 assist; active, creative final-third presence",
    "Smart Club fit  —  Strong Benfica academy upside model alignment",
]
_full.CONCERNS = [
    "0 goals in 756 min  —  xG underperformance (2.13 xG, 0 G); finishing concern",
    "Physical score  —  28.6; very low model score, SkillCorner sample only 75 min",
    "Ball security  —  32.2 ball security score; can be bypassed under pressure",
    "Contract length  —  Sparta Praha 2029; permanent transfer not feasible this window",
    "Limited SC data  —  75 min / 6 performances in SkillCorner; treat as indicative",
]
_full.NEXT_STEPS = [
    "Confirm loan availability with Sparta Praha for 2026/27 season",
    "Request full video package (min. 8 matches, focus on chance conversion)",
    "Arrange bilateral meeting with Sparta Praha agent by 31 July 2026",
    "Structure: 12-month loan with option; target arrival August 2026",
    "Priority: Unlock xG underperformance — positional freedom is key",
]
_full.TRANSFER_FLAGS = [
    ("Deal Type",     "Loan Only"),
    ("Fee Range",     "Loan fee est."),
    ("Wage Risk",     "Lower"),
    ("Fee Risk",      "Potentially realistic"),
    ("Availability",  "Needs check"),
    ("Deal Realism",  "75 / 100"),
]


if __name__ == "__main__":
    _full.main()
