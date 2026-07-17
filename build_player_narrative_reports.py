"""
build_player_narrative_reports.py  —  FC Hradec Králové qualitative scouting reports
Matches the "Player Report on ..." one-page narrative format:
  Physique / Pace / Work-Rate / Attitude / Technical / Intelligence / Comments /
  Strengths / Weaknesses / Overall rating / Fit pro HK

Analysis text is written from the underlying Wyscout numbers (see reports/ for
the companion statistical "Full" PDFs, which share the same source data).

No club-crest image asset exists in this repo, so the header is text-only.

Output: reports/{Slug}_Player_Report.pdf  (one per player in PLAYERS)
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

# Base-14 Helvetica can't render extended Latin (e.g. "ć") — register a Unicode TTF.
_FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
pdfmetrics.registerFont(TTFont("Helvetica",      _FONT_DIR / "LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Helvetica-Bold",  _FONT_DIR / "LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Helvetica-Oblique", _FONT_DIR / "LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Helvetica-BoldOblique", _FONT_DIR / "LiberationSans-BoldItalic.ttf"))

TEXT      = "#111827"
TEXT_DIM  = "#374151"
RULE      = "#9CA3AF"

styles = {
    "title": ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=18, leading=22,
        alignment=TA_CENTER, textColor=TEXT, spaceAfter=10,
    ),
    "meta": ParagraphStyle(
        "meta", fontName="Helvetica", fontSize=9, leading=12.5,
        alignment=TA_CENTER, textColor=TEXT_DIM, spaceAfter=3,
    ),
    "h2": ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
        alignment=TA_LEFT, textColor=TEXT, spaceBefore=7, spaceAfter=2,
    ),
    "body": ParagraphStyle(
        "body", fontName="Helvetica", fontSize=9.3, leading=12.4,
        alignment=TA_LEFT, textColor=TEXT, spaceAfter=1,
    ),
    "bullet": ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=9.3, leading=12.4,
        alignment=TA_LEFT, textColor=TEXT, leftIndent=10, spaceAfter=0.5,
    ),
    "rating": ParagraphStyle(
        "rating", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
        alignment=TA_LEFT, textColor=TEXT, spaceBefore=5,
    ),
}


def build_report(cfg: dict) -> Path:
    out_path = OUT_DIR / f"{cfg['slug']}_Player_Report.pdf"
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=1.3 * cm, bottomMargin=1.3 * cm,
    )

    story = []
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(f"Player Report on {cfg['name']}", styles["title"]))
    story.append(Paragraph(
        f"<u>Season</u> {cfg['season']} &nbsp;&nbsp; <u>Club</u> <b>{cfg['club']}</b>",
        styles["meta"]))
    story.append(Paragraph(
        f"<u>Date of Report</u> {cfg['report_date']} &nbsp;&nbsp; "
        f"<u>Position</u> {cfg['position']} &nbsp;&nbsp; "
        f"<u>Age</u> {cfg['age']} &nbsp;&nbsp; "
        f"<u>Foot</u> {cfg['foot']} &nbsp;&nbsp; "
        f"<u>Height</u> {cfg['height']}",
        styles["meta"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(HRFlowable(width="100%", color=RULE, thickness=0.6))
    story.append(Spacer(1, 0.05 * cm))

    for section in ["Physique", "Pace", "Work-Rate", "Attitude", "Technical", "Intelligence", "Comments"]:
        story.append(Paragraph(f"<u>{section}</u>", styles["h2"]))
        story.append(Paragraph(cfg[section.lower().replace("-", "_")], styles["body"]))

    story.append(Paragraph("<u>Strengths</u>", styles["h2"]))
    for s in cfg["strengths"]:
        story.append(Paragraph(f"- {s}", styles["bullet"]))

    story.append(Paragraph("<u>Weaknesses</u>", styles["h2"]))
    for w in cfg["weaknesses"]:
        story.append(Paragraph(f"- {w}", styles["bullet"]))

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(f"<u>Overall rating:</u>  {cfg['overall_rating']} / 10", styles["rating"]))
    story.append(Paragraph(f"<u>Fit pro HK:</u>  {cfg['fit_hk']} / 10", styles["rating"]))

    doc.build(story)
    return out_path


PLAYERS = [
    dict(
        slug="K_Vinicius",
        name="Kahuan Vinicius",
        season="2025/2026", club="FK Karviná",
        report_date="17-07-2026", position="CF", age=22, foot="Unknown (not recorded)",
        height="Unknown (not recorded)",
        physique=(
            "Height/weight are not recorded in the data, but his role-fit profile (73% Finisher, "
            "68% Target Man) and a respectable 42% aerial-duel win rate point to a physically "
            "competitive presence inside the box rather than a slight, elusive forward."
        ),
        pace=(
            "Very low mobility numbers — 0.44 progressive runs/90 and just 0.15 accelerations/90 — "
            "place him among the least mobile forwards in the sample. He is not a runner in behind; "
            "his game is built on where he stands, not how fast he gets there."
        ),
        work_rate=(
            "Moderate defensive workload for a lone striker (3.34 defensive actions/90, 1.6 "
            "interceptions/90). Pressing CF role-fit sits at only 31%, so he is not a high-intensity "
            "front-foot forward, but he is not entirely passive off the ball either."
        ),
        attitude=(
            "Engages in duels frequently (21.2/90) but wins only 37.7% of them, and offensive duels "
            "specifically are won just 23.5% of the time. High engagement, low return — persistence "
            "without control."
        ),
        technical=(
            "Passing is tidy (86% accuracy) but his end product with the ball at feet is weak: only "
            "16.7% of dribble attempts succeed, and creative output is negligible (0.02 xA/90, 0.15 "
            "key passes/90). He is not a carrier or creator."
        ),
        intelligence=(
            "Gets into good scoring areas — 3.78 box touches/90 and 0.51 xG/90 are solid underlying "
            "numbers for the level — but 2 goals from 3.52 xG (-1.52) over the sample is a significant "
            "finishing shortfall relative to the chances he generates."
        ),
        comments=(
            "A young Brazilian No.9 whose underlying chance generation is more encouraging than his "
            "output. On a genuinely small sample (11 matches, 619 minutes) he combines real box "
            "presence with almost no mobility or ball-carrying quality, and a finishing return well "
            "below his xG. For Hradec Králové specifically: the club's vertical, transition-based "
            "attacking model rewards forwards who can progress and accelerate — Vinicius's numbers "
            "run counter to that identity, echoing the same style mismatch flagged for comparable "
            "target forwards in this scouting cycle."
        ),
        strengths=[
            "Generates good chance volume and box presence for his role (0.51 xG/90, 3.78 box touches/90)",
            "Competitive in the air (42.1% aerial duels won)",
            "Reliable passer under pressure (86% pass accuracy)",
        ],
        weaknesses=[
            "Significant finishing underperformance versus his own chance quality (-1.52 G-xG)",
            "Very low mobility / progression (0.44 progressive runs/90, 0.15 accelerations/90)",
            "Poor dribble retention (16.7% success) and minimal creative output",
            "Tiny sample (11 matches) — treat all of the above as provisional",
        ],
        overall_rating="5.5", fit_hk="4",
    ),
    dict(
        slug="N_Milic",
        name="N. Milić",
        season="2025/2026", club="FK Karviná",
        report_date="17-07-2026", position="LB / LCB", age=22, foot="Left",
        height="1.85m",
        physique=(
            "A good frame for a full-back (185cm/77kg), backed up by a strong 52.2% aerial-duel win "
            "rate — well above what is typical for a player in this role, suggesting genuine physical "
            "competitiveness against crosses and long balls."
        ),
        pace=(
            "Moderate ball-carrying instincts from deep — 1.0 progressive run/90 and 1.33 dribbles/90 "
            "at a healthy 50% success rate — though the sample shows no recorded accelerations, which "
            "is worth checking on video rather than taking at face value given the small minutes total."
        ),
        work_rate=(
            "Very heavily involved in build-up: 37 passes/90 is a high volume for a full-back, with "
            "9.17 progressive passes/90 standing out. Defensive work is solid rather than spectacular — "
            "7.0 defensive actions/90 and 2.5 interceptions/90."
        ),
        attitude=(
            "Competes actively (17.3 duels/90, winning 53.9%) without over-committing — fouls stay "
            "controlled at 1.5/90. Reads as a disciplined, engaged defender rather than a reckless one."
        ),
        technical=(
            "Left-footed with genuine passing range — 82% accuracy at high volume, plus strong crossing "
            "volume (2.5/90) though accuracy on those crosses is modest (33.3%). Passing, not crossing, "
            "is the standout tool."
        ),
        intelligence=(
            "Role-fit data confirms a genuine Ball-Playing Full-Back profile (62%, primary role) — he "
            "progresses the team through passing more than through dribbling or crossing, and offers "
            "a reliable outlet from deep positions."
        ),
        comments=(
            "A 22-year-old, left-footed, ball-progressing full-back / wide centre-back from Karviná. "
            "The sample is small (10 matches, 540 minutes) and the near-zero attacking output (0 xG, "
            "minimal key passes) is expected for the role rather than a concern. His standout trait — "
            "progressive passing volume — is a good stylistic match for a team that wants to build "
            "quickly through deep, secure passers, but his 1v1 defensive numbers need live or video "
            "verification before trusting them at a higher level."
        ),
        strengths=[
            "Strong progressive-passing range and volume for a full-back (9.17 progressive passes/90)",
            "Aerially competitive for the position (52.2% duels won)",
            "Composed and controlled in duels (53.9% win rate, low foul count)",
            "Positional versatility (LB and LCB)",
        ],
        weaknesses=[
            "Weaker isolated 1v1 defending signal relative to league peers (low 'Stopper' role-fit)",
            "Crossing accuracy is poor (33.3%) despite reasonable volume",
            "Very small sample (10 matches) — defensive robustness unverified",
            "No offensive end product recorded (0 goals, 0 xG) — expected for the role, not a flaw per se",
        ],
        overall_rating="6", fit_hk="6",
    ),
    dict(
        slug="A_Marcolino",
        name="A. Do Marcolino",
        season="2025/2026", club="FK Ústí nad Labem",
        report_date="17-07-2026", position="CF", age=24, foot="Right",
        height="1.94m",
        physique=(
            "A genuinely imposing frame (194cm/83kg) that shows up directly in the data — an elite "
            "90% Target Man role-fit score — though his actual aerial win rate (42.3%) is more modest "
            "than the height alone would suggest, meaning there is still more to unlock in the air."
        ),
        pace=(
            "Not an explosive runner, but has enough mobility to be useful in transition — 1.75 "
            "progressive runs/90 and 0.52 accelerations/90 — and it shows up in his carrying: 2.27 "
            "dribbles/90 at a strong 54.6% success rate, clearly the best carrying output of this group."
        ),
        work_rate=(
            "Competitive defensive workload for a No.9 — 4.75 defensive actions/90, 1.75 "
            "interceptions/90 — with a Pressing CF role-fit of 50%, a functional rather than elite "
            "pressing forward."
        ),
        attitude=(
            "A standout defensive-duel win rate of 70.7% signals real physical dominance and "
            "competitiveness in 1v1 duels, even though his overall duel win rate (38.4%) is dragged "
            "down by a weaker return in offensive duels (27.6%)."
        ),
        technical=(
            "Right-footed with the most complete technical package of the three: 46.2% of shots on "
            "target (clearly the best conversion of chances into test-the-keeper efforts), functional "
            "passing (77.3% accuracy), and by far the best dribble success rate of this group (54.6%)."
        ),
        intelligence=(
            "Elite movement into the box (5.88 box touches/90) combines with a positive goals-minus-xG "
            "of +1.03 across 872 minutes (6 goals from 4.97 xG) — a real signal of composure and "
            "finishing instinct beyond what his underlying chances alone would predict."
        ),
        comments=(
            "The strongest and most complete CF profile of the three covered in this cycle. A "
            "physically mature 24-year-old target forward, one tier below Hradec Králové's level "
            "(Czech National Football League), who is outscoring his own xG, carrying the ball well "
            "for a player of his size, and rates as an elite Finisher/Target Man on role-fit (93% / "
            "90%). Genuinely ready for a look at a higher level, with the usual caveat that a step up "
            "a division brings its own adjustment risk."
        ),
        strengths=[
            "Positive finishing signal — outscoring his own xG by +1.03 over the sample (6 G vs 4.97 xG)",
            "Elite Finisher and Target Man role-fit scores (93% / 90%)",
            "Best shot quality (46.2% on target) and ball-carrying success (54.6% dribbles) of the group",
            "Standout physical/competitive signal in defensive duels (70.7% won)",
        ],
        weaknesses=[
            "Below-average offensive-duel win rate for his frame (27.6%)",
            "Limited creative output — finishes far more than he creates (0.05 xA/90, 0.31 key passes/90)",
            "Aerial win rate (42.3%) is modest relative to his height — underdelivering in the air so far",
            "One tier below Hradec's level — the jump to the Czech top flight is the main adaptation risk",
        ],
        overall_rating="7.5", fit_hk="7",
    ),
]


if __name__ == "__main__":
    for cfg in PLAYERS:
        path = build_report(cfg)
        print(f"  Saved → {path}")
