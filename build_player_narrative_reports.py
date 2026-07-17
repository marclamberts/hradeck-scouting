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
            "Height and weight are not confirmed, but he holds his own physically inside the box and "
            "competes well in the air, giving the impression of a sturdy, competitive frame rather than "
            "a light, elusive forward."
        ),
        pace=(
            "Not a mobile or explosive forward. He offers almost no running in behind and rarely drives "
            "with the ball from deeper positions — his game is built on where he stands in the box, not "
            "on how fast he gets there."
        ),
        work_rate=(
            "Limited pressing intensity for a lone striker, though he is not entirely passive off the "
            "ball and does put in a moderate defensive shift when needed."
        ),
        attitude=(
            "Engages in duels often but wins few of them. That points to persistence and a willingness "
            "to compete rather than genuine physical control or dominance in those contests."
        ),
        technical=(
            "A tidy, reliable passer, but his work with the ball at his feet is weak — dribbles rarely "
            "come off — and he offers almost nothing creatively for teammates. Not a carrier or a "
            "creator."
        ),
        intelligence=(
            "Finds good scoring positions consistently and generates encouraging underlying chances, "
            "but badly underdelivers in front of goal relative to the openings he creates for himself — "
            "a clear composure and finishing gap."
        ),
        comments=(
            "A young Brazilian No.9 whose movement and underlying chance generation are more "
            "encouraging than his output. On a small sample, he combines a real presence inside the box "
            "with almost no mobility or ball-carrying quality, and a finishing return that lags well "
            "behind the chances he creates. For Hradec Králové specifically: the club's vertical, "
            "transition-based attacking model rewards forwards who can progress and accelerate the ball "
            "themselves — Vinicius's profile runs counter to that identity."
        ),
        strengths=[
            "Consistently finds good scoring positions and threatens the box",
            "Competitive and combative in the air",
            "Reliable, tidy passer under pressure",
        ],
        weaknesses=[
            "Significant finishing underperformance relative to the chances he creates",
            "Very limited mobility and ball progression from deep",
            "Poor ball retention when dribbling, with minimal creative output",
            "Small sample size — the profile should be treated as provisional",
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
            "A good, physically mature frame for a full-back, backed up by genuine competitiveness in "
            "the air against crosses and long balls — well above what is typical for the position."
        ),
        pace=(
            "Shows some ball-carrying instinct from deep positions and a moderate ability to break "
            "lines with the dribble, though he is not an explosive or especially frequent runner — "
            "worth confirming on video given the limited sample available."
        ),
        work_rate=(
            "Very heavily involved in build-up play — consistently on the ball and progressing it "
            "forward. Defensive work off the ball is solid without being spectacular."
        ),
        attitude=(
            "Competes actively in duels without over-committing or becoming reckless. Reads as a "
            "disciplined, engaged defender rather than an aggressive risk-taker."
        ),
        technical=(
            "Left-footed with genuine passing range and composure on the ball. His crossing volume is "
            "decent but end product from wide areas is inconsistent — passing, not crossing, is his "
            "standout tool."
        ),
        intelligence=(
            "Profiles as a genuine ball-playing full-back — he progresses the team primarily through "
            "passing rather than dribbling or crossing, and offers a reliable outlet from deep "
            "positions."
        ),
        comments=(
            "A young, left-footed, ball-progressing full-back / wide centre-back from Karviná. The "
            "sample is small and the near-absence of attacking end product is expected for the role "
            "rather than a concern. His standout trait — progressive passing — is a good stylistic "
            "match for a team that wants to build quickly through deep, secure passers, but his "
            "one-on-one defending needs live or video verification before trusting it at a higher "
            "level."
        ),
        strengths=[
            "Strong progressive passing range and composure in possession",
            "Competitive in the air for his position",
            "Composed and controlled in duels, rarely reckless",
            "Positional versatility across left-back and left centre-back",
        ],
        weaknesses=[
            "Isolated one-on-one defending looks like the weaker part of his game relative to peers",
            "Crossing end product is inconsistent despite reasonable delivery volume",
            "Small sample size — defensive robustness still unverified",
            "No attacking end product on the season, though this is expected for the role",
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
            "A genuinely imposing physical frame that shows up clearly in how he plays — a real focal "
            "point and target man, though there is still more to unlock in the air relative to his "
            "size."
        ),
        pace=(
            "Not an explosive runner, but carries enough mobility to be useful in transition, and it "
            "shows up clearly in his ball-carrying, which is comfortably the best of this group."
        ),
        work_rate=(
            "A competitive defensive workload for a No.9 — functional rather than elite in his "
            "pressing contribution."
        ),
        attitude=(
            "Shows real physical dominance and competitiveness in defensive duels, even though his "
            "overall duel return is dragged down by a weaker showing in offensive duels."
        ),
        technical=(
            "The most complete technical package of the three covered here — good shot quality, "
            "functional passing, and clearly the best dribbling success rate of the group."
        ),
        intelligence=(
            "Excellent movement into the box combined with a finishing return that outstrips his "
            "underlying chance quality — a real signal of composure and instinct in front of goal."
        ),
        comments=(
            "The strongest and most complete centre-forward profile of the three covered in this "
            "cycle. A physically mature target forward, one tier below Hradec Králové's level, who is "
            "outperforming his own underlying chances, carrying the ball well for a player of his "
            "size, and profiles as an elite finisher and target man. Genuinely ready for a look at a "
            "higher level, with the usual caveat that a step up a division brings its own adjustment "
            "risk."
        ),
        strengths=[
            "Outperforming his own underlying chance quality — a real finishing signal",
            "Profiles as an elite finisher and target man",
            "Best shot quality and ball-carrying success of the group",
            "Standout physical and competitive signal in defensive duels",
        ],
        weaknesses=[
            "Offensive duel win rate is below what his frame would suggest",
            "Limited creative output — finishes far more than he creates for others",
            "Aerial dominance is modest relative to his height — underdelivering in the air so far",
            "One tier below Hradec's level — the jump to the Czech top flight is the main adaptation risk",
        ],
        overall_rating="7.5", fit_hk="7",
    ),
]


if __name__ == "__main__":
    for cfg in PLAYERS:
        path = build_report(cfg)
        print(f"  Saved → {path}")
