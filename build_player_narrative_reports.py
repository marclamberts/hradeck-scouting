"""
build_player_narrative_reports.py  —  FC Hradec Králové qualitative scouting reports

Replicates the "Player Report on ..." one-page narrative format one-for-one
with the original reference document: club crest top-centre, centred title
and bio block (with the club name rendered larger, matching the source),
underlined section headers (Physique / Pace / Work-Rate / Attitude /
Technical / Intelligence / Comments / Strengths / Weaknesses), and a tight
two-line ratings block at the bottom. Layout metrics (font sizes, colour,
underline styling, margins, spacing) were reverse-engineered directly from
the reference PDF's internal structure (fonts, spans, vector underlines).

Analysis text is grounded in each player's Wyscout numbers (see reports/
for the companion statistical "Full" PDFs, which share the same source
data), but is written as plain scouting prose: no stats, percentages, or
data/analytics language, and no em dashes.

Output: reports/{Slug}_Player_Report.pdf  (one per player in PLAYERS)
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)
CREST   = ROOT / "assets" / "fchk_crest.png"

# Base-14 Helvetica can't render extended Latin (e.g. "ć") — register a Unicode TTF.
_FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
pdfmetrics.registerFont(TTFont("Helvetica",      _FONT_DIR / "LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Helvetica-Bold",  _FONT_DIR / "LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Helvetica-Oblique", _FONT_DIR / "LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Helvetica-BoldOblique", _FONT_DIR / "LiberationSans-BoldItalic.ttf"))

INK = "#2C303A"   # single ink colour used throughout the reference document

styles = {
    "title": ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=21, leading=25,
        alignment=TA_CENTER, textColor=INK, spaceAfter=13,
    ),
    "meta": ParagraphStyle(
        "meta", fontName="Helvetica", fontSize=7.9, leading=13,
        alignment=TA_CENTER, textColor=INK, spaceAfter=6,
    ),
    "h2": ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=12, leading=14,
        alignment=TA_LEFT, textColor=INK, spaceBefore=11, spaceAfter=2,
    ),
    "body": ParagraphStyle(
        "body", fontName="Helvetica", fontSize=11, leading=13,
        alignment=TA_LEFT, textColor=INK, spaceAfter=1,
    ),
    "bullet": ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=11, leading=13,
        alignment=TA_LEFT, textColor=INK, spaceAfter=0.5,
    ),
    "rating": ParagraphStyle(
        "rating", fontName="Helvetica-Bold", fontSize=12, leading=14,
        alignment=TA_LEFT, textColor=INK, spaceBefore=0,
    ),
}


def build_report(cfg: dict) -> Path:
    out_path = OUT_DIR / f"{cfg['slug']}_Player_Report.pdf"
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=27, rightMargin=88,   # matches the reference document's margins (pt)
        topMargin=14, bottomMargin=10,
    )

    story = []
    if CREST.exists():
        story.append(Image(str(CREST), width=49, height=60, hAlign="CENTER"))
    story.append(Spacer(1, 3))
    story.append(Paragraph(f"Player Report on {cfg['name']}", styles["title"]))
    story.append(Paragraph(
        f'<u>Season</u> {cfg["season"]} &nbsp;&nbsp; <u>Club</u> '
        f'<font size="11"><b>{cfg["club"]}</b></font>',
        styles["meta"]))
    story.append(Paragraph(
        f"<u>Date of Report</u> {cfg['report_date']} &nbsp;&nbsp; "
        f"<u>Position</u> {cfg['position']} &nbsp;&nbsp; "
        f"<u>Age</u> {cfg['age']} &nbsp;&nbsp; "
        f"<u>Foot</u> {cfg['foot']} &nbsp;&nbsp; "
        f"<u>Height</u> {cfg['height']}",
        styles["meta"]))

    for section in ["Physique", "Pace", "Work-Rate", "Attitude", "Technical", "Intelligence", "Comments"]:
        story.append(Paragraph(f"<u>{section}</u>", styles["h2"]))
        story.append(Paragraph(cfg[section.lower().replace("-", "_")], styles["body"]))

    story.append(Paragraph("<u>Strengths</u>", styles["h2"]))
    for s in cfg["strengths"]:
        story.append(Paragraph(f"- {s}", styles["bullet"]))

    story.append(Paragraph("<u>Weaknesses</u>", styles["h2"]))
    for w in cfg["weaknesses"]:
        story.append(Paragraph(f"- {w}", styles["bullet"]))

    rating_style = ParagraphStyle("rating_tight", parent=styles["rating"], spaceBefore=10)
    story.append(Paragraph(f"<u>Overall rating:</u>  {cfg['overall_rating']} / 10", rating_style))
    story.append(Paragraph(f"<u>Fit pro HK:</u>  {cfg['fit_hk']} / 10",
                            ParagraphStyle("rating_tight2", parent=styles["rating"], spaceBefore=2)))

    doc.build(story)
    return out_path


PLAYERS = [
    dict(
        slug="K_Vinicius",
        name="Kahuan Vinicius",
        season="2025/2026", club="FK Karviná",
        report_date="17-07-2026", position="CF", age=22, foot="n/a",
        height="n/a",
        physique=(
            "He looks physically sturdy inside the box and competes well in the air, giving the "
            "impression of a strong, competitive frame rather than a light, elusive forward."
        ),
        pace=(
            "Not a mobile or explosive forward. He rarely runs in behind and rarely drives with the "
            "ball from deeper areas. His game is built on where he stands in the box, not on how fast "
            "he gets there."
        ),
        work_rate=(
            "Limited pressing for a lone striker, though he is not entirely passive off the ball and "
            "does put in a moderate shift defensively when needed."
        ),
        attitude=(
            "He competes for the ball often but does not come away with it as often as he would like. "
            "That points to persistence and a willingness to fight rather than genuine physical "
            "dominance."
        ),
        technical=(
            "A tidy, reliable passer, but his work with the ball at his feet is weak. Dribbles rarely "
            "come off, and he offers almost nothing creatively for teammates. Not a carrier or a "
            "creator."
        ),
        intelligence=(
            "He finds good scoring positions consistently and moves well in the box, but he badly "
            "underdelivers in front of goal relative to the chances he creates for himself. A clear "
            "composure and finishing gap."
        ),
        comments=(
            "A young Brazilian number nine whose movement is more encouraging than his finishing. He "
            "combines a real presence inside the box with almost no mobility or ball carrying quality, "
            "and a finishing touch that lags well behind the chances he creates for himself. For "
            "Hradec Králové specifically, the club's vertical, transition based attacking style "
            "rewards forwards who can progress and accelerate the ball themselves. Vinicius's game "
            "runs counter to that identity."
        ),
        strengths=[
            "Consistently finds good scoring positions and threatens the box",
            "Competitive and combative in the air",
            "Reliable, tidy passer under pressure",
        ],
        weaknesses=[
            "Clear finishing underperformance relative to the chances he creates",
            "Very limited mobility and ball progression from deep",
            "Poor ball retention when dribbling, with minimal creative touch",
            "A small number of matches watched so far, the picture should be treated as early",
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
            "A good, physically mature frame for a full back, and he competes well in the air against "
            "crosses and long balls, well above what is typical for the position."
        ),
        pace=(
            "Shows some ball carrying instinct from deep positions and a moderate ability to break "
            "lines with the dribble, though he is not an explosive or especially frequent runner. "
            "Worth confirming on more video."
        ),
        work_rate=(
            "Very heavily involved in build up play. Consistently on the ball and looking to move it "
            "forward. Defensive work off the ball is solid without being spectacular."
        ),
        attitude=(
            "Competes actively in duels without over committing or becoming reckless. Reads as a "
            "disciplined, engaged defender rather than an aggressive risk taker."
        ),
        technical=(
            "Left footed with genuine passing range and composure on the ball. His crossing looks "
            "decent in volume but the end product from wide areas is inconsistent. Passing, not "
            "crossing, is his standout tool."
        ),
        intelligence=(
            "Looks like a genuine ball playing full back. He progresses the team mainly through "
            "passing rather than dribbling or crossing, and offers a reliable outlet from deep "
            "positions."
        ),
        comments=(
            "A young, left footed, ball progressing full back or wide centre back from Karviná. The "
            "near absence of goal involvement is expected for the role rather than a concern. His "
            "standout trait, progressive passing, is a good stylistic match for a team that wants to "
            "build quickly through deep, secure passers, but his one on one defending needs further "
            "live or video confirmation before trusting it at a higher level."
        ),
        strengths=[
            "Strong progressive passing and composure in possession",
            "Competitive in the air for his position",
            "Composed and controlled in duels, rarely reckless",
            "Positional versatility across left back and left centre back",
        ],
        weaknesses=[
            "Isolated one on one defending looks like the weaker part of his game relative to peers",
            "Crossing end product is inconsistent despite reasonable delivery",
            "Limited matches watched so far, defensive robustness still unconfirmed",
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
            "A genuinely imposing physical frame that shows up clearly in how he plays. A real focal "
            "point and target man, though there is still more to unlock in the air relative to his "
            "size."
        ),
        pace=(
            "Not an explosive runner, but he carries enough mobility to be useful in transition, and "
            "it shows up clearly in his ball carrying, which is comfortably the best of this group."
        ),
        work_rate=(
            "A competitive defensive shift for a number nine. Functional rather than elite in his "
            "pressing."
        ),
        attitude=(
            "Shows real physical dominance and competitiveness in defensive duels, even though he is "
            "less effective in duels going forward."
        ),
        technical=(
            "The most complete technical package of the three covered here. Good shot quality, "
            "functional passing, and clearly the best dribbling of the group."
        ),
        intelligence=(
            "Excellent movement into the box combined with a finishing touch that looks better than "
            "the chances themselves would suggest. A real sign of composure and instinct in front of "
            "goal."
        ),
        comments=(
            "The strongest and most complete centre forward of the three covered in this cycle. A "
            "physically mature target forward, one tier below Hradec Králové's level, who finishes "
            "better than his chances alone would suggest, carries the ball well for a player of his "
            "size, and looks like an elite finisher and target man. Genuinely ready for a look at a "
            "higher level, with the usual caveat that a step up a division brings its own adjustment "
            "risk."
        ),
        strengths=[
            "Finishes better than his chances alone would suggest, a real sign of quality in front of goal",
            "Looks like an elite finisher and target man",
            "Best shot quality and ball carrying of the group",
            "Standout physical and competitive edge in defensive duels",
        ],
        weaknesses=[
            "Less effective in duels going forward than his frame would suggest",
            "Limited creative touch, finishes far more than he creates for others",
            "Aerial presence is modest relative to his height, underdelivering in the air so far",
            "One tier below Hradec's level, the jump to the Czech top flight is the main adjustment risk",
        ],
        overall_rating="7.5", fit_hk="7",
    ),
    dict(
        slug="P_Kikianis",
        name="P. Kikianis",
        season="2025/2026", club="Adelaide United",
        report_date="20-07-2026", position="LCB", age=21, foot="n/a",
        height="n/a",
        physique=(
            "Size is not confirmed, but he holds up well in the air and is rarely beaten cheaply. He "
            "reads as a composed, positionally sound defender more than an imposing physical "
            "specimen."
        ),
        pace=(
            "Not a runner or a carrier out from the back. His value is in reading the game and "
            "distributing, not in driving forward with the ball himself."
        ),
        work_rate=(
            "An ever present figure at the back who barely misses a match. A genuine workhorse who "
            "gets across to cover consistently."
        ),
        attitude=(
            "Wins the large majority of the challenges he enters. Calm, controlled, and disciplined "
            "enough to avoid unnecessary fouls."
        ),
        technical=(
            "An outstanding distributor for a centre back. Very comfortable in possession, accurate "
            "in his passing, and shows a genuine habit of moving the ball forward through his passing "
            "rather than just recycling it sideways."
        ),
        intelligence=(
            "Reading the game is his standout trait. He steps in to intercept at an excellent rate for "
            "the position, regularly snuffing out danger before it develops."
        ),
        comments=(
            "A genuinely reliable, ever present defensive foundation. Not a ball carrier or a physical "
            "dominator, but an excellent reader of the game and one of the best passing centre backs "
            "covered in this cycle. The archetype of a modern, possession comfortable stopper who lets "
            "others do the carrying."
        ),
        strengths=[
            "Excellent reading of the game and interceptions",
            "Outstanding passing for a centre back",
            "Controlled and disciplined, rarely fouls",
            "Ever present across a long season",
        ],
        weaknesses=[
            "Almost no ball carrying threat from the back",
            "Modest aerial presence for a centre back",
            "Limited attacking upside",
            "Needs a system that values distribution over physical dominance",
        ],
        overall_rating="6.5", fit_hk="6",
    ),
    dict(
        slug="W_Freney",
        name="W. Freney",
        season="2025/2026", club="Perth Glory",
        report_date="20-07-2026", position="LCMF / LDMF", age=20, foot="n/a",
        height="n/a",
        physique=(
            "Nothing in how he plays marks him out physically. A functional rather than imposing "
            "midfield frame."
        ),
        pace=(
            "Shows genuine carrying instinct and a willingness to drive with the ball rather than just "
            "recycle it, without being an especially explosive mover."
        ),
        work_rate=(
            "A committed defensive contributor for his age, consistently involved without being the "
            "main destroyer in his midfield."
        ),
        attitude=(
            "Competes actively without being reckless. Engaged in the game rather than passive."
        ),
        technical=(
            "Tidy in possession with a decent passing platform. Modest end product going forward but "
            "shows flashes of creativity."
        ),
        intelligence=(
            "Still developing. His overall involvement in the game suggests a genuine two way player "
            "rather than a specialist, but nothing yet marks him out as elite in either direction."
        ),
        comments=(
            "A young Australian central midfielder with a broad, two way game rather than one "
            "standout trait. Encouraging for his age but not yet a finished or specialist player. "
            "Worth monitoring for development rather than an immediate difference maker."
        ),
        strengths=[
            "Genuine two way contribution for his age",
            "Tidy on the ball with some carrying and creative instinct",
            "Competitive without being reckless",
        ],
        weaknesses=[
            "No standout elite trait yet in any single phase",
            "Limited end product going forward",
            "Physical presence and duelling are unremarkable",
            "Young, developing player who needs time",
        ],
        overall_rating="5.5", fit_hk="5",
    ),
    dict(
        slug="T_Nemcik",
        name="T. Nemčík",
        season="2025", club="Rosenborg BK",
        report_date="20-07-2026", position="LCB", age=25, foot="Left",
        height="1.89m",
        physique=(
            "A good, physically mature frame for a centre back. Genuinely competitive in the air."
        ),
        pace=(
            "Surprisingly comfortable carrying the ball out from the back for a centre back, with an "
            "unusually high success rate on the occasions he does dribble past a man."
        ),
        work_rate=(
            "Disciplined and controlled positioning, reflected in a very low number of fouls "
            "conceded."
        ),
        attitude=(
            "Reads danger early rather than relying on recovery challenges. A proactive rather than "
            "reactive defender."
        ),
        technical=(
            "An excellent distributor. Very comfortable and accurate in possession, genuinely happy "
            "building play from the back."
        ),
        intelligence=(
            "His reading of the game and interceptions are the standout traits of his entire game. "
            "Among the best readers of danger covered in this cycle."
        ),
        comments=(
            "A composed, left footed centre back with genuine passing range and sharp anticipation, "
            "though the picture is still based on a small number of matches watched. He looks like "
            "exactly the kind of ball playing defender a possession based build up wants, but needs a "
            "larger body of video evidence before trusting the picture fully."
        ),
        strengths=[
            "Excellent reading of the game and interceptions",
            "Excellent passing range and accuracy",
            "Comfortable carrying the ball out from the back",
            "Left footed variety at the back",
        ],
        weaknesses=[
            "Small number of matches watched so far, a heavy caveat on all of the above",
            "Aerial presence is solid but not standout",
            "Needs a larger run of matches to confirm the picture",
        ],
        overall_rating="6.5", fit_hk="6",
    ),
    dict(
        slug="I_Hughes",
        name="I. Hughes",
        season="2025/2026", club="Wellington Phoenix",
        report_date="20-07-2026", position="LCB / RCB", age=22, foot="Right",
        height="1.83m",
        physique=(
            "A well built, physically dominant centre back. Wins the vast majority of his defensive "
            "and aerial challenges, a genuine physical presence."
        ),
        pace=(
            "Not a mobile ball carrier, but positioned to defend rather than to progress play himself."
        ),
        work_rate=(
            "Extremely disciplined off the ball. Rarely fouls, rarely caught out of position."
        ),
        attitude=(
            "Among the most dominant defenders covered in this cycle in one on one and aerial "
            "contests. Real physical control of his box."
        ),
        technical=(
            "A capable, accurate passer who can move the ball forward when needed, without being a "
            "specialist distributor."
        ),
        intelligence=(
            "Reads danger well and times his interventions cleanly rather than relying on recovery "
            "pace. An old school stopper with modern passing ability, and he even chips in with the "
            "odd goal from set piece situations."
        ),
        comments=(
            "The most dominant defensive player of the centre backs covered here. Outstanding in "
            "duels and in the air, disciplined, and capable enough on the ball not to be a liability "
            "in possession. A genuinely strong defensive foundation."
        ),
        strengths=[
            "Dominant in duels and in the air",
            "Very disciplined, low number of fouls",
            "Reliable passer when asked to move the ball forward",
            "Occasional goal threat from set pieces",
        ],
        weaknesses=[
            "Not a ball carrier or progressive runner",
            "Less involved in duels going forward",
            "Needs to be paired with more mobile cover alongside him",
        ],
        overall_rating="7", fit_hk="6.5",
    ),
    dict(
        slug="L_Vickery",
        name="L. Vickery",
        season="2025/2026", club="Macarthur FC",
        report_date="20-07-2026", position="RW / RAMF", age=20, foot="n/a",
        height="n/a",
        physique=(
            "Undersized for aerial contests, which clearly is not his game. This is a ground based, "
            "dribbling wide threat rather than a physical presence."
        ),
        pace=(
            "A genuinely explosive carrier. His dribbling is the standout trait of his entire game, "
            "comfortably the most dynamic ball carrier covered in this cycle."
        ),
        work_rate=(
            "Limited defensive contribution. His value is almost entirely in the final third."
        ),
        attitude=(
            "Competitive and direct in one on one attacking situations, less involved physically "
            "without the ball."
        ),
        technical=(
            "A genuine dribbler and crosser with real end product for his age. Goals and the way he "
            "gets chances both point to a live goal threat from wide areas."
        ),
        intelligence=(
            "Gets into the box consistently and creates shooting chances for himself through direct "
            "running rather than combination play."
        ),
        comments=(
            "The most explosive attacking talent covered in this cycle. A young, direct wide forward "
            "whose dribbling and movement into the box stand out clearly. Raw defensively, but a "
            "genuine attacking weapon at his age."
        ),
        strengths=[
            "Elite dribbling and directness",
            "Genuine goal threat for a winger",
            "Gets into the box consistently",
            "Direct, progressive ball carrier",
        ],
        weaknesses=[
            "Minimal aerial or physical presence",
            "Limited defensive work",
            "Crossing end product is inconsistent",
            "Young, the defensive side of his game still needs development",
        ],
        overall_rating="7", fit_hk="6",
    ),
    dict(
        slug="S_Figueredo",
        name="S. Figueredo",
        season="2025/2026", club="CD Leganés",
        report_date="20-07-2026", position="RB / RWB / RCB", age=24, foot="Right",
        height="1.80m",
        physique=(
            "Nothing about his frame stands out. A functional rather than imposing physical presence."
        ),
        pace=(
            "Some dribbling and carrying instinct, without anything that marks him out as an "
            "explosive athlete."
        ),
        work_rate=(
            "A steady defensive contributor without being a standout destroyer."
        ),
        attitude=(
            "Competes evenly across his duels without dominating any single phase."
        ),
        technical=(
            "Genuine crossing and passing range for a full back, with a real creative touch despite "
            "almost no goals of his own."
        ),
        intelligence=(
            "More of a creator than a scorer from wide areas. His creative contribution stands out "
            "more than his defending."
        ),
        comments=(
            "A Uruguayan full back whose value leans towards service and creation from wide areas "
            "rather than physical defending or goal threat. A solid, functional player without a "
            "standout trait in any one direction."
        ),
        strengths=[
            "Genuine creative touch from a defensive position",
            "Reasonable crossing range",
            "Positional versatility across the back line",
        ],
        weaknesses=[
            "No goal threat of his own",
            "Defending in duels is unremarkable",
            "No standout physical or defensive trait",
        ],
        overall_rating="5.5", fit_hk="5",
    ),
    dict(
        slug="R_Sanusi",
        name="R. Sanusi",
        season="2025/2026", club="FK Podbrezová",
        report_date="20-07-2026", position="LWB / LB / RWB", age=23, foot="n/a",
        height="n/a",
        physique=(
            "Nothing marks him out physically. A functional rather than dominant frame."
        ),
        pace=(
            "A genuine ball carrier for a wing back. Comfortable dribbling and moving the ball "
            "forward."
        ),
        work_rate=(
            "Engages in a lot of duels and delivers a real amount of service into the box."
        ),
        attitude=(
            "Very willing to engage in duels, though his discipline is a mild concern."
        ),
        technical=(
            "Genuine crossing range and volume, and involved in chance creation more than most "
            "defenders covered in this cycle. The way he gets into good positions in the final third "
            "suggests real quality that simply has not turned into goals of his own yet."
        ),
        intelligence=(
            "A modern, attack minded wing back who gets forward and contributes to the final third "
            "consistently, though his own defending lags behind his attacking involvement."
        ),
        comments=(
            "A Nigerian wing back with real attacking upside. Good ball carrying, crossing, and "
            "involvement in the final third, but his defending and discipline lag behind the "
            "attacking side of his game. A boom or bust player who leans attacking."
        ),
        strengths=[
            "Genuine attacking involvement and chance creation from wing back",
            "Good ball carrying and crossing range",
            "Willing and frequent in duels",
        ],
        weaknesses=[
            "Defending in duels is below what's needed at a higher level",
            "Discipline and fouls are a concern",
            "His own attacking play has not yet turned into goals",
        ],
        overall_rating="6", fit_hk="5.5",
    ),
    dict(
        slug="P_Cancar",
        name="P. Cancar",
        season="2025/2026", club="Western Sydney Wanderers",
        report_date="20-07-2026", position="RB / RCB", age=24, foot="Right",
        height="1.86m",
        physique=(
            "A good, physically competitive frame. Wins more than his share in the air for a full "
            "back."
        ),
        pace=(
            "A willing carrier from deep positions, without being an especially explosive runner."
        ),
        work_rate=(
            "A functional defensive shift, consistently involved without being a standout destroyer."
        ),
        attitude=(
            "Competes fairly evenly across his duels without dominating any one phase."
        ),
        technical=(
            "A reliable, accurate passer with genuine range for a full back."
        ),
        intelligence=(
            "Reads danger early and gets across to intercept at a healthy rate for the position."
        ),
        comments=(
            "A versatile, dual nationality full back or wide centre back on a small number of matches "
            "watched so far. Solid across the board without one standout trait yet, worth monitoring "
            "as more games are seen."
        ),
        strengths=[
            "Competitive in the air for his position",
            "Reliable passer with genuine range",
            "Positional versatility",
            "Reads danger well",
        ],
        weaknesses=[
            "Small number of matches watched so far, treat as an early picture",
            "No standout attacking or defensive trait yet",
            "Ball carrying is limited",
        ],
        overall_rating="5.5", fit_hk="5",
    ),
    dict(
        slug="I_Diomande",
        name="I. Diomandé",
        season="2025", club="FK RFS",
        report_date="20-07-2026", position="CF / LAMF", age=22, foot="Right",
        height="n/a",
        physique=(
            "Size is not confirmed, but his willingness to get stuck into a lot of duels suggests he "
            "does not shy away from contact, even if he does not always come away with the ball."
        ),
        pace=(
            "A direct, high volume dribbler who gets into the box and into shooting positions very "
            "consistently. Clearly comfortable running at defenders."
        ),
        work_rate=(
            "Constantly engaged. A very high amount of duelling marks him as a persistently involved, "
            "combative forward."
        ),
        attitude=(
            "Very willing to engage physically, though a concerning number of fouls conceded suggests "
            "that competitiveness sometimes tips into rash challenges."
        ),
        technical=(
            "Direct and productive in front of goal, though his overall involvement in general "
            "passing play is minimal. A penalty box operator more than a build up participant."
        ),
        intelligence=(
            "Gets into the box and shoots more than almost anyone covered in this cycle relative to "
            "his minutes on the pitch. A real, live goal threat, and his finishing has been slightly "
            "ahead of the chances themselves so far."
        ),
        comments=(
            "A young Ivorian forward with a genuine, direct goal threat and very high energy and "
            "duelling, on a small number of matches watched so far. The discipline concerns and low "
            "general involvement in passing suggest a raw, high ceiling talent who needs polishing "
            "rather than a finished product."
        ),
        strengths=[
            "Goal output slightly ahead of the chances he gets, a good early sign",
            "Very high box arrival and dribbling",
            "Combative and persistent",
        ],
        weaknesses=[
            "Discipline and fouls are a concern",
            "Minimal involvement in general passing play",
            "Comes away with the ball in duels less often than his engagement would suggest",
            "Small number of matches watched so far, treat as an early picture",
        ],
        overall_rating="6.5", fit_hk="5.5",
    ),
    dict(
        slug="J_Nisbet",
        name="J. Nisbet",
        season="2025/2026", club="Roda JC Kerkrade",
        report_date="20-07-2026", position="LDMF / LCMF", age=26, foot="Right",
        height="1.60m",
        physique=(
            "A notably small, light frame for a central midfielder, yet it clearly does not limit how "
            "effective he is across almost every phase of the game."
        ),
        pace=(
            "Comfortable carrying the ball through midfield with a genuine dribbling and progression "
            "game."
        ),
        work_rate=(
            "A true ever present. Barely misses a match, and the sheer number of matches played "
            "speaks to durability and trust from his coaching staff."
        ),
        attitude=(
            "Wins the clear majority of his duels despite his frame. Competes on technique and "
            "anticipation rather than physicality."
        ),
        technical=(
            "A genuinely well rounded passer and dribbler, contributing directly to goals through "
            "both his own finishing and his creative play."
        ),
        intelligence=(
            "A complete central midfield player. Contributes going forward, wins the ball back, and "
            "rarely stops playing. One of the most well rounded players covered in this cycle."
        ),
        comments=(
            "The standout performer of this batch. A physically undersized central midfielder who "
            "compensates entirely through technique, energy, and availability. Genuine goal and "
            "assist involvement from central midfield, excellent defending in duels, and remarkable "
            "durability. His physical frame is the only real question mark against an otherwise "
            "complete package."
        ),
        strengths=[
            "Genuine goal and assist involvement from central midfield",
            "Excellent defending in duels despite his size",
            "Outstanding durability and availability",
            "Well rounded passing and carrying game",
        ],
        weaknesses=[
            "Notably light physical frame, likely to be tested aerially and physically at a higher level",
            "Crossing and wide delivery are modest",
        ],
        overall_rating="8", fit_hk="7",
    ),
]


if __name__ == "__main__":
    for cfg in PLAYERS:
        path = build_report(cfg)
        print(f"  Saved → {path}")
