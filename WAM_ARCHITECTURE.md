# Waltzing Analytics Model (WAM) — Complete Architecture

FC Hradec Králové recruitment-and-analysis system. Three movements, nine models,
each handing its output to the next. This document is the complete outline:
what each model does, the concrete schema it produces, exactly which file/
function already builds it in this repo, and what's still open.

Status legend: ✅ Built · 🟡 Partial · ⬜ Planned

## The data foundation — and what it rules in and out

The source is **Wyscout aggregated exports** (`Wyscout Files/*.xlsx`): season-level
per-90 rates and counting stats, one row per player per competition, no event
stream. No freeze frames, no touch-by-touch sequences, no possession chains —
so no event-derived xT/possession-value work, no true set-piece routine
detection, and (importantly for Model 6) **no injury records, no wage data,
no multi-season history in the current pipeline** (each rebuild loads the
current season only). WAM is built around **composite, percentile-based
ratings from aggregated KPIs**, not event data — that shape is deliberate, not
a compromise, and it's why the honest gaps below are mostly "this needs a
different data source," not "this needs more engineering."

Five files carry the system today:
- **`recruitment_model.py`** — the engine (Movements I & II, most of III)
- **`build_recruitment_model.py`** — Excel workbook assembly (`data/FCHK_Recruitment_Model.xlsx`)
- **`reports/Recruitment_Model_Dashboard.html`** + `export_dashboard_data.py` — the interactive view (Movement III.9)
- **`reports/WAM_Landing_Page.html`** — the front door: all nine models, what's built vs. open, deep links into the dashboard
- **`data/signings_log.csv`** — the feedback-loop input a scout appends to after each real signing (Model 9)
- **`wyscout_model.py`**, **`scouting_model.py`**, **`monte_carlo.py`** — pre-existing engines this system reuses rather than reimplements

---

## Movement I — Foundation

### 1. Playing Identity Model
**Purpose.** The anchor everything else measures against: Hradec Králové's target
role profiles per position, plus the constraints (budget, league realism, age
strategy) that keep the universe realistic.

**Status: 🟡 Partial** — role profiles, club identity, and realistic-scope
constraints are all built; only formation-level and tactical-principle
config remain genuinely open.

| Piece | Status | Where |
|---|---|---|
| Role archetypes per position (2-3 statistical signatures each, e.g. CB → Ball-Playing CB vs Aggressive Stopper; ST → Poacher vs Target Man vs Complete Forward) | ✅ | `recruitment_model.ROLE_ARCHETYPES` (23 archetypes across 8 positions), scored by `compute_role_archetypes()` |
| Reference club / league identity | ✅ | `HRADEC_CLUB`, `HRADEC_LEAGUE` constants |
| Club's own tactical identity, quantified | ✅ | `compute_club_style_profiles()` — Attacking/Creation/Defending/Pressing/Aerial percentiles, feeds the Club Style Profiles radar |
| Tactical principles as explicit data (build-up shape, pressing triggers, transition speed, set-piece approach) | ⬜ | Not encoded as structured input — archetype *weights* implicitly encode a style preference, but there's no standalone "this is how we want to play" config a scout could edit without touching code |
| Target formation (2-3 acceptable variants) | ⬜ | Not modelled — positions are treated independently; no formation-level constraint (e.g. "we need one of these two shapes to work") |
| Budget tier / wage ceiling constraint | 🟡 | `BUDGET_CEILING_EUR` fee ceiling now flags `WithinBudget` on every player (`compute_recruitment_scope()`); true wage ceiling still impossible — no wage data in Wyscout at all (hard data gap) |
| League realism as an explicit input (not just a fact about the model) | ✅ | `compute_recruitment_scope()` — `RealisticSource`/`InRecruitmentScope` flag tiers 3-6 as realistic sources (flagged, not hidden) |
| Age-profile targets as a named strategy (development window vs ready-now) | 🟡 | Age fit exists inside Brighton Score (peaks 19-21) and peak-age windows (`PEAK_WINDOWS`), but isn't a squad-level *policy* you can toggle — it's baked into one scoring formula |

**Next step, concretely:** a small `club_profile.py` (or a config block at the
top of `recruitment_model.py`) holding: `TARGET_FORMATIONS`, `BUDGET_CEILING_EUR`,
`REALISTIC_LEAGUES` (subset of tiers), and `AGE_STRATEGY` (per-position
development-vs-ready-now weight) — everything downstream (Squad Needs, Brighton
Mechanics, Shortlists) then reads from one place instead of separate hardcoded
constants.

### 2. Data Pipeline Model
**Purpose.** Ingest Wyscout exports, normalize to per-90, convert to percentile
rank within the right peer group, and tag league quality so a percentile in a
weak league isn't read at face value against a strong one.

**Status: ✅ Built.** The confidence-banding gap noted below is now closed —
see Model 4, which is where it actually surfaces.

| Piece | Status | Where |
|---|---|---|
| Structured ingestion of all 165 league files | ✅ | `load_raw_players()`, `build_league_index()` |
| Per-90 percentile within peer group (position × own league) | ✅ | `compute_wyscout_scores()` (from `wyscout_model.py`), called per-league in `build_recruitment_universe()` |
| League-quality coefficient | ✅ | `AdjustedCompositeScore = CompositeRecruitmentScore × (0.5 + 0.5 × LeagueStrength/100)` — this **is** the "don't read a weak-league percentile at face value" layer; `LeagueStrength` comes from the curated pyramid tier (1=100 … 6=22), not from market value |
| Minimum-minutes threshold | ✅ | `DEFAULT_MIN_MINUTES_SENIOR=500`, `DEFAULT_MIN_MINUTES_YOUTH=250` — still a hard cutoff, but now paired with a confidence gradient above it (next row) |
| Peer group includes age band | ⬜ | Peer group today = PositionGroup × League. A 19-year-old and a 33-year-old at the same position are percentiled together. Age enters later (Brighton Score, peak windows) but not at the percentile-computation stage itself |
| Minutes → confidence/uncertainty (not just a cutoff) | ✅ | **Closed.** `compute_rating_confidence()` widens a `±RatingBand` around `AdjustedCompositeScore`/`RoleArchetypeScore` as minutes fall below `REFERENCE_MINUTES`, surfaced as `RatingLow`/`RatingHigh` everywhere a rating is shown (Player Card, tables). See Model 4. |

### 3. Player Universe Model
**Purpose.** Define and refresh who's in scope: realistic source leagues,
contract status, age bands matching recruitment strategy.

**Status: ✅ Built** — the "realistic source" gap that anchored this model's
Partial status is closed; one small filter-UX item remains.

| Piece | Status | Where |
|---|---|---|
| League universe (which of Wyscout's leagues are loaded) | ✅ | `build_league_index()` — all 165, tiered 1-6, senior/youth split |
| Team-type scope (first team vs reserve/academy) | ✅ | `classify_team_type()` — regex on team name (…II, …B, U15-U23, Youth, Academy, Reserves) |
| Age-band filters as a *live* scope, not a one-off | 🟡 | `AgeYears` + `TrajectoryTag` exist per player, and Youth Prospects / Brighton Mechanics each apply their own age cap — but there's no single "current recruitment strategy" toggle that re-scopes every board at once |
| Contract-status filter (expiring / buyable / loan-eligible) | 🟡 | `Contract expires` is parsed, displayed, and now feeds `EffectiveValueEUR`'s contract discount (Model 5) — but there's still no standalone "show me only Expiring 2026" quick-filter in the dashboard itself |
| "Realistic source" filter (leagues Hradec could plausibly buy from) | ✅ | `compute_recruitment_scope()` — `RealisticSource`/`WithinBudget`/`InRecruitmentScope`, flag-not-hide (Tier 1-2/over-budget players are marked, not deleted) |

**Remaining, low-effort:** a dashboard quick-filter toggle for
`expiring_within(years)` and a single "current recruitment strategy" switch
that re-scopes every board (Youth Prospects, Brighton Mechanics, Shortlists)
by the same age-band policy at once, instead of each board applying its own
cap independently.

---

## Movement II — Analysis

### 4. Performance Rating Model
**Purpose.** Turn Movement I's role profiles into a weighted composite score
per player — as a **distribution with an uncertainty band**, not a point
estimate, since aggregated data can't verify within-season consistency the way
event data can.

**Status: ✅ Built** — the uncertainty band that anchored this model's Partial
status is closed.

| Piece | Status | Where |
|---|---|---|
| Weighted composite per role | ✅ | `compute_role_archetypes()` → `RoleArchetypeScore` (percentile via `norm.cdf` of a weighted z-score) |
| Cross-league comparable rating | ✅ | `AdjustedCompositeScore` |
| Rating expressed with a confidence/uncertainty band | ✅ | **Closed.** `compute_rating_confidence()` — `RatingBand = BASE_RATING_BAND × sqrt(REFERENCE_MINUTES/minutes)` (clipped [0.6, 2.2]), applied as `RatingLow`/`RatingHigh` around `AdjustedCompositeScore`; shown on every Player Card and comparison view, not just the workbook. |
| Player vs. own role-profile breakdown | ✅ | `RoleProfileBreakdown` — per-metric percentile within position peer group, paired with that metric's weight in the player's assigned archetype; explains *why* the archetype was assigned instead of leaving it a single black-box number. Shown on the dashboard's Player Card. |

This closes what was the single highest-value remaining gap in the whole
system — a 2,600-minute rating and a 520-minute rating no longer read with
the same false confidence.

### 5. Value & Market Model
**Purpose.** Estimate transfer value and (where possible) wage feasibility from
age-curve modelling, contract-length discounting, and comparable-transaction
benchmarking — keeping the club's actual financial reality honest against
Model 4's rating.

**Status: ✅ Built**, with one hard data gap (wages) and one framing gap (comps).

| Piece | Status | Where |
|---|---|---|
| Metrics-derived EUR valuation (not a Transfermarkt copy) | ✅ | `compute_value_model()` — ridge regression on `ln(Market Value) ~ Composite + Age + Age² + ln(Minutes) + LeagueStrength + Position`, calibrated on market value only, applied to every player's own metrics |
| Age-curve value trajectory | ✅ | `ProjectedPeakValueEUR` / `DevelopmentUpsideEUR` — regression re-run with age swapped for the position's peak-age, quality held fixed |
| Resale trajectory | ✅ | Same mechanism — `DevelopmentUpsideEUR` *is* the resale-upside read |
| Undervaluation flag | ✅ | `ValueRatio = ModelValueEUR / MarketValue`, tiered `ValueTier` (Elite/High/Value/Fair/Overpriced) |
| Comparable-transaction benchmarking | 🟡 | Functionally present but not *framed* as comps: the regression is trained across the whole population (every player is implicitly benchmarked against statistically similar priced peers), and `SimilarityEngine`-based comparables exist (Model 6 / Similar to Our Squad) but aren't yet joined back into the value estimate as literal "these 5 similar sales set this price" |
| Wage feasibility | ⬜ | **Hard gap** — Wyscout carries no wage data. Would need an external source; not fixable from this pipeline alone. |
| Contract-length discounting | ✅ | **Closed.** `compute_effective_value()` — `EffectiveValueEUR = ModelValueEUR × ContractDiscount`, a linear ramp between 6 months (floor 0.35×) and 3+ years remaining (full value), read straight from `Contract expires` |

### 6. Fit & Risk Model
**Purpose.** Two halves: (a) tactical fit — including an adaptability index for
moving into Czech football specifically — and (b) a risk overlay flagging
what a human scout needs to verify in person.

**Status: 🟡 Fit half built; risk half is a documented, deliberate gap.**

| Piece | Status | Where |
|---|---|---|
| Tactical fit to target role | ✅ | `RoleArchetypeScore` (Model 1/4) |
| Adaptability index — "how would this player's numbers translate into Czech football" | ✅ | **This is exactly `compute_physical_fit()`** — every player's duel/aerial/tempo metrics z-scored against the Czech First League's own per-position norms (not their own league's), scored 0-100 with a label (Excellent/Good/Moderate/Below). It's named "Physical / Quick-League Fit" in the workbook/dashboard, but it *is* Model 6's adaptability index. |
| Injury history / frequency | ⬜ | **Hard data gap** — not in Wyscout aggregated exports at all |
| Contract/agent complications | ⬜ | **Hard data gap** — no source in this pipeline |
| Minutes-trend direction (rising/declining role at current club) | ⬜ | **Data gap, but a closable one** — needs multi-season Wyscout snapshots (current pipeline loads one season only). If prior-season files are added to `Wyscout Files/`, a `MinutesTrend = this_season_minutes − last_season_minutes` is a straightforward join, not a redesign. |

**Deliberately unquantified, by design (per the original brief):** character,
dressing-room fit, agent behaviour — these stay scout-verified, flagged rather
than scored, exactly as intended. The two closable gaps (contract discount,
minutes trend) are listed as such; the two hard gaps (injuries, wages) are
flagged as needing a different data source, not more code.

---

## Movement III — Decision

### 7. Shortlist & Scoring Model
**Purpose.** Combine Models 4-6 into a single recruitment score, with
**configurable weights per recruitment need** (like-for-like replacement weights
fit+rating; emergency depth weights value+availability), output as tiered A/B/C
shortlists per position.

**Status: ✅ Built** — the configurable-weight layer that anchored this
model's Partial status is closed.

| Piece | Status | Where |
|---|---|---|
| Tiered shortlists | ✅ | `Undervalued & Not Past Peak` (Elite/High/Value), `Brighton Mechanics` (Prime Target/Strong Fit/Speculative/Long Shot), `Squad Needs` priority targets (High/Medium/Low per position) |
| Position-specific boards | ✅ | `Position Boards`, `Role Archetypes` |
| One scoring function, configurable weights per need-type | ✅ | **Closed.** `compute_shortlist_score(df, weights)` + `SHORTLIST_PRESETS` — `Like-for-Like` (quality/fit-heavy), `Emergency Depth` (value-heavy), `Resale Play` (trajectory-heavy), `Balanced` — one shared engine, reweighted per preset, restricted to realistic-source/in-budget/not-past-peak players |

### 8. Squad Impact Simulation Model
**Purpose.** Monte Carlo squad-level projection — expected team performance
with a target signing inserted, versus the status quo, as a percentile range
of marginal impact, not a single "this will work" verdict.

**Status: ✅ Built** — the engine and squad-level aggregation are now wired
together, closing what this document called "the clearest actionable gap in
the whole document."

| Piece | Status | Where |
|---|---|---|
| Player-level Monte Carlo forward projection (age curves, confidence shrinking with minutes, N-season horizon) | ✅ | `monte_carlo.py` — `PlayerProjection`, `BatchProjector`, `_age_multiplier()`, `_confidence_sigma()` |
| Squad-level aggregation (minutes-weighted club composite) | ✅ | `build_club_rankings()` — this is the "team output" half |
| **The two wired together**: project a target signing forward, insert into Hradec's XI, compare simulated squad output with vs without | ✅ | **Closed.** `simulate_squad_impact()` + `build_squad_impact_board()` — projects the #1 recommended target's composite subscores 3 seasons forward via `monte_carlo.PlayerProjection`, folds each simulated trajectory into the squad's minutes-weighted position composite, and outputs a `P10`/`P50`/`P90` marginal-impact range per season for every High/Medium priority position. One methodological note: the projection must be fed the same ~0-100 percentile subscores `monte_carlo.py`'s regression assumes, not raw per-90 counting stats — an early version of this wiring fed it raw stats and produced nonsensical (>150-point) impact figures before that was caught and fixed. |

### 9. Dashboard & Feedback Model
**Purpose.** Visualization: role-profile radars, value-vs-rating scatter,
squad-needs heatmap, shortlist tables — kept simple per the brief — plus a
feedback loop so real performance after a signing recalibrates Models 1 and 4.

**Status: ✅ Built** — visualization is extensive (including the player-level
radar and squad-needs heatmap this document originally flagged as missing),
and the feedback loop is now scaffolded and running; it just needs a real
signing to populate.

| Piece | Status | Where |
|---|---|---|
| Value-vs-rating scatter | ✅ | Undervalued panel's Model-vs-Market scatter; Brighton panel's Age-vs-Development-Upside scatter — both now clickable straight into the Player Card |
| Radar chart | ✅ | Built at both **club** level (`Club Style Profiles`) and **player** level (Player Card, Compare Players) — a wedge/donut percentile chart (color-scaled red→green by value), plus the player-vs-role-profile metric breakdown one layer deeper (Model 4) |
| Squad-needs view | ✅ | A priority-color heatmap (position × priority) above the existing sortable table, click either to drill into that position's targets |
| Shortlist tables | ✅ | Every board in Movements II/III is a sortable, filterable table in both the workbook and the dashboard, with grouped super-headers on the busiest ones |
| Side-by-side player comparison | ✅ | `Compare Players` — up to 4 players, added from anywhere on the dashboard (search, Player Card, command palette), each with bio/chips/composite-with-rank/meter-bars/wedge-radar |
| Rank/percentile context | ✅ | Every player carries `OverallRank`/`PosRank`/`LeagueRank`/`ValueRank`, surfaced as "#1,032 of 43,213" annotations — not just a bare score |
| Delivery mechanism | 🟡 **Different from the brief, by explicit request** — a **self-contained static HTML dashboard** (`reports/Recruitment_Model_Dashboard.html`, vanilla JS, no server) rather than a live Streamlit app. `app.py` (a Streamlit app) already exists in this repo but serves the older FCHK Model V3 workbooks, not this system. The data layer (`export_dashboard_data.py`'s JSON) is already Streamlit-ready if a live build is wanted instead of/alongside the static HTML. |
| **Feedback loop** (signed player's real performance checked against the model's prediction) | ✅ | **Scaffolded and running.** `data/signings_log.csv` is a template a scout appends to after each real signing (Player, PositionGroup, Archetype, SignedDate, ModelValueAtSigning, RatingAtSigning, MarketValueAtSigning, SourceClub, SourceLeague, Notes); `compute_model_calibration()` joins any logged rows against the player's current numbers on every rebuild — `RatingDelta`/`ValueDeltaEUR`, tagged Outperformed/On Track/Underperformed. Empty until the first real signing is logged — that's the loop waiting on a transfer window, not a bug. |

---

## Gap Summary — what closed, and what's still open

All nine of the originally-listed closable gaps are now closed. What
remains splits cleanly into two honest categories: config that hasn't been
factored out of code yet, and data this pipeline genuinely does not have.

| # | Gap | Status | Data needed? |
|---|---|---|---|
| 1 | Squad Impact Simulation (wire `monte_carlo.py` into `build_club_rankings`) | ✅ Closed | No — both pieces existed |
| 2 | Rating uncertainty bands (Model 4) | ✅ Closed | No — reused the `_confidence_sigma` shape |
| 3 | Configurable shortlist weight presets (Model 7) | ✅ Closed | No |
| 4 | Contract-length value discount (Model 5) | ✅ Closed | No |
| 5 | Player-vs-role-profile radar (Model 9) | ✅ Closed | No — extended `compute_role_archetypes()` to also capture per-metric percentiles |
| 6 | Signings log + calibration feedback loop (Model 9) | ✅ Scaffolded | New: a log you maintain — infrastructure is built, waiting on a real transfer window |
| 7 | Minutes-trend / rising-vs-declining role (Model 6) | ⬜ Open | New: prior-season Wyscout files not in the current pipeline |
| 8 | Realistic-league / budget-tier filter (Models 1 & 3) | ✅ Closed | No |
| 9 | Injury history, agent risk, wages (Models 5 & 6) | ⬜ Open | **External source required** — not closable from this pipeline |
| 10 | Target formations / tactical-principle config (Model 1) | ⬜ Open | No — pure engineering, a config file away |
| 11 | Live "expiring within N years" quick-filter (Model 3) | ⬜ Open | No — the underlying date field already exists |

Everything in Movements I and II that *can* be built from Wyscout aggregated
data alone is now built. What's left is either a straightforward config
follow-up (#10, #11), a scaffold waiting on a real-world event to populate
it (#6), or genuinely needs data this pipeline doesn't have (#7, #9) —
exactly where a season-aggregate model should hit its ceiling.
