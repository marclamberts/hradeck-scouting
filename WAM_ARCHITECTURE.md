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

Three files carry the system today:
- **`recruitment_model.py`** — the engine (Movements I & II, most of III)
- **`build_recruitment_model.py`** — Excel workbook assembly (`data/FCHK_Recruitment_Model.xlsx`)
- **`reports/Recruitment_Model_Dashboard.html`** + `export_dashboard_data.py` — the interactive view (Movement III.9)
- **`wyscout_model.py`**, **`scouting_model.py`**, **`monte_carlo.py`** — pre-existing engines this system reuses rather than reimplements

---

## Movement I — Foundation

### 1. Playing Identity Model
**Purpose.** The anchor everything else measures against: Hradec Králové's target
role profiles per position, plus the constraints (budget, league realism, age
strategy) that keep the universe realistic.

**Status: 🟡 Partial.**

| Piece | Status | Where |
|---|---|---|
| Role archetypes per position (2-3 statistical signatures each, e.g. CB → Ball-Playing CB vs Aggressive Stopper; ST → Poacher vs Target Man vs Complete Forward) | ✅ | `recruitment_model.ROLE_ARCHETYPES` (23 archetypes across 8 positions), scored by `compute_role_archetypes()` |
| Reference club / league identity | ✅ | `HRADEC_CLUB`, `HRADEC_LEAGUE` constants |
| Club's own tactical identity, quantified | ✅ | `compute_club_style_profiles()` — Attacking/Creation/Defending/Pressing/Aerial percentiles, feeds the Club Style Profiles radar |
| Tactical principles as explicit data (build-up shape, pressing triggers, transition speed, set-piece approach) | ⬜ | Not encoded as structured input — archetype *weights* implicitly encode a style preference, but there's no standalone "this is how we want to play" config a scout could edit without touching code |
| Target formation (2-3 acceptable variants) | ⬜ | Not modelled — positions are treated independently; no formation-level constraint (e.g. "we need one of these two shapes to work") |
| Budget tier / wage ceiling constraint | ⬜ | No wage data exists in Wyscout at all (hard data gap, not an engineering gap); fee ceiling could be added as a filter on `ModelValueEUR`/`Mkt Val` but isn't yet |
| League realism as an explicit input (not just a fact about the model) | 🟡 | League tiers exist (`build_league_index`) but aren't yet filtered *down* to "leagues we'd actually buy from" — every tier is shown, not just realistic ones |
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

**Status: ✅ Built**, with one real gap (confidence banding).

| Piece | Status | Where |
|---|---|---|
| Structured ingestion of all 165 league files | ✅ | `load_raw_players()`, `build_league_index()` |
| Per-90 percentile within peer group (position × own league) | ✅ | `compute_wyscout_scores()` (from `wyscout_model.py`), called per-league in `build_recruitment_universe()` |
| League-quality coefficient | ✅ | `AdjustedCompositeScore = CompositeRecruitmentScore × (0.5 + 0.5 × LeagueStrength/100)` — this **is** the "don't read a weak-league percentile at face value" layer; `LeagueStrength` comes from the curated pyramid tier (1=100 … 6=22), not from market value |
| Minimum-minutes threshold | ✅ | `DEFAULT_MIN_MINUTES_SENIOR=500`, `DEFAULT_MIN_MINUTES_YOUTH=250` — but this is a **hard cutoff**, not a confidence gradient |
| Peer group includes age band | ⬜ | Peer group today = PositionGroup × League. A 19-year-old and a 33-year-old at the same position are percentiled together. Age enters later (Brighton Score, peak windows) but not at the percentile-computation stage itself |
| Minutes → confidence/uncertainty (not just a cutoff) | ⬜ | **The real gap.** `monte_carlo._confidence_sigma()` already implements exactly this shape (`vol × max(0.5, min(1.8, 2000/minutes)) × sqrt(horizon)`) for projection — it is not yet wired into the *rating* itself. See Model 4. |

### 3. Player Universe Model
**Purpose.** Define and refresh who's in scope: realistic source leagues,
contract status, age bands matching recruitment strategy.

**Status: 🟡 Partial.**

| Piece | Status | Where |
|---|---|---|
| League universe (which of Wyscout's leagues are loaded) | ✅ | `build_league_index()` — all 165, tiered 1-6, senior/youth split |
| Team-type scope (first team vs reserve/academy) | ✅ | `classify_team_type()` — regex on team name (…II, …B, U15-U23, Youth, Academy, Reserves) |
| Age-band filters as a *live* scope, not a one-off | 🟡 | `AgeYears` + `TrajectoryTag` exist per player, and Youth Prospects / Brighton Mechanics each apply their own age cap — but there's no single "current recruitment strategy" toggle that re-scopes every board at once |
| Contract-status filter (expiring / buyable / loan-eligible) | 🟡 | `Contract expires` is parsed and displayed (`Contract` column) everywhere, but nothing *filters on it* yet — an "Expiring 2026" style cut (present in the older Lamberts model) hasn't been ported into this system |
| "Realistic source" filter (leagues Hradec could plausibly buy from) | ⬜ | Same gap as Model 1 — every tier is in scope; there's no budget-linked league cut |

**Next step:** a `--contract-window` / `expiring_within(years)` filter and a
`realistic_leagues_only` flag, both reading from the Model 1 config once that
exists.

---

## Movement II — Analysis

### 4. Performance Rating Model
**Purpose.** Turn Movement I's role profiles into a weighted composite score
per player — as a **distribution with an uncertainty band**, not a point
estimate, since aggregated data can't verify within-season consistency the way
event data can.

**Status: 🟡 Partial — point estimates are solid, uncertainty is the open piece.**

| Piece | Status | Where |
|---|---|---|
| Weighted composite per role | ✅ | `compute_role_archetypes()` → `RoleArchetypeScore` (percentile via `norm.cdf` of a weighted z-score) |
| Cross-league comparable rating | ✅ | `AdjustedCompositeScore` |
| Rating expressed with a confidence/uncertainty band | ⬜ | **Gap.** Every score today is a point estimate. The fix is mechanical, not conceptual — reuse `monte_carlo._confidence_sigma(minutes, horizon=0, metric_type)` (or a simplified same-season variant) to widen a ± band around `AdjustedCompositeScore` and `RoleArchetypeScore` as minutes fall, then carry `RatingLow` / `RatingHigh` alongside the point score everywhere a rating is shown. |

**Concrete plan:** add `compute_rating_confidence(df)` to `recruitment_model.py`:
`band = BASE_SIGMA × max(0.5, min(1.8, REF_MINUTES / minutes))`, applied as
`±band` around the percentile (clipped to [0,100]). Surface as `RatingLow`/
`RatingHigh` columns and, in the dashboard, as an error-bar or shaded range on
every score bar rather than a single number — this is the single highest-value
remaining gap in the whole system, because it's the thing that keeps a 2,600-
minute rating from being read with the same confidence as a 520-minute one.

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
| Contract-length discounting | 🟡 | Contract expiry is known (`Contract expires`) and shown, but doesn't yet discount `ModelValueEUR` (an expiring contract should lower the effective fee) |

**Next step:** a simple discount — `EffectiveValueEUR = ModelValueEUR × f(years_to_expiry)`
(e.g. linear floor at ~40% value inside the last 6 months) — is a half-day add
and closes the contract-discounting gap without new data.

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

**Status: 🟡 Partial — the shortlists exist; the configurable-weight layer doesn't.**

| Piece | Status | Where |
|---|---|---|
| Tiered shortlists | ✅ | `Undervalued & Not Past Peak` (Elite/High/Value), `Brighton Mechanics` (Prime Target/Strong Fit/Speculative/Long Shot), `Squad Needs` priority targets (High/Medium/Low per position) |
| Position-specific boards | ✅ | `Position Boards`, `Role Archetypes` |
| One scoring function, configurable weights per need-type | ⬜ | Today each shortlist has its **own fixed formula** (Brighton Score's 30/30/25/15 split; Squad Needs' percentile-threshold priority). There is no single `score(player, weights)` that a scout can re-weight for "like-for-like" vs "emergency depth" without editing code. |

**Next step:** factor the shared inputs (`AdjustedCompositeScore`, `RoleArchetypeScore`,
`ValueRatio`, `PhysicalLeagueFitScore`, `TrajectoryTag`) into one
`compute_shortlist_score(df, weights: dict)` with named presets
(`LIKE_FOR_LIKE`, `EMERGENCY_DEPTH`, `RESALE_PLAY`) — the four existing boards
become presets of the same function rather than four separate ones.

### 8. Squad Impact Simulation Model
**Purpose.** Monte Carlo squad-level projection — expected team performance
with a target signing inserted, versus the status quo, as a percentile range
of marginal impact, not a single "this will work" verdict.

**Status: 🟡 The engine exists; the squad-impact wiring doesn't.**

| Piece | Status | Where |
|---|---|---|
| Player-level Monte Carlo forward projection (age curves, confidence shrinking with minutes, N-season horizon) | ✅ | `monte_carlo.py` — `PlayerProjection`, `BatchProjector`, `_age_multiplier()`, `_confidence_sigma()` |
| Squad-level aggregation (minutes-weighted club composite) | ✅ | `build_club_rankings()` — this is the "team output" half |
| **The two wired together**: project a target signing forward, insert into Hradec's XI, compare simulated squad output with vs without | ⬜ | **Not built.** `monte_carlo.py` has zero references from `recruitment_model.py` today — it's a standalone engine. |

**Concrete plan (the clearest actionable gap in the whole document):**
1. Take a shortlisted target's current metrics through `monte_carlo.BatchProjector` → N simulated trajectories.
2. For each simulated trajectory, recompute their contribution to Hradec's club composite the same way `build_club_rankings()` weights any squad player (minutes-weighted `CompositeRecruitmentScore`).
3. Compare simulated "squad + player" composite against the real "squad as-is" composite, across all N runs.
4. Output a percentile range (`p10`/`p50`/`p90` marginal impact) per target — pairs naturally with `Squad Needs`' priority targets list as the next column: not just "better than our starter today" but "here's the range of impact if we sign them."

### 9. Dashboard & Feedback Model
**Purpose.** Visualization: role-profile radars, value-vs-rating scatter,
squad-needs heatmap, shortlist tables — kept simple per the brief — plus a
feedback loop so real performance after a signing recalibrates Models 1 and 4.

**Status: 🟡 Visualization is substantially built; the feedback loop is not.**

| Piece | Status | Where |
|---|---|---|
| Value-vs-rating scatter | ✅ | Undervalued panel's Model-vs-Market scatter; Brighton panel's Age-vs-Development-Upside scatter |
| Radar chart | 🟡 | Built at **club** level (`Club Style Profiles` — 5-axis Attacking/Creation/Defending/Pressing/Aerial radar, any club selectable). Not yet built at **player-vs-role-profile** level (a player's own per-90s plotted against their archetype's target weights) — a natural extension of the same `drawRadar()` component, different data. |
| Squad-needs view | 🟡 | Built as a sortable table with priority chips (`Squad Needs` panel/sheet), not a heatmap. Visually a heatmap (position × priority-color grid) is a small step from the current table. |
| Shortlist tables | ✅ | Every board in Movements II/III is a sortable, filterable table in both the workbook and the dashboard |
| Delivery mechanism | 🟡 **Different from the brief, by explicit request** — this session built a **self-contained static HTML dashboard** (`reports/Recruitment_Model_Dashboard.html`, vanilla JS, no server) rather than a live Streamlit app. `app.py` (a Streamlit app) already exists in this repo but serves the older FCHK Model V3 workbooks, not this system. If a live "SetPlayPro-shaped" Streamlit build is wanted instead of/alongside the static HTML, that's a straightforward follow-up — the data layer (`export_dashboard_data.py`'s JSON) is already Streamlit-ready. |
| **Feedback loop** (signed player's real performance recalibrates Model 1 role weights and Model 4 ratings) | ⬜ | **Not built, and needs a new piece of state that doesn't exist yet: a signings log.** Nothing in this pipeline currently records "we signed player X into role Y on date Z." Without that record there's nothing to compare next season's real output against. |

**Concrete plan:** a `data/signings_log.csv` (Player, Club, Position, Archetype,
Signed Date, Model Value at signing, Rating at signing) updated by hand or by
a small script each window. On each rebuild, join current-season actuals back
onto that log to produce a `Model Calibration` sheet — predicted vs actual
rating and value, per signing — closing the loop the brief describes.

---

## Gap Summary — ordered by effort vs impact

| # | Gap | Data needed? | Effort | Impact |
|---|---|---|---|---|
| 1 | Squad Impact Simulation (wire `monte_carlo.py` into `build_club_rankings`) | No — both pieces exist | Medium | **High** — turns "is this a good buy" into "how much better does our XI get" |
| 2 | Rating uncertainty bands (Model 4) | No — reuse `_confidence_sigma` | Low–Medium | **High** — every score in the system currently hides its own reliability |
| 3 | Configurable shortlist weight presets (Model 7) | No | Low | Medium — mostly a refactor of existing formulas |
| 4 | Contract-length value discount (Model 5) | No | Low | Medium |
| 5 | Player-vs-role-profile radar (Model 9) | No — reuse `drawRadar()` | Low | Medium |
| 6 | Signings log + calibration feedback loop (Model 9) | New: a log you maintain | Medium | High, but only pays off *after* a transfer window passes |
| 7 | Minutes-trend / rising-vs-declining role (Model 6) | New: prior-season Wyscout files | Medium | Medium |
| 8 | Realistic-league / budget-tier filter (Models 1 & 3) | No | Low | Medium — mostly a config/UX add |
| 9 | Injury history, agent risk, wages (Models 5 & 6) | **External source required** | — | Not closable from this pipeline |

Everything in Movements I and II that *can* be built from Wyscout aggregated
data alone is built. What's left splits cleanly into "wire two things that
already exist together" (#1), "surface uncertainty that's already computed
elsewhere" (#2), and "needs data this pipeline doesn't have" (#9) — which is
exactly where a season-aggregate model should hit its ceiling.
