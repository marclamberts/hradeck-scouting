"""
monte_carlo.py
──────────────
Monte Carlo player projection engine — 3-season forward simulation.

Model design
────────────
For each player we simulate N trajectories for each key metric:

  score(age + t) = score_now
                   × age_multiplier(age, age+t, position, metric_type)
                   × regression_factor(minutes, t)
                   × exp(noise_t)

  noise_t ~ Normal(0, σ)  where σ grows with projection horizon and
                           shrinks with sample size / data confidence.

Age curves are position- and metric-type-specific, built on a
skewed Gaussian:
  - Rise side: wider (slower development)
  - Decline side: narrower (faster decline post-peak)

Output per simulation × season:
  raw trajectories, percentile bands, scenario labels,
  composite projected score and per-metric table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# ── Age curve parameters ───────────────────────────────────────────────────────
# (peak_age, left_sigma, right_sigma)
# left_sigma  = years before peak where performance is at ~60 % of peak
# right_sigma = years after  peak where performance is at ~60 % of peak

MetricType = Literal["scoring", "creative", "physical", "defensive", "gk"]

AGE_CURVES: dict[str, dict[MetricType, tuple[float, float, float]]] = {
    "ST":  {"scoring": (25.5, 3.5, 4.5), "creative": (26.0, 4.0, 5.5), "physical": (23.0, 3.0, 4.0), "defensive": (26.0, 4.5, 5.5), "gk": (27.0, 3.0, 4.0)},
    "W":   {"scoring": (25.0, 3.5, 4.5), "creative": (26.5, 4.0, 5.0), "physical": (23.0, 2.5, 3.5), "defensive": (25.0, 3.5, 4.5), "gk": (27.0, 3.0, 4.0)},
    "AM":  {"scoring": (25.5, 3.5, 4.5), "creative": (27.0, 4.5, 5.5), "physical": (24.0, 3.0, 4.0), "defensive": (26.0, 4.0, 5.0), "gk": (27.0, 3.0, 4.0)},
    "CM":  {"scoring": (26.0, 4.0, 5.0), "creative": (27.5, 4.5, 5.5), "physical": (24.5, 3.5, 4.5), "defensive": (26.5, 4.0, 5.5), "gk": (27.0, 3.0, 4.0)},
    "DM":  {"scoring": (27.0, 4.5, 5.5), "creative": (27.5, 4.5, 5.5), "physical": (25.0, 3.5, 4.5), "defensive": (27.0, 4.5, 5.5), "gk": (27.0, 3.0, 4.0)},
    "FB":  {"scoring": (25.0, 3.5, 4.5), "creative": (26.0, 4.0, 5.0), "physical": (24.0, 3.0, 4.0), "defensive": (27.0, 4.5, 5.5), "gk": (27.0, 3.0, 4.0)},
    "CB":  {"scoring": (27.0, 4.5, 5.5), "creative": (27.5, 5.0, 6.0), "physical": (25.5, 4.0, 5.0), "defensive": (27.5, 4.5, 5.5), "gk": (27.0, 3.0, 4.0)},
    "GK":  {"scoring": (28.0, 4.5, 6.0), "creative": (28.0, 4.5, 5.5), "physical": (27.0, 4.0, 5.0), "defensive": (28.0, 4.5, 6.0), "gk": (27.5, 4.0, 5.5)},
}

# Metric → metric type mapping (for IMPECT model scores)
METRIC_TYPE_MAP: dict[str, MetricType] = {
    "ScoringThreatScore":         "scoring",
    "ASA_GoalsAddedScore":        "scoring",
    "ExpectedThreatScore":        "scoring",
    "CreativeProgressionScore":   "creative",
    "DecisionScore":              "creative",
    "ValueRecruitmentScore":      "creative",
    "CompositeRecruitmentScore":  "creative",
    "PressingScore":              "physical",
    "BallSecurityScore":          "physical",
    "DefensiveDisruptionScore":   "defensive",
    "PerformanceReliabilityScore": "defensive",
    "AgeResaleScore":             "creative",
    # Wyscout per-90 metrics
    "Goals per 90":               "scoring",
    "Non-penalty goals per 90":   "scoring",
    "xG per 90":                  "scoring",
    "Shots per 90":               "scoring",
    "Touches in box per 90":      "scoring",
    "Assists per 90":             "creative",
    "xA per 90":                  "creative",
    "Key passes per 90":          "creative",
    "Progressive passes per 90":  "creative",
    "Smart passes per 90":        "creative",
    "Passes per 90":              "creative",
    "Accurate passes, %":         "creative",
    "Dribbles per 90":            "physical",
    "Progressive runs per 90":    "physical",
    "Successful dribbles, %":     "physical",
    "Successful defensive actions per 90": "defensive",
    "Defensive duels won, %":     "defensive",
    "Aerial duels won, %":        "defensive",
    "Interceptions per 90":       "defensive",
    "Save rate, %":               "gk",
    "Prevented goals per 90":     "gk",
    "Exits per 90":               "gk",
}

# Volatility (season-to-season std dev as fraction of current value)
BASE_VOLATILITY: dict[MetricType, float] = {
    "scoring":   0.22,   # goals/xG volatile
    "creative":  0.14,   # playmaking more stable
    "physical":  0.16,
    "defensive": 0.13,
    "gk":        0.12,
}

# Regression-to-mean strength per horizon (fraction of gap that closes)
REGRESSION_STRENGTH: list[float] = [0.12, 0.20, 0.28]  # seasons 1, 2, 3

# Confidence band labels
SCENARIO_PERCENTILES = {
    "breakout":  90,
    "optimistic": 75,
    "expected":  50,
    "cautious":  25,
    "decline":   10,
}


# ── Utility functions ──────────────────────────────────────────────────────────

def _age_multiplier(
    current_age: float,
    future_age: float,
    position: str,
    metric_type: MetricType,
) -> float:
    """
    Ratio of expected performance at future_age vs current_age.
    Uses asymmetric Gaussian (skewed around peak).
    """
    pos     = position if position in AGE_CURVES else "CM"
    params  = AGE_CURVES[pos].get(metric_type, AGE_CURVES[pos]["creative"])
    peak, sigma_l, sigma_r = params

    def _curve(age: float) -> float:
        delta = age - peak
        sigma = sigma_l if delta < 0 else sigma_r
        return np.exp(-0.5 * (delta / sigma) ** 2)

    c_now    = _curve(current_age)
    c_future = _curve(future_age)
    if c_now < 1e-9:
        return 1.0
    return c_future / c_now


def _confidence_sigma(
    minutes: float,
    horizon: int,
    metric_type: MetricType,
    base_vol: float | None = None,
) -> float:
    """
    Uncertainty (std dev of log-return) for a given projection horizon.
    Shrinks with minutes played; grows with horizon.
    """
    vol = base_vol if base_vol is not None else BASE_VOLATILITY[metric_type]
    # More minutes → more reliable → lower uncertainty
    minutes_factor = max(0.5, min(1.8, 2000.0 / max(minutes, 200.0)))
    # Uncertainty scales with sqrt of horizon
    horizon_factor = np.sqrt(horizon)
    return vol * minutes_factor * horizon_factor


# ── Main dataclass ─────────────────────────────────────────────────────────────

@dataclass
class PlayerProjection:
    player_name:   str
    team:          str
    position:      str
    current_age:   float
    metrics:       dict[str, float]   # metric_name → current value
    minutes:       float = 900.0
    n_seasons:     int   = 3
    n_simulations: int   = 1_000
    random_state:  int   = 42

    # Filled by .run()
    trajectories:  dict[str, np.ndarray] = field(default_factory=dict)  # metric → (n_sims, n_seasons)
    percentiles:   dict[str, dict[int, list[float]]] = field(default_factory=dict)
    composite_trajectories: np.ndarray | None = None

    def run(self) -> "PlayerProjection":
        rng = np.random.default_rng(self.random_state)
        pos = self.position if self.position in AGE_CURVES else "CM"

        for metric, value in self.metrics.items():
            if not np.isfinite(value) or value <= 0:
                continue
            mtype = METRIC_TYPE_MAP.get(metric, "creative")
            traj  = np.zeros((self.n_simulations, self.n_seasons))

            for t in range(1, self.n_seasons + 1):
                future_age   = self.current_age + t
                age_mult     = _age_multiplier(self.current_age, future_age, pos, mtype)
                reg_strength = REGRESSION_STRENGTH[t - 1]
                sigma        = _confidence_sigma(self.minutes, t, mtype)

                # Mean log-return = log(age_mult) + regression pull toward 0
                mean_lr  = np.log(max(age_mult, 0.01)) - reg_strength * np.log(max(value / 50.0, 0.01))
                noise    = rng.normal(0.0, sigma, self.n_simulations)
                traj[:, t - 1] = value * np.exp(mean_lr + noise)

            self.trajectories[metric] = traj

            # Percentiles per season
            self.percentiles[metric] = {}
            for pct in [10, 25, 50, 75, 90]:
                self.percentiles[metric][pct] = np.percentile(traj, pct, axis=0).tolist()

        # Composite: equal-weighted average across all metrics, normalised
        if self.trajectories:
            stack = np.stack(list(self.trajectories.values()), axis=2)  # (sims, seasons, metrics)
            # Normalise each metric by its current value so they're on equal footing
            current_vals = np.array([self.metrics[m] for m in self.trajectories])
            current_vals = np.where(current_vals == 0, 1e-9, current_vals)
            relative     = stack / current_vals[None, None, :]
            self.composite_trajectories = relative.mean(axis=2) * 100  # (sims, seasons)

        return self

    def summary_table(self) -> pd.DataFrame:
        """
        Returns a DataFrame: rows = metric, cols = (Season 1 / 2 / 3) × (P10 / P50 / P90).
        """
        rows = []
        for metric in self.metrics:
            if metric not in self.percentiles:
                continue
            row: dict[str, object] = {"Metric": metric, "Current": round(self.metrics[metric], 3)}
            for t in range(self.n_seasons):
                label = f"S{t+1}"
                row[f"{label} P10"] = round(self.percentiles[metric][10][t], 3)
                row[f"{label} P50"] = round(self.percentiles[metric][50][t], 3)
                row[f"{label} P90"] = round(self.percentiles[metric][90][t], 3)
            rows.append(row)
        return pd.DataFrame(rows)

    def composite_bands(self) -> pd.DataFrame:
        """
        Returns (n_seasons+1) × percentile bands for the composite score.
        Season 0 = current.
        """
        if self.composite_trajectories is None:
            return pd.DataFrame()
        rows = [{"Season": "Now", "P10": 100.0, "P25": 100.0, "P50": 100.0, "P75": 100.0, "P90": 100.0}]
        for t in range(self.n_seasons):
            col = self.composite_trajectories[:, t]
            rows.append({
                "Season": f"Season {t+1}",
                "P10": round(float(np.percentile(col, 10)), 2),
                "P25": round(float(np.percentile(col, 25)), 2),
                "P50": round(float(np.percentile(col, 50)), 2),
                "P75": round(float(np.percentile(col, 75)), 2),
                "P90": round(float(np.percentile(col, 90)), 2),
            })
        return pd.DataFrame(rows)

    def scenario_table(self) -> pd.DataFrame:
        """Best / Expected / Worst trajectories for composite score."""
        if self.composite_trajectories is None:
            return pd.DataFrame()
        final_scores = self.composite_trajectories[:, -1]
        idx = {
            "Breakout":   np.argmax(final_scores >= np.percentile(final_scores, 90)),
            "Optimistic": np.argmax(final_scores >= np.percentile(final_scores, 75)),
            "Expected":   np.argmin(np.abs(final_scores - np.percentile(final_scores, 50))),
            "Cautious":   np.argmin(np.abs(final_scores - np.percentile(final_scores, 25))),
            "Decline":    np.argmin(final_scores),
        }
        rows = []
        for label, sim_idx in idx.items():
            row: dict[str, object] = {"Scenario": label}
            row["Now"] = 100.0
            for t in range(self.n_seasons):
                row[f"Season {t+1}"] = round(float(self.composite_trajectories[sim_idx, t]), 1)
            rows.append(row)
        return pd.DataFrame(rows)


# ── Batch projector ────────────────────────────────────────────────────────────

class BatchProjector:
    """
    Run Monte Carlo projections for every player in a DataFrame and
    return a summary suitable for scouting.
    """

    def __init__(
        self,
        metrics: list[str],
        n_simulations: int = 1_000,
        n_seasons: int = 3,
    ) -> None:
        self.metrics       = metrics
        self.n_simulations = n_simulations
        self.n_seasons     = n_seasons

    def project_player(self, row: pd.Series) -> PlayerProjection | None:
        name    = str(row.get("PlayerName", row.get("Player", "Unknown")))
        team    = str(row.get("TeamName", row.get("Team", "")))
        pos     = str(row.get("PositionGroup", row.get("Position", "CM")))
        age_raw = pd.to_numeric(row.get("AgeYears", row.get("Age")), errors="coerce")
        age     = float(age_raw) if pd.notna(age_raw) else 25.0
        mins    = float(pd.to_numeric(row.get("MinutesPlayed", row.get("Minutes played", 900)), errors="coerce") or 900)

        metric_vals: dict[str, float] = {}
        for m in self.metrics:
            v = pd.to_numeric(row.get(m), errors="coerce")
            if pd.notna(v) and float(v) > 0:
                metric_vals[m] = float(v)

        if not metric_vals:
            return None

        proj = PlayerProjection(
            player_name=name, team=team, position=pos,
            current_age=age, metrics=metric_vals,
            minutes=mins, n_seasons=self.n_seasons,
            n_simulations=self.n_simulations,
        )
        proj.run()
        return proj

    def batch_summary(self, df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
        """
        Returns a flat DataFrame: one row per player with projected
        composite P10/P50/P90 for each future season.
        """
        rows: list[dict] = []
        for _, row in df.head(top_n).iterrows():
            proj = self.project_player(row)
            if proj is None or proj.composite_trajectories is None:
                continue
            r: dict[str, object] = {
                "Player":   proj.player_name,
                "Team":     proj.team,
                "Position": proj.position,
                "Age":      proj.current_age,
            }
            for t in range(self.n_seasons):
                col = proj.composite_trajectories[:, t]
                r[f"S{t+1} P10"] = round(float(np.percentile(col, 10)), 1)
                r[f"S{t+1} P50"] = round(float(np.percentile(col, 50)), 1)
                r[f"S{t+1} P90"] = round(float(np.percentile(col, 90)), 1)
            rows.append(r)
        return pd.DataFrame(rows)
