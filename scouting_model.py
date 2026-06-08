"""
Anomaly, similarity, and set-piece scouting model.

Three engines:
- AnomalyEngine  : z-score / MAD / IQR outlier flagging with classification
- SimilarityEngine: cosine, Euclidean, and Pearson similarity search
- SetPieceAnalyzer: Wyscout-powered set-piece profiling and anomaly detection
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# ── Set-piece metric catalogue ────────────────────────────────────────────────

SET_PIECE_DELIVERY = [
    "Corners per 90",
    "Free kicks per 90",
    "Direct free kicks per 90",
    "Direct free kicks on target, %",
    "Crosses per 90",
    "Accurate crosses, %",
    "Deep completed crosses per 90",
    "Key passes per 90",
]

SET_PIECE_AERIAL = [
    "Aerial duels per 90",
    "Aerial duels won, %",
    "Head goals per 90",
    "Shots blocked per 90",
]

SET_PIECE_SHOOTING = [
    "xG per 90",
    "Shots per 90",
    "Shots on target, %",
    "Goal conversion, %",
    "Non-penalty goals per 90",
    "Touches in box per 90",
]

SET_PIECE_ALL = list(dict.fromkeys(SET_PIECE_DELIVERY + SET_PIECE_AERIAL + SET_PIECE_SHOOTING))

IMPECT_SET_PIECE_METRICS = [
    "SuccessfulAerialDuels_per90",
    "Goals_per90",
    "xG_per90",
    "xA_per90",
    "Shots_per90",
    "ProgressivePasses_per90",
]

Method = Literal["z-score", "mad", "iqr"]
SimilarityMethod = Literal["cosine", "euclidean", "pearson"]


# ── AnomalyEngine ─────────────────────────────────────────────────────────────

class AnomalyEngine:
    """
    Computes per-position anomaly scores and classifies player types.

    Usage::

        engine = AnomalyEngine(threshold=1.8, method="z-score")
        result = engine.fit_transform(df, metrics)
        anomalies = engine.filter_anomalies(result)
    """

    ANOMALY_TYPES = [
        "Hidden Gem",
        "Specialist Elite",
        "Multi-dimensional",
        "Age-adjusted Gem",
        "Consistent Overperformer",
    ]

    def __init__(
        self,
        threshold: float = 1.8,
        method: Method = "z-score",
        groupby: str = "PositionGroup",
    ) -> None:
        self.threshold = threshold
        self.method    = method
        self.groupby   = groupby

    # ── Internal scoring ──────────────────────────────────────────────────────

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        if self.method == "mad":
            med = np.nanmedian(X, axis=0)
            mad = np.nanmedian(np.abs(X - med), axis=0)
            mad = np.where(mad == 0, 1e-9, mad)
            return 0.6745 * (X - med) / mad
        if self.method == "iqr":
            q1  = np.nanpercentile(X, 25, axis=0)
            q3  = np.nanpercentile(X, 75, axis=0)
            iqr = np.where((q3 - q1) == 0, 1e-9, q3 - q1)
            return (X - (q3 + 1.5 * iqr)) / iqr
        # default: z-score
        mu  = np.nanmean(X, axis=0)
        sig = np.nanstd(X, axis=0, ddof=0)
        sig = np.where(sig == 0, 1e-9, sig)
        return (X - mu) / sig

    def _score_group(
        self, grp: pd.DataFrame, metrics: list[str]
    ) -> pd.DataFrame:
        out = grp.copy()
        X   = grp[metrics].values.astype(float)
        Z   = self._standardise(X)
        for i, col in enumerate(metrics):
            out[f"_z_{col}"] = Z[:, i]
        return out

    def _classify(self, row: pd.Series, z_cols: list[str]) -> str:
        peak    = float(row.get("_peak_z", 0))
        breadth = int(row.get("_anomaly_breadth", 0))
        age     = float(pd.to_numeric(row.get("AgeYears"), errors="coerce") or 99)
        comp    = float(pd.to_numeric(row.get("CompositeRecruitmentScore"), errors="coerce") or 50)
        t       = self.threshold
        if peak >= t and comp <= 40:
            return "Hidden Gem"
        if peak >= t * 1.5 and breadth <= 2:
            return "Specialist Elite"
        if breadth >= 4:
            return "Multi-dimensional"
        if peak >= t and age <= 23:
            return "Age-adjusted Gem"
        if breadth >= 2:
            return "Consistent Overperformer"
        return "Specialist Elite"

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        metrics: list[str],
    ) -> pd.DataFrame:
        """
        Compute z-scores, peak z, mean z, anomaly breadth, composite score,
        and anomaly type classification. Returns the enriched DataFrame.
        """
        metrics = [m for m in metrics if m in df.columns]
        if not metrics:
            raise ValueError("None of the requested metrics are present in df.")

        if self.groupby and self.groupby in df.columns:
            frames = [
                self._score_group(grp, metrics)
                for _, grp in df.groupby(self.groupby)
            ]
            zdf = pd.concat(frames, ignore_index=True)
        else:
            zdf = self._score_group(df, metrics)

        z_cols = [f"_z_{m}" for m in metrics]
        zdf["_peak_z"]          = zdf[z_cols].max(axis=1)
        zdf["_mean_z"]          = zdf[z_cols].mean(axis=1)
        zdf["_anomaly_breadth"] = (zdf[z_cols] >= self.threshold).sum(axis=1)
        zdf["_anomaly_score"]   = (
            0.45 * zdf["_peak_z"].clip(lower=0)
            + 0.35 * zdf["_anomaly_breadth"]
            + 0.20 * zdf["_mean_z"].clip(lower=0)
        )
        zdf["_anomaly_type"] = zdf.apply(
            lambda r: self._classify(r, z_cols), axis=1
        )
        return zdf

    def filter_anomalies(
        self,
        zdf: pd.DataFrame,
        types: list[str] | None = None,
        hidden_gem_only: bool = False,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """Return rows where _peak_z >= threshold, sorted by _anomaly_score."""
        out = zdf.loc[zdf["_peak_z"] >= self.threshold].copy()
        if hidden_gem_only and "CompositeRecruitmentScore" in out.columns:
            out = out.loc[out["CompositeRecruitmentScore"].fillna(100) <= 40]
        if types:
            out = out.loc[out["_anomaly_type"].isin(types)]
        return out.sort_values("_anomaly_score", ascending=False).head(top_n)

    def z_summary(self, zdf: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        """Return per-metric z-score summary statistics."""
        z_cols = [f"_z_{m}" for m in metrics if f"_z_{m}" in zdf.columns]
        rows   = []
        for zc in z_cols:
            metric = zc.replace("_z_", "")
            col    = pd.to_numeric(zdf[zc], errors="coerce").dropna()
            rows.append({
                "Metric":    metric,
                "Mean z":    round(float(col.mean()), 3),
                "Std z":     round(float(col.std()), 3),
                "Max z":     round(float(col.max()), 3),
                "Min z":     round(float(col.min()), 3),
                "N anomaly": int((col >= self.threshold).sum()),
            })
        return pd.DataFrame(rows).sort_values("N anomaly", ascending=False)


# ── SimilarityEngine ──────────────────────────────────────────────────────────

class SimilarityEngine:
    """
    Multi-method player similarity search.

    Supported methods: 'cosine', 'euclidean', 'pearson'.
    All methods first standardise features to z-scores before comparison.

    Usage::

        eng = SimilarityEngine(features)
        result = eng.find_similar(df, target_row, method="cosine", n=10)
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features

    def _prepare(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        cols = [f for f in self.features if f in df.columns]
        mat  = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
        mu   = mat.mean(axis=0)
        sig  = mat.std(axis=0)
        sig  = np.where(sig == 0, 1e-9, sig)
        return cols, mat, mu, sig

    # ── Similarity methods ────────────────────────────────────────────────────

    @staticmethod
    def _cosine_sim(target_z: np.ndarray, matrix_z: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix_z, axis=1)
        norms = np.where(norms == 0, 1e-9, norms)
        t_norm = np.linalg.norm(target_z) or 1e-9
        return (matrix_z @ target_z) / (norms * t_norm)

    @staticmethod
    def _euclidean_sim(target_z: np.ndarray, matrix_z: np.ndarray) -> np.ndarray:
        dists = np.sqrt(((matrix_z - target_z) ** 2).sum(axis=1))
        max_d = dists.max() or 1e-9
        return 1.0 - (dists / max_d)

    @staticmethod
    def _pearson_sim(target_z: np.ndarray, matrix_z: np.ndarray) -> np.ndarray:
        n  = matrix_z.shape[1]
        if n < 2:
            return np.zeros(len(matrix_z))
        t_centered = target_z - target_z.mean()
        m_centered = matrix_z - matrix_z.mean(axis=1, keepdims=True)
        num   = m_centered @ t_centered
        denom = np.linalg.norm(m_centered, axis=1) * (np.linalg.norm(t_centered) or 1e-9)
        denom = np.where(denom == 0, 1e-9, denom)
        return num / denom

    # ── Main API ──────────────────────────────────────────────────────────────

    def find_similar(
        self,
        df: pd.DataFrame,
        target_row: pd.Series,
        method: SimilarityMethod = "cosine",
        n: int = 10,
        same_position: bool = True,
    ) -> pd.DataFrame:
        """
        Find the *n* most similar players to *target_row*.

        Returns a copy of df with a '_similarity' column (range −1 to 1 for
        cosine/Pearson, 0 to 1 for Euclidean) and '_sim_method' label.
        """
        pool = df.copy()
        if same_position and "PositionGroup" in df.columns:
            pos = target_row.get("PositionGroup", "")
            pool = pool.loc[pool["PositionGroup"].astype(str) == str(pos)].copy()

        cols, mat, mu, sig = self._prepare(pool)
        if not cols or len(pool) < 2:
            return pd.DataFrame()

        z_mat    = (mat - mu) / sig
        t_raw    = np.array([
            float(pd.to_numeric(target_row.get(c, 0), errors="coerce") or 0)
            for c in cols
        ])
        t_z      = (t_raw - mu) / sig

        if method == "cosine":
            sims = self._cosine_sim(t_z, z_mat)
        elif method == "pearson":
            sims = self._pearson_sim(t_z, z_mat)
        else:
            sims = self._euclidean_sim(t_z, z_mat)

        pool["_similarity"]  = sims
        pool["_sim_method"]  = method

        # Exclude the target player
        name_col = next(
            (c for c in ["PlayerName", "Player", "Name"] if c in pool.columns), None
        )
        if name_col:
            target_name = str(target_row.get(name_col, ""))
            pool = pool.loc[pool[name_col].astype(str) != target_name]

        return pool.sort_values("_similarity", ascending=False).head(n)

    def compare_methods(
        self,
        df: pd.DataFrame,
        target_row: pd.Series,
        n: int = 10,
        same_position: bool = True,
    ) -> dict[SimilarityMethod, pd.DataFrame]:
        """Run all three methods and return a dict of results."""
        return {
            "cosine":    self.find_similar(df, target_row, "cosine",    n, same_position),
            "euclidean": self.find_similar(df, target_row, "euclidean", n, same_position),
            "pearson":   self.find_similar(df, target_row, "pearson",   n, same_position),
        }

    def similarity_matrix(self, df: pd.DataFrame, method: SimilarityMethod = "cosine") -> pd.DataFrame:
        """Return an N×N similarity matrix for all players in df."""
        _, mat, mu, sig = self._prepare(df)
        z = (mat - mu) / sig
        if method == "cosine":
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            z_n   = z / norms
            S     = z_n @ z_n.T
        elif method == "pearson":
            z_c = z - z.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(z_c, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            z_cn  = z_c / norms
            S     = z_cn @ z_cn.T
        else:
            dists = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
            max_d = dists.max() or 1e-9
            S     = 1.0 - dists / max_d

        name_col = next(
            (c for c in ["PlayerName", "Player"] if c in df.columns), None
        )
        labels = df[name_col].astype(str).tolist() if name_col else [str(i) for i in range(len(df))]
        return pd.DataFrame(S, index=labels, columns=labels)


# ── SetPieceAnalyzer ──────────────────────────────────────────────────────────

SET_PIECE_ROLES = {
    "Corner Taker":         {"Corners per 90": 2.0, "Accurate crosses, %": 1.5, "Deep completed crosses per 90": 1.5},
    "Dead Ball Specialist":  {"Direct free kicks per 90": 2.5, "Direct free kicks on target, %": 2.0, "Free kicks per 90": 1.0},
    "Crossing Threat":       {"Crosses per 90": 2.0, "Accurate crosses, %": 1.5, "Deep completed crosses per 90": 1.0},
    "Aerial Threat":         {"Aerial duels won, %": 2.0, "Head goals per 90": 2.5, "Aerial duels per 90": 1.0},
    "Box Presence":          {"Touches in box per 90": 2.0, "xG per 90": 1.5, "Shots per 90": 1.0},
    "Set Piece Blocker":     {"Shots blocked per 90": 2.5, "Aerial duels won, %": 1.5},
}


class SetPieceAnalyzer:
    """
    Set-piece profiling and anomaly detection on Wyscout data.

    Computes z-scores for set-piece metrics, classifies players into set-piece
    roles (Corner Taker, Dead Ball Specialist, etc.) and surfaces anomalies.

    Usage::

        analyzer = SetPieceAnalyzer(threshold=1.5)
        enriched = analyzer.fit_transform(ws_df)
        top_roles = analyzer.top_players_by_role(enriched)
    """

    def __init__(self, threshold: float = 1.5) -> None:
        self.threshold = threshold
        self._anomaly_engine = AnomalyEngine(threshold=threshold, method="z-score", groupby=None)

    def _available_metrics(self, df: pd.DataFrame, metric_list: list[str]) -> list[str]:
        return [m for m in metric_list if m in df.columns]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich df with set-piece z-scores, composite scores, and role
        classifications. Works on any Wyscout-format DataFrame.
        """
        df = df.copy()
        all_sp = self._available_metrics(df, SET_PIECE_ALL)
        if not all_sp:
            return df

        # Standardise each metric group separately
        for metric in all_sp:
            col = pd.to_numeric(df[metric], errors="coerce")
            mu  = col.mean()
            sig = col.std() or 1e-9
            df[f"_spz_{metric}"] = (col - mu) / sig

        # Role composite scores
        for role, weights in SET_PIECE_ROLES.items():
            score = pd.Series(0.0, index=df.index)
            total_w = 0.0
            for m, w in weights.items():
                z_col = f"_spz_{m}"
                if z_col in df.columns:
                    score += w * df[z_col].fillna(0)
                    total_w += w
            df[f"_sp_role_{role}"] = (score / (total_w or 1)).clip(-3, 3)

        # Composite set-piece score (average of positive role scores)
        role_cols = [f"_sp_role_{r}" for r in SET_PIECE_ROLES]
        role_cols_present = [c for c in role_cols if c in df.columns]
        df["_sp_composite"] = df[role_cols_present].clip(lower=0).mean(axis=1)

        # Primary role assignment
        df["_sp_primary_role"] = (
            df[role_cols_present].idxmax(axis=1)
            .str.replace("_sp_role_", "", regex=False)
        )

        # Anomaly flags: any metric z ≥ threshold
        sp_z_cols = [f"_spz_{m}" for m in all_sp]
        df["_sp_peak_z"]     = df[sp_z_cols].max(axis=1)
        df["_sp_breadth"]    = (df[sp_z_cols] >= self.threshold).sum(axis=1)
        df["_sp_is_anomaly"] = df["_sp_peak_z"] >= self.threshold

        return df

    def top_players_by_role(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> dict[str, pd.DataFrame]:
        """Return the top-N players per set-piece role."""
        result: dict[str, pd.DataFrame] = {}
        for role in SET_PIECE_ROLES:
            col = f"_sp_role_{role}"
            if col not in df.columns:
                continue
            name_col = next(
                (c for c in ["Player", "PlayerName"] if c in df.columns), None
            )
            keep = [c for c in [name_col, "Team", "Position", "Age", col, "_sp_composite"] if c and c in df.columns]
            result[role] = (
                df[keep]
                .sort_values(col, ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
        return result

    def anomaly_table(self, df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """Return top anomalies sorted by peak z-score."""
        if "_sp_peak_z" not in df.columns:
            df = self.fit_transform(df)
        name_col = next(
            (c for c in ["Player", "PlayerName"] if c in df.columns), None
        )
        keep = [
            c for c in [
                name_col, "Team", "Position", "Age", "_League",
                "_sp_primary_role", "_sp_peak_z", "_sp_breadth", "_sp_composite",
            ]
            if c and c in df.columns
        ]
        return (
            df.loc[df["_sp_is_anomaly"]]
            .sort_values("_sp_peak_z", ascending=False)
            [keep]
            .head(top_n)
            .reset_index(drop=True)
            .rename(columns={
                name_col: "Player" if name_col else "Player",
                "_League": "League",
                "_sp_primary_role": "Primary Role",
                "_sp_peak_z": "Peak Z",
                "_sp_breadth": "Metric Breadth",
                "_sp_composite": "SP Composite",
            })
        )

    def delivery_similarity(
        self,
        df: pd.DataFrame,
        target_row: pd.Series,
        method: SimilarityMethod = "cosine",
        n: int = 10,
    ) -> pd.DataFrame:
        """Find similar set-piece delivery players."""
        feats   = self._available_metrics(df, SET_PIECE_DELIVERY)
        engine  = SimilarityEngine(feats)
        return engine.find_similar(df, target_row, method=method, n=n, same_position=False)
