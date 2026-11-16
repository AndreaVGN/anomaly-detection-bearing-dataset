"""Mahalanobis detector: one model per bearing, fitted on a period assumed healthy.

score(x) = sqrt(z^T Σ^-1 z), with z the features standardised on the healthy period and
Σ their sample covariance on the same period.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

MAX_CONDITION_NUMBER = 1e8


@dataclass
class BearingModel:
    mean: list[float]
    std: list[float]
    precision: list[list[float]]
    threshold: float
    condition_number: float         # of the healthy covariance: large = nearly singular
    n_healthy: int

    # --- core maths -------------------------------------------------------
    def _z(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) - np.asarray(self.mean)) / np.asarray(self.std)

    def score(self, X: np.ndarray) -> np.ndarray:
        return mahalanobis(self._z(X), np.asarray(self.precision))

    def to_dict(self) -> dict:
        return asdict(self)


def mahalanobis(Z: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each row of Z (already centred and standardised) from the origin."""
    Z = np.atleast_2d(Z)
    return cdist(Z, np.zeros((1, Z.shape[1])), metric="mahalanobis", VI=precision).ravel()


def fit_params(X: np.ndarray) -> dict:
    """Mean, std and precision (inverse covariance) of the healthy feature matrix."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    if np.any(std == 0):
        raise ValueError("A feature is constant in the healthy period: cannot standardise")
    cov = np.cov((X - mean) / std, rowvar=False)
    cond = float(np.linalg.cond(cov))
    if not np.isfinite(cond) or cond > MAX_CONDITION_NUMBER:
        raise ValueError(f"Healthy covariance is (nearly) singular, condition number {cond:.1e}: "
                         "some features are redundant")
    return {"mean": mean, "std": std, "precision": np.linalg.inv(cov), "condition_number": cond}


def score_with(X: np.ndarray, p: dict) -> np.ndarray:
    return mahalanobis((X - p["mean"]) / p["std"], p["precision"])


def oof_scores(X_h: np.ndarray, fit_fn=fit_params, score_fn=score_with, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold scores on the healthy period: contiguous blocks, never shuffled (time series)."""
    oof = np.empty(len(X_h))
    for tr, te in KFold(n_splits=n_splits, shuffle=False).split(X_h):
        oof[te] = score_fn(X_h[te], fit_fn(X_h[tr]))
    return oof


def fit_bearing(X_h: np.ndarray, quantile: float, n_splits: int) -> BearingModel:
    p = fit_params(X_h)
    threshold = float(np.quantile(oof_scores(X_h, n_splits=n_splits), quantile))
    return BearingModel(
        mean=p["mean"].tolist(), std=p["std"].tolist(), precision=p["precision"].tolist(),
        threshold=threshold, condition_number=p["condition_number"],
        n_healthy=int(len(X_h)),
    )


# --- alarm logic ----------------------------------------------------------------
def persistent_alarms(above, k: int) -> np.ndarray:
    """True only inside runs of at least k consecutive True values."""
    above = np.asarray(above, dtype=bool)
    runs, _ = ndimage.label(above)                  # one id per run of consecutive True values
    run_length = np.bincount(runs)[runs]            # length of the run each acquisition belongs to
    return above & (run_length >= k)


def alarm_intervals(ts, alarm) -> pd.DataFrame:
    """Start, end and length of every alarm run."""
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    runs, _ = ndimage.label(np.asarray(alarm, dtype=bool))
    groups = ts[runs > 0].groupby(runs[runs > 0])
    return pd.DataFrame({"start": groups.min(), "end": groups.max(), "acquisitions": groups.size()},
                        columns=["start", "end", "acquisitions"]).reset_index(drop=True)

