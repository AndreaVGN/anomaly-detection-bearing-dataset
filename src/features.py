"""Time-domain features: each 20,480-sample signal is summarised by five numbers."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import BEARINGS, FEATURES, LOG_FEATURES
from .data_io import list_acquisitions, parse_timestamp, read_signals


def time_features(x: np.ndarray) -> dict[str, float]:
    """Features of one channel. The mean is removed first: it is a sensor offset, not vibration."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    if rms == 0.0:                       # flat signal: shape features are undefined
        return {"rms": 0.0, "peak": peak, "p2p": float(np.ptp(x)),
                "kurtosis": float("nan"), "skewness": float("nan")}
    return {
        "rms": rms,
        "peak": peak,
        "p2p": float(np.ptp(x)),
        "kurtosis": float(stats.kurtosis(x, fisher=False)),   # 3 for a Gaussian signal
        "skewness": float(stats.skew(x)),
    }


def transform(feats: pd.DataFrame) -> np.ndarray:
    """Model input: log of the positive long-tailed features, skewness unchanged."""
    X = feats[FEATURES].astype(float).copy()
    for col in LOG_FEATURES:
        X[col] = np.log(X[col])
    return X.to_numpy()


def acquisition_features(signals: np.ndarray, timestamp, bearings: list[str]) -> list[dict]:
    """One row per bearing for a single acquisition of shape (n_samples, n_bearings)."""
    return [{"timestamp": timestamp, "bearing": b, **time_features(signals[:, j])}
            for j, b in enumerate(bearings)]


def dataset_features(folder: Path, bearings: list[str] = BEARINGS, progress=None) -> pd.DataFrame:
    """Features of every acquisition in a dataset folder (long format: timestamp, bearing, features)."""
    files = list_acquisitions(folder)
    rows = []
    for i, f in enumerate(files):
        rows += acquisition_features(read_signals(f, bearings), parse_timestamp(f.name), bearings)
        if progress is not None and i % 100 == 0:
            progress(i, len(files))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
