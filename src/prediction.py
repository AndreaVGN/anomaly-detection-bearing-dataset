"""Prediction pipeline: acquisitions -> features -> score and outcome per bearing.

Usable from Python, from the API (POST /predict) and from the command line:
    python -m src.prediction data/2nd_test/2004.02.17.07.12.39 data/2nd_test/2004.02.17.07.22.39
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import registry
from .config import MODEL_DIR
from .data_io import parse_timestamp, read_signals
from .features import time_features, transform
from .model import persistent_alarms

NORMAL, ANOMALOUS = "normal", "anomalous"


class PredictionError(ValueError):
    pass


@dataclass
class Acquisition:
    signals: dict[str, np.ndarray]          # bearing -> raw signal
    timestamp: datetime | None = None
    id: str | None = None


def validate(acq: Acquisition, model: registry.DetectorModel, index: int) -> None:
    name = acq.id or acq.timestamp or f"#{index}"
    if not acq.signals:
        raise PredictionError(f"Acquisition {name}: no signals")
    unknown = set(acq.signals) - set(model.bearings)
    if unknown:
        raise PredictionError(f"Acquisition {name}: unknown bearings {sorted(unknown)}; "
                              f"the model knows {sorted(model.bearings)}")
    for b, x in acq.signals.items():
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) != model.samples_per_acquisition:
            raise PredictionError(f"Acquisition {name}, {b}: expected {model.samples_per_acquisition} "
                                  f"samples (1 s of signal), got {x.size}")
        if not np.isfinite(x).all():
            raise PredictionError(f"Acquisition {name}, {b}: signal contains NaN or infinite values")
        if np.ptp(x) == 0:
            raise PredictionError(f"Acquisition {name}, {b}: constant signal, features are undefined")


def score_bearing(x: np.ndarray, m) -> dict:
    feats = time_features(x)
    score = float(m.score(transform(pd.DataFrame([feats])))[0])
    above = score > m.threshold
    return {
        "score": round(score, 4),
        "threshold": round(m.threshold, 4),
        "status": ANOMALOUS if above else NORMAL,
        "above_threshold": bool(above),
        "features": {k: _r(v) for k, v in feats.items()},
    }


def predict(model: registry.DetectorModel, acquisitions: list[Acquisition]) -> list[dict]:
    """Score every acquisition. Results are returned in the input order.

    status: per-acquisition outcome, normal / anomalous (stateless).
    alarm:  True when the acquisition belongs to a run of at least `persistence` consecutive
            anomalous acquisitions inside this request, in time order. It needs a time series:
            with fewer acquisitions than `persistence` it is always False.
    """
    if not acquisitions:
        raise PredictionError("No acquisitions to score")
    for a in acquisitions:                      # compare aware and naive timestamps safely
        if a.timestamp is not None and a.timestamp.tzinfo is not None:
            a.timestamp = a.timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    for i, acq in enumerate(acquisitions):
        validate(acq, model, i)

    results = [{"id": a.id, "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                "bearings": {b: score_bearing(np.asarray(x, dtype=float), model.bearings[b])
                             for b, x in a.signals.items()}}
               for a in acquisitions]

    # time order for the persistence rule (input order when timestamps are missing)
    if all(a.timestamp is not None for a in acquisitions):
        order = sorted(range(len(acquisitions)), key=lambda i: acquisitions[i].timestamp)
    else:
        order = list(range(len(acquisitions)))
    k = int(model.config["persistence"])
    for b in model.bearings:
        idx = [i for i in order if b in results[i]["bearings"]]
        above = [results[i]["bearings"][b]["above_threshold"] for i in idx]
        for i, a in zip(idx, persistent_alarms(above, k)):
            results[i]["bearings"][b]["alarm"] = bool(a)

    for r in results:
        r["anomalous_bearings"] = [b for b, v in r["bearings"].items() if v["status"] == ANOMALOUS]
    return results


def acquisition_from_file(path: Path, bearings: list[str]) -> Acquisition:
    signals = read_signals(path, bearings)
    return Acquisition(signals={b: signals[:, j] for j, b in enumerate(bearings)},
                       timestamp=parse_timestamp(Path(path).name), id=Path(path).name)


def _r(v: float) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 6)


def main():
    parser = argparse.ArgumentParser(description="Score IMS acquisition files with a trained model")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()
    model = registry.load(MODEL_DIR, args.model_version)
    bearings = list(model.bearings)
    results = predict(model, [acquisition_from_file(f, bearings) for f in args.files])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
