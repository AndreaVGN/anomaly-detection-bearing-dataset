"""Training pipeline: dataset folder -> features -> one detector per bearing -> saved model.

Usable from Python, from the API (POST /train) and from the command line:
    python -m src.training --dataset 2nd_test
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd

from . import registry
from .config import DATA_DIR, MODEL_DIR, TrainConfig
from .data_io import list_acquisitions, read_signals
from .features import dataset_features, transform
from .model import alarm_intervals, fit_bearing, persistent_alarms

log = logging.getLogger(__name__)

MIN_HEALTHY_ACQUISITIONS = 30


class TrainingError(ValueError):
    pass


def resolve_dataset(name: str, data_root: Path = DATA_DIR) -> Path:
    """Dataset folder under the data root. Absolute paths and '..' are rejected."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if not name.strip() or name.startswith(("/", "\\")) or ".." in parts or ":" in name:
        raise TrainingError(f"Dataset '{name}' must be a folder inside the data directory")
    return Path(data_root) / Path(*parts)      # symlinked datasets are allowed on purpose


def fit_from_features(feats: pd.DataFrame, config: TrainConfig) -> tuple[dict, dict]:
    """Fit one model per bearing on a features table. Returns (models, healthy_period)."""
    start = feats["timestamp"].min()
    healthy_end = start + pd.Timedelta(hours=config.healthy_hours)

    data = feats.sort_values(["bearing", "timestamp"])

    models = {}
    for b in config.bearings:
        h = data[(data["bearing"] == b) & (data["timestamp"] <= healthy_end)]
        if len(h) < MIN_HEALTHY_ACQUISITIONS:
            raise TrainingError(
                f"{b}: only {len(h)} acquisitions in the healthy period, at least "
                f"{MIN_HEALTHY_ACQUISITIONS} are needed. Increase healthy_hours.")
        try:
            models[b] = fit_bearing(transform(h), config.quantile, config.n_splits)
        except ValueError as exc:                       # constant feature or singular covariance
            raise TrainingError(f"{b}: {exc}") from exc

    period = {"start": str(start), "end": str(healthy_end)}
    return models, period


def summarise(feats: pd.DataFrame, models: dict, config: TrainConfig) -> dict:
    """Apply the fitted models to the training data: per-bearing anomaly onset and alarm intervals."""
    summary = {}
    for b, m in models.items():
        s = feats[feats["bearing"] == b].sort_values("timestamp")
        scores = m.score(transform(s))
        alarm = persistent_alarms(scores > m.threshold, config.persistence)
        ts = s["timestamp"].to_numpy()
        intervals = alarm_intervals(ts, alarm)
        summary[b] = {
            "threshold": round(m.threshold, 4),
            "healthy_acquisitions": m.n_healthy,
            "anomalous_acquisitions": int(alarm.sum()),
            "anomaly_onset": str(pd.Timestamp(ts[alarm].min())) if alarm.any() else None,
            "alarm_intervals": [{k: str(v) if k != "acquisitions" else int(v) for k, v in r.items()}
                                for r in intervals.to_dict("records")],
        }
    return summary


def train(dataset: str, config: TrainConfig | None = None, data_root: Path = DATA_DIR,
          model_dir: Path = MODEL_DIR, save: bool = True) -> registry.DetectorModel:
    """Full training pipeline. Returns the trained model (saved in model_dir if save=True)."""
    config = config or TrainConfig()
    folder = resolve_dataset(dataset, data_root)
    files = list_acquisitions(folder)
    n_samples = read_signals(files[0], config.bearings).shape[0]

    log.info("Training on %s: %d acquisitions, %d samples each", folder, len(files), n_samples)
    feats = dataset_features(folder, config.bearings,
                             progress=lambda i, n: log.info("features %d/%d", i, n))
    models, period = fit_from_features(feats, config)

    model = registry.DetectorModel(
        version=registry.new_version(model_dir),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=dataset,
        samples_per_acquisition=int(n_samples),
        config=config.to_dict(),
        healthy_period=period,
        bearings=models,
        training_summary={"acquisitions": len(files), "bearings": summarise(feats, models, config)},
    )
    if save:
        path = registry.save(model, model_dir)
        log.info("Model %s saved to %s", model.version, path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the bearing anomaly detector")
    parser.add_argument("--dataset", default="2nd_test", help="folder inside DATA_DIR")
    parser.add_argument("--healthy-hours", type=float, default=TrainConfig.healthy_hours)
    parser.add_argument("--quantile", type=float, default=TrainConfig.quantile)
    parser.add_argument("--persistence", type=int, default=TrainConfig.persistence)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = TrainConfig(healthy_hours=args.healthy_hours, quantile=args.quantile, persistence=args.persistence)
    model = train(args.dataset, cfg)
    for b, s in model.training_summary["bearings"].items():
        print(f"{b}: threshold={s['threshold']:.2f}  onset={s['anomaly_onset']}")
    print(f"Model version: {model.version}")


if __name__ == "__main__":
    main()

