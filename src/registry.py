"""Saving and loading trained models.

Each model is a JSON file in MODEL_DIR named after its version (UTC timestamp of training).
JSON instead of pickle: human-readable, diff-able and safe to load. A file named LATEST holds
the version used by default.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import FEATURES, LOG_FEATURES
from .model import BearingModel

_VERSION_RE = re.compile(r"^\d{8}T\d{6}(_\d+)?Z$")


class ModelNotFoundError(LookupError):
    pass


@dataclass
class DetectorModel:
    version: str
    created_at: str
    dataset: str
    samples_per_acquisition: int
    config: dict
    healthy_period: dict
    bearings: dict[str, BearingModel]
    training_summary: dict = field(default_factory=dict)
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    log_features: list[str] = field(default_factory=lambda: list(LOG_FEATURES))
    code_version: str = __version__

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bearings"] = {b: m.to_dict() for b, m in self.bearings.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorModel":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        keep = BearingModel.__dataclass_fields__
        d["bearings"] = {b: BearingModel(**{k: v for k, v in m.items() if k in keep})
                         for b, m in d["bearings"].items()}
        return cls(**d)


def new_version(model_dir: Path) -> str:
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    version, i = f"{base}Z", 1
    while (Path(model_dir) / f"{version}.json").exists():     # two trainings in the same second
        version, i = f"{base}_{i}Z", i + 1
    return version


def save(model: DetectorModel, model_dir: Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{model.version}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model.to_dict(), indent=2, default=str))
    tmp.replace(path)                                   # atomic: no half-written models
    (model_dir / "LATEST").write_text(model.version)
    return path


def list_versions(model_dir: Path) -> list[str]:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return []
    return sorted(p.stem for p in model_dir.glob("*.json") if _VERSION_RE.match(p.stem))


def latest_version(model_dir: Path) -> str | None:
    marker = Path(model_dir) / "LATEST"
    if marker.is_file():
        v = marker.read_text().strip()
        if v in list_versions(model_dir):
            return v
    versions = list_versions(model_dir)
    return versions[-1] if versions else None


def load(model_dir: Path, version: str | None = None) -> DetectorModel:
    if version is None:
        version = latest_version(model_dir)
        if version is None:
            raise ModelNotFoundError("No trained model available. Call POST /train first.")
    if not _VERSION_RE.match(version):
        raise ModelNotFoundError(f"Invalid model version '{version}'")
    path = Path(model_dir) / f"{version}.json"
    if not path.is_file():
        raise ModelNotFoundError(f"Model version '{version}' not found")
    return DetectorModel.from_dict(json.loads(path.read_text()))
