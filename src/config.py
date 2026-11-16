"""Default parameters and runtime paths.

Paths come from environment variables so the same code runs locally and in Docker:
- DATA_DIR:  root folder that contains the datasets (e.g. data/2nd_test). Mounted read-only in Docker.
- MODEL_DIR: folder where trained models are saved. Mounted as a volume in Docker.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data")).resolve()
MODEL_DIR = Path(os.getenv("MODEL_DIR", REPO_ROOT / "models")).resolve()

BEARINGS = ["B1", "B2", "B3", "B4"]
SAMPLING_RATE_HZ = 20480          # verified in the notebook (section 8): the README says 20 kHz

# crest factor left out: after the log it is exactly log(peak) - log(rms), which makes the covariance singular
FEATURES = ["rms", "peak", "p2p", "kurtosis", "skewness"]
LOG_FEATURES = ["rms", "peak", "p2p", "kurtosis"]   # positive, long-tailed


@dataclass
class TrainConfig:
    """Hyper-parameters of the detector. Defaults are the values chosen in the notebook."""

    healthy_hours: float = 48.0      # initial period assumed healthy, from the first acquisition
    quantile: float = 0.995          # threshold = this quantile of out-of-fold healthy scores
    persistence: int = 3             # consecutive acquisitions above threshold to raise an alarm
    n_splits: int = 5                # contiguous blocks for the out-of-fold threshold
    bearings: list[str] = field(default_factory=lambda: list(BEARINGS))

    def to_dict(self) -> dict:
        return asdict(self)
