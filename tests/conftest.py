"""Synthetic IMS-like dataset: Gaussian vibration, B1 degrades near the end, almost no signal in the last two files."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

N_FILES = 120          # 20 hours, one acquisition every 10 minutes
N_SAMPLES = 2048
FAULT_FROM = 90        # B1 amplitude grows from this file on
START = datetime(2004, 2, 12, 10, 32, 39)


def make_signals(i: int, rng) -> np.ndarray:
    x = rng.normal(0.0, 0.1, size=(N_SAMPLES, 4)) - 0.01
    if i >= FAULT_FROM:
        x[:, 0] *= 1.5 + 0.05 * (i - FAULT_FROM)
    if i >= N_FILES - 2:
        x *= 0.01                                  # rig almost silent, like the real last two files
    return x


def write_dataset(folder: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(N_FILES):
        p = folder / (START + timedelta(minutes=10 * i)).strftime("%Y.%m.%d.%H.%M.%S")
        np.savetxt(p, make_signals(i, rng), fmt="%.3f", delimiter="\t")
        paths.append(p)
    return paths


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    write_dataset(root / "synthetic")
    return root
