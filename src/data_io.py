"""Reading IMS acquisition files.

One file = one acquisition: tab-separated ASCII, no header, one column per bearing.
The file name is the acquisition timestamp, e.g. 2004.02.12.10.32.39.
"""

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTAMP_FORMAT = "%Y.%m.%d.%H.%M.%S"


class DataError(ValueError):
    """Raised when a dataset folder or an acquisition file is not valid."""


def parse_timestamp(name: str) -> datetime:
    try:
        return datetime.strptime(name, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise DataError(f"File name '{name}' is not a timestamp in the format YYYY.MM.DD.hh.mm.ss") from exc


def list_acquisitions(folder: Path) -> list[Path]:
    """Acquisition files of a dataset folder, in chronological order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise DataError(f"Dataset folder not found: {folder}")
    files = []
    for f in folder.iterdir():
        if f.is_file() and not f.name.startswith("."):
            parse_timestamp(f.name)          # fails on unexpected files
            files.append(f)
    if not files:
        raise DataError(f"No acquisition files in {folder}")
    return sorted(files, key=lambda f: parse_timestamp(f.name))


def read_signals(source, bearings: list[str]) -> np.ndarray:
    """Read one acquisition (path, bytes or text) as an array of shape (n_samples, n_bearings)."""
    if isinstance(source, bytes):
        source = io.StringIO(source.decode("ascii", errors="strict"))
    elif isinstance(source, str) and "\t" in source:
        source = io.StringIO(source)
    try:
        df = pd.read_csv(source, sep="\t", header=None, dtype=float)
    except Exception as exc:
        raise DataError(f"Cannot parse acquisition: {exc}") from exc
    if df.shape[1] != len(bearings):
        raise DataError(f"Expected {len(bearings)} columns ({', '.join(bearings)}), found {df.shape[1]}")
    values = df.to_numpy()
    if not np.isfinite(values).all():
        raise DataError("Acquisition contains missing or non-numeric values")
    return values
