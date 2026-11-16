import numpy as np
import pandas as pd
import pytest

from src import registry
from src.config import TrainConfig
from src.features import time_features
from src.model import alarm_intervals, persistent_alarms
from src.prediction import ANOMALOUS, NORMAL, Acquisition, PredictionError, acquisition_from_file, predict
from src.training import TrainingError, resolve_dataset, train

from .conftest import FAULT_FROM, N_FILES

CFG = TrainConfig(healthy_hours=8)


def test_time_features_gaussian():
    x = np.random.default_rng(1).normal(5.0, 2.0, 200_000)       # offset must not matter
    f = time_features(x)
    assert f["rms"] == pytest.approx(2.0, rel=0.01)
    assert f["kurtosis"] == pytest.approx(3.0, abs=0.05)
    assert f["skewness"] == pytest.approx(0.0, abs=0.02)
    assert set(f) == {"rms", "peak", "p2p", "kurtosis", "skewness"}


def test_persistent_alarms():
    above = [1, 0, 1, 1, 0, 1, 1, 1, 1, 0]
    assert persistent_alarms(above, 3).astype(int).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert persistent_alarms(above, 1).astype(int).tolist() == above
    ts = pd.date_range("2004-01-01", periods=len(above), freq="10min")
    iv = alarm_intervals(ts, persistent_alarms(above, 2))
    assert iv["acquisitions"].tolist() == [2, 4]


def test_train_finds_the_degrading_bearing(data_root, tmp_path):
    model = train("synthetic", CFG, data_root=data_root, model_dir=tmp_path)
    summary = model.training_summary["bearings"]
    onset = pd.Timestamp(summary["B1"]["anomaly_onset"])
    first_faulty = pd.Timestamp(model.healthy_period["start"]) + pd.Timedelta(minutes=10 * FAULT_FROM)
    assert first_faulty <= onset <= first_faulty + pd.Timedelta(minutes=30)
    for b in ["B2", "B3", "B4"]:          # the two silent files at the end are fewer than the persistence (3)
        assert summary[b]["anomaly_onset"] is None

    loaded = registry.load(tmp_path)                           # saved as the latest version
    assert loaded.version == model.version
    assert loaded.bearings["B1"].threshold == pytest.approx(model.bearings["B1"].threshold)


def test_predict_outcomes(data_root, tmp_path):
    model = train("synthetic", CFG, data_root=data_root, model_dir=tmp_path)
    files = sorted((data_root / "synthetic").iterdir())
    acqs = [acquisition_from_file(f, list(model.bearings)) for f in files[N_FILES - 8:]]
    res = predict(model, acqs[::-1])                           # order of the input is kept...
    assert res[0]["id"] == files[-1].name
    res = {r["id"]: r for r in res}
    last, faulty = res[files[-1].name], res[files[-3].name]
    assert last["bearings"]["B1"]["status"] == ANOMALOUS          # an (almost) silent sensor is far from normal
    assert faulty["bearings"]["B1"]["status"] == ANOMALOUS
    assert faulty["bearings"]["B1"]["alarm"] is True           # ...but persistence uses time order
    assert faulty["bearings"]["B2"]["status"] == NORMAL


def test_bad_inputs(data_root, tmp_path):
    with pytest.raises(TrainingError):
        resolve_dataset("../outside", data_root)
    with pytest.raises(TrainingError):
        train("synthetic", TrainConfig(healthy_hours=2), data_root=data_root, model_dir=tmp_path)
    with pytest.raises(registry.ModelNotFoundError):
        registry.load(tmp_path / "empty")
    model = train("synthetic", CFG, data_root=data_root, model_dir=tmp_path)
    flat = Acquisition(signals={"B1": np.full(model.samples_per_acquisition, 0.1)})
    with pytest.raises(PredictionError, match="constant signal"):
        predict(model, [flat])
