import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from src.config import BEARINGS

from .conftest import N_SAMPLES


@pytest.fixture
def client(data_root, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", data_root)
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path / "models")
    main._load_version.cache_clear()
    return TestClient(main.app)


@pytest.fixture
def trained(client):
    r = client.post("/train", json={"dataset": "synthetic", "healthy_hours": 8})
    assert r.status_code == 200, r.text
    return r.json()


def signals(scale=1.0, n=N_SAMPLES):
    rng = np.random.default_rng(3)
    return {b: (rng.normal(0, 0.1 * scale, n)).round(3).tolist() for b in BEARINGS}


def test_health_and_untrained_model(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_available"] is False
    r = client.post("/predict", json={"acquisitions": [{"signals": signals()}]})
    assert r.status_code == 409 and "POST /train" in r.json()["detail"]


def test_train_errors(client):
    assert client.post("/train", json={"dataset": "missing"}).status_code == 404
    assert client.post("/train", json={"dataset": "../etc"}).status_code == 422
    assert client.post("/train", json={"dataset": "synthetic", "quantile": 2}).status_code == 422
    r = client.post("/train", json={"dataset": "synthetic", "healthy_hours": 1})
    assert r.status_code == 422 and "healthy period" in r.json()["detail"]


def test_train_and_predict_json(client, trained):
    assert trained["bearings"]["B1"]["anomaly_onset"] is not None
    assert trained["bearings"]["B2"]["anomaly_onset"] is None
    body = {"acquisitions": [
        {"id": "ok", "timestamp": "2004-02-20T10:00:00", "signals": signals()},
        {"id": "loud", "timestamp": "2004-02-20T10:10:00", "signals": {"B1": signals(scale=4)["B1"]}},
    ]}
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    out = {x["id"]: x for x in r.json()["results"]}
    assert out["ok"]["anomalous_bearings"] == []
    assert out["loud"]["anomalous_bearings"] == ["B1"]
    assert out["loud"]["bearings"]["B1"]["alarm"] is False      # one acquisition < persistence 3
    assert r.json()["model_version"] == trained["model_version"]


def test_predict_validation(client, trained):
    bad_len = {"acquisitions": [{"signals": {"B1": [0.1] * 100}}]}
    r = client.post("/predict", json=bad_len)
    assert r.status_code == 422 and f"expected {N_SAMPLES} samples" in r.json()["detail"]
    r = client.post("/predict", json={"acquisitions": [{"signals": {"B9": signals()["B1"]}}]})
    assert r.status_code == 422 and "unknown bearings" in r.json()["detail"]
    assert client.post("/predict", json={"acquisitions": []}).status_code == 422
    r = client.post("/predict", json={"acquisitions": [{"signals": {"B1": [0.1] * N_SAMPLES}}]})
    assert r.status_code == 422 and "constant signal" in r.json()["detail"]
    r = client.post("/predict", json={"acquisitions": [{"signals": signals()}], "model_version": "20000101T000000Z"})
    assert r.status_code == 404


def test_predict_files(client, trained, data_root):
    files = sorted((data_root / "synthetic").iterdir())[-6:-2]          # 4 faulty acquisitions
    upload = [("files", (f.name, f.read_bytes(), "text/plain")) for f in files]
    r = client.post("/predict/files", files=upload)
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert all(x["bearings"]["B1"]["alarm"] for x in res)
    r = client.post("/predict/files", files=[("files", ("x", b"1\t2\n3\t4\n", "text/plain"))])
    assert r.status_code == 422


def test_models_endpoints(client, trained):
    r = client.get("/models").json()
    assert r["latest"] == trained["model_version"] and r["versions"] == [trained["model_version"]]
    info = client.get(f"/models/{trained['model_version']}").json()
    assert set(info["bearings"]) == set(BEARINGS)
    assert client.get("/models/nope").status_code == 404
