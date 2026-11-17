# Bearing anomaly detection — NASA IMS Set 2

Unsupervised anomaly detection on bearing vibration data, served as a REST API.
Each 1-second acquisition is summarised by five time-domain features per bearing; a Mahalanobis distance
from an initial healthy period is the anomaly score. The analysis behind these choices is in
`notebooks/01_anomaly_detection.ipynb`; this README only explains how to run the code.

## Repository structure
```
├── README.md
├── notebooks/01_anomaly_detection.ipynb   # Part 1: exploration, model, evaluation
├── src/                                   # library shared by the notebook and the service
│   ├── config.py        # default parameters, DATA_DIR / MODEL_DIR
│   ├── data_io.py       # reading IMS files
│   ├── features.py      # time-domain features
│   ├── model.py         # Mahalanobis detector, out-of-fold threshold, persistence rule
│   ├── training.py      # training pipeline   (also a CLI)
│   ├── prediction.py    # prediction pipeline (also a CLI)
│   └── registry.py      # model saving, loading and versioning
├── app/                 # FastAPI application (endpoints and request/response schemas)
├── examples/            # curl script and Python client
├── tests/               # pytest, on a synthetic dataset
├── data/                # dataset, not in the repository (see Data)
├── models/              # saved models; the one trained on Set 2 is included
├── Dockerfile, docker-compose.yml
├── requirements.txt     # service dependencies
├── requirements-dev.txt # + notebook and tests
└── setup.sh             # creates the local virtual environment
```

**Two dependency files**, installed at different moments:

| File | Contents | Installed by |
|---|---|---|
| `requirements.txt` | runtime dependencies of the service only | the `Dockerfile`, when the image is built (`docker compose up --build`) |
| `requirements-dev.txt` | `requirements.txt` + notebook (Jupyter, matplotlib) and tests (pytest, httpx) | `setup.sh`, in the local virtual environment `.venv` |

This keeps the Docker image small: it contains only what the service needs to run.

## Data

The dataset is not in the repository. Download the NASA IMS Bearing Dataset (e.g. from [Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)) and put the 984 acquisition files of Set 2 **directly** in `data/2nd_test/`:

```
data/
└── 2nd_test/
    ├── 2004.02.12.10.32.39
    ├── 2004.02.12.10.42.39
    ├── ...
    └── 2004.02.19.06.22.39      (984 files, one per acquisition)
```

If the downloaded archive contains a nested folder (e.g. `2nd_test/2nd_test/`), copy the inner one.

## Local setup (notebook, tests, CLI)

Requires Python ≥ 3.11.

```bash
./setup.sh                    # creates .venv and installs requirements-dev.txt
source .venv/bin/activate

jupyter lab notebooks/01_anomaly_detection.ipynb    # Part 1
pytest                                              # tests
```

Training and prediction also run without the API:

```bash
python -m src.training --dataset 2nd_test                 # saves the model in models/
python -m src.prediction data/2nd_test/2004.02.16.03.22.39 data/2nd_test/2004.02.16.03.32.39
```

## Run the service with Docker

```bash
docker compose up --build
```

The service listens on http://localhost:8000; interactive documentation at http://localhost:8000/docs.

**Where data and models live.** The image contains only the code.
- `./data` on the host is mounted read-only at `/data` in the container (`DATA_DIR`). `/train` reads `/data/<dataset>`.
- `./models` on the host is mounted at `/models` (`MODEL_DIR`). Trained models are written there, so they survive container restarts and can be inspected on the host.

## API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | service status, latest model available |
| POST | `/train` | train a new model on a dataset folder and save it |
| POST | `/predict` | score one or more acquisitions sent as JSON |
| POST | `/predict/files` | score one or more raw IMS files (multipart upload) |
| GET  | `/models` | list of saved model versions |
| GET  | `/models/{version}` | full model: parameters, healthy period, training summary |

### POST /train

Body (all fields optional; defaults are the values chosen in the analysis):

```json
{
  "dataset": "2nd_test",
  "healthy_hours": 48,
  "quantile": 0.995,
  "persistence": 3
}
```

| Field | Meaning |
|---|---|
| `dataset` | folder inside the data directory |
| `healthy_hours` | initial period assumed healthy, from the first acquisition |
| `quantile` | threshold = this quantile of the out-of-fold healthy scores |
| `persistence` | consecutive anomalous acquisitions needed to raise an alarm |

The response contains the new `model_version` and, for each bearing,
the threshold, the anomaly onset and the alarm intervals found on the training data:

```json
{
  "model_version": "20261116T173527Z",
  "dataset": "2nd_test",
  "acquisitions": 984,
  "healthy_period": {"start": "2004-02-12 10:32:39", "end": "2004-02-14 10:32:39"},
  "config": {"healthy_hours": 48.0, "quantile": 0.995, "persistence": 3, "...": "..."},
  "bearings": {
    "B1": {"threshold": 5.2467, "healthy_acquisitions": 289,
           "anomalous_acquisitions": 451, "anomaly_onset": "2004-02-16 03:22:39",
           "alarm_intervals": [{"start": "2004-02-16 03:22:39", "end": "2004-02-19 06:22:39", "acquisitions": 451}]},
    "...": {}
  }
}
```

### POST /predict

Body: 1 to 100 acquisitions. `signals` holds the raw samples of any subset of the model's bearings;
each signal must have the same length as the training files (20,480 samples = 1 s).
`timestamp` and `id` are optional.

```json
{
  "acquisitions": [
    {"id": "2004.02.16.03.22.39", "timestamp": "2004-02-16T03:22:39",
     "signals": {"B1": [-0.049, -0.042, ...], "B2": [...], "B3": [...], "B4": [...]}}
  ],
  "model_version": null
}
```

`model_version` is optional: without it the latest model is used. Response, one result per acquisition in the input order:

```json
{
  "model_version": "20261116T173527Z",
  "persistence": 3,
  "results": [{
    "id": "2004.02.16.03.22.39",
    "timestamp": "2004-02-16T03:22:39",
    "anomalous_bearings": ["B1"],
    "bearings": {
      "B1": {"score": 6.5702, "threshold": 5.2467, "status": "anomalous", "above_threshold": true, "alarm": false,
             "features": {"rms": 0.084295, "peak": 0.427701, "p2p": 0.813, "kurtosis": 3.450845, "skewness": -0.029107}},
      "B2": {"score": 2.632, "status": "normal", "...": "..."}
    }
  }]
}
```

| Field | Meaning |
|---|---|
| `score` | Mahalanobis distance from the healthy period |
| `status` | `normal` / `anomalous` (score above threshold) |
| `alarm` | `true` if the acquisition belongs to a run of at least `persistence` consecutive anomalous acquisitions **within the same request**, ordered by timestamp (input order if timestamps are missing). A single acquisition can never raise an alarm on its own: send a time series to use this field |
| `features` | the five features computed on the signal |

`status` is the per-acquisition outcome required by the exercise and needs no history. `alarm` adds the persistence rule of the analysis;
the service is stateless, so the history must come with the request.

### POST /predict/files

Same output as `/predict`, but the acquisitions are raw IMS files uploaded as multipart (`files` field, one or more).
The file name, if it is a timestamp, gives the acquisition time.

```bash
curl -X POST http://localhost:8000/predict/files \
     -F "files=@data/2nd_test/2004.02.16.03.22.39" -F "files=@data/2nd_test/2004.02.16.03.32.39"
```

### Errors

| Code | When |
|---|---|
| 404 | dataset folder or model version not found |
| 409 | no trained model yet (`POST /train` first), or a training already in progress |
| 422 | invalid input: wrong signal length, unknown bearing, NaN values, constant signal, unparseable file, invalid training parameters, healthy period too short |
| 500 | unexpected error (details in the service log) |

All errors have the form `{"detail": "..."}`.

## Examples

With the service running:

```bash
./examples/requests.sh                               # health, train, predict (files and JSON), an error, models
python examples/client.py train
python examples/client.py predict data/2nd_test/2004.02.16.03.{12,22,32,42}.39
python examples/client.py payload data/2nd_test/2004.02.18.12.02.39 -o payload.json   # JSON body for /predict
```

`API_URL` changes the target (default `http://localhost:8000`).

## Models and versioning

The repository ships with the model trained on Set 2 with the default parameters (`models/20261116T173527Z.json`, the version in `models/LATEST`): `/predict` works right after `docker compose up`, without training first. Training is still needed only to change parameters or dataset.

Each training writes `models/<version>.json`, where the version is the UTC training time (e.g. `20261115T132738Z`),
and updates `models/LATEST`. Models are plain JSON (no pickle): configuration, healthy period, per-bearing
mean, standard deviation, precision matrix (inverse covariance) and threshold, plus the training summary. Old versions are kept and can be
used with `model_version` in `/predict`.
