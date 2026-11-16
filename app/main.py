"""FastAPI service: train and query the bearing anomaly detector.

The endpoints only translate HTTP <-> Python: all the logic lives in src/training.py and src/prediction.py.
"""

import logging
import threading
import time
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from src import __version__, registry
from src.config import DATA_DIR, MODEL_DIR, TrainConfig
from src.data_io import DataError, parse_timestamp, read_signals
from src.prediction import Acquisition, PredictionError, predict
from src.training import TrainingError, resolve_dataset, train

from .schemas import (
    MAX_ACQUISITIONS_PER_REQUEST,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = FastAPI(
    title="Bearing anomaly detection",
    version=__version__,
    description="Mahalanobis detector on time-domain vibration features (NASA IMS bearing dataset).",
)

_training_lock = threading.Lock()
ERRORS = {409: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


# --- helpers ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_version(version: str) -> registry.DetectorModel:
    return registry.load(MODEL_DIR, version)       # versions are immutable: safe to cache


def get_model(version: str | None) -> registry.DetectorModel:
    if version is None:
        version = registry.latest_version(MODEL_DIR)
        if version is None:
            raise HTTPException(409, "No trained model available. Call POST /train first.")
    try:
        return _load_version(version)
    except registry.ModelNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def run_prediction(model, acquisitions) -> PredictResponse:
    try:
        results = predict(model, acquisitions)
    except PredictionError as exc:
        raise HTTPException(422, str(exc)) from exc
    return PredictResponse(model_version=model.version, persistence=model.config["persistence"], results=results)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code,
             1000 * (time.perf_counter() - t0))
    return response


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    log.exception("Unexpected error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error, see the service logs"})


# --- endpoints --------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    latest = registry.latest_version(MODEL_DIR)
    return HealthResponse(status="ok", model_available=latest is not None, latest_model=latest,
                          training_in_progress=_training_lock.locked())


@app.post("/train", response_model=TrainResponse, responses=ERRORS, tags=["model"])
def train_endpoint(req: TrainRequest | None = None):
    """Train a new model on a dataset folder and save it as the latest version (takes ~15-30 s on Set 2).

    The body is optional: without it the defaults chosen in the analysis are used.
    """
    req = req or TrainRequest()
    folder = resolve_or_raise(req.dataset)
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(409, "A training is already in progress")
    try:
        cfg = TrainConfig(healthy_hours=req.healthy_hours, quantile=req.quantile, persistence=req.persistence)
        log.info("Training on %s with %s", folder, cfg.to_dict())
        model = train(req.dataset, cfg, data_root=DATA_DIR, model_dir=MODEL_DIR)
    except (TrainingError, DataError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        _training_lock.release()
    return TrainResponse(model_version=model.version, created_at=model.created_at, dataset=model.dataset,
                         acquisitions=model.training_summary["acquisitions"],
                         healthy_period=model.healthy_period, config=model.config,
                         bearings=model.training_summary["bearings"])


def resolve_or_raise(dataset: str):
    try:
        folder = resolve_dataset(dataset, DATA_DIR)
    except TrainingError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not folder.is_dir():
        raise HTTPException(404, f"Dataset '{dataset}' not found in the data directory")
    return folder


@app.post("/predict", response_model=PredictResponse, responses=ERRORS, tags=["model"])
def predict_endpoint(req: PredictRequest):
    """Score one or more acquisitions sent as JSON arrays of raw samples."""
    model = get_model(req.model_version)
    acquisitions = [Acquisition(signals=a.signals, timestamp=a.timestamp, id=a.id) for a in req.acquisitions]
    return run_prediction(model, acquisitions)


@app.post("/predict/files", response_model=PredictResponse, responses=ERRORS, tags=["model"])
async def predict_files_endpoint(files: list[UploadFile] = File(..., description="Raw IMS files"),
                                 model_version: str | None = Query(None)):
    """Score one or more raw IMS files (same format as the dataset; the file name is the timestamp)."""
    if len(files) > MAX_ACQUISITIONS_PER_REQUEST:
        raise HTTPException(422, f"At most {MAX_ACQUISITIONS_PER_REQUEST} files per request")
    model = get_model(model_version)
    bearings = list(model.bearings)
    acquisitions = []
    for f in files:
        try:
            signals = read_signals(await f.read(), bearings)
        except (DataError, UnicodeDecodeError) as exc:
            raise HTTPException(422, f"{f.filename}: {exc}") from exc
        try:
            ts = parse_timestamp(f.filename or "")
        except DataError:
            ts = None                                   # name is not a timestamp: keep upload order
        acquisitions.append(Acquisition(signals={b: signals[:, j] for j, b in enumerate(bearings)},
                                        timestamp=ts, id=f.filename))
    return run_prediction(model, acquisitions)


@app.get("/models", response_model=ModelsResponse, tags=["model"])
def list_models():
    return ModelsResponse(latest=registry.latest_version(MODEL_DIR), versions=registry.list_versions(MODEL_DIR))


@app.get("/models/{version}", responses=ERRORS, tags=["model"])
def model_info(version: str):
    """Full model: configuration, healthy period, per-bearing parameters and training summary."""
    return get_model(version).to_dict()
