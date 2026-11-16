"""Request and response formats of the API (also shown in the OpenAPI docs at /docs)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

MAX_ACQUISITIONS_PER_REQUEST = 100


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field("2nd_test", description="Folder with the acquisition files, inside the data directory")
    healthy_hours: float = Field(48.0, gt=0, description="Initial period assumed healthy, from the first acquisition")
    quantile: float = Field(0.995, gt=0.5, lt=1, description="Threshold = this quantile of out-of-fold healthy scores")
    persistence: int = Field(3, ge=1, le=144, description="Consecutive anomalous acquisitions needed for an alarm")


class AlarmInterval(BaseModel):
    start: str
    end: str
    acquisitions: int


class BearingTrainingSummary(BaseModel):
    threshold: float
    healthy_acquisitions: int
    anomalous_acquisitions: int
    anomaly_onset: str | None
    alarm_intervals: list[AlarmInterval]


class TrainResponse(BaseModel):
    model_version: str
    created_at: str
    dataset: str
    acquisitions: int
    healthy_period: dict
    config: dict
    bearings: dict[str, BearingTrainingSummary]


class AcquisitionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(None, description="Optional client identifier, echoed in the response")
    timestamp: datetime | None = Field(None, description="Acquisition time; used to order the persistence rule")
    signals: dict[str, list[float]] = Field(
        ..., description="Raw signal per bearing, e.g. {'B1': [20480 values], ...}. Any subset of the model's bearings.")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acquisitions: list[AcquisitionIn] = Field(..., min_length=1, max_length=MAX_ACQUISITIONS_PER_REQUEST)
    model_version: str | None = Field(None, description="Model to use; the latest one if omitted")


class BearingResult(BaseModel):
    score: float = Field(..., description="Mahalanobis distance from the healthy period")
    threshold: float
    status: str = Field(..., description="normal | anomalous")
    above_threshold: bool
    alarm: bool = Field(..., description="Part of >= persistence consecutive anomalous acquisitions in this request")
    features: dict[str, float | None]


class AcquisitionResult(BaseModel):
    id: str | None
    timestamp: str | None
    anomalous_bearings: list[str]
    bearings: dict[str, BearingResult]


class PredictResponse(BaseModel):
    model_version: str
    persistence: int
    results: list[AcquisitionResult]


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    latest_model: str | None
    training_in_progress: bool


class ModelsResponse(BaseModel):
    latest: str | None
    versions: list[str]


class ErrorResponse(BaseModel):
    detail: str
