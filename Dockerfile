FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    MODEL_DIR=/models

WORKDIR /service

# dependencies first: this layer is rebuilt only when requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY app/ app/
# make the code readable by the non-root user whatever the permissions on the host
RUN chmod -R a+rX /service

# the dataset is NOT in the image: it is mounted read-only at /data.
# trained models are written to /models, mounted as a volume so they survive the container.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data /models \
    && chown appuser /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
