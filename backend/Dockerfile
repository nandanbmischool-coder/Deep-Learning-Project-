FROM python:3.11-slim
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app/backend
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# CPU-only PyTorch (change the index URL for a CUDA build)
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements-postgres.txt ./
RUN pip install -r requirements.txt -r requirements-postgres.txt
COPY backend/ .
ENV DATABASE_URL=sqlite:////app/data/app.db MEDIA_DIR=/app/data/media ARTIFACTS_DIR=/app/data/artifacts DATASET_DIR=/app/data/raw/chest_xray
RUN useradd -m app && mkdir -p /app/data && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
