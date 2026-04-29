FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy lockfile + project metadata first (layer cache)
COPY pyproject.toml uv.lock ./

# Install deps from lockfile — no pip involved
RUN uv sync --frozen --no-dev

# Pre-download model weights
RUN uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN uv run python -c "from deepface import DeepFace; DeepFace.build_model('Facenet')"

COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN mkdir -p data/alerts data/criminal_db data/sample_videos

WORKDIR /app/backend

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]