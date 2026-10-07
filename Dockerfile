# syntax=docker/dockerfile:1
# Serving image: API + dashboard + daily pipeline. Research-only libraries
# (Prophet, Darts/PyTorch) are an optional extra and are NOT installed here.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
# xgboost needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Install dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
# The cache mount keeps uv's download cache OUT of the image (saves ~0.9 GB).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serve --no-install-project
COPY configs ./configs
COPY src ./src
COPY dashboard ./dashboard
COPY reports ./reports
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serve

# data/ and artifacts/ are mounted as volumes (see docker-compose.yml)
EXPOSE 8000 8501
CMD ["uvicorn", "demand_forecast.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
