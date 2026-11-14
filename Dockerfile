# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
# pip 25.2 can resume interrupted downloads; pip 24 cannot.
# Keep this step separate so upgrading pip is cached independently.
RUN --mount=type=secret,id=proxy_ca \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    python -m pip install --timeout 180 --retries 10 --upgrade pip==25.2
COPY requirements.txt .
RUN --mount=type=secret,id=proxy_ca \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    python -m pip install --timeout 180 --retries 10 --resume-retries 20 -r requirements.txt
# DVC needs Git inside the container, including for the mounted repository.
# Install after Python dependencies to preserve their existing build cache.
RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 update \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 install -y --no-install-recommends git \
    && git config --system --add safe.directory /app \
    && git --version \
    && dvc --version \
    && rm -rf /var/lib/apt/lists/*
COPY . .
ENV PYTHONUNBUFFERED=1
