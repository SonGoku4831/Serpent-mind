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
COPY . .
ENV PYTHONUNBUFFERED=1
