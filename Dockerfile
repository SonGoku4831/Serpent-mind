# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# Keep completed downloads between builds and tolerate slow connections.
RUN --mount=type=secret,id=proxy_ca \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip install --timeout 180 --retries 10 -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
