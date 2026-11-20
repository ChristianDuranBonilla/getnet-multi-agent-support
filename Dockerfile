FROM python:3.12-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120
# Dependencies come from the lockfile (make lock); the package itself is installed
# without dependencies so the lock is the only source of versions.
COPY requirements.lock pyproject.toml ./
RUN pip install --retries 5 --prefix=/install -r requirements.lock
COPY app ./app
RUN pip install --retries 5 --prefix=/install --no-deps .


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 \
    CHROMA_DIR=/app/data/chroma STATE_DIR=/app/data/state
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY app ./app
COPY scripts ./scripts
COPY evals ./evals
COPY data/users.json data/settlements.json data/terminals.json data/api_keys.json ./data/
COPY data/raw ./data/raw

RUN mkdir -p /app/data/chroma /app/data/state \
    && useradd --create-home --uid 1000 getnet \
    && chown -R getnet:getnet /app
USER getnet

# The index needs embeddings from Vertex AI, so it is built on first start when the
# mounted volume is empty and credentials are present (see docker-entrypoint.sh).
COPY --chown=getnet:getnet docker-entrypoint.sh ./
# The executable bit does not survive every checkout (Windows, some sync tools).
RUN chmod +x docker-entrypoint.sh
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD curl -fsS http://127.0.0.1:8080/ready || exit 1
ENTRYPOINT ["./docker-entrypoint.sh"]
