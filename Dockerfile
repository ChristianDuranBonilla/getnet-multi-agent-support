FROM python:3.12-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120
COPY pyproject.toml ./
COPY app ./app
RUN pip install --retries 5 --prefix=/install .


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 \
    CHROMA_DIR=/app/data/chroma
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app ./app
COPY scripts ./scripts
COPY evals/run_eval.py evals/scenarios.jsonl ./evals/
COPY data/users.json data/settlements.json data/terminals.json ./data/
COPY data/raw ./data/raw

RUN mkdir -p /app/data/chroma \
    && useradd --create-home --uid 1000 getnet \
    && chown -R getnet:getnet /app
USER getnet

# The index needs embeddings from Vertex AI, so it is built on first start when
# the mounted volume is empty (see docker-entrypoint.sh), never at build time.
COPY --chown=getnet:getnet docker-entrypoint.sh ./
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1
ENTRYPOINT ["./docker-entrypoint.sh"]
