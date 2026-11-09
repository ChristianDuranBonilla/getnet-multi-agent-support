#!/usr/bin/env bash
# One-command evaluation: build and start the service with Docker, wait until the
# knowledge index is ready, then run the 14 regression scenarios against the API
# from inside the container. Only Docker is required on the host.
#
# Usage: scripts/evaluate.sh [path/to/sa-key.json]
#   The key must belong to a service account with roles/aiplatform.user on a GCP
#   project with Vertex AI enabled. Without the argument, .secrets/sa-key.json is used.
set -euo pipefail

cd "$(dirname "$0")/.."
KEY_SOURCE="${1:-}"
KEY_TARGET=".secrets/sa-key.json"
HEALTH_URL="http://localhost:8080/health"
TIMEOUT_SECONDS=240

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose (v2) is required" >&2
  exit 1
fi

if [ -n "$KEY_SOURCE" ]; then
  mkdir -p .secrets
  cp "$KEY_SOURCE" "$KEY_TARGET"
fi
if [ ! -f "$KEY_TARGET" ]; then
  echo "service-account key not found at $KEY_TARGET; pass its path as the first argument" >&2
  exit 1
fi

echo "== building and starting the service"
docker compose up --build -d

echo "== waiting for the knowledge index (first start builds it, about one minute)"
elapsed=0
until curl -fsS "$HEALTH_URL" 2>/dev/null | grep -q '"index_chunks":[1-9]'; do
  if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
    echo "service did not become ready within ${TIMEOUT_SECONDS}s; last logs:" >&2
    docker compose logs --tail 40 >&2
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done
curl -sS "$HEALTH_URL"
echo

echo "== running the regression scenarios against the API"
docker compose exec -T api python evals/run_eval.py --api http://127.0.0.1:8080
status=$?

echo
echo "UI:      http://localhost:8080/"
echo "OpenAPI: http://localhost:8080/docs"
echo "Stop:    docker compose down"
exit $status
