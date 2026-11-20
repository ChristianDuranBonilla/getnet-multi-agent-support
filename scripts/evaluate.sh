#!/usr/bin/env bash
# One-command evaluation: build and start the service with Docker, wait until the
# knowledge index is ready, then run the regression scenarios (including the operator
# round trip), the semantic judge and the retrieval metrics against the API from inside
# the container. Only Docker is required on the host.
#
# Usage: scripts/evaluate.sh [path/to/sa-key.json]
#   The key must belong to a service account with roles/aiplatform.user on a GCP
#   project with Vertex AI enabled. Without the argument, .secrets/sa-key.json is used.
#   The operator round trip uses the demo operator key from evals/demo_keys.json (a test
#   fixture stored hashed in data/api_keys.json); the service itself gets a random
#   OPERATOR_API_KEY and LOG_SALT appended to .env when they are missing.
set -euo pipefail

cd "$(dirname "$0")/.."
KEY_SOURCE="${1:-}"
KEY_TARGET=".secrets/sa-key.json"
READY_URL="http://localhost:8080/ready"
TIMEOUT_SECONDS=300

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

# Secrets never come from .env.example: when .env lacks an operator credential or a log
# salt, random values are appended so the container runs with secrets of its own.
random_hex() { openssl rand -hex 16 2>/dev/null || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
touch .env
if ! grep -qE '^OPERATOR_API_KEY=.+' .env; then
  sed -i.bak '/^OPERATOR_API_KEY=/d' .env && rm -f .env.bak
  echo "OPERATOR_API_KEY=op_$(random_hex)" >> .env
  echo "== added a generated OPERATOR_API_KEY to .env"
fi
if ! grep -qE '^LOG_SALT=.+' .env; then
  sed -i.bak '/^LOG_SALT=/d' .env && rm -f .env.bak
  echo "LOG_SALT=$(random_hex)" >> .env
  echo "== added a generated LOG_SALT to .env"
fi
OPERATOR_API_KEY="$(grep -E '^OPERATOR_API_KEY=' .env | tail -1 | cut -d= -f2-)"
# The merchant demo keys are test fixtures of the repository, not part of the image:
# they are handed to the evaluator inside the container through the environment.
EVAL_API_KEYS="$(cat evals/demo_keys.json)"

echo "== building and starting the service"
docker compose up --build -d

echo "== waiting for the service to be ready (first start builds the index, about one minute)"
elapsed=0
until curl -fsS "$READY_URL" >/dev/null 2>&1; do
  if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
    echo "service did not become ready within ${TIMEOUT_SECONDS}s; last logs:" >&2
    docker compose logs --tail 40 >&2
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done
curl -sS http://localhost:8080/health
echo

status=0
echo "== regression scenarios against the API"
STAMP="$(date -u +%F)"
docker compose exec -T -e EVAL_API_KEYS="$EVAL_API_KEYS" api \
  python evals/run_eval.py --api http://127.0.0.1:8080 \
  --out "evals/results/${STAMP}-regression-api.json" || status=1
echo
echo "== semantic judge over the answers"
docker compose exec -T api python evals/judge.py \
  --results "evals/results/${STAMP}-regression-api.json" \
  --out "evals/results/${STAMP}-judge-api.json" || status=1
echo
echo "== retrieval metrics (dense vs hybrid)"
docker compose exec -T api python evals/retrieval_eval.py || status=1

echo
echo "Results inside the container: /app/evals/results/${STAMP}-*-api.json (copy with: docker compose cp api:/app/evals/results ./evals/)"
echo "UI:      http://localhost:8080/   (paste a merchant key from evals/demo_keys.json; operator key in .env)"
echo "OpenAPI: http://localhost:8080/docs"
echo "Stop:    docker compose down"
exit $status
