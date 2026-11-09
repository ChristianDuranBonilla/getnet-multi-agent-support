#!/bin/sh
set -e

if [ ! -f "${CHROMA_DIR:-/app/data/chroma}/chroma.sqlite3" ]; then
  echo "index not found, building it from data/raw"
  python scripts/ingest.py --offline
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
