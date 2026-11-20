#!/bin/sh
set -e

CHROMA="${CHROMA_DIR:-/app/data/chroma}"
if [ ! -f "$CHROMA/chroma.sqlite3" ]; then
  if { [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; } \
     || [ -n "$GOOGLE_API_KEY" ] || [ -n "$GEMINI_API_KEY" ]; then
    echo "index not found, building it from data/raw"
    python scripts/ingest.py --offline
  else
    echo "index not found and no credentials present; starting without an index (/ready reports 503)"
  fi
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
