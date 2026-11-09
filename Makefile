.PHONY: install run ingest ingest-offline test lint eval eval-api docker clean

PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

run:
	$(BIN)/uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

ingest:
	$(BIN)/python scripts/ingest.py

ingest-offline:
	$(BIN)/python scripts/ingest.py --offline

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

test: lint
	$(BIN)/pytest

eval:
	$(BIN)/python evals/run_eval.py

docker:
	docker compose up --build

clean:
	rm -rf data/chroma evals/last_run.json .pytest_cache .ruff_cache

eval-api:
	$(BIN)/python evals/run_eval.py --api http://localhost:8080
