.PHONY: install lock run ingest ingest-offline test lint eval eval-api docker clean

PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.lock
	$(BIN)/pip install -e . --no-deps

lock:
	$(BIN)/pip-compile --quiet --strip-extras --output-file=requirements.lock pyproject.toml
	$(BIN)/pip-compile --quiet --strip-extras --extra=dev --extra=otel --output-file=requirements-dev.lock pyproject.toml

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
	$(BIN)/python evals/judge.py
	$(BIN)/python evals/retrieval_eval.py
	$(BIN)/python evals/calibration_sheet.py

eval-api:
	$(BIN)/python evals/run_eval.py --api http://localhost:8080
	$(BIN)/python evals/judge.py

docker:
	docker compose up --build

clean:
	rm -rf data/chroma data/state .pytest_cache .ruff_cache
