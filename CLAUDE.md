# CLAUDE.md — working agreement for this repository

Read `SPEC.md` first. It is the source of truth: build exactly what it describes, in the order of its 60-minute plan, and nothing it leaves out of scope. When the spec and this file disagree, the spec wins; when something is missing from both, ask before inventing.

## Language and style

- Everything in English: code, docstrings, comments, commit messages, README, log messages, prompts.
- Python 3.12. Type hints on every public function and method. Pydantic v2 models for all structured data.
- Google-style docstrings on every public module, class and function (`Args:`, `Returns:`, `Raises:`). One-line docstrings are fine when the signature already says it all.
- Comments only where the code cannot explain itself (a non-obvious threshold, a workaround, a Gemini restriction). No comments that restate the line below them. No commented-out code.
- PEP 8 as enforced by `ruff format` and `ruff check` with the configuration in `pyproject.toml`: two blank lines between top-level definitions, one between methods, no trailing whitespace, exactly one newline at end of file, 100-character lines.
- Imports sorted (ruff `I` rules). Absolute imports from `app`.
- No emojis anywhere.

## Design rules

- One node per module under `app/agents/`. Prompts live in `app/prompts/*.md` and are loaded, never hard-coded.
- Dependency inversion: agents and tools receive the LLM and the `SupportDataRepository` through factories in `app/llm.py` and `app/repositories/`; nothing instantiates a Vertex client at import time.
- Keep the Gemini restriction in mind: a model with `with_structured_output` must not also bind `google_search`.
- Errors never escape as tracebacks in HTTP responses; log them with `request_id` and return a 500 envelope.
- Tests under `tests/` must pass with no network and no credentials. Anything that calls Vertex belongs in `evals/`.

## Commands

- `make install` — create the environment and install dependencies
- `make run` — start the API locally on port 8080
- `make ingest` — crawl the seed URLs and build the Chroma index (`--offline` rebuilds from `data/raw/`)
- `make test` — ruff check, ruff format --check, pytest
- `make eval` — run the 10 challenge scenarios against the real graph
- `make docker` — build and start with docker compose

Run `make test` before declaring any block of the plan done.

## Environment

Variables come from `.env` (see `.env.example`). Vertex AI location is `global`; Gemini 3.x models are not served from regional endpoints in this project. Never commit `.env`, `.secrets/`, `data/chroma/` or `evals/last_run.json`.

## Git

Small commits, one per block of the plan, imperative subject lines ("Add support tools and JSON repository"). Do not add co-author trailers.
