# CLAUDE.md — working agreement for this repository

Read `SPEC.md` completely before touching anything. Part I describes Revision 1 (the 60-minute
core, already built and evaluated); Part II describes Revision 2, which is what is being built
now. Build exactly what Part II describes, block by block (SPEC section 23), and nothing it leaves
out of scope. When the spec and this file disagree, the spec wins; when both are silent, choose
the option that keeps the model out of every authorization decision, then ask.

## Non-negotiable invariants

These are the properties the evaluation checks first. A change that breaks one is wrong even if
every test passes.

1. **Identity never comes from the model or the body.** The only source of the merchant identity
   is `Principal`, built in `app/auth.py` from the `X-API-Key` header. Tools receive the merchant
   id as an `InjectedToolArg`; `run_tool` forwards only the arguments a tool declares (an
   allowlist, so no spelling of an identity gets through) and injects `principal.merchant_id`. No prompt may contain a merchant id. No other code path
   may invoke a Support tool.
2. **Ownership is checked at the repository boundary.** A conversation, a ticket or any
   merchant-scoped record is returned only when `merchant_id` matches the principal; "missing" and
   "belongs to someone else" produce the same answer (404) so existence is never leaked.
3. **Secrets and raw identifiers stay out of logs and out of the image.** Logs carry `merchant_ref`
   (salted hash) and `key_id`, never the merchant id, the API key or an attempted override value.
   Credentials are mounted at runtime; nothing under `.secrets/`, `.env` or `data/state/` is
   committed.
4. **Deterministic policy, probabilistic language.** Routing decisions, hop caps, escalation
   triggers (in `app/agents/replan.py`, from tool status and evidence, never from the answer
   text), rate limits and identity checks are code with tests. Models write text and choose
   among allowed operations; they never choose a policy outcome.
5. **Tests prove invariants.** Every invariant above has at least one test that fails when the
   invariant is removed. Do not weaken, skip or delete an existing test or scenario; add to them.
6. **Exactly-once on resume.** Ticket creation is idempotent on the turn's server-generated
   thread id (never on a client header); an operator reply resolves a ticket at most once; a
   retried request cannot duplicate a turn.

## Language and style

- Everything in English: code, docstrings, comments, commit messages, README, log messages,
  prompts, scenario files.
- Python 3.12. Type hints on every public function and method. Pydantic v2 models for all
  structured data; `TypedDict` only for the LangGraph state.
- Google-style docstrings on every public module, class and function (`Args:`, `Returns:`,
  `Raises:`). One-line docstrings when the signature already says it all. Tool docstrings double
  as the description the model reads, so they describe what the tool returns, never how it is
  implemented and never the merchant id.
- Comments only where the code cannot explain itself (a non-obvious threshold, a workaround, a
  Gemini restriction, a security reason). No comments that restate the line below them. No
  commented-out code. No emojis.
- PEP 8 as enforced by `ruff format` and `ruff check` with the configuration in
  `pyproject.toml`: 100-character lines, two blank lines between top-level definitions, imports
  sorted, absolute imports from `app`.
- Keep modules small and single-purpose: one node per module under `app/agents/`, one repository
  implementation per module under `app/repositories/`, prompts in `app/prompts/*.md` loaded with
  `render_prompt`, never hard-coded.

## Design rules

- Dependency inversion as in Revision 1: agents and tools obtain the LLM, the retriever and the
  repositories through the factories in `app/llm.py`, `app/rag/retriever.py` and
  `app/repositories/__init__.py`. Nothing instantiates a Vertex client, a Chroma client or a
  SQLite connection at import time. Tests replace factories with fakes through `monkeypatch`.
- The Gemini restriction stands: a model with `with_structured_output` must not also bind
  `google_search`. `thinking_level="low"` everywhere; default temperature.
- Latency is a feature. Nothing new may add a model call to the path of a simple question: the
  replanner is deterministic, the hop request is a no-I/O tool, the rolling summary runs after the
  answer is composed. Measure before and after with `make eval` and record both in the README.
- State discipline: per-turn fields are reset by `initial_state` at every turn; durable memory
  lives in the conversation repository, not in the graph state. The `trace` reducer stays
  additive. Any new reducer needs a test.
- Errors never escape as tracebacks in HTTP responses: log them with `request_id` and return the
  error envelope with the codes listed in SPEC section 20.
- SQLite usage: one file `data/state/app.sqlite`, WAL mode, schema created on first use by a
  single `ensure_schema()` function, parameterised queries only, one connection per call
  (no connection shared across threads).
- Settings: every new tunable is a field in `app/config.py` with a docstring line, a default, and
  a line in `.env.example` with the same default. Keep the two in sync.

## Testing and evaluation

- `tests/` runs with no network and no credentials in under five seconds. Use `InMemorySaver` for
  the checkpointer and a temporary SQLite file (`tmp_path`) for repositories.
- Anything that calls Vertex AI lives in `evals/` and is run by `make eval`. Results are written to
  `evals/results/<YYYY-MM-DD>-<kind>.json` and committed; they are the evidence the README cites.
- Add at least one scenario to `evals/scenarios.jsonl` for every new behaviour, with a
  `reference_answer` when the answer is factual.
- Before declaring a block done: `make test` green, the block's acceptance checks in the build
  prompt satisfied, and a two-line note of anything that deviates from the spec.

## Commands

- `make install` — create the environment and install from `requirements.lock` plus dev tools
- `make lock` — regenerate `requirements.lock` from `pyproject.toml`
- `make run` — start the API locally on port 8080
- `make ingest` — crawl the seed URLs and build the Chroma and BM25 indexes (`--offline` from `data/raw/`)
- `make test` — ruff check, ruff format --check, pytest
- `make eval` — regression + judge + retrieval evaluations against Vertex AI; writes `evals/results/`
- `make eval-api` — regression scenarios against a running API (`http://localhost:8080`)
- `make docker` — build and start with docker compose

## Environment

Variables come from `.env` (see `.env.example`). Vertex AI location is `global`; Gemini 3.x models
are not served from regional endpoints in this project. New in Revision 2: `OPERATOR_API_KEY`,
`LOG_SALT`, `RATE_LIMIT_PER_MINUTE`, `MEMORY_WINDOW_TURNS`, `MEMORY_SUMMARY_EVERY`, `MAX_HOPS`,
`RAG_MODE`, `LLM_PRICE_INPUT_PER_MTOK`, `LLM_PRICE_OUTPUT_PER_MTOK`, `OTEL_EXPORTER_OTLP_ENDPOINT`.
Never commit `.env`, `.secrets/`, `data/chroma/`, `data/state/`.

## Git

The owner commits. Do not run `git add`, `git commit`, `git stash`, `git reset` or anything that
changes history or the index; leave the working tree ready for review and say which files changed
and why. Commit subjects are imperative ("Bind Support tools to the authenticated merchant"). No
co-author trailers.
