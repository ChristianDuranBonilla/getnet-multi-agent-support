# SPEC — Getnet Multi-Agent Support System

Spec-Driven Development: this document is written before any code and is the single source of truth during the 60-minute core build. Anything not described here is not built.

## 0. Scope

| In scope (core, 60 min) | Out of scope (documented as next steps) |
|---|---|
| Three agents: Router, Knowledge, Support, plus Guardrail and Escalation nodes | Multi-turn conversation memory |
| Explicit LangGraph orchestration with shared state | Endpoint authentication and rate limiting |
| RAG over Getnet public pages (global site + Brazil site) with local Chroma index | Vertex AI Vector Search, reranking |
| Web search through Gemini's native `google_search` tool | Streaming responses |
| Three Support tools over mock JSON data | Real customer database |
| Input guardrail (off-topic, PII, unsafe, prompt injection) | Trained guardrail classifier |
| Human handoff when confidence is low or tools fail | Real ticketing integration |
| Structured JSON logs with `request_id` and per-agent trace | Cloud Logging dashboards and alerting (described) |
| FastAPI `POST /chat`, `GET /health`, minimal chat UI at `/` | Cloud Run deployment (described) |
| Unit tests without network, regression eval over the 10 challenge scenarios | LLM-as-judge evaluation (described) |

Bonus items (fourth agent as Escalation node, guardrails, handoff, evaluation and observability) are deliberately part of the core: each is a small node or module, and together they are the difference between a demo and a service.

## 1. Stack

- Python 3.12 (container image `python:3.12-slim`; several dependencies do not yet support 3.14)
- FastAPI, Uvicorn, Pydantic v2, pydantic-settings
- LangGraph, LangChain, `langchain-google-genai`
- Vertex AI, location `global`: `gemini-3.8-flash` for every agent with `thinking_level="low"` (the lightest level the model accepts; `minimal` is rejected), default temperature as Google recommends for Gemini 3, a 30 s request timeout with retries; `gemini-embedding-2` (3072 dims, one text per request on Vertex)
- Chroma (persistent, `data/chroma/`) for retrieval
- `httpx`, `beautifulsoup4` for ingestion; `pytest`, `ruff` for quality
- Credentials: Application Default Credentials locally; service account `getnet-agents` mounted as a Docker secret

## 2. Architecture

```
POST /chat {message, user_id}
        │
        ▼
 ┌─────────────┐  injection (regex)  ┌───────────────┐
 │  Sanitize   │────────────────────▶│  Safe reply   │
 │ mask PII    │                     └───────────────┘
 └──────┬──────┘
        │ fan-out (one parallel superstep)
   ┌────┴─────┐
   ▼          ▼
┌─────────┐ ┌─────────┐
│Guardrail│ │ Router  │   both: structured output over the sanitized message
└────┬────┘ └────┬────┘
     └────┬──────┘
          ▼
   ┌─────────────┐  guardrail blocked / off-topic → finalizer
   │  Dispatch   │
   └──┬───────┬──┘
      ▼       ▼
┌─────────┐ ┌─────────┐
│Knowledge│ │ Support │   tools: get_user_profile, get_settlement_status, get_terminal_status
│RAG + web│ │ ReAct   │
└────┬────┘ └────┬────┘
     └─────┬─────┘
           ▼
    ┌─────────────┐  low confidence, tool error, unknown user, account under review
    │ Escalation  │──▶ create_ticket, handoff=true
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  Finalizer  │  guarantees an answer, adds the PII notice
    └─────────────┘
```

Inter-agent communication is the LangGraph shared state plus conditional edges. The Guardrail classifier and the Router are independent model calls over the same sanitized message, so the graph fans out to both in one superstep and joins at Dispatch; this removes one sequential model call (about two seconds) from every request. The `trace` field uses an additive reducer so parallel nodes can append without conflicts; steps are numbered when the response is built. The README explains when an event bus would replace in-process orchestration (long-running jobs, fan-out across services, asynchronous human steps).

### 2.1 Graph state

```python
class AgentState(TypedDict):
    request_id: str
    user_id: str
    message: str
    language: str  # detected by the Guardrail, e.g. "en", "pt", "es"
    guardrail: GuardrailResult | None
    route: RouterDecision | None
    answer: str | None
    sources: list[Source]  # {"type": "rag" | "web" | "tool", "ref": str}
    tool_calls: list[ToolCall]  # {"name", "args", "ok", "summary"}
    confidence: float
    handoff: bool
    ticket_id: str | None
    trace: list[TraceStep]  # {"step", "node", "ms", "detail"}
```

### 2.2 Sanitize and Guardrail nodes

Sanitize runs deterministic checks first (no model cost): Luhn-valid card numbers and Brazilian CPF patterns are masked in place and obvious injection phrases ("ignore your instructions", "system prompt") are blocked immediately. Guardrail then makes one structured-output call on the masked message, in parallel with the Router:

```python
class GuardrailResult(BaseModel):
    allowed: bool
    category: Literal["ok", "off_topic", "pii", "unsafe", "prompt_injection"]
    language: str
    reason: str = Field(max_length=200)
```

Policy: `unsafe` and `prompt_injection` are blocked with a short safe reply; `pii` continues with the masked message and a notice to the user; `off_topic` (recipes, politics) gets a polite redirect without calling the specialist agents. Business questions are always allowed.

### 2.3 Router agent

`gemini-3.8-flash`, `temperature=0`, `with_structured_output(RouterDecision, method="json_schema")`, no tools.

```python
class RouterDecision(BaseModel):
    agent: Literal["knowledge", "support"]
    intent: Literal[
        "product_info", "general_question", "account_or_transaction", "device_issue", "other"
    ]
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=200)
```

Routing rule (lives in the system prompt): `knowledge` for products, plans, fees, Pix, receivables prepayment, crediário, Payment Link, and general world questions (weather, exchange rates); `support` for anything that depends on the specific user: settlements and deposits, the user's terminal, declined transactions, account status. Confidence below 0.5 routes to `knowledge` and sets `handoff=true`.

### 2.4 Knowledge agent

Model bound to `[{"google_search": {}}]`, no structured output (Gemini does not allow combining both). Flow: retrieve top-5 chunks from Chroma with scores; if the best distance is under the threshold (0.38, calibrated: relevant chunks score 0.13–0.32, unrelated queries 0.41+), answer from context and cite URLs; otherwise, or when intent is `general_question`, use `google_search` and say so. Answers in the user's language even though most sources are Portuguese. Never invents fees or deadlines; never touches account data.

### 2.5 Support agent

ReAct loop (LangGraph prebuilt agent or a manual tools node, at most 4 iterations) with three tools backed by a `SupportDataRepository` Protocol (JSON implementation today, database tomorrow):

```python
@tool
def get_user_profile(user_id: str) -> dict:
    """Return name, plan, terminal model, account status and whether a bank account is linked."""


@tool
def get_settlement_status(user_id: str) -> dict:
    """Return recent sales with date, amount, method (debit, credit, Pix) and expected deposit date."""


@tool
def get_terminal_status(user_id: str) -> dict:
    """Return the terminal model, connectivity state, last heartbeat and known error codes."""
```

Mock users: `cliente1988` (Get Smart, yesterday's credit sale pending deposit, terminal online), `cliente2001` (Get Clássica, terminal offline on Wi-Fi), `cliente3050` (account under review, declined transactions). An unknown `user_id` produces a polite answer and `handoff=true`.

### 2.6 Escalation node (fourth agent)

Runs after the specialist when `handoff` conditions hold: `route.confidence < 0.5`, any tool error, Knowledge found no context and web search was not applicable, or Support could not resolve (unknown user, account under review). It calls `create_ticket(user_id, summary, priority)` on the repository, appends a human-handoff sentence in the user's language and records `ticket_id`.

### 2.7 Finalizer

Assembles the response object, computes total latency, emits one JSON log line.

## 3. API

`POST /chat`

```json
{ "message": "When will the money from yesterday's sales be deposited?", "user_id": "cliente1988" }
```

```json
{
  "request_id": "7f1c…",
  "user_id": "cliente1988",
  "agent": "support",
  "intent": "account_or_transaction",
  "guardrail": "ok",
  "response": "Your credit sale of R$ 450.00 from yesterday is scheduled to be deposited on …",
  "confidence": 0.93,
  "handoff": false,
  "ticket_id": null,
  "sources": [{"type": "tool", "ref": "get_settlement_status"}],
  "trace": [{"step": 1, "node": "guardrail", "ms": 210}, {"step": 2, "node": "router", "ms": 330}],
  "latency_ms": 1780
}
```

`GET /health` returns `{"status": "ok", "index_chunks": N, "model": "gemini-3.8-flash"}`. `GET /` serves the minimal chat UI. Validation errors return 422 (empty message, over 2000 characters, missing `user_id`); unexpected errors return 500 with `request_id` and never a traceback.

## 4. RAG pipeline

Ingestion → storage → retrieval → generation.

- Seeds (depth-1 crawl, same host only, at most 40 pages, stored as cleaned `.txt` in `data/raw/` and versioned so the Docker build needs no network):
  - Global site: `https://www.getnet.net/en/`, `/en/our-solutions/in-person-payments`, `/en/our-solutions/online-payments`, `/en/our-solutions/omnichannel`, `/en/our-solutions/value-added-solutions`, `/en/our-solutions/agentic-commerce`, `/en/corporate/about-us`
  - Brazil site (Portuguese, where the products live): `https://site.getnet.com.br/`, `/todas-as-maquininhas/`, `/get-tap/`, `/pix/`, `/crediario/`, `/link-de-pagamento/`, `/ecommerce/`, `/taxas-personalizadas/`, `/duvidas/`, `/get-ajuda/`
- Cleaning: BeautifulSoup, drop `nav`, `footer`, `script`, `style`, cookie banners; keep headings as section metadata.
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)`; metadata `{url, title, section, lang}`.
- Embeddings: `gemini-embedding-2`, `task_type=RETRIEVAL_DOCUMENT` at index time and `RETRIEVAL_QUERY` at query time.
- Storage: Chroma persistent collection `getnet_kb`; `scripts/ingest.py --offline` rebuilds from `data/raw/` during the Docker build.

## 5. Repository layout

```
getnet-agents/
├── app/
│   ├── main.py              # FastAPI app, routes, request_id middleware
│   ├── config.py            # Settings from environment
│   ├── graph.py             # LangGraph construction and compile
│   ├── state.py             # AgentState and Pydantic models
│   ├── llm.py               # LLM and embeddings factories (single place to swap models)
│   ├── agents/              # guardrail.py, router.py, knowledge.py, support.py, escalation.py, finalizer.py
│   ├── tools/support_tools.py
│   ├── repositories/        # protocols.py (SupportDataRepository), json_repository.py
│   ├── rag/                 # ingest.py, retriever.py
│   ├── prompts/             # one .md per agent, loaded at startup
│   ├── observability.py     # JSON logger, node timer
│   └── static/index.html    # minimal chat UI
├── data/                    # raw/*.txt, users.json, settlements.json, terminals.json, chroma/ (ignored)
├── evals/                   # scenarios.jsonl, run_eval.py, last_run.json (ignored)
├── tests/                   # test_guardrail.py, test_router.py, test_tools.py, test_api.py
├── scripts/                 # ingest.py (CLI wrapper), evaluate.sh (one-command Docker evaluation), setup_gcp.sh
├── Dockerfile  docker-compose.yml  Makefile  pyproject.toml  .env.example  .gitignore
├── SPEC.md  CLAUDE.md  README.md
```

## 6. Design principles applied (named in the README)

- Single responsibility: one node per module; prompts live outside the code.
- Dependency inversion: agents receive the LLM and the repository through constructors or factories in `llm.py`, so tests inject fakes and run without network.
- Open/closed: agents are registered in a dictionary consumed by `graph.py`; adding a fifth agent does not modify the Router or the graph wiring beyond one entry.
- Liskov and interface segregation: `SupportDataRepository` is a small Protocol with exactly the four methods the tools need.
- DRY: one `structured_llm()` helper, one `timed_node` decorator, one prompt loader, one response builder.
- Not applied on purpose: abstract factories, class hierarchies for three agents, interfaces without a second implementation.

## 7. Observability

- Middleware assigns `request_id`, measures latency and writes one JSON line per request: `ts, request_id, user_id, agent, intent, confidence, handoff, ticket_id, latency_ms, usage`.
- Each node is wrapped by `timed_node`, which appends `{node, ms, detail}` to `state["trace"]` through an additive reducer; the API numbers the steps and returns the trace in the response and in the log.
- README: stdout JSON goes to Cloud Logging unchanged on Cloud Run; metrics to alert on: handoff rate, p95 latency, tool error rate, daily cost; traces exportable to Cloud Trace with OpenTelemetry.

## 8. Testing and evaluation

- `tests/` runs with no network: Guardrail regexes and policy, Router wiring with a fake LLM, tools over the JSON repository, API contract with `TestClient` (health, 422, response schema).
- `evals/scenarios.jsonl`: the 10 challenge scenarios with `expected_agent` and `must_contain`; `make eval` runs the real graph, prints expected vs. obtained and writes `evals/last_run.json`. This is the regression gate for prompt or model changes.
- README describes the full integration strategy: contract tests per edge, golden dataset with human review, LLM-as-judge for answer quality, prompt canaries, load test on `/chat`.

## 9. Docker

Multi-stage `Dockerfile`: builder installs dependencies from `pyproject.toml`; runtime is `python:3.12-slim` with a non-root user, `EXPOSE 8080` and an entrypoint that builds the Chroma index from `data/raw/` on first start when the mounted volume is empty, then runs uvicorn. The index cannot be built at image build time because embeddings require Vertex AI credentials, which never enter the image. `docker-compose.yml` runs the `api` service with `env_file: .env`, the service-account key as a Compose secret at `/run/secrets/sa-key` and a named volume for `data/chroma`. `docker compose up --build` followed by `curl localhost:8080/health` must work on a clean machine with the key in `.secrets/sa-key.json`.

## 10. 60-minute plan

| Minute | Block | Deliverable |
|---|---|---|
| 0–8 | Skeleton | `config.py`, `llm.py`, `main.py` with `/health`, `state.py`, Makefile targets working |
| 8–18 | Data and tools | repository Protocol, JSON repository, three tools, `test_tools.py` green |
| 18–30 | RAG | `ingest.py` (seeds, cleaning, chunks, Chroma), index built, `retriever.py` |
| 30–46 | Agents and graph | guardrail, router, knowledge, support, escalation, finalizer, `graph.py`, `/chat` wired |
| 46–53 | Docker | `Dockerfile`, compose, successful `curl` inside the container |
| 53–60 | Evals | `run_eval.py` over `scenarios.jsonl`, fix Router prompt if any scenario misroutes |

Outside the clock (allowed by the challenge): environment setup (done), prompts, mock data, README, chat UI, video.

## 11. README outline

README: overview → architecture diagram → quick start (local, Docker, tests, eval) → life of a request → RAG pipeline → agents and tools → guardrails and escalation → observability → evaluation strategy → design principles → trade-offs and next steps → note on Vertex AI `global` location.
