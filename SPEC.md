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

---

# Part II — Revision 2 (after evaluation feedback)

Revision 1 (Part I above) was built in the 60-minute core and evaluated. The evaluation
confirmed the architecture, the RAG pipeline, the regression evaluation and the engineering
judgment, and found one structural defect plus a set of scope gaps. Revision 2 fixes the defect
first and then closes the gaps, in the order of their weight. Part I is kept unchanged as the
historical record; where Part II contradicts it, Part II wins.

## 12. What the evaluation found

| Finding | Severity | Root cause in Revision 1 |
|---|---|---|
| The model chooses which merchant's data a tool reads: `user_id` is a tool argument produced by the LLM and `run_tool` executes it unchanged | Critical (security gate) | Identity was passed through the prompt instead of being bound by the runtime |
| No caller authentication; `user_id` in the body is trusted | High | Authentication was declared out of scope |
| No conversation memory; every request starts from an empty state | High | Declared out of scope |
| Knowledge and Support never collaborate; no replanning after observing a result | Medium | Router picks exactly one specialist |
| Tickets live in memory; no operator flow; no resume | Medium | Escalation was a notice, not a case |
| Evaluation checks routing, tools, sources and keywords, never the factual quality of the answer; retrieval quality not measured | Medium | Semantic evaluation declared "described" |
| No CI, no lockfile, raw `user_id` in logs, no token or cost accounting | Low | Time |

## 13. Revision 2 scope

| In scope (Revision 2) | Still out of scope (documented) |
|---|---|
| Server-bound identity: tools cannot receive a merchant id from the model | OIDC/JWT federation with a real identity provider (described) |
| API-key authentication per merchant, hashed at rest, with scopes and revocation; per-principal rate limiting | Mutual TLS, API gateway |
| Multi-turn memory: conversations owned by a merchant, persisted in SQLite, rolling summary, bounded history | Cross-device session federation |
| Persistent LangGraph checkpointer (SQLite) for durable turns and interrupt/resume | Distributed checkpointer (Firestore/Postgres) (described) |
| Specialist collaboration: Router emits a plan of up to two specialists; a specialist can request one hop at runtime; a deterministic replanner enforces the cap | A model-driven planner with arbitrary depth |
| Human-in-the-loop: persistent tickets, operator endpoints, graph `interrupt()` on escalation and `Command(resume=...)` when the operator answers, exactly-once | Operator UI, SLA routing |
| Semantic evaluation: golden answers, evidence-based LLM judge, human calibration, versioned results | Online evaluation on production traffic |
| Retrieval metrics (recall@5, MRR, nDCG@5) on a held-out query set; hybrid retrieval (dense + BM25 with RRF) compared against dense-only | Learned reranker, Vertex AI Vector Search |
| Token and cost accounting per request; pseudonymous merchant ids in logs; optional OpenTelemetry export | Dashboards and alerting (described) |
| CI (lint, tests, image build, smoke), dependency lockfile, `/ready` distinct from `/health` | Multi-region deployment |

The HTTP contract of the challenge, `POST /chat {message, user_id}`, remains valid: `user_id`
becomes an assertion that the server verifies against the authenticated principal.

## 14. Identity and authentication

### 14.1 Principal

```python
class Principal(BaseModel):
    merchant_id: str
    auth_method: Literal["api_key"]
    scopes: tuple[str, ...]  # ("chat",) for merchants; ("operator",) for operators
    key_id: str  # first 8 hex chars of the key hash, for logs and revocation
```

The principal is created in `app/auth.py` from the `X-API-Key` header and nowhere else. It is
stored in `AgentState["principal"]` and is the only identity the tools, the repository and the
escalation node ever see. `ChatRequest.user_id` is optional; when present it must equal
`principal.merchant_id`, otherwise the API answers `403 {"detail": {"error": "identity_mismatch"}}`
without running the graph. This keeps the challenge contract and makes the body field harmless.

### 14.2 API keys

- `data/api_keys.json` stores `{"<sha256 of key>": {"merchant_id": ..., "scopes": [...], "status": "active"|"revoked", "label": ...}}`. Plaintext keys are never stored.
- Demo keys for the three mock merchants plus one revoked key live in `evals/demo_keys.json`; they are mock credentials for mock merchants and are documented in the README as such. `scripts/issue_api_key.py <merchant_id>` prints a fresh key and appends its hash.
- Lookup hashes the presented key with SHA-256 and compares with `hmac.compare_digest`. Missing or unknown key: `401 {"error": "unauthorized"}`. Revoked: `403 {"error": "key_revoked"}`. Wrong scope: `403 {"error": "forbidden"}`.
- The operator credential comes from the environment: `OPERATOR_API_KEY` (hash compared the same way). It has scope `operator` and no `merchant_id`.
- Rate limit: token bucket per `key_id`, `RATE_LIMIT_PER_MINUTE` (default 30) with burst 10, in-process; exceeded requests get `429` with `Retry-After`. Documented as per-instance; a shared store is the production step.

### 14.3 Server-bound tools

Tools lose the `user_id` parameter from the schema the model sees. Each tool declares the merchant
as an injected argument:

```python
@tool
def get_settlement_status(merchant_id: Annotated[str, InjectedToolArg]) -> dict:
    """Return the merchant's recent sales: date, amount, method, status and expected deposit date."""
```

`InjectedToolArg` removes the parameter from `tool.tool_call_schema`, so the model cannot even
name it. `run_tool(name, args, principal)` in `app/agents/support.py`:

1. forwards only the keys declared in the tool's `tool_call_schema` (an allowlist; see section 25 for the change from the blocklist first specified) and, when a dropped key looks like an identity (`IDENTITY_KEYS = {"user_id", "merchant_id", "customer_id", "account_id"}` or any spelling built on those stems), logs `identity_override_attempt` with the `key_id` and the tool name (never the attempted value) and sets `AgentState["security_events"]`;
2. invokes the tool with `{**args, "merchant_id": principal.merchant_id}`;
3. is the only code path that executes tools; tests assert this by inspecting the schemas and by replaying an adversarial tool call.

The repository keeps its `user_id: str` parameters (it is the trusted layer); the principal's
`merchant_id` is the only value ever passed to them from the agents. The support prompt no longer
mentions the id at all.

### 14.4 Tests that prove the boundary (`tests/test_identity.py`)

- Every tool's `tool_call_schema` has no field named in `IDENTITY_KEYS`.
- A fake model emits `get_user_profile(user_id="cliente2001")` while the principal is `cliente1988`: the repository receives `cliente1988`, the answer contains cliente1988's data, a `security_events` entry is recorded, and the attempted value does not appear in logs.
- `POST /chat` without `X-API-Key` → 401; with the revoked demo key → 403; with a valid key and `user_id` of another merchant → 403 `identity_mismatch`; with a valid key and no `user_id` → 200 and `user_id` in the response equals the principal.
- Operator key cannot call `/chat` (403); merchant key cannot call `/tickets` (403).
- Rate limit: the 41st request in a minute from one key gets 429.

### 14.5 Guardrail failure policy

Revision 1 fails open: when the classifier call fails, `classify()` returns `allowed=True`.
Revision 2 makes the policy explicit and configurable: `GUARDRAIL_FAIL_MODE=closed|open`,
default `closed`. In `closed` mode a classifier failure answers with a templated "temporarily
unavailable, please try again" message in the detected language, sets `guardrail="unavailable"`,
runs no specialist, and records `guardrail_unavailable` in `security_events`; the deterministic
sanitizer still runs first. `open` mode keeps the Revision 1 behaviour for development and is
logged as a warning at startup. A test covers both modes.

## 15. Conversation memory

### 15.1 Model

A conversation is owned by one merchant and is a sequence of turns. A turn is one graph run.
Separating durable conversation memory from per-turn execution state keeps the graph reducers
simple (the `trace` reducer stays additive within a run) and makes ownership checks explicit.

```python
class Turn(BaseModel):
    role: Literal["merchant", "assistant", "operator"]
    content: str
    agent: str | None = None
    created_at: datetime


class ConversationRepository(Protocol):
    def create(self, merchant_id: str) -> str: ...

    # None when the conversation is missing or owned by another merchant
    def get(self, conversation_id: str, merchant_id: str) -> Conversation | None: ...

    def append(self, conversation_id: str, turn: Turn) -> None: ...

    def update_summary(self, conversation_id: str, summary: str, through_turn: int) -> None: ...
```

`SqliteConversationRepository` in `app/repositories/sqlite_repository.py`, file
`data/state/app.sqlite` (same file holds tickets, section 17), WAL mode, one connection per
call, schema created on first use. The JSON merchant data stays where it is.

### 15.2 Request flow

- `ChatRequest.conversation_id: str | None`. Absent → a new conversation is created for the principal. Present → `get(conversation_id, principal.merchant_id)`; `None` → `404 {"error": "conversation_not_found"}` (the same answer for missing and foreign conversations, so existence is not leaked).
- `ChatResponse.conversation_id` and `ChatResponse.turn` (1-based index of this assistant turn).
- `GET /conversations/{id}` (merchant scope) returns the turns and the summary; ownership enforced the same way.
- A `recall` node runs first in the graph and loads `history` (last `MEMORY_WINDOW_TURNS`, default 6 turns of each role) and `summary` into the state. A `remember` node runs last and appends the merchant turn and the assistant turn. When the conversation exceeds `MEMORY_SUMMARY_EVERY` (default 10) turns past the last summary, `remember` asks the model for a rolling summary (structured output, 120 words max, in the conversation's language) and stores it; the summary is generated after the answer is produced so it never adds latency to the user-visible path (it runs in the same request, after the answer is composed; acceptable at this scale and documented).
- Router, Knowledge and Support prompts receive `summary` and `history` as a labelled block before the current message. Guardrail classifies only the current message. The scenario "My card machine is offline" → "It's the Smart" must route the second turn to Support with the terminal context.
- `MEMORY_WINDOW_TURNS` and `MEMORY_SUMMARY_EVERY` are settings.

### 15.3 Checkpointer

The compiled graph uses `SqliteSaver` (`langgraph-checkpoint-sqlite`) on `data/state/checkpoints.sqlite`.
The thread id is `"{conversation_id}:{turn}"`: one thread per turn. The checkpointer makes a
turn durable across a restart and is what `interrupt()`/`Command(resume=...)` need in section 17.
In tests the graph is compiled with `InMemorySaver`.

## 16. Collaboration and replanning

### 16.1 Plan

`RouterDecision` gains `plan: list[AgentName]` (length 1 or 2, first element equals `agent`)
and the router prompt gains the rule: when a message needs both the merchant's data and product
knowledge, put `support` first and `knowledge` second. Example: "When will I be paid and how many
instalments does crediário allow?".

### 16.2 Runtime hop request

Support gets a fourth tool, `request_product_knowledge(question: str) -> dict`, that performs no
I/O: it records `hop_request = {"agent": "knowledge", "question": question}` in the loop and
returns `{"ok": true, "note": "the Knowledge agent will add the official procedure"}`. The model
uses it when a tool result shows that an official procedure or product rule is needed (an
offline terminal that needs the reset procedure, a fee question after a settlement lookup). It
costs no latency inside the Support loop.

### 16.3 Replanner node (deterministic)

`replan_node` runs after every specialist and decides, in this order:

1. `handoff` → `escalation`; a failed merchant-data tool or an account under review (reported by Support in `tool_calls`, decided here) → `escalation`; a Knowledge answer without evidence → `escalation`;
2. `hop_request` present and `hops < MAX_HOPS` (default 1) → the requested specialist, with `hop_question`, `partial_answers` carrying the previous specialist's answer and tool summaries;
3. remaining steps in `route.plan` and `hops < MAX_HOPS` → the next specialist;
4. otherwise → `finalizer`.

The policy is code, not a model call, so it is tested exhaustively and adds no latency. The
specialist that runs as the second hop receives `partial_answers` in its prompt and must produce one
combined answer; Knowledge's prompt gets a paragraph for this case. `hops` and `plan` are returned
in `ChatResponse` and every hop appears in the trace (`support` → `replan` → `knowledge` →
`replan` → `finalizer`).

### 16.4 Tests

- Plan of two: both specialists run in order, one `replan` between them, `hops == 1`.
- Hop request from Support: Knowledge runs with the question and the partial answer in its prompt.
- Cap: a second hop request is ignored and the graph finalises; a plan of two plus a hop request still yields exactly one hop.
- Handoff wins over any hop.

## 17. Human in the loop

### 17.1 Persistent tickets

Tickets move from the in-memory dict to the SQLite file: `tickets(ticket_id, merchant_id,
conversation_id, request_id, summary, priority, status, created_at, claimed_by, resolved_at,
operator_reply)`. `status ∈ {open, claimed, resolved}`. `create_ticket` is idempotent on
`request_id`: calling it twice for the same turn returns the same ticket (this is what makes
resume safe, 17.3).

### 17.2 Operator endpoints (scope `operator`)

- `GET /tickets?status=open` → list.
- `POST /tickets/{id}/claim` → `claimed_by = key_id`; 409 if already claimed by another operator.
- `POST /tickets/{id}/reply {"message": str, "resolution": "resolved"|"needs_merchant"}` → stores the reply, resumes the graph (17.3), marks the ticket `resolved`; a second reply returns 409 `already_resolved`.
- `GET /tickets/{id}` with merchant scope returns only the merchant's own tickets (status and reply), otherwise 404.

### 17.3 Interrupt and resume

The escalation path is split in two nodes:

- `ticket_node`: creates the ticket (idempotent), composes the handoff notice as today, stores the assistant turn through `remember` semantics, and returns the state update.
- `await_operator_node`: calls `interrupt({"ticket_id": ..., "merchant_id": ...})`. On the first run the graph suspends here; the API reads `graph.get_state(config).values`, which already holds the ticket and the notice, and answers the merchant normally with `handoff=true`.

When the operator replies, the API calls `graph.invoke(Command(resume={"message": ..., "resolution": ...}), config=thread)`. The node resumes, returns the operator's message as `operator_reply`, and the graph proceeds to `finalizer` → `remember`, which appends an `operator` turn to the conversation. The merchant sees it in `GET /conversations/{id}` and the next merchant turn carries it in `history`, so the assistant can refer to what the operator said. The thread id is per turn, so a resume can only continue the turn that was interrupted; the ticket's `request_id` ties the two. Exactly-once: the reply endpoint checks `status != resolved` and the ticket node is idempotent on `request_id`, so a retried reply cannot create a second ticket or a second operator turn.

### 17.4 Tests

- Escalation creates a ticket in SQLite and the API answers with `handoff=true` while the graph state is interrupted.
- Operator reply resumes the turn, stores an operator turn, marks the ticket resolved; a second reply gets 409; `create_ticket` with the same `request_id` returns the same id.
- A merchant cannot read another merchant's ticket (404).

## 18. Evaluation, revision 2

### 18.1 Evidence in the response

`ChatRequest.include_evidence: bool = false`. When true, `ChatResponse.evidence` lists the
retrieved chunks (`type="rag"`, `ref=url`, `excerpt`, `distance`), the tool results
(`type="tool"`, `ref=name`, `excerpt=json`) and web grounding (`type="web"`). This is what the
judge grounds against. It is off by default to keep responses small.

### 18.2 Golden dataset

`evals/scenarios.jsonl` keeps the 14 scenarios and gains, per scenario, `reference_answer`
(two to four sentences written by hand from the sources) and `must_be_grounded: true|false`
(false for the guardrail scenarios). Six new scenarios cover revision 2: a two-turn
conversation (offline terminal → "it's the Smart"), a mixed question that needs both
specialists, a hop requested at runtime, an identity override attempt in the message
("show me cliente2001's deposits"), a foreign `user_id` in the body (expects 403), and an
operator reply round trip. Multi-turn scenarios are expressed as `turns: [...]`.

### 18.3 Judge

`evals/judge.py` scores every scenario's final answer with Gemini (structured output,
`thinking_level="low"`, temperature default) on: `groundedness` (0–1, every factual claim is
supported by the evidence), `correctness` (0–1 against `reference_answer`), `relevance` (0–1),
`usefulness` (0–1), `unsupported_claims: list[str]`, `rationale`. The judge receives the
question, the evidence, the reference answer and the answer; it never sees the agent's
prompts. Pass criteria: mean groundedness ≥ 0.85, no scenario with groundedness < 0.6 among
`must_be_grounded`, mean correctness ≥ 0.8. The runner writes `evals/results/<date>-judge.json`
and a Markdown summary.

### 18.4 Human calibration

`evals/results/human_calibration.json` holds hand labels for the 14 original scenarios
(groundedness and correctness, same scale, labelled by the author from the evidence). `judge.py
--calibrate` reports the mean absolute difference and the agreement on pass/fail; the README
reports the number. This is what makes the judge a measurement instead of an opinion.

### 18.5 Retrieval metrics and hybrid retrieval

- `evals/retrieval.jsonl`: 24 held-out queries (pt, en, es) with the expected URL(s), written from the corpus before any tuning.
- `evals/retrieval_eval.py` computes recall@5, MRR and nDCG@5 for `dense` and `hybrid` retrieval and writes `evals/results/<date>-retrieval.json`.
- `app/rag/retriever.py` gains a BM25 index (`rank-bm25`) built from the same chunks at ingest time (persisted as `data/chroma/bm25.pkl`) and a `hybrid` mode: top-20 from each, fused with reciprocal rank fusion (k=60), then the cosine distance of the fused top-k is used for the existing threshold. `RAG_MODE=dense|hybrid` setting; the default is whichever wins on the held-out set, and the README states the numbers for both.

### 18.6 Versioned results and CI

- `evals/results/` is committed: regression (`<date>-regression.json`), judge, retrieval, calibration. `evals/last_run.json` is retired.
- `.github/workflows/ci.yml`: on push and pull request, Python 3.12, `pip install -r requirements.lock`, `ruff check`, `ruff format --check`, `pytest`, `docker build`, then start the container without credentials and assert `GET /health` returns 200 with `"ready": false`. Vertex-backed evaluations do not run in CI (no credentials in the repository); the workflow says so in a comment.
- `make eval` runs regression + judge + retrieval and writes the three result files.

## 19. Observability and production, revision 2

- Token accounting: every model call reads `usage_metadata` from the `AIMessage` (`input_tokens`, `output_tokens`); `timed_node` sums them per node into `TraceStep.tokens_in/tokens_out`; `ChatResponse.usage = {input_tokens, output_tokens, estimated_cost_usd}` using `LLM_PRICE_INPUT_PER_MTOK` and `LLM_PRICE_OUTPUT_PER_MTOK` settings (defaults in `.env.example` with the date they were read).
- Logs never carry a raw merchant id: `merchant_ref = sha256(LOG_SALT + merchant_id)[:12]`; the `chat` log event uses `merchant_ref` and `key_id`. `LOG_SALT` is required in production and defaults to a fixed dev value with a warning at startup.
- Optional OpenTelemetry: when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, `app/observability.py` creates one span per request and one per node with the trace fields as attributes (`opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `opentelemetry-instrumentation-fastapi` in the `otel` extra). Without the variable, nothing is imported.
- `GET /health` stays liveness (`status`, `index_chunks`, `model`, `ready`). `GET /ready` returns 200 only when the index has chunks, the SQLite file is writable and credentials are configured; otherwise 503 with the failing check. The Docker healthcheck uses `/ready`; the entrypoint skips index building when no credentials are present and logs it, so the image boots in CI.
- `requirements.lock` generated with `pip-compile` (pip-tools) from `pyproject.toml`; the Dockerfile and CI install from it. `make lock` regenerates it.

## 20. API, revision 2

`POST /chat` — header `X-API-Key` (merchant scope).

```json
{ "message": "It's the Smart.", "user_id": "cliente1988", "conversation_id": "cnv_8f3a…", "include_evidence": false }
```

```json
{
  "request_id": "7f1c…", "conversation_id": "cnv_8f3a…", "turn": 2, "user_id": "cliente1988",
  "agent": "support", "plan": ["support"], "hops": 0, "intent": "device_issue", "guardrail": "ok",
  "response": "…", "confidence": 0.91, "handoff": false, "ticket_id": null,
  "sources": [{"type": "tool", "ref": "get_terminal_status"}],
  "usage": {"input_tokens": 2310, "output_tokens": 164, "estimated_cost_usd": 0.0011},
  "trace": [{"step": 1, "node": "recall", "ms": 3}, {"step": 2, "node": "sanitize", "ms": 0}, …],
  "latency_ms": 5120
}
```

Errors: 401 `unauthorized`, 403 `identity_mismatch` | `key_revoked` | `forbidden`, 404
`conversation_not_found`, 409 `already_claimed` | `already_resolved`, 422 validation, 429 rate
limit, 500 envelope. `GET /conversations/{id}`, `GET /tickets`, `POST /tickets/{id}/claim`,
`POST /tickets/{id}/reply`, `GET /tickets/{id}`, `GET /ready` as described above. The UI gains a
key selector (demo keys), a conversation that persists across messages, and a small operator panel
behind the operator key.

## 21. Graph, revision 2

```
START → recall → sanitize → (guardrail || router) → dispatch → specialist → replan
        replan → specialist (hop) | ticket → await_operator → finalizer | finalizer
        finalizer → remember → END
```

`SPECIALISTS`, `CLASSIFIERS` and the conditional edges keep their names. `AgentState` gains:
`principal`, `conversation_id`, `turn`, `history`, `summary`, `plan`, `hops`, `hop_request`,
`partial_answers`, `security_events`, `operator_reply`, `evidence`, `usage`.

## 22. Rubric mapping (the evidence each criterion can be checked against)

| Criterion | Evidence in the repository |
|---|---|
| Business & Challenge Fit | unchanged contract plus the revision 2 scenarios |
| Agentic Reasoning & Orchestration | plan, hop request tool, deterministic replanner, hops in trace, tests |
| Context, Knowledge & RAG | recall@5/MRR/nDCG on held-out queries, dense vs hybrid, versioned results |
| Tool & Action Engineering | injected merchant id, `run_tool` boundary, identity tests |
| Security & Identity | API keys hashed, principal, body verification, rate limit, pseudonymous logs, tests |
| Memory, State & Resilience | SQLite conversations with ownership, rolling summary, SqliteSaver, multi-turn scenario |
| Human Oversight & Risk Management | persistent tickets, operator endpoints, interrupt/resume, exactly-once tests |
| Evaluation & Quality Engineering | golden answers, judge, calibration, versioned results, CI, black-box runner |
| Observability, Reliability & Cost | tokens and cost per request, pseudonymous logs, optional OTel, `/ready` |
| Production Engineering | CI with image build and smoke, lockfile, readiness, one-command evaluation |
| Engineering Judgment | this part of the spec, the README "Revision 2" section, measurements behind every default |

## 23. Revision 2 plan

Six blocks, each ending with `make test` green and one commit by the author. Model calls are
never made from tests; everything that touches Vertex AI lives in `evals/`.

| Block | Deliverable |
|---|---|
| R2-1 Identity and authentication | `app/auth.py`, `Principal`, API keys hashed, rate limit, injected tool argument, `run_tool` boundary, `tests/test_identity.py`, UI key selector |
| R2-2 Memory | `ConversationRepository`, SQLite, `recall`/`remember` nodes, prompts with history, `conversation_id` in the API, `GET /conversations/{id}`, checkpointer, tests |
| R2-3 Collaboration | `plan` in `RouterDecision`, `request_product_knowledge` tool, `replan_node`, `partial_answers` in prompts, `plan`/`hops` in the response, tests |
| R2-4 Human in the loop | tickets in SQLite, operator endpoints, `ticket_node` + `await_operator_node` with `interrupt`, resume, UI operator panel, tests |
| R2-5 Evaluation | `include_evidence`, golden answers, 6 new scenarios, `judge.py`, calibration file and flag, `retrieval.jsonl`, `retrieval_eval.py`, hybrid retrieval, `evals/results/` |
| R2-6 Observability and production | tokens and cost, pseudonymous logs, optional OTel, `/ready`, lockfile, CI, README "Revision 2" section, SPEC consistency pass |

## 24. README additions

A section "Revision 2: what changed and why" right after the overview, with one paragraph per
finding of section 12, the measured numbers (regression, judge, calibration, retrieval dense vs
hybrid, latency before and after), the demo keys, the operator flow with a `curl` sequence, and
the updated trade-offs. The architecture diagram and the "Life of a request" section are updated
to the revision 2 graph.

## 25. Deviations from this part, as built

- `scripts/issue_api_key.py` handles both `issue` and `revoke`; there is no separate revoke script.
- The rate limiter is a token bucket with `RATE_LIMIT_BURST` tokens refilled at `RATE_LIMIT_PER_MINUTE / 60` per second; the test proves the bucket semantics (burst exhausted, refill, independence per key) rather than a literal "41st request".
- `recall` loads the last `2 × MEMORY_WINDOW_TURNS` turns in total (merchant and assistant interleaved) instead of counting per role.
- BM25 is implemented in `app/rag/lexical.py` (about eighty lines) instead of adding `rank-bm25`; the lexical index is stored as `lexical.json` next to the Chroma files, not as a pickle. Chunk ids are deterministic (`<slug>-<n>`) so both indexes agree.
- The hop request tool is defined with the merchant-data tools in `app/tools/support_tools.py`; `SUPPORT_TOOLS` includes it and `MERCHANT_DATA_TOOLS` does not.
- `ChatResponse` gained `tool_calls` (names) so the black-box evaluator can check tools without parsing the trace, and `GET /demo-keys` serves the demo credentials to the UI when `EXPOSE_DEMO_KEYS` is true.
- Two lockfiles are generated (`requirements.lock` for the image, `requirements-dev.lock` for CI and `make install`) with `--strip-extras`; hashes are not pinned.
- `remember` runs before `await_operator`, so the merchant and assistant turns of an escalated turn are stored before the graph waits for the operator; the operator's reply is stored by a second pass through `remember` after the resume.
- `evals/last_run.json` is retired; every run writes a dated file under `evals/results/`.
- The demo credentials are test fixtures: `evals/demo_keys.json` is excluded from the image, the API has no endpoint that serves credentials, the UI takes a pasted key, and `scripts/evaluate.sh` hands the keys to the in-container evaluator through the environment.
- The operator credential has no working default: an empty or placeholder `OPERATOR_API_KEY` never authenticates; `scripts/evaluate.sh` generates one (and a `LOG_SALT`) into `.env` when none exists.
- Ticket idempotency is keyed on the server-generated thread id of the turn, not on the client-controllable `X-Request-ID`; thread ids carry a random suffix so concurrent or retried requests never share a checkpoint. A retried merchant request is therefore a new turn, by design.
- `resolution: needs_merchant` leaves the ticket in `waiting_merchant` instead of resolving it; a later `resolved` reply closes it.
- Messages blocked by the guardrail are persisted as a marker, never verbatim, so they cannot re-enter later prompts through the history block.
- No web source is reported without grounding metadata; web evidence carries the answer text that each grounding chunk supports.
- `/health` no longer probes the disk; the write probe belongs to `/ready` only.
- The merchant data repository reads the current date at every call unless a date is pinned (tests), so relative dates do not drift in a long-running process.
- `LLM_FALLBACK_MODEL` (default `gemini-3.5-flash`, the only other Gemini 3 model served to this project, also on `global`) answers once when the primary raises; the SDK retry count is one; every model call retries once more when it failed in under ten seconds (`llm.invoke_resilient`). The trace records the model that answered each node.
- Ingestion strips the header, help menu and footer that every `site.getnet.com.br` page repeats, which cut the index from 498 to 401 chunks and removed near-duplicate boilerplate from the retrieved context.
- When Gemini runs a search but returns no per-result attribution, the executed queries are reported as the web source (`google_search: <query>`); a world question that comes back with no grounding at all is asked once more with the search made explicit.
- The ticket node adds the ticket record to the evidence, so the handoff sentence is grounded for the judge.
- In the regression runner, `expected_agent` names the first specialist the Router chose and `expected_final_agent` the one that wrote the final answer after a hop; `--only 15,16` runs a subset.
- Hybrid retrieval was measured against dense on the 24 held-out queries and did not win (see `evals/results/`); `RAG_MODE=dense` stays the default and hybrid remains available as a setting.
- The replanner escalates a Knowledge answer that has no retrieval or web evidence behind it (`handoff_reason: answer without sources`); the rolling summary keeps its cursor when the summary call fails; an operator reply is resumed into the graph only when the thread is parked at `await_operator`, otherwise it is appended directly (`resumed: false`).
- `GET /metrics` (operator scope) exposes Prometheus-style counters; `/health` does not say whether an operator credential is configured (that is an internal fact, checked by `/ready`'s owner, not advertised).
- `run_tool` is an allowlist, not a blocklist: only the arguments present in the tool's `tool_call_schema` are forwarded. `IDENTITY_KEYS` and the identity stems now only classify a dropped argument for logging (`identity_override_attempt` versus `unexpected_tool_arguments`, which records the stray names and never a value). A test checks the declaration that is actually sent to the model (`convert_to_openai_tool`).
- The escalation policy lives in the replanner, not in the Support loop. Support only reports (`ToolCall.ok`, `ToolCall.account_status`, evidence); `replan.decide` turns a failed merchant-data lookup or an account status in `under_review`, `blocked`, `suspended` into a ticket with the reason, in order: handoff from a node (model error, exhausted loop, empty answer) → data policy → unsourced Knowledge answer → hop request → remaining plan → finalizer.
- Operator keys may also live in `data/api_keys.json` with the `operator` scope, so several operators have distinct `key_id`s and a second operator gets `409 already_claimed` on a ticket another one holds; `scripts/issue_api_key.py issue --operator` issues one. Issued keys are random (`gk_` or `op_` plus 32 hex characters); the demo fixtures were regenerated accordingly and include `operator_demo`.
- The checkpointer serializer registers the project's Pydantic models (`Principal`, the state models) in `JsonPlusSerializer`'s msgpack allowlist, so a parked turn restores them as objects after a restart; the tests use the same serializer with `InMemorySaver` and with a `SqliteSaver` file.
- Ingestion removes near-duplicate chunks (`dedupe_chunks`): the FAQ hub `/duvidas/` republishes every category page and some product pages repeat their FAQ. A chunk whose sentences are at least 90% present on a more specific page (deeper URL path; earlier page on a tie) is dropped, 401 → 347 chunks; the held-out MRR rose from 0.744 to 0.793 and nDCG@5 from 0.764 to 0.807 with recall unchanged.
- `pytest-asyncio` is not a dependency (no async tests); the dev lockfile was regenerated without it.
- `evals/retrieval.jsonl` carries 8 `off_topic` queries besides the 24 labelled ones; `retrieval_eval.py` reports the gap between the best relevant distance (max 0.281) and the best off-topic distance (min 0.392) around `RAG_DISTANCE_THRESHOLD=0.38`, and how many off-topic chunks would enter the prompt (0).
- Tickets carry a `reason` column; a conversation gets one ticket per cause (an unresolved ticket with the same reason is re-used and the turn does not park again: `ticket_reused` in the state), and a ticket resolved by an operator for a cause suppresses the automatic trigger for that cause in the same conversation (`replan.resolved_by_operator`). `recall` adds operator replies in the window to the evidence as `type: operator`, and the judge prompt treats them as evidence of what the support team decided.
- The judge does not score scenarios whose expected outcome is an HTTP status (`expected_status`, carried into the results by the runner); they are listed as `status_checks`. `judge.py --out` and `run_eval.py --out` let `scripts/evaluate.sh` keep the black-box files apart (`<date>-regression-api.json`, `<date>-judge-api.json`) from the in-process ones.
- The answer text's sources block is rendered by the finalizer from the verified `sources` (`render_sources`: page URLs, web results by attributed domain, unattributed searches by query, tool sources omitted) and the Knowledge node strips any sources block the model wrote (`strip_sources_block`), after a run showed the model citing remembered domains (xe.com, climatempo) that no grounding metadata had returned. `Source.title` carries the domain from the grounding chunk.
- The Knowledge node's forced-search retry also fires when a search ran but returned only its queries (`needs_search_retry`): without result attributions the judge cannot verify a single claim of the web part of the answer, and a second call usually comes back attributed.
- `evals/calibration_sheet.py` renders the labelling sheet (`evals/results/CALIBRATION_SHEET.md`) from the newest regression and judge results and creates an empty `human_calibration.json` when none exists, so the calibration step is reproducible after every re-run. The labels committed were written for `2026-11-20-regression.json`; `judge.py --calibrate` reports the mean absolute error, the sign of the judge's bias and the pass/fail agreement.
