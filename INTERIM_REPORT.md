# Oracle — Interim Report

Status snapshot of the Oracle QA-evaluation platform.

**Branch:** `main` · **Tests:** 28 passing (`pytest`, network-free) · **Stack:** FastAPI +
React; all LLM calls go to the kiz1 proxy; document embeddings are computed locally on
the Apple **MPS** device.

## What Oracle is

A platform to compare question-answering approaches along a spectrum of increasing
sophistication: **world** (plain LLM) → **rag** (retrieval-augmented) → **a-rag**
(agentic RAG). The backend is an importable library (testable without the HTTP layer);
the frontend is a two-page React app.

## Delivered & committed (on `main`)

### Core backend & evaluation
- `oracle/` package: `config` (YAML QA configs), `llm` (proxy client, token from
  `.llmtoken`), `qa/` (systems behind a `QASystem` interface + factory), `eval/`
  (LLM-as-a-judge + runner).
- **World QA** — plain LLM answer with token/latency metrics.
- **Eval harness** — `bin/run_eval.py`; judge verdicts: correct / wrong / orthogonal.
- **Model registry** — `configs/models.yaml` + `oracle/models.py`: declares each model's
  reasoning style; exposed via `GET /api/models`.

### GUI (React, two routed pages, HSG-green theme)
- **Chat** (default `/`) — multi-turn chat with model + mode (World / RAG / Agentic RAG)
  selectors and a settings modal.
- **Comparison** (`/compare`) — ask one question across N model entries (the same model
  with and without reasoning), with charts (response time; prompt/reasoning/answer
  tokens), per-model answer cards, and per-model error surfacing.
- `make dev` runs backend (:8000) and frontend (:5173) together.

### RAG
- `oracle/retrieval/` — chunk wiki-10k `text`, embed locally (sentence-transformers, MPS),
  FAISS `IndexFlatIP` behind a `Retriever`; `bin/build_index.py` builds + caches the index
  to `data/rag-index/`.
- `oracle/qa/rag.py` — `RagQA`: embeds the question directly (no reformulation), retrieves
  top-k (default 10, configurable), renders a Jinja2 template; `answer_chat` retrieves on
  the **first** message only and extends context thereafter.
- `/api/chat` carries conversation history; the chat GUI's RAG mode works end to end.

### Agentic RAG
- `oracle/tools/` — smolagents tools: `search` (takes `k` / `min_similarity`, may be
  empty) and `query_rewrite`.
- `oracle/qa/arag.py` — `AgenticRagQA`: a smolagents `ToolCallingAgent` (→ kiz1) that first
  **reasons whether retrieval is needed**, rewrites short queries, and retries an empty
  search at most twice. The agent's reasoning trace is shown in the chat (collapsible).

## Verification

- **28 tests pass** — config, judge parser, `/api/*` contract, model registry, and the
  agentic message-sanitizer & reasoning-trace logic. All network-free.
- RAG index built for a **300-article subset** (12,905 chunks, 384-dim, cached).
- Live-verified: world / rag / a-rag answers, multi-model comparison, chat (RAG +
  Agentic RAG), and the reasoning display.

## Recent stability fixes

- **Agentic crash on "What is a conundrum"** — empty-search path produced an empty
  assistant message Mistral rejected; the sanitizer now handles list-content + enum-role
  messages.
- **Empty reasoning trace** — build it from `agent.memory.steps` (real step objects), not
  `result.steps` (plain dicts).
- **No HF calls at request time** — the embedder loads `local_files_only`; only kiz1 is
  contacted while serving.
- **Agentic "hang"** — bound the agent's LLM client (90 s timeout, 1 retry, `max_tokens`)
  instead of smolagents' ~30-min default; a stalled proxy call now fails cleanly.
- **`reasoning_effort` 400 on Mistral-Small** *(uncommitted)* — send
  `allowed_openai_params=['reasoning_effort']` so litellm passes the param through.

## Known limitations

- **RAG index is a 300-doc subset** (fast dev). Build the full 10k with
  `bin/build_index.py` for real coverage (~40 min on MPS).
- **Mistral-Medium agentic tool-calling intermittently stalls** on the proxy. Now bounded
  (~180 s, clean error) rather than a 30-min hang; retry, or use a different model for a-rag.
- **Mistral-Medium + high reasoning is slow** and can time out — a proxy/model speed limit.
- The proxy's reasoning-parameter handling is unstable and has changed repeatedly during
  development.

## Uncommitted right now

- `oracle/models.py` + `tests/test_models.py` — the `reasoning_effort` allowed-params fix
  (verified; 28 tests green).

## Suggested next steps

1. Commit the `reasoning_effort` fix.
2. Build the full 10k RAG index.
3. Author real wiki-grounded eval query/answer pairs (replace the trivial stub) and run
   `bin/run_eval.py` across world / rag / a-rag for a quality comparison.
4. Consider a more reliable tool-calling model for a-rag (or a per-model agent override),
   given Mistral-Medium's intermittent stalls.
