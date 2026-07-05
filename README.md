# Oracle

**A platform for evaluating question-answering approaches — from plain LLM world
knowledge, through retrieval-augmented generation, up to a fully agentic RAG system
with a local tool router, web/YouTube tools, multi-hop planning, and answer
verification.**

Oracle lets you ask the same question across different QA "modes", see exactly how
each one arrived at its answer (tools used, tokens spent, time taken), compare them
side by side, and even run a **quiz** where a host model grades a player mode against
a dataset.

- **Backend** — a Python library (`oracle/`) that runs QA + evaluation without the
  HTTP layer, plus a thin **FastAPI** app.
- **Frontend** — a **React** SPA with three pages: **Chat**, **Comparison**, **Quiz**.
- **LLMs** — answering/judging go to an OpenAI-compatible proxy; the small tool
  **router runs locally**, and document **embeddings are computed locally**.

---

## Table of contents

- [The QA modes](#the-qa-modes)
- [Agentic RAG in depth](#agentic-rag-in-depth)
  - [The local tool router](#1-the-local-tool-router)
  - [The toolbox](#2-the-toolbox)
  - [RAG-first grounding](#3-rag-first-grounding)
  - [Online tools & offline resilience](#4-online-tools--offline-resilience)
  - [Multi-hop planning](#5-multi-hop-planning)
  - [Recursive sub-agents](#6-recursive-sub-agents)
  - [Verification / backtracking](#7-verification--backtracking)
  - [Termination & loop avoidance](#8-termination--loop-avoidance)
  - [Traceability & token accounting](#9-traceability--token-accounting)
- [The GUI](#the-gui)
  - [Quiz page](#quiz-page)
- [Setup](#setup)
- [Running](#running)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project layout](#project-layout)

---

## The QA modes

Every question can be answered by one of three interchangeable **modes**:

| Mode | Type | What it does |
|------|------|--------------|
| **World** | `world` | Answers purely from the model's own world knowledge — no retrieval. |
| **RAG** | `rag` | Retrieves passages from a local wiki index and answers grounded in them. |
| **Agentic RAG** | `a-rag` | An agent that **reasons about which tools to use**, gathers evidence, and synthesizes an answer — the flagship mode described below. |

RAG and Agentic RAG retrieve over the [`NeelNanda/wiki-10k`](https://huggingface.co/datasets/NeelNanda/wiki-10k)
corpus, chunked and embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`)
into an in-memory **FAISS** index (build it once with `bin/build_index.py`).

---

## Agentic RAG in depth

Agentic RAG answers bar/pub-quiz-style questions by orchestrating a set of tools. It
is built around a clear division of labour: a **small, local model routes tools**
cheaply, while a **large proxy model synthesizes** the final answer.

```
question
   │
   ├─▶ RAG-first: always search the local index first
   │
   ├─▶ local router (small LLM) decides the next tool …           ┐
   │      search · wiki_lookup · google_search · youtube ·        │  bounded loop,
   │      calculator · date_tool · unit_convert · list_pick …     │  always terminates
   │   … until it has enough                                      ┘
   │
   ├─▶ (optional) multi-hop: split into sub-questions, answer each,
   │             recurse for hard chains, thread facts forward
   │
   ├─▶ synthesis (big proxy model) composes the final answer
   │
   └─▶ (optional) verification: re-check vs the evidence, backtrack if wrong
```

Everything a run does is captured in a **structured trace** with full **token
accounting**.

### 1. The local tool router

A small Hugging Face instruct model (default **`Qwen/Qwen2.5-1.5B-Instruct`**) is
loaded in-process via `transformers` and prompted to emit a JSON decision
(`{"tool": …, "arguments": …}` or `{"finished": true}`). It runs **locally** — no
proxy call for routing — so tool selection stays cheap and private. The large proxy
model is reserved for the final synthesis.

- Configurable model / device (auto-picks **MPS** on Apple Silicon, else CUDA/CPU)
  with a memory-ceiling guardrail; documented fallbacks include `SmolLM2-1.7B` and
  `Llama-3.2-1B`, and a `Qwen2.5-3B` step-up for stronger routing.
- Weights are loaded once and shared across the router and the planner.

### 2. The toolbox

Each tool is a small, single-purpose function exposed to the router. Local tools are
offline and cost **0 tokens**; online tools are keyless and open-source.

| Tool | Purpose |
|------|---------|
| `search` | Semantic search over the local wiki index (FAISS cosine). |
| `wiki_lookup` | Fetch a specific article by title. |
| `query_rewrite` | Expand a terse query so its embedding is more characteristic. |
| `calculator` | Safe arithmetic **and** a whitelist of math functions (`sqrt`, `log`, trig, `factorial`, `comb`/`perm`, `gcd`/`lcm`, constants `pi`/`e`, …). |
| `date_tool` | Weekdays, date differences, date arithmetic. |
| `unit_convert` | Length / mass / temperature conversions. |
| `list_pick` | Choose one candidate from a set by a criterion (largest, longest, …). |
| `google_search` 🌐 | Web search via **DuckDuckGo** (`ddgs`), returns the **10 best** results. |
| `youtube` 🌐 | Finds videos (`yt-dlp`) and extracts **transcripts** (`youtube-transcript-api`) — for questions whose answer lives in a video. |

### 3. RAG-first grounding

Before the router picks anything, the agent **always runs a local `search`** on the
question, so answers are grounded in the corpus and cheaper/offline paths are tried
first. The router only escalates to other tools when the local result is
insufficient.

### 4. Online tools & offline resilience

`google_search` and `youtube` are **on by default** and use cost-free, keyless
libraries (DuckDuckGo, yt-dlp, youtube-transcript-api). They can be turned off for a
deliberately offline run.

- **Video questions** (e.g. *"In Mark Rober's squirrel-maze video, how many
  obstacles…"*) **force a YouTube lookup up front** so the transcript is always in
  the evidence.
- **No internet ≠ crash**: on a connectivity failure the tools return a clear
  *"No internet connection — … unavailable"* observation, the run still terminates,
  and the answer tells the user the lookup couldn't be performed.

### 5. Multi-hop planning

Some questions need several dependent steps (*"In which country was the director of
the highest-grossing 1997 film born?"*). With **multi-hop** on (default), a planner
decomposes the question into ordered sub-questions; each is answered by the bounded
loop, and earlier answers are **threaded forward as known facts**. A final synthesis
composes them.

- **Choose the planner per question** — the fast **small local** model, or the
  **big answer model** (better on hard questions).
- Toggle multi-hop **per question** right in the chat.

### 6. Recursive sub-agents

A sub-question can itself be planned and decomposed, down to a configurable
**`max_depth`** (`1` = flat). Nested steps are tagged with their recursion depth and
**indented in the trace**; the whole tree stays bounded and always terminates.

### 7. Verification / backtracking

After composing an answer, an optional **bounded verification hop** (on by default)
re-checks it against the gathered facts with the big model and, on a contradiction,
returns a **corrected answer** (a one-shot backtrack). It runs at most once.

### 8. Termination & loop avoidance

The agent is guaranteed to reach a conclusion:

- **Step budget** (`max_steps`) — on exhaustion it synthesizes from what it has.
- **Wall-clock timeout** — forces the final answer rather than hanging.
- **Loop / duplicate guards** — an exact **or reworded-equivalent** re-query of a
  tool that already succeeded is treated as *"already ran — finishing"*, and a
  **shared result cache** means an identical `(tool, args)` runs at most once per
  question.
- Every run ends in exactly one synthesis; the trace records **why** it stopped
  (`done` · `step-budget` · `timeout` · `loop-guard`).

### 9. Traceability & token accounting

Every answer carries a **structured, per-step trace** rendered in the GUI:

- Which **tool** was called, with what **arguments**, and its **result**.
- Which **model** produced each step (router vs synthesis vs planner).
- **Tokens per step and per question**, split into **input / output / reasoning**,
  attributed across **planner / router / tools / synthesis / verify**.
- Total time, recursion depth, and the stop reason. Judge/verifier costs are kept
  separate from the answer's accounting where relevant.

---

## The GUI

The React SPA has three pages:

- **Chat** — ask questions in any mode. For Agentic RAG you get the per-question
  multi-hop toggle, planner choice, a Settings panel (web/YouTube tools, `max_steps`
  / `max_hops` / `max_depth`, verify, temperature), a live progress indicator, and
  the full trace + token breakdown under each answer. Chats and settings persist for
  the session and chats can be deleted.
- **Comparison** — ask one question across several models at once and compare
  response time and token cost with charts.
- **Quiz** — described below.

### Quiz page

A **host** (a strong model with a Q&A dataset) quizzes a **player** (any QA mode) and
scores its performance.

- **Dataset** — defaults to **Who Wants to Be a Millionaire** (`millionaire.csv` from
  [`RedBlock/parrot`](https://huggingface.co/datasets/RedBlock/parrot)), downloaded
  and cached once. The player is asked the **bare question** (no multiple-choice
  options). The loader is **pluggable** (dataset id + column mapping) for other sets.
- **Player** — any mode (**Agentic RAG by default**), with **all its settings**
  available, exactly as in Chat.
- **Flow** — the user picks how many questions (any number, or **All**; fewer than
  all are **randomly sampled**) and starts. The host asks each question, the player
  answers, and the host marks it **right or wrong** against the reference (binary
  LLM-as-judge). The reference is revealed only after grading.
- **Statistics** — live and final: **correct / wrong / accuracy**, **time** (total
  and per question), and **tokens** (input / output / reasoning, total and per
  question), plus the per-answer agentic trace. The host's own cost is excluded from
  the player's stats.
- **Runs across tab switches** — the quiz keeps running if you navigate to another
  page, and the results are still there when you return.

---

## Setup

Requires **Python 3.12** (the ML stack has no 3.14 wheels yet) and **Node** for the
frontend.

```bash
# Backend: venv + dependencies
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Frontend
npm --prefix frontend install
```

Provide a proxy token in a git-ignored **`.llmtoken`** file at the repo root (or via
`LLM_TOKEN` / `OPENAI_API_KEY` / `HF_TOKEN`).

Build the RAG index once (needed for RAG / Agentic RAG retrieval):

```bash
.venv/bin/python bin/build_index.py            # all 10k articles
.venv/bin/python bin/build_index.py --max-docs 500   # a quick subset
```

> On first use, Agentic RAG downloads the small router model (~3 GB, once), the Quiz
> downloads its dataset (once), and RAG builds/loads the local embedding index.

## Running

```bash
make dev        # backend (:8000) + frontend (:5173) together
# or individually:
make backend    # FastAPI on :8000
make frontend   # Vite dev server on :5173
```

Open **http://localhost:5173**. The FastAPI backend also serves the production
frontend build from `frontend/dist/` on **:8000**.

Key endpoints: `POST /api/chat`, `POST /api/compare`, and the quiz endpoints
`GET /api/quiz/datasets`, `POST /api/quiz/start`, `POST /api/quiz/answer`.

## Configuration

QA systems are configured per run (YAML in `configs/`, or per API request). Notable
Agentic-RAG settings and their defaults:

| Setting | Default | Meaning |
|---------|---------|---------|
| `router_model` | `Qwen/Qwen2.5-1.5B-Instruct` | local tool-routing model |
| `enable_online_tools` | `true` | allow `google_search` + `youtube` |
| `multi_hop` | `true` | decompose into sub-questions |
| `planner_model` | `router` | `router` (small) or `answer` (big model) |
| `max_steps` | `6` | tool calls before forced synthesis |
| `max_hops` | `3` | sub-questions per planning step |
| `max_depth` | `1` | recursive planning levels (`>1` nests sub-agents) |
| `verify` | `true` | bounded verification/backtracking hop |
| `rag_first` | `true` | always search the local index first |

## Testing

The test suite is **network-free** (the LLM proxy, the local router, HTTP tools, and
datasets are all mocked):

```bash
make test        # or: .venv/bin/python -m pytest -q tests/
```

## Project layout

```
oracle/
  config.py         QAConfig
  llm.py            proxy client + token accounting (UsageMetrics)
  qa/               world / rag / a-rag QA systems + factory
  retrieval/        chunking, local embeddings, FAISS retriever, index build/load
  tools/            the agent toolbox (search, calculator, google_search, youtube, …)
  agent/            router, bounded loop, planner, multi-hop, verifier, trace
  eval/             LLM-as-a-judge + evaluation runner
  quiz/             quiz dataset loader + host judge
  api.py            FastAPI app (chat / compare / quiz + serves the SPA)
bin/                helper scripts (fetch_dataset, build_index, run_eval)
configs/            QA-system configs + model registry
frontend/           React SPA (Chat, Comparison, Quiz pages)
tests/              pytest (no network)
```

Design notes and per-feature specs live in [`tools.md`](tools.md) (agentic tools) and
[`quiz.md`](quiz.md) (quiz), with the roadmap in [`TODO.md`](TODO.md).
