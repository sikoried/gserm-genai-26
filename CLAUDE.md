# System Overview

Oracle is a platform to evaluate different question-answering approaches, starting from a basic LLM-generated response (world knowledge), reaching over different RAG-flavors to an agentic RAG approch.
It consists of an API on the backend (which needs to be testable for scripted/batch evaluation runs) and a frontend that is javascript based.

All LLM calls will be routed to an external server; document embeddings shall be computed locally.


## Preferred Toolkits

- Frontend: React
- Backend: fastapi, HF openai chat library (token from `.llmtoken`, falling back to ENV)


## Project Layout

- `oracle/` — core library (importable, so QA + eval run without the HTTP layer):
  `config.py`, `llm.py`, `qa/` (QA systems), `eval/` (judge + runner).
- `bin/` — all helper programs / scripts (dataset fetching, eval runner, …).
- `configs/` — QA-system configs (one YAML per system, e.g. `world.yaml`).
- `eval/` — evaluation query/answer sets.
- `tests/` — unit tests (`pytest`, no network).
- `data/` — local dataset + embedding caches + eval results (gitignored).
- `.venv/` — Python virtual environment (gitignored).


## Python Environment

- Python **3.12** (the ML stack — torch, sentence-transformers — has no wheels for 3.14 yet).
- Virtual environment in `.venv`; dependencies in `requirements.txt`.
- Setup: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- Run helpers via the venv interpreter, e.g. `.venv/bin/python bin/fetch_dataset.py`.


## Data / Corpus

- RAG variants retrieve over [`NeelNanda/wiki-10k`](https://huggingface.co/datasets/NeelNanda/wiki-10k).
- Shape: 10,000 rows, columns `id`, `url`, `title`, `text` — **full Wikipedia articles**
  (text length min ~10, mean ~22k, max ~200k chars), so documents must be **chunked**
  before embedding/retrieval.
- `bin/fetch_dataset.py` downloads and explores it (schema, row count, length stats, samples).
  Cached under `data/hf-cache`.


## Embeddings & Vector Store

- Document embeddings are computed **locally** (not via the LLM proxy) with
  `sentence-transformers` (default model `all-MiniLM-L6-v2`, 384-dim).
- Vector store: **FAISS in-memory** (`faiss-cpu`) — a basic exact flat index
  (`IndexFlatIP` over L2-normalized embeddings = cosine similarity); the app owns the
  id→document mapping. Retrieval sits behind a small, swappable `Retriever` interface.
- Backlog: ChromaDB (in-memory `EphemeralClient`) as an alternative when metadata
  filtering / persistence is wanted — see `TODO.md`.


## LLM Access & Secrets

- All LLM calls (answering and judging) go to the proxy `endpoint` via the OpenAI-compatible
  chat API (`openai` client) — see `oracle/llm.py`.
- Token: read from the git-ignored `.llmtoken` file at repo root (preferred), else env
  (`LLM_TOKEN` / `OPENAI_API_KEY` / `HF_TOKEN`).
- Default answering model: `mistralai/Mistral-Medium-3.5-128B`. Judge model: `openai/gpt-oss-120b`.


## Configuration Schema

The different QA-systems will follow a yaml scheme:

```yaml
type: The type of the model, for now either of world, rag or a-rag.
model: Any huggingface LLM, default to mistralai/Mistral-Medium-3.5-128B
endpoint: https://kiz1.in.ohmportal.de/llmproxy/v1
temperature:
```


## Evaluation

We'll use LLM-as-a-judge (`openai/gpt-oss-120b`) to measure the quality and correctness of answers.

- The judge classifies each answer into one verdict: **correct** (same essential information as
  the reference), **wrong** (contradicts / factually incorrect), or **orthogonal** (neither right
  nor wrong — refuses, off-topic, tangential). Unparseable judge replies are bucketed as `error`.
- Run end-to-end: `.venv/bin/python bin/run_eval.py --config configs/world.yaml` — answers each
  pair with the configured QA system, judges it, prints per-item verdicts + a summary, and can
  write JSON results (`--out`).
- Real query/answer pairs (grounded in the wiki-10k corpus) will be authored later; for now
  `eval/qa_pairs.yaml` holds a small stub of trivial examples (`id`, `question`, `reference_answer`).
