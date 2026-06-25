# Requirements — Advanced RAG (Simon)

> **Purpose of this document.** This is the implementation brief for the next development
> week. It turns my short task list into a detailed, code-grounded spec so a fresh Claude
> Code session (or a teammate) can start implementing immediately without re-discovering
> the codebase. It is written against the **current state** of `oracle/` on branch
> `feature/advanced-rag-retrieval`.

---

## 0. Original task list (verbatim) and what it maps to

My raw notes were:

- **Chunking**
  - Meaningful chunks — e.g. with an LLM
  - Multi-Resolution
- **Retrieval** (#, Rank, etc.)
- **Context size** → context compression?
- **Small LLM?**
- **Choice of Q → RS** — maybe compress, etc.

Decoded into concrete features (IDs used throughout this doc):

| ID  | Feature                          | One-line description |
|-----|----------------------------------|----------------------|
| F1  | **Meaningful chunking**          | Replace blind 800-char windows with structure-/semantic-aware chunks; optionally LLM-assisted. |
| F2  | **Multi-resolution chunking**    | Index the corpus at several granularities (e.g. sentence / paragraph / section) and/or use small-to-big "retrieve small, return big". |
| F3  | **Retrieval controls & ranking** | Make `#` of hits, score thresholds, and **re-ranking** (cross-encoder / MMR diversity) first-class and configurable. |
| F4  | **Context size & compression**   | Budget the prompt context by tokens (not chunk count); compress retrieved passages before they hit the answering LLM. |
| F5  | **Small LLM for sub-tasks**      | Route cheap auxiliary jobs (chunking hints, compression, query rewrite, rerank) to a small/cheap model instead of the main answering model. |
| F6  | **Query → Result-Set handling**  | Improve how the query is formed/selected before retrieval (rewrite, expansion, multi-query, HyDE) and optionally compress the query/result set. |

Each feature has its own section below with: **Motivation → Design → Code touchpoints →
Config knobs → Acceptance criteria → Tests**. A suggested build order is in §8.

---

## 1. Current state (what already exists — do not rebuild)

Read these before writing code; the whole feature set is *extensions* of this stack.

### Retrieval stack — `oracle/retrieval/`
- **`chunking.py`** — `chunk_text(text, size=800, overlap=150)` slices into overlapping
  **character** windows; `chunk_document(doc, …)` returns `list[Chunk]`. `Chunk` is a
  dataclass: `text, doc_id, title, url`. **No notion of sentence/section boundaries today.**
- **`store.py`** —
  - `Embedder(model_name="all-MiniLM-L6-v2")`: sentence-transformers, auto-picks
    `mps`/`cuda`/`cpu`, `encode()` returns **L2-normalized float32** (so dot product = cosine).
    Loads with `local_files_only=True` first (offline), falls back to a one-time download.
  - `Retriever` (ABC) with `search(query_embedding, k) -> list[Hit]`.
  - `FaissRetriever`: exact `IndexFlatIP` over normalized vectors. `Hit = {text, title, url, score}`.
- **`index.py`** — `build_index(max_docs, chunk_size, overlap, model_name, cache_dir)` chunks
  + embeds + caches to `data/rag-index/` as `embeddings.npy` + `chunks.jsonl` + `meta.json`.
  `load_retriever(cache_dir)` rebuilds the in-memory FAISS index from that cache.
  **The cache schema is the integration seam for F1/F2** (see §2.1).

### QA systems — `oracle/qa/`
- **`base.py`** — `QASystem` ABC (`answer(question) -> Answer`); `Answer = {content, metrics, reasoning?}`.
- **`rag.py`** — `RagQA`: embeds the (first) user message directly (no rewrite), `search(top_k)`,
  renders hits via the **`rag_prompt.j2`** Jinja2 template into a system message, then answers.
  `answer_chat()` retrieves **only on the first user message**; later turns just extend context.
  Retriever + embedder are **class-level singletons** (heavy; shared per process).
- **`arag.py`** — `AgenticRagQA`: smolagents `ToolCallingAgent` that decides whether to retrieve,
  with `search` + `query_rewrite` tools and a `_build_trace()` reasoning summary. Reuses
  `rag.DEFAULT_SYSTEM_PROMPT`.
- **`rag_prompt.j2`** — minimal: numbered `[i] {title}\n{text}` blocks.
- **`__init__.py`** — `build_qa_system(config)` factory dispatches on `config.type`
  (`world` / `rag` / `a-rag`).

### Agent tools — `oracle/tools/`
- `search.py` (`@tool search(query, k=10, min_similarity=0.0)`), `rewrite.py`
  (`@tool query_rewrite(query)` via the LLM), `runtime.py` (module-level retriever/embedder/LLM
  the tools read — configured per process by `AgenticRagQA`).

### Config / API / scripts
- **`config.py`** — `QAConfig` (pydantic). Relevant fields today: `type, model, endpoint,
  temperature, system_prompt, reasoning_effort, top_k=10, embedding_model="all-MiniLM-L6-v2"`.
  Loaded from YAML via `QAConfig.from_yaml`. **All new knobs go here** (see §7.1).
- **`api.py`** — FastAPI. `/api/chat` maps a `mode` (`World`/`RAG`/`Agentic RAG`) → QA `type`.
  Note: `/api/chat` currently builds `QAConfig` **without** passing `top_k`/`embedding_model`,
  so it uses defaults — wire new knobs through here too (see §7.2).
- **`bin/build_index.py`** — CLI wrapper over `build_index` (`--max-docs --chunk-size --overlap --model`).
- **`configs/*.yaml`** — one self-contained QA system per file. Add new example configs here (§7.3).
- **Index cache lives in `data/rag-index/` (gitignored).** Rebuild it whenever chunking changes.

### Constraints baked into the project (keep these!)
- **Embeddings local only**; only the **LLM proxy** (`kiz1…/llmproxy/v1`) is contacted at request time.
- `all-MiniLM-L6-v2` truncates at **256 tokens (~1000 chars)** → chunks must stay under that to
  embed without truncation. Any new chunker must respect the embedding model's window.
- Library must stay **importable & testable without the HTTP layer**; unit tests are **no-network**
  (see `tests/`, `conftest.py`). New unit tests must not hit the proxy or download models.
- Python **3.12**, deps in `requirements.txt`. New deps must have 3.12 wheels.

---

## 2. F1 — Meaningful chunking

### Motivation
Fixed 800-char windows cut mid-sentence and mix unrelated topics, which dilutes embeddings and
yields noisy hits. We want chunks whose boundaries follow the document's structure/meaning so each
embedding represents one coherent idea.

### Design — three strategies behind one interface
Introduce a **`Chunker` strategy** so chunking is swappable like `Retriever` already is.

```
oracle/retrieval/chunking.py
  class Chunker(Protocol):  def split(self, doc: dict) -> list[Chunk]
  - FixedChunker        # current behavior (size/overlap) — keep as default & baseline
  - StructuralChunker   # split on Wikipedia structure: blank lines / headings / sentences
  - SemanticChunker     # embedding-similarity boundary detection (no LLM)
  - LlmChunker          # OPTIONAL: LLM proposes split points (small model, F5)
```

1. **StructuralChunker (must):** split on paragraph breaks (`\n\n`), then pack consecutive
   paragraphs/sentences greedily up to a target token budget (respect the 256-token embed window),
   carrying a small sentence-level overlap. Use a lightweight sentence splitter (regex on
   `.?!` + abbreviation guard, or `blingfire`/`nltk punkt` if we accept a dep). This is cheap,
   deterministic, no network — the **recommended default** for the meaningful-chunk requirement.
2. **SemanticChunker (should):** embed sentences, start a new chunk when cosine similarity between
   consecutive sentences drops below a threshold (or use a rolling window). Pure local embeddings,
   no LLM. More faithful to topic shifts; more expensive to build.
3. **LlmChunker (could, ties to F5):** ask a **small** LLM to return split offsets/section labels
   for an article. Cache aggressively (build-time only). Gate behind config because it costs tokens
   for 10k articles — likely run on a `--max-docs` subset for the demo.

Keep `size`/`overlap` measured in **tokens** going forward (add a tokenizer helper) rather than raw
chars, since the embedding limit is a token limit; expose char-mode for backward compatibility.

### Code touchpoints
- `oracle/retrieval/chunking.py` — add `Chunker` protocol + the implementations; keep
  `chunk_text`/`chunk_document` as the `FixedChunker` path (don't break existing imports/tests).
- `oracle/retrieval/index.py` — `build_index` takes a `chunker` (or a `chunking:` config block) and
  writes the **chunking strategy + params into `meta.json`** so `load_retriever` and eval can report it.
- `bin/build_index.py` — add `--chunker {fixed,structural,semantic,llm}` and strategy params.

### Config knobs (see §7.1 for the full block)
```yaml
chunking:
  strategy: structural        # fixed | structural | semantic | llm
  target_tokens: 220          # stay < 256 embed window
  overlap_tokens: 40
  semantic_threshold: 0.55    # SemanticChunker only
  llm_model: <small model>    # LlmChunker only (F5)
```

### Acceptance criteria
- `bin/build_index.py --chunker structural --max-docs 200` builds a cache whose `meta.json`
  records the strategy; chunk count differs from `fixed` and no chunk exceeds the token window.
- A handful of spot-checked chunks start/end at sentence/paragraph boundaries (not mid-word).
- Retrieval quality on the eval set (see §9) is **≥** the fixed baseline (don't regress).

### Tests (no network)
- Unit-test `StructuralChunker` on a synthetic multi-paragraph string: boundaries land on `\n\n`,
  no chunk over the token budget, overlap present.
- `SemanticChunker` can be tested with a **fake embedder** (inject a stub returning canned vectors)
  to assert it splits where similarity drops — keep it network-free.

---

## 3. F2 — Multi-resolution chunking & retrieval

### Motivation
Small chunks retrieve precisely but lack context for the answer; large chunks give context but are
noisy to match. Multi-resolution gives us both: **match on small, answer on large.**

### Design — pick one (A is recommended for a 1-week scope)
- **A. Small-to-big / parent-child (recommended).** Chunk each doc at a **fine** level (child) for
  embedding+search, but keep a pointer to a **coarse** parent (e.g. the section or a ±N-neighbor
  window). Retrieve on children, then **expand each hit to its parent** before building context
  (deduplicating parents). Only the child level is embedded → one index, cheap.
- **B. Multi-index.** Build **separate** indices at 2–3 resolutions (sentence / paragraph / section),
  query each, and merge/rerank (depends on F3). More flexible, more memory + plumbing.

For either, the `Chunk` dataclass needs to carry hierarchy:
```
Chunk += { chunk_id, parent_id, level, doc_id, start_char, end_char }
```
and the cache must store the **parent text** (or enough offsets to reconstruct it from the original
doc) so `RagQA.retrieve` can expand hits.

### Code touchpoints
- `oracle/retrieval/chunking.py` — emit child chunks with `parent_id` + level; produce a
  `parents` map (`parent_id -> text/title/url`).
- `oracle/retrieval/index.py` — persist `parents.jsonl` alongside `chunks.jsonl`; `load_retriever`
  loads both. Bump `meta.json` with `resolutions`.
- `oracle/retrieval/store.py` — add an optional **expand step**: `FaissRetriever.search` stays the
  same (child-level), expansion happens in a thin wrapper (`MultiResolutionRetriever`) or in
  `RagQA.retrieve` to keep the store generic. Deduplicate parents, keep best child score per parent.
- `oracle/qa/rag.py` — after `search`, map child hits → parent passages for the template.

### Config knobs
```yaml
retrieval:
  multi_resolution: small_to_big   # off | small_to_big | multi_index
  parent_level: section            # what "big" means: section | neighbors
  parent_window: 1                 # ± neighbor chunks if parent_level=neighbors
```

### Acceptance criteria
- With `small_to_big`, a query that hits a child chunk produces context that includes the
  surrounding parent text (verifiable by length/inclusion), with no duplicate parents.
- Index build still completes on `--max-docs 200` within a reasonable time and the cache documents
  the resolutions used.

### Tests
- Unit: build child+parent chunks from a synthetic doc; assert each child's `parent_id` resolves and
  expansion returns the parent text exactly once when two children share a parent.

---

## 4. F3 — Retrieval controls & ranking

### Motivation
"`#`, Rank, etc." — make retrieval depth, thresholds, and **ordering quality** explicit and tunable
instead of the single `top_k`. Bi-encoder cosine alone misorders near-duplicates and ignores
diversity.

### Design
1. **Two-stage retrieve-then-rerank (should):** fetch a wider candidate set `fetch_k` (e.g. 30) with
   FAISS, then re-score the top candidates with a **cross-encoder reranker**
   (`sentence-transformers` `CrossEncoder`, e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`, **local**)
   and keep the top `top_k`. This is the single highest-leverage quality win.
2. **MMR diversity (could):** Maximal Marginal Relevance over candidate embeddings to drop
   near-duplicate chunks (λ tradeoff between relevance and novelty). Useful because Wikipedia
   articles repeat phrasing.
3. **Score threshold / min-similarity (must):** already exists in the agent `search` tool; lift it
   to `RagQA` so low-confidence hits can be dropped (and the prompt can honestly say "not found").
4. **Expose counts in the trace/response:** record `fetch_k`, `top_k`, post-rerank order, and scores
   so we can show *why* a passage was used (good for the demo + eval).

### Code touchpoints
- `oracle/retrieval/store.py` — add a `Reranker` interface + `CrossEncoderReranker`; an `MmrReranker`
  (pure numpy over candidate embeddings). Keep them **optional & lazy-imported** (don't load the
  cross-encoder unless configured — it's a second model download).
- `oracle/retrieval/__init__.py` — export the new types.
- `oracle/qa/rag.py` — pipeline becomes: `embed → search(fetch_k) → [rerank] → [mmr] → threshold →
  top_k → template`.
- `oracle/tools/search.py` — optionally honor the same rerank settings so `a-rag` benefits too.

### Config knobs
```yaml
retrieval:
  top_k: 6                  # final passages into the prompt
  fetch_k: 30               # candidates pulled before rerank
  min_similarity: 0.25
  rerank: cross_encoder     # off | cross_encoder
  rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  mmr: false
  mmr_lambda: 0.5
```
(Move `top_k` under a `retrieval:` block; keep top-level `top_k` working as an alias for back-compat
— see §7.1.)

### Acceptance criteria
- With `rerank: cross_encoder`, the ordering of returned hits differs from pure cosine on at least
  some eval queries, and end-to-end eval accuracy is **≥** the no-rerank baseline.
- `min_similarity` filters out hits below threshold; when all are filtered the system answers
  "not found in context" rather than hallucinating.
- Reranker/MMR are **never loaded** when disabled (verify by config + lazy import).

### Tests
- `MmrReranker` unit test with canned embeddings: asserts a near-duplicate is demoted.
- Cross-encoder path tested with a **stub reranker** (network-free) to verify the pipeline reorders.

---

## 5. F4 — Context size management & context compression

### Motivation
`top_k` chunks of arbitrary length can blow the prompt budget and bury the answer in filler.
We want to **budget context by tokens** and **compress** passages to the parts that matter.

### Design
1. **Token budgeting (must):** add a `context_budget_tokens` cap. After ranking, greedily add
   passages until the budget is hit (count tokens with a tokenizer helper, §7.4). Replaces/augments
   the raw `top_k` cut. Report how many candidates were dropped for budget.
2. **Extractive compression (should):** for each retained passage, keep only the **sentences most
   similar to the query** (re-use the local embedder: embed sentences, score vs. query, keep top-N
   or above-threshold). Cheap, deterministic, no LLM. This is the recommended compression for the
   demo.
3. **Abstractive compression (could, ties to F5):** a **small** LLM summarizes/condenses each
   passage (or the whole context) toward the query before the answering model sees it
   ("context distillation"). Costs tokens; gate behind config and prefer the small model.
4. **Template update:** `rag_prompt.j2` should render compressed passages and keep the
   `[i] title` citation handles intact.

### Code touchpoints
- New `oracle/retrieval/compress.py` — `extractive_compress(hits, query, embedder, …)` and an
  optional `llm_compress(...)`. Pure functions; the extractive path takes an injected embedder so
  it's testable.
- `oracle/qa/rag.py` — insert compression between ranking and template render; enforce the token budget.
- `oracle/qa/rag_prompt.j2` — minor: show compressed text; optionally mark elided spans with `…`.
- Tokenizer helper (§7.4) shared with F1/F3.

### Config knobs
```yaml
context:
  budget_tokens: 1500
  compression: extractive    # off | extractive | llm
  keep_sentences: 3          # extractive: top-N sentences per passage
  compress_model: <small>    # llm compression (F5)
```

### Acceptance criteria
- Total rendered context never exceeds `budget_tokens` (assert in a test with a tiny budget).
- With `extractive`, rendered passages are shorter than the raw chunks yet still contain the
  answer span for eval queries; eval accuracy does not regress vs. uncompressed.

### Tests
- Token-budget enforcement test (tiny budget → context truncated, citations preserved).
- Extractive compression with a fake embedder: asserts the query-relevant sentence survives and an
  irrelevant one is dropped.

---

## 6. F5 — Small LLM for auxiliary sub-tasks

### Motivation
Chunking hints (F1), query rewrite (F6), abstractive compression (F4), and (optionally) LLM
re-ranking are **cheap, high-volume** jobs. Running them on the big answering model wastes tokens and
latency. Route them to a **small/cheap** model.

### Design
- Add an **`aux_model`** (a.k.a. "small model") concept to config, separate from the answering
  `model`. Default it to a small instruct model available on the proxy (confirm availability against
  `oracle/models.py` / `configs/models.yaml` — **TODO: pick the smallest model the proxy exposes**).
- Provide a single helper `aux_client_and_model(config)` so every sub-task (rewrite/compress/chunk)
  pulls the same small model. The agent `tools/runtime.py` already holds an LLM handle — extend it to
  hold **both** the answering and the aux model, and point `query_rewrite` at the aux one.
- Make it a **no-op fallback**: if `aux_model` is unset, sub-tasks reuse the main model so nothing
  breaks.

### Code touchpoints
- `oracle/config.py` — add `aux_model: str | None`.
- `oracle/llm.py` — small helper to build a client+model for aux tasks (reuse `make_client`).
- `oracle/tools/runtime.py` + `oracle/tools/rewrite.py` — use the aux model.
- `oracle/retrieval/compress.py` (F4) + `LlmChunker` (F1) — accept the aux model.

### Config knobs
```yaml
aux_model: <small model id>   # used for rewrite / compression / llm-chunking; falls back to `model`
```

### Acceptance criteria
- Setting `aux_model` routes rewrite/compression calls to it (verify via a recorded call / stub),
  while the final answer still comes from `model`.
- Unset `aux_model` → identical behavior to today (regression-safe).

### Tests
- Network-free: stub the LLM client and assert the aux path is invoked with the configured small
  model id.

---

## 7. F6 — Query → Result-Set handling ("Choice of Q → RS")

### Motivation
The basic `rag` system embeds the user message **verbatim**. Short or ambiguous queries embed poorly.
Improving the **query** (and optionally compressing it / the result set) lifts retrieval quality
without touching the index.

### Design — opt-in query transforms (all behind config, default off to preserve `rag` semantics)
1. **Query rewrite/expansion (should):** reuse the existing `query_rewrite` (now on the aux model,
   F5) to turn a terse query into a descriptive sentence before embedding. The agent already does
   this; bring it to basic `rag` as an **optional** step (note: rag-spec says basic RAG embeds
   directly, so keep this **off by default** and clearly flagged).
2. **Multi-query (could):** generate K paraphrases, retrieve for each, then **union + dedupe + rerank**
   (F3). Improves recall for ambiguous questions.
3. **HyDE (could):** ask the aux LLM for a hypothetical answer, embed *that*, retrieve against it.
   Strong for knowledge questions; one extra small-LLM call.
4. **Query compression (could):** for long/chatty inputs, compress to keywords/intent before
   embedding (aux model or simple stopword/keyphrase extraction).
5. **Result-set compression** is covered by F4 (it's the same idea applied to the retrieved set).

### Code touchpoints
- New `oracle/retrieval/query.py` (or extend `tools/rewrite.py`) with `transform_query(query, mode,
  aux_llm) -> list[str]` returning one or more query strings.
- `oracle/qa/rag.py` — call the transform before embedding; when multiple queries come back, retrieve
  for each and merge before rerank/compress.

### Config knobs
```yaml
query:
  transform: none        # none | rewrite | multi | hyde
  num_queries: 3         # multi-query
```

### Acceptance criteria
- Default (`none`) reproduces today's `rag` behavior exactly (verbatim embed).
- `multi` retrieves a merged, deduped candidate set larger than single-query before rerank cuts it back.

### Tests
- `transform_query` with a stub LLM: `rewrite` returns one transformed string; `multi` returns
  `num_queries` distinct strings; merge dedupes overlapping hits.

---

## 8. Cross-cutting: config, API, GUI, configs, tokenizer

### 8.1 `QAConfig` changes (`oracle/config.py`)
Add nested pydantic sub-models so the YAML stays readable, **while keeping the existing flat
`top_k`/`embedding_model` working** (back-compat: map a top-level `top_k` onto `retrieval.top_k`).
Target shape of a full advanced-RAG config:

```yaml
type: rag
model: mistralai/Mistral-Medium-3.5-128B
endpoint: https://kiz1.in.ohmportal.de/llmproxy/v1
temperature: 0.0
embedding_model: all-MiniLM-L6-v2
aux_model: <small model>            # F5

chunking:                            # F1/F2 — affects the INDEX (rebuild required)
  strategy: structural
  target_tokens: 220
  overlap_tokens: 40
  multi_resolution: small_to_big
  parent_level: section

retrieval:                           # F3
  top_k: 6
  fetch_k: 30
  min_similarity: 0.25
  rerank: cross_encoder
  rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  mmr: false

context:                             # F4
  budget_tokens: 1500
  compression: extractive
  keep_sentences: 3

query:                               # F6
  transform: none
```

**Important distinction to document in the config:** `chunking.*` changes require **rebuilding the
index** (`bin/build_index.py`), whereas `retrieval.*`, `context.*`, `query.*`, and `aux_model` are
**query-time** and need no rebuild. Make `build_index` write the chunking params into `meta.json`
and have `load_retriever`/`RagQA` **warn on mismatch** between the config's chunking block and the
cached index.

### 8.2 API exposure (`oracle/api.py`)
- `/api/chat` currently drops `top_k`/`embedding_model` when building `QAConfig`. When wiring RAG
  knobs, decide the surface: simplest is to **load RAG settings from a named config file** server-side
  (e.g. a `rag_profile` field) rather than exploding the chat request with a dozen fields.
  Recommended: add `rag_profile: str = "rag"` to `ChatRequest`, load `configs/<profile>.yaml`,
  override `model`/`temperature` from the request. This keeps the GUI simple and lets us demo
  several RAG flavors by switching profiles.
- Optionally surface retrieved-passage metadata (titles/scores) in `ChatResponse` for a "sources"
  panel in the GUI (nice for the demo; low effort since `Hit` already has `title`/`url`/`score`).

### 8.3 GUI (React, `frontend/`)
- Add a **RAG profile selector** (dropdown of available `configs/*.yaml` of type `rag`/`a-rag`).
- Optional: a **"Sources" disclosure** showing the retrieved titles + scores returned by the API.
- Keep it minimal — the backend is where the assignment's substance is.

### 8.4 Example configs (`configs/`)
Author one config per flavor so they can be compared side-by-side (mirrors the existing
`world.yaml` / `world_plus.yaml` pattern):
- `rag.yaml` — current baseline (fixed chunk, no rerank) — **keep as the control**.
- `rag_structural.yaml` — F1 structural chunking.
- `rag_multi.yaml` — F2 small-to-big.
- `rag_rerank.yaml` — F3 cross-encoder.
- `rag_compress.yaml` — F4 extractive compression + token budget.
- `rag_full.yaml` — everything on (the showcase config).

### 8.5 Tokenizer helper
Several features need token counts. Add `oracle/retrieval/tokens.py` with `count_tokens(text)` /
`truncate_to_tokens(...)`. Prefer the **embedding model's own tokenizer** (`Embedder.model.tokenizer`)
for chunk/embed limits, and a generic `tiktoken`-style or word-count approximation for the prompt
budget if the proxy model's exact tokenizer isn't available. Keep it dependency-light and offline.

---

## 9. Evaluation (how we prove each feature helps)

The project already has an eval harness — **use it as the success metric for every feature.**
- `bin/run_eval.py --config configs/<flavor>.yaml --out data/eval/<flavor>.json` answers + judges
  (`openai/gpt-oss-120b`, verdicts: correct/wrong/orthogonal) over `eval/qa_pairs.yaml`.
- **Action items:**
  1. Author **real query/answer pairs grounded in wiki-10k** (the current `eval/qa_pairs.yaml` is a
     trivial stub — see TODO.md). ~15–30 questions answerable from the corpus, ideally a mix of
     easy/lookup and multi-fact. This is a prerequisite for meaningful comparison.
  2. Run each `configs/rag_*.yaml` flavor and record verdict counts + token usage in a small table.
  3. Report **retrieval-level** metrics too (recall@k / whether the gold passage was retrieved) where
     we have gold passages, not just end-to-end judge verdicts — this isolates retrieval wins from
     answering-model noise.
- Add a tiny `bin/compare_rag.py` (or extend `run_eval`) to run several configs and print a comparison
  table — makes the demo a one-command story.

---

## 10. Suggested build order (1 week, priority-tagged)

Build in this order; each step is independently demoable and falls back safely.

1. **Foundations** — tokenizer helper (§8.5); nested `QAConfig` with back-compat (§8.1);
   `rag.yaml` baseline config + authored eval pairs (§9). *(must)*
2. **F3 retrieval controls** — `fetch_k`, `min_similarity`, cross-encoder rerank. Highest quality/
   effort ratio, query-time only (no rebuild). *(must/should)*
3. **F1 structural chunking** — `StructuralChunker` + `build_index` wiring + `meta.json` strategy.
   *(must)*
4. **F4 token budget + extractive compression.** *(must/should)*
5. **F2 small-to-big multi-resolution.** *(should)*
6. **F5 aux/small model** — wire rewrite/compression to it. *(should)*
7. **F6 query transforms** — rewrite/multi-query/HyDE (opt-in). *(could)*
8. **SemanticChunker / LlmChunker, MMR, abstractive compression** — stretch goals if time remains.
   *(could)*
9. **API `rag_profile` + GUI selector + sources panel.** *(should — needed for the live demo)*
10. **Eval comparison table across all flavors + write-up.** *(must — this is the deliverable's proof)*

**Definition of done for the week:** baseline + ≥3 advanced flavors (rerank, structural chunking,
compression) implemented behind config, each with no-network unit tests, all selectable in the GUI,
and an eval table showing their effect on judge verdicts and token usage.

---

## 11. Open questions / decisions to make early

- **Small model id (F5):** which model on the proxy is the "small" one? Check `oracle/models.py` /
  `configs/models.yaml` and confirm with the proxy. Blocks F5/abstractive paths.
- **New dependencies:** cross-encoder reranking adds a second `sentence-transformers` model download
  (still local inference — OK). A sentence splitter may add `blingfire`/`nltk`. Confirm 3.12 wheels
  and that the demo machine can fetch the model once (then `local_files_only`).
- **Index size with multi-resolution:** building child+parent over all 10k articles may be slow/large.
  Decide whether the demo runs on a `--max-docs` subset (recommended) and document it in `meta.json`.
- **Does basic `rag` stay verbatim?** rag-spec.md says basic RAG embeds the message directly. Keep the
  baseline `rag.yaml` faithful to that; put query transforms only in the advanced flavors so we don't
  silently change the control.
- **a-rag reuse:** F3/F4/F5 should also benefit `a-rag` via `tools/search.py` + `tools/runtime.py`.
  Decide whether to do that in-scope or note it as follow-up.

---

## 12. Quick reference — files you'll most likely touch

| Area            | Files |
|-----------------|-------|
| Chunking F1/F2  | `oracle/retrieval/chunking.py`, `oracle/retrieval/index.py`, `bin/build_index.py` |
| Retrieval F3    | `oracle/retrieval/store.py`, `oracle/retrieval/__init__.py`, `oracle/qa/rag.py`, `oracle/tools/search.py` |
| Compression F4  | `oracle/retrieval/compress.py` (new), `oracle/qa/rag.py`, `oracle/qa/rag_prompt.j2` |
| Small LLM F5    | `oracle/config.py`, `oracle/llm.py`, `oracle/tools/runtime.py`, `oracle/tools/rewrite.py` |
| Query F6        | `oracle/retrieval/query.py` (new) or `oracle/tools/rewrite.py`, `oracle/qa/rag.py` |
| Config/API/GUI  | `oracle/config.py`, `oracle/api.py`, `configs/rag*.yaml`, `frontend/` |
| Eval            | `bin/run_eval.py`, `bin/compare_rag.py` (new), `eval/qa_pairs.yaml` |
| Tests           | `tests/test_*.py` (network-free; inject fake embedder/LLM stubs) |

> Remember after any **chunking** change: rebuild the index
> (`.venv/bin/python bin/build_index.py --chunker … --max-docs …`) before evaluating.
