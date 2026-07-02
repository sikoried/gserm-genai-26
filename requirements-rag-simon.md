# Requirements — Advanced RAG (Implementation Spec for Next Sprint)

> **Purpose**  
> This document defines the implementation requirements for the next development sprint.  
> It is written so a new engineer/session can start coding immediately without rediscovering architecture decisions.

> **Codebase baseline**  
> Target state is based on the current `oracle/` code on branch `feature/advanced-rag-retrieval`.

---

## 0 Sprint objective and success criteria

### Objective
Ship a robust, configurable Advanced RAG pipeline with measurable retrieval-quality improvements, while preserving baseline behavior as a control setup.

### Minimum deliverables (must ship)
1. Backward-compatible config model with advanced RAG blocks.
2. Retrieval controls (fetch_k, min_similarity, optional rerank).
3. Structural chunking (token-budget aware) integrated into index build.
4. Context token budgeting + extractive compression.
5. Eval comparison across baseline + at least 3 advanced profiles.

### Definition of done (DoD)
- Baseline config still behaves exactly like current `rag` behavior.
- At least 3 advanced profiles are selectable and runnable.
- No-network unit tests for new logic.
- Eval report table includes quality deltas + token usage deltas.
- Chunking configuration mismatch is detectable and warned at runtime.

---

## 1 Existing system (do not rebuild, extend only)

### Retrieval (`oracle/retrieval/`)
- `chunking.py`: fixed char-window chunking (`size=800`, `overlap=150`), no semantic structure.
- `store.py`: local sentence-transformers embedder, normalized vectors, FAISS IP retrieval.
- `index.py`: builds cache in `data/rag-index/` (`embeddings.npy`, `chunks.jsonl`, `meta.json`).

### QA (`oracle/qa/`)
- `rag.py`: direct query embedding, top_k retrieval, prompt rendering via `rag_prompt.j2`.
- `arag.py`: agentic retrieval via tools (`search`, `query_rewrite`).

### Tools/config/api
- `tools/search.py`, `tools/rewrite.py`, `tools/runtime.py`.
- `config.py`: `QAConfig` includes `top_k`, `embedding_model` etc.
- `api.py`: chat endpoint currently does not wire all RAG knobs through.
- `bin/build_index.py`: index CLI.
- `configs/*.yaml`: baseline profiles.

### Constraints (must remain true)
- Embeddings local-only.
- No network in unit tests.
- Python 3.12.
- Respect embedding window limit (MiniLM truncation constraints).
- Importable/testable without HTTP layer.

---

## 2 Feature requirements

---

### F1 — Meaningful chunking

#### Goal
Replace blind fixed-character splitting with structure-aware chunking that preserves semantic coherence and respects embedding token limits.

#### Required implementation
- Introduce chunker strategy interface:
  - `FixedChunker` (existing behavior, default-compatible)
  - `StructuralChunker` (**must**)
  - `SemanticChunker` (**should**)
  - `LlmChunker` (**optional**, gated)
- Keep existing `chunk_text` / `chunk_document` imports operational.

#### Clarified behavior
- Chunk size and overlap are token-based by default.
- Structural chunking must prefer paragraph and sentence boundaries.
- No chunk may exceed embedder token budget.
- Overlap is sentence-aware where possible.

#### Config
```yaml
chunking:
  strategy: structural         # fixed | structural | semantic | llm
  target_tokens: 220
  overlap_tokens: 40
  semantic_threshold: 0.55
  llm_model: null
```

#### Acceptance criteria
- `meta.json` stores chunking strategy and parameters.
- Structural output differs from fixed and avoids mid-word/mid-sentence cuts.
- No chunk exceeds configured token window.

#### Tests
- Structural chunk boundary correctness on synthetic multi-paragraph text.
- Token-budget enforcement.
- Semantic split behavior with fake embedder vectors.

---

### F2 — Multi-resolution retrieval (small-to-big)

#### Goal
Retrieve with high precision on small chunks, answer with broader context from parent chunks.

#### Required implementation
- Add chunk hierarchy fields:
  - `chunk_id`, `parent_id`, `level`, `start_char`, `end_char`
- Persist parent payload (e.g., `parents.jsonl`) in index cache.
- Expand child hits to parent context before prompt assembly.
- Deduplicate parents; keep highest child score per parent.

#### Config
```yaml
retrieval:
  multi_resolution: small_to_big   # off | small_to_big | multi_index
  parent_level: section            # section | neighbors
  parent_window: 1
```

#### Acceptance criteria
- Parent expansion works deterministically and without duplicate context blocks.
- Cache metadata documents resolution mode.

#### Tests
- Child→parent resolution integrity.
- Two children of same parent produce one parent context block.

---

### F3 — Retrieval controls & ranking

#### Goal
Make candidate depth, filtering, and ranking quality explicit and tunable.

#### Required implementation
- Pipeline in `RagQA`:
  1. embed query
  2. retrieve `fetch_k`
  3. optional rerank
  4. optional MMR
  5. `min_similarity` filter
  6. cut to `top_k`
- Add reranker abstractions:
  - `Reranker` interface
  - optional `CrossEncoderReranker` (lazy load only)
  - `MmrReranker` (numpy-based)

#### Config
```yaml
retrieval:
  top_k: 6
  fetch_k: 30
  min_similarity: 0.25
  rerank: cross_encoder            # off | cross_encoder
  rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  mmr: false
  mmr_lambda: 0.5
```

#### Acceptance criteria
- Rerank changes ordering for some queries.
- Similarity threshold can produce empty context safely.
- Disabled rerank/MMR does not load extra models.

#### Tests
- MMR unit test with canned embeddings.
- Rerank pipeline ordering test with stub reranker.

---

### F4 — Context budgeting & compression

#### Goal
Control prompt context by token budget and reduce irrelevant text before answer generation.

#### Required implementation
- Add context budget enforcement by tokens.
- Add extractive compression (sentence-level relevance scoring).
- Keep citation handles stable (`[i]` mapping must remain valid).
- Optional LLM compression behind config (disabled by default).

#### Config
```yaml
context:
  budget_tokens: 1500
  compression: extractive          # off | extractive | llm
  keep_sentences: 3
  compress_model: null
```

#### Acceptance criteria
- Rendered context never exceeds budget.
- Extractive mode shortens passages while preserving query-relevant content.

#### Tests
- Tiny-budget truncation test.
- Extractive compression with fake embedder: relevant sentence survives.

---

### F5 — Small model for auxiliary tasks

#### Goal
Route high-volume helper tasks to a smaller model to reduce latency/cost.

#### Required implementation
- Add `aux_model` config with fallback to main `model`.
- Central helper for aux model/client resolution.
- Use aux model for rewrite/compression/chunking-helper tasks.

#### Config
```yaml
aux_model: null   # fallback to `model` if null
```

#### Acceptance criteria
- Aux-enabled requests route auxiliary calls to aux model.
- Final answer model remains unchanged.
- Null aux_model reproduces old behavior.

#### Tests
- Stub client verifies model id routing for aux tasks.

---

### F6 — Query transformation before retrieval

#### Goal
Improve retrieval recall/precision for ambiguous or underspecified queries.

#### Required implementation
- Add transform stage in basic RAG (default off):
  - `none` (baseline exact behavior)
  - `rewrite`
  - `multi`
  - `hyde`
- For `multi`, merge + dedupe candidate sets before rerank/compression.

#### Config
```yaml
query:
  transform: none      # none | rewrite | multi | hyde
  num_queries: 3
```

#### Acceptance criteria
- `none` equals current baseline behavior.
- `multi` expands candidate pool pre-cut.
- Deduplication is deterministic.

#### Tests
- Transform function with stub LLM.
- Multi-query merge dedupe behavior.

---

## 3 Cross-cutting requirements

### 3.1 Config model (`oracle/config.py`)
- Introduce nested models: `chunking`, `retrieval`, `context`, `query`.
- Keep `top_k` and `embedding_model` back-compat aliases.
- Document rebuild-boundary clearly:
  - **requires index rebuild**: `chunking.*`
  - **query-time only**: `retrieval.*`, `context.*`, `query.*`, `aux_model`

### 3.2 Index metadata integrity
- `meta.json` must store chunking and resolution details.
- At runtime, warn if config chunking differs from index metadata.

### 3.3 API profile loading (`oracle/api.py`)
- Add `rag_profile` field to chat request.
- Load `configs/<profile>.yaml` server-side.
- Override only request-level runtime knobs (e.g. model, temperature) as intended.
- Keep API shape stable and minimal.

### 3.4 GUI minimal support (`frontend/`)
- Add RAG profile selector.
- Optional: source panel with title + score + URL.

### 3.5 Token helper (`oracle/retrieval/tokens.py`)
- Provide `count_tokens()` and `truncate_to_tokens()`.
- Must work offline and be dependency-light.
- Use embedding tokenizer where available.

---

## 4 Required configs to add/update (`configs/`)

- `rag.yaml` (baseline control, unchanged semantics)
- `rag_structural.yaml`
- `rag_multi.yaml`
- `rag_rerank.yaml`
- `rag_compress.yaml`
- `rag_full.yaml`

Each profile must be self-contained and runnable by eval scripts.

---

## 5 Evaluation requirements

### 5.1 Dataset
- Replace trivial eval stub with real corpus-grounded QA pairs (15–30 minimum).

### 5.2 Execution
- Run `bin/run_eval.py` for each profile.
- Store outputs in `data/eval/<profile>.json`.

### 5.3 Reporting
- Provide comparison table with:
  - correct / wrong / orthogonal
  - avg token usage
  - retrieval diagnostics where available (e.g., recall@k proxy)

### 5.4 Additional script
- Add `bin/compare_rag.py` (or extend existing eval) to run profile set + print summary table.

---

## 6 Sprint plan (recommended order)

1. Config foundation + tokenizer helper + eval set.
2. Retrieval controls (fetch_k/min_similarity/rerank).
3. Structural chunking + index metadata.
4. Context budget + extractive compression.
5. Small-to-big expansion.
6. Aux model routing.
7. Query transforms (rewrite/multi/hyde).
8. Optional stretch: semantic/LLM chunking, MMR tuning, LLM compression.
9. API profile + GUI selector + source panel.
10. Final evaluation report.

---

## 7 Non-functional requirements

- No network in unit tests.
- Lazy-load optional heavy components.
- Keep import-time side effects minimal.
- Clear warnings instead of silent fallback for index/config mismatches.
- Preserve backward compatibility by default.

---

## 8 Risks and mitigations

- **Cross-encoder model availability** → lazy load + clear error message + fallback to off.
- **Index size growth** → support `--max-docs` demo subset + meta records build scope.
- **Tokenizer mismatch** → central helper + deterministic approximation fallback.
- **Behavior drift in baseline** → locked `rag.yaml` as control profile + regression tests.

---

## 9 Final acceptance checklist

**Status (2026-07-01): all items met.** `SemanticChunker` (F1 "should") also
implemented beyond the minimum scope. Verification: `tests/` 60 passing (no
network); live eval over the 40-doc subset (25 QA pairs) across baseline + 6
advanced profiles, 0 errors.

- [x] Baseline behavior unchanged with baseline config.
      — `configs/rag.yaml` locked as control; `RagQA._is_baseline_query()` routes
      to the byte-identical legacy path; back-compat regression tests; eval 25/25.
- [x] Structural chunking implemented and indexed.
      — `StructuralChunker` (token-budget, paragraph/sentence-aware); `build_index`
      records strategy + params in `meta.json`.
- [x] Retrieval controls active and tested.
      — `fetch_k` → rerank → MMR → `min_similarity` → `top_k` in `RagQA._rank`;
      MMR/rerank/threshold unit tests.
- [x] Token-budgeted context assembly implemented.
      — `context.assemble_context` (never exceeds budget); tiny-budget test.
- [x] Extractive compression implemented.
      — `context.compress_extractive` (query-relevant sentences, stable `[i]`
      citations); relevant-sentence test.
- [x] At least 3 advanced profiles evaluated against baseline.
      — 6 evaluated (`rag_structural`, `rag_semantic`, `rag_multi`, `rag_rerank`,
      `rag_compress`, `rag_full`) via `bin/compare_rag.py`.
- [x] No-network unit tests added for all new core logic.
      — `tests/test_rag_features.py` (chunking incl. semantic, tokens, rerank/MMR,
      parent expansion, budgeting/compression, transforms, aux routing, config).
- [x] Demo-ready comparison output produced.
      — `bin/compare_rag.py` prints the verdict/token-delta/retrieval table and
      writes `data/eval/summary.json`.
