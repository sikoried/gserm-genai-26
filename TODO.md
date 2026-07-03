# Oracle — TODO

Working roadmap / backlog. Design decisions live in `CLAUDE.md`; the advanced-RAG
sprint spec (with final acceptance checklist) is `requirements-rag-simon.md`.

## Done
- [x] Python 3.12 venv + dependency stack (`requirements.txt`)
- [x] Dataset fetch + explore helper — `bin/fetch_dataset.py` (NeelNanda/wiki-10k)
- [x] Local embeddings verified — sentence-transformers `all-MiniLM-L6-v2`
- [x] FAISS in-memory retrieval verified (`IndexFlatIP`, cosine)
- [x] `world` QA system + config loader (`oracle/`) — answers via proxy (default Mistral-Medium)
- [x] LLM-as-a-judge (`openai/gpt-oss-120b`) — verdicts: correct / wrong / orthogonal
- [x] End-to-end eval runner — `bin/run_eval.py`
- [x] FastAPI HTTP layer (`oracle/api.py`) — answer/compare/chat + RAG profile loading
- [x] Retrieval: chunk → embed → FAISS index behind a swappable `Retriever` (`oracle/retrieval/`)
- [x] `rag` QA type: retrieve-and-answer over the FAISS index
- [x] Agentic RAG (`a-rag`) QA type — tool-driven search + query rewrite
- [x] Frontend (React) — chat with mode/model selection, RAG profile selector, source panel
- [x] **Advanced RAG pipeline** (see `requirements-rag-simon.md`, all acceptance criteria met):
      structural/semantic chunking, small→big parent expansion, fetch_k/min_similarity/
      rerank/MMR controls, token-budgeted context + extractive compression, aux model
      routing, query transforms (rewrite/multi/hyde)
- [x] Authored eval set — 25 corpus-grounded QA pairs (`eval/qa_pairs.yaml`)
- [x] Profile comparison — `bin/compare_rag.py` (baseline + 6 advanced profiles,
      results in `data/eval/`, summary table with verdicts/token/retrieval deltas)
- [x] Unit tests — `tests/` (62 passing, no network)

## Later / Backlog
- [ ] **Vector store: ChromaDB (in-memory `EphemeralClient`)** as an alternative to FAISS —
      adds built-in metadata filtering + optional persistence. Swap behind the `Retriever`
      interface so it's a contained change.
- [ ] Eval on the full 10k corpus (current indexes/eval use the 40-doc `--max-docs` demo subset)
- [ ] Concurrency in the eval runner (parallel answer/judge calls) once sets get larger
- [ ] Gated options currently warning + falling back: `chunking.strategy: llm`,
      `context.compression: llm`
