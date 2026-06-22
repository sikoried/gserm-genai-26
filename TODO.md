# Oracle — TODO

Working roadmap / backlog. Design decisions live in `CLAUDE.md`.

## Done
- [x] Python 3.12 venv + dependency stack (`requirements.txt`)
- [x] Dataset fetch + explore helper — `bin/fetch_dataset.py` (NeelNanda/wiki-10k)
- [x] Local embeddings verified — sentence-transformers `all-MiniLM-L6-v2`
- [x] FAISS in-memory retrieval verified (`IndexFlatIP`, cosine)
- [x] Eval stub — `eval/qa_pairs.yaml` (trivial placeholder pairs)
- [x] `world` QA system + config loader (`oracle/`) — answers via proxy (default Mistral-Medium)
- [x] LLM-as-a-judge (`openai/gpt-oss-120b`) — verdicts: correct / wrong / orthogonal
- [x] End-to-end eval runner — `bin/run_eval.py` (verified: world = 4/4 correct on the stub)
- [x] Unit tests — `tests/` (config + judge parser, no network)

## Next
- [ ] FastAPI HTTP layer wrapping the QA systems (thin; the library is already testable without it)
- [ ] Retrieval: chunk articles → embed → FAISS in-memory index, behind a swappable `Retriever`
- [ ] `rag` QA type: retrieve-and-answer over the FAISS index
- [ ] Frontend (React)

## Later / Backlog
- [ ] **Vector store: ChromaDB (in-memory `EphemeralClient`)** as an alternative to FAISS —
      adds built-in metadata filtering + optional persistence. Swap behind the `Retriever`
      interface so it's a contained change.
- [ ] Authored query/answer eval pairs grounded in wiki-10k (replace the trivial stub)
- [ ] Concurrency in the eval runner (parallel answer/judge calls) once sets get larger
- [ ] Agentic RAG (`a-rag`) QA type
