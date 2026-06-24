# Basic RAG System

Revisit the sections `Data` and `Embeddings` from the `CLAUDE.md`.

Draft a plan to implement the basic `rag` system in the backend; make it work in python first, then expose in API and subsequently in the GUI.

The high-level plan is:

- Set up the `sentence-piece`/`faiss-cpu` (use Apple `mps` device) for in-memory similarity.

- Fill the database by indexing the `text` field of the `wiki-10k` dataset.

- Write a new function for RAG conversations; note that this should not end up being agentic.
	- RAG will only be done on first message; subsequent will just extend the context
	- Embed the message directly, without reformulating it.
	- Write a basic RAG template that can be used to fill in the retrieved context; consider Jinja2
	- Limit the context to 10 hits for now but make it configurable.
