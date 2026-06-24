# Agentic RAG

Familiarize with the `rag` system, as designed per `reg-spec.md`.

Change the overall dialog to an agentic RAG system, that is: the agent needs to first reason whether a retrieval step is necessary, before inferring the actual response.


## Technical Considerations

- Since we are in the `huggingface` and `faiss` ecosystem, use the `smolagents` toolkit for a very lightweight implementation.

- Put all tools into a separate `tools` package.

- Provide at least the following tools:
	- Query rewrite: The user query may be only short. Extend the query so that the semantic embedding is more characteristic.
	- Actual search tool: specify a sentence, run the embedding, do the search; should take an argument specifiying the number of hits, or a minimum similarity. Resultset may be empty.

- If a search comes back empty, rephrase the query once and retry; do not try more than two times.

- For each subsequent chat interaction, reason if a new search query is necessary.

- Re-use the general prompt structure from basic `rag`
