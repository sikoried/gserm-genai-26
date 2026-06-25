# Feature: Tools

Familiarize with the agentic-RAG system (`oracle/qa/arag.py`) and the existing
`tools` package (`oracle/tools/`), as designed per `arag-spec.md`.

This feature extends the **agentic-RAG** QA type with a richer toolbox so the agent
can answer **bar / pub quiz questions** — short, factual trivia spanning many
domains (geography, history, science, sports, music, film, dates, simple maths).

## Goal

Give the agent a set of focused tools and let a **small, fast LLM act as the
router** that decides — per step — which tool (if any) to call before producing
the final answer. The large answering model is reserved for synthesis; the small
model keeps tool selection cheap and quick.

## Technical Considerations

- Built on the **`smolagents`** toolkit from Hugging Face (already used for the
  agentic system); each tool is a module-level `@tool` function in the `tools`
  package, sharing state via `tools/runtime.py`.
- **Routing model:** a small LLM (e.g. a Mistral-Small / GPT-OSS class model,
  configurable) is wired as the agent's reasoning/tool-calling model and decides
  which tool to invoke when. The answering model stays configurable separately.
- Each tool is **single-purpose**, has a clear docstring (smolagents exposes it to
  the router), validates its arguments, and returns a short string — never raises
  for "no result"; it returns an explicit empty/`"No result."` message instead.
- Reuse the retry/early-stop discipline from `arag-spec.md`: bound the number of
  tool calls, and don't loop indefinitely.
- Tools must stay **offline-friendly** where possible (local wiki index, local
  compute); any tool that reaches the network is clearly marked and optional.

## Proposed Tools

Existing (keep, already implemented):

- **`search`** — semantic search over the local wiki-10k index; returns top-k
  passages (args: query, `k`, `min_similarity`).
- **`query_rewrite`** — expand a terse quiz question into a fuller sentence so its
  embedding is more characteristic before searching.

New tools to add for quiz answering:

- **`calculator`** — evaluate a safe arithmetic expression (sums, products,
  percentages, powers). Many quiz questions reduce to a small computation.
- **`date_tool`** — day-of-week for a date, date differences, "what year was N
  years before X", current date. Handles the common "on which weekday…" trivia.
- **`unit_convert`** — convert between common units (length, mass, temperature,
  currency-agnostic numeric scaling) for "how many X in Y" questions.
- **`wiki_lookup`** — fetch a specific entity/article by title (vs. fuzzy
  `search`), for "what is the capital of…", "who wrote…" style direct lookups.
- **`list_pick` / `compare`** — given a small set of candidates, return the one
  matching a criterion (largest, earliest, etc.) for "which of these…" questions.
- **`final_answer`** *(smolagents built-in)* — return the concise answer; quiz
  answers should be **short and exact** (a name, number, year), not an essay.

## Out of Scope (for now)

- Live web search and external paid APIs — note them as a backlog extension behind
  the same tool interface.
- Multi-hop planning beyond the bounded step limit.

## Acceptance

- The agent answers a small set of representative bar-quiz questions end-to-end,
  selecting tools via the small router model.
- Each new tool has a unit test (no network) covering a typical and an empty/edge
  case.
- The reasoning trace (`Answer.reasoning`) shows which tools were chosen, so the
  routing decisions are inspectable in the GUI.
