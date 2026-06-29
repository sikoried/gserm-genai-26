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
- **Routing model:** a small LLM, **run locally**, is wired as the agent's
  reasoning/tool-calling model and decides which tool to invoke when. The large
  answering model stays configurable and separate. See **Routing Model** below.
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

## Routing Model (local, small LLM)

The tool router must run **locally** — no proxy call for the routing decision — so
selection stays cheap and private. It is wired into the smolagents agent as the
reasoning/tool-calling model (e.g. via `TransformersModel`, which loads a Hugging
Face model in-process), while the large proxy model is kept for final answer
synthesis.

- **Model choice:** any Hugging Face instruct model that supports tool/function
  calling and **fits in local memory**. Default to **`Qwen/Qwen2.5-1.5B-Instruct`**
  (~3 GB fp16, ~1 GB 4-bit; strong tool-calling for its size). Document at least
  two fallbacks — e.g. `HuggingFaceTB/SmolLM2-1.7B-Instruct` and
  `meta-llama/Llama-3.2-1B-Instruct` — and one step-up option
  (`Qwen/Qwen2.5-3B-Instruct`) for machines with more memory.
- **Configurable:** the router model id, device, and quantization are set in
  config (e.g. new `router_model` / `router_device` keys), defaulting to the MPS
  device on Apple Silicon and CPU otherwise, consistent with `oracle/retrieval`.
- **Loaded once** per process and shared across requests (the model is heavy);
  download is cached locally and a one-time fetch is allowed on first run, mirroring
  the `Embedder` policy.
- **Division of labour:** the local router selects tools and drives the loop; the
  large model is invoked only for final synthesis of the gathered evidence. Both
  contribute to the token accounting below.
- **Footprint guardrails:** the spec should state an approximate memory ceiling and
  refuse / warn if the chosen model would not fit, so a too-large model is caught
  early rather than OOM-ing mid-run.

## Traceability & Token Accounting

The Oracle user must be able to **see what the agent did and what it cost**, per
question. Extend the existing `Answer.reasoning` trace (today a plain string) into
a **structured, per-step trace** surfaced in the GUI:

- **Per step / per tool call**, record and display:
  - which **tool** was called, with its **arguments**;
  - the **order/when** (step index, and ideally a timestamp or elapsed offset);
  - a short rendering of the **observation/result**;
  - the **tokens used by that step** — for LLM-backed steps (the router decision,
    LLM tools like `query_rewrite`, synthesis); local tools (calculator, date,
    unit-convert) report **0 tokens** explicitly.
- **Per question (aggregate)**, display the **total tokens** to answer it, split
  into prompt vs completion and attributed across **router**, **tools**, and
  **synthesis**, plus total wall-clock time. This reuses the existing
  `UsageMetrics` model and the project's "judge excluded" accounting rule.
- **Transport:** extend the QA `Answer` / API `ChatResponse` to carry the
  structured trace and the token breakdown (not just a string), so the frontend can
  render a per-step table and a per-question total.
- **GUI:** in `ChatPage`, the existing "Reasoning" `<details>` becomes a step list
  (tool · args · result · tokens) with a summary line showing total tokens and time
  for the whole answer. Keep it collapsible.

## Termination & Loop Avoidance

The agentic system **must always reach a conclusion** and must not get stuck
repeatedly calling tools. Specify the following guarantees:

- **Hard step budget:** a bounded `max_steps` (carry over the agent's existing
  limit, e.g. 6). On exhaustion the system does **not** error — it forces a final
  synthesis from whatever evidence was gathered (a best-effort answer, or an
  explicit "could not determine" if there is none).
- **Repeat/no-progress detection:** if the same tool is called with the same (or
  near-identical) arguments, or a tool returns the same observation as a prior step,
  the agent must not repeat it — stop or move on. This generalises the existing
  "never search more than twice / rephrase once on empty" rule.
- **Wall-clock timeout:** an overall time budget for answering one question; on
  timeout, force the final-answer path rather than hanging the GUI (consistent with
  the bounded client timeouts already in `arag.py`).
- **Always-terminating contract:** every answer path ends in exactly one
  `final_answer` — by the agent deciding it's done, or by the budget/loop guards
  forcing it. The trace records **why** it stopped (done · step-budget · timeout ·
  loop-guard) so the termination reason is visible to the user.

## Out of Scope (for now)

- Live web search and external paid APIs — note them as a backlog extension behind
  the same tool interface.
- Multi-hop planning beyond the bounded step limit.

## Acceptance

- The agent answers a small set of representative bar-quiz questions end-to-end,
  selecting tools via the **local small router model** (no proxy call for routing).
- The router model loads locally within the stated memory ceiling and is shared
  across requests; its id/device are configurable with a documented default.
- Each new tool has a unit test (no network) covering a typical and an empty/edge
  case.
- The GUI shows a **structured trace** — which tool was called, when, with what
  arguments and result — plus **token usage per tool call and a per-question
  total** (router + tools + synthesis), and the total time.
- Every question **terminates** in a single final answer; tests cover the loop
  guards (step-budget exhaustion, repeated call, timeout) forcing a conclusion
  rather than looping or erroring, and the trace records the termination reason.
