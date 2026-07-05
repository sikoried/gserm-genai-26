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
  **Extended maths:** beyond `+ - * / // % **`, support a curated allow-list of
  functions and constants — `sqrt`, `abs`, `round`, `floor`, `ceil`, `log`,
  `log10`, `ln`, `exp`, trig (`sin`, `cos`, `tan` + inverses, `radians`,
  `degrees`), `factorial`, `gcd`, `lcm`, combinatorics (`comb`/`nCr`, `perm`/`nPr`),
  `min`, `max`, `sum`, and the constants `pi` and `e`. Stay safe: keep the
  AST/allow-list evaluator (named functions from a whitelist only — never `eval`,
  no attribute access, no arbitrary names), and still return an explicit error
  string on bad input rather than raising.
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

### Online tools (network — optional, key-gated)

These reach the public internet, so they are **opt-in**: clearly marked as online,
disabled unless configured, and gated behind an API key / setting so an offline run
never calls out. Each still obeys the tool contract (returns a short string, never
raises on "no result"), and each network round-trip is recorded in the trace.

- **`google_search`** — run a web search for the query and return the **5 best
  results**, each as `title · url · snippet`. Use a search API (e.g. a
  SerpAPI / Programmable Search style backend); the key/endpoint come from config.
  Bound results to 5 and truncate snippets so the router prompt stays small.
- **`youtube`** — find the **5 best matching videos** for the query, then **extract
  the information needed to answer** from them: pull each video's transcript /
  captions (and title + description as fallback) and return a compact,
  answer-oriented digest the synthesizer can read the answer out of — not just
  links. Prefer transcript text; note when captions are unavailable. Cap at 5
  videos and cap transcript length per video so token cost stays bounded.

Because both are token-heavy (search snippets, long transcripts), they must feed
the **token accounting** above and respect the **step budget / timeout**; a single
`youtube` call can dominate cost, so document its expected footprint.

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
  - which **model** produced the step — the **router model** id for a routing /
    tool-selection decision, the **synthesis model** id for the final answer, and
    `—` (none) for a local tool that used no model;
  - the **order/when** (step index, and ideally a timestamp or elapsed offset);
  - a short rendering of the **observation/result**;
  - the **tokens used by that step**, broken down by type (see below); local tools
    (calculator, date, unit-convert) report **0 tokens** explicitly.
- **Token types — distinguish three, not two.** Every LLM-backed step and the
  per-question total must separate:
  - **input tokens** — the prompt sent to the model;
  - **output tokens** — the visible answer/decision tokens generated;
  - **reasoning tokens** — the hidden chain-of-thought tokens a reasoning model
    spends (a *subset* of generation, already surfaced as
    `UsageMetrics.reasoning_tokens`). Report it as its own column so a reasoning
    model's cost is visible and not conflated with output.
  A local tool has input = output = reasoning = 0.
- **Per question (aggregate)**, display the **total tokens** to answer it, split
  into **input / output / reasoning** and attributed across **router**, **tools**,
  and **synthesis**, plus total wall-clock time. This reuses the existing
  `UsageMetrics` model (which already carries `reasoning_tokens`) and the project's
  "judge excluded" accounting rule.
- **Model names in the GUI.** Surface, per answer, both **which model routed** (the
  local router model id) and **which model composed the final answer** (the
  synthesis model id) — e.g. in the trace summary line and/or the per-step "model"
  column — so the user can see the small-router / large-synthesizer division of
  labour at a glance.
- **Transport:** extend the QA `Answer` / API `ChatResponse` to carry the
  structured trace and the token breakdown (not just a string), including the model
  id per step and the input/output/reasoning split, so the frontend can render a
  per-step table and a per-question total.
- **GUI:** in `ChatPage`, the existing "Reasoning" `<details>` becomes a step list
  (tool · args · model · result · input/output/reasoning tokens) with a summary
  line showing the router & synthesis model names, total tokens (by type), and time
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

## Multi-hop planning (exploratory)

Some quiz questions need **several dependent hops** — e.g. "In which country was the
director of the highest-grossing 1997 film born?" (find film → find director → find
birthplace → find country). Today's loop is *reactive*: the router picks one tool at
a time under a flat step budget, which makes deep chains fragile. Ideas for going
beyond the bounded limit — captured for discussion, **not committed**:

- **Plan-then-execute.** Before tool calls, ask the router (or the large model once)
  to emit an explicit ordered **plan** of sub-questions. Execute each hop, feeding
  the previous hop's answer into the next. The step budget then bounds *plans*, and
  each hop can carry its own small sub-budget.
- **Sub-goal decomposition with a scratchpad.** Maintain a running "known facts"
  memory: each hop writes its result as a named fact (`director = …`) that later
  hops and the synthesizer read. This makes dependencies explicit and lets the
  loop-guard reason about *progress* (new facts learned) rather than just repeated
  calls.
- **Budget as depth, not count.** Replace the flat `max_steps` with a **hop-depth**
  limit plus a per-hop step limit (e.g. depth ≤ 3, ≤ 3 tool calls per hop), so a
  legitimately deep question isn't starved by a single global cap while still
  terminating.
- **Recursive sub-agents.** Spawn a bounded child agent per sub-question (its own
  tools, trace, and budget); the parent composes their results. Traces nest, and
  token accounting rolls the children's cost up into the parent total.
- **Verification / backtracking hop.** After a candidate answer, allow one optional
  "check" hop (re-query to confirm), and on contradiction backtrack to an earlier
  fact instead of failing — bounded so it can't loop.
- **Guardrails carry over.** Whatever the shape, the **always-terminating contract**
  holds: total depth × per-hop budget and the wall-clock timeout bound the whole
  tree, every branch ends in a single synthesis, and the trace records the plan, the
  hops taken, and the stop reason.

## Out of Scope (for now)

- **Unauthenticated / uncapped** external calls. Live web search and YouTube are now
  in scope (above) but only **key-gated and disabled by default**; running them
  without a configured key, or without the per-tool result/length caps, stays out
  of scope.
- Other external paid APIs beyond the two documented tools.
- Full multi-hop planning beyond the bounded step limit — see **Multi-hop planning
  (exploratory)** below for how it *could* work; it is not committed for this
  iteration.

## Acceptance

- The agent answers a small set of representative bar-quiz questions end-to-end,
  selecting tools via the **local small router model** (no proxy call for routing).
- The router model loads locally within the stated memory ceiling and is shared
  across requests; its id/device are configurable with a documented default.
- Each new tool has a unit test covering a typical and an empty/edge case; the
  offline tools test without network, and the online tools (`google_search`,
  `youtube`) test against a **mocked** HTTP client so the suite stays network-free.
- The extended `calculator` evaluates the added functions/constants and still
  rejects unsafe input (no `eval`, no non-whitelisted names).
- The GUI shows a **structured trace** — tool, arguments, **the model that produced
  each step**, result — plus **token usage per tool call and a per-question total**
  broken into **input / output / reasoning** and attributed across router / tools /
  synthesis, the **router and synthesis model names**, and the total time.
- Every question **terminates** in a single final answer; tests cover the loop
  guards (step-budget exhaustion, repeated call, timeout) forcing a conclusion
  rather than looping or erroring, and the trace records the termination reason.
