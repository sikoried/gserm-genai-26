# Feature: Tools

Familiarize with the agentic-RAG system (`oracle/qa/arag.py`) and the existing
`tools` package (`oracle/tools/`), as designed per `arag-spec.md`.

This feature extends the **agentic-RAG** QA type with a richer toolbox so the agent
can answer **bar / pub quiz questions** — short, factual trivia spanning many
domains (geography, history, science, sports, music, film, dates, simple maths).

> **Status: implemented.** Everything below is built. Where the delivered design
> deviates from the original intent it is called out inline (e.g. the online tools
> are keyless open-source rather than key-gated, and the local router drives a
> custom bounded orchestrator rather than smolagents' `ToolCallingAgent`). See
> **Configuration & GUI** and **Implementation notes** at the end for the as-built
> details (module map, config keys, defaults).

## Goal

Give the agent a set of focused tools and let a **small, fast LLM act as the
router** that decides — per step — which tool (if any) to call before producing
the final answer. The large answering model is reserved for synthesis; the small
model keeps tool selection cheap and quick.

## Technical Considerations

- Built on the **`smolagents`** toolkit from Hugging Face: each tool is a plain,
  unit-testable function wrapped as a smolagents `Tool` in `oracle/tools/__init__.py`
  (exposing its name / description / arg schema to the router), sharing state via
  `tools/runtime.py`. *As built,* the tools run under a **custom bounded orchestrator**
  (`oracle/agent/loop.py`) rather than smolagents' `ToolCallingAgent`, because the
  termination guarantees, per-step token attribution and proxy synthesis need
  explicit control the opaque agent loop doesn't expose.
- **Routing model:** a small LLM, **run locally**, decides which tool to invoke
  when; the large answering model stays configurable and separate and is used only
  for the final synthesis. See **Routing Model** below.
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
- **Final answer** — *as built,* not a tool but a single **proxy synthesis** step
  over the gathered evidence (see Termination). Quiz answers should be **short and
  exact** (a name, number, year), not an essay — the synthesis prompt enforces this.

### Online tools (network — enabled by default)

These reach the public internet. They use **cost-free, keyless open-source methods**
(no API key, no paid service) and are **enabled by default** (`enable_online_tools`
defaults to **true**) so current-events questions work out of the box. They remain
individually disableable via the setting for a deliberately offline run. Each obeys
the tool contract (returns a short string, never raises on "no result" or a network
error), and each round-trip is recorded in the trace; the backend HTTP call is
isolated in an injectable function so tests mock it and stay network-free.

- **Offline resilience — must never crash.** When there is **no internet
  connection** (DNS failure, timeout, connection refused), the online tools must
  **fail gracefully**: catch the network error and return a short, clear message
  such as `"No internet connection — web search unavailable."` (rather than raising),
  so the agent can still finish via the local tools / its own knowledge. The message
  is a normal tool observation, so it flows into the trace and lets the synthesizer
  **tell the user** that the web/YouTube lookup could not be performed instead of
  producing a confusing or empty answer. A lightweight connectivity check (or simply
  treating the first failed round-trip as "offline") is acceptable; the requirement
  is: **inform the user, do not crash.**

- **`google_search`** — web search returning the **5 best results**, each as
  `title · url · snippet`. *As built,* backed by **DuckDuckGo** via the cost-free
  `ddgs` library (no key). Results are capped at 5 and snippets truncated so the
  router prompt stays small. (`oracle/tools/websearch.py`.)
- **`youtube`** — find the **5 best matching videos** and **extract the information
  needed to answer** from them: pull each video's transcript/captions (title +
  description as fallback) and return a compact, answer-oriented digest the
  synthesizer can read the answer out of — not just links. *As built,* video search
  uses **`yt-dlp`** (`ytsearchN:`, metadata only — nothing is downloaded) and
  transcripts use **`youtube-transcript-api`**; both are keyless/open-source.
  Capped at 5 videos with a per-video transcript-length cap so token cost stays
  bounded. (`oracle/tools/youtube.py`.)

Both are token-light for the tools themselves (no LLM call → 0 tokens), but they
feed large text into the synthesizer, so they respect the **step budget / timeout**
and their cost shows up in the synthesis step of the **token accounting** above.

## Routing Model (local, small LLM)

The tool router must run **locally** — no proxy call for the routing decision — so
selection stays cheap and private, while the large proxy model is kept for final
answer synthesis. *As built* (`oracle/agent/router.py`): a small Hugging Face
instruct model is loaded in-process via `transformers` and prompted to emit a
**JSON decision** (`{"tool": …, "arguments": …}` or `{"finished": true}`), which the
orchestrator executes — a structured-JSON router rather than smolagents'
`TransformersModel` tool-calling, chosen for deterministic control and clean token
counts. The router prompt lists the active tools' descriptions plus explicit
guidance on when to prefer `google_search` (current/after-cutoff facts) or `youtube`
(video-content questions) vs. the local `search`/`wiki_lookup`.

- **Model choice:** any Hugging Face instruct model that **fits in local memory**.
  Default to **`Qwen/Qwen2.5-1.5B-Instruct`** (~3 GB fp16, ~1 GB 4-bit; strong
  tool-calling for its size). Documented fallbacks — `HuggingFaceTB/SmolLM2-1.7B-Instruct`
  and `meta-llama/Llama-3.2-1B-Instruct` — and a step-up option
  (`Qwen/Qwen2.5-3B-Instruct`) for machines with more memory (a good lever if the
  1.5B model misroutes).
- **Configurable:** `router_model` and `router_device` (default: auto — MPS on Apple
  Silicon, else CUDA/CPU, consistent with `oracle/retrieval`).
- **Loaded once** per process and shared across router **and** planner instances via
  a `(model_id, device)` weight cache; download is cached locally and a one-time
  fetch is allowed on first run, mirroring the `Embedder` policy.
- **Division of labour:** the local router selects tools and drives the loop; the
  large model is invoked only for final synthesis of the gathered evidence. Both
  contribute to the token accounting below.
- **Footprint guardrails:** a memory ceiling (`router_max_gb`, default 6 GB); the
  router refuses a known model whose estimated footprint exceeds it, so a too-large
  model is caught early rather than OOM-ing mid-run.

### RAG-first

The agent must **always consult the local RAG index first**, and reach for other
tools only when that isn't good enough — so answers stay grounded in the corpus and
the cheaper/offline path is tried before the web.

- **Forced first step.** Before the router picks anything, the orchestrator runs one
  local `search` on the question and adds its result to the observations (recorded as
  a normal `tool` step in the trace). *As built,* controlled by `rag_first` (default
  **true**); a no-op if the toolset has no `search`. In multi-hop, each hop searches
  the index first.
- **Escalate only if needed.** The router then sees the RAG result and is instructed
  to **finish immediately if it already answers the question**, and to call another
  tool (`google_search`, `youtube`, calculator, …) **only** if the local result is
  missing, irrelevant, or insufficient. This keeps the local index as the default
  source and the other tools as fallbacks.

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
  for the whole answer. Keep it collapsible. *As built,* rendered by a dedicated
  `TraceView` component.
- **Progress feedback:** because agentic runs (local router + tools + synthesis) can
  take a while, while a request is in flight the GUI shows a **live elapsed-seconds
  counter and an animated progress bar** so a long run doesn't look like a crash.

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
- **Always-terminating contract:** every answer path ends in exactly one concluding
  step — *as built,* a single **proxy synthesis** over the gathered evidence
  (rather than a smolagents `final_answer` tool) — reached either by the router
  deciding it's done or by the budget/loop guards forcing it. The trace records
  **why** it stopped (done · step-budget · timeout · loop-guard) so the termination
  reason is visible to the user.

## Multi-hop planning

Some quiz questions need **several dependent hops** — e.g. "In which country was the
director of the highest-grossing 1997 film born?" (find film → find director → find
birthplace → find country). The plain loop is *reactive*: the router picks one tool
at a time under a flat step budget, which makes deep chains fragile.

**Implemented (`oracle/agent/planner.py` + `oracle/agent/multihop.py`) —
plan-then-execute with a facts scratchpad.** It is **on by default** (`multi_hop`
defaults to **true**) and toggled **per question from the chat** (see Configuration
& GUI):

- **Plan.** A `LocalPlanner` (reusing the local router model) emits an ordered list
  of **sub-questions** (JSON), bounded by `max_hops`.
- **Execute per hop with a scratchpad.** Each sub-question is answered by the bounded
  `run_agent` loop; the answers of earlier hops are threaded forward as **known
  facts** in the next hop's context, making dependencies explicit.
- **Compose.** A final synthesis composes the hop answers into the answer to the
  original question. A single-hop plan skips the extra compose and reproduces the
  plain, non-planning behaviour exactly.
- **Budget as depth, not just count.** Depth is bounded by `max_hops` and each hop
  carries the loop's own step budget / timeout, so the whole tree always terminates.
- **Nested trace.** The trace shows a `planner` step, then every hop's steps tagged
  with their hop index, then the final `synthesis`; token accounting rolls the hops'
  cost up into the per-question total (with a dedicated `planner` attribution).

Ideas **not** taken this iteration, kept for later: recursive bounded **sub-agents**
per sub-question (nested child traces), and a **verification / backtracking hop**
(re-query to confirm a candidate; on contradiction, backtrack to an earlier fact) —
both bounded so they can't loop.

## Configuration & GUI

As-built configuration (`oracle/config.py`, per QA-system config / `QAConfig`):

- **Router:** `router_model` (default `Qwen/Qwen2.5-1.5B-Instruct`), `router_device`
  (default auto), `router_max_gb` (default 6).
- **Termination:** `max_steps` (default 6), `agent_timeout_seconds` (default 120).
- **Online tools:** `enable_online_tools` (default **true**).
- **Multi-hop:** `multi_hop` (default **true**), `max_hops` (default 3).
- **RAG-first:** `rag_first` (default **true**) — force a local index search before
  other tools.

GUI placement:

- **Multi-hop planning** is a **per-question toggle in the chat itself** (next to the
  input / mode selector, not buried in Settings), so the user chooses hop-by-hop
  whether to plan. It is **on by default** and shown **only in `Agentic RAG` mode**
  (hidden/disabled for World and RAG, where it has no effect). The toggle carries a
  short **explanation** (tooltip / info affordance) making the difference clear:
  - **ON** — the question is broken into a chain of sub-questions, each answered in
    turn with the previous answer fed into the next; best for questions that depend
    on an intermediate fact (e.g. *"In which country was the director of the
    highest-grossing 1997 film born?"*).
  - **OFF (single-hop)** — the question is answered in one pass with direct tool
    calls; faster, and best for simple, self-contained questions (e.g. *"What is the
    capital of France?"*).
- **Web + YouTube tools** stay a setting (`enable_online_tools`), now **on by
  default**; expose it wherever settings live so a user can turn the network off for
  a deliberately offline run.
- `/api/chat` carries both flags per request and threads them into the config; since
  both now default to **on**, current-events and multi-step questions work without
  the user changing anything.

## Out of Scope (for now)

- **Uncapped** external calls or **paid** search/video APIs — the online tools are
  keyless/open-source and capped (5 results, truncated text); anything beyond that
  (paid APIs, uncapped fetches) stays out of scope.
- Recursive sub-agents and a verification/backtracking hop for multi-hop (see that
  section) — deferred to a later iteration.

## Implementation notes

Module map of the as-built feature:

- `oracle/tools/` — tool logic as plain functions (`calculator`, `datetool`,
  `convert`, `pick`, `wiki_lookup`, `search`, `rewrite`, `websearch`, `youtube`),
  wrapped as smolagents `Tool`s in `__init__.py`; `runtime.py` holds shared
  retriever/embedder/client and the per-tool token tally.
- `oracle/agent/` — `router.py` (local JSON router + weight cache + memory
  guardrail), `loop.py` (bounded, always-terminating orchestrator + `SynthesisResult`),
  `planner.py` + `multihop.py` (multi-hop), `trace.py` (structured trace + token
  totals by attribution and by input/output/reasoning type).
- `oracle/qa/arag.py` — composes the active toolset, router, and (optional) planner,
  and provides the proxy synthesis.
- `oracle/api.py` — `/api/chat` carries `enable_online_tools` / `multi_hop` in and the
  structured `trace` + `usage` (token breakdown + model names) out.
- `frontend/src/components/TraceView.jsx` + `pages/ChatPage.jsx` — trace rendering,
  the per-question multi-hop toggle in the chat (Agentic RAG only), the web/YouTube
  setting, and the in-flight progress indicator.
- Tests: `tests/test_tools.py`, `test_online_tools.py` (mocked HTTP),
  `test_agent_loop.py` (guards + token accounting), `test_multihop.py`,
  `test_api.py`.

Deviations from the original intent, for the record: online tools are **keyless
open-source** (DuckDuckGo / yt-dlp / youtube-transcript-api) gated by a setting
rather than an API key; the tools are smolagents `Tool`s but run under a **custom
bounded orchestrator** rather than `ToolCallingAgent`; and the concluding step is a
**proxy synthesis**, not a smolagents `final_answer`.

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
- **Online tools are on by default** — a current-events question (e.g. "Who is the
  current CEO of OpenAI?") is answered without the user changing any setting.
- **No internet does not crash.** With connectivity down, the online tools return a
  clear "no internet / web unavailable" observation (not an exception), the run still
  terminates, and the final answer tells the user the web/YouTube lookup could not be
  performed. A test simulates a network failure and asserts graceful handling.
- **Multi-hop is a per-question chat toggle**, on by default, shown **only in
  `Agentic RAG` mode**, with a tooltip/info affordance explaining the ON vs OFF
  behaviour; a test asserts the chosen value is threaded through `/api/chat` per
  request.
- **RAG-first**: the local index is searched before any other tool (a forced
  `search` step appears first in the trace); the router only escalates to other
  tools when the local result is insufficient. A test asserts the forced search runs
  even when the router would finish immediately.
