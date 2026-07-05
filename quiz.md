# Feature: Quiz

A new GUI window, **Quiz**, where a **host** (a big model with a question/answer
dataset) quizzes a **player** (any QA mode — Agentic RAG by default) and scores how
well it does. It reuses the existing QA systems (`oracle/qa`, incl. the agentic-RAG
toolbox from `tools.md`), the LLM-as-judge pattern (`oracle/eval/judge.py`), the
token accounting / trace (`oracle/agent/trace.py`, `UsageMetrics`), and the HF
dataset-caching approach (`bin/fetch_dataset.py`, `data/hf-cache`).

## Goal

Let the Oracle user **start a quiz**: the host asks questions from a dataset, the
player answers each one, the host marks each answer **right or wrong** against the
dataset's reference, and the GUI reports the player's **performance** (accuracy,
time, tokens). This turns the platform into a measurable head-to-head between the
different QA approaches.

## Roles

### 1) Host
- A **big model** (via the proxy, from the models in `configs/models.yaml`) that
  **holds a Q&A dataset** and **judges** the player's answers.
- The host is the only side that sees the dataset's **reference answers** — the
  player never receives them (it must answer from its own capabilities).
- **Default dataset: `RedBlock/parrot`** (Hugging Face). Downloaded **once** and
  cached (under `data/hf-cache`, like the wiki-10k corpus) the first time a quiz is
  run; reused thereafter.
- **Pluggable datasets.** The dataset is behind a small loader interface so other
  HF (or local) datasets can be used. A dataset config gives the **dataset id** and
  a **column mapping** (which field is the question, which is the reference answer),
  since schemas differ. The exact `RedBlock/parrot` columns are inspected on first
  download and mapped in config (default mapping documented once known).

### 2) Player
- **Any QA mode can be a player** — `world`, `rag`, or `a-rag`. **Agentic RAG is the
  default.**
- **All settings available for Agentic RAG are available in the Quiz**: model,
  online tools, multi-hop (+ per-question toggle), planner model, `max_steps` /
  `max_hops` / `max_depth`, verify, temperature, etc. — the same controls as the
  Chat page, reused so behaviour is identical.
- The player answers each question through the normal QA pipeline (`build_qa_system`
  + `answer`), producing an answer plus its **metrics and trace** (tools used,
  time, tokens).

## Quiz flow

1. The user configures the quiz (dataset, number of questions, player mode +
   settings) and **starts** it.
2. The host **asks** the player a question from the dataset.
3. The **player answers** (running its QA pipeline; the agentic trace is captured).
4. The host **judges** the answer against the dataset's reference → **right** or
   **wrong** (binary).
5. Repeat for the chosen number of questions; then show a **final summary**.

- **Number of questions** is chosen by the user, with **"All questions in the
  dataset"** as an explicit option. When fewer than all are chosen, the questions
  are **randomly sampled** from the dataset (without replacement). "All" plays every
  question (order does not matter for scoring).

## Judging (right / wrong)

- The host judges each answer with a **big model** using the LLM-as-judge pattern,
  reduced to a **binary verdict**: *right* (the player's answer conveys the correct
  information from the reference) or *wrong* (everything else — contradiction,
  refusal, off-topic, empty). This adapts `oracle/eval/judge.py` (which already
  classifies correct/wrong/orthogonal) by collapsing non-*correct* into *wrong*.
- The judge/host model is **configurable** (default: a strong judge model such as
  the existing `JUDGE_MODEL`, or the selected answering model). Its own time/token
  cost is **not** attributed to the player's statistics (consistent with the
  project's "judge excluded" accounting rule).

## Statistics (player performance)

Shown live during the quiz and summarised at the end. At minimum:

- **Correct** count and **wrong** count (and accuracy = correct / total).
- **Time**: total wall-clock and **per question** (from the player's
  `elapsed_seconds`).
- **Tokens**: **input / output / reasoning**, totalled and **per question**
  (reusing the player's `UsageMetrics` / trace token breakdown; judge tokens
  excluded).
- Per-question rows (question, player answer, reference, verdict, time, tokens) plus
  the agentic **trace** per answer (tools/steps/tokens), like the Chat page.

## Backend / API (proposed)

Reuses existing infra; keeps the backend **stateless** (no server-side session
store — the frontend drives the loop).

- **Dataset loader** (`oracle/quiz/dataset.py`, proposed): load + cache a HF dataset
  once, expose `len`, and return items `{id, question, reference_answer}` for a
  requested index/subset. Default `RedBlock/parrot`; column mapping in config.
- `GET /api/quiz/datasets` *(optional)* — list configured datasets (id, size).
- `POST /api/quiz/start` — body `{dataset?, count | "all"}`; returns the quiz plan:
  `{total, indices: [...]}` (the **randomly sampled** question indices to play; all
  indices when `count = "all"`). References are **not** sent to the client here.
- `POST /api/quiz/answer` — body `{dataset?, index, mode, ...player settings}`;
  server loads that dataset row, runs the **player** on the question, has the
  **host** judge the answer vs the row's reference, and returns
  `{question, answer, reference, verdict, metrics, trace}`. The frontend calls this
  once per index and accumulates statistics.
- Player settings reuse the existing `ChatRequest` fields (model, mode,
  `enable_online_tools`, `multi_hop`, `planner_model`, `max_steps`, `max_hops`,
  `max_depth`, `verify`, temperature). The reference answer stays server-side until
  the answer is graded, then is returned for display.

## GUI

- A new routed page **Quiz** in the nav (next to Chat and Comparison).
- **Setup panel:** dataset selector (default `RedBlock/parrot`), **number of
  questions** (incl. **All**), **player mode** (default Agentic RAG) and the **full
  Agentic-RAG settings** (reused components/controls), and a **Start** button.
- **Play view:** progress (question *i / N*), the current question, the player's
  answer with its **trace + tokens** (reuse `TraceView`), the host **verdict**
  (right/wrong) and the **revealed reference** after grading.
- **Statistics panel:** live correct/wrong/accuracy, total & per-question time,
  tokens (input/output/reasoning) total & per question; a **final summary** at the
  end.
- **Progress feedback** during long agentic answers (the elapsed timer / progress
  bar already used in Chat), since a full quiz can take a while.

## Configuration

Proposed keys (a quiz config / request):

- `dataset` (default `RedBlock/parrot`), plus `question_column` / `answer_column`
  mapping for pluggable datasets.
- `count` — number of questions, or `all`.
- `host_model` / judge model (default a strong judge model).
- All existing Agentic-RAG player settings (as above).

## Out of scope (for now)

- Multi-player / simultaneous comparison of several modes in one quiz run (the
  Comparison page already covers side-by-side single questions; a "quiz compare"
  could be a later extension).
- Persisted leaderboards across sessions (stats are per quiz run).
- Difficulty weighting / adaptive selection (random uniform sampling only).
- Fuzzy/partial credit — grading is strictly binary right/wrong for now.

## Acceptance

- A new **Quiz** page lets the user pick a dataset (default `RedBlock/parrot`,
  downloaded/cached once), a **number of questions** (including **All**; fewer than
  all are **randomly sampled**), and a **player** (default Agentic RAG with all its
  settings), then **Start**.
- The host asks each question, the player answers via its normal pipeline, and the
  host returns a **binary right/wrong** verdict against the dataset reference; the
  reference is revealed only after grading.
- The GUI shows, live and in a final summary: **correct/wrong/accuracy**, **time
  (total + per question)**, and **tokens input/output/reasoning (total + per
  question)**, plus the per-answer agentic trace. Judge cost is excluded from the
  player's stats.
- The dataset layer is **pluggable** (id + column mapping) and tested with a
  **mocked/local** dataset so the suite stays network-free; the player/host wiring
  reuses existing QA + judge components.
