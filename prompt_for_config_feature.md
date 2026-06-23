# Feature: Configurable QA System Prompt

Implement a feature: the QA system prompt should be configurable via YAML config
files instead of being hardcoded in `oracle/qa/world.py`. We want to be able to
keep multiple config files around — same code, different models and/or system
prompts per file — so this change adds the field and a second example config that
overrides it.

## Setup

First, create and switch to a new feature branch:

```bash
git checkout -b feature/system-prompt-config
```

Read `CLAUDE.md`, `oracle/config.py`, `oracle/qa/world.py`, `oracle/qa/base.py`,
`configs/world.yaml`, and `tests/test_config.py` before changing anything, so the
change matches the existing style.

## Requirements

1. Add a `system_prompt` field to the `QAConfig` model in `oracle/config.py`. It
   should default to the current text of the `SYSTEM_PROMPT` constant in
   `oracle/qa/world.py`, so existing behavior is unchanged when a config omits it.
   Keep the default text as a module-level constant (e.g. `DEFAULT_SYSTEM_PROMPT`)
   next to the other `DEFAULT_*` constants rather than inlining a long string in
   the field definition.
2. Update `oracle/qa/world.py` so `WorldQA` reads the prompt from
   `self.config.system_prompt` instead of the local `SYSTEM_PROMPT` constant.
   Remove the now-unused constant from `world.py`.
3. Leave `configs/world.yaml` unchanged — it stays the baseline that relies on the
   default prompt. Instead, create a NEW config file `configs/world_plus.yaml`
   that demonstrates the feature: same `type: world`, but with its own
   `system_prompt:` (use a YAML block scalar `|` for readability) and feel free to
   point `model:` at a different value to illustrate that different config files
   can pair different models with different prompts. Add a short comment at the top
   explaining that this is an example of an overriding config.
4. Extend `tests/test_config.py`: one test that `QAConfig()` has the expected
   default `system_prompt`, and one that `from_yaml` on `configs/world_*.yaml`
   loads its overridden `system_prompt` (and that it differs from the default). No
   network calls in tests.
5. Update the Configuration Schema section in `CLAUDE.md` to document the new
   `system_prompt` field, and mention that multiple config files can coexist
   (e.g. `world.yaml` vs `world_plus.yaml`).

## After Implementing

Run the test suite (`.venv/bin/python -m pytest`, or `.venv\Scripts\python` on
Windows) and confirm everything passes. Then show me a summary of the diff. Do NOT
commit or push — I'll review and commit myself.
