"""LLM access: token loading + an OpenAI-compatible client for the proxy.

All LLM calls (answering and judging) are routed to the external proxy via the
HF/OpenAI-compatible chat API. The token is read from the git-ignored
``.llmtoken`` file (preferred) or an environment variable.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = REPO_ROOT / ".llmtoken"
_TOKEN_ENV_VARS = ("LLM_TOKEN", "OPENAI_API_KEY", "HF_TOKEN")


@dataclass
class UsageMetrics:
    """Token counts and wall-clock time for a single QA LLM call (judge excluded)."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float


def load_token() -> str:
    """Return the API token from ``.llmtoken`` or a known env var."""
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    for name in _TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value.strip()
    raise RuntimeError(
        f"No API token found. Put it in {TOKEN_FILE} "
        f"or set one of {', '.join(_TOKEN_ENV_VARS)}."
    )


def make_client(endpoint: str) -> OpenAI:
    """Build an OpenAI client pointed at the proxy `endpoint`."""
    return OpenAI(base_url=endpoint, api_key=load_token(), timeout=60.0, max_retries=2)


def chat(client: OpenAI, model: str, messages: list[dict], temperature: float = 0.0,
         **kwargs) -> str:
    """Single chat-completion call; returns the assistant message content.

    Used by the judge — intentionally does NOT capture metrics.
    """
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kwargs
    )
    return resp.choices[0].message.content or ""


def chat_with_metrics(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    **kwargs,
) -> tuple[str, UsageMetrics]:
    """Chat-completion that also returns token counts and wall-clock time.

    Use this for QA system calls. Do NOT use for judge calls.
    """
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kwargs
    )
    elapsed = time.perf_counter() - t0
    usage = resp.usage
    metrics = UsageMetrics(
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        elapsed_seconds=round(elapsed, 3),
    )
    return resp.choices[0].message.content or "", metrics
