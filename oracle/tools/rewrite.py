"""Query-rewrite tool: expand a short query so its embedding is more characteristic.

This tool calls the LLM, so it records its token usage into the runtime tally
(``runtime.add_tool_tokens``) for the orchestrator to attribute to this step.
"""
from __future__ import annotations

from . import runtime
from ..llm import chat_with_metrics

_SYSTEM = (
    "Rewrite the user's search query into a single, more descriptive sentence that "
    "captures its intent for semantic retrieval. Return only the rewritten query."
)


def query_rewrite(query: str) -> str:
    """Rewrite a short query into a fuller, more descriptive search sentence.

    Use this when a query is terse so its embedding becomes more characteristic for
    retrieval, and to rephrase before retrying a search that returned nothing.

    Args:
        query: The short query to expand.
    """
    client, model = runtime.llm()
    content, metrics = chat_with_metrics(
        client, model,
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query}],
        temperature=0.0,
    )
    runtime.add_tool_tokens(metrics.prompt_tokens, metrics.completion_tokens)
    return content.strip()
