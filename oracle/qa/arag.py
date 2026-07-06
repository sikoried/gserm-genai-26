"""`a-rag` (agentic RAG): split-model architecture with local router + cloud answerer.

A small local LLM (the "router") classifies whether the question needs retrieval
from the wiki-10k knowledge base or can be answered from world knowledge alone.
If retrieval is needed, the router may also rewrite the search query.  The large
cloud LLM then generates the final answer — with or without retrieved context.

When the local router is unavailable the system falls back to always retrieving
(the safer default).
"""
from __future__ import annotations

import json
import logging
import re
import time

from .base import Answer, QASystem
from .rag import DEFAULT_SYSTEM_PROMPT as RAG_SYSTEM_PROMPT, _context_template, _first_user_message
from .world import DEFAULT_SYSTEM_PROMPT as WORLD_SYSTEM_PROMPT
from ..config import QAConfig
from ..llm import UsageMetrics, chat_with_metrics, make_client
from ..models import reasoning_request_kwargs
from ..retrieval import Embedder, Hit, load_retriever

log = logging.getLogger(__name__)

ROUTER_PROMPT = (
    "You are a routing classifier. You have access to a knowledge base of 10,000 "
    "Wikipedia articles covering diverse topics (people, places, science, history, "
    "technology, sports, culture, etc.).\n\n"
    "Decide whether the following question should be answered using retrieval from "
    "this knowledge base (route: rag) or from general world knowledge (route: world).\n\n"
    "Route to RAG when the question asks about specific facts, people, organizations, "
    "events, technical details, or anything that would benefit from a precise source. "
    "Route to WORLD only for very simple, universally known facts (e.g. 'what is 2+2').\n\n"
    "When in doubt, prefer RAG — retrieving unnecessary context is harmless, but "
    "missing needed context produces wrong answers.\n\n"
    "Question: {question}\n\n"
    "Respond with JSON only. Set search_query to the best search terms for retrieval, "
    "or null if route is world.\n"
    'Example (rag): {{"route": "rag", "search_query": "Marie Curie Nobel Prize chemistry"}}\n'
    'Example (world): {{"route": "world", "search_query": null}}'
)


def _parse_router_response(raw: str) -> dict:
    """Extract {"route": ..., "search_query": ...} from the router's output."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            route = str(obj.get("route", "rag")).strip().lower()
            if route not in ("world", "rag"):
                route = "rag"
            search_query = obj.get("search_query")
            if search_query and str(search_query).lower() in ("null", "none", ""):
                search_query = None
            return {"route": route, "search_query": search_query}
        except json.JSONDecodeError:
            pass
    return {"route": "rag", "search_query": None}


class AgenticRagQA(QASystem):
    _retriever = None
    _embedder = None

    def __init__(self, config: QAConfig):
        super().__init__(config)
        self.cloud_client = make_client(config.endpoint)
        self.rag_system_prompt = config.system_prompt or RAG_SYSTEM_PROMPT
        self.world_system_prompt = WORLD_SYSTEM_PROMPT
        self.top_k = config.top_k

        if AgenticRagQA._retriever is None:
            AgenticRagQA._retriever = load_retriever()
        if AgenticRagQA._embedder is None:
            AgenticRagQA._embedder = Embedder(config.embedding_model)

        self.router_client = None
        self.router_model = config.router_model
        if config.router_endpoint and config.router_model:
            try:
                self.router_client = make_client(config.router_endpoint)
            except Exception as exc:
                log.warning("Could not create router client: %s — will always retrieve", exc)

    def _route(self, question: str) -> dict:
        """Ask the local router whether retrieval is needed. Falls back to RAG."""
        if self.router_client is None:
            return {"route": "rag", "search_query": None, "fallback": True}
        try:
            prompt = ROUTER_PROMPT.format(question=question)
            resp = self.router_client.chat.completions.create(
                model=self.router_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=60.0,
            )
            raw = resp.choices[0].message.content or ""
            result = _parse_router_response(raw)
            result["fallback"] = False
            result["router_tokens"] = resp.usage.total_tokens if resp.usage else 0
            return result
        except Exception as exc:
            log.warning("Router call failed: %s — falling back to RAG", exc)
            return {"route": "rag", "search_query": None, "fallback": True}

    def _retrieve(self, query: str) -> list[Hit]:
        embedding = AgenticRagQA._embedder.encode([query])[0]
        return AgenticRagQA._retriever.search(embedding, self.top_k)

    def _answer_with_context(self, question: str, context_query: str | None,
                             history: list[dict] | None = None) -> Answer:
        """Generate the final answer using the cloud model, with or without retrieval."""
        t0 = time.perf_counter()

        route_result = self._route(question)
        route = route_result["route"]
        search_query = route_result.get("search_query")
        router_tokens = route_result.get("router_tokens", 0)
        is_fallback = route_result.get("fallback", False)

        trace_lines: list[str] = []
        if is_fallback:
            trace_lines.append("Router unavailable — defaulting to RAG retrieval.")
        else:
            trace_lines.append(f"Router decision: {route}")

        if route == "rag":
            query_for_search = search_query or context_query or question
            if search_query:
                trace_lines.append(f"Router rewrote query: {search_query!r}")
            hits = self._retrieve(query_for_search)
            context = _context_template.render(hits=hits)
            system_content = f"{self.rag_system_prompt}\n\nContext:\n{context}"
            trace_lines.append(f"Retrieved {len(hits)} chunks.")
        else:
            system_content = self.world_system_prompt
            trace_lines.append("Skipped retrieval — answering from world knowledge.")

        messages = [{"role": "system", "content": system_content}]
        if history:
            messages.extend(history)
        else:
            messages.append({"role": "user", "content": question})

        content, metrics = chat_with_metrics(
            self.cloud_client, self.config.model, messages,
            temperature=self.config.temperature,
            **reasoning_request_kwargs(self.config.model, self.config.reasoning_effort),
        )

        total_elapsed = round(time.perf_counter() - t0, 3)
        metrics = UsageMetrics(
            prompt_tokens=metrics.prompt_tokens + router_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.total_tokens + router_tokens,
            elapsed_seconds=total_elapsed,
            reasoning_tokens=metrics.reasoning_tokens,
        )

        return Answer(
            content=content.strip(),
            metrics=metrics,
            reasoning="\n".join(trace_lines),
        )

    def answer(self, question: str) -> Answer:
        return self._answer_with_context(question, context_query=None)

    def answer_chat(self, history: list[dict]) -> Answer:
        first_q = _first_user_message(history)
        return self._answer_with_context(first_q, context_query=first_q, history=history)
