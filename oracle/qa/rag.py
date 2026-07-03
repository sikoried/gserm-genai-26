"""`rag` QA system: retrieve from the local wiki-10k index, then answer with context.

With a baseline config it is the original system: one direct query embedding, a
top-k cosine search, and the hits rendered via ``rag_prompt.j2`` — byte-identical
to before, so ``rag.yaml`` stays a faithful control.

When advanced knobs are set it runs the full retrieval pipeline (requirements
§F3/F4/F6):

    transform query → retrieve fetch_k → (rerank) → (MMR) → min_similarity
                     → cut top_k → (small→big parent expansion)
                     → (extractive compression) → token-budgeted context

Each stage is config-gated and off by default, so heavy components (cross-encoder
model, aux LLM calls) are only touched when explicitly enabled.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .base import Answer, QASystem
from .context import (
    assemble_context, blocks_from_hits, compress_blocks,
    compression_fallback_warning, render_context,
)
from .query import make_generator, merge_candidates, transform_queries
from ..config import QAConfig
from ..llm import chat_with_metrics, make_client
from ..models import reasoning_request_kwargs
from ..retrieval import (
    CrossEncoderReranker, Embedder, Hit, MmrReranker, check_chunking_mismatch,
    load_index,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a question-answering assistant. Use only the context provided below to "
    "answer the question. If the answer is not contained in the context, say so briefly "
    "rather than guessing."
)

_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).resolve().parent)),
    autoescape=False, trim_blocks=True, lstrip_blocks=True,
)
_context_template = _env.get_template("rag_prompt.j2")


def _first_user_message(history: list[dict]) -> str:
    return next((m.get("content", "") for m in history if m.get("role") == "user"), "")


def _sources(hits) -> list[dict]:
    return [{"title": h.title, "score": round(float(h.score), 4), "url": h.url}
            for h in hits]


class RagQA(QASystem):
    # Index + embedder are heavy; share them across instances in the process.
    _index = None
    _embedder = None

    def __init__(self, config: QAConfig):
        super().__init__(config)
        self.client = make_client(config.endpoint)
        self.system_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT
        self.r = config.retrieval
        self.q = config.query
        self.ctx = config.context
        self.last_sources: list[dict] = []  # diagnostics for the GUI source panel
        if RagQA._index is None:
            RagQA._index = load_index()
        if RagQA._embedder is None:
            RagQA._embedder = Embedder(config.embedding_model)
        self.index = RagQA._index
        self.retriever = self.index.retriever
        # Warn (don't fail) when the cache was built with different chunking (§3.2),
        # or when a gated-but-unimplemented option would silently do nothing (§7).
        for warning in (check_chunking_mismatch(config, self.index.meta),
                        compression_fallback_warning(self.ctx.compression)):
            if warning:
                import warnings
                warnings.warn(warning, RuntimeWarning, stacklevel=2)
        # Lazy: built only when their stage is enabled.
        self._reranker = (
            CrossEncoderReranker(self.r.rerank_model)
            if self.r.rerank == "cross_encoder" else None
        )

    # -- baseline detection --------------------------------------------------
    def _is_baseline_query(self) -> bool:
        """True when every query-time knob is at its baseline (legacy) setting."""
        return (
            self.q.transform == "none"
            and self.r.rerank == "off"
            and not self.r.mmr
            and self.r.min_similarity <= 0
            and self.r.multi_resolution == "off"
            and self.ctx.compression == "off"
        )

    # -- pipeline stages -----------------------------------------------------
    def _embed(self, text: str):
        return RagQA._embedder.encode([text])[0]

    def _queries(self, query: str) -> list[str]:
        if self.q.transform == "none":
            return [query]
        generate = make_generator(self.client, self.config.aux, self.config.temperature)
        return transform_queries(query, self.q, generate)

    def _candidates(self, queries: list[str]) -> list[Hit]:
        lists = [self.retriever.search(self._embed(q), self.r.fetch_k) for q in queries]
        return merge_candidates(lists) if len(lists) > 1 else lists[0]

    def _rank(self, query: str, query_vec, hits: list[Hit]) -> list[Hit]:
        if self._reranker is not None:
            hits = self._reranker.rerank(query, hits)
        if self.r.mmr and hits:
            idxs = [h.idx for h in hits if h.idx >= 0]
            if len(idxs) == len(hits):
                vecs = self.retriever.vectors(idxs)
                hits = MmrReranker(self.r.mmr_lambda).rerank_vectors(
                    query_vec, hits, vecs, k=len(hits))
        if self.r.min_similarity > 0:
            hits = [h for h in hits if h.score >= self.r.min_similarity]
        return hits[: self.r.top_k]

    def _expand_parents(self, hits: list[Hit]) -> list[Hit]:
        """Small→big: replace each child hit with its parent, deduped (§F2)."""
        if self.r.multi_resolution == "off" or not self.index.parents:
            return hits
        best: dict[str, Hit] = {}
        for h in hits:
            parent = self.index.parents.get(h.parent_id or "")
            if parent is None:
                best.setdefault(h.chunk_id or h.text[:32], h)
                continue
            key = parent.chunk_id
            if key not in best or h.score > best[key].score:
                best[key] = Hit(text=parent.text, title=parent.title, url=parent.url,
                                score=h.score, chunk_id=parent.chunk_id,
                                parent_id=None, doc_id=parent.doc_id, idx=-1)
        return sorted(best.values(), key=lambda h: -h.score)

    def _advanced_context(self, query: str) -> str:
        query_vec = self._embed(query)
        hits = self._candidates(self._queries(query))
        hits = self._rank(query, query_vec, hits)
        hits = self._expand_parents(hits)
        self.last_sources = _sources(hits)
        blocks = blocks_from_hits(hits)
        if self.ctx.compression == "extractive":
            blocks = compress_blocks(blocks, query_vec, RagQA._embedder.encode,
                                     self.ctx.keep_sentences)
        blocks = assemble_context(blocks, self.ctx.budget_tokens)
        return render_context(blocks)

    # -- public retrieval (back-compat) --------------------------------------
    def retrieve(self, query: str) -> list[Hit]:
        """Baseline retrieval: direct embedding, top-k cosine search."""
        return self.retriever.search(self._embed(query), self.r.top_k)

    def _context(self, query: str) -> str:
        if self._is_baseline_query():
            hits = self.retrieve(query)
            self.last_sources = _sources(hits)
            return _context_template.render(hits=hits)
        return self._advanced_context(query)

    def sources(self) -> list[dict]:
        """Title/score/URL of the passages used for the most recent answer."""
        return self.last_sources

    def _system_message(self, query: str) -> dict:
        context = self._context(query)
        return {"role": "system", "content": f"{self.system_prompt}\n\nContext:\n{context}"}

    def _complete(self, messages: list[dict]) -> Answer:
        content, metrics = chat_with_metrics(
            self.client, self.config.model, messages,
            temperature=self.config.temperature,
            **reasoning_request_kwargs(self.config.model, self.config.reasoning_effort),
        )
        return Answer(content=content.strip(), metrics=metrics)

    def answer(self, question: str) -> Answer:
        return self._complete(
            [self._system_message(question), {"role": "user", "content": question}]
        )

    def answer_chat(self, history: list[dict]) -> Answer:
        """Multi-turn: retrieve only on the first user message; later turns extend context."""
        return self._complete([self._system_message(_first_user_message(history)), *history])
