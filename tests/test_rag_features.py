"""No-network unit tests for the advanced-RAG core logic (requirements §F1–F6).

Everything here runs offline: chunking and token counting use the deterministic
approximation, rerank/MMR/compression are exercised with canned vectors and stub
callables, and the RagQA pipeline stages are tested on instances built without
loading the index or any model.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from oracle.config import QAConfig, RetrievalConfig, ContextConfig, QueryConfig
from oracle.qa import context as ctxmod
from oracle.qa import query as qmod
from oracle.qa.rag import RagQA
from oracle.retrieval.chunking import (
    FixedChunker, SemanticChunker, StructuralChunker, build_chunker,
)
from oracle.retrieval.index import check_chunking_mismatch
from oracle.retrieval.rerank import CrossEncoderReranker, MmrReranker, mmr_select
from oracle.retrieval.store import Hit
from oracle.retrieval.tokens import count_tokens, truncate_to_tokens


# ---------------------------------------------------------------------------
# Config (§3.1 back-compat)
# ---------------------------------------------------------------------------

def test_legacy_top_k_folds_into_retrieval():
    c = QAConfig(type="rag", top_k=5)
    assert c.retrieval.top_k == 5
    assert c.top_k == 5  # back-compat property


def test_nested_blocks_default_to_baseline():
    c = QAConfig(type="rag")
    assert c.chunking.strategy == "fixed"
    assert c.retrieval.rerank == "off" and not c.retrieval.mmr
    assert c.query.transform == "none"
    assert c.context.compression == "off"
    assert c.aux == c.model  # aux falls back to main model


def test_explicit_retrieval_top_k_wins_over_legacy():
    c = QAConfig(type="rag", top_k=5, retrieval={"top_k": 9})
    assert c.retrieval.top_k == 9


# ---------------------------------------------------------------------------
# Token helper (§3.5)
# ---------------------------------------------------------------------------

def test_count_tokens_offline_and_monotonic():
    assert count_tokens("") == 0
    assert count_tokens("one two three") == 3
    assert count_tokens("a, b.") == 4  # words + punctuation count as tokens


def test_truncate_never_exceeds_budget():
    text = " ".join(f"w{i}" for i in range(50))
    out = truncate_to_tokens(text, 10)
    assert count_tokens(out) <= 10
    assert out and not out.endswith("w")  # cut on a token boundary, no partial word run cut


# ---------------------------------------------------------------------------
# F1 — structural chunking
# ---------------------------------------------------------------------------

_DOC = {
    "id": "1", "title": "T", "url": "u",
    "text": ("Intro sentence one. Intro sentence two.\n"
             "Second paragraph has several sentences. " + ("alpha beta " * 80) + "done."),
}


def test_structural_emits_parent_per_paragraph_and_children():
    children, parents = StructuralChunker(target_tokens=30, overlap_tokens=8).split(_DOC)
    assert len(parents) == 2  # two non-empty lines -> two parents
    assert all(p.level == 1 and p.parent_id is None for p in parents)
    assert all(c.level == 0 and c.parent_id is not None for c in children)
    # every child's parent_id refers to a real parent
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_structural_respects_token_budget():
    children, _ = StructuralChunker(target_tokens=30, overlap_tokens=8).split(_DOC)
    assert children, "expected child chunks"
    assert max(count_tokens(c.text) for c in children) <= 30


def test_structural_differs_from_fixed_and_avoids_midword_cuts():
    structural, _ = StructuralChunker(target_tokens=30, overlap_tokens=8).split(_DOC)
    fixed, _ = FixedChunker(size=120, overlap=20).split(_DOC)
    assert [c.text for c in structural] != [c.text for c in fixed]
    # structural chunks should not end mid-word (they end on token boundaries)
    for c in structural:
        assert not c.text.endswith("alph"), "mid-word cut leaked"


def test_first_short_paragraph_is_single_child_equal_to_parent():
    doc = {"id": "9", "title": "T", "url": "u", "text": "Just one short line."}
    children, parents = StructuralChunker(target_tokens=220, overlap_tokens=40).split(doc)
    assert len(children) == 1 and len(parents) == 1
    assert children[0].text == parents[0].text == "Just one short line."


def test_build_chunker_dispatch():
    assert isinstance(build_chunker(QAConfig(type="rag").chunking), FixedChunker)
    structural = QAConfig(type="rag", chunking={"strategy": "structural"}).chunking
    assert isinstance(build_chunker(structural), StructuralChunker)
    semantic = QAConfig(type="rag", chunking={"strategy": "semantic"}).chunking
    chunker = build_chunker(semantic, embed_fn=lambda xs: np.zeros((len(xs), 2)))
    assert isinstance(chunker, SemanticChunker)


# ---------------------------------------------------------------------------
# F1 — semantic chunking (topical split with a fake embedder, offline)
# ---------------------------------------------------------------------------

def _topic_embed(sentences):
    """Fake normalized embedder: 'apple' sentences -> [1,0], else -> [0,1]."""
    return np.array([[1.0, 0.0] if "apple" in s.lower() else [0.0, 1.0]
                     for s in sentences], dtype="float32")


_TWO_TOPIC_DOC = {
    "id": "7", "title": "T", "url": "u",
    "text": ("Apples are a red fruit. Apple trees blossom in spring. "
             "The Moon orbits the Earth. The Moon has many craters."),
}


def test_semantic_splits_on_topic_change():
    chunker = SemanticChunker(target_tokens=200, overlap_tokens=0,
                              semantic_threshold=0.55, embed_fn=_topic_embed)
    children, parents = chunker.split(_TWO_TOPIC_DOC)
    assert parents == []  # semantic output is flat (no small->big)
    assert len(children) == 2  # one apple chunk, one moon chunk
    assert "Apple" in children[0].text and "Moon" not in children[0].text
    assert "Moon" in children[1].text and "Apple" not in children[1].text


def test_semantic_respects_token_budget():
    # One topic (no semantic breaks) but long -> budget forces multiple children.
    text = "Apple data point alpha beta. " * 30  # ~5 tokens/sentence, 30 sentences
    doc = {"id": "8", "title": "T", "url": "u", "text": text}
    children, _ = SemanticChunker(target_tokens=20, overlap_tokens=0,
                                  embed_fn=_topic_embed).split(doc)
    assert len(children) > 1
    assert max(count_tokens(c.text) for c in children) <= 20


def test_semantic_overlap_carries_context_without_exceeding_budget():
    # Same topic, varied sentence lengths so a budget break leaves room for a
    # short trailing sentence to be carried into the next chunk.
    text = ("Apple alpha beta gamma delta epsilon. Apple two. "
            "Apple three. Apple four.")
    doc = {"id": "8", "title": "T", "url": "u", "text": text}
    no_ovl, _ = SemanticChunker(target_tokens=12, overlap_tokens=0,
                                embed_fn=_topic_embed).split(doc)
    ovl, _ = SemanticChunker(target_tokens=12, overlap_tokens=4,
                             embed_fn=_topic_embed).split(doc)
    assert len(ovl) == len(no_ovl) == 2
    # Overlap pulls the later chunk's start back into the prior chunk's span...
    assert ovl[1].start_char < no_ovl[1].start_char
    assert "Apple two" in ovl[1].text and "Apple two" not in no_ovl[1].text
    # ...but never lets a chunk exceed the token budget.
    assert max(count_tokens(c.text) for c in ovl) <= 12


def test_semantic_without_embedder_degrades_to_structural():
    # No embed_fn -> behaves like StructuralChunker (emits small->big parents).
    chunker = SemanticChunker(target_tokens=30, overlap_tokens=8, embed_fn=None)
    children, parents = chunker.split(_DOC)
    structural_children, structural_parents = StructuralChunker(
        target_tokens=30, overlap_tokens=8).split(_DOC)
    assert [c.text for c in children] == [c.text for c in structural_children]
    assert len(parents) == len(structural_parents) > 0


# ---------------------------------------------------------------------------
# §3.2 — index/config mismatch warning
# ---------------------------------------------------------------------------

def test_chunking_mismatch_detected_and_clean():
    cfg = QAConfig(type="rag", chunking={"strategy": "structural", "target_tokens": 220})
    same = {"chunking": {"strategy": "structural", "target_tokens": 220}, "model": cfg.embedding_model}
    assert check_chunking_mismatch(cfg, same) is None
    drift = {"chunking": {"strategy": "fixed"}, "model": cfg.embedding_model}
    warning = check_chunking_mismatch(cfg, drift)
    assert warning and "strategy" in warning


# ---------------------------------------------------------------------------
# F3 — rerank & MMR
# ---------------------------------------------------------------------------

def _hit(cid, score, idx, text="x"):
    return Hit(text=text, title=cid, url="", score=score, chunk_id=cid, idx=idx)


def test_mmr_select_prefers_diversity_at_low_lambda():
    q = np.array([1.0, 0.0])
    docs = np.array([[1.0, 0.0],   # 0: most relevant
                     [1.0, 0.0],   # 1: identical to 0 (redundant)
                     [0.0, 1.0]])  # 2: orthogonal (diverse)
    order = mmr_select(q, docs, k=3, lambda_=0.2)
    assert order[0] == 0          # relevance picks 0 first
    assert order[1] == 2          # diversity prefers the orthogonal doc next
    assert order == [0, 2, 1]


def test_mmr_reranker_reorders_hits_by_vectors():
    hits = [_hit("a", 0.9, 0), _hit("b", 0.89, 1), _hit("c", 0.2, 2)]
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out = MmrReranker(lambda_=0.2).rerank_vectors(np.array([1.0, 0.0]), hits, vecs, k=3)
    assert [h.chunk_id for h in out] == ["a", "c", "b"]


class _StubReranker(CrossEncoderReranker):
    """A reranker that reverses the candidate order without loading any model."""
    def __init__(self):
        pass

    def rerank(self, query, hits):
        return list(reversed(hits))


def test_rank_pipeline_applies_reranker_then_cut():
    rag = RagQA.__new__(RagQA)
    rag._reranker = _StubReranker()
    rag.r = RetrievalConfig(top_k=2, min_similarity=0.0, mmr=False)
    hits = [_hit("a", 0.3, 0), _hit("b", 0.5, 1), _hit("c", 0.9, 2)]
    out = rag._rank("q", np.array([1.0, 0.0]), hits)
    assert [h.chunk_id for h in out] == ["c", "b"]  # reversed, then cut to top_k=2


def test_min_similarity_can_empty_context_safely():
    rag = RagQA.__new__(RagQA)
    rag._reranker = None
    rag.r = RetrievalConfig(top_k=5, min_similarity=0.95, mmr=False)
    out = rag._rank("q", np.array([1.0, 0.0]), [_hit("a", 0.3, 0), _hit("b", 0.5, 1)])
    assert out == []


# ---------------------------------------------------------------------------
# F2 — small-to-big parent expansion
# ---------------------------------------------------------------------------

def _make_rag_with_parents():
    rag = RagQA.__new__(RagQA)
    rag.r = RetrievalConfig(multi_resolution="small_to_big")
    parent = SimpleNamespace(chunk_id="d#p0", title="T", url="u", text="PARENT TEXT", doc_id="d")
    rag.index = SimpleNamespace(parents={"d#p0": parent})
    return rag


def test_two_children_same_parent_collapse_to_one_block():
    rag = _make_rag_with_parents()
    hits = [Hit("c0", "T", "u", 0.4, chunk_id="d#c0", parent_id="d#p0", doc_id="d"),
            Hit("c1", "T", "u", 0.8, chunk_id="d#c1", parent_id="d#p0", doc_id="d")]
    out = rag._expand_parents(hits)
    assert len(out) == 1
    assert out[0].text == "PARENT TEXT"
    assert out[0].score == 0.8  # highest child score kept for the parent


def test_expand_parents_off_returns_hits_unchanged():
    rag = RagQA.__new__(RagQA)
    rag.r = RetrievalConfig(multi_resolution="off")
    rag.index = SimpleNamespace(parents={})
    hits = [_hit("a", 0.4, 0)]
    assert rag._expand_parents(hits) is hits


# ---------------------------------------------------------------------------
# F4 — context budgeting & extractive compression
# ---------------------------------------------------------------------------

def test_assemble_context_never_exceeds_tiny_budget():
    blocks = [ctxmod.ContextBlock("T1", "word " * 100, "u", 0.9),
              ctxmod.ContextBlock("T2", "word " * 100, "u", 0.8)]
    kept = ctxmod.assemble_context(blocks, budget_tokens=12)
    rendered = ctxmod.render_context(kept)
    assert count_tokens(rendered) <= 12


def test_render_context_numbers_blocks_for_stable_citations():
    blocks = [ctxmod.ContextBlock("A", "x", "u", 0.9),
              ctxmod.ContextBlock("B", "y", "u", 0.8)]
    rendered = ctxmod.render_context(blocks)
    assert "[1] A" in rendered and "[2] B" in rendered


def test_extractive_compression_keeps_relevant_sentence():
    text = "Cats are mammals. The capital of France is Paris. Bananas are yellow."
    query_vec = np.array([1.0, 0.0])

    def fake_embed(sentences):
        # the Paris sentence aligns with the query; others are orthogonal
        return np.array([[1.0, 0.0] if "Paris" in s else [0.0, 1.0] for s in sentences],
                        dtype="float32")

    out = ctxmod.compress_extractive(text, query_vec, fake_embed, keep_sentences=1)
    assert "Paris" in out
    assert "Bananas" not in out


# ---------------------------------------------------------------------------
# F5 — aux model routing
# ---------------------------------------------------------------------------

def test_make_generator_routes_to_given_model():
    seen = {}

    class _FakeCompletions:
        def create(self, model, messages, temperature):
            seen["model"] = model
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok"))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    generate = qmod.make_generator(fake_client, "aux/small-model")
    assert generate("sys", "user") == "ok"
    assert seen["model"] == "aux/small-model"


# ---------------------------------------------------------------------------
# F6 — query transforms & multi-query merge
# ---------------------------------------------------------------------------

def test_transform_none_is_identity():
    assert qmod.transform_queries("q", QueryConfig(transform="none"), None) == ["q"]


def test_transform_rewrite_uses_generate():
    out = qmod.transform_queries("q", QueryConfig(transform="rewrite"),
                                 lambda s, u: "rewritten descriptive query")
    assert out == ["rewritten descriptive query"]


def test_transform_multi_includes_original_and_dedupes():
    def gen(system, user):
        return "q\nVariant one\nvariant one\nVariant two"  # dup + original echoed
    out = qmod.transform_queries("q", QueryConfig(transform="multi", num_queries=3), gen)
    assert out[0] == "q"
    lowered = [o.lower() for o in out]
    assert len(lowered) == len(set(lowered))  # deduped, case-insensitive
    assert "variant one" in lowered and "variant two" in lowered


def test_merge_candidates_dedupes_by_chunk_id_keeping_max_score():
    a = [_hit("x", 0.3, 0), _hit("y", 0.7, 1)]
    b = [_hit("x", 0.9, 0), _hit("z", 0.5, 2)]
    merged = qmod.merge_candidates([a, b])
    by_id = {h.chunk_id: h.score for h in merged}
    assert by_id == {"x": 0.9, "y": 0.7, "z": 0.5}  # x kept at max score
    # deterministic order: score desc, then chunk_id
    assert [h.chunk_id for h in merged] == ["x", "y", "z"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
