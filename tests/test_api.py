"""Network-free tests for the FastAPI app — the LLM call is mocked.

Covers the /api/compare contract: one result per entry (in order, duplicates
kept), failures surfaced via `error`, and per-entry reasoning threaded through.
"""
import pytest
from fastapi.testclient import TestClient

import oracle.api as api
from oracle.api import AnswerResponse, app

client = TestClient(app)


def _fake_answer_one(question, model, endpoint, temperature, reasoning_effort=None):
    """Stand-in for the real LLM call: succeeds unless the model name has 'bad'."""
    if "bad" in model:
        raise RuntimeError(f"model {model} not available")
    return AnswerResponse(
        model=model,
        reasoning_effort=reasoning_effort,
        answer=f"answer from {model} (effort={reasoning_effort})",
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=3 if reasoning_effort else 0,
        total_tokens=15,
        elapsed_seconds=0.1,
    )


@pytest.fixture(autouse=True)
def _patch_answer(monkeypatch):
    monkeypatch.setattr(api, "_answer_one", _fake_answer_one)


def _post(*entries):
    return client.post("/api/compare", json={"question": "q", "entries": list(entries)})


def test_compare_returns_one_entry_per_input_in_order():
    resp = _post({"model": "m/one"}, {"model": "m/two"}, {"model": "m/three"})
    assert resp.status_code == 200
    data = resp.json()
    assert [r["model"] for r in data] == ["m/one", "m/two", "m/three"]
    assert all(r["error"] is None for r in data)


def test_compare_surfaces_errors_without_dropping():
    resp = _post({"model": "m/one"}, {"model": "m/bad"}, {"model": "m/three"})
    assert resp.status_code == 200
    data = resp.json()
    assert [r["model"] for r in data] == ["m/one", "m/bad", "m/three"]  # order kept, none dropped
    assert data[1]["error"] is not None
    assert data[0]["error"] is None


def test_compare_same_model_with_and_without_reasoning():
    # The new capability: one model can appear twice with different settings.
    resp = _post({"model": "m/one"}, {"model": "m/one", "reasoning_effort": "high"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["model"] == data[1]["model"] == "m/one"
    assert data[0]["reasoning_effort"] is None and data[0]["reasoning_tokens"] == 0
    assert data[1]["reasoning_effort"] == "high" and data[1]["reasoning_tokens"] == 3
    assert "effort=high" in data[1]["answer"]


def test_compare_all_failed_still_returns_entries():
    resp = _post({"model": "bad/x"}, {"model": "bad/y"})
    assert resp.status_code == 200
    data = resp.json()
    assert [r["model"] for r in data] == ["bad/x", "bad/y"]
    assert all(r["error"] is not None for r in data)


def test_compare_empty_entries_is_400():
    resp = client.post("/api/compare", json={"question": "q", "entries": []})
    assert resp.status_code == 400


def test_answer_happy_path():
    resp = client.post("/api/answer", json={"question": "q", "model": "m/one"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "m/one"
    assert data["error"] is None
    assert data["total_tokens"] == 15


def test_chat_world_mode(monkeypatch):
    from oracle.llm import UsageMetrics
    from oracle.qa.base import Answer

    class _FakeQA:
        def answer(self, q):
            return Answer(content=f"echo: {q}", metrics=UsageMetrics(1, 1, 2, 0.1))

    monkeypatch.setattr(api, "build_qa_system", lambda cfg: _FakeQA())
    resp = client.post("/api/chat", json={"question": "hi", "mode": "World"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "echo: hi"
    assert body["mode"] == "World"


def test_chat_unknown_mode_is_400():
    resp = client.post("/api/chat", json={"question": "hi", "mode": "Nope"})
    assert resp.status_code == 400


def test_rag_profiles_lists_configs():
    resp = client.get("/api/rag-profiles")
    assert resp.status_code == 200
    profiles = resp.json()
    assert "rag" in profiles and "rag_full" in profiles  # configs/rag*.yaml stems


def test_chat_with_rag_profile_loads_config_and_returns_sources(monkeypatch):
    from oracle.llm import UsageMetrics
    from oracle.qa.base import Answer

    seen = {}

    class _FakeRag:
        def answer_chat(self, conversation):
            return Answer(content="profile answer", metrics=UsageMetrics(1, 1, 2, 0.1))

        def sources(self):
            return [{"title": "Algeria", "score": 0.42, "url": "https://x"}]

    real_loader = api._load_profile_config

    def _spy_loader(profile, model, temperature):
        seen["profile"] = profile
        cfg = real_loader(profile, model, temperature)
        seen["type"] = cfg.type
        return cfg

    monkeypatch.setattr(api, "build_qa_system", lambda cfg: _FakeRag())
    monkeypatch.setattr(api, "_load_profile_config", _spy_loader)
    resp = client.post("/api/chat", json={
        "question": "what is the capital of Algeria?",
        "mode": "RAG", "rag_profile": "rag_rerank",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "profile answer"
    assert body["profile"] == "rag_rerank"
    assert body["sources"][0]["title"] == "Algeria"
    assert seen["profile"] == "rag_rerank" and seen["type"] == "rag"


def test_chat_unknown_rag_profile_is_400():
    resp = client.post("/api/chat", json={
        "question": "q", "mode": "RAG", "rag_profile": "nope_not_real",
    })
    assert resp.status_code == 400


def test_chat_conversation_aware_qa_gets_history_then_question(monkeypatch):
    from oracle.llm import UsageMetrics
    from oracle.qa.base import Answer

    seen = {}

    class _FakeRag:  # has answer_chat -> conversation-aware path
        def answer_chat(self, conversation):
            seen["conversation"] = conversation
            return Answer(content="rag answer", metrics=UsageMetrics(1, 1, 2, 0.1))

    monkeypatch.setattr(api, "build_qa_system", lambda cfg: _FakeRag())
    resp = client.post("/api/chat", json={
        "question": "q2", "mode": "RAG",
        "history": [{"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"}],
    })
    assert resp.status_code == 200
    assert resp.json()["answer"] == "rag answer"
    # history + current question, in order
    assert [m["content"] for m in seen["conversation"]] == ["q1", "a1", "q2"]
