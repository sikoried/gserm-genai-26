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


def test_chat_agentic_carries_structured_trace(monkeypatch):
    # The a-rag path attaches an AgentTrace; the API must surface it as `trace` + `usage`.
    from oracle.llm import UsageMetrics
    from oracle.qa.base import Answer
    from oracle.agent.trace import AgentTrace, TraceStep

    trace = AgentTrace(stop_reason="done", elapsed_seconds=0.3, steps=[
        TraceStep(0, "router", "calculator", {"expression": "2+2"}, "call calculator", 5, 2),
        TraceStep(1, "tool", "calculator", {"expression": "2+2"}, "4", 0, 0, 0.001),
        TraceStep(2, "synthesis", None, None, "The answer is 4.", 11, 4, 0.2),
    ])

    class _FakeARag:
        def answer_chat(self, conversation):
            return Answer(content="The answer is 4.",
                          metrics=UsageMetrics(16, 6, 22, 0.3),
                          reasoning=trace.as_text(), trace=trace)

    monkeypatch.setattr(api, "build_qa_system", lambda cfg: _FakeARag())
    resp = client.post("/api/chat", json={"question": "2+2?", "mode": "Agentic RAG"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The answer is 4."
    # structured trace present, with the tool step visible
    steps = body["trace"]["steps"]
    assert [s["kind"] for s in steps] == ["router", "tool", "synthesis"]
    assert steps[1]["tool"] == "calculator" and steps[1]["total_tokens"] == 0
    # per-question token breakdown
    assert body["usage"]["total_tokens"] == 22
    assert body["usage"]["tools"]["total"] == 0
    assert body["trace"]["stop_reason"] == "done"


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
