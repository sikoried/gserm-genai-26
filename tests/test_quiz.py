"""Network-free tests for the Quiz feature: dataset, host grading, and API.

The HF dataset, the player QA system, and the host judge are all mocked so nothing
leaves the process.
"""
import pytest
from fastapi.testclient import TestClient

import oracle.api as api
from oracle.api import app
from oracle.quiz import QuizDataset, grade, list_datasets
import oracle.quiz.host as host

client = TestClient(app)

_ROWS = [
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "2 + 2?", "answer": "4"},
    {"question": "Chemical symbol for water?", "answer": "H2O"},
    {"question": "Author of Romeo and Juliet?", "answer": "Shakespeare"},
]


# --- dataset ------------------------------------------------------------------

def test_dataset_item_and_len():
    ds = QuizDataset("mock", rows=_ROWS)
    assert len(ds) == 4
    it = ds.item(0)
    assert it.question == "Capital of France?" and it.reference_answer == "Paris"
    assert it.id == "0"


def test_sample_indices_random_subset_and_all():
    ds = QuizDataset("mock", rows=_ROWS)
    sub = ds.sample_indices(2, seed=1)
    assert len(sub) == 2 and all(0 <= i < 4 for i in sub) and len(set(sub)) == 2
    assert ds.sample_indices(None) == [0, 1, 2, 3]      # "all"
    assert ds.sample_indices(99) == [0, 1, 2, 3]        # count >= size → all


def test_custom_column_mapping():
    ds = QuizDataset("mock", question_col="q", answer_col="a",
                     rows=[{"q": "hi", "a": "yo"}])
    assert ds.item(0).question == "hi" and ds.item(0).reference_answer == "yo"


def test_options_column_is_appended_to_question():
    # Multiple-choice datasets (millionaire) append options to the question.
    ds = QuizDataset("mock", question_col="question", answer_col="normalized_correct_opt",
                     options_col="normalized_options",
                     rows=[{"question": "A local what?",
                            "normalized_options": "A: Yogurt, B: Yam, C: Yokel, D: Yoko Ono",
                            "normalized_correct_opt": "C: Yokel"}])
    it = ds.item(0)
    assert it.question.startswith("A local what?")
    assert "Options: A: Yogurt, B: Yam, C: Yokel, D: Yoko Ono" in it.question
    assert it.reference_answer == "C: Yokel"


def test_list_datasets_defaults_to_millionaire():
    ids = [d["id"] for d in list_datasets()]
    assert "millionaire" in ids and "parrot" not in ids


# --- host grading (binary) ----------------------------------------------------

def test_grade_collapses_to_right_wrong(monkeypatch):
    from oracle.eval.judge import Verdict
    monkeypatch.setattr(host, "_judge", lambda q, r, c, **k: Verdict("correct", "ok", "raw"))
    v = grade("q", "Paris", "Paris", endpoint="x")
    assert v.right is True and v.verdict == "right"

    monkeypatch.setattr(host, "_judge", lambda q, r, c, **k: Verdict("orthogonal", "meh", "raw"))
    v = grade("q", "Paris", "I refuse", endpoint="x")
    assert v.right is False and v.verdict == "wrong"   # non-correct collapses to wrong

    monkeypatch.setattr(host, "_judge", lambda q, r, c, **k: Verdict("error", "bad", "raw"))
    assert grade("q", "r", "c", endpoint="x").verdict == "error"


# --- API ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_dataset(monkeypatch):
    monkeypatch.setattr(api.quizmod, "get_dataset",
                        lambda key, **kw: QuizDataset("mock", rows=_ROWS))


def test_quiz_datasets_endpoint():
    resp = client.get("/api/quiz/datasets")
    assert resp.status_code == 200
    assert any(d["id"] == "millionaire" for d in resp.json())


def test_quiz_start_samples_indices():
    resp = client.post("/api/quiz/start", json={"dataset": "mock", "count": 2, "seed": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 4 and len(body["indices"]) == 2


def test_quiz_start_all():
    resp = client.post("/api/quiz/start", json={"dataset": "mock", "count": None})
    assert resp.json()["indices"] == [0, 1, 2, 3]


def test_quiz_answer_runs_player_and_host(monkeypatch):
    from oracle.llm import UsageMetrics
    from oracle.qa.base import Answer

    class _FakeQA:
        def answer(self, q):
            return Answer(content="Paris", metrics=UsageMetrics(10, 4, 14, 0.5, reasoning_tokens=1))

    monkeypatch.setattr(api, "build_qa_system", lambda cfg: _FakeQA())
    monkeypatch.setattr(api.quizmod, "grade",
                        lambda q, r, c, **k: host.QuizVerdict(True, "right", "matches", "raw"))

    resp = client.post("/api/quiz/answer", json={"dataset": "mock", "index": 0, "mode": "Agentic RAG"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "Capital of France?"
    assert body["reference"] == "Paris"
    assert body["answer"] == "Paris"
    assert body["right"] is True and body["verdict"] == "right"
    # player metrics: input/output/reasoning split (output = completion - reasoning)
    assert body["metrics"]["input_tokens"] == 10
    assert body["metrics"]["output_tokens"] == 3
    assert body["metrics"]["reasoning_tokens"] == 1


def test_quiz_answer_unknown_mode_is_400():
    resp = client.post("/api/quiz/answer", json={"dataset": "mock", "index": 0, "mode": "Nope"})
    assert resp.status_code == 400
