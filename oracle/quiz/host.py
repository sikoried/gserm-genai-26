"""The quiz host: grade a player's answer against the dataset reference.

Grading is **binary** — *right* or *wrong* — done by a strong model via the existing
LLM-as-judge (`oracle/eval/judge.py`), collapsing its correct/wrong/orthogonal
verdict into right (== correct) vs wrong (everything else). The host's own token /
time cost is not captured, so it never counts toward the player's statistics
(consistent with the project's "judge excluded" rule).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..eval.judge import JUDGE_MODEL, judge as _judge

# The host defaults to a strong judge model.
DEFAULT_HOST_MODEL = JUDGE_MODEL


@dataclass
class QuizVerdict:
    right: bool
    verdict: str      # "right" | "wrong" | "error"
    reasoning: str
    raw: str = ""


def grade(question: str, reference: str, candidate: str, *, endpoint: str,
          model: str = DEFAULT_HOST_MODEL, client=None) -> QuizVerdict:
    """Judge the player's ``candidate`` answer vs the dataset ``reference``."""
    v = _judge(question, reference, candidate, endpoint=endpoint, model=model, client=client)
    if v.label == "error":
        return QuizVerdict(False, "error", v.reasoning, v.raw)
    right = v.label == "correct"
    return QuizVerdict(right, "right" if right else "wrong", v.reasoning, v.raw)
