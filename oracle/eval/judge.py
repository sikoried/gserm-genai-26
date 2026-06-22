"""LLM-as-a-judge: classify a candidate answer vs. a reference.

Verdict is one of three labels:
  - correct    : conveys the same essential information as the reference
  - wrong      : contradicts the reference / factually incorrect
  - orthogonal : neither right nor wrong (refuses, off-topic, tangential)

`error` is reserved for cases where the judge response could not be parsed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..llm import chat, make_client

JUDGE_MODEL = "openai/gpt-oss-120b"
VERDICTS = ("correct", "wrong", "orthogonal")

SYSTEM_PROMPT = """You are a strict evaluator for a question-answering system.
You are given a QUESTION, a REFERENCE answer (the gold answer), and a CANDIDATE answer.
Classify the CANDIDATE into exactly one verdict:
- "correct": the candidate conveys the same essential information as the reference \
(paraphrases, different wording, or extra correct detail are fine).
- "wrong": the candidate contradicts the reference or is factually incorrect for the question.
- "orthogonal": the candidate is neither right nor wrong with respect to the reference \
— e.g. it refuses, is off-topic, hedges without answering, or addresses a different aspect.
Respond with ONLY a JSON object, no extra text:
{"verdict": "<correct|wrong|orthogonal>", "reasoning": "<one short sentence>"}"""


@dataclass
class Verdict:
    label: str
    reasoning: str
    raw: str


def parse_verdict(raw: str) -> Verdict:
    """Parse a judge response into a Verdict (robust to non-JSON output)."""
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            label = str(obj.get("verdict", "")).strip().lower()
            if label in VERDICTS:
                return Verdict(label, str(obj.get("reasoning", "")).strip(), raw)
        except json.JSONDecodeError:
            pass
    # Fallback: scan for a bare verdict keyword.
    lowered = text.lower()
    for label in VERDICTS:
        if label in lowered:
            return Verdict(label, "(parsed from non-JSON response)", raw)
    return Verdict("error", "(unparseable judge response)", raw)


def judge(question: str, reference: str, candidate: str, *, endpoint: str,
          model: str = JUDGE_MODEL, client=None) -> Verdict:
    client = client or make_client(endpoint)
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE:\n{reference}\n\n"
        f"CANDIDATE:\n{candidate}"
    )
    raw = chat(
        client,
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return parse_verdict(raw)
