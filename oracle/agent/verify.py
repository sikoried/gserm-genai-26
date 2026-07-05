"""Verification / backtracking hop for the agentic loop.

After a candidate answer is composed, an optional **bounded** verification step
re-checks it against the gathered facts. If the verifier finds a contradiction it
returns a corrected answer (a one-shot backtrack); it never loops. ``ProxyVerifier``
asks the big answering model; tests use a fake ``Verifier``.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VerifyResult:
    ok: bool
    corrected_answer: str | None = None
    note: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    model_id: str | None = None


class Verifier(ABC):
    @abstractmethod
    def verify(self, question: str, answer: str, facts: list[tuple[str, str]]) -> VerifyResult:
        ...


_SYSTEM = (
    "You verify a candidate answer against the facts gathered while answering a quiz "
    "question. If the candidate is consistent with the facts and correctly answers the "
    "question, reply {\"ok\": true}. If it is contradicted or wrong, reply "
    "{\"ok\": false, \"answer\": \"<corrected short answer>\"}. Reply with ONLY the JSON."
)


def parse_verdict(raw: str, fallback_answer: str) -> tuple[bool, str | None, str]:
    """(ok, corrected_answer, note) parsed from a verifier reply (robust)."""
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "ok" in obj:
            ok = bool(obj.get("ok"))
            if ok:
                return True, None, "verified"
            corrected = str(obj.get("answer", "")).strip() or None
            return False, corrected, "corrected on contradiction"
    # Unparseable → treat as OK so a fragile verifier can't discard a good answer.
    return True, None, "unparseable verifier reply — kept candidate"


class ProxyVerifier(Verifier):
    """Verify with the big proxy model (``chat`` returns (content, UsageMetrics))."""

    def __init__(self, chat_with_metrics, client, model_id: str, *, temperature: float = 0.0):
        self._chat = chat_with_metrics
        self._client = client
        self.model_id = model_id
        self.temperature = temperature

    def verify(self, question: str, answer: str, facts: list[tuple[str, str]]) -> VerifyResult:
        fact_text = "\n".join(f"- {q}: {a}" for q, a in facts) or "(none)"
        user = (f"QUESTION:\n{question}\n\nFACTS:\n{fact_text}\n\n"
                f"CANDIDATE ANSWER:\n{answer}")
        content, metrics = self._chat(
            self._client, self.model_id,
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            temperature=self.temperature,
        )
        ok, corrected, note = parse_verdict(content, answer)
        return VerifyResult(
            ok=ok, corrected_answer=corrected, note=note,
            prompt_tokens=metrics.prompt_tokens, completion_tokens=metrics.completion_tokens,
            reasoning_tokens=metrics.reasoning_tokens, model_id=self.model_id,
        )
