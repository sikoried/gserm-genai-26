"""Multi-hop planner: decompose a question into ordered sub-questions.

Some quiz questions need several dependent hops (find film → find its director →
find their birthplace). The planner emits a short ordered plan; the multi-hop
orchestrator (``agent.multihop``) then answers each sub-question with the bounded
loop, threading earlier answers forward as facts.

``LocalPlanner`` reuses the local router model (a JSON call), so planning stays
local and cheap. Tests use a fake ``Planner``. A single-hop plan (just the
original question) reproduces the non-planning behaviour.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PlanResult:
    subquestions: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    model_id: str | None = None
    raw: str = ""


class Planner(ABC):
    @abstractmethod
    def plan(self, question: str, history: list[dict]) -> PlanResult:
        ...


_SYSTEM = (
    "You break a quiz question into the minimal ordered list of sub-questions needed "
    "to answer it, where each later sub-question may depend on earlier answers. If the "
    "question is already atomic, return just it. Reply with ONLY a JSON array of "
    'strings, e.g. ["...", "..."]. Use at most {max_hops} items.'
)


def parse_plan(raw: str, question: str, max_hops: int) -> list[str]:
    """Parse a planner reply into a bounded list of sub-questions (robust)."""
    text = (raw or "").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
        except json.JSONDecodeError:
            arr = None
        if isinstance(arr, list):
            subs = [str(x).strip() for x in arr if str(x).strip()]
            if subs:
                return subs[:max_hops]
    # Fallback: treat as a single hop so the run still proceeds.
    return [question]


class SingleHopPlanner(Planner):
    """Trivial planner: always one hop (the original question)."""

    def plan(self, question: str, history: list[dict]) -> PlanResult:
        return PlanResult(subquestions=[question])


class LocalPlanner(Planner):
    """Plan with the local router model (``generate`` returns (text, ptok, ctok))."""

    def __init__(self, model_id: str, generate, max_hops: int = 3):
        self.model_id = model_id
        self._generate = generate  # callable(messages) -> (text, prompt_tokens, completion_tokens)
        self.max_hops = max_hops

    def plan(self, question: str, history: list[dict]) -> PlanResult:
        convo = ""
        if history:
            convo = "Conversation so far:\n" + "\n".join(
                f"{m.get('role')}: {m.get('content')}" for m in history) + "\n\n"
        messages = [
            {"role": "system", "content": _SYSTEM.format(max_hops=self.max_hops)},
            {"role": "user", "content": f"{convo}Question: {question}\n\nSub-questions as JSON:"},
        ]
        text, ptok, ctok = self._generate(messages)
        return PlanResult(
            subquestions=parse_plan(text, question, self.max_hops),
            prompt_tokens=ptok, completion_tokens=ctok, model_id=self.model_id, raw=text,
        )
