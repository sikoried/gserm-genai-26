"""End-to-end evaluation: answer each pair with a QA system, then judge it."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import yaml

from ..config import QAConfig
from ..qa import build_qa_system
from .judge import JUDGE_MODEL, VERDICTS, judge


@dataclass
class ItemResult:
    id: str
    question: str
    reference: str
    answer: str
    verdict: str
    reasoning: str


def load_pairs(path: str | Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text()) or []


def run_eval(config: QAConfig, pairs: list[dict], *, judge_model: str = JUDGE_MODEL,
             judge_endpoint: str | None = None,
             progress: Callable[[ItemResult], None] | None = None) -> list[ItemResult]:
    qa = build_qa_system(config)
    judge_endpoint = judge_endpoint or config.endpoint
    results: list[ItemResult] = []
    for i, pair in enumerate(pairs):
        rid = str(pair.get("id", i))
        question = pair["question"]
        reference = pair["reference_answer"]
        try:
            answer = qa.answer(question)
        except Exception as exc:  # answering failed — record and continue
            results.append(ItemResult(rid, question, reference,
                                      f"<answer error: {exc}>", "error",
                                      "answer call failed"))
            if progress:
                progress(results[-1])
            continue
        try:
            verdict = judge(question, reference, answer,
                            endpoint=judge_endpoint, model=judge_model)
            label, reasoning = verdict.label, verdict.reasoning
        except Exception as exc:  # judging failed
            label, reasoning = "error", f"judge call failed: {exc}"
        results.append(ItemResult(rid, question, reference, answer, label, reasoning))
        if progress:
            progress(results[-1])
    return results


def summarize(results: list[ItemResult]) -> dict[str, int]:
    counts = {label: 0 for label in (*VERDICTS, "error")}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return counts


def results_as_dicts(results: list[ItemResult]) -> list[dict]:
    return [asdict(r) for r in results]
