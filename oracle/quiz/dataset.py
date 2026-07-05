"""Quiz datasets: load + cache a HF question/answer dataset, pluggably.

The host quizzes the player from a dataset of ``{question, reference_answer}`` items.
The default is the **Who Wants to Be a Millionaire** set (``millionaire.csv`` from
``RedBlock/parrot``): multiple-choice questions whose options are appended to the
question so the player sees the choices. Other datasets work by giving a dataset id,
optional ``data_files``, and a column mapping. The HF download is cached once under
``data/hf-cache`` (like the wiki-10k corpus) and kept in memory for the process.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

_HF_CACHE = Path(__file__).resolve().parents[2] / "data" / "hf-cache"

# Named datasets with their default column mapping. Add entries to offer more.
# ``options_col`` (optional) is appended to the question (multiple-choice datasets).
DATASETS: dict[str, dict] = {
    "millionaire": {
        "name": "RedBlock/parrot",
        "data_files": "millionaire.csv",  # only this file — not jeopardy.csv
        "label": "Who Wants to Be a Millionaire (RedBlock/parrot)",
        "question_col": "question",
        "answer_col": "normalized_correct_opt",
        # No options_col: the player is asked the bare question (no A/B/C/D choices).
    },
}
DEFAULT_DATASET = "millionaire"

# In-memory cache of loaded HF datasets, keyed by (name, data_files, split).
_LOADED: dict = {}


@dataclass
class QuizItem:
    id: str
    question: str
    reference_answer: str


class QuizDataset:
    """A question/answer dataset the host draws from.

    Pass ``rows`` (a list of dicts) to use an in-memory dataset (tests); otherwise the
    HF dataset ``name`` is downloaded once and cached.
    """

    def __init__(self, name: str, question_col: str = "question",
                 answer_col: str = "answer", *, label: str | None = None,
                 split: str = "train", data_files: str | None = None,
                 options_col: str | None = None, rows: list[dict] | None = None):
        self.name = name
        self.label = label or name
        self.question_col = question_col
        self.answer_col = answer_col
        self.options_col = options_col
        self.split = split
        self.data_files = data_files
        self._rows = rows

    def _data(self):
        if self._rows is None:
            key = (self.name, self.data_files, self.split)
            if key not in _LOADED:
                from datasets import load_dataset
                kwargs = {"split": self.split, "cache_dir": str(_HF_CACHE)}
                if self.data_files:
                    kwargs["data_files"] = self.data_files
                _LOADED[key] = load_dataset(self.name, **kwargs)
            self._rows = _LOADED[key]
        return self._rows

    def __len__(self) -> int:
        data = self._data()
        return data.num_rows if hasattr(data, "num_rows") else len(data)

    def item(self, index: int) -> QuizItem:
        row = self._data()[index]
        question = str(row[self.question_col])
        # Multiple-choice datasets: append the options so the player sees the choices.
        if self.options_col and row.get(self.options_col):
            question = f"{question}\nOptions: {row[self.options_col]}"
        return QuizItem(id=str(index), question=question,
                        reference_answer=str(row[self.answer_col]))

    def sample_indices(self, count: int | None, seed: int | None = None) -> list[int]:
        """Random question indices to play. ``count`` None/``>= len`` → all of them."""
        n = len(self)
        if not count or count >= n:
            return list(range(n))
        rng = random.Random(seed)
        return sorted(rng.sample(range(n), count))


def get_dataset(key: str = DEFAULT_DATASET, *, question_col: str | None = None,
                answer_col: str | None = None) -> QuizDataset:
    """Build a ``QuizDataset`` for a registered key (or a raw HF id), with optional
    column overrides for a custom/pluggable dataset."""
    spec = DATASETS.get(key)
    if spec is None:
        # Treat an unknown key as a raw HF dataset id (pluggable path).
        return QuizDataset(key, question_col or "question", answer_col or "answer")
    return QuizDataset(
        spec["name"], question_col or spec["question_col"],
        answer_col or spec["answer_col"], label=spec["label"],
        data_files=spec.get("data_files"), options_col=spec.get("options_col"),
    )


def list_datasets() -> list[dict]:
    """Registry entries for the GUI (id, label, columns) — no download needed."""
    return [{"id": k, "label": v["label"], "question_column": v["question_col"],
             "answer_column": v["answer_col"]} for k, v in DATASETS.items()]
