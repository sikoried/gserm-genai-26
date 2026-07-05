"""Quiz feature: a host (dataset + judge) quizzes a player (a QA mode).

See ``quiz.md``. ``dataset`` loads/caches the Q&A dataset (default RedBlock/parrot);
``host`` grades the player's answers right/wrong with a strong model.
"""
from __future__ import annotations

from .dataset import (
    DATASETS, DEFAULT_DATASET, QuizDataset, QuizItem, get_dataset, list_datasets,
)
from .host import DEFAULT_HOST_MODEL, QuizVerdict, grade

__all__ = [
    "DATASETS", "DEFAULT_DATASET", "QuizDataset", "QuizItem", "get_dataset",
    "list_datasets", "DEFAULT_HOST_MODEL", "QuizVerdict", "grade",
]
