"""Model registry: available models and how each accepts reasoning control.

Loaded from ``configs/models.yaml`` so per-model quirks (which reasoning
parameter the proxy expects, and which effort values are valid) are data, not
code. Used by the API to populate the model picker and by the QA layer to send
reasoning in the form each model accepts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    reasoning: str | None = None      # "effort" | "nested" | None
    efforts: tuple[str, ...] = ()     # valid effort values for this model


def _load() -> list[ModelInfo]:
    if not _CONFIG.exists():
        return []
    data = yaml.safe_load(_CONFIG.read_text()) or {}
    return [
        ModelInfo(
            id=m["id"],
            label=m.get("label", m["id"]),
            reasoning=m.get("reasoning"),
            efforts=tuple(m.get("efforts") or ()),
        )
        for m in data.get("models", [])
    ]


MODELS: list[ModelInfo] = _load()
_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODELS}


def get_model(model_id: str) -> ModelInfo | None:
    return _BY_ID.get(model_id)


def reasoning_request_kwargs(model_id: str, effort: str | None) -> dict:
    """Return the ``create()`` kwargs that enable `effort` reasoning for a model.

    The form is dictated by the model's ``reasoning`` style in the config:
    top-level ``reasoning_effort`` ("effort") or nested ``reasoning.effort``
    ("nested"). Models without a reasoning style (or no effort) get no kwargs.
    """
    if not effort:
        return {}
    info = _BY_ID.get(model_id)
    style = info.reasoning if info else None
    if style == "effort":
        # litellm rejects top-level reasoning_effort for some model routes (it classifies
        # them as the "openai" provider); allowed_openai_params forces the param through.
        return {"reasoning_effort": effort,
                "extra_body": {"allowed_openai_params": ["reasoning_effort"]}}
    if style == "nested":
        return {"extra_body": {"reasoning": {"effort": effort}}}
    return {}
