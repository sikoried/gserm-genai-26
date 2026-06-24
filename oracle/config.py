"""QA-system configuration (see CLAUDE.md `Configuration Schema`)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_MODEL = "mistralai/Mistral-Medium-3.5-128B"
DEFAULT_ENDPOINT = "https://kiz1.in.ohmportal.de/llmproxy/v1"


class QAConfig(BaseModel):
    """Configuration for a single QA system."""

    type: str = "world"  # world | rag | a-rag
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    temperature: float = 0.0
    system_prompt: str | None = None  # overrides the QA system's built-in prompt
    reasoning_effort: str | None = None  # "low"/"medium"/"high"; passed to the model when set
    # --- RAG ---
    top_k: int = 10  # number of retrieved chunks used as context (rag)
    embedding_model: str = "all-MiniLM-L6-v2"  # local embedding model for retrieval

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QAConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
