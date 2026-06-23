"""Tests for the model registry and config-driven reasoning dispatch."""
from oracle.models import MODELS, get_model, reasoning_request_kwargs


def test_registry_loaded():
    assert MODELS, "configs/models.yaml should define models"
    assert get_model("openai/gpt-oss-120b") is not None


def test_effort_style_uses_top_level_param():
    # Mistral-family models are configured with reasoning: effort.
    assert reasoning_request_kwargs("mistralai/Mistral-Medium-3.5-128B", "high") == {
        "reasoning_effort": "high"
    }


def test_nested_style_uses_extra_body():
    assert reasoning_request_kwargs("openai/gpt-oss-120b", "high") == {
        "extra_body": {"reasoning": {"effort": "high"}}
    }


def test_no_effort_or_uncontrollable_model_sends_nothing():
    assert reasoning_request_kwargs("openai/gpt-oss-120b", None) == {}
    # Magistral has no reasoning style configured.
    assert reasoning_request_kwargs("mistralai/Magistral-Small-2509", "high") == {}


def test_unknown_model_sends_nothing():
    assert reasoning_request_kwargs("some/unknown-model", "high") == {}
