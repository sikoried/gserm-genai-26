"""Unit tests for QAConfig loading (no network)."""
from oracle.config import QAConfig


def test_defaults():
    c = QAConfig()
    assert c.type == "world"
    assert "Mistral" in c.model
    assert c.temperature == 0.0
    assert c.system_prompt is None


def test_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("type: world\nmodel: foo\ntemperature: 0.3\n")
    c = QAConfig.from_yaml(p)
    assert c.model == "foo"
    assert c.temperature == 0.3


def test_system_prompt_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("type: world\nsystem_prompt: Be terse.\n")
    c = QAConfig.from_yaml(p)
    assert c.system_prompt == "Be terse."
