"""Unit tests for QAConfig loading (no network)."""
from oracle.config import QAConfig


def test_defaults():
    c = QAConfig()
    assert c.type == "world"
    assert "Mistral" in c.model
    assert c.temperature == 0.0


def test_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("type: world\nmodel: foo\ntemperature: 0.3\n")
    c = QAConfig.from_yaml(p)
    assert c.model == "foo"
    assert c.temperature == 0.3
