"""Unit tests for the judge verdict parser (no network)."""
from oracle.eval.judge import parse_verdict


def test_parse_clean_json():
    v = parse_verdict('{"verdict": "correct", "reasoning": "matches reference"}')
    assert v.label == "correct"
    assert v.reasoning == "matches reference"


def test_parse_json_with_surrounding_text():
    v = parse_verdict('Here is my judgment: {"verdict":"wrong","reasoning":"contradicts"} ok')
    assert v.label == "wrong"


def test_keyword_fallback():
    v = parse_verdict("I think the candidate is orthogonal to the reference answer.")
    assert v.label == "orthogonal"


def test_unparseable_is_error():
    v = parse_verdict("no idea what to say here")
    assert v.label == "error"
