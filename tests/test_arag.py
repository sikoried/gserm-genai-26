"""Regression tests for the agentic-RAG message sanitizer.

The query "What is a conundrum" crashed end-to-end: that word isn't in the index, so
the agent's searches come back empty and the model emits an empty assistant step.
smolagents sends it with the content as a *list of parts* (``[{"type":"text","text":""}]``)
and the role as a ``MessageRole`` enum, which the original sanitizer missed — so the
empty assistant message reached Mistral, which rejects it (HTTP 400).
"""
from oracle.qa.arag import (
    _build_trace, _message_text, _sanitize_messages, references_video,
)


def test_references_video_detection():
    assert references_video("In Mark Rober's squirrel maze video, how many obstacles?")
    assert references_video("What happens in the MrBeast YouTube episode?")
    assert references_video("watch the trailer and tell me the release date")
    # Not video questions
    assert not references_video("What is the capital of France?")
    assert not references_video("Who directed Titanic?")


def test_message_text_flattens_string_and_parts():
    assert _message_text("hi") == "hi"
    assert _message_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _message_text([{"type": "text", "text": ""}]) == ""
    assert _message_text(None) == ""


def test_sanitize_fixes_empty_assistant_list_content():
    # The exact shape from the crash: assistant, list content with empty text, no tool calls.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "instructions"}]},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]
    _sanitize_messages(messages)
    assert messages[1]["content"] == [{"type": "text", "text": "(no output)"}]


def test_sanitize_fixes_empty_assistant_with_enum_role():
    class _Role:  # mimic smolagents' MessageRole enum (str shows "MessageRole.ASSISTANT")
        def __str__(self):
            return "MessageRole.ASSISTANT"

    messages = [{"role": _Role(), "content": [{"type": "text", "text": ""}]}]
    _sanitize_messages(messages)
    assert messages[0]["content"][0]["text"] == "(no output)"


def test_sanitize_fixes_empty_assistant_string_content():
    messages = [{"role": "assistant", "content": "", "tool_calls": None}]
    _sanitize_messages(messages)
    assert messages[0]["content"] == "(no output)"


class _Call:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class ActionStep:  # name must match for _build_trace's type-name check
    def __init__(self, tool_calls, observations):
        self.tool_calls = tool_calls
        self.observations = observations


def test_build_trace_shows_tool_calls():
    # Regression: reasoning was always "Answered directly" because result.steps are
    # dicts; with the real ActionStep objects the trace must show the tool calls.
    steps = [
        ActionStep([_Call("query_rewrite", {"query": "conundrum"})], "a confusing problem"),
        ActionStep([_Call("search", {"query": "...", "k": 5})], "[1] Foo\nbar"),
        ActionStep([_Call("final_answer", {"answer": "done"})], "done"),
    ]
    trace = _build_trace(steps)
    assert "query_rewrite(query='conundrum')" in trace
    assert "search(" in trace
    assert "final_answer" not in trace          # final answer is excluded from the trace
    assert "Answered directly" not in trace


def test_build_trace_direct_answer():
    steps = [ActionStep([_Call("final_answer", {"answer": "x"})], "x")]
    assert _build_trace(steps) == "Answered directly — no retrieval was needed."


def test_sanitize_leaves_valid_messages_untouched():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "real answer"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},  # empty but has a tool call
        {"role": "user", "content": [{"type": "text", "text": ""}]},        # user, not our concern
    ]
    _sanitize_messages(messages)
    assert messages[0]["content"] == [{"type": "text", "text": "real answer"}]
    assert messages[1]["content"] == ""
    assert messages[2]["content"] == [{"type": "text", "text": ""}]
