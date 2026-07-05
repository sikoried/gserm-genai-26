"""Structured trace + token accounting for one agentic answer.

A trace is a list of ``TraceStep``s, each attributed to a *kind* — ``planner``
(multi-hop plan), ``router`` (a local routing decision), ``tool`` (a tool call),
or ``synthesis`` (a proxy answer / hop answer). Each step also records **which
model** produced it and its tokens split into **input / output / reasoning**
(reasoning is the hidden chain-of-thought subset of generation). ``TokenTotals``
aggregates by kind and by type so the GUI can show both per-call cost and a per-
question total. Local tools record 0 tokens and no model. The judge is never part
of this, consistent with the project's "judge excluded" accounting rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Why the loop stopped — surfaced to the user.
STOP_REASONS = ("done", "step-budget", "timeout", "loop-guard")
KINDS = ("planner", "router", "tool", "synthesis", "verify")
# Kinds whose token cost is produced by an LLM (for attribution / model names).
_MODEL_KINDS = ("planner", "router", "synthesis", "verify")


@dataclass
class TraceStep:
    index: int
    kind: str                    # "planner" | "router" | "tool" | "synthesis"
    tool: str | None             # tool name (router decision / tool call)
    arguments: dict | None       # tool arguments
    result: str                  # short rendering of the observation / decision
    prompt_tokens: int = 0       # input tokens
    completion_tokens: int = 0   # generated tokens (includes reasoning)
    reasoning_tokens: int = 0    # hidden chain-of-thought subset of completion
    elapsed_seconds: float = 0.0
    model_id: str | None = None  # which model produced this step (None for local tools)
    hop: int | None = None       # multi-hop sub-question index (None = top level)
    depth: int = 0               # recursion depth for nested sub-agents (0 = top level)

    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens

    @property
    def output_tokens(self) -> int:
        """Visible generation = completion minus the hidden reasoning part."""
        return max(self.completion_tokens - self.reasoning_tokens, 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "model_id": self.model_id,
            "hop": self.hop,
            "depth": self.depth,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass
class _Bucket:
    input: int = 0
    output: int = 0
    reasoning: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.reasoning

    def to_dict(self) -> dict:
        return {"input": self.input, "output": self.output,
                "reasoning": self.reasoning, "total": self.total}


@dataclass
class TokenTotals:
    """Token counts split by attribution (planner/router/tools/synthesis/verify) and type."""
    planner: _Bucket = field(default_factory=_Bucket)
    router: _Bucket = field(default_factory=_Bucket)
    tools: _Bucket = field(default_factory=_Bucket)
    synthesis: _Bucket = field(default_factory=_Bucket)
    verify: _Bucket = field(default_factory=_Bucket)

    def _all(self):
        return (self.planner, self.router, self.tools, self.synthesis, self.verify)

    @property
    def input_tokens(self) -> int:
        return sum(b.input for b in self._all())

    @property
    def output_tokens(self) -> int:
        return sum(b.output for b in self._all())

    @property
    def reasoning_tokens(self) -> int:
        return sum(b.reasoning for b in self._all())

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens + self.reasoning_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    def to_dict(self) -> dict:
        return {
            "planner": self.planner.to_dict(),
            "router": self.router.to_dict(),
            "tools": self.tools.to_dict(),
            "synthesis": self.synthesis.to_dict(),
            "verify": self.verify.to_dict(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class AgentTrace:
    steps: list[TraceStep] = field(default_factory=list)
    stop_reason: str = "done"
    elapsed_seconds: float = 0.0

    def add(self, step: TraceStep) -> None:
        self.steps.append(step)

    def totals(self) -> TokenTotals:
        t = TokenTotals()
        buckets = {"planner": t.planner, "router": t.router,
                   "tools": t.tools, "synthesis": t.synthesis, "verify": t.verify}
        for s in self.steps:
            key = "tools" if s.kind == "tool" else s.kind
            b = buckets.get(key)
            if b is None:
                continue
            b.input += s.input_tokens
            b.output += s.output_tokens
            b.reasoning += s.reasoning_tokens
        return t

    def model_names(self) -> dict:
        """First model id seen per LLM kind — the router / synthesis / planner models."""
        names: dict[str, str] = {}
        for s in self.steps:
            if s.kind in _MODEL_KINDS and s.model_id and s.kind not in names:
                names[s.kind] = s.model_id
        return names

    def as_text(self) -> str:
        """Human-readable trace, kept for the legacy ``Answer.reasoning`` string."""
        tool_steps = [s for s in self.steps if s.kind == "tool"]
        if not tool_steps:
            lines = ["Answered directly — no tools were called."]
        else:
            lines = []
            for s in tool_steps:
                args = ", ".join(f"{k}={v!r}" for k, v in (s.arguments or {}).items())
                lines.append(f"• {s.tool}({args})")
                if s.result:
                    snippet = " ".join(str(s.result).split())
                    lines.append(f"    → {snippet[:200]}{'…' if len(snippet) > 200 else ''}")
        lines.append(f"[stopped: {self.stop_reason}]")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "stop_reason": self.stop_reason,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "totals": self.totals().to_dict(),
            "models": self.model_names(),
        }
