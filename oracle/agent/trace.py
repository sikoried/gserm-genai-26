"""Structured trace + token accounting for one agentic answer.

A trace is a list of ``TraceStep``s, each attributed to a *kind* — ``router``
(a local routing decision), ``tool`` (a tool call), or ``synthesis`` (the final
proxy answer). Local tools record 0 tokens. ``TokenTotals`` aggregates the per-
step counts and breaks them down by kind so the GUI can show both per-call cost
and a per-question total. The judge is never part of this (it has no place in a
QA trace), consistent with the project's "judge excluded" accounting rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Why the loop stopped — surfaced to the user.
STOP_REASONS = ("done", "step-budget", "timeout", "loop-guard")
KINDS = ("router", "tool", "synthesis")


@dataclass
class TraceStep:
    index: int
    kind: str                    # "router" | "tool" | "synthesis"
    tool: str | None             # tool name (router decision / tool call)
    arguments: dict | None       # tool arguments
    result: str                  # short rendering of the observation / decision
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0

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
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass
class TokenTotals:
    """Token counts split by attribution plus a grand total."""
    router_prompt: int = 0
    router_completion: int = 0
    tools_prompt: int = 0
    tools_completion: int = 0
    synthesis_prompt: int = 0
    synthesis_completion: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.router_prompt + self.tools_prompt + self.synthesis_prompt

    @property
    def completion_tokens(self) -> int:
        return self.router_completion + self.tools_completion + self.synthesis_completion

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "router": {"prompt": self.router_prompt, "completion": self.router_completion,
                       "total": self.router_prompt + self.router_completion},
            "tools": {"prompt": self.tools_prompt, "completion": self.tools_completion,
                      "total": self.tools_prompt + self.tools_completion},
            "synthesis": {"prompt": self.synthesis_prompt, "completion": self.synthesis_completion,
                          "total": self.synthesis_prompt + self.synthesis_completion},
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
        for s in self.steps:
            if s.kind == "router":
                t.router_prompt += s.prompt_tokens
                t.router_completion += s.completion_tokens
            elif s.kind == "tool":
                t.tools_prompt += s.prompt_tokens
                t.tools_completion += s.completion_tokens
            elif s.kind == "synthesis":
                t.synthesis_prompt += s.prompt_tokens
                t.synthesis_completion += s.completion_tokens
        return t

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
        }
