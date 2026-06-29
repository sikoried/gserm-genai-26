"""The tool router: a small LLM, run locally, that picks the next tool.

The router is deliberately separate from the large answering model: it only
decides *which tool to call next* (or that it is done), keeping selection cheap
and private. ``LocalRouter`` loads a small Hugging Face instruct model in-process
via ``transformers`` (default ``Qwen/Qwen2.5-1.5B-Instruct``) and emits a JSON
decision; tests use a fake ``Router`` instead.

Heavy imports (torch / transformers) are deferred to first use so importing this
module — and the rest of Oracle — stays light.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Rough on-disk / in-memory footprint (GB, fp16) for the models we document, used
# only for the memory guardrail. Unknown models are allowed with a warning.
_MODEL_GB = {
    "Qwen/Qwen2.5-1.5B-Instruct": 3.1,
    "Qwen/Qwen2.5-3B-Instruct": 6.2,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 3.4,
    "meta-llama/Llama-3.2-1B-Instruct": 2.5,
    "meta-llama/Llama-3.2-3B-Instruct": 6.5,
}


@dataclass
class RouterDecision:
    finished: bool
    tool: str | None = None
    arguments: dict | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""


class Router(ABC):
    """Chooses the next tool (or finishing) given the question and observations."""

    @abstractmethod
    def decide(self, question: str, history: list[dict],
               observations: list[dict]) -> RouterDecision:
        ...


def _pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def estimate_gb(model_id: str) -> float | None:
    """Approximate fp16 footprint for a known model id (else None)."""
    return _MODEL_GB.get(model_id)


_SYSTEM = (
    "You are a tool router for a quiz-answering agent. Given the question and the "
    "tools already used (with their results), decide the single next step.\n"
    "Reply with ONLY a JSON object, no prose:\n"
    '  {"tool": "<name>", "arguments": {...}}  to call a tool, or\n'
    '  {"finished": true}                      when enough is known to answer.\n'
    "Prefer to finish as soon as the question can be answered. Never repeat a tool "
    "call you already made with the same arguments."
)


def _render_tools(metas: list[dict]) -> str:
    lines = []
    for m in metas:
        args = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in (m.get("inputs") or {}).items())
        lines.append(f"- {m['name']}({args}): {m['description'].splitlines()[0]}")
    return "\n".join(lines)


def _render_observations(observations: list[dict]) -> str:
    if not observations:
        return "(no tools used yet)"
    out = []
    for o in observations:
        args = ", ".join(f"{k}={v!r}" for k, v in (o.get("arguments") or {}).items())
        result = " ".join(str(o.get("result", "")).split())
        out.append(f"- {o.get('tool')}({args}) -> {result[:200]}")
    return "\n".join(out)


def parse_decision(raw: str) -> RouterDecision:
    """Parse a router model reply into a RouterDecision (robust to extra prose)."""
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            if obj.get("finished") is True or obj.get("tool") in (None, "", "final_answer"):
                return RouterDecision(finished=True, raw=raw)
            args = obj.get("arguments") or obj.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            return RouterDecision(finished=False, tool=str(obj["tool"]),
                                  arguments=args, raw=raw)
    # Unparseable → finish, so the loop always makes progress toward an answer.
    return RouterDecision(finished=True, raw=raw)


class LocalRouter(Router):
    """Local small-LLM router backed by a transformers causal model."""

    def __init__(self, model_id: str, tool_metas: list[dict], *,
                 device: str | None = None, max_gb: float = 6.0,
                 max_new_tokens: int = 192):
        self.model_id = model_id
        self.tool_metas = tool_metas
        self.device = device or _pick_device()
        self.max_new_tokens = max_new_tokens
        self._check_fits(max_gb)
        self._tokenizer = None
        self._model = None

    def _check_fits(self, max_gb: float) -> None:
        gb = estimate_gb(self.model_id)
        if gb is not None and gb > max_gb:
            raise RuntimeError(
                f"Router model {self.model_id!r} needs ~{gb:.1f} GB but the ceiling "
                f"is {max_gb:.1f} GB. Pick a smaller model (e.g. "
                f"Qwen/Qwen2.5-1.5B-Instruct) or raise router_max_gb."
            )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto"
        ).to(self.device)

    def _generate(self, messages: list[dict]) -> tuple[str, int, int]:
        self._ensure_model()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new = generated[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new, skip_special_tokens=True)
        return text, int(inputs["input_ids"].shape[1]), int(new.shape[0])

    def _messages(self, question: str, history: list[dict],
                  observations: list[dict]) -> list[dict]:
        convo = ""
        if history:
            convo = "Conversation so far:\n" + "\n".join(
                f"{m.get('role')}: {m.get('content')}" for m in history) + "\n\n"
        user = (
            f"{convo}Question: {question}\n\n"
            f"Available tools:\n{_render_tools(self.tool_metas)}\n\n"
            f"Tools used so far:\n{_render_observations(observations)}\n\n"
            "Next step as JSON:"
        )
        return [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user}]

    def decide(self, question: str, history: list[dict],
               observations: list[dict]) -> RouterDecision:
        text, ptok, ctok = self._generate(self._messages(question, history, observations))
        decision = parse_decision(text)
        decision.prompt_tokens = ptok
        decision.completion_tokens = ctok
        return decision
