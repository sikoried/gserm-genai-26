"""FastAPI application for Oracle.

Endpoints:
  POST /api/answer   — answer a single question with one model
  POST /api/compare  — answer the same question with multiple models in parallel
"""
from __future__ import annotations

import concurrent.futures
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

from .config import DEFAULT_ENDPOINT, QAConfig
from .models import MODELS
from .qa import build_qa_system
from . import quiz as quizmod

app = FastAPI(title="Oracle API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AnswerRequest(BaseModel):
    question: str
    model: str = QAConfig().model
    endpoint: str = DEFAULT_ENDPOINT
    temperature: float = 0.0
    reasoning_effort: str | None = None  # "low"/"medium"/"high"; omitted when None


class AnswerResponse(BaseModel):
    model: str
    reasoning_effort: str | None = None  # echo of the requested setting (None = off)
    answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0  # subset of completion_tokens spent on reasoning
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None  # populated when this model's call failed


class CompareEntry(BaseModel):
    model: str
    reasoning_effort: str | None = None  # "low"/"medium"/"high"; None = no reasoning


class CompareRequest(BaseModel):
    question: str
    entries: list[CompareEntry]
    endpoint: str = DEFAULT_ENDPOINT
    temperature: float = 0.0


class ModelInfoResponse(BaseModel):
    id: str
    label: str
    efforts: list[str]  # reasoning effort values this model accepts (may be empty)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []  # prior turns, oldest first
    mode: str = "World"  # World | RAG | Agentic RAG
    model: str = QAConfig().model
    temperature: float = 0.0
    enable_online_tools: bool = True  # a-rag: allow google_search + youtube (default on)
    multi_hop: bool = True             # a-rag: decompose into sub-questions (default on)
    planner_model: str = "router"      # a-rag: "router" (small local) | "answer" (big model)
    max_hops: int = 3                  # a-rag: max sub-questions per planning step
    max_steps: int = 6                 # a-rag: max tool calls before forced synthesis
    max_depth: int = 1                 # a-rag: recursive planning levels (1 = flat)
    verify: bool = True                # a-rag: bounded verification/backtracking hop (default on)


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str
    reasoning: str | None = None  # human-readable trace (agentic RAG)
    trace: dict | None = None     # structured per-step trace + token totals (a-rag)
    usage: dict | None = None      # per-question token breakdown (router/tools/synthesis)


# A chat "mode" maps onto a QA-system type.
MODE_TO_TYPE = {"World": "world", "RAG": "rag", "Agentic RAG": "a-rag"}


class PlayerSettings(BaseModel):
    """The QA-mode + agentic-RAG settings shared by chat and the quiz player."""
    mode: str = "Agentic RAG"
    model: str = QAConfig().model
    temperature: float = 0.0
    enable_online_tools: bool = True
    multi_hop: bool = True
    planner_model: str = "router"
    max_hops: int = 3
    max_steps: int = 6
    max_depth: int = 1
    verify: bool = True


def _player_config(p: "PlayerSettings | ChatRequest") -> QAConfig:
    """Build a QAConfig for a player from its mode + agentic-RAG settings."""
    qa_type = MODE_TO_TYPE.get(p.mode)
    if qa_type is None:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {p.mode!r}")
    return QAConfig(type=qa_type, model=p.model, endpoint=DEFAULT_ENDPOINT,
                    temperature=p.temperature,
                    enable_online_tools=p.enable_online_tools, multi_hop=p.multi_hop,
                    planner_model=p.planner_model, max_hops=p.max_hops,
                    max_steps=p.max_steps, max_depth=p.max_depth, verify=p.verify)


def _answer_payload(result) -> dict:
    """Serialize an Answer's metrics + trace for the client (shared by chat/quiz)."""
    trace = result.trace.to_dict() if result.trace is not None else None
    usage = {**trace["totals"], "models": trace.get("models", {})} if trace else None
    m = result.metrics
    return {
        "answer": result.content, "reasoning": result.reasoning,
        "trace": trace, "usage": usage,
        "metrics": {
            "input_tokens": m.prompt_tokens,
            "output_tokens": max(m.completion_tokens - m.reasoning_tokens, 0),
            "reasoning_tokens": m.reasoning_tokens,
            "total_tokens": m.total_tokens,
            "elapsed_seconds": m.elapsed_seconds,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer_one(question: str, model: str, endpoint: str, temperature: float,
                reasoning_effort: str | None = None) -> AnswerResponse:
    config = QAConfig(type="world", model=model, endpoint=endpoint,
                      temperature=temperature, reasoning_effort=reasoning_effort)
    qa = build_qa_system(config)
    result = qa.answer(question)
    return AnswerResponse(
        model=model,
        reasoning_effort=reasoning_effort,
        answer=result.content,
        prompt_tokens=result.metrics.prompt_tokens,
        completion_tokens=result.metrics.completion_tokens,
        reasoning_tokens=result.metrics.reasoning_tokens,
        total_tokens=result.metrics.total_tokens,
        elapsed_seconds=result.metrics.elapsed_seconds,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/models", response_model=list[ModelInfoResponse])
def get_models() -> list[ModelInfoResponse]:
    return [ModelInfoResponse(id=m.id, label=m.label, efforts=list(m.efforts)) for m in MODELS]


@app.post("/api/answer", response_model=AnswerResponse)
def post_answer(req: AnswerRequest) -> AnswerResponse:
    try:
        return _answer_one(req.question, req.model, req.endpoint, req.temperature,
                           req.reasoning_effort)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/compare", response_model=list[AnswerResponse])
def post_compare(req: CompareRequest) -> list[AnswerResponse]:
    if not req.entries:
        raise HTTPException(status_code=400, detail="At least one model entry is required.")

    # Each entry (model + optional reasoning) runs independently; the same model
    # may appear more than once (e.g. with and without reasoning). Results are
    # keyed by position so duplicates are kept and failures surface as errors.
    results: list[AnswerResponse | None] = [None] * len(req.entries)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(req.entries)) as pool:
        futures = {
            pool.submit(_answer_one, req.question, entry.model, req.endpoint,
                        req.temperature, entry.reasoning_effort): i
            for i, entry in enumerate(req.entries)
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            entry = req.entries[i]
            try:
                results[i] = future.result()
            except Exception as exc:
                results[i] = AnswerResponse(model=entry.model,
                                            reasoning_effort=entry.reasoning_effort,
                                            error=str(exc))

    return results


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest) -> ChatResponse:
    qa_type = MODE_TO_TYPE.get(req.mode)
    if qa_type is None:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode!r}")
    config = QAConfig(type=qa_type, model=req.model, endpoint=DEFAULT_ENDPOINT,
                      temperature=req.temperature,
                      enable_online_tools=req.enable_online_tools,
                      multi_hop=req.multi_hop, planner_model=req.planner_model,
                      max_hops=req.max_hops, max_steps=req.max_steps,
                      max_depth=req.max_depth, verify=req.verify)
    try:
        qa = build_qa_system(config)
    except NotImplementedError as exc:  # RAG / a-rag not implemented yet
        raise HTTPException(status_code=501, detail=str(exc))
    try:
        if hasattr(qa, "answer_chat"):
            # Conversation-aware (rag retrieves on the first message, then extends context).
            conversation = [m.model_dump() for m in req.history]
            conversation.append({"role": "user", "content": req.question})
            result = qa.answer_chat(conversation)
        else:
            result = qa.answer(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    trace = result.trace.to_dict() if result.trace is not None else None
    usage = None
    if trace:
        usage = {**trace["totals"], "models": trace.get("models", {})}
    return ChatResponse(answer=result.content, mode=req.mode, model=req.model,
                        reasoning=result.reasoning, trace=trace, usage=usage)


# ---------------------------------------------------------------------------
# Quiz: a host (dataset + judge) quizzes a player (a QA mode). See quiz.md.
# ---------------------------------------------------------------------------

class QuizStartRequest(BaseModel):
    dataset: str = quizmod.DEFAULT_DATASET
    count: int | None = None  # None or >= size → all questions; else a random sample
    seed: int | None = None


class QuizStartResponse(BaseModel):
    dataset: str
    total: int          # size of the dataset
    indices: list[int]  # the (randomly sampled) question indices to play


class QuizAnswerRequest(PlayerSettings):
    dataset: str = quizmod.DEFAULT_DATASET
    index: int
    host_model: str = quizmod.DEFAULT_HOST_MODEL


class QuizAnswerResponse(BaseModel):
    index: int
    question: str
    reference: str
    answer: str
    verdict: str              # "right" | "wrong" | "error"
    right: bool
    judge_reasoning: str
    reasoning: str | None = None
    trace: dict | None = None
    usage: dict | None = None
    metrics: dict


@app.get("/api/quiz/datasets")
def get_quiz_datasets() -> list[dict]:
    return quizmod.list_datasets()


@app.post("/api/quiz/start", response_model=QuizStartResponse)
def post_quiz_start(req: QuizStartRequest) -> QuizStartResponse:
    try:
        ds = quizmod.get_dataset(req.dataset)  # downloads + caches on first use
        indices = ds.sample_indices(req.count, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load dataset: {exc}")
    return QuizStartResponse(dataset=req.dataset, total=len(ds), indices=indices)


@app.post("/api/quiz/answer", response_model=QuizAnswerResponse)
def post_quiz_answer(req: QuizAnswerRequest) -> QuizAnswerResponse:
    try:
        ds = quizmod.get_dataset(req.dataset)
        item = ds.item(req.index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load question: {exc}")

    config = _player_config(req)  # 400 on unknown mode
    try:
        qa = build_qa_system(config)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    try:
        result = qa.answer(item.question)  # the player answers (no reference given)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Player failed: {exc}")

    # The host grades the answer against the reference (its cost is not the player's).
    verdict = quizmod.grade(item.question, item.reference_answer, result.content,
                            endpoint=DEFAULT_ENDPOINT, model=req.host_model)

    payload = _answer_payload(result)
    return QuizAnswerResponse(
        index=req.index, question=item.question, reference=item.reference_answer,
        answer=payload["answer"], verdict=verdict.verdict, right=verdict.right,
        judge_reasoning=verdict.reasoning, reasoning=payload["reasoning"],
        trace=payload["trace"], usage=payload["usage"], metrics=payload["metrics"],
    )


# ---------------------------------------------------------------------------
# Serve the React frontend (production build under frontend/dist/)
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{path:path}", include_in_schema=False)
    def serve_spa(path: str = "") -> FileResponse:
        index = _FRONTEND_DIST / "index.html"
        return FileResponse(str(index))
