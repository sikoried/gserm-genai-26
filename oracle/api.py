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
    enable_online_tools: bool = False  # a-rag: allow google_search + youtube
    multi_hop: bool = False            # a-rag: decompose into sub-questions


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str
    reasoning: str | None = None  # human-readable trace (agentic RAG)
    trace: dict | None = None     # structured per-step trace + token totals (a-rag)
    usage: dict | None = None      # per-question token breakdown (router/tools/synthesis)


# A chat "mode" maps onto a QA-system type.
MODE_TO_TYPE = {"World": "world", "RAG": "rag", "Agentic RAG": "a-rag"}


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
                      multi_hop=req.multi_hop)
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
