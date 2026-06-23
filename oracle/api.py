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


class AnswerResponse(BaseModel):
    model: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float


class CompareRequest(BaseModel):
    question: str
    models: list[str]
    endpoint: str = DEFAULT_ENDPOINT
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer_one(question: str, model: str, endpoint: str, temperature: float) -> AnswerResponse:
    config = QAConfig(type="world", model=model, endpoint=endpoint, temperature=temperature)
    qa = build_qa_system(config)
    result = qa.answer(question)
    return AnswerResponse(
        model=model,
        answer=result.content,
        prompt_tokens=result.metrics.prompt_tokens,
        completion_tokens=result.metrics.completion_tokens,
        total_tokens=result.metrics.total_tokens,
        elapsed_seconds=result.metrics.elapsed_seconds,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/answer", response_model=AnswerResponse)
def post_answer(req: AnswerRequest) -> AnswerResponse:
    try:
        return _answer_one(req.question, req.model, req.endpoint, req.temperature)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/compare", response_model=list[AnswerResponse])
def post_compare(req: CompareRequest) -> list[AnswerResponse]:
    if not req.models:
        raise HTTPException(status_code=400, detail="At least one model is required.")
    results: list[AnswerResponse] = []
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(req.models)) as pool:
        futures = {
            pool.submit(_answer_one, req.question, model, req.endpoint, req.temperature): model
            for model in req.models
        }
        for future in concurrent.futures.as_completed(futures):
            model = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f"{model}: {exc}")

    if errors and not results:
        raise HTTPException(status_code=500, detail="; ".join(errors))

    # Return in the original requested order
    order = {m: i for i, m in enumerate(req.models)}
    results.sort(key=lambda r: order.get(r.model, 999))
    return results


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
