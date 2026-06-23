"""FastAPI backend — routes chat requests through QA systems."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import QAConfig, DEFAULT_MODEL, DEFAULT_ENDPOINT
from .qa import build_qa_system

app = FastAPI(title="Oracle API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODE_TO_TYPE = {
    "World": "world",
    "RAG": "rag",
    "Agentic RAG": "a-rag",
}


class ChatRequest(BaseModel):
    question: str
    mode: str = "World"
    model: str = DEFAULT_MODEL
    temperature: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    qa_type = MODE_TO_TYPE.get(req.mode)
    if qa_type is None:
        raise HTTPException(400, f"Unknown mode: {req.mode!r}")

    config = QAConfig(
        type=qa_type,
        model=req.model,
        endpoint=DEFAULT_ENDPOINT,
        temperature=req.temperature,
    )

    try:
        qa = build_qa_system(config)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc))

    answer = qa.answer(req.question)
    return ChatResponse(answer=answer, mode=req.mode, model=req.model)
