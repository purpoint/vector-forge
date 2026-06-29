"""FastAPI web layer -- the engine's front door to the outside world.

Exposes the RAGPipeline over HTTP and serves the web UI. The pipeline is
untouched; we just translate HTTP requests into pipeline calls and pick the
real or offline providers based on which API keys are present.

Run it:   uvicorn vectorforge.api:app --reload
Then open http://localhost:8000/        for the web UI
       or http://localhost:8000/docs    for interactive API docs
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .embeddings import HashEmbedder, VoyageEmbedder
from .llm import build_llm
from .pipeline import RAGPipeline

load_dotenv()  # read .env into the environment

app = FastAPI(title="VectorForge", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


def _make_pipeline() -> RAGPipeline:
    # Real Voyage embeddings if the key is present; offline HashEmbedder otherwise.
    embedder = VoyageEmbedder() if os.getenv("VOYAGE_API_KEY") else HashEmbedder()
    # Real Claude only if an Anthropic key is set; otherwise the free offline EchoLLM.
    llm = build_llm("anthropic") if os.getenv("ANTHROPIC_API_KEY") else build_llm("echo")
    return RAGPipeline(embedder=embedder, llm=llm)


pipeline = _make_pipeline()


# --- request/response shapes (Pydantic validates these automatically) ---

class IngestRequest(BaseModel):
    text: str
    doc_id: str | None = None


class IngestResponse(BaseModel):
    doc_id: str | None
    chunks_added: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str
    k: int | None = None


class HitModel(BaseModel):
    text: str
    doc_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[int]
    context: list[HitModel]


# --- endpoints ---

@app.get("/")
def index():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Liveness check: is the server up, and how many chunks are indexed?"""
    return {"status": "ok", "chunks_indexed": len(pipeline.store)}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Add a document to the engine."""
    added = pipeline.ingest(req.text, doc_id=req.doc_id)
    return IngestResponse(
        doc_id=req.doc_id,
        chunks_added=added,
        total_chunks=len(pipeline.store),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Ask a question; get a grounded answer with citations."""
    answer = pipeline.query(req.question, k=req.k)
    return QueryResponse(
        answer=answer.text,
        citations=answer.citations,
        context=[
            HitModel(
                text=h.record.text,
                doc_id=h.record.doc_id,
                score=round(h.score, 4),
            )
            for h in answer.context_used
        ],
    )