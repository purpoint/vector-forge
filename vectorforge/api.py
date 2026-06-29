"""FastAPI web layer — the engine's front door to the outside world.

This is the 'waiter': it exposes the RAGPipeline over HTTP so anything that
speaks the web (a browser, curl, a frontend) can use the engine. The pipeline
itself is untouched; we just translate HTTP requests into pipeline calls.

Run it:   uvicorn vectorforge.api:app --reload
Then open http://localhost:8000/docs  for interactive, auto-generated docs.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

# One pipeline instance, shared across all requests.
app = FastAPI(title="VectorForge", version="0.1.0")
pipeline = RAGPipeline()


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

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")