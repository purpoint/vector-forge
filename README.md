# VectorForge

### 🔗 [Live demo →](https://vector-forge-xvi8.onrender.com)
*Paste any text, ask a question, get a grounded answer with the exact passages it used — ranked by similarity.*

A retrieval-augmented generation (RAG) engine built **from scratch** — no Pinecone, no LangChain, no vector-database black box. Documents go in, grounded answers with citations come out. Cosine similarity is hand-written with NumPy; every external dependency (the vector store, the embedder, the LLM) sits behind a small interface so it can be swapped without touching the pipeline.

Built with **Python + FastAPI**. The live demo uses real semantic embeddings (Voyage AI, free tier); the codebase also runs end-to-end with **zero API keys** (offline embedder + offline generator), so it's cloneable and testable in under a minute.

`Python 3.10+` · `41 passing tests` · `REST API` · `live demo` · `self-measuring eval harness`

> Note: the demo runs on a free tier that sleeps when idle — the first request after a while can take ~50 seconds to wake.

---

## What it is

A plain language model answers from memory — which means it can hallucinate, can't see your private documents, and can't be verified. VectorForge turns it into an *open-book* system: it finds the most relevant passages from your own documents first, then makes the model answer **only** from those passages, citing each one.

```
upload docs → chunk (with overlap) → embed → store vectors
                                                   │
question → embed → cosine top-k → assemble context → grounded answer + citations
```

## Design decisions (the interesting part)

Three deliberate choices, each made for a reason an interviewer would ask about:

- **Hand-written cosine similarity, behind a `VectorStore` interface.** The point of the project is to demonstrate that I understand what a vector database actually does, so retrieval is exact brute-force cosine in NumPy. Because it lives behind an interface, the production scaling path (Postgres + **pgvector** with an HNSW index) drops in as an alternate backend without changing the pipeline. Fundamentals *and* the scaling story, in one design.

- **Everything external is swappable.** The embedder, the vector store, and the LLM each sit behind a tiny contract. The pipeline depends only on those contracts, never on a concrete implementation — so the offline embedder swaps for a real model (Voyage / OpenAI), and the offline generator swaps for Claude, with a one-line change.

- **The engine measures itself.** An evaluation harness scores retrieval quality with a real number (top-k hit rate + MRR), so every future improvement is *proven*, not guessed.

## Architecture

| Module | Responsibility |
|---|---|
| `chunking.py` | Sentence-aware splitting into overlapping windows |
| `embeddings.py` | `Embedder` interface + offline `HashEmbedder` (Voyage / OpenAI swap in) |
| `similarity.py` | Hand-written L2-normalize, cosine similarity, top-k selection |
| `vectorstore.py` | `VectorStore` interface: in-memory NumPy backend + documented pgvector path |
| `pipeline.py` | Ties everything together: `ingest`, `retrieve`, `query` |
| `llm.py` | Grounded generation with citations (offline `EchoLLM` / real `AnthropicLLM`) |
| `api.py` | FastAPI REST layer (`/ingest`, `/query`, `/health`) |
| `eval/harness.py` | Retrieval quality: top-k hit rate and MRR |

Each module owns one concern and depends only on the small interfaces of the others, so any part can be swapped in isolation.

## Quickstart

```bash
# clone and set up
git clone https://github.com/purpoint/vector-forge.git
cd vector-forge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# run the tests
pytest -q

# measure retrieval quality
python -m eval.harness

# serve the API, then open http://localhost:8000/docs
uvicorn vectorforge.api:app --reload
```

Use it from Python directly:

```python
from vectorforge import RAGPipeline

rag = RAGPipeline()
rag.ingest("Photosynthesis happens in leaves. Chlorophyll absorbs sunlight. Plants release oxygen.")
answer = rag.query("what do plants release?")

print(answer.text)        # grounded answer with [n] citations
print(answer.citations)   # e.g. [1]
```

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check + number of indexed chunks |
| `POST` | `/ingest` | Add a document (chunked, embedded, stored) |
| `POST` | `/query` | Ask a question; get a grounded, cited answer |

FastAPI auto-generates interactive docs at `/docs` — a clickable browser UI for the API, with no frontend code.

## Evaluation

Generation quality is hard to score automatically; **retrieval quality is not**. For each labeled question we know which document holds the answer, so the harness measures whether the right chunk lands in the top-k. If retrieval misses, no LLM can recover — so this is the most honest health signal for the engine.

- **hit_rate@k** — fraction of questions whose relevant document appears in the top-k results.
- **MRR** — mean reciprocal rank; rewards ranking the right chunk *higher*, not just including it.

Results on the bundled 7-question set (run `python -m eval.harness` to reproduce):

| k | hit_rate@k | MRR |
|---|------------|-------|
| 1 | 0.857 | 0.857 |
| 3 | 1.000 | 0.929 |
| 5 | 1.000 | 0.929 |

Read this as: with only the single best chunk (`k=1`), retrieval finds the right document ~86% of the time; given the top 3, it finds it **100%** of the time. These run on the **offline lexical embedder** — a deliberately weak baseline. Swapping in a real embedding model and re-running this same harness shows the measured lift, rather than a guess.

## Scaling path

The in-memory store does exact brute-force search — exact and fast for tens of thousands of chunks. When the corpus outgrows memory, the `VectorStore` interface lets a `PgVectorStore` (Postgres + pgvector, HNSW index) drop in with no change to the pipeline. The stub documents the exact schema and query.

## Project structure

```
vector-forge/
├── vectorforge/
│   ├── chunking.py        # split documents with overlap
│   ├── embeddings.py      # Embedder interface + offline HashEmbedder
│   ├── similarity.py      # hand-written cosine + top-k
│   ├── vectorstore.py     # VectorStore interface + in-memory + pgvector stub
│   ├── pipeline.py        # ingest / retrieve / query
│   ├── llm.py             # grounded generation with citations
│   └── api.py             # FastAPI REST layer
├── eval/
│   ├── harness.py         # hit_rate@k and MRR
│   └── sample_qa.json     # labeled evaluation set
├── tests/                 # 41 tests across all modules
├── pyproject.toml
└── requirements.txt
```

## Tech stack

Python · FastAPI · NumPy · Pydantic · pytest. Optional real providers: Voyage / OpenAI (embeddings), Anthropic Claude (generation).

---

Built by **Manan Ghodasara** — [github.com/purpoint](https://github.com/purpoint)
