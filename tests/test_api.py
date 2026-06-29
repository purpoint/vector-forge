from fastapi.testclient import TestClient
from vectorforge.api import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_then_query_flow():
    # ingest a document
    r1 = client.post("/ingest", json={"text": "The Eiffel Tower is 330 metres tall.", "doc_id": "eiffel"})
    assert r1.status_code == 200
    assert r1.json()["chunks_added"] >= 1

    # query it
    r2 = client.post("/query", json={"question": "How tall is the Eiffel Tower?"})
    assert r2.status_code == 200
    body = r2.json()
    assert "answer" in body
    assert len(body["context"]) >= 1


def test_query_validation_rejects_bad_input():
    # missing required 'question' field -> FastAPI auto-rejects with 422
    r = client.post("/query", json={})
    assert r.status_code == 422