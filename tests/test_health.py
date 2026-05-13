from fastapi.testclient import TestClient

from rag_pc4u.api.main import app


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "client_id" in body
    assert "collection" in body
