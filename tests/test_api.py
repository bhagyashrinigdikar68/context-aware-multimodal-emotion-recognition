
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


def test_health_endpoint_returns_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_rejects_unsupported_file_type():
    response = client.post(
        "/analyze",
        files={"file": ("document.txt", b"test content", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
