from fastapi.testclient import TestClient

from rag_book_agent.web import app as web_app


def test_status_endpoint_exposes_no_secret():
    client = TestClient(web_app.app)
    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "generation" in data
    assert "api_key" not in str(data).lower()


def test_upload_rejects_unsupported_type():
    client = TestClient(web_app.app)
    response = client.post(
        "/api/upload",
        files=[("files", ("unsupported.docx", b"test", "application/octet-stream"))],
    )

    assert response.status_code == 400


def test_upload_and_ask_complete_local_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "PROJECT_DIR", tmp_path)
    client = TestClient(web_app.app)
    response = client.post(
        "/api/upload",
        files=[
            (
                "files",
                (
                    "notes.md",
                    "# Reranking\n\nReranking improves selected evidence.",
                    "text/markdown",
                ),
            )
        ],
    )
    assert response.status_code == 200
    assert response.json()["import"]["imported"] == 1

    answer = client.post("/api/ask", json={"question": "What does reranking improve?"})
    assert answer.status_code == 200
    assert answer.json()["sources"]
    assert answer.json()["mode"] == "extractive"
