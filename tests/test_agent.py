from fastapi.testclient import TestClient

from rag_book_agent.web import app as web_app


def test_langgraph_agent_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "PROJECT_DIR", tmp_path)
    client = TestClient(web_app.app)
    uploaded = client.post(
        "/api/upload",
        files=[("files", ("agent.md", "# RAG\n\nRRF combines retrieval results.", "text/markdown"))],
    )
    assert uploaded.status_code == 200
    response = client.post("/api/ask", json={"question": "RRF combines what?", "agent_engine": "langgraph"})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"].startswith("langgraph-")
    assert data["retrieval"]["agent"] is True
    assert "local_search" in data["retrieval"]["agent_tools"]
