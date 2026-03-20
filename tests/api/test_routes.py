import pytest
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_ping(async_client):
    response = await async_client.get("/api/v1/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}

@pytest.mark.asyncio
async def test_health(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_query_success(async_client):
    mock_result = {
        "query": "LangGraph란?",
        "search_results": [],
        "summaries": [],
        "critique": None,
        "should_retry": False,
        "retry_count": 0,
        "final_answer": "## 답변\nLangGraph는 에이전트 라이브러리입니다.",
        "messages": [{"role": "assistant", "content": "## 답변\nLangGraph는 에이전트 라이브러리입니다."}],
    }

    with patch("app.api.routes.research_graph") as mock_graph, \
         patch("app.api.routes.save_conversation", new_callable=AsyncMock):
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)
        response = await async_client.post("/api/v1/query", json={
            "query": "LangGraph란?",
            "session_id": "test-session",
        })

    assert response.status_code == 200
    assert response.json()["answer"].startswith("## 답변")
    assert response.json()["session_id"] == "test-session"

@pytest.mark.asyncio
async def test_query_agent_failure(async_client):
    with patch("app.api.routes.research_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("에이전트 오류"))
        response = await async_client.post("/api/v1/query", json={
            "query": "LangGraph란?",
            "session_id": "test-session",
        })

    assert response.status_code == 500

@pytest.mark.asyncio
async def test_history_success(async_client):
    mock_history = [
        {
            "id": 1,
            "session_id": "test-session",
            "query": "LangGraph란?",
            "answer": "## 답변\nLangGraph는...",
            "created_at": "2026-03-20T12:00:00",
        }
    ]

    with patch("app.api.routes.get_history", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_history
        response = await async_client.get("/api/v1/history/test-session")

    assert response.status_code == 200
    assert len(response.json()["history"]) == 1
    assert response.json()["history"][0]["query"] == "LangGraph란?"