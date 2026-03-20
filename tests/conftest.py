import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from app.main import app
from app.core.state import AgentState

@pytest.fixture
def base_state() -> AgentState:
    return {
        "query": "LangGraph란?",
        "search_results": [],
        "summaries": [],
        "critique": None,
        "should_retry": False,
        "retry_count": 0,
        "final_answer": None,
        "messages": [{"role": "user", "content": "LangGraph란?"}],
    }

@pytest.fixture
def state_with_results(base_state) -> AgentState:
    return {
        **base_state,
        "search_results": [
            {"snippet": "LangGraph is a library", "title": "LangGraph", "link": "https://example.com"},
            {"snippet": "LangGraph helps build agents", "title": "Agents", "link": "https://example2.com"},
        ],
    }

@pytest.fixture
def state_with_summaries(state_with_results) -> AgentState:
    return {
        **state_with_results,
        "summaries": ["LangGraph는 에이전트 오케스트레이션 라이브러리입니다."],
    }

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client