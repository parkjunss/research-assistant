import pytest
from unittest.mock import patch, MagicMock
from app.agents.search_agent import search_node

@pytest.mark.asyncio
async def test_search_node_success(base_state):
    mock_results = [
        {"snippet": "LangGraph is great", "title": "LangGraph", "link": "https://example.com"},
    ]
    with patch("app.agents.search_agent.search_tool") as mock_tool:
        mock_tool.invoke.return_value = mock_results
        result = await search_node(base_state)

    assert result["search_results"] == mock_results
    assert result["retry_count"] == 0

@pytest.mark.asyncio
async def test_search_node_uses_improved_query(base_state):
    state = {
        **base_state,
        "critique": {"improved_query": "LangGraph 에이전트 오케스트레이션"},
    }
    with patch("app.agents.search_agent.search_tool") as mock_tool:
        mock_tool.invoke.return_value = []
        await search_node(state)
        mock_tool.invoke.assert_called_once_with("LangGraph 에이전트 오케스트레이션")

@pytest.mark.asyncio
async def test_search_node_failure_returns_empty(base_state):
    with patch("app.agents.search_agent.search_tool") as mock_tool:
        mock_tool.invoke.side_effect = Exception("Search API 오류")
        result = await search_node(base_state)

    assert result["search_results"] == []