import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.agents.summarizer_agent import summarize_node

@pytest.mark.asyncio
async def test_summarize_node_success(state_with_results):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="요약 결과입니다."))

    with patch("app.agents.summarizer_agent.get_llm", return_value=mock_llm):
        result = await summarize_node(state_with_results)

    assert len(result["summaries"]) == 2
    assert all(isinstance(s, str) for s in result["summaries"])

@pytest.mark.asyncio
async def test_summarize_node_empty_results(base_state):
    with patch("app.agents.summarizer_agent.get_llm"):
        result = await summarize_node(base_state)

    assert result["summaries"] == []

@pytest.mark.asyncio
async def test_summarize_node_partial_failure(state_with_results):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        MagicMock(content="첫 번째 요약"),
        Exception("LLM 오류"),
    ])

    with patch("app.agents.summarizer_agent.get_llm", return_value=mock_llm):
        result = await summarize_node(state_with_results)

    assert len(result["summaries"]) == 1
    assert result["summaries"][0] == "첫 번째 요약"