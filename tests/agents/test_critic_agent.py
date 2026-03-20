import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.agents.critic_agent import critic_node

@pytest.mark.asyncio
async def test_critic_node_pass(state_with_summaries):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="통과 여부: PASS\n이유: 요약이 정확합니다.\n개선 쿼리: 없음"
    ))

    with patch("app.agents.critic_agent.get_llm", return_value=mock_llm):
        result = await critic_node(state_with_summaries)

    assert result["should_retry"] is False
    assert result["retry_count"] == 1

@pytest.mark.asyncio
async def test_critic_node_retry(state_with_summaries):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="통과 여부: RETRY\n이유: 부정확한 정보\n개선 쿼리: LangGraph 공식 문서"
    ))

    with patch("app.agents.critic_agent.get_llm", return_value=mock_llm):
        result = await critic_node(state_with_summaries)

    assert result["should_retry"] is True
    assert result["critique"]["improved_query"] == "LangGraph 공식 문서"

@pytest.mark.asyncio
async def test_critic_node_empty_summaries(base_state):
    with patch("app.agents.critic_agent.get_llm"):
        result = await critic_node(base_state)

    assert result["should_retry"] is True