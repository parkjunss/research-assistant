import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.agents.formatter_agent import format_node

@pytest.mark.asyncio
async def test_format_node_success(state_with_summaries):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="## 답변\nLangGraph는 에이전트 라이브러리입니다."
    ))

    with patch("app.agents.formatter_agent.get_llm", return_value=mock_llm):
        result = await format_node(state_with_summaries)

    assert result["final_answer"].startswith("## 답변")
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"

@pytest.mark.asyncio
async def test_format_node_llm_failure(state_with_summaries):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM 오류"))

    with patch("app.agents.formatter_agent.get_llm", return_value=mock_llm):
        with pytest.raises(Exception, match="LLM 오류"):
            await format_node(state_with_summaries)