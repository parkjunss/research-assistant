from langchain_core.messages import HumanMessage
from app.core.state import AgentState
from app.core.prompts import CRITIC_PROMPT
from app.core.utils import get_llm
from app.core.logger import get_logger

logger = get_logger("critic_agent")

async def critic_node(state: AgentState) -> AgentState:
    llm = get_llm()

    if not state["summaries"]:
        logger.warning("요약 없음 — 검증 스킵, 재검색 트리거")
        return {
            **state,
            "should_retry": True,
            "retry_count": state.get("retry_count", 0) + 1,
            "critique": {"content": "요약 없음", "improved_query": state["query"]},
        }

    logger.info("검증 시작")
    try:
        prompt = CRITIC_PROMPT.format(
            search_results=state["search_results"],
            summaries="\n".join(state["summaries"]),
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content

        should_retry = "RETRY" in content.upper()
        improved_query = None

        if should_retry:
            for line in content.split("\n"):
                if "개선 쿼리" in line and ":" in line:
                    improved_query = line.split(":", 1)[-1].strip()
                    break
            logger.warning(f"검증 실패 — 재검색: {improved_query}")
        else:
            logger.info("검증 통과")

        return {
            **state,
            "should_retry": should_retry,
            "retry_count": state.get("retry_count", 0) + 1,
            "critique": {
                "content": content,
                "improved_query": improved_query or state["query"],
            },
        }
    except Exception as e:
        logger.error(f"검증 실패: {e}")
        return {
            **state,
            "should_retry": False,
            "retry_count": state.get("retry_count", 0) + 1,
            "critique": {"content": str(e), "improved_query": state["query"]},
        }