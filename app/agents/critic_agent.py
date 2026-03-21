from langchain_core.messages import HumanMessage
from app.core.state import AgentState
from app.core.prompts import CRITIC_PROMPT
from app.core.utils import get_llm
from app.core.logger import get_logger

logger = get_logger("critic_agent")


def make_critic_node(model_name: str | None = None):
    """model_name을 주입한 critic_node를 반환하는 팩토리."""
    llm = get_llm(model_name)

    async def node(state: AgentState) -> AgentState:
        return await _run_critic(state, llm)

    node.__name__ = "critic_node"
    return node


async def _run_critic(state: AgentState, llm) -> AgentState:
    if not state["summaries"]:
        logger.warning("요약 없음 — 검증 스킵, 재검색 트리거")
        return {
            **state,
            "should_retry": True,
            "retry_count": state.get("retry_count", 0) + 1,
            "critique": {"content": "요약 없음", "improved_query": state["query"]},
        }

    logger.info("검증 시작")
    improved_query = None
    try:
        prompt = CRITIC_PROMPT.format(
            search_results=state["search_results"],
            summaries="\n".join(state["summaries"]),
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content

        should_retry = "RETRY" in content.upper()

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
        logger.warning(f"검증 실패 — retry_count: {state.get('retry_count', 0)} / 에러: {e}")
        return {
            **state,
            "should_retry": False,
            "retry_count": state.get("retry_count", 0) + 1,
            "critique": {"content": str(e), "improved_query": state["query"]},
        }


async def critic_node(state: AgentState) -> AgentState:
    """기본 LLM을 사용하는 critic 노드 (model_name 미지정 시 fallback)."""
    return await _run_critic(state, get_llm())