from langchain_core.messages import HumanMessage
from app.core.state import AgentState
from app.core.prompts import REASONING_PROMPT
from app.core.utils import get_llm
from app.agents.code_agent import is_coding_question
from app.core.logger import get_logger

logger = get_logger("reasoning_agent")

VALID_TYPES = {"code", "planning", "writing", "search"}

async def reasoning_node(state: AgentState) -> AgentState:
    """질문 유형을 LLM으로 판단하고 route_type을 설정한다."""
    query = state["query"]

    context_parts = []
    if state.get("rag_context"):
        context_parts.append(state["rag_context"])
    if state.get("memory_context"):
        context_parts.append(state["memory_context"])
    context = "\n".join(context_parts) or "없음"

    try:
        llm = get_llm()
        prompt = REASONING_PROMPT.format(query=query, context=context)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        route_type = response.content.strip().lower()

        if route_type not in VALID_TYPES:
            logger.warning(f"알 수 없는 유형: {route_type} → 키워드 폴백")
            route_type = _keyword_fallback(query)
        else:
            logger.info(f"Reasoning Agent 판단: {route_type}")

    except Exception as e:
        logger.error(f"Reasoning Agent 실패 → 키워드 폴백: {e}")
        route_type = _keyword_fallback(query)

    return {**state, "route_type": route_type}


def _keyword_fallback(query: str) -> str:
    """LLM 실패 시 키워드 기반 폴백."""
    if is_coding_question(query):
        logger.info("키워드 폴백 → code")
        return "code"
    logger.info("키워드 폴백 → search")
    return "search"


def route_after_reasoning(state: AgentState) -> str:
    """reasoning_node 결과로 다음 노드를 결정한다."""
    return state.get("route_type", "search")