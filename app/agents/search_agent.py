from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.state import AgentState
from app.core.config import settings
from app.core.tools import get_today_date
from app.core.utils import get_llm
from app.core.logger import get_logger
from app.core.prompts import SEARCH_OPTIMIZE_PROMPT

logger = get_logger("search_agent")

search_tool = DuckDuckGoSearchResults(
    num_results=settings.duckduckgo_max_results,
    output_format="list",
)



def make_search_node(model_name: str | None = None):
    """model_name을 주입한 search_node를 반환하는 팩토리."""
    llm = get_llm(model_name)

    async def node(state: AgentState) -> AgentState:
        return await _run_search(state, llm)

    node.__name__ = "search_node"
    return node


async def _run_search(state: AgentState, llm) -> AgentState:
    query = state["query"]

    if state.get("critique"):
        improved = state["critique"].get("improved_query")
        if improved:
            logger.info(f"재검색 쿼리 적용: {improved}")
            query = improved

    try:
        llm_with_tools = llm.bind_tools([get_today_date])

        response = await llm_with_tools.ainvoke([
            SystemMessage(content=SEARCH_OPTIMIZE_PROMPT),
            HumanMessage(content=query),
        ])

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "get_today_date":
                    date_result = get_today_date.invoke({})
                    logger.info(f"날짜 툴 호출: {date_result}")
                    query = f"{query} ({date_result})"

        if response.content and isinstance(response.content, str):
            optimized_query = response.content.strip()
            if optimized_query:
                logger.info(f"쿼리 최적화: {query} → {optimized_query}")
                query = optimized_query

    except Exception as e:
        logger.warning(f"쿼리 최적화 실패, 원본 쿼리 사용: {e}")

    logger.info(f"검색 시작: {query}")
    try:
        results = search_tool.invoke(query)
        logger.info(f"검색 완료: {len(results)}개 결과")
    except Exception as e:
        logger.error(f"검색 실패: {e}")
        results = []

    return {
        **state,
        "search_results": results,
        "retry_count": state.get("retry_count", 0),
    }


async def search_node(state: AgentState) -> AgentState:
    """기본 LLM을 사용하는 search 노드 (model_name 미지정 시 fallback)."""
    return await _run_search(state, get_llm())