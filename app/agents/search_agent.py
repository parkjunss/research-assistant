from langchain_community.tools import DuckDuckGoSearchResults
from app.core.state import AgentState
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("search_agent")

search_tool = DuckDuckGoSearchResults(
    num_results=settings.duckduckgo_max_results,
    output_format="list",
)

async def search_node(state: AgentState) -> AgentState:
    query = state["query"]

    if state.get("critique"):
        improved = state["critique"].get("improved_query")
        if improved:
            logger.info(f"재검색 쿼리 적용: {improved}")
            query = improved

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