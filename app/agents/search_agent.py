from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.state import AgentState
from app.core.config import settings
from app.core.tools import get_today_date
from app.core.utils import get_llm
from app.core.logger import get_logger

logger = get_logger("search_agent")

search_tool = DuckDuckGoSearchResults(
    num_results=settings.duckduckgo_max_results,
    output_format="list",
)

QUERY_SYSTEM_PROMPT = """당신은 검색 쿼리 최적화 전문가입니다.
사용자 질문을 분석해서 검색에 최적화된 쿼리를 반환하세요.
'오늘', '최근', '이번 주', '2025년' 등 날짜 관련 표현이 있으면 반드시 get_today_date 툴을 호출해서 실제 날짜를 확인하세요.
최적화된 검색 쿼리만 텍스트로 반환하세요."""

async def search_node(state: AgentState) -> AgentState:
    query = state["query"]

    if state.get("critique"):
        improved = state["critique"].get("improved_query")
        if improved:
            logger.info(f"재검색 쿼리 적용: {improved}")
            query = improved

    try:
        llm = get_llm()
        llm_with_tools = llm.bind_tools([get_today_date])

        response = await llm_with_tools.ainvoke([
            SystemMessage(content=QUERY_SYSTEM_PROMPT),
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