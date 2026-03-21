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
사용자 질문을 검색엔진에 최적화된 짧은 키워드로 변환하세요.
'오늘', '최근', '이번 주' 등 날짜 관련 표현이 있으면 반드시 get_today_date 툴을 호출하세요.

규칙:
- 반드시 검색 키워드만 반환하세요 (질문 형태 금지)
- 10단어 이내로 작성하세요
- 핵심 명사 위주로 작성하세요

예시:
- 입력: "LangGraph의 StateGraph는 어떻게 사용하나요?" → 출력: "LangGraph StateGraph 사용법 예제"
- 입력: "오늘 최신 AI 뉴스 알려줘" → 출력: "AI 뉴스 2026년 3월 21일"
"""


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


async def search_node(state: AgentState) -> AgentState:
    """기본 LLM을 사용하는 search 노드 (model_name 미지정 시 fallback)."""
    return await _run_search(state, get_llm())