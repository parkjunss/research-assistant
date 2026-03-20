import asyncio
from langchain_core.messages import HumanMessage
from app.core.state import AgentState
from app.core.config import settings
from app.core.prompts import SUMMARIZE_PROMPT
from app.core.utils import get_llm
from app.core.logger import get_logger

logger = get_logger("summarizer_agent")

async def summarize_node(state: AgentState) -> AgentState:
    llm = get_llm()
    search_results = state["search_results"]
    query = state["query"]

    if not search_results:
        logger.warning("검색 결과 없음 — 요약 스킵")
        return {**state, "summaries": []}

    logger.info(f"요약 시작: {len(search_results)}개 청크")

    async def summarize_chunk(chunk):
        try:
            prompt = SUMMARIZE_PROMPT.format(
                search_results=chunk,
                query=query,
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"청크 요약 실패: {e}")
            return ""

    summaries = await asyncio.gather(
        *[summarize_chunk(result) for result in search_results]
    )
    summaries = [s for s in summaries if s]

    logger.info(f"요약 완료: {len(summaries)}개")
    return {**state, "summaries": list(summaries)}