import asyncio
from langchain_core.messages import HumanMessage
from app.core.state import AgentState
from app.core.prompts import SUMMARIZE_PROMPT
from app.core.utils import get_llm
from app.core.logger import get_logger

logger = get_logger("summarizer_agent")


def make_summarize_node(model_name: str | None = None):
    """model_name을 주입한 summarize_node를 반환하는 팩토리."""
    llm = get_llm(model_name)

    async def node(state: AgentState) -> AgentState:
        return await _run_summarize(state, llm)

    node.__name__ = "summarize_node"
    return node


async def _run_summarize(state: AgentState, llm) -> AgentState:
    search_results = state["search_results"]
    query = state["query"]
    memory_context = state.get("memory_context", "")
    rag_context = state.get("rag_context", "")

    if not search_results:
        logger.warning("검색 결과 없음 — 요약 스킵")
        return {**state, "summaries": []}

    logger.info(f"요약 시작: {len(search_results)}개 청크")

    async def summarize_chunk(chunk):
        try:
            prompt = SUMMARIZE_PROMPT.format(
                search_results=chunk,
                query=query,
                memory_context=memory_context or "없음",
                rag_context=rag_context or "없음",
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


async def summarize_node(state: AgentState) -> AgentState:
    """기본 LLM을 사용하는 summarize 노드 (model_name 미지정 시 fallback)."""
    return await _run_summarize(state, get_llm())