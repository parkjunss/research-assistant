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
    # 1. 모든 소스를 수집
    search_results = state.get("search_results", [])
    query = state["query"]
    memory_context = state.get("memory_context", "")
    rag_context = state.get("rag_context", "")

    # [수정] 검색 결과는 없더라도 메모리나 RAG 컨텍스트가 있다면 진행!
    if not search_results and not rag_context and not memory_context:
        logger.warning("요약할 소스가 전혀 없음 (검색/RAG/메모리 모두 비어있음)")
        return {**state, "summaries": []}

    # 2. 요약 대상 리스트 생성
    # 검색 결과가 있다면 그것들을 쓰고, 없다면 RAG 컨텍스트 자체를 하나의 청크로 취급
    chunks_to_summarize = search_results if search_results else [rag_context or memory_context]
    
    logger.info(f"요약 시작: {len(chunks_to_summarize)}개 소스 활용")

    async def summarize_chunk(chunk_content):
        try:
            # 텍스트가 너무 짧으면 요약할 필요 없음 (방어 로직)
            if not chunk_content or len(str(chunk_content)) < 10:
                return ""
                
            prompt = SUMMARIZE_PROMPT.format(
                search_results=chunk_content, # 여기가 실제 요약 대상
                query=query,
                memory_context=memory_context or "없음",
                rag_context=rag_context or "없음",
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"요약 실패: {e}")
            return ""

    # 3. 병렬 처리
    summaries = await asyncio.gather(
        *[summarize_chunk(c) for c in chunks_to_summarize]
    )
    summaries = [s for s in summaries if s]

    logger.info(f"요약 완료: {len(summaries)}개")
    return {**state, "summaries": list(summaries)}


async def summarize_node(state: AgentState) -> AgentState:
    """기본 LLM을 사용하는 summarize 노드 (model_name 미지정 시 fallback)."""
    return await _run_summarize(state, get_llm())