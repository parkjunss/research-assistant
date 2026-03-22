from langchain_core.messages import HumanMessage, SystemMessage
from app.core.state import AgentState
from app.core.prompts import CODE_AGENT_PROMPT
from app.core.utils import get_llm
from app.db.vector_store import get_rag_store
from app.db.postgres import get_agent_model_name
from app.core.logger import get_logger

logger = get_logger("code_agent")

# 코딩 관련 키워드
CODE_KEYWORDS = {
    "코드", "구현", "함수", "클래스", "메서드", "알고리즘",
    "프로그램", "스크립트", "코딩", "개발", "작성해줘",
    "만들어줘", "짜줘", "debug", "버그", "에러", "오류",
    "리팩토링", "최적화", "테스트 코드", "unittest", "pytest",
    "code", "implement", "function", "class", "algorithm",
}

def is_coding_question(query: str) -> bool:
    """쿼리가 코딩 관련 질문인지 판단한다."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in CODE_KEYWORDS)

async def code_node(state: AgentState) -> AgentState:
    """코딩 관련 질문을 처리하는 에이전트"""
    query = state["query"]
    logger.info(f"Code Agent 실행: {query[:50]}...")

    # 컨텍스트 수집 (RAG + 메모리)
    context_parts = []

    if state.get("rag_context"):
        context_parts.append(f"=== 관련 문서 ===\n{state['rag_context']}")

    if state.get("memory_context"):
        context_parts.append(f"=== 과거 대화 ===\n{state['memory_context']}")

    if state.get("summaries"):
        newline = '\n'
        context_parts.append(f"=== 검색 결과 요약 ===\n{newline.join(state['summaries'])}")

    context = "\n\n".join(context_parts) if context_parts else "없음"

    try:
        model_name = await get_agent_model_name("code")
        llm = get_llm(model_name)

        prompt = CODE_AGENT_PROMPT.format(
            query=query,
            context=context,
        )

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        logger.info("Code Agent 완료")

        return {
            **state,
            "final_answer": response.content,
            "messages": [{"role": "assistant", "content": response.content}],
        }

    except Exception as e:
        logger.error(f"Code Agent 실패: {e}")
        raise