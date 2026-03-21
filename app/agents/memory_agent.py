from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.state import AgentState
from app.core.utils import get_llm
from app.db.vector_store import get_memory_store
from app.core.logger import get_logger

logger = get_logger("memory_agent")

MEMORY_SYSTEM_PROMPT = """당신은 대화 기억 전문가입니다.
과거 관련 대화 내용을 참고해서 현재 질문에 더 나은 답변을 제공하세요.
과거 대화가 없거나 관련 없으면 무시하세요."""

async def memory_save_node(state: AgentState) -> AgentState:
    """대화 완료 후 장기 메모리에 저장"""
    try:
        store = get_memory_store()
        doc = Document(
            page_content=f"질문: {state['query']}\n답변: {state['final_answer']}",
            metadata={
                "query": state["query"],
                "session_id": state.get("session_id", "default"),
            }
        )
        store.add_documents([doc])
        logger.info(f"장기 메모리 저장 완료: {state['query'][:30]}...")
    except Exception as e:
        logger.error(f"장기 메모리 저장 실패: {e}")

    return state

async def memory_retrieve_node(state: AgentState) -> AgentState:
    """질문과 관련된 과거 대화 검색"""
    try:
        store = get_memory_store()
        docs = store.similarity_search(state["query"], k=3)

        if docs:
            memory_context = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"관련 메모리 {len(docs)}개 검색됨")
        else:
            memory_context = ""
            logger.info("관련 메모리 없음")

        return {
            **state,
            "memory_context": memory_context,
        }
    except Exception as e:
        logger.error(f"장기 메모리 검색 실패: {e}")
        return {**state, "memory_context": ""}