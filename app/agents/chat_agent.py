from langchain_core.messages import HumanMessage
from app.core.state import AgentState
from app.core.utils import get_llm
from app.core.logger import get_logger

logger = get_logger("chat_agent")

CHAT_PROMPT = """
당신은 친절한 AI 어시스턴트입니다.
아래 메시지에 자연스럽게 답변하세요.
검색이나 문서 참조 없이 바로 답변하세요.

메시지: {query}
"""

async def chat_node(state: AgentState) -> AgentState:
    query = state["query"]
    logger.info(f"Chat Agent 실행: {query[:50]}")

    try:
        llm = get_llm()
        prompt = CHAT_PROMPT.format(query=query)
        response = await llm.ainvoke([HumanMessage(content=prompt)])

        logger.info("Chat Agent 완료")
        return {
            **state,
            "final_answer": response.content,
            "messages": [{"role": "assistant", "content": response.content}],
        }
    except Exception as e:
        logger.error(f"Chat Agent 실패: {e}")
        raise