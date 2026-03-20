from langchain_core.messages import HumanMessage
from app.core.state import AgentState
from app.core.prompts import FORMAT_PROMPT
from app.core.utils import get_llm
from app.core.logger import get_logger

logger = get_logger("formatter_agent")

async def format_node(state: AgentState) -> AgentState:
    llm = get_llm()

    urls = [
        result.get("link", "")
        for result in state["search_results"]
        if result.get("link")
    ]
    urls_str = "\n".join(f"- {url}" for url in urls)

    logger.info("최종 답변 포맷 시작")
    try:
        prompt = FORMAT_PROMPT.format(
            query=state["query"],
            summaries="\n".join(state["summaries"]),
            urls=urls_str,
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        logger.info("최종 답변 포맷 완료")

        return {
            **state,
            "final_answer": response.content,
            "messages": [{"role": "assistant", "content": response.content}],
        }
    except Exception as e:
        logger.error(f"포맷 실패: {e}")
        raise