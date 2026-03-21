from langchain_core.messages import HumanMessage, SystemMessage
from app.core.state import AgentState
from app.core.prompts import FORMAT_PROMPT
from app.core.utils import get_llm
from app.core.tools import create_file, read_file, get_workspace_path
from app.core.logger import get_logger

logger = get_logger("formatter_agent")

FORMATTER_SYSTEM_PROMPT = """당신은 리포트 작성 전문가입니다.
아래 요약본을 읽기 좋은 마크다운 형식으로 구성하세요.
사용자가 파일 저장을 요청하면 create_file 툴을 호출하세요.
사용자가 파일 읽기를 요청하면 read_file 툴을 호출하세요.
형식 외 추가 설명은 하지 마세요."""

async def format_node(state: AgentState) -> AgentState:
    llm = get_llm()
    llm_with_tools = llm.bind_tools([create_file, read_file, get_workspace_path])

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
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=FORMATTER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "create_file":
                    result = create_file.invoke(tool_call["args"])
                    logger.info(f"파일 생성: {result}")
                elif tool_call["name"] == "read_file":
                    result = read_file.invoke(tool_call["args"])
                    logger.info(f"파일 읽기: {result}")

        logger.info("최종 답변 포맷 완료")
        return {
            **state,
            "final_answer": response.content,
            "messages": [{"role": "assistant", "content": response.content}],
        }
    except Exception as e:
        logger.error(f"포맷 실패: {e}")
        raise