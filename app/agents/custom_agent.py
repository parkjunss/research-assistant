"""
custom_agent.py

system_prompt 하나만 받아서 AgentState를 처리하는 범용 노드 함수를 반환하는 팩토리.
커스텀 에이전트는 이전 노드들의 결과(summaries, search_results 등)를
컨텍스트로 받아 messages 에 결과를 추가한다.
"""

from app.core.state import AgentState
from app.core.utils import get_llm
from app.core.logger import get_logger
from langchain_core.messages import SystemMessage, HumanMessage

logger = get_logger("custom_agent")


def make_custom_node(name: str, system_prompt: str, model_name: str | None = None):
    """
    Args:
        name:          에이전트 이름 (로그용)
        system_prompt: 이 에이전트의 역할 정의 프롬프트
        model_name:    사용할 LLM 모델 (None이면 기본 LLM)
                       형식: "ollama/qwen2.5:14b" | "gemini/gemini-2.0-flash" | None

    Returns:
        async node 함수 (AgentState → dict)
    """
    llm = get_llm(model_name)
    logger.info(
        f"[{name}] 노드 생성 — "
        f"model: {model_name or 'default'} / "
        f"llm: {type(llm).__name__}"
    )

    async def node(state: AgentState) -> dict:
        logger.info(f"[{name}] 커스텀 에이전트 실행")

        # 이전 파이프라인 결과를 컨텍스트로 조립
        context_parts = []

        if state.get("memory_context"):
            context_parts.append(f"[장기 메모리]\n{state['memory_context']}")

        if state.get("rag_context"):
            context_parts.append(f"[문서 컨텍스트]\n{state['rag_context']}")

        if state.get("summaries"):
            joined = "\n".join(state["summaries"])
            context_parts.append(f"[요약 결과]\n{joined}")

        context = "\n\n".join(context_parts) if context_parts else "없음"

        user_content = (
            f"사용자 질문: {state['query']}\n\n"
            f"현재까지 수집된 컨텍스트:\n{context}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        response = await llm.ainvoke(messages)
        result_text = response.content

        logger.info(f"[{name}] 완료 — {len(result_text)}자")

        return {
            "messages": [{"role": "assistant", "content": f"[{name}] {result_text}"}],
        }

    # 디버깅용으로 함수 이름 지정
    node.__name__ = f"custom_node_{name}"
    return node