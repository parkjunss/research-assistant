import json
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.state import AgentState
from app.core.prompts import FORMAT_PROMPT
from app.core.utils import get_llm
from app.core.tools import create_file, read_file, get_workspace_path, send_email
from app.core.logger import get_logger

logger = get_logger("formatter_agent")

# 프롬프트 보강: 한국어 고정 및 출처 표기 규칙 상세화
FORMATTER_SYSTEM_PROMPT = """당신은 리포트 작성 전문가입니다.
반드시 한국어로 답변하세요. 아래 요약본을 가독성이 뛰어난 마크다운 형식으로 구성하세요.

[출처 표기 규칙]
1. URL이 있다면 해당 링크를 '참고 URL' 섹션에 불렛 포인트로 나열하세요.
2. 내부 문서(RAG) 정보가 있다면 '[내부 문서] 파일명 > 섹션명' 형식으로 표기하세요.
3. 어떤 출처도 없다면 '내부 지식 및 대화 문맥'이라고 표기하세요.

사용자가 파일 저장/읽기/이메일 등을 요청하면 관련 툴을 호출하세요. 형식 외 추가 설명은 하지 마세요."""

_TOOLS = [create_file, read_file, get_workspace_path, send_email]

def make_format_node(model_name: str | None = None):
    llm = get_llm(model_name)
    async def node(state: AgentState) -> AgentState:
        return await _run_format(state, llm)
    node.__name__ = "format_node"
    return node

async def _run_format(state: AgentState, llm) -> AgentState:
    llm_with_tools = llm.bind_tools(_TOOLS)

    # 1. 출처 수집 로직 통합 (Search + RAG)
    all_sources = []
    
    # 외부 URL 추출
    if state.get("search_results"):
        for res in state["search_results"]:
            url = res.get("link") or res.get("url")
            if url:
                all_sources.append(f"- [참고 URL] {url}")

    # 내부 RAG 문서 추출 (파일명 및 섹션 활용)
    # rag_docs 객체가 있다면 그것을 사용하고, 없다면 텍스트에서 유추
    rag_docs = state.get("rag_docs", [])
    for doc in rag_docs:
        fname = doc.metadata.get("filename")
        heading = doc.metadata.get("section_heading", "본문")
        if fname:
            source_item = f"- [내부 문서] {fname} ({heading})"
            if source_item not in all_sources:
                all_sources.append(source_item)

    sources_str = "\n".join(all_sources) if all_sources else "내부 지식 및 대화 문맥"

    logger.info("최종 답변 포맷 시작")
    try:
        # 2. 프롬프트 구성 (urls -> sources로 확장)
        prompt = FORMAT_PROMPT.format(
            query=state["query"],
            summaries="\n".join(state["summaries"]),
            urls=sources_str, # 기존 프롬프트 템플릿의 {urls} 자리를 활용
        )
        
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=FORMATTER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        # 3. 툴 호출 처리 (기존 유지)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                if t_name == "create_file":
                    create_file.invoke(t_args)
                elif t_name == "read_file":
                    read_file.invoke(t_args)
                elif t_name == "send_email":
                    send_email.invoke(t_args)

        logger.info("최종 답변 포맷 완료")
        return {
            **state,
            "final_answer": response.content,
            "messages": [{"role": "assistant", "content": response.content}],
        }
    except Exception as e:
        logger.error(f"포맷 실패: {e}")
        raise
    

async def format_node(state: AgentState) -> AgentState:
    """기본 LLM을 사용하는 format 노드 (model_name 미지정 시 fallback)."""
    return await _run_format(state, get_llm())