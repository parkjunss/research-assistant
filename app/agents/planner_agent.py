"""
planner_agent.py

사용자 요청(query)과 RAG 컨텍스트(기획서 등)를 바탕으로
실행 가능한 작업 목록(task list)으로 분해한다.

결과는 RAG 벡터 스토어에 task_type 메타데이터를 태깅해서 저장하며,
이후 Writer Agent가 RAG에서 꺼내 사용한다.
"""

import json
import re
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.core.utils import get_llm
from app.core.prompts import PLANNER_PROMPT
from app.db.vector_store import get_rag_store, get_memory_store
from app.core.logger import get_logger

logger = get_logger("planner_agent")


async def run_planner(
    query: str,
    session_id: str = "default",
    model_name: str | None = None,
) -> dict:
    """
    작업 분해를 실행하고 결과를 RAG에 저장한다.

    Args:
        query:      사용자 요청 (목표 또는 작업 설명)
        session_id: 대화 세션 ID (메모리 검색용)
        model_name: 사용할 LLM (None이면 기본 LLM)

    Returns:
        {
            "title":   str,
            "summary": str,
            "tasks":   List[dict],
            "chunks":  int,         # RAG에 저장된 청크 수
        }
    """
    # 1. 컨텍스트 수집 (RAG + 메모리)
    context = await _gather_context(query)
    logger.info(f"컨텍스트 수집 완료: {len(context)}자")

    # 2. LLM으로 작업 분해
    parsed = await _decompose(query, context, model_name)
    title   = parsed.get("title", "")
    summary = parsed.get("summary", "")
    tasks   = parsed.get("tasks", [])

    if not tasks:
        raise ValueError("작업 분해 결과가 없습니다. 요청을 더 구체적으로 작성해주세요.")

    # title 폴백
    if not title:
        title = query[:50]

    logger.info(f"작업 분해 완료: '{title}' — {len(tasks)}개 태스크")

    # 3. RAG 저장
    chunks = _save_to_rag(title, summary, tasks, query, session_id)
    logger.info(f"RAG 저장 완료: {chunks}개 청크")

    return {
        "title":   title,
        "summary": summary,
        "tasks":   tasks,
        "chunks":  chunks,
    }


# ── 컨텍스트 수집 ─────────────────────────────────────────────

async def _gather_context(query: str) -> str:
    """RAG(기획서 등)와 메모리에서 관련 컨텍스트를 가져온다."""
    parts = []

    try:
        rag_store = get_rag_store()
        rag_docs = rag_store.similarity_search(query, k=5)
        if rag_docs:
            rag_text = "\n\n".join([
                f"[{doc.metadata.get('section_type', 'doc')}] {doc.page_content}"
                for doc in rag_docs
            ])
            parts.append(f"=== 기획서 / 문서 컨텍스트 ===\n{rag_text}")
    except Exception as e:
        logger.warning(f"RAG 컨텍스트 수집 실패: {e}")

    try:
        mem_store = get_memory_store()
        mem_docs = mem_store.similarity_search(query, k=3)
        if mem_docs:
            mem_text = "\n\n".join([doc.page_content for doc in mem_docs])
            parts.append(f"=== 과거 대화 컨텍스트 ===\n{mem_text}")
    except Exception as e:
        logger.warning(f"메모리 컨텍스트 수집 실패: {e}")

    return "\n\n".join(parts) if parts else "없음"


# ── LLM 작업 분해 ─────────────────────────────────────────────

async def _decompose(query: str, context: str, model_name: str | None) -> dict:
    """LLM으로 작업을 분해한다. 실패 시 ValueError."""
    llm = get_llm(model_name)
    prompt = PLANNER_PROMPT.format(query=query, context=context)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # ```json ... ``` 펜스 제거
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"작업 분해 JSON 파싱 실패: {e}")
        raise ValueError(f"LLM 응답을 파싱할 수 없습니다: {e}")
    except Exception as e:
        logger.error(f"작업 분해 실패: {e}")
        raise


# ── RAG 저장 ──────────────────────────────────────────────────

def _save_to_rag(
    title: str,
    summary: str,
    tasks: list[dict],
    query: str,
    session_id: str,
) -> int:
    """
    분해된 태스크를 RAG에 저장한다.

    저장 단위:
    - 전체 요약 1개 Document (task_type="plan_summary")
    - 태스크별 1개 Document  (task_type="task")
    """
    store = get_rag_store()
    now   = datetime.utcnow().isoformat()
    docs  = []

    # 전체 요약 Document
    summary_content = (
        f"프로젝트: {title}\n"
        f"요약: {summary}\n"
        f"원본 요청: {query}\n"
        f"총 태스크 수: {len(tasks)}"
    )
    docs.append(Document(
        page_content=summary_content,
        metadata={
            "source":      "planner",
            "task_type":   "plan_summary",
            "plan_title":  title,
            "session_id":  session_id,
            "created_at":  now,
        },
    ))

    # 태스크별 Document
    for task in tasks:
        task_content = (
            f"[{task.get('type', 'other')} / {task.get('priority', 'P1')}] "
            f"{task.get('title', '')}\n"
            f"{task.get('description', '')}"
        )
        depends = task.get("depends_on", [])
        docs.append(Document(
            page_content=task_content,
            metadata={
                "source":      "planner",
                "task_type":   "task",
                "task_id":     task.get("id"),
                "task_title":  task.get("title", ""),
                "task_kind":   task.get("type", "other"),
                "priority":    task.get("priority", "P1"),
                "depends_on":  ",".join(str(d) for d in depends),
                "plan_title":  title,
                "session_id":  session_id,
                "created_at":  now,
            },
        ))

    store.add_documents(docs)
    return len(docs)