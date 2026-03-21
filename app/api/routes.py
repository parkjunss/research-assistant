from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional

from app.agents.orchestrator import research_graph, rebuild_graph
from app.agents.rag_agent import ingest_document
from app.core.state import AgentState
from app.db.postgres import (
    save_conversation,
    get_history,
    get_all_agents,
    get_agent_by_name,
    create_agent,
    update_agent,
    delete_agent,
)
from app.db.vector_store import get_rag_store
from app.core.logger import get_logger

import pypdf
import io

logger = get_logger("routes")

router = APIRouter()


# ── 공통 스키마 ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    session_id: str


# ── 에이전트 스키마 ───────────────────────────────────────────

class AgentCreateRequest(BaseModel):
    name: str = Field(..., description="에이전트 고유 이름 (영문, 중복 불가)")
    system_prompt: str = Field(..., description="에이전트 역할 정의 프롬프트")
    position: int = Field(
        ...,
        ge=10, le=89,
        description="파이프라인 순서 (10~89). 낮을수록 먼저 실행. 10=search, 20=summarize, 30=critic 주의",
    )
    enabled: bool = Field(True, description="활성화 여부")
    model_name: Optional[str] = Field(
        None,
        description=(
            "사용할 LLM 모델. 미지정 시 기본 LLM 사용.\n"
            "형식: 'ollama/<model>' 또는 'gemini/<model>'\n"
            "예: 'ollama/qwen2.5:14b', 'gemini/gemini-2.0-flash'"
        ),
    )

class AgentUpdateRequest(BaseModel):
    system_prompt: Optional[str] = None
    position: Optional[int] = Field(None, ge=10, le=89)
    enabled: Optional[bool] = None
    model_name: Optional[str] = Field(
        None,
        description="'ollama/<model>' | 'gemini/<model>' | null(기본 LLM으로 초기화)",
    )


# ── 기존 엔드포인트 ───────────────────────────────────────────

@router.get("/ping")
async def ping():
    return {"message": "pong"}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    logger.info(f"쿼리 수신: session={request.session_id} query={request.query}")
    try:
        initial_state: AgentState = {
            "query": request.query,
            "search_results": [],
            "summaries": [],
            "critique": None,
            "should_retry": False,
            "retry_count": 0,
            "final_answer": None,
            "memory_context": None,
            "rag_context": None,
            "messages": [{"role": "user", "content": request.query}],
        }

        result = await research_graph.ainvoke(initial_state)

        await save_conversation(
            session_id=request.session_id,
            query=request.query,
            answer=result["final_answer"],
        )

        logger.info(f"쿼리 완료: session={request.session_id}")
        return QueryResponse(
            answer=result["final_answer"],
            session_id=request.session_id,
        )

    except Exception as e:
        logger.error(f"쿼리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def history(session_id: str, limit: int = 10):
    logger.info(f"히스토리 조회: session={session_id}")
    try:
        records = await get_history(session_id=session_id, limit=limit)
        return {"session_id": session_id, "history": records}
    except Exception as e:
        logger.error(f"히스토리 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"문서 업로드: {file.filename}")
    try:
        content_bytes = await file.read()

        if file.filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            content = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif file.filename.endswith((".txt", ".md")):
            content = content_bytes.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="PDF, TXT, MD 파일만 지원합니다.")

        chunk_count = await ingest_document(
            filename=file.filename,
            content=content,
            metadata={"content_type": file.content_type},
        )

        return {
            "filename":    file.filename,
            "chunk_count": chunk_count,
            "message":     f"문서 업로드 완료 ({chunk_count}개 청크)",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    try:
        store = get_rag_store()
        results = store.similarity_search("", k=100)
        filenames = list(set([doc.metadata.get("filename", "unknown") for doc in results]))
        return {"documents": filenames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 에이전트 CRUD ─────────────────────────────────────────────

@router.get("/agents")
async def list_agents():
    """
    현재 등록된 모든 커스텀 에이전트 목록을 반환한다.
    내장 고정 노드(memory, rag, format 등)는 포함되지 않는다.
    """
    try:
        agents = await get_all_agents()
        return {"agents": agents, "count": len(agents)}
    except Exception as e:
        logger.error(f"에이전트 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents", status_code=201)
async def add_agent(request: AgentCreateRequest):
    """
    커스텀 에이전트를 추가하고 파이프라인 그래프를 재빌드한다.

    position 주의사항:
    - 10 = search, 20 = summarize, 30 = critic (내장 노드와 같은 position 지정 시 대체됨)
    - 11~19: search 직후, 21~29: summarize 직후 등으로 중간 삽입 가능
    """
    try:
        agent = await create_agent(
            name=request.name,
            system_prompt=request.system_prompt,
            position=request.position,
            enabled=request.enabled,
            model_name=request.model_name,
        )
        await rebuild_graph()
        logger.info(f"에이전트 추가 완료: {request.name} (position={request.position})")
        return {"agent": agent, "message": f"에이전트 '{request.name}' 추가 완료. 그래프 재빌드됨."}

    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"에이전트 추가 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/agents/{name}")
async def modify_agent(name: str, request: AgentUpdateRequest):
    """
    에이전트 설정을 부분 수정하고 그래프를 재빌드한다.
    수정 가능 필드: system_prompt, position, enabled, model_name
    """
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    # enabled=False 는 위 필터에서 빠지므로 별도 처리
    if request.enabled is not None:
        updates["enabled"] = request.enabled
    if not updates:
        raise HTTPException(status_code=400, detail="수정할 필드가 없습니다.")

    try:
        agent = await update_agent(name, **updates)
        await rebuild_graph()
        logger.info(f"에이전트 수정 완료: {name}")
        return {"agent": agent, "message": f"에이전트 '{name}' 수정 완료. 그래프 재빌드됨."}

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"에이전트 수정 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{name}", status_code=200)
async def remove_agent(name: str):
    """
    커스텀 에이전트를 삭제하고 그래프를 재빌드한다.
    내장 고정 노드(search, summarize, critic 등)는 삭제되지 않는다.
    """
    try:
        await delete_agent(name)
        await rebuild_graph()
        logger.info(f"에이전트 삭제 완료: {name}")
        return {"message": f"에이전트 '{name}' 삭제 완료. 그래프 재빌드됨."}

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"에이전트 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))