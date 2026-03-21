from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.agents.orchestrator import research_graph
from app.agents.rag_agent import ingest_document
from app.core.state import AgentState
from app.db.postgres import save_conversation, get_history
from app.db.vector_store import get_rag_store
from app.core.logger import get_logger

import pypdf
import io

logger = get_logger("routes")

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    session_id: str

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
            "filename": file.filename,
            "chunk_count": chunk_count,
            "message": f"문서 업로드 완료 ({chunk_count}개 청크)",
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