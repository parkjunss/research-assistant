from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.orchestrator import research_graph
from app.core.state import AgentState
from app.db.postgres import save_conversation, get_history
from app.core.logger import get_logger

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