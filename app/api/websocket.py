import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage
from app.agents import orchestrator
from app.core.state import AgentState
from app.db.postgres import save_conversation
from app.core.logger import get_logger

logger = get_logger("websocket")

async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket 연결: session={session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            query = message.get("query", "").strip()

            if not query:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "query가 비어있습니다.",
                }))
                continue

            await _run_pipeline(websocket, session_id, query)

    except WebSocketDisconnect:
        logger.info(f"WebSocket 연결 종료: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass


async def _run_pipeline(websocket: WebSocket, session_id: str, query: str):
    await websocket.send_text(json.dumps({
        "type": "start",
        "query": query,
    }))

    initial_state: AgentState = {
        "query": query,
        "search_results": [],
        "summaries": [],
        "critique": None,
        "should_retry": False,
        "retry_count": 0,
        "final_answer": None,
        "memory_context": None,
        "rag_context": None,
        "is_coding": False,
        "route_type": None,
        "messages": [{"role": "user", "content": query}],
    }

    try:
        final_state = None

        # astream으로 노드 단위 이벤트 + 최종 상태 동시 수집
        async for event in orchestrator.research_graph.astream(
            initial_state,
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                await websocket.send_text(json.dumps({
                    "type": "agent_update",
                    "agent": node_name,
                    "output": _serialize_output(node_name, node_output),
                }))
                # 최종 상태 누적
                if final_state is None:
                    final_state = {**initial_state}
                final_state.update(node_output)

        final_answer = final_state.get("final_answer", "") if final_state else ""

        # 토큰 단위 스트리밍
        words = final_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            await websocket.send_text(json.dumps({
                "type": "token",
                "content": chunk,
            }))
            await asyncio.sleep(0.02)

        # 저장 + 완료
        await save_conversation(
            session_id=session_id,
            query=query,
            answer=final_answer,
        )

        await websocket.send_text(json.dumps({
            "type": "done",
            "answer": final_answer,
            "session_id": session_id,
        }))

    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e),
        }))
        

def _serialize_output(node_name: str, output: dict) -> dict:
    """노드 출력에서 직렬화 가능한 정보만 추출한다."""
    result = {}

    if node_name == "search":
        results = output.get("search_results", [])
        result["search_count"] = len(results)

    elif node_name == "summarize":
        summaries = output.get("summaries", [])
        result["summary_count"] = len(summaries)

    elif node_name == "critic":
        result["should_retry"] = output.get("should_retry", False)
        result["retry_count"] = output.get("retry_count", 0)

    elif node_name == "reasoning":
        result["route_type"] = output.get("route_type", "search")

    elif node_name == "memory_retrieve":
        ctx = output.get("memory_context", "")
        result["has_memory"] = bool(ctx)

    elif node_name == "rag_retrieve":
        ctx = output.get("rag_context", "")
        result["has_rag"] = bool(ctx)

    return result