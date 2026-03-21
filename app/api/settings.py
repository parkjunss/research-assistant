import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db.postgres import get_setting, upsert_setting
from app.core.logger import get_logger

logger = get_logger("settings")

router = APIRouter()

WORKSPACE_KEY = "workspace_path"
DEFAULT_WORKSPACE = "/tmp/research_workspace"

class WorkspaceRequest(BaseModel):
    path: str

@router.get("/workspace")
async def get_workspace():
    path = await get_setting(WORKSPACE_KEY, DEFAULT_WORKSPACE)
    return {"workspace_path": path}

@router.patch("/workspace")
async def update_workspace(request: WorkspaceRequest):
    path = request.path.strip()

    if not os.path.isabs(path):
        raise HTTPException(status_code=400, detail="절대 경로로 입력해 주세요")

    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"폴더 생성 실패: {e}")

    await upsert_setting(WORKSPACE_KEY, path)
    logger.info(f"작업 폴더 변경: {path}")
    return {"workspace_path": path}