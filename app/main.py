from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.postgres import init_db
from app.db.redis_client import get_redis, close_redis
from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await get_redis()
    print("DB and Redis connected")
    yield
    await close_redis()
    print("Shutdown complete")

app = FastAPI(
    title="AI Research Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}