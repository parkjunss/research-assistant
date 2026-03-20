from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Text, DateTime, Integer, select
from datetime import datetime, timezone
from app.core.config import settings

DATABASE_URL = (
    f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def save_conversation(session_id: str, query: str, answer: str):
    async with AsyncSessionLocal() as session:
        record = ConversationHistory(
            session_id=session_id,
            query=query,
            answer=answer,
        )
        session.add(record)
        await session.commit()

async def get_history(session_id: str, limit: int = 10) -> list:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ConversationHistory)
            .where(ConversationHistory.session_id == session_id)
            .order_by(ConversationHistory.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "id": row.id,
                "session_id": row.session_id,
                "query": row.query,
                "answer": row.answer,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]