from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, select
from datetime import datetime
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

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    query      = Column(Text, nullable=False)
    answer     = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AppSettings(Base):
    __tablename__ = "app_settings"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    key        = Column(String(64), unique=True, nullable=False)
    value      = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentConfig(Base):
    """
    커스텀 에이전트 설정 테이블.

    position 규칙 (고정 노드는 건드리지 않음):
        0   = memory_retrieve  (고정)
        5   = rag_retrieve     (고정)
        10  = search           (기본 동적)
        20  = summarize        (기본 동적)
        30  = critic           (기본 동적)
        90  = format           (고정)
        95  = memory_save      (고정)

    커스텀 에이전트는 10~89 사이 임의 position 지정 가능.
    같은 position 불허 (unique 제약).
    """
    __tablename__ = "agent_configs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String(100), unique=True, nullable=False)
    system_prompt = Column(Text, nullable=False)
    position      = Column(Integer, unique=True, nullable=False)
    enabled       = Column(Boolean, default=True, nullable=False)
    model_name    = Column(String(100), nullable=True)   # None → 기본 LLM (#4 연결 포인트)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── 기존 함수 ────────────────────────────────────────────────

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
                "id":         row.id,
                "session_id": row.session_id,
                "query":      row.query,
                "answer":     row.answer,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]


async def get_setting(key: str, default: str = "") -> str:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AppSettings).where(AppSettings.key == key)
        )
        row = result.scalar_one_or_none()
        return row.value if row else default


async def upsert_setting(key: str, value: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AppSettings).where(AppSettings.key == key)
        )
        row = result.scalar_one_or_none()
        if row:
            row.value      = value
            row.updated_at = datetime.utcnow()
        else:
            session.add(AppSettings(key=key, value=value))
        await session.commit()


# ── AgentConfig CRUD ─────────────────────────────────────────

async def get_all_agents() -> list[dict]:
    """전체 에이전트 목록을 position 순으로 반환."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AgentConfig).order_by(AgentConfig.position)
        )
        rows = result.scalars().all()
        return [_agent_to_dict(row) for row in rows]


async def get_agent_by_name(name: str) -> dict | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AgentConfig).where(AgentConfig.name == name)
        )
        row = result.scalar_one_or_none()
        return _agent_to_dict(row) if row else None


async def create_agent(
    name: str,
    system_prompt: str,
    position: int,
    enabled: bool = True,
    model_name: str | None = None,
) -> dict:
    """
    커스텀 에이전트를 추가한다.
    name 또는 position 중복 시 ValueError.
    """
    _validate_position(position)

    async with AsyncSessionLocal() as session:
        # 중복 검사
        dup_name = await session.execute(
            select(AgentConfig).where(AgentConfig.name == name)
        )
        if dup_name.scalar_one_or_none():
            raise ValueError(f"이미 존재하는 에이전트 이름입니다: {name}")

        dup_pos = await session.execute(
            select(AgentConfig).where(AgentConfig.position == position)
        )
        if dup_pos.scalar_one_or_none():
            raise ValueError(f"이미 사용 중인 position입니다: {position}")

        agent = AgentConfig(
            name=name,
            system_prompt=system_prompt,
            position=position,
            enabled=enabled,
            model_name=model_name,
        )
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        return _agent_to_dict(agent)


async def update_agent(name: str, **kwargs) -> dict:
    """
    에이전트 설정을 부분 수정한다.
    수정 가능 필드: system_prompt, position, enabled, model_name
    """
    allowed = {"system_prompt", "position", "enabled", "model_name"}
    invalid = set(kwargs) - allowed
    if invalid:
        raise ValueError(f"수정 불가능한 필드: {invalid}")

    if "position" in kwargs:
        _validate_position(kwargs["position"])

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AgentConfig).where(AgentConfig.name == name)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise KeyError(f"에이전트를 찾을 수 없습니다: {name}")

        # position 중복 체크 (자기 자신 제외)
        if "position" in kwargs:
            dup = await session.execute(
                select(AgentConfig)
                .where(AgentConfig.position == kwargs["position"])
                .where(AgentConfig.name != name)
            )
            if dup.scalar_one_or_none():
                raise ValueError(f"이미 사용 중인 position입니다: {kwargs['position']}")

        for field, value in kwargs.items():
            setattr(agent, field, value)
        agent.updated_at = datetime.utcnow()

        await session.commit()
        await session.refresh(agent)
        return _agent_to_dict(agent)


async def delete_agent(name: str) -> None:
    """커스텀 에이전트를 삭제한다."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AgentConfig).where(AgentConfig.name == name)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise KeyError(f"에이전트를 찾을 수 없습니다: {name}")
        await session.delete(agent)
        await session.commit()


# ── 내부 헬퍼 ────────────────────────────────────────────────

def _agent_to_dict(agent: AgentConfig) -> dict:
    return {
        "id":            agent.id,
        "name":          agent.name,
        "system_prompt": agent.system_prompt,
        "position":      agent.position,
        "enabled":       agent.enabled,
        "model_name":    agent.model_name,
        "created_at":    agent.created_at.isoformat(),
        "updated_at":    agent.updated_at.isoformat() if agent.updated_at else None,
    }


# position 10~89 만 허용 (고정 노드 범위 보호)
_RESERVED_POSITIONS = {0, 5, 90, 95}
_MIN_CUSTOM_POSITION = 10
_MAX_CUSTOM_POSITION = 89

def _validate_position(position: int) -> None:
    if position in _RESERVED_POSITIONS:
        raise ValueError(f"position {position}은 고정 노드 예약값입니다.")
    if not (_MIN_CUSTOM_POSITION <= position <= _MAX_CUSTOM_POSITION):
        raise ValueError(
            f"position은 {_MIN_CUSTOM_POSITION}~{_MAX_CUSTOM_POSITION} 범위여야 합니다."
        )