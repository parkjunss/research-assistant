from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
from app.core.config import settings
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, select, text

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
    에이전트 설정 테이블. 내장 에이전트와 커스텀 에이전트를 함께 관리한다.

    is_builtin=True  → 내장 에이전트 (model_name만 수정 가능)
    is_builtin=False → 커스텀 에이전트 (모든 필드 수정 가능)

    position 규칙:
        0   = memory_retrieve  (고정, DB 미관리)
        5   = rag_retrieve     (고정, DB 미관리)
        10  = search           (내장)
        20  = summarize        (내장)
        30  = critic           (내장)
        40  = format           (내장)
        90  = memory_save      (고정, DB 미관리)

    커스텀 에이전트는 10~89 사이 임의 position 지정 가능 (내장과 중복 불가).
    """
    __tablename__ = "agent_configs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String(100), unique=True, nullable=False)
    system_prompt = Column(Text, nullable=False)
    position      = Column(Integer, unique=True, nullable=False)
    enabled       = Column(Boolean, default=True, nullable=False)
    is_builtin    = Column(Boolean, default=False, nullable=False)
    model_name    = Column(String(100), nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# 내장 에이전트 seed 데이터
_BUILTIN_AGENTS = [
    {"name": "search",    "position": 10, "system_prompt": "DuckDuckGo 검색 및 쿼리 최적화"},
    {"name": "summarize", "position": 20, "system_prompt": "검색 결과 map-reduce 요약"},
    {"name": "critic",    "position": 30, "system_prompt": "사실 검증 및 재검색 판단"},
    {"name": "format",    "position": 40, "system_prompt": "마크다운 응답 생성"},
    {"name": "code",      "position": 50, "system_prompt": "코드 생성, 리뷰, 버그 분석"},
]

# ── 기존 함수 ────────────────────────────────────────────────


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)
        # langchain_pg_embedding 테이블에 FTS 인덱스 추가
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_embedding_fts
            ON langchain_pg_embedding
            USING gin(to_tsvector('simple', document))
        """))
    await _seed_builtin_agents()


async def _seed_builtin_agents():
    """내장 에이전트가 DB에 없으면 초기 데이터를 삽입한다."""
    async with AsyncSessionLocal() as session:
        for data in _BUILTIN_AGENTS:
            exists = await session.execute(
                select(AgentConfig).where(AgentConfig.name == data["name"])
            )
            if not exists.scalar_one_or_none():
                session.add(AgentConfig(
                    name=data["name"],
                    system_prompt=data["system_prompt"],
                    position=data["position"],
                    enabled=True,
                    is_builtin=True,
                    model_name=None,
                ))
        await session.commit()


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
    - 내장 에이전트: model_name만 수정 가능
    - 커스텀 에이전트: system_prompt, position, enabled, model_name 수정 가능
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AgentConfig).where(AgentConfig.name == name)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise KeyError(f"에이전트를 찾을 수 없습니다: {name}")

        if agent.is_builtin:
            allowed = {"model_name"}
        else:
            allowed = {"system_prompt", "position", "enabled", "model_name"}

        invalid = set(kwargs) - allowed
        if invalid:
            raise ValueError(
                f"수정 불가능한 필드: {invalid}. "
                f"{'내장 에이전트는 model_name만 수정 가능합니다.' if agent.is_builtin else ''}"
            )

        if "position" in kwargs:
            _validate_position(kwargs["position"])
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
    """커스텀 에이전트를 삭제한다. 내장 에이전트는 삭제 불가."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AgentConfig).where(AgentConfig.name == name)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise KeyError(f"에이전트를 찾을 수 없습니다: {name}")
        if agent.is_builtin:
            raise ValueError(f"내장 에이전트는 삭제할 수 없습니다: {name}")
        await session.delete(agent)
        await session.commit()


async def get_agent_model_name(name: str) -> str | None:
    """에이전트의 model_name을 반환한다. 없으면 None (기본 LLM 사용)."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AgentConfig.model_name).where(AgentConfig.name == name)
        )
        row = result.scalar_one_or_none()
        return row  # None이면 기본 LLM


# ── 내부 헬퍼 ────────────────────────────────────────────────

def _agent_to_dict(agent: AgentConfig) -> dict:
    return {
        "id":            agent.id,
        "name":          agent.name,
        "system_prompt": agent.system_prompt,
        "position":      agent.position,
        "enabled":       agent.enabled,
        "is_builtin":    agent.is_builtin,
        "model_name":    agent.model_name,
        "created_at":    agent.created_at.isoformat(),
        "updated_at":    agent.updated_at.isoformat() if agent.updated_at else None,
    }


# position 예약값: 고정 노드 + 내장 에이전트
_RESERVED_POSITIONS = {0, 5, 10, 20, 30, 40, 90}
_MIN_CUSTOM_POSITION = 10
_MAX_CUSTOM_POSITION = 89

def _validate_position(position: int) -> None:
    if position in _RESERVED_POSITIONS:
        raise ValueError(
            f"position {position}은 예약값입니다 (고정 노드 또는 내장 에이전트). "
            f"예약된 position: {sorted(_RESERVED_POSITIONS)}"
        )
    if not (_MIN_CUSTOM_POSITION <= position <= _MAX_CUSTOM_POSITION):
        raise ValueError(
            f"position은 {_MIN_CUSTOM_POSITION}~{_MAX_CUSTOM_POSITION} 범위여야 합니다."
        )