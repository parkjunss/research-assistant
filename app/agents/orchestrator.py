"""
orchestrator.py

고정 노드 (memory_retrieve, rag_retrieve, format, memory_save) 는 항상 포함.
DB의 agent_configs 테이블을 position 순으로 읽어 동적 노드를 중간에 삽입한다.

position 구조:
    0   memory_retrieve  (고정)
    5   rag_retrieve     (고정)
    10  search           (기본 동적 — DB에 없으면 기본값으로 포함)
    20  summarize        (기본 동적 — DB에 없으면 기본값으로 포함)
    30  critic           (기본 동적 — DB에 없으면 기본값으로 포함)
    10~89 커스텀 에이전트 (DB에서 로드)
    90  format           (고정)
    95  memory_save      (고정)
"""

import asyncio
from langgraph.graph import StateGraph, END

from app.core.state import AgentState
from app.agents.search_agent import search_node
from app.agents.summarizer_agent import summarize_node
from app.agents.critic_agent import critic_node
from app.agents.formatter_agent import format_node
from app.agents.memory_agent import memory_retrieve_node, memory_save_node
from app.agents.rag_agent import rag_retrieve_node
from app.agents.custom_agent import make_custom_node
from app.core.logger import get_logger

logger = get_logger("orchestrator")

# ── 기본 동적 노드 정의 (position 기반) ──────────────────────
# DB에 해당 position 이 없을 때 사용하는 내장 노드 매핑
_BUILTIN_NODES: dict[int, tuple[str, callable]] = {
    10: ("search",    search_node),
    20: ("summarize", summarize_node),
    30: ("critic",    critic_node),
}

# critic 노드 이름 (조건 분기 판단에 사용)
_CRITIC_NODE_NAME = "critic"


def should_retry(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    should = state.get("should_retry", False)
    logger.info(f"재시도 판단 — should_retry: {should} / retry_count: {retry_count}")
    if should and retry_count < 2:
        return "search"
    return "format"


def _build_dynamic_sequence(agent_configs: list[dict]) -> list[tuple[str, callable]]:
    """
    DB의 agent_configs 와 내장 노드를 합쳐 position 순 노드 시퀀스를 반환.
    enabled=False 인 노드는 제외.

    Returns:
        [(node_name, node_fn), ...]  — position 오름차순
    """
    # DB에 있는 position 목록
    db_positions = {cfg["position"] for cfg in agent_configs}

    sequence: list[tuple[int, str, callable]] = []

    # 내장 노드: DB에 같은 position 이 있으면 대체, 없으면 기본값 사용
    for pos, (name, fn) in _BUILTIN_NODES.items():
        # DB에 이 position 을 덮어쓰는 커스텀 에이전트가 있으면 건너뜀
        if pos not in db_positions:
            sequence.append((pos, name, fn))

    # DB 커스텀 에이전트 추가
    for cfg in agent_configs:
        if not cfg["enabled"]:
            logger.info(f"비활성 에이전트 건너뜀: {cfg['name']} (position={cfg['position']})")
            continue
        node_fn = make_custom_node(cfg["name"], cfg["system_prompt"], cfg.get("model_name"))
        sequence.append((cfg["position"], cfg["name"], node_fn))

    # position 오름차순 정렬
    sequence.sort(key=lambda x: x[0])
    return [(name, fn) for _, name, fn in sequence]


async def build_graph_async():
    """DB에서 에이전트 설정을 읽어 그래프를 빌드한다."""
    from app.db.postgres import get_all_agents
    agent_configs = await get_all_agents()

    dynamic_nodes = _build_dynamic_sequence(agent_configs)

    graph = StateGraph(AgentState)

    # ── 고정 노드 등록 ──
    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("rag_retrieve",    rag_retrieve_node)
    graph.add_node("format",          format_node)
    graph.add_node("memory_save",     memory_save_node)

    # ── 동적 노드 등록 ──
    for name, fn in dynamic_nodes:
        graph.add_node(name, fn)
        logger.info(f"노드 등록: {name}")

    # ── 엣지 연결 ──
    graph.set_entry_point("memory_retrieve")
    graph.add_edge("memory_retrieve", "rag_retrieve")

    # rag_retrieve → 첫 번째 동적 노드 (없으면 format 으로)
    if dynamic_nodes:
        graph.add_edge("rag_retrieve", dynamic_nodes[0][0])
    else:
        graph.add_edge("rag_retrieve", "format")

    # 동적 노드들 순서대로 연결
    for i, (name, _) in enumerate(dynamic_nodes):
        is_last = (i == len(dynamic_nodes) - 1)

        if name == _CRITIC_NODE_NAME:
            # critic 은 조건 분기
            # retry 대상: search 가 있으면 search, 없으면 첫 번째 동적 노드
            retry_target = _find_retry_target(dynamic_nodes)
            graph.add_conditional_edges(
                _CRITIC_NODE_NAME,
                should_retry,
                {"search": retry_target, "format": "format"},
            )
        elif is_last:
            graph.add_edge(name, "format")
        else:
            next_name = dynamic_nodes[i + 1][0]
            # critic 바로 다음 노드는 conditional_edges 가 담당하므로 스킵
            if name != _CRITIC_NODE_NAME:
                graph.add_edge(name, next_name)

    graph.add_edge("format",      "memory_save")
    graph.add_edge("memory_save", END)

    compiled = graph.compile()
    logger.info(f"그래프 빌드 완료 — 동적 노드: {[n for n, _ in dynamic_nodes]}")
    return compiled


def _find_retry_target(dynamic_nodes: list[tuple[str, callable]]) -> str:
    """재시도 시 돌아갈 노드 이름. search 가 있으면 search, 없으면 첫 번째 노드."""
    names = [n for n, _ in dynamic_nodes]
    return "search" if "search" in names else (names[0] if names else "format")


def build_graph():
    """동기 컨텍스트에서 호출 가능한 래퍼 (앱 시작 시 사용)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 이벤트 루프가 실행 중이면 (FastAPI lifespan 등) future 반환
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, build_graph_async())
                return future.result()
        else:
            return loop.run_until_complete(build_graph_async())
    except RuntimeError:
        return asyncio.run(build_graph_async())


async def rebuild_graph():
    """
    API에서 에이전트 추가/삭제/수정 후 호출.
    전역 research_graph 를 갱신한다.
    """
    global research_graph
    research_graph = await build_graph_async()
    logger.info("research_graph 재빌드 완료")


# 앱 시작 시 초기 그래프 (DB가 비어있으면 기본 구성)
research_graph = build_graph()