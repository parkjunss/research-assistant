import asyncio
from langgraph.graph import StateGraph, END

from app.core.state import AgentState
from app.agents.search_agent import search_node, make_search_node
from app.agents.summarizer_agent import summarize_node, make_summarize_node
from app.agents.critic_agent import critic_node, make_critic_node
from app.agents.formatter_agent import format_node, make_format_node
from app.agents.memory_agent import memory_retrieve_node, memory_save_node
from app.agents.rag_agent import rag_retrieve_node
from app.agents.custom_agent import make_custom_node
from app.agents.code_agent import code_node, is_coding_question
from app.agents.reasoning_agent import reasoning_node, route_after_reasoning
from app.agents.chat_agent import chat_node
from app.core.logger import get_logger

logger = get_logger("orchestrator")

_CRITIC_NODE_NAME = "critic"

# 내장 노드 팩토리 매핑: name → (position, factory_fn, fallback_fn)
_BUILTIN_FACTORIES = {
    "search":    (10, make_search_node,    search_node),
    "summarize": (20, make_summarize_node, summarize_node),
    "critic":    (30, make_critic_node,    critic_node),
    "format":    (40, make_format_node,    format_node),
}


def should_retry(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    should = state.get("should_retry", False)
    logger.info(f"재시도 판단 — should_retry: {should} / retry_count: {retry_count}")
    if should and retry_count < 2:
        return "search"
    return "format"


def _build_dynamic_sequence(agent_configs: list[dict]) -> list[tuple[str, callable]]:
    """
    DB의 agent_configs(내장+커스텀)를 position 순 노드 시퀀스로 변환.
    enabled=False 인 노드는 제외.
    내장 에이전트는 model_name을 팩토리에 주입, 없으면 fallback 함수 사용.

    Returns:
        [(node_name, node_fn), ...]  — position 오름차순
    """
    # DB 설정을 name으로 인덱싱
    cfg_by_name = {cfg["name"]: cfg for cfg in agent_configs}

    sequence: list[tuple[int, str, callable]] = []

    # 내장 에이전트 처리
    for name, (pos, factory_fn, fallback_fn) in _BUILTIN_FACTORIES.items():
        cfg = cfg_by_name.get(name)

        if cfg and not cfg["enabled"]:
            logger.info(f"내장 에이전트 비활성 건너뜀: {name}")
            continue

        model_name = cfg.get("model_name") if cfg else None

        if model_name:
            node_fn = factory_fn(model_name)
            logger.info(f"내장 에이전트 model 주입: {name} → {model_name}")
        else:
            node_fn = fallback_fn

        sequence.append((pos, name, node_fn))

    # 커스텀 에이전트 처리
    for cfg in agent_configs:
        if cfg["is_builtin"]:
            continue
        if not cfg["enabled"]:
            logger.info(f"커스텀 에이전트 비활성 건너뜀: {cfg['name']} (position={cfg['position']})")
            continue
        node_fn = make_custom_node(cfg["name"], cfg["system_prompt"], cfg.get("model_name"))
        sequence.append((cfg["position"], cfg["name"], node_fn))

    sequence.sort(key=lambda x: x[0])
    return [(name, fn) for _, name, fn in sequence]

async def build_graph_async():
    from app.db.postgres import get_all_agents
    agent_configs = await get_all_agents()

    dynamic_nodes = _build_dynamic_sequence(agent_configs)
    node_names = [n for n, _ in dynamic_nodes]

    graph = StateGraph(AgentState)

    # 1. 노드 등록 (동일)
    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("memory_save", memory_save_node)
    graph.add_node("code", code_node)
    graph.add_node("chat", chat_node)

    for name, fn in dynamic_nodes:
        graph.add_node(name, fn)

    # 2. 고정 엣지 연결
    graph.set_entry_point("memory_retrieve")
    graph.add_edge("memory_retrieve", "rag_retrieve")
    graph.add_edge("rag_retrieve", "reasoning")

    # 3. 타겟 노드 이름 미리 정의
    node_map = {name: name for name, _ in dynamic_nodes}
    first = node_names[0] if node_names else "memory_save"
    search_node_name = node_map.get("search", first)
    summarize_node_name = node_map.get("summarize", first)

    # [핵심] Reasoning 결과에 따른 지능형 분기
    graph.add_conditional_edges("reasoning", route_after_reasoning, {
        "chat": "chat",
        "code": "code",
        "search": search_node_name,    # 정보 부족 -> 검색부터
        "planning": search_node_name, 
        "writing": summarize_node_name # 정보 충분 -> 검색 건너뛰고 요약부터!
    })

    # 4. 동적 노드 간 엣지 순회 연결
    for i, (name, _) in enumerate(dynamic_nodes):
        is_last = (i == len(dynamic_nodes) - 1)
        next_name = dynamic_nodes[i + 1][0] if not is_last else "memory_save"

        if name == _CRITIC_NODE_NAME:
            # critic 실패 시 돌아갈 곳 명시
            retry_target = "search" if "search" in node_names else first
            graph.add_conditional_edges(
                _CRITIC_NODE_NAME,
                should_retry,
                {"search": retry_target, "format": next_name},
            )
        else:
            graph.add_edge(name, next_name)

    # 5. 마무리 엣지
    graph.add_edge("chat", "memory_save")
    graph.add_edge("code", "memory_save")
    graph.add_edge("memory_save", END)

    compiled = graph.compile()
    logger.info(f"그래프 빌드 완료 — 동적 노드: {node_names}")
    return compiled


async def init_graph():
    """lifespan에서 init_db() 이후 호출. 초기 그래프를 빌드한다."""
    global research_graph
    research_graph = await build_graph_async()
    logger.info("초기 그래프 빌드 완료")


async def rebuild_graph():
    """API에서 에이전트 추가/삭제/수정 후 호출. 전역 research_graph를 갱신한다."""
    global research_graph
    research_graph = await build_graph_async()
    logger.info("research_graph 재빌드 완료")


research_graph = None  # lifespan의 init_graph()에서 초기화