from langgraph.graph import StateGraph, END
from app.core.state import AgentState
from app.agents.search_agent import search_node
from app.agents.summarizer_agent import summarize_node
from app.agents.critic_agent import critic_node
from app.agents.formatter_agent import format_node
from app.core.logger import get_logger

logger = get_logger("critic_agent")

def should_retry(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    should = state.get("should_retry", False)
    logger.info(f"재시도 판단 — should_retry: {should} / retry_count: {retry_count}")
    if should and retry_count < 2:
        return "search"
    return "format"

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search",    search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("critic",    critic_node)
    graph.add_node("format",    format_node)

    graph.set_entry_point("search")
    graph.add_edge("search",    "summarize")
    graph.add_edge("summarize", "critic")
    graph.add_conditional_edges("critic", should_retry, {
        "search": "search",
        "format": "format",
    })
    graph.add_edge("format", END)

    return graph.compile()

research_graph = build_graph()