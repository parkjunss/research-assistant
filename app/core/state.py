from typing import TypedDict, List, Optional, Annotated
import operator

class AgentState(TypedDict):
    query: str
    search_results: List[dict]
    summaries: List[str]
    critique: Optional[dict]
    should_retry: bool
    retry_count: int
    final_answer: Optional[str]
    memory_context: Optional[str]
    rag_context: Optional[str]
    messages: Annotated[List[dict], operator.add]
    plan_tasks: Optional[List[dict]]