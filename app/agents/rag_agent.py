from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.state import AgentState
from app.core.logger import get_logger
from app.db.vector_store import get_rag_store, hybrid_search_all_stores, rerank, COLLECTION_RAG, COLLECTION_MEMORY

logger = get_logger("rag_agent")

_general_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", ". ", " ", ""],
    length_function=len,
)

SECTION_TYPE_HINTS = {
    "기술 스택": "technical",
    "아키텍처": "technical",
    "에이전트 파이프라인": "technical",
    "인프라": "technical",
    "기능": "requirements",
    "요구사항": "requirements",
    "일정": "schedule",
    "마일스톤": "schedule",
    "리스크": "risk",
    "이해관계자": "stakeholder",
    "목적": "objective",
    "배경": "objective",
}

def _detect_section_type(query: str) -> str | None:
    for keyword, section_type in SECTION_TYPE_HINTS.items():
        if keyword in query:
            return section_type
    return None

async def rag_retrieve_node(state: AgentState) -> AgentState:
    """하이브리드 검색 + Re-ranking으로 관련 문서 청크 검색"""
    try:
        query = state["query"]
        
        # 1. 모든 저장소(RAG + Memory)에서 하이브리드 검색 수행
        # 이제 COLLECTION_RAG가 비어있어도 COLLECTION_MEMORY에서 긁어옵니다.
        docs = await hybrid_search_all_stores(
            query=query, 
            collections=[COLLECTION_RAG, COLLECTION_MEMORY], 
            k=20
        )

        # 2. Re-ranking (여기서 지식과 메모리 중 가장 관련성 높은 5개만 남김)
        docs = await rerank(query=query, docs=docs, top_k=5)

        rag_context = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"통합 검색 완료: 최종 {len(docs)}개 반환")
        
        return {**state, "rag_context": rag_context}

    except Exception as e:
        logger.error(f"RAG 검색 실패: {e}")
        return {**state, "rag_context": ""}
    

async def ingest_document(filename: str, content: str, metadata: dict = {}) -> int:
    """문서를 청킹해서 벡터 스토어에 저장"""
    try:
        if filename.endswith(".md"):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=80,
                separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            )
        elif filename.endswith(".pdf"):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", ". ", " ", ""],
            )
        else:
            splitter = _general_splitter

        chunks = splitter.split_text(content)
        docs = [
            Document(
                page_content=chunk,
                metadata={"filename": filename, **metadata},
            )
            for chunk in chunks
        ]

        store = get_rag_store()
        store.add_documents(docs)
        logger.info(f"문서 저장 완료: {filename} ({len(docs)}개 청크)")
        return len(docs)
    except Exception as e:
        logger.error(f"문서 저장 실패: {e}")
        raise

