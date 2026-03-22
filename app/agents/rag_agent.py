from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.state import AgentState
from app.db.vector_store import get_rag_store, hybrid_search, COLLECTION_RAG
from app.core.logger import get_logger

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
    """하이브리드 검색으로 관련 문서 청크 검색"""
    try:
        query = state["query"]

        # 하이브리드 검색 실행
        docs = await hybrid_search(query=query, collection_name=COLLECTION_RAG, k=5)

        # section_type 필터링 (힌트 있을 때)
        section_type = _detect_section_type(query)
        if section_type and docs:
            filtered = [d for d in docs if d.metadata.get("section_type") == section_type]
            if filtered:
                docs = filtered[:3]
                logger.info(f"섹션 필터 적용: {section_type} → {len(docs)}개")

        rag_context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
        logger.info(f"RAG 하이브리드 검색 완료: {len(docs)}개")
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