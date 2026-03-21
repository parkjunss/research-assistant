from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.state import AgentState
from app.db.vector_store import get_rag_store
from app.core.logger import get_logger

logger = get_logger("rag_agent")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# 쿼리 키워드 → section_type 매핑
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
    """쿼리에서 section_type 힌트를 감지한다."""
    for keyword, section_type in SECTION_TYPE_HINTS.items():
        if keyword in query:
            return section_type
    return None

async def rag_retrieve_node(state: AgentState) -> AgentState:
    """질문과 관련된 문서 청크 검색"""
    try:
        store = get_rag_store()
        query = state["query"]

        # section_type 힌트 감지
        section_type = _detect_section_type(query)

        if section_type:
            # 전체 검색 후 section_type 필터링
            all_docs = store.similarity_search(query, k=20)
            filtered = [
                doc for doc in all_docs
                if doc.metadata.get("section_type") == section_type
            ]
            # 필터 결과 없으면 전체 결과 사용
            docs = filtered[:3] if filtered else all_docs[:3]
            logger.info(f"RAG 섹션 필터: {section_type} → {len(filtered)}개 중 {len(docs)}개")
        else:
            docs = store.similarity_search(query, k=3)
            logger.info(f"RAG 문서 {len(docs)}개 검색됨")

        rag_context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
        return {**state, "rag_context": rag_context}

    except Exception as e:
        logger.error(f"RAG 검색 실패: {e}")
        return {**state, "rag_context": ""}

async def ingest_document(filename: str, content: str, metadata: dict = {}) -> int:
    """문서를 청킹해서 벡터 스토어에 저장"""
    try:
        docs = text_splitter.create_documents(
            texts=[content],
            metadatas=[{"filename": filename, **metadata}],
        )
        store = get_rag_store()
        store.add_documents(docs)
        logger.info(f"문서 저장 완료: {filename} ({len(docs)}개 청크)")
        return len(docs)
    except Exception as e:
        logger.error(f"문서 저장 실패: {e}")
        raise