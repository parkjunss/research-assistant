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

async def rag_retrieve_node(state: AgentState) -> AgentState:
    """질문과 관련된 문서 청크 검색"""
    try:
        store = get_rag_store()
        docs = store.similarity_search(state["query"], k=3)

        if docs:
            rag_context = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"RAG 문서 {len(docs)}개 검색됨")
        else:
            rag_context = ""
            logger.info("관련 RAG 문서 없음")

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