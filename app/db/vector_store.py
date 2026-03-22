from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
from app.core.logger import get_logger
import requests

from sentence_transformers import CrossEncoder
import asyncio

_reranker_model = None

logger = get_logger("vector_store")

COLLECTION_MEMORY = "long_term_memory"
COLLECTION_RAG    = "rag_documents"

CONNECTION_STRING = (
    f"postgresql+psycopg://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
)

ASYNC_CONNECTION_STRING = (
    f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
)

async_engine = create_async_engine(ASYNC_CONNECTION_STRING, echo=False)

def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )

def get_memory_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_MEMORY,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

def get_rag_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_RAG,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

async def hybrid_search(
    query: str,
    collection_name: str = COLLECTION_RAG,
    k: int = 5,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[Document]:
    """
    벡터 검색 + BM25 키워드 검색을 혼합한 하이브리드 검색.
    Reciprocal Rank Fusion(RRF)으로 결과를 통합한다.
    """
    store = get_rag_store() if collection_name == COLLECTION_RAG else get_memory_store()

    # 1. 벡터 검색 (상위 20개)
    vector_docs = store.similarity_search(query, k=20)
    vector_ranks = {doc.page_content: i + 1 for i, doc in enumerate(vector_docs)}

    # 2. BM25 키워드 검색 (PostgreSQL FTS)
    bm25_docs = await _bm25_search(query, collection_name, limit=20)
    bm25_ranks = {doc.page_content: i + 1 for i, doc in enumerate(bm25_docs)}

    # 3. RRF 점수 계산
    all_contents = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores: dict[str, float] = {}

    for content in all_contents:
        v_rank = vector_ranks.get(content, 21)  # 없으면 최하위
        b_rank = bm25_ranks.get(content, 21)
        # RRF 공식: 1/(k+rank), k=60 권장값
        rrf_scores[content] = (
            vector_weight * (1 / (60 + v_rank)) +
            bm25_weight   * (1 / (60 + b_rank))
        )

    # 4. 점수 기준 정렬 후 상위 k개 반환
    sorted_contents = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]

    # 5. 원본 Document 객체 복원
    content_to_doc: dict[str, Document] = {}
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in content_to_doc:
            content_to_doc[doc.page_content] = doc

    result = [content_to_doc[c] for c in sorted_contents if c in content_to_doc]
    logger.info(f"하이브리드 검색 완료: vector={len(vector_docs)} bm25={len(bm25_docs)} → merged={len(result)}")
    return result


async def _bm25_search(
    query: str,
    collection_name: str,
    limit: int = 20,
) -> list[Document]:
    """PostgreSQL FTS로 BM25 키워드 검색을 수행한다."""
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT
                        e.document,
                        e.cmetadata,
                        ts_rank(
                            to_tsvector('simple', e.document),
                            plainto_tsquery('simple', :query)
                        ) AS rank
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = :collection_name
                      AND to_tsvector('simple', e.document) @@ plainto_tsquery('simple', :query)
                    ORDER BY rank DESC
                    LIMIT :limit
                """),
                {
                    "query": query,
                    "collection_name": collection_name,
                    "limit": limit,
                }
            )
            rows = result.fetchall()

        docs = []
        for row in rows:
            import json
            metadata = row.cmetadata if isinstance(row.cmetadata, dict) else json.loads(row.cmetadata or "{}")
            docs.append(Document(page_content=row.document, metadata=metadata))

        return docs

    except Exception as e:
        logger.warning(f"BM25 검색 실패 → 빈 결과 반환: {e}")
        return []
    
def _get_reranker() -> CrossEncoder:
    """CrossEncoder 모델을 싱글톤으로 로드한다."""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Re-ranker 모델 로딩: {settings.reranker_model}")
        _reranker_model = CrossEncoder(
            settings.reranker_model,
            max_length=512,
        )
        logger.info("Re-ranker 모델 로딩 완료")
    return _reranker_model

async def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """
    CrossEncoder로 문서를 재순위화한다.
    실패 시 원본 순서 반환.
    """
    if not docs:
        return docs

    try:
        reranker = _get_reranker()

        # CrossEncoder 입력: [(query, doc), ...]
        pairs = [(query, doc.page_content) for doc in docs]

        # CPU 블로킹 작업을 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: reranker.predict(pairs)
        )

        # 점수 기준 정렬
        scored_docs = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked = [doc for _, doc in scored_docs[:top_k]]
        logger.info(f"Re-ranking 완료: {len(docs)}개 → {len(reranked)}개")
        return reranked

    except Exception as e:
        logger.warning(f"Re-ranking 실패 → 원본 순서 반환: {e}")
        return docs[:top_k]