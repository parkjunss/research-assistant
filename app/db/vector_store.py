from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
import requests
from app.core.config import settings
from app.core.logger import get_logger


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

async def hybrid_search_all_stores(
    query: str, 
    collections: list[str] = [COLLECTION_RAG, COLLECTION_MEMORY], 
    k: int = 15
) -> list[Document]:
    """모든 컬렉션에서 검색 결과를 가져와 통합한다."""
    all_docs = []
    
    for coll in collections:
        # 각 컬렉션별로 하이브리드 검색 수행
        docs = await hybrid_search(query=query, collection_name=coll, k=k)
        # 출처를 메타데이터에 명시 (디버깅용)
        for d in docs:
            d.metadata["origin_collection"] = coll
        all_docs.extend(docs)
    
    # 중복 제거 (내용 기준)
    seen = set()
    unique_docs = []
    for d in all_docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)
            
    return unique_docs

async def hybrid_search(
    query: str,
    collection_name: str = COLLECTION_RAG,
    k: int = 5,
    filter: dict = None,  # 필터 추가
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[Document]:
    """
    벡터 검색 + BM25 키워드 검색을 혼합한 하이브리드 검색.
    Reciprocal Rank Fusion(RRF)으로 결과를 통합한다.
    """
    store = get_rag_store() if collection_name == COLLECTION_RAG else get_memory_store()

    # 1. 벡터 검색 (상위 20개)
    vector_docs = store.similarity_search(query, k=20, filter=filter)
    vector_ranks = {doc.page_content: i + 1 for i, doc in enumerate(vector_docs)}

    # 2. BM25 키워드 검색 (PostgreSQL FTS)
    bm25_docs = await _bm25_search(query, collection_name, limit=20, filter=filter)
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
    filter: dict = None  # [필터 추가]
) -> list[Document]:
    try:
        # 필터 조건 동적 생성
        filter_clause = ""
        params = {"query": query, "collection_name": collection_name, "limit": limit}
        
        if filter:
            # metadata->>'key' = 'value' 형태의 조건을 동적으로 추가
            for key, value in filter.items():
                filter_clause += f" AND e.cmetadata->>'{key}' = :{key}"
                params[key] = str(value)

        async with async_engine.connect() as conn:
            result = await conn.execute(
                text(f"""
                    SELECT 
                        e.document, 
                        e.cmetadata,
                        ts_rank(to_tsvector('simple', e.document), plainto_tsquery('simple', :query)) AS rank
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = :collection_name
                      AND to_tsvector('simple', e.document) @@ plainto_tsquery('simple', :query)
                      {filter_clause}  -- [필터 조건 삽입]
                    ORDER BY rank DESC
                    LIMIT :limit
                """),
                params
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

async def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """
    Ollama embed API로 query-document 유사도를 계산해서 재순위화한다.
    실패 시 원본 순서 반환.
    """
    if not docs:
        return docs

    try:
        # query + doc 쌍을 하나의 배열로 임베딩
        inputs = [f"query: {query}"] + [f"passage: {doc.page_content}" for doc in docs]

        response = requests.post(
            f"{settings.ollama_base_url}/api/embed",
            json={
                "model": settings.reranker_model,
                "input": inputs,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = data["embeddings"]
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # 코사인 유사도 계산
        scores = [_cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs]

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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도를 계산한다."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# DB와 직접 소통하는 로직은 여기서만 관리
async def deactivate_documents_by_filename(filename: str):
    async with async_engine.begin() as conn:
        await conn.execute(
            text("""
                UPDATE langchain_pg_embedding 
                SET cmetadata = cmetadata || '{"is_active": false}'::jsonb 
                WHERE cmetadata->>'filename' = :filename
                  AND (cmetadata->>'is_active')::boolean = true
            """),
            {"filename": filename}
        )