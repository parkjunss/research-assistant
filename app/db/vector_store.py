from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from app.core.config import settings

COLLECTION_MEMORY = "long_term_memory"
COLLECTION_RAG = "rag_documents"

CONNECTION_STRING = (
    f"postgresql+psycopg://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
)

def get_embeddings():
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model="nomic-embed-text",
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