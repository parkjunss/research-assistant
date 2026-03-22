from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    gemini_api_key: str
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    embedding_model: str = "bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Search
    duckduckgo_max_results: int = 5

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # PostgreSQL
    db_host: str = "localhost"
    db_port: int = 5433
    db_name: str = "research_db"
    db_user: str = "dev"
    db_password: str = "dev"

    # SMTP
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    # App
    app_env: str = "development"
    app_port: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()