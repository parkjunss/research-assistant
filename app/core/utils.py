from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.core.config import settings


def get_llm():
    if settings.app_env == "development":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    try:
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
        )
    except Exception:
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )