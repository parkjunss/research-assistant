from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.core.config import settings


def get_llm(model_name: str | None = None):
    """
    LLM 인스턴스를 반환한다.

    Args:
        model_name: 사용할 모델. 형식:
            - None         → 환경변수 기반 기본 LLM
            - "ollama/<model>"  → Ollama, 지정 모델
            - "gemini/<model>"  → Google Gemini, 지정 모델
            - "<model>"         → 현재 app_env 프로바이더로 지정 모델 사용

    Examples:
        get_llm()                          # 기본 LLM
        get_llm("ollama/qwen2.5:14b")      # Ollama 지정 모델
        get_llm("gemini/gemini-2.0-flash") # Gemini 지정 모델
    """
    if model_name is None:
        return _default_llm()

    # 프로바이더 prefix 파싱
    if "/" in model_name:
        provider, model = model_name.split("/", 1)
    else:
        # prefix 없으면 현재 환경의 프로바이더 사용
        provider = "gemini" if settings.app_env == "production" else "ollama"
        model = model_name

    if provider == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model,
        )

    if provider == "gemini":
        try:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.gemini_api_key,
            )
        except Exception:
            # Gemini 실패 시 Ollama 폴백
            return ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )

    raise ValueError(
        f"지원하지 않는 프로바이더: '{provider}'. "
        f"'ollama' 또는 'gemini' 를 사용하세요."
    )


def _default_llm():
    """환경변수 기반 기본 LLM."""
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