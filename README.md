# AI Research Assistant

멀티 에이전트 아키텍처 기반의 AI 리서치 어시스턴트입니다.
사용자 질문에 대해 검색 → 요약 → 검증 → 응답 파이프라인을 자동으로 실행합니다.

## 아키텍처

```
사용자 질문
  → Search Agent    (DuckDuckGo 검색)
  → Summarizer Agent (map-reduce 요약)
  → Critic Agent    (사실 검증 + 재검색 판단)
  → Formatter Agent (마크다운 응답 생성)
```

## 기술 스택

| 역할 | 기술 |
|---|---|
| API 서버 | FastAPI, Uvicorn |
| 에이전트 오케스트레이션 | LangGraph |
| LLM | Google Gemini (production), Ollama (development) |
| 검색 | DuckDuckGo Search |
| 데이터베이스 | PostgreSQL (대화 히스토리) |
| 캐시 / 큐 | Redis |
| 테스트 | pytest, pytest-asyncio |
| 패키지 관리 | uv |
| 컨테이너 | Docker, Docker Compose |

## 시작하기

### 사전 요구사항

- Docker, Docker Compose
- Ollama (로컬 개발 시)
- Google Gemini API 키 (production 시)

### 환경 변수 설정

```bash
cp .env.example .env
# .env 파일에 API 키 입력
```

### 개발 환경 실행

인프라만 Docker로 실행하고 FastAPI는 로컬에서 실행합니다.

```bash
# 인프라 실행 (Redis, PostgreSQL)
docker compose -f docker-compose.dev.yml up -d

# Ollama 모델 준비
ollama pull qwen2.5:14b

# 서버 실행
uv run uvicorn app.main:app --reload --port 8000
```

### 전체 스택 실행 (포트폴리오 시연)

```bash
docker compose up --build
```

## API 명세

### 질문하기

```
POST /api/v1/query
```

```json
// Request
{
  "query": "LangGraph란 무엇인가요?",
  "session_id": "user-123"
}

// Response
{
  "answer": "## 답변\nLangGraph는...",
  "session_id": "user-123"
}
```

### 대화 히스토리 조회

```
GET /api/v1/history/{session_id}?limit=10
```

```json
// Response
{
  "session_id": "user-123",
  "history": [
    {
      "id": 1,
      "session_id": "user-123",
      "query": "LangGraph란?",
      "answer": "## 답변\nLangGraph는...",
      "created_at": "2026-03-20T12:00:00"
    }
  ]
}
```

### 서버 상태 확인

```
GET /health
```

## 에이전트 설계 의사결정

### LangGraph를 선택한 이유
Critic Agent의 검증 결과에 따라 Search Agent로 되돌아가는 조건부 분기가 필요했습니다.
단순 순차 실행이 아닌 루프 구조를 명시적으로 표현하기 위해 LangGraph의 `StateGraph`와 `add_conditional_edges`를 활용했습니다.

### 재시도 상한을 2회로 설정한 이유
무한 루프 방지와 API 비용 제어 사이의 트레이드오프를 고려했습니다.
검증 실패 시 개선된 쿼리로 재검색하지만, 2회 초과 시 현재까지의 결과로 응답을 생성합니다.

### Gemini / Ollama 전략
`APP_ENV` 환경변수로 LLM을 전환합니다.
- `development` → Ollama (로컬, 무료)
- `production` → Gemini (빠른 응답, API 비용 발생)

## 테스트

```bash
uv run pytest -v
```

```
16 passed in 0.51s
```

## 프로젝트 구조

```
app/
├── main.py
├── api/
│   └── routes.py
├── agents/
│   ├── orchestrator.py
│   ├── search_agent.py
│   ├── summarizer_agent.py
│   ├── critic_agent.py
│   └── formatter_agent.py
├── core/
│   ├── config.py
│   ├── state.py
│   ├── prompts.py
│   ├── utils.py
│   └── logger.py
└── db/
    ├── postgres.py
    └── redis_client.py
```