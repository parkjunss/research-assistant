# AI Research Assistant

멀티 에이전트 아키텍처 기반의 AI 리서치 어시스턴트입니다.
사용자 질문에 대해 검색 → 요약 → 검증 → 응답 파이프라인을 자동으로 실행합니다.

## 아키텍처

```
사용자 질문
  → Memory Retrieve  (장기 메모리 검색)
  → RAG Retrieve     (업로드 문서 검색)
  → Search Agent     (DuckDuckGo 검색 + 날짜 툴)
  → Summarizer Agent (map-reduce 요약)
  → Critic Agent     (사실 검증 + 재검색 판단)
  → Formatter Agent  (마크다운 응답 + 파일/이메일 툴)
  → Memory Save      (장기 메모리 저장)
```

## 기술 스택

| 역할 | 기술 |
|---|---|
| API 서버 | FastAPI, Uvicorn |
| 에이전트 오케스트레이션 | LangGraph |
| LLM | Google Gemini (production), Ollama (development) |
| 임베딩 | Ollama nomic-embed-text |
| 검색 | DuckDuckGo Search |
| 벡터 DB | pgvector (PostgreSQL 확장) |
| 데이터베이스 | PostgreSQL (대화 히스토리, 설정) |
| 캐시 | Redis |
| 테스트 | pytest, pytest-asyncio |
| 패키지 관리 | uv |
| 컨테이너 | Docker, Docker Compose |

## 에이전트 & 툴

### 에이전트
| 에이전트 | 역할 |
|---|---|
| Memory Retrieve | 과거 대화에서 관련 컨텍스트 검색 |
| RAG Retrieve | 업로드된 문서에서 관련 내용 검색 |
| Search Agent | DuckDuckGo 검색 + 쿼리 최적화 |
| Summarizer Agent | 검색 결과 map-reduce 요약 |
| Critic Agent | 사실 검증 + 재검색 트리거 |
| Formatter Agent | 마크다운 응답 생성 |
| Memory Save | 대화 결과 장기 메모리 저장 |

### 툴
| 툴 | 역할 |
|---|---|
| get_today_date | 현재 날짜 반환 (KST) |
| get_workspace_path | 작업 폴더 경로 반환 |
| create_file | .md / .txt 파일 생성 |
| read_file | 파일 읽기 |
| send_email | Gmail SMTP 이메일 전송 |

## 시작하기

### 사전 요구사항

- Docker, Docker Compose
- Ollama
- Google Gemini API 키 (production 시)
- Gmail 앱 비밀번호 (이메일 툴 사용 시)

### Ollama 모델 준비

```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### 환경 변수 설정

```bash
cp .env.example .env
# .env 파일에 API 키 입력
```

### 개발 환경 실행

```bash
# 인프라 실행 (Redis, PostgreSQL + pgvector)
docker compose -f docker-compose.dev.yml up -d

# 서버 실행
uv run uvicorn app.main:app --reload --port 8000
```

### 전체 스택 실행

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

### 문서 업로드 (RAG)
```
POST /api/v1/documents
Content-Type: multipart/form-data
```
```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@document.pdf"
```

### 업로드된 문서 목록
```
GET /api/v1/documents
```

### 작업 폴더 설정
```
GET  /api/v1/settings/workspace
PATCH /api/v1/settings/workspace
```
```json
{ "path": "/mnt/f/15_Project/workspace" }
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

### 장기 메모리 + RAG 분리 이유
장기 메모리는 과거 대화 컨텍스트(사용자 맥락)를, RAG는 외부 문서 컨텍스트(도메인 지식)를 담당합니다.
두 컨텍스트를 분리해서 각각 독립적으로 검색하고 Summarizer Agent에 주입합니다.

### pgvector를 선택한 이유
별도 벡터 DB(Pinecone, Weaviate) 없이 기존 PostgreSQL에 확장으로 추가할 수 있어
인프라 복잡도를 낮추면서 벡터 검색 기능을 구현할 수 있습니다.

### Gemini / Ollama 전략
`APP_ENV` 환경변수로 LLM을 전환합니다.
- `development` → Ollama (로컬, 무료)
- `production` → Gemini (빠른 응답)

## 테스트

```bash
uv run pytest -v
```

## 프로젝트 구조

```
app/
├── main.py
├── api/
│   ├── routes.py
│   └── settings.py
├── agents/
│   ├── orchestrator.py
│   ├── memory_agent.py
│   ├── rag_agent.py
│   ├── search_agent.py
│   ├── summarizer_agent.py
│   ├── critic_agent.py
│   ├── custom_agent.py
│   ├── plan_parser_agent.py
│   ├── planner_agent.py
│   ├── writer_agent.py
│   └── formatter_agent.py
├── core/
│   ├── config.py
│   ├── state.py
│   ├── prompts.py
│   ├── tools.py
│   ├── utils.py
│   └── logger.py
└── db/
    ├── postgres.py
    ├── vector_store.py
    └── redis_client.py