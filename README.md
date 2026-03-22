# AI Research Assistant

멀티 에이전트 아키텍처 기반의 AI 리서치 어시스턴트입니다.
사용자 질문에 대해 자동으로 검색 → 요약 → 검증 → 응답 파이프라인을 실행하며,
코딩, 문서 작성, 논문 분석, 작업 분해 등 다양한 작업을 지원합니다.

## 아키텍처

```
사용자 질문
  → Memory Retrieve   (장기 메모리 검색)
  → RAG Retrieve      (하이브리드 검색 + Re-ranking)
  → Reasoning Agent   (질문 유형 분류 → 동적 라우팅)
      ├─ code     → Code Agent        (코드 생성/리뷰/버그 분석)
      ├─ planning → Planner Agent     (작업 분해)
      ├─ writing  → Writer Agent      (문서 작성)
      └─ search   → Search Agent      (DuckDuckGo + 날짜 툴)
                    → Summarizer Agent (map-reduce 요약)
                    → Critic Agent     (사실 검증 + 재검색)
                    → Formatter Agent  (마크다운 응답)
  → Memory Save       (장기 메모리 저장)
```

## 기술 스택

| 역할 | 기술 |
|---|---|
| API 서버 | FastAPI, Uvicorn |
| 에이전트 오케스트레이션 | LangGraph |
| LLM | Google Gemini (production), Ollama (development) |
| 임베딩 | Ollama bge-m3 |
| Re-ranking | Ollama embed 코사인 유사도 |
| 검색 | DuckDuckGo Search + BM25 하이브리드 |
| 벡터 DB | pgvector (PostgreSQL 확장) |
| 데이터베이스 | PostgreSQL |
| 캐시 | Redis |
| 테스트 | pytest, pytest-asyncio |
| 패키지 관리 | uv |
| 컨테이너 | Docker, Docker Compose |

## 에이전트 & 툴

### 에이전트

| 에이전트 | 역할 |
|---|---|
| Reasoning Agent | LLM 기반 질문 유형 분류 + 동적 라우팅 |
| Memory Retrieve | 과거 대화 컨텍스트 검색 |
| RAG Retrieve | 하이브리드 검색 + Re-ranking |
| Search Agent | DuckDuckGo 검색 + 쿼리 최적화 |
| Summarizer Agent | 검색 결과 map-reduce 요약 |
| Critic Agent | 사실 검증 + 재검색 트리거 |
| Formatter Agent | 마크다운 응답 생성 |
| Code Agent | 코드 생성, 리뷰, 버그 분석 |
| Planner Agent | 기획서 기반 작업 분해 |
| Writer Agent | 컨텍스트 기반 문서 작성 |
| Memory Save | 대화 결과 장기 메모리 저장 |

### 툴

| 툴 | 역할 |
|---|---|
| get_today_date | 현재 날짜 반환 (KST) |
| get_workspace_path | 작업 폴더 경로 반환 |
| create_file | .md / .txt 파일 생성 |
| read_file | 파일 읽기 |
| send_email | Gmail SMTP 이메일 전송 |

## RAG 파이프라인

```
문서 업로드
  → 문서 타입 분류 (PLAN/REPORT/ARTICLE/LEGAL/RESUME/GENERAL)
  → 타입별 LLM 파싱 + 섹션 메타데이터 태깅
  → 타입별 청킹 전략 적용
  → pgvector 저장

질문 시 검색
  → 하이브리드 검색 (벡터 0.7 + BM25 0.3, RRF 통합)
  → Re-ranking (bge-m3 코사인 유사도)
  → 상위 5개 컨텍스트 → LLM 주입
```

## 논문 분석 파이프라인

```
POST /api/v1/paper
  → Paper Analyzer  (알고리즘/입출력/제약조건 추출 → 기술 명세서)
  → Code Generator  (명세서 기반 Python 코드 생성)
  → Code Critic     (subprocess 실행 + Self-Correction, 최대 2회)
  → Service Builder (FastAPI 래핑 + Dockerfile 생성)
  → 파일 저장       (solution.py, service.py, spec.json, Dockerfile)
```

## 시작하기

### 사전 요구사항

- Docker, Docker Compose
- Ollama
- Google Gemini API 키 (production 시)
- Gmail 앱 비밀번호 (이메일 툴 사용 시)

### Ollama 모델 준비

```bash
ollama pull qwen2.5:14b       # LLM
ollama pull bge-m3            # 임베딩 + Re-ranking
ollama pull nomic-embed-text  # 폴백용
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
{
  "query": "LangGraph란 무엇인가요?",
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
```
```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@document.pdf"
```

### 기획서 업로드 (구조화 파싱)
```
POST /api/v1/documents/plan
```

### 논문 분석 → 코드 생성
```
POST /api/v1/paper
```
```bash
curl -X POST http://localhost:8000/api/v1/paper \
  -F "file=@paper.pdf"
```

### 작업 분해 (Planner)
```
POST /api/v1/plan
```

### 문서 작성 (Writer)
```
POST /api/v1/write
```

### 작업 폴더 설정
```
GET  /api/v1/settings/workspace
PATCH /api/v1/settings/workspace
```

### 에이전트 관리
```
GET    /api/v1/agents
POST   /api/v1/agents
PATCH  /api/v1/agents/{name}
DELETE /api/v1/agents/{name}
```

### 서버 상태
```
GET /health
```

## 설계 의사결정

### LangGraph를 선택한 이유
Critic Agent의 검증 결과에 따라 Search Agent로 되돌아가는 조건부 분기가 필요했습니다.
단순 순차 실행이 아닌 루프 구조를 명시적으로 표현하기 위해 LangGraph의 `StateGraph`와
`add_conditional_edges`를 활용했습니다.

### Reasoning Agent 설계
LLM 기반 질문 유형 분류가 실패할 경우를 대비해 키워드 기반 폴백을 구현했습니다.
LLM 호출 1회 추가 비용이 있지만 맥락을 이해한 정확한 라우팅이 가능합니다.

### 하이브리드 검색 + Re-ranking
벡터 검색만으로는 한글 기술 용어, 고유명사, 함수명 검색이 부정확했습니다.
BM25 키워드 검색을 RRF로 혼합하고, bge-m3 임베딩 코사인 유사도로 재순위화하여
검색 정확도를 크게 향상시켰습니다.

### pgvector를 선택한 이유
별도 벡터 DB 없이 기존 PostgreSQL 확장으로 벡터 검색을 구현하여
인프라 복잡도를 낮추었습니다.

### subprocess 코드 실행
논문 분석 파이프라인에서 생성된 코드를 검증하기 위해 subprocess로 격리 실행합니다.
10초 타임아웃과 stderr 캡처로 안전하게 실행하고, 실패 시 LLM이 코드를 수정하는
Self-Correction 루프를 구현했습니다.

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
│   ├── reasoning_agent.py
│   ├── memory_agent.py
│   ├── rag_agent.py
│   ├── search_agent.py
│   ├── summarizer_agent.py
│   ├── critic_agent.py
│   ├── formatter_agent.py
│   ├── code_agent.py
│   ├── plan_parser_agent.py
│   ├── planner_agent.py
│   └── writer_agent.py
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