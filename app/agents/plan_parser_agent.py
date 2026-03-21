"""
plan_parser_agent.py

기획서 파일(PDF, DOCX, MD, TXT)을 받아 LLM으로 섹션 구조를 분석하고,
섹션 유형을 메타데이터로 태깅한 뒤 RAG 벡터 스토어에 저장한다.

일반 문서 업로드(/documents)와의 차이:
- 단순 청킹이 아닌 LLM 기반 섹션 분류
- section_type, section_heading, plan_title 메타데이터 태깅
- 쿼리 시 "요구사항만", "일정만" 같은 필터링 검색 가능
"""

import json
import re
import io

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import get_llm
from app.core.prompts import PLAN_PARSE_PROMPT
from app.db.vector_store import get_rag_store
from app.core.logger import get_logger

logger = get_logger("plan_parser_agent")

# 섹션 하나당 최대 청크 길이 (초과 시 추가 분할)
_MAX_SECTION_LENGTH = 800


async def parse_and_ingest_plan(
    filename: str,
    file_bytes: bytes,
    content_type: str,
    model_name: str | None = None,
) -> dict:
    """
    기획서를 파싱해서 RAG에 저장한다.

    Args:
        filename:     업로드된 파일명
        file_bytes:   파일 바이트
        content_type: MIME 타입
        model_name:   파싱에 사용할 LLM (None이면 기본 LLM)

    Returns:
        {"title": str, "sections": int, "chunks": int}
    """
    # 1. 텍스트 추출
    text = _extract_text(filename, file_bytes)
    if not text.strip():
        raise ValueError("파일에서 텍스트를 추출할 수 없습니다.")

    logger.info(f"텍스트 추출 완료: {filename} ({len(text)}자)")

    # 2. LLM으로 섹션 분류
    parsed = await _parse_sections(text, model_name)
    title = parsed.get("title", "")
    sections = parsed.get("sections", [])

    if not sections:
        logger.warning("섹션 파싱 실패 — 전체 텍스트를 단일 청크로 저장")
        sections = [{"type": "other", "heading": filename, "content": text}]

    logger.info(f"섹션 파싱 완료: {len(sections)}개 섹션 / 제목: {title or '(없음)'}")

    # 3. 섹션별 Document 생성 + RAG 저장
    docs = _build_documents(filename, title, sections)
    store = get_rag_store()
    store.add_documents(docs)

    logger.info(f"RAG 저장 완료: {filename} → {len(docs)}개 청크")
    return {"title": title, "sections": len(sections), "chunks": len(docs)}


# ── 텍스트 추출 ───────────────────────────────────────────────

def _extract_text(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return _extract_pdf(file_bytes)

    if lower.endswith(".docx"):
        return _extract_docx(file_bytes)

    if lower.endswith((".md", ".txt")):
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError(f"지원하지 않는 파일 형식입니다: {filename}")


def _extract_pdf(file_bytes: bytes) -> str:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx가 설치되지 않았습니다. `pip install python-docx`를 실행하세요.")

    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


# ── LLM 섹션 파싱 ─────────────────────────────────────────────

async def _parse_sections(text: str, model_name: str | None) -> dict:
    """LLM으로 기획서 섹션을 분류한다. 실패 시 빈 sections 반환."""
    llm = get_llm(model_name)

    # 토큰 절약: 텍스트가 너무 길면 앞 4000자만 사용
    trimmed = text[:4000] if len(text) > 4000 else text

    prompt = PLAN_PARSE_PROMPT.format(content=trimmed)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # ```json ... ``` 펜스 제거
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"섹션 파싱 JSON 오류: {e} / raw: {raw[:200]}")
        return {}
    except Exception as e:
        logger.error(f"섹션 파싱 실패: {e}")
        return {}


# ── Document 빌드 ─────────────────────────────────────────────

def _build_documents(filename: str, title: str, sections: list[dict]) -> list[Document]:
    """
    섹션 리스트를 langchain Document로 변환한다.
    섹션 내용이 _MAX_SECTION_LENGTH를 초과하면 추가 분할한다.
    """
    docs = []

    for i, section in enumerate(sections):
        section_type    = section.get("type", "other")
        section_heading = section.get("heading", f"섹션 {i + 1}")
        content         = section.get("content", "").strip()

        if not content:
            continue

        base_metadata = {
            "filename":        filename,
            "plan_title":      title,
            "section_type":    section_type,
            "section_heading": section_heading,
            "source":          "plan",          # 일반 문서와 구분용 태그
        }

        # 섹션이 짧으면 그대로 1개 Document
        if len(content) <= _MAX_SECTION_LENGTH:
            docs.append(Document(page_content=content, metadata=base_metadata))
            continue

        # 길면 문장 단위로 추가 분할
        chunks = _split_by_length(content, _MAX_SECTION_LENGTH)
        for j, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={**base_metadata, "chunk_index": j},
            ))

    return docs


def _split_by_length(text: str, max_len: int) -> list[str]:
    """텍스트를 max_len 이하 청크로 분할. 문장 경계 우선."""
    sentences = re.split(r"(?<=[.。!?])\s+", text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_len:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current)
            # 단일 문장이 max_len 초과 시 강제 분할
            while len(sentence) > max_len:
                chunks.append(sentence[:max_len])
                sentence = sentence[max_len:]
            current = sentence

    if current:
        chunks.append(current)

    return chunks