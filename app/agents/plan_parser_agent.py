"""
plan_parser_agent.py
"""

import json
import re
import io

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.core.utils import get_llm
from app.core.prompts import CLASSIFY_PROMPT, PARSE_PROMPTS
from app.db.vector_store import get_rag_store
from app.core.logger import get_logger

logger = get_logger("plan_parser_agent")

_MAX_SECTION_LENGTH = 800
_HEADER_LENGTH = 500  # 문서 타입 분류에 사용할 앞부분 길이


async def parse_and_ingest_plan(
    filename: str,
    file_bytes: bytes,
    content_type: str,
    model_name: str | None = None,
) -> dict:
    # 1. 텍스트 추출
    text = _extract_text(filename, file_bytes)
    if not text.strip():
        raise ValueError("파일에서 텍스트를 추출할 수 없습니다.")

    logger.info(f"텍스트 추출 완료: {filename} ({len(text)}자)")

    # 2. 문서 타입 분류 (앞 500자만 사용 — 비용 절약)
    doc_type = await _classify_document(text[:_HEADER_LENGTH], model_name)
    logger.info(f"문서 타입 분류: {doc_type}")

    # 3. 타입별 프롬프트로 섹션 파싱 (최대 4000자)
    parsed = await _parse_sections(text[:4000], doc_type, model_name)
    title = parsed.get("title", "")
    sections = parsed.get("sections", [])

    if not sections:
        logger.warning("섹션 파싱 실패 — 전체 텍스트를 단일 청크로 저장")
        sections = [{"type": "other", "heading": filename, "content": text}]

    logger.info(f"섹션 파싱 완료: {len(sections)}개 / 제목: {title or '(없음)'}")

    # 4. Document 생성 + RAG 저장
    docs = _build_documents(filename, title, doc_type, sections)
    store = get_rag_store()
    store.add_documents(docs)

    logger.info(f"RAG 저장 완료: {filename} → {len(docs)}개 청크")
    return {
        "title": title,
        "doc_type": doc_type,
        "sections": len(sections),
        "chunks": len(docs),
    }


# ── 문서 타입 분류 ────────────────────────────────────────────

async def _classify_document(header: str, model_name: str | None) -> str:
    """문서 앞부분만 읽어 타입을 분류한다. 실패 시 GENERAL 반환."""
    llm = get_llm(model_name)
    prompt = CLASSIFY_PROMPT.format(header=header)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        doc_type = response.content.strip().upper()

        # 유효한 타입인지 확인, 아니면 GENERAL로 폴백
        valid_types = {"PLAN", "REPORT", "ARTICLE", "LEGAL", "RESUME", "GENERAL"}
        if doc_type not in valid_types:
            logger.warning(f"알 수 없는 문서 타입: {doc_type} → GENERAL로 폴백")
            return "GENERAL"

        return doc_type

    except Exception as e:
        logger.error(f"문서 타입 분류 실패: {e} → GENERAL로 폴백")
        return "GENERAL"


# ── LLM 섹션 파싱 ─────────────────────────────────────────────

async def _parse_sections(text: str, doc_type: str, model_name: str | None) -> dict:
    """타입별 프롬프트로 섹션을 파싱한다. 실패 시 빈 dict 반환."""
    llm = get_llm(model_name)

    # 타입별 프롬프트 선택 — 없으면 GENERAL로 폴백
    prompt_template = PARSE_PROMPTS.get(doc_type, PARSE_PROMPTS["GENERAL"])
    prompt = prompt_template.format(content=text)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # ```json ... ``` 펜스 제거
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        # title 빈 문자열 폴백
        if not parsed.get("title"):
            parsed["title"] = text.strip().splitlines()[0].strip()

        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"섹션 파싱 JSON 오류: {e} / raw: {raw[:200]}")
        return {}
    except Exception as e:
        logger.error(f"섹션 파싱 실패: {e}")
        return {}


# ── Document 빌드 ─────────────────────────────────────────────

def _build_documents(
    filename: str,
    title: str,
    doc_type: str,
    sections: list[dict],
) -> list[Document]:
    docs = []

    for i, section in enumerate(sections):
        section_type    = section.get("type", "other")
        section_heading = section.get("heading", f"섹션 {i + 1}")
        content         = section.get("content", "")

        # content가 dict나 list로 올 경우 JSON 문자열로 변환
        if isinstance(content, dict) or isinstance(content, list):
            logger.warning(f"섹션 {i+1} content가 {type(content).__name__} 타입 — 문자열 변환")
            import json
            content = json.dumps(content, ensure_ascii=False)

        content = str(content).strip()

        if not content:
            continue

        base_metadata = {
            "filename":        filename,
            "plan_title":      title,
            "doc_type":        doc_type,
            "section_type":    section_type,
            "section_heading": section_heading,
            "source":          "plan",
        }

        if len(content) <= _MAX_SECTION_LENGTH:
            docs.append(Document(page_content=content, metadata=base_metadata))
            continue

        chunks = _split_by_length(content, _MAX_SECTION_LENGTH)
        for j, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={**base_metadata, "chunk_index": j},
            ))

    return docs

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
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(file_bytes: bytes) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx가 필요합니다. `pip install python-docx`")

    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _split_by_length(text: str, max_len: int) -> list[str]:
    sentences = re.split(r"(?<=[.。!?])\s+", text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_len:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current)
            while len(sentence) > max_len:
                chunks.append(sentence[:max_len])
                sentence = sentence[max_len:]
            current = sentence

    if current:
        chunks.append(current)

    return chunks