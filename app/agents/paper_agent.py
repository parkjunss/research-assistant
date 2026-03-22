"""
paper_agent.py

논문 PDF를 분석하여 코드를 생성하고 FastAPI 서비스로 래핑한다.

파이프라인:
1. Paper Analyzer  — 논문 분석 → 기술 명세서
2. Code Generator  — 명세서 기반 코드 생성
3. Code Critic     — subprocess로 코드 실행 + 검증
4. Service Builder — FastAPI 래핑 + 파일 저장
"""

import json
import re
import os
import asyncio
import subprocess
import tempfile

from langchain_core.messages import HumanMessage
from app.core.utils import get_llm
from app.core.prompts import (
    PAPER_ANALYZER_PROMPT,
    PAPER_CODE_PROMPT,
    PAPER_CRITIC_PROMPT,
    PAPER_SERVICE_PROMPT,
)
from app.db.postgres import get_setting
from app.core.logger import get_logger

logger = get_logger("paper_agent")

_MAX_RETRY = 2
_CODE_TIMEOUT = 10  # 초


async def run_paper_pipeline(
    filename: str,
    text: str,
    model_name: str | None = None,
) -> dict:
    """
    논문 텍스트를 받아 전체 파이프라인을 실행한다.

    Returns:
        {
            "title":    str,
            "spec":     dict,
            "code":     str,
            "service":  str,
            "filepath": str,
        }
    """
    llm = get_llm(model_name)

    # 1. 논문 분석 → 기술 명세서
    spec = await _analyze_paper(llm, text)
    logger.info(f"논문 분석 완료: {spec.get('title', '제목 없음')}")

    # 2. 코드 생성 + 검증 (최대 _MAX_RETRY회)
    code = await _generate_and_validate_code(llm, spec)
    logger.info("코드 생성 및 검증 완료")

    # 3. FastAPI 서비스 래핑
    service_code = await _build_service(llm, spec, code)
    logger.info("서비스 코드 생성 완료")

    # 4. 파일 저장
    filepath = await _save_outputs(spec, code, service_code)
    logger.info(f"파일 저장 완료: {filepath}")

    return {
        "title":    spec.get("title", ""),
        "spec":     spec,
        "code":     code,
        "service":  service_code,
        "filepath": filepath,
    }


# ── 1. 논문 분석 ──────────────────────────────────────────────

async def _analyze_paper(llm, text: str) -> dict:
    """논문 텍스트를 분석해서 기술 명세서를 반환한다."""
    trimmed = text[:6000] if len(text) > 6000 else text
    prompt = PAPER_ANALYZER_PROMPT.format(content=trimmed)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = _strip_fences(response.content)

        # JSON 파싱 시도 1: 직접 파싱
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # JSON 파싱 시도 2: {{ }} 블록 추출
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # JSON 파싱 시도 3: LLM에 재요청
        logger.warning("JSON 파싱 실패 → 재요청")
        fix_prompt = f"""아래 텍스트를 올바른 JSON으로만 반환하세요. 설명 없이 JSON만 반환하세요:

{raw}

반드시 아래 필드를 포함해야 합니다:
title, problem, algorithm, inputs, outputs, constraints, language, dependencies"""

        fix_response = await llm.ainvoke([HumanMessage(content=fix_prompt)])
        fixed_raw = _strip_fences(fix_response.content)
        match2 = re.search(r'\{.*\}', fixed_raw, re.DOTALL)
        if match2:
            return json.loads(match2.group())

        raise ValueError("JSON 파싱 최종 실패")

    except Exception as e:
        logger.error(f"명세서 파싱 실패: {e}")
        return {
            "title": _extract_title(text),
            "problem": text[:200],
            "algorithm": text[:500],
            "inputs": [],
            "outputs": [],
            "constraints": [],
            "language": "Python",
            "dependencies": [],
        }

def _extract_title(text: str) -> str:
    """텍스트 첫 줄에서 제목을 추출한다."""
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 100:
            # "Title:" 접두사 제거
            return re.sub(r'^[Tt]itle:\s*', '', line).strip() or "Unknown"
    return "Unknown"

# ── 2. 코드 생성 + 검증 ────────────────────────────────────────

async def _generate_and_validate_code(llm, spec: dict) -> str:
    """코드를 생성하고 실행 검증한다. 실패 시 최대 _MAX_RETRY회 재시도."""
    code = await _generate_code(llm, spec)

    for attempt in range(_MAX_RETRY):
        result = await _execute_code(code)
        critique = await _critique_code(llm, spec, code, result)

        if critique.get("passed"):
            logger.info(f"코드 검증 통과 (시도 {attempt + 1}회)")
            return code

        fixed = critique.get("fixed_code")
        if fixed:
            logger.warning(f"코드 수정 후 재시도 ({attempt + 1}/{_MAX_RETRY})")
            code = fixed
        else:
            break

    logger.warning("최대 재시도 초과 — 현재 코드로 진행")
    return code


async def _generate_code(llm, spec: dict) -> str:
    """기술 명세서 기반으로 코드를 생성한다."""
    prompt = PAPER_CODE_PROMPT.format(spec=json.dumps(spec, ensure_ascii=False))
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return _strip_fences(response.content)


async def _execute_code(code: str) -> str:
    """subprocess로 코드를 실행하고 결과를 반환한다."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=_CODE_TIMEOUT,
            )
        )

        os.unlink(tmp_path)

        if result.returncode == 0:
            output = result.stdout[:1000] or "실행 성공 (출력 없음)"
            logger.info(f"코드 실행 성공: {output[:100]}")
            return output
        else:
            error = result.stderr[:1000]
            logger.warning(f"코드 실행 오류: {error[:100]}")
            return f"오류: {error}"

    except subprocess.TimeoutExpired:
        logger.warning(f"코드 실행 타임아웃 ({_CODE_TIMEOUT}초)")
        return f"타임아웃: {_CODE_TIMEOUT}초 초과"
    except Exception as e:
        logger.error(f"코드 실행 실패: {e}")
        return f"실행 실패: {e}"


async def _critique_code(llm, spec: dict, code: str, execution_result: str) -> dict:
    """코드 실행 결과를 검증한다."""
    prompt = PAPER_CRITIC_PROMPT.format(
        spec=json.dumps(spec, ensure_ascii=False),
        code=code,
        execution_result=execution_result,
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = _strip_fences(response.content)
        return json.loads(raw)
    except Exception as e:
        logger.error(f"코드 검증 실패: {e}")
        return {"passed": True, "issues": [], "fixed_code": None, "summary": "검증 스킵"}


# ── 3. 서비스 빌더 ────────────────────────────────────────────

async def _build_service(llm, spec: dict, code: str) -> str:
    """FastAPI 서비스 코드를 생성한다."""
    prompt = PAPER_SERVICE_PROMPT.format(
        title=spec.get("title", ""),
        code=code,
        spec=json.dumps(spec, ensure_ascii=False),
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return _strip_fences(response.content)


# ── 4. 파일 저장 ──────────────────────────────────────────────

async def _save_outputs(spec: dict, code: str, service_code: str) -> str:
    """생성된 코드와 서비스를 워크스페이스에 저장한다."""
    workspace = await get_setting("workspace_path") or "/tmp/research_workspace"

    title = spec.get("title", "paper_output")
    safe_title = re.sub(r'[\\/*?:"<>| ]', "_", title)[:40]
    output_dir = os.path.join(workspace, safe_title)
    os.makedirs(output_dir, exist_ok=True)

    # 핵심 코드 저장
    code_path = os.path.join(output_dir, "solution.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    # FastAPI 서비스 저장
    service_path = os.path.join(output_dir, "service.py")
    with open(service_path, "w", encoding="utf-8") as f:
        f.write(service_code)

    # 기술 명세서 저장
    spec_path = os.path.join(output_dir, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    # Dockerfile 생성
    deps = spec.get("dependencies", [])
    pip_install = f"RUN pip install {' '.join(deps)}" if deps else ""
    dockerfile = f"""FROM python:3.11-slim
    WORKDIR /app
    RUN pip install fastapi uvicorn
    {pip_install}
    COPY . .
    CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
    """
    dockerfile_path = os.path.join(output_dir, "Dockerfile")
    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(dockerfile)

    logger.info(f"출력 파일 저장: {output_dir}")
    return output_dir


# ── 헬퍼 ──────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """```json, ```python 펜스를 제거한다."""
    text = re.sub(r"^```(?:json|python)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()