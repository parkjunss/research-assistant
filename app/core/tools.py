from langchain_core.tools import tool
from datetime import datetime, timezone, timedelta
import os
import asyncio
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

KST = timezone(timedelta(hours=9))

@tool
def get_today_date() -> str:
    """오늘 날짜와 요일을 반환합니다. 사용자가 '오늘', '최근', '이번 주' 등의 표현을 사용할 때 호출하세요."""
    now = datetime.now(KST)
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    weekday = weekdays[now.weekday()]
    return f"오늘 날짜: {now.strftime('%Y년 %m월 %d일')} ({weekday}요일)"

@tool
def get_workspace_path() -> str:
    """현재 작업 폴더 경로를 반환합니다. 파일 생성/읽기 전에 호출하세요."""
    from app.db.postgres import AsyncSessionLocal, AppSettings
    from sqlalchemy import select

    async def _get():
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AppSettings).where(AppSettings.key == "workspace_path")
            )
            row = result.scalar_one_or_none()
            return row.value if row else "/tmp/research_workspace"

    return asyncio.get_event_loop().run_until_complete(_get())

@tool
def create_file(filename: str, content: str) -> str:
    """파일을 생성하고 내용을 저장합니다.
    filename: 파일명 (예: report.md, summary.txt)
    content: 저장할 내용
    """
    try:
        workspace = get_workspace_path.invoke({})
        os.makedirs(workspace, exist_ok=True)

        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith((".md", ".txt")):
            return "오류: .md 또는 .txt 파일만 생성 가능합니다."

        filepath = os.path.join(workspace, safe_filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return f"파일 생성 완료: {filepath}"
    except Exception as e:
        return f"파일 생성 실패: {e}"

@tool
def read_file(filename: str) -> str:
    """파일을 읽어서 내용을 반환합니다.
    filename: 파일명 (예: report.md, summary.txt)
    """
    try:
        workspace = get_workspace_path.invoke({})
        safe_filename = os.path.basename(filename)
        filepath = os.path.join(workspace, safe_filename)

        if not os.path.exists(filepath):
            return f"오류: 파일을 찾을 수 없습니다 ({filepath})"

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        return f"파일 내용:\n{content}"
    except Exception as e:
        return f"파일 읽기 실패: {e}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """이메일을 전송합니다.
    to: 수신자 이메일 주소
    subject: 이메일 제목
    body: 이메일 본문 (마크다운 지원)
    """
    from app.core.config import settings

    async def _send():
        try:
            message = MIMEMultipart("alternative")
            message["From"] = settings.smtp_user
            message["To"] = to
            message["Subject"] = subject

            message.attach(MIMEText(body, "plain", "utf-8"))

            await aiosmtplib.send(
                message,
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_user,
                password=settings.smtp_password,
                start_tls=True,
            )
            return f"이메일 전송 완료: {to}"
        except Exception as e:
            return f"이메일 전송 실패: {e}"

    return asyncio.get_event_loop().run_until_complete(_send())