from langchain_core.tools import tool
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

@tool
def get_today_date() -> str:
    """오늘 날짜와 요일을 반환합니다. 사용자가 '오늘', '최근', '이번 주' 등의 표현을 사용할 때 호출하세요."""
    now = datetime.now(KST)
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    weekday = weekdays[now.weekday()]
    return f"오늘 날짜: {now.strftime('%Y년 %m월 %d일')} ({weekday}요일)"