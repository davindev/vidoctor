"""테스트 공통 헬퍼."""

from vidoctor.graph.state import Word


def _w(text: str, start: float, end: float, score: float | None = 0.9) -> Word:
    """단위 테스트용 Word 인스턴스 팩토리."""
    return Word(text=text, start=start, end=end, score=score)
