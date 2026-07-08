"""ASR 어댑터 레지스트리. 환경변수 키 존재 여부로 사용 가능 어댑터를 판별한다.

우선순위 (MEETING_SCRIBE_ASR 로 강제 지정 가능):
  1. rtzr   — ReturnZero (한국어 특화, 키워드 부스팅, 단어 타임스탬프)
  2. groq   — Groq Whisper large-v3-turbo (무료 티어, 단어 타임스탬프)
  3. gemini — Gemini Flash (무료 티어, 단어 타임스탬프 없음 → 화자 매칭 정밀도 낮음)
  4. local  — faster-whisper (완전 로컬, 민감 회의용, CPU에서 느림)
"""
import os

from .base import AsrAdapter, AsrSegment, AsrWord

_PRIORITY = ["rtzr", "groq", "gemini", "local"]


def _configured(name: str) -> bool:
    if name == "rtzr":
        return bool(os.environ.get("RTZR_CLIENT_ID") and os.environ.get("RTZR_CLIENT_SECRET"))
    if name == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if name == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if name == "local":
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False
    return False


def available_adapters() -> list[str]:
    return [n for n in _PRIORITY if _configured(n)]


def get_adapter(name: str | None = None) -> AsrAdapter:
    name = name or os.environ.get("MEETING_SCRIBE_ASR")
    if not name:
        avail = available_adapters()
        if not avail:
            raise RuntimeError(
                "사용 가능한 ASR 엔진이 없습니다. GROQ_API_KEY / GEMINI_API_KEY / "
                "RTZR_CLIENT_ID+SECRET 환경변수를 설정하거나 "
                "'pip install meeting-scribe[local]'로 로컬 엔진을 설치하세요.")
        name = avail[0]
    if name == "rtzr":
        from .rtzr import RtzrAdapter
        return RtzrAdapter()
    if name == "groq":
        from .groq import GroqAdapter
        return GroqAdapter()
    if name == "gemini":
        from .gemini import GeminiAdapter
        return GeminiAdapter()
    if name == "local":
        from .local_whisper import LocalWhisperAdapter
        return LocalWhisperAdapter()
    raise ValueError(f"알 수 없는 ASR 어댑터: {name}")


__all__ = ["AsrAdapter", "AsrSegment", "AsrWord", "get_adapter", "available_adapters"]
