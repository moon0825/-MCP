"""Gemini Flash 오디오 전사 어댑터. 단어 타임스탬프가 없어 화자 배정 정밀도가 낮다 —
문장 단위 타임스탬프를 프롬프트로 요구하고 세그먼트 단위로 화자를 배정한다.
사용자가 이미 쓰던 Gemini 워크플로우의 자동화 버전. 무료 티어 가능."""
import json
import os
import re
from pathlib import Path

from .base import AsrAdapter, AsrSegment

PROMPT = """다음 오디오를 한국어로 전사하세요.

출력은 JSON 배열만: [{{"start": 초(숫자), "end": 초(숫자), "text": "발화 내용"}}, ...]
- 문장/발화 단위로 나누고 시작·끝 시간을 초 단위 숫자로 추정해 기록
- 다음 용어/이름이 등장할 수 있으니 정확히 표기: {hotwords}
{context_hint}
- JSON 외 다른 텍스트 금지"""


class GeminiAdapter(AsrAdapter):
    name = "gemini"
    word_timestamps = False

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = os.environ.get("MEETING_SCRIBE_GEMINI_MODEL", model)

    def transcribe(self, wav_path: Path, language: str,
                   hotwords: list[str], context_hint: str = "") -> list[AsrSegment]:
        from google import genai

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
        uploaded = client.files.upload(file=str(wav_path))
        prompt = PROMPT.format(
            hotwords=", ".join(hotwords) or "(없음)",
            context_hint=f"- 회의 맥락: {context_hint}" if context_hint else "",
        )
        resp = client.models.generate_content(model=self.model, contents=[uploaded, prompt])
        text = resp.text or ""
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            raise RuntimeError(f"Gemini 응답에서 JSON을 찾지 못함: {text[:300]}")
        items = json.loads(m.group(0))
        return [
            AsrSegment(text=str(it.get("text", "")).strip(),
                       start=float(it.get("start", 0)), end=float(it.get("end", 0)))
            for it in items if str(it.get("text", "")).strip()
        ]
