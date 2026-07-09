"""Groq Whisper large-v3-turbo 어댑터. 무료 티어 + 단어 타임스탬프. 25MB 제한 → 10분 청크 분할."""
import os
import time
from pathlib import Path

import httpx

from ..audio import split_wav
from .base import AsrAdapter, AsrSegment, AsrWord, offset_segments

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
CHUNK_SECONDS = 600  # 16kHz mono PCM16 10분 ≈ 19MB < 25MB
MAX_RETRIES = 3  # 429/5xx 재시도 횟수 (무료 티어 오디오 한도 7200초/시간 대응)
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_AFTER_CAP = 3600.0  # Retry-After가 비정상적으로 커도 이 이상 대기하지 않음


class GroqAdapter(AsrAdapter):
    name = "groq"
    word_timestamps = True

    def __init__(self, model: str = "whisper-large-v3-turbo"):
        self.model = os.environ.get("MEETING_SCRIBE_GROQ_MODEL", model)
        self.api_key = os.environ["GROQ_API_KEY"]

    def transcribe(self, wav_path: Path, language: str,
                   hotwords: list[str], context_hint: str = "") -> list[AsrSegment]:
        prompt = ", ".join(hotwords)[:800]  # Whisper prompt ~224토큰 제한 대응
        chunks = split_wav(wav_path, CHUNK_SECONDS, wav_path.parent / "chunks")
        all_segments: list[AsrSegment] = []
        with httpx.Client(timeout=300) as client:
            for chunk_path, offset in chunks:
                resp = _post_with_retry(
                    client, chunk_path, self.api_key,
                    data={
                        "model": self.model,
                        "language": language,
                        "response_format": "verbose_json",
                        # 단어(화자 배정용) + 세그먼트(원문·신뢰도용) 둘 다 요청
                        "timestamp_granularities[]": ["word", "segment"],
                        **({"prompt": prompt} if prompt else {}),
                    },
                )
                segments = _parse_verbose_json(resp.json())
                all_segments.extend(offset_segments(segments, offset))
        return all_segments


def _post_with_retry(client: httpx.Client, chunk_path: Path, api_key: str,
                     data: dict) -> httpx.Response:
    """429/5xx는 재시도 (무료 티어 한도에 긴 회의 전사가 통째로 죽지 않게).

    대기 시간은 Retry-After 헤더 우선, 없으면 지수 백오프. 파일 스트림은 소진되므로
    시도마다 다시 연다."""
    for attempt in range(MAX_RETRIES + 1):
        with open(chunk_path, "rb") as f:
            resp = client.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data=data,
                files={"file": (chunk_path.name, f, "audio/wav")},
            )
        if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
            time.sleep(_retry_wait(resp, attempt))
            continue
        resp.raise_for_status()
        return resp
    raise AssertionError("unreachable")  # 루프는 항상 return 또는 raise


def _retry_wait(resp: httpx.Response, attempt: int) -> float:
    try:
        return min(max(float(resp.headers.get("retry-after", "")), 1.0), RETRY_AFTER_CAP)
    except ValueError:
        return 2.0 * (2 ** attempt)  # 2, 4, 8초


def _parse_verbose_json(data: dict) -> list[AsrSegment]:
    words = [AsrWord(w["word"].strip(), float(w["start"]), float(w["end"]))
             for w in data.get("words", [])]
    segments: list[AsrSegment] = []
    for s in data.get("segments", []):
        seg = AsrSegment(
            text=s["text"].strip(), start=float(s["start"]), end=float(s["end"]),
            avg_logprob=s.get("avg_logprob"),
        )
        seg.words = [w for w in words if s["start"] - 0.01 <= w.start < s["end"] + 0.01]
        segments.append(seg)
    if not segments and words:  # segments 없이 words만 온 경우
        segments = [AsrSegment(
            text=" ".join(w.text for w in words), start=words[0].start, end=words[-1].end,
            words=words)]
    if not segments and data.get("text"):
        segments = [AsrSegment(text=data["text"].strip(), start=0.0, end=0.0)]
    return segments
