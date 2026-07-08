"""Groq Whisper large-v3-turbo 어댑터. 무료 티어 + 단어 타임스탬프. 25MB 제한 → 10분 청크 분할."""
import os
from pathlib import Path

import httpx

from ..audio import split_wav
from .base import AsrAdapter, AsrSegment, AsrWord, offset_segments

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
CHUNK_SECONDS = 600  # 16kHz mono PCM16 10분 ≈ 19MB < 25MB


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
                with open(chunk_path, "rb") as f:
                    resp = client.post(
                        API_URL,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data={
                            "model": self.model,
                            "language": language,
                            "response_format": "verbose_json",
                            "timestamp_granularities[]": "word",
                            **({"prompt": prompt} if prompt else {}),
                        },
                        files={"file": (chunk_path.name, f, "audio/wav")},
                    )
                resp.raise_for_status()
                data = resp.json()
                segments = _parse_verbose_json(data)
                all_segments.extend(offset_segments(segments, offset))
        return all_segments


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
