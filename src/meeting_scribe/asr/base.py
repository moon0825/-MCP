"""ASR 어댑터 공통 인터페이스."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AsrWord:
    text: str
    start: float
    end: float


@dataclass
class AsrSegment:
    text: str
    start: float
    end: float
    avg_logprob: float | None = None
    words: list[AsrWord] = field(default_factory=list)


class AsrAdapter(ABC):
    name: str = "base"
    word_timestamps: bool = True  # False면 병합 시 세그먼트 단위 화자 배정으로 폴백

    @abstractmethod
    def transcribe(self, wav_path: Path, language: str,
                   hotwords: list[str], context_hint: str = "") -> list[AsrSegment]:
        """16kHz mono wav를 전사. hotwords는 부스팅용 핵심 용어 목록."""


def offset_segments(segments: list[AsrSegment], offset: float) -> list[AsrSegment]:
    for s in segments:
        s.start += offset
        s.end += offset
        for w in s.words:
            w.start += offset
            w.end += offset
    return segments
