"""faster-whisper 로컬 어댑터 — 완전 오프라인(민감 회의용).

GPU(CUDA) 자동 감지: RTX 3050급(VRAM 4GB+)이면 large-v3-turbo가 실시간보다 빠르게 돌고,
GTX 1050(2GB)은 small로 자동 축소. CUDA 초기화 실패(cuDNN 미설치 등) 시 CPU int8 폴백.
CPU에서는 오디오 길이의 1~3배 시간이 걸릴 수 있으므로 야간 배치에 적합.

환경변수 강제 지정: MEETING_SCRIBE_WHISPER_MODEL, MEETING_SCRIBE_WHISPER_DEVICE(cuda|cpu)
"""
import os
from pathlib import Path

from ..hw import local_whisper_plan
from .base import AsrAdapter, AsrSegment, AsrWord


class LocalWhisperAdapter(AsrAdapter):
    name = "local"
    word_timestamps = True

    def __init__(self):
        plan = local_whisper_plan()
        self.model_size = os.environ.get("MEETING_SCRIBE_WHISPER_MODEL", plan["model"])
        self.device = os.environ.get("MEETING_SCRIBE_WHISPER_DEVICE", plan["device"])
        self.compute_type = plan["compute_type"] if self.device == "cuda" else "int8"

    def _load(self):
        from faster_whisper import WhisperModel
        try:
            return WhisperModel(self.model_size, device=self.device,
                                compute_type=self.compute_type)
        except Exception:
            if self.device == "cuda":  # cuDNN/cuBLAS 미설치 등 → CPU 폴백
                return WhisperModel(self.model_size, device="cpu", compute_type="int8")
            raise

    def transcribe(self, wav_path: Path, language: str,
                   hotwords: list[str], context_hint: str = "") -> list[AsrSegment]:
        model = self._load()
        segments_iter, _info = model.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,  # 프롬프트 주입 시 환각 방지 표준 완화책
            hotwords=", ".join(hotwords) if hotwords else None,
        )
        out: list[AsrSegment] = []
        for s in segments_iter:
            words = [AsrWord(w.word.strip(), w.start, w.end) for w in (s.words or [])]
            out.append(AsrSegment(text=s.text.strip(), start=s.start, end=s.end,
                                  avg_logprob=s.avg_logprob, words=words))
        return out
