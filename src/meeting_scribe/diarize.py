"""로컬 CPU 화자분리(sherpa-onnx) + 화자 임베딩 추출/매칭.

- 화자분리: pyannote segmentation-3.0 ONNX + CAM++ 임베딩 클러스터링
- 화자 학습: 클러스터별 임베딩을 SQLite 보이스프린트와 코사인 매칭 → '제안'만 하고
  확정(이름 부여)은 사용자의 confirm_speaker 호출로만 이뤄진다 (동의 기반 등록).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CLUSTER_THRESHOLD, SPEAKER_SUGGEST_THRESHOLD
from .models import ensure_models


@dataclass
class DiarSegment:
    start: float
    end: float
    cluster: str  # "화자1", "화자2", ...

    @property
    def duration(self) -> float:
        return self.end - self.start


def _cluster_label(idx: int) -> str:
    return f"화자{idx + 1}"


def diarize(samples: np.ndarray, sample_rate: int,
            num_speakers: int | None = None,
            num_threads: int = 2) -> list[DiarSegment]:
    import sherpa_onnx

    seg_model, emb_model = ensure_models()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model)),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model), num_threads=num_threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers else -1,
            threshold=CLUSTER_THRESHOLD,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    if sd.sample_rate != sample_rate:
        raise ValueError(f"화자분리 모델은 {sd.sample_rate}Hz 입력 필요 (입력: {sample_rate}Hz)")
    result = sd.process(samples).sort_by_start_time()
    return [DiarSegment(r.start, r.end, _cluster_label(r.speaker)) for r in result]


def extract_cluster_embeddings(samples: np.ndarray, sample_rate: int,
                               segments: list[DiarSegment],
                               min_seg: float = 1.0, max_total: float = 30.0,
                               num_threads: int = 2) -> dict[str, np.ndarray]:
    """클러스터별 대표 임베딩. 긴 구간 우선으로 최대 max_total초 사용 (원거리 마이크 잡음 완화)."""
    import sherpa_onnx

    _, emb_model = ensure_models()
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=num_threads))

    by_cluster: dict[str, list[DiarSegment]] = {}
    for s in segments:
        by_cluster.setdefault(s.cluster, []).append(s)

    out: dict[str, np.ndarray] = {}
    for cluster, segs in by_cluster.items():
        segs = sorted(segs, key=lambda s: s.duration, reverse=True)
        picked, total = [], 0.0
        for s in segs:
            if s.duration < min_seg and picked:
                continue
            picked.append(s)
            total += s.duration
            if total >= max_total:
                break
        if not picked:
            continue
        audio = np.concatenate([
            samples[int(s.start * sample_rate):int(s.end * sample_rate)] for s in picked
        ])
        stream = ex.create_stream()
        stream.accept_waveform(sample_rate, audio)
        stream.input_finished()
        emb = np.asarray(ex.compute(stream), dtype=np.float32)
        out[cluster] = emb
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def match_speakers(cluster_embs: dict[str, np.ndarray],
                   enrolled: list[tuple[int, str, np.ndarray]],
                   threshold: float = SPEAKER_SUGGEST_THRESHOLD,
                   ) -> dict[str, tuple[int | None, str | None, float]]:
    """클러스터 → (speaker_id, 이름, 유사도). 등록 화자별 최대 유사도로 매칭, 임계값 미만이면 None."""
    result: dict[str, tuple[int | None, str | None, float]] = {}
    for cluster, emb in cluster_embs.items():
        best: tuple[int | None, str | None, float] = (None, None, 0.0)
        per_speaker: dict[int, tuple[str, float]] = {}
        for sid, name, ref in enrolled:
            sim = cosine(emb, ref)
            if sid not in per_speaker or sim > per_speaker[sid][1]:
                per_speaker[sid] = (name, sim)
        for sid, (name, sim) in per_speaker.items():
            if sim > best[2]:
                best = (sid, name, sim)
        if best[2] < threshold:
            best = (None, None, best[2])
        result[cluster] = best
    return result
