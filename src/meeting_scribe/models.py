"""화자분리/임베딩 ONNX 모델 자동 다운로드 (최초 1회, 총 ~35MB)."""
import tarfile
from pathlib import Path

import httpx

from .config import MODELS_DIR

# GitHub 릴리스가 1순위, 차단 환경(사내망 등) 대비 HuggingFace 미러 폴백
SEGMENTATION_URLS = [
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
]
SEGMENTATION_ONNX_URLS = [
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
]
SEGMENTATION_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"

# 3D-Speaker CAM++ (Apache-2.0, 중국어/영어 학습 — 동아시아 화자에 무난, 512차원)
_EMB_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
EMBEDDING_URLS = [
    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{_EMB_FILE}",
    f"https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/{_EMB_FILE}",
]
EMBEDDING_MODEL = MODELS_DIR / "3dspeaker_campplus_zh_en_advanced.onnx"


def _download(urls: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for url in urls:
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            tmp.rename(dest)
            return
        except Exception as e:  # 다음 미러 시도
            last_err = e
    raise RuntimeError(f"모델 다운로드 실패 ({dest.name}): {last_err}")


def ensure_models() -> tuple[Path, Path]:
    """(segmentation_model, embedding_model) 경로 반환. 없으면 다운로드."""
    if not SEGMENTATION_MODEL.exists():
        try:
            archive = MODELS_DIR / "segmentation.tar.bz2"
            _download(SEGMENTATION_URLS, archive)
            with tarfile.open(archive, "r:bz2") as tf:
                tf.extractall(MODELS_DIR)
            archive.unlink(missing_ok=True)
        except RuntimeError:
            _download(SEGMENTATION_ONNX_URLS, SEGMENTATION_MODEL)
        if not SEGMENTATION_MODEL.exists():
            raise RuntimeError(f"세그멘테이션 모델 준비 실패: {SEGMENTATION_MODEL}")
    if not EMBEDDING_MODEL.exists():
        _download(EMBEDDING_URLS, EMBEDDING_MODEL)
    return SEGMENTATION_MODEL, EMBEDDING_MODEL


def models_ready() -> bool:
    return SEGMENTATION_MODEL.exists() and EMBEDDING_MODEL.exists()
