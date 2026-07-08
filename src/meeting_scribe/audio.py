"""ffmpeg 기반 오디오 정규화. 시스템 ffmpeg가 없으면 imageio-ffmpeg 번들 바이너리 사용."""
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def normalize(input_path: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    """어떤 포맷이든 16kHz mono PCM16 wav로 변환."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(), "-y", "-i", str(input_path),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        "-vn", str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 변환 실패: {proc.stderr[-800:]}")
    return out_wav


def read_wav_float32(wav_path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM wav → float32 [-1, 1] 샘플과 샘플레이트."""
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        channels = w.getnchannels()
    if width != 2:
        raise ValueError(f"PCM16 wav만 지원 (sampwidth={width})")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sr


def duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def split_wav(wav_path: Path, chunk_seconds: int, out_dir: Path) -> list[tuple[Path, float]]:
    """긴 wav를 청크로 분할. (경로, 시작오프셋초) 목록 반환. API 업로드 용량 제한 대응."""
    total = duration_seconds(wav_path)
    if total <= chunk_seconds:
        return [(wav_path, 0.0)]
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    offset = 0.0
    i = 0
    while offset < total:
        out = out_dir / f"chunk_{i:03d}.wav"
        cmd = [
            ffmpeg_exe(), "-y", "-i", str(wav_path),
            "-ss", str(offset), "-t", str(chunk_seconds),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 분할 실패: {proc.stderr[-500:]}")
        chunks.append((out, offset))
        offset += chunk_seconds
        i += 1
    return chunks
