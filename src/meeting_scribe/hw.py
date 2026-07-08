"""하드웨어 감지: NVIDIA GPU(VRAM), CUDA 런타임, RAM."""
import shutil
import subprocess


def ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / (1024 ** 2), 1)
    except OSError:  # Windows/macOS
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except ImportError:
            return None


def nvidia_gpu() -> dict | None:
    """nvidia-smi로 GPU 이름·VRAM 조회. 없으면 None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
        if not line:
            return None
        name, mem = line.rsplit(",", 1)
        return {"name": name.strip(), "vram_gb": round(int(mem.strip()) / 1024, 1)}
    except Exception:
        return None


def cuda_runtime_available() -> bool:
    """faster-whisper(CTranslate2)가 실제로 CUDA를 쓸 수 있는지 (cuBLAS/cuDNN 포함)."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def local_whisper_plan() -> dict:
    """로컬 whisper 실행 계획: (device, compute_type, model). GPU VRAM에 맞춰 자동 선택.
    - VRAM ≥ 4GB (RTX 3050급~): large-v3-turbo — 실시간보다 빠름, 한국어 품질 최상
    - VRAM ≥ 3GB: medium
    - VRAM < 3GB (GTX 1050 2GB 등): small
    - GPU 불가: CPU int8, RAM 기준 모델 선택"""
    gpu = nvidia_gpu()
    if gpu and cuda_runtime_available():
        vram = gpu["vram_gb"]
        model = "large-v3-turbo" if vram >= 4 else ("medium" if vram >= 3 else "small")
        return {"device": "cuda", "compute_type": "int8_float16", "model": model, "gpu": gpu}
    ram = ram_gb() or 8.0
    model = "small" if ram < 8 else ("medium" if ram < 12 else "large-v3-turbo")
    return {"device": "cpu", "compute_type": "int8", "model": model, "gpu": gpu}
